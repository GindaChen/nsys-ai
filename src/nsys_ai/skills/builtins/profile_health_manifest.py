"""Profile Health Manifest — one-shot summary for external AI agents.

Returns a compact JSON summary (~500 bytes) that captures the essential
profile characteristics in a single tool call, eliminating the need for
5-8 sequential skill invocations during agent exploration.

Internally orchestrates: overlap_breakdown, nccl_breakdown,
nccl_communicator_analysis, gpu_idle_gaps, root_cause_matcher,
aggregate_nvtx_ranges, and iteration_timing.
"""

import dataclasses
import logging
from datetime import datetime, timezone

from ..base import Skill, SkillParam

_log = logging.getLogger(__name__)


def _safe_skill_run(skill_name: str, conn, **kwargs):
    """Run a skill by name, returning [] on any error."""
    import sqlite3

    import duckdb

    from nsys_ai.exceptions import SkillExecutionError

    from ..registry import get_skill

    skill = get_skill(skill_name)
    if skill is None:
        return []
    try:
        return skill.execute(conn, **kwargs)
    except (sqlite3.Error, duckdb.Error, SkillExecutionError) as exc:
        _log.debug("manifest: %s failed: %s", skill_name, exc, exc_info=True)
        return []


# Threshold past which a no-trim manifest is auto-narrowed to a
# representative window. Below this, the full scan is fast enough that
# sub-skills (nvtx_layer_breakdown IEJoin, root_cause_matcher, …) won't
# blow past their soft budgets even on a multi-GPU export.
_AUTO_TRIM_PROFILE_SPAN_THRESHOLD_NS = 120 * 10**9   # 120 s
# Target width of the auto-selected window. Picked to be (a) large enough
# to capture at least a few steady-state iterations of any DiT / transformer
# block, (b) small enough that a 15 M-row NVTX IEJoin completes in seconds.
_AUTO_TRIM_TARGET_WINDOW_NS = 20 * 10**9             # 20 s

# Values that turn NSYS_AI_MANIFEST_AUTO_TRIM off. Matches the parsing
# of NSYS_AI_DEFER_NVTX_KERNEL_MAP in parquet_cache.py so users get
# the same on/off semantics across env vars in this repo.
_AUTO_TRIM_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})

# ── Roll-up finding thresholds ──────────────────────────────────────
# Shared between _infer_bottleneck (the human-readable string) and
# _to_findings (the structured roll-up emitter) so a tweak in one place
# can't drift from the other. Same constants-at-module-top convention as
# PR #149 (kernel_launch_overhead) and PR #153 (top_kernels).
_OVERHEAD_CONTAMINATED_PCT = 1.0
_SYNC_BOUND_DENSITY_PCT = 20.0
_COMM_BOUND_OVERLAP_PCT = 30.0
_IDLE_DOMINANT_PCT = 15.0
_KERNEL_HOTSPOT_PCT = 60.0
_ITER_VARIANCE_SPIKE_RATIO = 1.5
_MIN_ITERATIONS_FOR_NVTX_COVERAGE = 3


def _auto_trim_enabled() -> bool:
    """Read NSYS_AI_MANIFEST_AUTO_TRIM with 0/false/no/off → disabled."""
    import os

    return os.environ.get("NSYS_AI_MANIFEST_AUTO_TRIM", "1").strip().lower() not in _AUTO_TRIM_FALSE_TOKENS


def _resolve_nvtx_table_for_auto_trim(prof) -> str | None:
    """Find a NVTX-bearing table or view we can scan for iteration markers.

    Preference order:
      * ``nvtx_high`` view — DuckDB cache, aten::* already filtered.
      * ``nvtx`` view — DuckDB cache, full NVTX with resolved text.
      * Versioned canonical table — ``NVTX_EVENTS`` / ``NVTX_EVENTS_V2`` /
        ``NVTX_EVENTS_V3`` (the actual nsys export schema for the version
        we attached). Resolved via ``prof.schema.tables`` rather than
        hardcoded so newer Nsight exports without an unversioned alias
        still work.

    Returns the chosen name or ``None`` when no NVTX source can be opened.
    """
    import sqlite3

    import duckdb

    # Cache-view candidates always have resolved text (no StringIds join
    # needed by the caller). Probe each; pick the first that can be
    # opened.
    for candidate in ("nvtx_high", "nvtx"):
        try:
            prof._duckdb_query(f"SELECT 1 FROM {candidate} LIMIT 1")
            return candidate
        except (sqlite3.Error, duckdb.Error):
            continue

    # Raw-SQLite path: walk the actual schema for an NVTX-prefixed name.
    try:
        tables = set(prof.schema.tables)
    except Exception:
        tables = set()
    if "NVTX_EVENTS" in tables:
        return "NVTX_EVENTS"
    for t in sorted(tables):
        if t.startswith("NVTX_EVENTS"):
            return t
    return None


def _build_auto_trim_nvtx_sql(prof, nvtx_table: str) -> str:
    """SQL that picks the middle instance of the dominant NVTX iteration.

    Cache views (``nvtx_high`` / ``nvtx``) already carry resolved text in
    the ``text`` column — no StringIds join needed. The raw
    ``NVTX_EVENTS`` family does need the join when the export stores
    labels via ``textId`` (newer Nsight schemas with
    ``sqlite_all_varchar=true`` attach), so we COALESCE the two and
    LEFT JOIN ``StringIds`` only on that path.
    """
    is_cache_view = nvtx_table in ("nvtx_high", "nvtx")
    if is_cache_view:
        text_expr = "n.text"
        text_join = ""
    else:
        text_expr = "COALESCE(n.text, s.value)"
        text_join = "LEFT JOIN StringIds s ON n.textId = s.id"
    return f"""
        WITH ranged AS (
            SELECT {text_expr} AS text, n.start, n.[end]
            FROM {nvtx_table} n
            {text_join}
            WHERE {text_expr} IS NOT NULL
              AND n.[end] > n.start
              AND {text_expr} NOT LIKE 'aten::%'
              AND {text_expr} NOT LIKE 'cudaLaunch%'
              AND {text_expr} NOT LIKE 'cudaMemcpy%'
        )
        SELECT text, start, [end] FROM ranged
        WHERE text = (
            SELECT text FROM ranged
            GROUP BY text
            HAVING COUNT(*) >= 3
            ORDER BY SUM([end] - start) DESC, text ASC
            LIMIT 1
        )
        ORDER BY start
    """


def _auto_select_trim_window(prof) -> tuple[int, int] | None:
    """Pick a representative steady-state window for a long profile.

    Returns ``(start_ns, end_ns)`` to feed into sub-skills as
    ``trim_start_ns`` / ``trim_end_ns``, or ``None`` when no trim should
    be applied — either because the profile span is at/under the
    threshold, or because the time_range itself is unusable (None,
    inverted, missing meta).

    Strategy:
      1. If profile span ``≤ _AUTO_TRIM_PROFILE_SPAN_THRESHOLD_NS``,
         return None — the full manifest will fit a normal turn budget.
      2. Otherwise, find the top non-``aten::*`` NVTX range name with
         ≥ 3 instances (iteration / stage marker) and pick the *middle*
         instance — skips index-0 JIT warmup. If the chosen instance is
         wider than ``_AUTO_TRIM_TARGET_WINDOW_NS``, trim further to the
         middle of that range.
      3. If no qualifying NVTX range exists, fall back to the middle
         ``_AUTO_TRIM_TARGET_WINDOW_NS`` of the profile span — better
         to land near steady state than to scan head-of-trace JIT plus
         tail teardown.

    Failures are absorbed — auto-trim is best-effort; the caller proceeds
    untrimmed when this returns None or raises.
    """
    import sqlite3

    import duckdb

    try:
        start_ns, end_ns = prof.meta.time_range
    except Exception:
        return None
    # Profiles missing or with a degenerate time_range can't be trimmed.
    if start_ns is None or end_ns is None or end_ns <= start_ns:
        return None
    span_ns = end_ns - start_ns
    # Use ≤ so the boundary matches the docstring contract — a span
    # exactly equal to the threshold is short enough to not need trim.
    if span_ns <= _AUTO_TRIM_PROFILE_SPAN_THRESHOLD_NS:
        return None

    # Probe candidates in preference order. The first one that can be
    # opened wins — `nvtx_high` is fastest because aten::* is already
    # filtered, but it only exists on _CACHE_VERSION ≥ 14 caches.
    # `nvtx` is the resolved-text cache view. `NVTX_EVENTS` (and its
    # versioned variants) is the raw SQLite table.
    nvtx_table = _resolve_nvtx_table_for_auto_trim(prof)
    if nvtx_table is None:
        # No NVTX source at all → skip the SQL path and fall straight to
        # the middle-of-span fallback below.
        rows: list = []
    else:
        sql = _build_auto_trim_nvtx_sql(prof, nvtx_table)
        try:
            rows = prof._duckdb_query(sql)
        except (sqlite3.Error, duckdb.Error) as exc:
            _log.debug("auto-trim NVTX query failed: %s", exc)
            rows = []

    if rows:
        # Skip the first instance (JIT warmup / one-time setup cost) and
        # pick the middle of what remains so we land in steady state.
        idx = max(1, len(rows) // 2)
        if idx >= len(rows):
            idx = len(rows) - 1
        chosen = rows[idx]
        c_start = int(chosen["start"])
        c_end = int(chosen["end"])
        # Trim very wide stage ranges down to a 20 s slice in their middle
        # so sub-skills still get steady-state behaviour but don't grind.
        if c_end - c_start > _AUTO_TRIM_TARGET_WINDOW_NS:
            mid = (c_start + c_end) // 2
            half = _AUTO_TRIM_TARGET_WINDOW_NS // 2
            return (mid - half, mid + half)
        return (c_start, c_end)

    # No usable NVTX iteration marker — take the middle 20 s of the
    # profile so we at least avoid head-of-trace JIT and tail teardown.
    mid = (start_ns + end_ns) // 2
    half = _AUTO_TRIM_TARGET_WINDOW_NS // 2
    return (mid - half, mid + half)


def _execute(conn, **kwargs):
    """Build a compact profile health manifest."""
    from ...profile import Profile

    device = int(kwargs.get("device", 0))
    overhead_ns = kwargs.get("overhead_ns", 0)

    # Forward trim kwargs if present. Treat trim as "explicit" only when
    # BOTH endpoints are supplied — a partial trim is unusable downstream
    # (sub-skills ignore it) and would otherwise silence auto-trim while
    # leaving the manifest scanning the full profile.
    trim_kwargs = {}
    for k in ("trim_start_ns", "trim_end_ns"):
        if kwargs.get(k) is not None:
            trim_kwargs[k] = kwargs[k]
    explicit_trim = (
        "trim_start_ns" in trim_kwargs and "trim_end_ns" in trim_kwargs
    )
    if trim_kwargs and not explicit_trim:
        # Drop the partial trim so it can't be forwarded to sub-skills
        # in a half-useless state; auto-trim (below) will replace it.
        _log.debug(
            "manifest: partial trim_kwargs %s discarded; will auto-trim if long",
            list(trim_kwargs),
        )
        trim_kwargs = {}

    # ── 1. Profile metadata ──────────────────────────────────────
    prof = Profile._from_conn(conn)

    # ── 1a. Auto-trim for very long profiles ────────────────────
    # Without this guard, manifest on a multi-GB / 10-minute profile
    # melts on the NVTX×kernel IEJoin (15M+ rows) and `iteration_timing`
    # full-table scans. Opt out with NSYS_AI_MANIFEST_AUTO_TRIM=0.
    auto_trim_meta: dict | None = None
    if not explicit_trim and _auto_trim_enabled():
        try:
            picked = _auto_select_trim_window(prof)
        except Exception as exc:  # noqa: BLE001 — best-effort, never block manifest
            _log.debug("manifest auto-trim selection failed: %s", exc, exc_info=True)
            picked = None
        if picked is not None:
            t0, t1 = picked
            trim_kwargs = {"trim_start_ns": int(t0), "trim_end_ns": int(t1)}
            # Preserve the original profile span — the standard
            # `profile_span_ms` below will reflect the trim window, so a
            # caller who wants to know "this is a 10-minute profile"
            # would otherwise see 20 s. The pre-trim span lives here.
            full_start, full_end = prof.meta.time_range
            auto_trim_meta = {
                "applied": True,
                "trim_start_ns": int(t0),
                "trim_end_ns": int(t1),
                "window_ms": round((t1 - t0) / 1e6, 1),
                "profile_full_span_ms": round((full_end - full_start) / 1e6, 1),
            }
            _log.info(
                "manifest auto-trimmed to %.2fs–%.2fs (%.1fs window) on long profile",
                t0 / 1e9,
                t1 / 1e9,
                (t1 - t0) / 1e9,
            )
    # Prefer the GPU name for the requested device, if available.
    gpu_name = "unknown"
    gpu_info = getattr(prof.meta, "gpu_info", None)
    if gpu_info is not None:
        device_info = None
        # Support both dict- and list-like gpu_info containers.
        if isinstance(gpu_info, dict):
            device_info = gpu_info.get(device)
        elif isinstance(gpu_info, list) and device < len(gpu_info):
            device_info = gpu_info[device]
        if device_info is not None:
            gpu_name = getattr(device_info, "name", "unknown")

    # Fallback if device info is missing/empty
    if not gpu_name or gpu_name == "unknown":
        from ...profile import get_first_gpu_name

        gpu_name = get_first_gpu_name(conn) or "unknown"
    start_ns, end_ns = prof.meta.time_range
    # Use the trim window (if provided), clamped to the profile range,
    # so the reported span matches the analysis window used by sub-skills.
    effective_start_ns = start_ns
    effective_end_ns = end_ns
    if trim_kwargs.get("trim_start_ns") is not None and trim_kwargs.get("trim_end_ns") is not None:
        effective_start_ns = max(effective_start_ns, trim_kwargs["trim_start_ns"])
        effective_end_ns = min(effective_end_ns, trim_kwargs["trim_end_ns"])
    profile_span_ns = (
        effective_end_ns - effective_start_ns if effective_end_ns > effective_start_ns else 0
    )
    profile_span_ms = round(profile_span_ns / 1e6, 1) if profile_span_ns > 0 else 0

    overhead_ms = round(overhead_ns / 1e6, 1)
    overhead_pct_raw = (overhead_ns / profile_span_ns * 100) if profile_span_ns > 0 else 0
    overhead_pct = round(overhead_pct_raw, 1)
    data_quality = {
        "profiler_overhead_ms": overhead_ms,
        "overhead_pct": overhead_pct,
        "overhead_pct_raw": overhead_pct_raw,
    }
    if auto_trim_meta is not None:
        data_quality["auto_trim"] = auto_trim_meta

    # ── 2. Top kernels (compact: top 5 only) ─────────────────────
    trim_tuple = None
    if trim_kwargs.get("trim_start_ns") is not None and trim_kwargs.get("trim_end_ns") is not None:
        trim_tuple = (trim_kwargs["trim_start_ns"], trim_kwargs["trim_end_ns"])

    try:
        import sqlite3

        import duckdb

        from nsys_ai.exceptions import SkillExecutionError

        # Use aggregate_kernels for correct device filtering
        agg_kernels = prof.aggregate_kernels(device=device, trim=trim_tuple, limit=None)
    except (sqlite3.Error, duckdb.Error, SkillExecutionError) as exc:
        _log.debug("manifest: aggregate_kernels failed: %s", exc, exc_info=True)
        agg_kernels = [{"demangled": f"Error: {exc}", "total_ns": 0, "count": 0}]

    top_kernels = []
    for r in agg_kernels[:5]:
        name = r.get("demangled", "?")
        if len(name) > 60:
            name = name[:57] + "..."
        top_kernels.append(
            {
                "name": name,
                "total_ms": round(r.get("total_ns", 0) / 1e6, 2),
                "count": r.get("count", 0),
            }
        )

    # Compute total kernel time over all kernels
    total_kernel_ms = sum(r.get("total_ns", 0) for r in agg_kernels) / 1e6

    # ── 3. Compute/NCCL overlap ──────────────────────────────────
    overlap_rows = _safe_skill_run("overlap_breakdown", conn, device=device, **trim_kwargs)
    overlap = {}
    if overlap_rows:
        ov = overlap_rows[0]
        if "error" in ov:
            # Preserve error details from overlap_breakdown instead of dropping them.
            # At minimum, surface the primary error message; keep any additional
            # fields that may provide context for callers.
            overlap = dict(ov)
        else:
            overlap = {
                "compute_only_ms": ov.get("compute_only_ms", 0),
                "nccl_only_ms": ov.get("nccl_only_ms", 0),
                "overlap_pct": ov.get("overlap_pct", 0),
                "idle_ms": ov.get("idle_ms", 0),
            }

    # ── 4. NCCL breakdown (compact summary) ──────────────────────
    nccl_rows = _safe_skill_run("nccl_breakdown", conn, device=device, **trim_kwargs)
    nccl_summary = {"streams": 0, "collectives": 0}
    if nccl_rows:
        stream_ids = {r.get("stream_id") for r in nccl_rows}
        nccl_summary["streams"] = len(stream_ids)
        nccl_summary["collectives"] = sum(r.get("count", 1) for r in nccl_rows)
        # Dominant collective by total_ms
        dominant = max(nccl_rows, key=lambda r: r.get("total_ms", 0))
        nccl_summary["dominant_type"] = dominant.get("type", "?")
        nccl_summary["dominant_pct"] = dominant.get("pct", 0)
        nccl_summary["total_nccl_ms"] = round(sum(r.get("total_ms", 0) for r in nccl_rows), 1)

    communicator_rows = _safe_skill_run("nccl_communicator_analysis", conn, device=device, **trim_kwargs)
    communicator_data = [r for r in communicator_rows if not r.get("_diagnostic")]
    communicator_summary = {"communicators": 0, "collective_rows": 0}
    if communicator_data:
        communicator_summary["communicators"] = len(
            {r.get("communicator_hex") for r in communicator_data if r.get("communicator_hex")}
        )
        communicator_summary["collective_rows"] = len(communicator_data)
        dominant_comm = max(communicator_data, key=lambda r: r.get("total_ms", 0))
        communicator_summary["dominant_collective"] = dominant_comm.get("collective_type", "?")
        communicator_summary["dominant_dimension"] = dominant_comm.get(
            "inferred_dimension", "single_rank_or_unknown"
        )
        communicator_summary["top_total_ms"] = round(dominant_comm.get("total_ms", 0), 1)
        communicator_summary["subgroup_count"] = sum(
            1
            for r in communicator_data
            if str(r.get("inferred_dimension", "")).startswith("subgroup_parallelism")
        )
        communicator_summary["low_efficiency_count"] = sum(
            1
            for r in communicator_data
            if r.get("efficiency_pct") is not None and r.get("efficiency_pct", 100) < 20.0
        )

    # ── 5. GPU idle gaps (summary only) ──────────────────────────
    idle_rows = _safe_skill_run("gpu_idle_gaps", conn, device=device, limit=1, **trim_kwargs)
    idle_summary = {"gap_count": 0, "idle_pct": 0}
    summary_row = next((r for r in idle_rows if r.get("_summary")), None)
    if summary_row:
        idle_summary["gap_count"] = summary_row.get("gap_count", 0)
        idle_summary["idle_pct"] = summary_row.get("pct_of_profile", 0)
        idle_summary["total_idle_ms"] = summary_row.get("total_idle_ms", 0)

    # ── 6. Sync Cost Analysis ────────────────────────────────────────
    sync_summary = {}
    try:
        sync_data = _safe_skill_run("sync_cost_analysis", conn, **trim_kwargs)
        if sync_data and "error" not in sync_data[0]:
            sync_summary = sync_data[0]
    except Exception as exc:
        _log.debug("manifest: sync_cost_analysis failed: %s", exc)

    # ── 8. NVTX summary ──────────────────────────────────────────
    nvtx_summary: dict = {"has_nvtx": False}
    try:
        nvtx_ranges = prof.aggregate_nvtx_ranges(limit=5, trim=trim_tuple)
        if nvtx_ranges:
            nvtx_summary["has_nvtx"] = True
            nvtx_summary["top_regions"] = [
                {
                    "name": (r.get("text") or "?")[:50],
                    "total_ms": round(r.get("total_ns", 0) / 1e6, 1),
                    "count": r.get("count", 0),
                }
                for r in nvtx_ranges[:5]
            ]
    except Exception as exc:
        _log.debug("manifest: aggregate_nvtx_ranges failed: %s", exc)

    iter_rows = _safe_skill_run("iteration_timing", conn, device=device, **trim_kwargs)
    if iter_rows:
        nvtx_summary["iteration_count"] = len(iter_rows)
        # Skip iter 0 (warm-up) when computing variance metrics
        steady = iter_rows[1:] if len(iter_rows) > 1 else iter_rows
        if steady:
            durs = sorted(r.get("duration_ms", 0) for r in steady)
            mid = len(durs) // 2
            nvtx_summary["median_iter_ms"] = round(durs[mid], 1)
            nvtx_summary["slowest_iter_ms"] = round(max(durs), 1)

    # ── 7. Root cause findings (count + top severity) ────────────
    # Pass precomputed communicator rows to avoid re-running the expensive
    # nccl_communicator_analysis inside root_cause_matcher.
    rc_rows = _safe_skill_run(
        "root_cause_matcher", conn, device=device,
        communicator_data=communicator_rows, **trim_kwargs,
    )
    root_causes = []
    for r in rc_rows:
        pattern = r.get("pattern", "")
        if pattern == "No Known Anti-Patterns Detected":
            continue
        root_causes.append(
            {
                "pattern": pattern,
                "severity": r.get("severity", "info"),
            }
        )

    sev_rank = {"critical": 0, "high": 1, "warning": 2, "medium": 3, "low": 4, "info": 5}
    root_causes.sort(key=lambda x: sev_rank.get(x["severity"].lower(), 99))

    # ── Assemble manifest ────────────────────────────────────────
    manifest = {
        "analysis_time_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "gpu": gpu_name,
        "fingerprint": dataclasses.asdict(prof.fingerprint) if prof.fingerprint else None,
        "profile_span_ms": profile_span_ms,
        "top_kernels": top_kernels,
        "total_kernel_ms": round(total_kernel_ms, 1),
        "overlap": overlap,
        "nccl": nccl_summary,
        "communicators": communicator_summary,
        "sync": sync_summary,
        "idle": idle_summary,
        "root_cause_count": len(root_causes),
        "root_causes": root_causes[:5],  # Cap at 5 to keep output compact
        "nvtx": nvtx_summary,
        "data_quality": data_quality,
    }

    # Infer suspected bottleneck
    bottleneck = _infer_bottleneck(manifest)
    if bottleneck:
        manifest["suspected_bottleneck"] = bottleneck

    return [manifest]


def _infer_bottleneck(m: dict) -> str:
    """Heuristic bottleneck inference from manifest data."""
    sync = m.get("sync", {})
    sync_density = sync.get("sync_density_pct", 0)
    if sync_density > _SYNC_BOUND_DENSITY_PCT:
        return f"High CPU Synchronization Blocking ({sync_density:.1f}% of span)"

    dq = m.get("data_quality", {})
    overhead_pct_val = dq.get("overhead_pct_raw", dq.get("overhead_pct", 0))
    if overhead_pct_val > _OVERHEAD_CONTAMINATED_PCT:
        return f"Profiler Overhead ({dq.get('overhead_pct', overhead_pct_val)}%) contaminated the profile"

    overlap = m.get("overlap", {})
    idle = m.get("idle", {})
    nccl = m.get("nccl", {})

    # Check NCCL serialization first (most impactful)
    if overlap.get("overlap_pct", 100) < _COMM_BOUND_OVERLAP_PCT and overlap.get("nccl_only_ms", 0) > 0:
        return f"NCCL serialization (overlap < {int(_COMM_BOUND_OVERLAP_PCT)}%)"

    # Check idle gaps
    if idle.get("idle_pct", 0) > _IDLE_DOMINANT_PCT:
        return f"GPU idle bubbles ({idle['idle_pct']}% of profile)"

    # Check if one kernel dominates
    top_k = m.get("top_kernels", [])
    total = m.get("total_kernel_ms", 0)
    if top_k and total > 0:
        top_pct = top_k[0]["total_ms"] / total * 100
        if top_pct > _KERNEL_HOTSPOT_PCT:
            return f"Kernel hotspot: {top_k[0]['name']} ({top_pct:.0f}%)"

    # Check NCCL dominance
    if nccl.get("total_nccl_ms", 0) > overlap.get("compute_only_ms", float("inf")):
        return "Communication-bound (NCCL > compute)"

    # Check iteration variance (spike pattern)
    nvtx = m.get("nvtx", {})
    median_ms = nvtx.get("median_iter_ms", 0)
    slowest_ms = nvtx.get("slowest_iter_ms", 0)
    if median_ms > 0 and slowest_ms > median_ms * _ITER_VARIANCE_SPIKE_RATIO:
        ratio = slowest_ms / median_ms
        return f"Iteration variance spike (slowest {slowest_ms:.0f}ms = {ratio:.1f}× median)"

    return ""


# Explanation strings interpolate the threshold constants at module load
# so the prose stays in sync with the thresholds it cites. Hardcoded
# values would silently drift if a threshold were ever tuned.
_OVERHEAD_EXPLANATION = (
    f"Per-event CUPTI instrumentation cost exceeded {_OVERHEAD_CONTAMINATED_PCT}% of "
    "profile span, which perturbs kernel durations and launch timings enough to "
    "mask real regressions. Re-capture with torch.cuda.profiler.start/stop() + "
    "--capture-range=cudaProfilerApi to scope nsys to the region of interest."
)
_SYNC_EXPLANATION = (
    "Synchronous CPU→GPU waits (cudaStreamSynchronize, cudaMemcpy, .item(), "
    f".cpu()) consumed >{_SYNC_BOUND_DENSITY_PCT}% of wall time. Look for "
    "unnecessary host-side tensor reads, non-pinned host memory in the "
    "dataloader, or eager evaluation inside the hot loop."
)
_COMM_EXPLANATION = (
    f"Compute/NCCL overlap fell below {int(_COMM_BOUND_OVERLAP_PCT)}% with "
    "non-trivial NCCL traffic on a separate stream — the rank is serializing "
    "communication after compute instead of hiding it. Candidate levers: "
    "dedicated NCCL stream + wait_stream ordering, Ulysses-Ring hybrid SP "
    "partitioning, larger fused all-reduces."
)
_IDLE_EXPLANATION = (
    f"GPU was idle for >{int(_IDLE_DOMINANT_PCT)}% of the profile, indicating "
    "launch-bound segments or host-side blocking gaps between kernels. Inspect "
    "gpu_idle_gaps findings for the specific stalls; consider CUDA Graphs / "
    "torch.compile for launch overhead, dataloader workers / pinned memory "
    "for host stalls."
)
_HOTSPOT_EXPLANATION = (
    f"A single kernel accounts for >{int(_KERNEL_HOTSPOT_PCT)}% of total kernel "
    "time — optimization effort spent anywhere else is Amdahl-limited. Triage "
    "this kernel first: check TC eligibility, fusion opportunities, dtype, "
    "occupancy."
)
_ITER_SPIKE_EXPLANATION = (
    f"At least one steady-state iteration ran ≥{_ITER_VARIANCE_SPIKE_RATIO}× the "
    "median, indicating non-uniform per-step cost (CFG batching, conditional "
    "branches, sync outliers, or the first NCCL collective of a new "
    "communicator). The median is what matters for throughput; the spike often "
    "signals a specific anti-pattern."
)
_NVTX_COVERAGE_EXPLANATION = (
    f"The profile has no NVTX annotations or fewer than {_MIN_ITERATIONS_FOR_NVTX_COVERAGE} "
    "detected iterations, which means iteration_timing / nvtx_kernel_map / "
    "overlap-by-region skills cannot anchor their analysis. Wrap the training "
    "loop in torch.cuda.nvtx.range_push/pop or use the existing FastVideo "
    "annotate_profile_regions context manager before re-capturing."
)

_OVERHEAD_ACTIONS = (
    "Use torch.cuda.profiler.start/stop() around the region of interest.",
    "Pass --capture-range=cudaProfilerApi to nsys profile.",
    "Disable per-event CUPTI features (--cuda-trace=runtime,driver only).",
)
_SYNC_ACTIONS = (
    "Audit .item() / .cpu() / .tolist() calls inside the hot loop.",
    "Switch host-side buffers to pinned memory with non_blocking=True copies.",
    "Move loss / metric reductions out of the inner loop.",
)
_COMM_ACTIONS = (
    "Route NCCL collectives onto a dedicated stream with wait_stream ordering.",
    "Evaluate Hybrid Ulysses-Ring SP partitioning for the dominant dimension.",
    "Profile per-stream NCCL breakdown to identify the serialized collective.",
)
_IDLE_ACTIONS = (
    "Inspect gpu_idle_gaps findings for the specific stall location.",
    "Enable torch.compile (or CUDA Graphs) on the launch-bound segment.",
    "Verify dataloader is non-blocking and uses pinned memory.",
)
_HOTSPOT_ACTIONS = (
    "Confirm Tensor-Core eligibility (dtype, dims, layout) for this kernel.",
    "Check for fusion opportunities with adjacent ops under torch.compile.",
    "Profile occupancy and register pressure with Nsight Compute.",
)
_ITER_SPIKE_ACTIONS = (
    "Inspect iteration_timing for the specific slow iteration's stack.",
    "Check whether the spike aligns with a CFG batch / first-of-kind collective.",
    "Compare median vs slowest with NVTX annotations to localize the divergence.",
)
_NVTX_COVERAGE_ACTIONS = (
    "Wrap the training loop in torch.cuda.nvtx.range_push/range_pop.",
    "Use the framework's existing region-annotation context manager.",
    "Re-capture with at least 5 steady-state iterations after warmup.",
)


def _to_findings(rows: list[dict], *, context: dict | None = None) -> list:
    """Emit roll-up findings characterizing the profile as a whole.

    Reads only the already-assembled manifest dict — never reaches into
    sub-skill findings — so this skill's findings are independent of any
    sub-skill's _to_findings status. Seven roll-up types map to the
    step-time decomposition (compute/communication/idle/sync) plus
    profile_quality coverage. ``_infer_bottleneck`` still produces the
    single human-readable string for ``_format``; this emits the full
    structured set.
    """
    from nsys_ai.annotation import EvidenceRow, Finding, TraceSelection

    findings: list = []
    if not rows:
        return findings
    m = rows[0]
    profile_id = (context or {}).get("profile_id", "unknown")

    def _emit(
        *,
        finding_id: str,
        label: str,
        severity: str,
        category: str,
        values: dict,
        units: dict,
        note: str,
        explanation: str,
        actions: tuple[str, ...],
        provenance_extra: dict | None = None,
    ) -> None:
        # All roll-ups are profile-global characterizations (no time
        # anchor); ``start_ns=0`` is the established "no anchor" sentinel
        # — see top_kernels.top_kernels_concentrated for the same
        # convention — and ``type="highlight"`` signals to consumers that
        # this is not a trace location.
        selection = TraceSelection(
            id=f"sel_{finding_id}",
            profile_id=profile_id,
            source="skill:profile_health_manifest",
            label=label,
        )
        provenance = {"skill": "profile_health_manifest", "row_kind": finding_id}
        if provenance_extra:
            provenance.update(provenance_extra)
        ev = EvidenceRow(
            id=f"ev_{finding_id}",
            source_skill="profile_health_manifest",
            values=values,
            units=units,
            selection_id=selection.id,
            provenance={"row_kind": finding_id},
        )
        findings.append(
            Finding(
                type="highlight",
                label=label,
                start_ns=0,
                end_ns=None,
                severity=severity,
                note=note,
                id=finding_id,
                category=category,
                evidence=[ev],
                selection=selection,
                explanation=explanation,
                suggested_actions=list(actions),
                provenance=provenance,
            )
        )

    # ── 1. Profiler overhead contamination ─────────────────────────
    dq = m.get("data_quality", {}) or {}
    overhead_pct = float(dq.get("overhead_pct_raw", dq.get("overhead_pct", 0)) or 0)
    if overhead_pct > _OVERHEAD_CONTAMINATED_PCT:
        overhead_ms = float(dq.get("profiler_overhead_ms", 0) or 0)
        _emit(
            finding_id="profile_overhead_contaminated",
            label=f"Profiler overhead contaminated profile ({overhead_pct:.1f}%)",
            severity="critical",
            category="profile_quality",
            values={
                "overhead_pct": round(overhead_pct, 2),
                "overhead_ms": round(overhead_ms, 2),
                "threshold_pct": _OVERHEAD_CONTAMINATED_PCT,
            },
            units={"overhead_pct": "percent", "overhead_ms": "ms", "threshold_pct": "percent"},
            note=(
                f"Per-event profiler overhead is {overhead_pct:.1f}% of profile span "
                f"({overhead_ms:.1f}ms). Above {_OVERHEAD_CONTAMINATED_PCT}% the captured "
                "kernel durations are no longer trustworthy."
            ),
            explanation=_OVERHEAD_EXPLANATION,
            actions=_OVERHEAD_ACTIONS,
        )

    # ── 2. CPU sync blocking ────────────────────────────────────────
    sync = m.get("sync", {}) or {}
    sync_density = float(sync.get("sync_density_pct", 0) or 0)
    if sync_density > _SYNC_BOUND_DENSITY_PCT:
        sync_wall_ms = float(sync.get("total_sync_wall_ms", 0) or 0)
        _emit(
            finding_id="profile_sync_bound",
            label=f"CPU sync blocks {sync_density:.1f}% of wall time",
            severity="warning",
            category="sync",
            values={
                "sync_density_pct": round(sync_density, 2),
                "sync_wall_ms": round(sync_wall_ms, 2),
                "threshold_pct": _SYNC_BOUND_DENSITY_PCT,
            },
            units={
                "sync_density_pct": "percent",
                "sync_wall_ms": "ms",
                "threshold_pct": "percent",
            },
            note=(
                f"Synchronous CPU→GPU waits consume {sync_density:.1f}% of wall time "
                f"({sync_wall_ms:.1f}ms). Threshold {_SYNC_BOUND_DENSITY_PCT}%."
            ),
            explanation=_SYNC_EXPLANATION,
            actions=_SYNC_ACTIONS,
        )

    # ── 3. Communication-bound ──────────────────────────────────────
    overlap = m.get("overlap", {}) or {}
    nccl = m.get("nccl", {}) or {}
    # `overlap_pct` of exactly 0.0 is a legitimate (and important) signal
    # — full serialization of compute and NCCL on one stream. The naive
    # ``x or 100`` idiom would mistake 0.0 for "missing" and silence the
    # comm_bound finding precisely when it should fire hardest. Caught
    # during L40S validation on the uncompiled perf.sqlite profile, where
    # overlap_pct=0.0 and nccl_only_ms=8654ms should have tripped the
    # low-overlap trigger but didn't.
    raw_overlap_pct = overlap.get("overlap_pct")
    overlap_pct = 100.0 if raw_overlap_pct is None else float(raw_overlap_pct)
    nccl_only_ms = float(overlap.get("nccl_only_ms", 0) or 0)
    total_nccl_ms = float(nccl.get("total_nccl_ms", 0) or 0)
    compute_only_ms = float(overlap.get("compute_only_ms", 0) or 0)
    low_overlap = overlap_pct < _COMM_BOUND_OVERLAP_PCT and nccl_only_ms > 0
    nccl_dominates_compute = total_nccl_ms > 0 and total_nccl_ms > compute_only_ms
    if low_overlap or nccl_dominates_compute:
        # Two distinct triggers can fire this; capture which in provenance
        # so downstream consumers (and Copilot-style reviewers) can tell
        # the low-overlap signal from the NCCL-dominance signal.
        triggers = []
        if low_overlap:
            triggers.append("low_overlap")
        if nccl_dominates_compute:
            triggers.append("nccl_exceeds_compute")
        # Build the label from whichever trigger(s) fired so the
        # human-readable summary reflects the actual signal — citing
        # "overlap 100%" when only nccl_exceeds_compute triggered would
        # be misleading.
        if low_overlap and nccl_dominates_compute:
            label_text = (
                f"Communication-bound (overlap {overlap_pct:.0f}%, "
                f"NCCL {total_nccl_ms:.0f}ms vs compute {compute_only_ms:.0f}ms)"
            )
        elif low_overlap:
            label_text = (
                f"Communication-bound: low overlap {overlap_pct:.0f}% "
                f"(NCCL {nccl_only_ms:.0f}ms unhidden)"
            )
        else:
            label_text = (
                f"Communication-bound: NCCL dominates "
                f"({total_nccl_ms:.0f}ms NCCL vs {compute_only_ms:.0f}ms compute)"
            )
        _emit(
            finding_id="profile_comm_bound",
            label=label_text,
            severity="warning",
            category="communication",
            values={
                "overlap_pct": round(overlap_pct, 2),
                "nccl_only_ms": round(nccl_only_ms, 2),
                "total_nccl_ms": round(total_nccl_ms, 2),
                "compute_only_ms": round(compute_only_ms, 2),
                "dominant_collective": nccl.get("dominant_type", "unknown"),
                "overlap_threshold_pct": _COMM_BOUND_OVERLAP_PCT,
            },
            units={
                "overlap_pct": "percent",
                "nccl_only_ms": "ms",
                "total_nccl_ms": "ms",
                "compute_only_ms": "ms",
                "overlap_threshold_pct": "percent",
            },
            note=(
                f"Compute/NCCL overlap is {overlap_pct:.0f}% (threshold "
                f"{int(_COMM_BOUND_OVERLAP_PCT)}%); NCCL total {total_nccl_ms:.0f}ms "
                f"vs compute-only {compute_only_ms:.0f}ms. Dominant collective: "
                f"{nccl.get('dominant_type', 'unknown')}."
            ),
            explanation=_COMM_EXPLANATION,
            actions=_COMM_ACTIONS,
            provenance_extra={"triggers": triggers},
        )

    # ── 4. GPU idle dominant ────────────────────────────────────────
    idle = m.get("idle", {}) or {}
    idle_pct = float(idle.get("idle_pct", 0) or 0)
    if idle_pct > _IDLE_DOMINANT_PCT:
        gap_count = int(idle.get("gap_count", 0) or 0)
        total_idle_ms = float(idle.get("total_idle_ms", 0) or 0)
        _emit(
            finding_id="profile_idle_dominant",
            label=f"GPU idle {idle_pct:.0f}% of profile ({gap_count} gaps)",
            severity="warning",
            category="idle",
            values={
                "idle_pct": round(idle_pct, 2),
                "gap_count": gap_count,
                "total_idle_ms": round(total_idle_ms, 2),
                "threshold_pct": _IDLE_DOMINANT_PCT,
            },
            units={
                "idle_pct": "percent",
                "total_idle_ms": "ms",
                "threshold_pct": "percent",
            },
            note=(
                f"GPU idle for {idle_pct:.0f}% of profile across {gap_count} gaps "
                f"({total_idle_ms:.0f}ms total). Threshold {int(_IDLE_DOMINANT_PCT)}%."
            ),
            explanation=_IDLE_EXPLANATION,
            actions=_IDLE_ACTIONS,
        )

    # ── 5. Kernel hotspot ───────────────────────────────────────────
    top_k = m.get("top_kernels", []) or []
    total_kernel_ms = float(m.get("total_kernel_ms", 0) or 0)
    if top_k and total_kernel_ms > 0:
        top_ms = float(top_k[0].get("total_ms", 0) or 0)
        top_pct = 100.0 * top_ms / total_kernel_ms
        if top_pct > _KERNEL_HOTSPOT_PCT:
            kname = top_k[0].get("name", "<unknown>")
            count = int(top_k[0].get("count", 0) or 0)
            _emit(
                finding_id="profile_kernel_hotspot",
                label=f"Kernel hotspot: {kname[:48]} ({top_pct:.0f}%)",
                severity="warning",
                category="compute",
                values={
                    "kernel_name": kname,
                    "kernel_total_ms": round(top_ms, 2),
                    "kernel_invocations": count,
                    "pct_of_total_kernel_ms": round(top_pct, 2),
                    "threshold_pct": _KERNEL_HOTSPOT_PCT,
                },
                units={
                    "kernel_total_ms": "ms",
                    "pct_of_total_kernel_ms": "percent",
                    "threshold_pct": "percent",
                },
                note=(
                    f"Kernel '{kname}' is {top_pct:.0f}% of total kernel time "
                    f"({top_ms:.0f}ms over {count} invocations). Threshold "
                    f"{int(_KERNEL_HOTSPOT_PCT)}%."
                ),
                explanation=_HOTSPOT_EXPLANATION,
                actions=_HOTSPOT_ACTIONS,
            )

    # ── 6. Iteration variance spike ─────────────────────────────────
    nvtx = m.get("nvtx", {}) or {}
    median_ms = float(nvtx.get("median_iter_ms", 0) or 0)
    slowest_ms = float(nvtx.get("slowest_iter_ms", 0) or 0)
    if median_ms > 0 and slowest_ms > median_ms * _ITER_VARIANCE_SPIKE_RATIO:
        ratio = slowest_ms / median_ms
        iter_count = int(nvtx.get("iteration_count", 0) or 0)
        _emit(
            finding_id="profile_iteration_variance_spike",
            label=f"Iteration variance: slowest {ratio:.1f}× median",
            severity="warning",
            category="nvtx",
            values={
                "median_iter_ms": round(median_ms, 2),
                "slowest_iter_ms": round(slowest_ms, 2),
                "ratio": round(ratio, 2),
                "iteration_count": iter_count,
                "threshold_ratio": _ITER_VARIANCE_SPIKE_RATIO,
            },
            units={
                "median_iter_ms": "ms",
                "slowest_iter_ms": "ms",
                "ratio": "x",
                "threshold_ratio": "x",
            },
            note=(
                f"Slowest steady-state iteration ({slowest_ms:.0f}ms) is "
                f"{ratio:.1f}× the median ({median_ms:.0f}ms) across "
                f"{iter_count} iterations. Threshold {_ITER_VARIANCE_SPIKE_RATIO}×."
            ),
            explanation=_ITER_SPIKE_EXPLANATION,
            actions=_ITER_SPIKE_ACTIONS,
        )

    # ── 7. Insufficient NVTX coverage ───────────────────────────────
    has_nvtx = bool(nvtx.get("has_nvtx", False))
    iter_count = int(nvtx.get("iteration_count", 0) or 0)
    if not has_nvtx or iter_count < _MIN_ITERATIONS_FOR_NVTX_COVERAGE:
        reason = "no NVTX annotations" if not has_nvtx else (
            f"only {iter_count} iteration(s) detected "
            f"(<{_MIN_ITERATIONS_FOR_NVTX_COVERAGE} required for steady-state)"
        )
        _emit(
            finding_id="profile_insufficient_nvtx_coverage",
            label=f"Insufficient NVTX coverage: {reason}",
            severity="info",
            category="profile_quality",
            values={
                "has_nvtx": has_nvtx,
                "iteration_count": iter_count,
                "min_iterations": _MIN_ITERATIONS_FOR_NVTX_COVERAGE,
            },
            units={},
            note=(
                f"Profile cannot anchor iteration / region analysis: {reason}. "
                "Downstream skills (iteration_timing, nvtx_kernel_map) will "
                "have reduced fidelity."
            ),
            explanation=_NVTX_COVERAGE_EXPLANATION,
            actions=_NVTX_COVERAGE_ACTIONS,
        )

    return findings


def _format(rows):
    if not rows:
        return "(No manifest data)"
    m = rows[0]
    lines = ["══ Profile Health Manifest ══"]

    fp = m.get("fingerprint")
    if fp:
        dist_str = "Distributed: yes" if fp.get("distributed") else "Distributed: no"
        mn_str = "Multi-node: yes" if fp.get("multi_node") else "Multi-node: no"
        lines.append(f"  Framework:    {fp.get('framework', 'Unknown')} ({dist_str}, {mn_str})")

    lines.append(f"  GPU:          {m.get('gpu', '?')}")
    lines.append(f"  Profile span: {m.get('profile_span_ms', 0):.1f}ms")

    dq = m.get("data_quality", {})
    overhead_pct_raw = dq.get("overhead_pct_raw", dq.get("overhead_pct", 0))
    if overhead_pct_raw >= 0.1:
        lines.append(
            f"  ⚠️ Profiler Overhead: {dq.get('profiler_overhead_ms', 0):.1f}ms ({dq.get('overhead_pct', overhead_pct_raw)}% of span)"
        )

    # Top kernels
    lines.append("")
    lines.append("  Top Kernels:")
    for k in m.get("top_kernels", []):
        lines.append(f"    {k['name'][:50]:<52s}  {k['total_ms']:>8.1f}ms  ×{k['count']}")

    # Overlap
    ov = m.get("overlap", {})
    if ov:
        lines.append("")
        lines.append("  Compute/NCCL Overlap:")
        err = ov.get("error")
        if err:
            # Preserve and surface overlap computation errors instead of showing 0.0ms metrics.
            if isinstance(err, dict):
                msg = err.get("message") or str(err)
            else:
                msg = str(err)
            lines.append(f"    ERROR: {msg}")
        else:
            lines.append(f"    Compute only: {ov.get('compute_only_ms', 0):.1f}ms")
            lines.append(f"    NCCL only:    {ov.get('nccl_only_ms', 0):.1f}ms")
            lines.append(f"    Overlap:      {ov.get('overlap_pct', 0)}%")

    # NCCL
    nccl = m.get("nccl", {})
    if nccl.get("streams", 0) > 0:
        lines.append("")
        lines.append("  NCCL Summary:")
        lines.append(f"    Streams: {nccl['streams']}, Collectives: {nccl['collectives']}")
        lines.append(
            f"    Dominant: {nccl.get('dominant_type', '?')} ({nccl.get('dominant_pct', 0)}%)"
        )
        lines.append(f"    Total: {nccl.get('total_nccl_ms', 0):.1f}ms")

    comm = m.get("communicators", {})
    if comm.get("communicators", 0) > 0:
        lines.append("")
        lines.append("  NCCL Communicators:")
        lines.append(
            f"    Communicators: {comm.get('communicators', 0)}, grouped rows: {comm.get('collective_rows', 0)}"
        )
        lines.append(
            f"    Dominant: {comm.get('dominant_collective', '?')} / {comm.get('dominant_dimension', '?')}"
        )
        if comm.get("low_efficiency_count", 0):
            lines.append(f"    Low efficiency groups: {comm.get('low_efficiency_count', 0)}")

    # Idle
    idle = m.get("idle", {})
    sync = m.get("sync", {})
    if idle.get("gap_count", 0) > 0:
        lines.append("")
        lines.append(f"  GPU Idle: {idle['gap_count']} gaps, {idle.get('idle_pct', 0)}% of profile")
    if sync.get("total_sync_wall_ms"):
        lines.append("")
        lines.append(
            f"  CPU Sync Block: {sync.get('total_sync_wall_ms', 0):.1f}ms ({sync.get('sync_density_pct', 0)}% of profile)"
        )

    # Root causes
    rcs = m.get("root_causes", [])
    if rcs:
        lines.append("")
        lines.append(f"  Root Causes ({m.get('root_cause_count', 0)} findings):")
        for rc in rcs:
            sev_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(rc["severity"], "⚪")
            lines.append(f"    {sev_icon} {rc['pattern']}")

    # NVTX summary
    nvtx = m.get("nvtx", {})
    if nvtx.get("has_nvtx"):
        lines.append("")
        iter_count = nvtx.get("iteration_count")
        if iter_count:
            median_ms = nvtx.get("median_iter_ms", 0)
            slowest_ms = nvtx.get("slowest_iter_ms", 0)
            lines.append(f"  NVTX Iterations: {iter_count} detected, median {median_ms:.1f}ms, slowest {slowest_ms:.1f}ms")
        top = nvtx.get("top_regions", [])
        if top:
            lines.append("  Top NVTX Regions:")
            for r in top:
                lines.append(f"    {r['name'][:48]:<50s}  {r['total_ms']:>8.1f}ms  ×{r['count']}")

    # Bottleneck
    bn = m.get("suspected_bottleneck", "")
    if bn:
        lines.append("")
        lines.append(f"  ⚡ Suspected bottleneck: {bn}")

    return "\n".join(lines)


SKILL = Skill(
    name="profile_health_manifest",
    title="Profile Health Manifest",
    description=(
        "One-shot profile health summary for AI agents. Returns a compact JSON manifest "
        "covering GPU info, top kernels, compute/NCCL overlap, NCCL summary, "
        "communicator-aware NCCL hints, idle gaps, root cause findings, and NVTX summary "
        "(top regions + iteration count/median/slowest) — all in a single call. "
        "If Profiler Overhead is >1%, advise the user to use torch.cuda.profiler.start/stop() "
        "and --capture-range=cudaProfilerApi instead of full-script profiling. "
        "Use this as the FIRST skill to call on any new profile. "
        "The nvtx.iteration_count, nvtx.median_iter_ms, and nvtx.slowest_iter_ms fields "
        "let you skip the first iteration_timing call for Mode 5 and Mode 9."
    ),
    category="utility",
    execute_fn=_execute,
    format_fn=_format,
    to_findings_fn=_to_findings,
    params=[
        SkillParam("device", "GPU device ID", "int", False, 0),
    ],
    tags=["manifest", "summary", "health", "overview", "agent", "triage"],
)
