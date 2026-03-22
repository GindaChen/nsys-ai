"""Root cause pattern matcher.

Programmatically detects known GPU performance anti-patterns from
the Book of Root Causes using existing skill data and direct SQL.

Each pattern has:
  - pattern: canonical root cause name (the unique identifier)
  - check: function that examines skill outputs and returns match info
  - severity: critical / warning / info
  - recommendation: actionable fix suggestion

This is a Python-level skill that runs other skills internally
to gather evidence, then matches against known patterns.
"""

import logging
import sqlite3

import duckdb

from nsys_ai.sql_compat import sqlite_to_duckdb

from ..base import Skill, SkillParam, _resolve_activity_tables

_log = logging.getLogger(__name__)


def _execute(conn: sqlite3.Connection, **kwargs):
    """Run all pattern matchers against the profile."""
    findings = []

    # Gather evidence from skills (forward trim kwargs)

    # Top kernels (request a larger slice to reduce bias in hotspot detection)
    top_kernels_data = _safe_execute("top_kernels", conn, limit=1000, **kwargs)
    # GPU idle gaps
    idle_gaps_data = _safe_execute("gpu_idle_gaps", conn, **kwargs)
    # Overlap
    overlap_data = _safe_execute("overlap_breakdown", conn, **kwargs)
    # Kernel launch overhead
    launch_data = _safe_execute("kernel_launch_overhead", conn, **kwargs)

    # --- GPU Bubbles (Pipeline Stalls) ---
    if idle_gaps_data:
        # Extract summary from enriched gpu_idle_gaps output
        gap_summary = next((g for g in idle_gaps_data if g.get("_summary")), None)
        gap_rows = [g for g in idle_gaps_data if not g.get("_summary")]
        gap_threshold = int(kwargs.get("min_gap_ns", 1_000_000))
        large_gaps = [g for g in gap_rows if g.get("gap_ns", 0) > gap_threshold]
        if len(large_gaps) >= 3:
            total_idle_ms = (
                gap_summary.get("total_idle_ms", 0)
                if gap_summary
                else sum(g.get("gap_ns", 0) / 1e6 for g in large_gaps)
            )
            pct = gap_summary.get("pct_of_profile", 0) if gap_summary else 0

            # Build attribution-aware recommendation
            attr_counts: dict[str, int] = {}
            for g in large_gaps:
                cat = (g.get("attribution") or {}).get("category", "")
                if cat:
                    attr_counts[cat] = attr_counts.get(cat, 0) + 1
            dominant = max(attr_counts, key=attr_counts.get) if attr_counts else ""
            rec_map = {
                "synchronization": (
                    "Remove explicit cudaDeviceSynchronize / cudaStreamSynchronize. "
                    "Use event-based dependencies or CUDA graphs."
                ),
                "cpu_stall": (
                    "CPU is not feeding GPU fast enough. "
                    "Increase DataLoader num_workers & prefetch_factor, "
                    "or check for Python GIL contention."
                ),
                "memory_transfer": (
                    "Gaps caused by blocking memory transfers. "
                    "Use cudaMemcpyAsync with pinned memory and overlap with compute."
                ),
                "kernel_launch": (
                    "Kernel launch overhead dominates gaps. "
                    "Use torch.compile() to fuse ops, or CUDA graphs."
                ),
            }
            rec = rec_map.get(
                dominant,
                (
                    "Use CUDA graphs, overlap data loading with compute, "
                    "or replace explicit cudaDeviceSynchronize with events."
                ),
            )

            evidence = (
                f"{len(large_gaps)} gaps > {gap_threshold / 1e6:.1f}ms detected, "
                f"totaling {total_idle_ms:.1f}ms of idle time"
            )
            if pct > 0:
                evidence += f" ({pct}% of profile)"

            findings.append(
                {
                    "pattern": "GPU Bubbles (Pipeline Stalls)",
                    "severity": "warning",
                    "evidence": evidence,
                    "recommendation": rec,
                }
            )

    # --- NCCL Serialization ---
    if overlap_data and len(overlap_data) > 0:
        ov = overlap_data[0]
        if "error" not in ov:
            overlap_pct = ov.get("overlap_pct", 100)
            nccl_only = ov.get("nccl_only_ms", 0)
            total = ov.get("total_ms", 1)
            if nccl_only > 0 and overlap_pct < 30:
                # Run deeper diagnosis to determine WHY overlap is low
                diagnosis = _diagnose_low_overlap(conn, **kwargs)
                cause = diagnosis.get("cause", "general")

                rec_map = {
                    "same_stream": (
                        "NCCL and compute kernels share the same CUDA stream — "
                        "they are serialized by design. Move AllReduce to a dedicated "
                        "stream. In PyTorch DDP, check if find_unused_parameters=True "
                        "is forcing synchronization."
                    ),
                    "sync_after_nccl": (
                        "Explicit synchronization detected after NCCL operations. "
                        "Remove torch.cuda.synchronize() / cudaStreamSynchronize "
                        "after communication calls. Use non_blocking=True for transfers."
                    ),
                    "general": (
                        "Tune DDP bucket sizes (bucket_cap_mb), "
                        "ensure NCCL runs on separate stream, "
                        "consider gradient compression or FSDP."
                    ),
                }
                rec = rec_map.get(cause, rec_map["general"])

                evidence = (
                    f"NCCL overlap only {overlap_pct}%, "
                    f"NCCL-only time: {nccl_only:.1f}ms / {total:.1f}ms"
                )
                diag_detail = diagnosis.get("detail", "")
                if diag_detail:
                    evidence += f". Diagnosis: {diag_detail}"

                findings.append(
                    {
                        "pattern": "NCCL Serialization",
                        "severity": "critical",
                        "evidence": evidence,
                        "recommendation": rec,
                    }
                )

    # --- Excessive H2D Transfers ---
    mem_data = _safe_execute("memory_bandwidth", conn, **kwargs)
    if mem_data:
        h2d = [r for r in mem_data if r.get("copyKind") == 1]
        if h2d:
            h2d_ms = h2d[0].get("total_dur_ms", 0)
            if h2d_ms > 50:  # > 50ms of H2D is suspicious
                findings.append(
                    {
                        "pattern": "Excessive H2D Transfers",
                        "severity": "warning",
                        "evidence": (
                            f"H2D transfers: {h2d_ms:.1f}ms total, "
                            f"{h2d[0].get('total_mb', 0):.1f}MB, "
                            f"{h2d[0].get('op_count', 0)} ops, "
                            f"avg bandwidth {h2d[0].get('avg_bandwidth_gbps', 0):.1f} GB/s"
                        ),
                        "recommendation": (
                            "Use pin_memory=True in DataLoader, keep "
                            "model params on GPU, accumulate metrics on GPU."
                        ),
                    }
                )

    # --- H2D Distribution Pattern ---
    h2d_dist_data = _safe_execute("h2d_distribution", conn, **kwargs)
    if h2d_dist_data:
        h2d_pattern = next((r for r in h2d_dist_data if r.get("_pattern")), None)
        if h2d_pattern:
            ptype = h2d_pattern.get("type", "")
            if ptype == "spread_out":
                findings.append(
                    {
                        "pattern": "Continuous H2D Transfers",
                        "severity": "warning",
                        "evidence": h2d_pattern.get(
                            "detail", "H2D transfers detected in every step"
                        ),
                        "recommendation": (
                            "Use pin_memory=True in DataLoader, increase num_workers, "
                            "set prefetch_factor>=2, and ensure tensors are pre-staged on GPU. "
                            "Check if .cpu() / .item() calls in the loop are pulling data back to host."
                        ),
                    }
                )
            elif ptype == "spike":
                spike_secs = h2d_pattern.get("spike_seconds", [])
                findings.append(
                    {
                        "pattern": "H2D Transfer Spike",
                        "severity": "info",
                        "evidence": h2d_pattern.get("detail", "H2D spikes detected"),
                        "recommendation": (
                            f"Check timeline at second(s) {spike_secs} for unexpected "
                            f"data movement. May be checkpoint saving, dynamic batching, "
                            f"or model reloading."
                        ),
                    }
                )

    # --- Small Kernel Overhead ---
    if launch_data:
        # overhead_us > kernel_ms*1000 means overhead > kernel duration
        high_overhead = [
            e
            for e in launch_data
            if e.get("kernel_ms", 0) > 0 and e.get("overhead_us", 0) > e["kernel_ms"] * 1000
        ]
        if len(high_overhead) >= 5:
            findings.append(
                {
                    "pattern": "Small Kernel Overhead",
                    "severity": "warning",
                    "evidence": f"{len(high_overhead)} kernels with launch overhead > kernel duration",
                    "recommendation": (
                        "Use torch.compile() to fuse element-wise ops, "
                        "enable cudnn.benchmark, or use CUDA graphs."
                    ),
                }
            )

    # --- Kernel Hotspot ---
    if top_kernels_data and len(top_kernels_data) >= 2:
        # Compute percentage from total_ms since top_kernels doesn't have pct
        total_all_ms = sum(k.get("total_ms", 0) for k in top_kernels_data)
        if total_all_ms > 0:
            top_k = top_kernels_data[0]
            pct = (top_k.get("total_ms", 0) / total_all_ms) * 100
            if pct > 50:
                findings.append(
                    {
                        "pattern": "Kernel Hotspot",
                        "severity": "info",
                        "evidence": (
                            f"'{top_k.get('kernel_name', '?')}' accounts for {pct:.0f}% "
                            f"of time in the profiled top kernels "
                            f"({top_k.get('total_ms', 0):.1f}ms)"
                        ),
                        "recommendation": (
                            "Ensure shapes are multiples of 128 (H100) / 64 (A100), "
                            "use FlashAttention, or profile with NCU for details."
                        ),
                    }
                )

    # --- Compute-Communication Imbalance ---
    if overlap_data and len(overlap_data) > 0:
        ov = overlap_data[0]
        if "error" not in ov:
            compute_ms = ov.get("compute_only_ms", 0)
            nccl_ms_total = ov.get("nccl_only_ms", 0) + ov.get("overlap_ms", 0)
            if nccl_ms_total > 0 and compute_ms > 0:
                ratio = compute_ms / nccl_ms_total
                if ratio < 0.5:
                    findings.append(
                        {
                            "pattern": "Compute-Communication Imbalance",
                            "severity": "critical",
                            "evidence": (
                                f"Compute/NCCL ratio = {ratio:.2f} (healthy > 2.0). "
                                f"Compute: {compute_ms:.1f}ms, NCCL: {nccl_ms_total:.1f}ms"
                            ),
                            "recommendation": (
                                "Reduce tensor parallel degree (e.g. TP=4 → TP=1 "
                                "if model fits on one GPU), rebalance pipeline stages, "
                                "or pad sequences to uniform length."
                            ),
                        }
                    )

    # Use a bounded default limit to avoid huge result sets; allow caller override
    layer_kwargs = dict(kwargs)
    layer_kwargs.setdefault("limit", 500)
    layer_data = _safe_execute("nvtx_layer_breakdown", conn, **layer_kwargs)
    if layer_data and len(layer_data) >= 2:
        try:
            nccl_hotspot_pct = float(kwargs.get("nccl_hotspot_pct", 40.0))
        except (ValueError, TypeError):
            nccl_hotspot_pct = 40.0

        try:
            imbalance_ratio = float(kwargs.get("imbalance_ratio", 3.0))
        except (ValueError, TypeError):
            imbalance_ratio = 3.0

        findings += _check_layer_nccl_hotspot(layer_data, threshold_pct=nccl_hotspot_pct)
        findings += _check_pipeline_imbalance(layer_data, threshold_ratio=imbalance_ratio)

    # --- nsys anti-pattern checks (direct SQL) ---
    # These cover the 4 expert-rule recipes from nsys:
    # cuda_api_sync, cuda_memcpy_sync, cuda_memcpy_async, cuda_memset_sync
    findings += _check_sync_apis(conn, **kwargs)
    findings += _check_sync_memcpy(conn, **kwargs)
    findings += _check_pageable_memcpy(conn, **kwargs)
    findings += _check_sync_memset(conn, **kwargs)

    if not findings:
        findings.append(
            {
                "pattern": "No Known Anti-Patterns Detected",
                "severity": "info",
                "evidence": "All checks passed — profile looks healthy",
                "recommendation": "Consider deep-diving with NCU for fine-grained analysis.",
            }
        )

    return findings


# -----------------------------------------------------------------------
# Overlap diagnosis helper
# -----------------------------------------------------------------------


def _diagnose_low_overlap(conn: sqlite3.Connection, **kwargs) -> dict:
    """Diagnose why compute/NCCL overlap is low.

    Checks:
      1. Same-stream: NCCL and compute kernels on the same CUDA stream
      2. Sync-after-NCCL: explicit sync call shortly after NCCL launch

    Returns dict with 'cause' ('same_stream', 'sync_after_nccl', 'general')
    and 'detail' string.
    """
    tables = _resolve_activity_tables(conn)
    kernel_tbl = tables.get("kernel")
    runtime_tbl = tables.get("runtime")

    if not kernel_tbl:
        return {"cause": "general", "detail": ""}

    device = int(kwargs.get("device", 0))

    # --- Check 1: Same-stream detection ---
    try:
        same_stream_rows = conn.execute(
            f"""
            SELECT k.streamId,
                SUM(CASE WHEN s.value LIKE '%nccl%' OR s.value LIKE '%NCCL%'
                    THEN 1 ELSE 0 END) AS nccl_count,
                SUM(CASE WHEN NOT (s.value LIKE '%nccl%' OR s.value LIKE '%NCCL%')
                    THEN 1 ELSE 0 END) AS compute_count
            FROM {kernel_tbl} k
            JOIN StringIds s ON k.shortName = s.id
            WHERE k.deviceId = ?
            GROUP BY k.streamId
            HAVING nccl_count > 0 AND compute_count > 0
            """,
            (device,),
        ).fetchall()
        if same_stream_rows:
            streams = [str(r[0]) for r in same_stream_rows]
            return {
                "cause": "same_stream",
                "detail": (f"Stream(s) [{', '.join(streams)}] run both NCCL and compute kernels"),
            }
    except (sqlite3.Error, duckdb.Error) as e:
        _log.debug("_diagnose_low_overlap (same_stream): %s", e)

    # --- Check 2: Sync-after-NCCL detection ---
    if runtime_tbl:
        try:
            # Find sync nameIds
            sync_names = conn.execute(
                """
                SELECT id FROM StringIds
                WHERE value LIKE 'cudaStreamSynchronize%'
                   OR value LIKE 'cudaDeviceSynchronize%'
                """
            ).fetchall()
            if sync_names:
                sync_id_list = [r[0] for r in sync_names]
                sync_placeholders = ",".join("?" for _ in sync_id_list)

                # Single query: check if ANY sync call starts within 1ms
                # after ANY NCCL kernel's end time (avoids N+1 loop).
                found = conn.execute(sqlite_to_duckdb(
                    f"""
                    SELECT 1 FROM {kernel_tbl} k
                    JOIN StringIds s ON k.shortName = s.id
                    WHERE (s.value LIKE '%nccl%' OR s.value LIKE '%NCCL%')
                      AND k.deviceId = ?
                      AND EXISTS (
                          SELECT 1 FROM {runtime_tbl} r
                          WHERE r.nameId IN ({sync_placeholders})
                            AND r.start >= k.[end]
                            AND r.start <= k.[end] + 1000000
                      )
                    LIMIT 1
                    """
                ), [device] + sync_id_list).fetchone()
                if found:
                    return {
                        "cause": "sync_after_nccl",
                        "detail": (
                            "cudaStreamSynchronize/cudaDeviceSynchronize "
                            "detected immediately after NCCL kernel completion"
                        ),
                    }
        except (sqlite3.Error, duckdb.Error) as e:
            _log.debug("_diagnose_low_overlap (sync_after_nccl): %s", e)

    return {"cause": "general", "detail": ""}


# -----------------------------------------------------------------------
# Per-layer NVTX breakdown pattern checkers
# -----------------------------------------------------------------------


def _check_layer_nccl_hotspot(
    layer_data: list[dict], threshold_pct: float = 40.0
) -> list[dict]:
    """Detect when one NVTX region dominates total NCCL time.

    Args:
        threshold_pct: Fire when a region exceeds this % of total NCCL (default 40).
    """
    total_nccl = sum(r.get("nccl_ms", 0) for r in layer_data)
    if total_nccl <= 0:
        return []

    findings = []
    for r in layer_data:
        nccl_ms = r.get("nccl_ms", 0)
        if nccl_ms <= 0:
            continue
        pct = 100.0 * nccl_ms / total_nccl
        if pct > threshold_pct:
            path = r.get("nvtx_path", r.get("nvtx_region", ""))
            findings.append(
                {
                    "pattern": "Layer NCCL Hotspot",
                    "severity": "warning",
                    "evidence": (
                        f"'{r['nvtx_region']}' accounts for {pct:.0f}% of total NCCL time "
                        f"({nccl_ms:.1f}ms / {total_nccl:.1f}ms). "
                        f"Path: {path}"
                    ),
                    "recommendation": (
                        "Consider activation recomputation for this layer to pipeline "
                        "backward computation and NCCL communication, "
                        "or check if gradient bucketing can be rebalanced."
                    ),
                }
            )
    return findings


def _check_pipeline_imbalance(
    layer_data: list[dict], threshold_ratio: float = 3.0
) -> list[dict]:
    """Detect compute time imbalance across NVTX layers.

    Args:
        threshold_ratio: Fire when max/min compute ratio exceeds this (default 3.0).
    """
    # Only consider layers with measurable compute
    compute_layers = [
        r for r in layer_data
        if r.get("compute_ms", 0) > 0.01  # > 10µs
    ]
    if len(compute_layers) < 2:
        return []

    compute_times = [r["compute_ms"] for r in compute_layers]
    max_compute = max(compute_times)
    min_compute = min(ct for ct in compute_times if ct > 0)
    ratio = max_compute / min_compute if min_compute > 0 else 0

    if ratio < threshold_ratio:
        return []

    # Find the heaviest and lightest layers
    heaviest = max(compute_layers, key=lambda r: r["compute_ms"])
    lightest = min(compute_layers, key=lambda r: r["compute_ms"])

    heaviest_label = heaviest.get("nvtx_path") or heaviest.get("nvtx_region", "?")
    lightest_label = lightest.get("nvtx_path") or lightest.get("nvtx_region", "?")

    return [
        {
            "pattern": "Pipeline Imbalance",
            "severity": "warning",
            "evidence": (
                f"Compute time varies {ratio:.1f}× across layers. "
                f"Heaviest: '{heaviest_label}' ({heaviest['compute_ms']:.1f}ms), "
                f"lightest: '{lightest_label}' ({lightest['compute_ms']:.1f}ms)"
            ),
            "recommendation": (
                "Rebalance pipeline stage partitioning, "
                "or investigate if the heavy layer has suboptimal kernel configuration "
                "(e.g. too many small kernels, poor tiling)."
            ),
        }
    ]


# -----------------------------------------------------------------------
# nsys anti-pattern checkers — inline SQL for expert-rule recipe parity
# -----------------------------------------------------------------------


def _check_sync_apis(conn: sqlite3.Connection, **kwargs):
    """Detect excessive cuda*Synchronize calls.

    Uses a percentage-based threshold: sync time must exceed 2% of total
    GPU kernel time to avoid false positives from initialization phases.

    Note: Nsight exports use versioned API names (e.g. cudaDeviceSynchronize_v3020),
    so we use LIKE prefix matching via a two-step nameId resolution.
    """
    from ...sql_compat import sqlite_to_duckdb
    tables = _resolve_activity_tables(conn)
    runtime_tbl = tables.get("runtime")
    kernel_tbl = tables.get("kernel")
    if not runtime_tbl:
        return []

    try:
        # Step 1: resolve nameIds from StringIds (fast, tiny table)
        sync_names = conn.execute(sqlite_to_duckdb("""
            SELECT id, value FROM StringIds
            WHERE value LIKE 'cudaDeviceSynchronize%'
               OR value LIKE 'cudaStreamSynchronize%'
               OR value LIKE 'cudaEventSynchronize%'
               OR value LIKE 'cudaStreamWaitEvent%'
        """)).fetchall()
        if not sync_names:
            return []

        name_ids = [r[0] for r in sync_names]
        placeholders = ",".join(str(nid) for nid in name_ids)

        # Step 2: count sync calls by nameId (fast with index)
        rows = conn.execute(sqlite_to_duckdb(f"""
            SELECT nameId, COUNT(*) AS call_count,
                   SUM([end] - start) AS total_ns
            FROM {runtime_tbl}
            WHERE nameId IN ({placeholders})
            GROUP BY nameId
        """)).fetchall()
        if not rows:
            return []

        # Map nameId back to name
        id_to_name = {r[0]: r[1] for r in sync_names}
        total_sync_ns = sum(r[2] for r in rows)
        total_sync_ms = total_sync_ns / 1e6
        call_count = sum(r[1] for r in rows)
        # Strip version suffixes for cleaner display (cudaDeviceSynchronize_v3020 → cudaDeviceSynchronize)
        api_names = ", ".join(sorted({id_to_name[r[0]].split("_v")[0] for r in rows}))

        # Total GPU kernel time as baseline for percentage threshold
        total_gpu_ns = 0
        if kernel_tbl:
            gpu_row = conn.execute(sqlite_to_duckdb(f"SELECT SUM([end] - start) FROM {kernel_tbl}")).fetchone()
            total_gpu_ns = gpu_row[0] or 0 if gpu_row else 0

        # Percentage-based threshold: sync time > 2% of total GPU time
        # Also require absolute minimum of 1ms to filter trivial cases
        sync_pct = (total_sync_ns / total_gpu_ns * 100) if total_gpu_ns > 0 else 100
        if total_sync_ms >= 1.0 and sync_pct >= 2.0:
            return [
                {
                    "pattern": "Excessive Synchronization",
                    "severity": "warning",
                    "evidence": (
                        f"{call_count} sync calls totalling {total_sync_ms:.1f}ms "
                        f"({sync_pct:.1f}% of GPU time). APIs: {api_names}"
                    ),
                    "recommendation": (
                        "Remove .item()/.cpu() from the training loop, "
                        "use torch.cuda.set_sync_debug_mode(1) to find hidden syncs, "
                        "replace cudaDeviceSynchronize with event-based dependencies. "
                        "Run `nsys recipe cuda_api_sync <profile.nsys-rep>` for a detailed breakdown."
                    ),
                }
            ]
    except (sqlite3.Error, duckdb.Error) as e:
        _log.debug("root_cause_matcher (_check_sync_apis): %s", e)
    return []


def _check_sync_memcpy(conn: sqlite3.Connection, **kwargs):
    """Detect synchronous cudaMemcpy (not cudaMemcpyAsync).

    Synchronous memcpy blocks the host until the transfer completes,
    preventing CPU/GPU overlap.

    Note: Nsight exports use versioned API names (e.g. cudaMemcpy_v3020).
    We match any name starting with 'cudaMemcpy' but NOT 'cudaMemcpyAsync'.
    """
    from ...sql_compat import sqlite_to_duckdb
    tables = _resolve_activity_tables(conn)
    runtime_tbl = tables.get("runtime")
    memcpy_tbl = tables.get("memcpy")
    if not runtime_tbl or not memcpy_tbl:
        return []

    try:
        # Step 1: find nameIds for sync cudaMemcpy (NOT async)
        sync_names = conn.execute(sqlite_to_duckdb("""
            SELECT id, value FROM StringIds
            WHERE value LIKE 'cudaMemcpy%'
              AND value NOT LIKE 'cudaMemcpyAsync%'
        """)).fetchall()
        if not sync_names:
            return []

        name_ids = [r[0] for r in sync_names]
        placeholders = ",".join(str(nid) for nid in name_ids)

        # Step 2: find memcpy ops correlated with sync runtime calls
        row = conn.execute(sqlite_to_duckdb(f"""
            SELECT COUNT(*) AS count,
                   COALESCE(SUM(m.bytes), 0) AS total_bytes,
                   COALESCE(SUM(m.[end] - m.start), 0) AS total_ns
            FROM {runtime_tbl} r
            JOIN {memcpy_tbl} m ON r.correlationId = m.correlationId
            WHERE r.nameId IN ({placeholders})
        """)).fetchone()
        if not row or row[0] == 0:
            return []

        count, total_bytes, total_ns = row
        total_ms = total_ns / 1e6
        total_mb = total_bytes / 1e6

        return [
            {
                "pattern": "Synchronous Memcpy",
                "severity": "warning",
                "evidence": (
                    f"{count} sync cudaMemcpy calls: {total_mb:.1f}MB in {total_ms:.1f}ms. "
                    f"These block the host thread."
                ),
                "recommendation": (
                    "Replace cudaMemcpy with cudaMemcpyAsync + pinned memory. "
                    "Use pin_memory=True in DataLoader and non_blocking=True in .to(device). "
                    "Run `nsys recipe cuda_memcpy_sync <profile.nsys-rep>` for a detailed breakdown."
                ),
            }
        ]
    except (sqlite3.Error, duckdb.Error) as e:
        _log.debug("root_cause_matcher (_check_sync_memcpy): %s", e)
    return []


def _check_pageable_memcpy(conn: sqlite3.Connection, **kwargs):
    """Detect async memcpy using pageable (non-pinned) memory.

    When cudaMemcpyAsync is called with pageable memory, the driver silently
    falls back to a synchronous copy, defeating the purpose of async.

    Memory kind values (Nsight CUPTI schema):
      0 = Unknown, 1 = Pageable, 2 = Device, 3 = Array,
      4 = Managed, 5 = Device Static, 6 = Managed Static, 7 = Pinned
    Source: CUPTI_ACTIVITY_KIND_MEMCPY table schema, Nsight Systems export.
    """
    from ...sql_compat import sqlite_to_duckdb
    tables = _resolve_activity_tables(conn)
    memcpy_tbl = tables.get("memcpy")
    if not memcpy_tbl:
        return []

    try:
        row = conn.execute(sqlite_to_duckdb(f"""
            SELECT COUNT(*) AS pageable_count,
                   COALESCE(SUM(bytes), 0) AS total_bytes,
                   COALESCE(SUM([end] - start), 0) AS total_ns
            FROM {memcpy_tbl}
            WHERE srcKind = 1 OR dstKind = 1
        """)).fetchone()
        if not row or row[0] == 0:
            return []

        count, total_bytes, total_ns = row
        total_ms = total_ns / 1e6
        total_mb = total_bytes / 1e6

        return [
            {
                "pattern": "Pageable Memory in Async Memcpy",
                "severity": "warning",
                "evidence": (
                    f"{count} memcpy ops using pageable memory: {total_mb:.1f}MB in "
                    f"{total_ms:.1f}ms. Pageable → async memcpy silently becomes sync."
                ),
                "recommendation": (
                    "Use pinned (page-locked) memory: cudaMallocHost() / "
                    "pin_memory=True in DataLoader. This enables true async H2D overlap. "
                    "Run `nsys recipe cuda_memcpy_async <profile.nsys-rep>` for details on pageable fallback."
                ),
            }
        ]
    except (sqlite3.Error, duckdb.Error) as e:
        _log.debug("root_cause_matcher (_check_pageable_memcpy): %s", e)
    return []


def _check_sync_memset(conn: sqlite3.Connection, **kwargs):
    """Detect synchronous cudaMemset (not cudaMemsetAsync).

    Synchronous memset blocks the host. Usually a minor issue but
    can add up in tight loops.

    Note: Nsight exports use versioned API names (e.g. cudaMemset_v3020).
    We match any name starting with 'cudaMemset' but NOT 'cudaMemsetAsync'.
    """
    from ...sql_compat import sqlite_to_duckdb
    tables = _resolve_activity_tables(conn)
    runtime_tbl = tables.get("runtime")
    memset_tbl = tables.get("memset")
    if not runtime_tbl or not memset_tbl:
        return []

    try:
        # Step 1: find nameIds for sync cudaMemset (NOT async)
        sync_names = conn.execute(sqlite_to_duckdb("""
            SELECT id, value FROM StringIds
            WHERE value LIKE 'cudaMemset%'
              AND value NOT LIKE 'cudaMemsetAsync%'
        """)).fetchall()
        if not sync_names:
            return []

        name_ids = [r[0] for r in sync_names]
        placeholders = ",".join(str(nid) for nid in name_ids)

        # Step 2: find memset ops correlated with sync runtime calls
        row = conn.execute(sqlite_to_duckdb(f"""
            SELECT COUNT(*) AS count,
                   COALESCE(SUM(ms.[end] - ms.start), 0) AS total_ns
            FROM {runtime_tbl} r
            JOIN {memset_tbl} ms ON r.correlationId = ms.correlationId
            WHERE r.nameId IN ({placeholders})
        """)).fetchone()
        if not row or row[0] == 0:
            return []

        count, total_ns = row
        total_ms = total_ns / 1e6

        return [
            {
                "pattern": "Synchronous Memset",
                "severity": "info",
                "evidence": (
                    f"{count} sync cudaMemset calls: {total_ms:.2f}ms total. "
                    f"These block the host thread."
                ),
                "recommendation": (
                    "Replace cudaMemset with cudaMemsetAsync on the appropriate stream. "
                    "Run `nsys recipe cuda_memset_sync <profile.nsys-rep>` for a detailed breakdown."
                ),
            }
        ]
    except (sqlite3.Error, duckdb.Error) as e:
        _log.debug("root_cause_matcher (_check_sync_memset): %s", e)
    return []


# -----------------------------------------------------------------------


def _safe_execute(skill_name, conn: sqlite3.Connection, **kwargs):
    """Execute a skill, returning [] on any error."""
    from ...skills.registry import get_skill

    error_types = [sqlite3.Error]
    try:
        import duckdb
        error_types.extend([duckdb.Error, duckdb.CatalogException])
    except ImportError:
        pass

    try:
        skill = get_skill(skill_name)
        if skill is None:
            return []
        return skill.execute(conn, **kwargs)
    except tuple(error_types) as e:
        _log.debug("root_cause_matcher (%s): %s", skill_name, e)
        return []


def _format(rows):
    if not rows:
        return "(No patterns checked)"
    lines = ["── Root Cause Pattern Analysis ──"]
    for f in rows:
        icon = {"critical": "🔴", "warning": "🟡", "info": "🟢"}.get(f["severity"], "⚪")
        lines.append(f"\n{icon} {f['pattern']}")
        lines.append(f"  Evidence: {f['evidence']}")
        lines.append(f"  Fix: {f['recommendation']}")
    return "\n".join(lines)


SKILL = Skill(
    name="root_cause_matcher",
    title="Root Cause Pattern Matcher",
    description=(
        "Automatically detects known GPU performance anti-patterns from the "
        "Book of Root Causes: GPU bubbles, NCCL serialization, kernel hotspots, "
        "small kernel overhead, compute-communication imbalance, excessive "
        "synchronization, synchronous memcpy/memset, pageable memory in async "
        "transfers. Returns matched patterns with evidence and fix recommendations."
    ),
    category="analysis",
    execute_fn=_execute,
    format_fn=_format,
    params=[SkillParam("device", "GPU device ID", "int", False, 0)],
    tags=[
        "root-cause",
        "pattern",
        "diagnosis",
        "analysis",
        "recommendation",
        "sync",
        "memcpy",
        "memset",
        "anti-pattern",
    ],
)
