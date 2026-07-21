"""Critical-path bound-class analysis.

Classifies a run (or a trimmed iteration) as ``cpu-bound``,
``gpu-compute-bound``, ``comm-bound``, or ``mixed`` by estimating the
critical path through the trace rather than merely partitioning wall time
by what happened to be running.

Approach (a testable, best-effort approximation of a longest-path walk):

The trace carries no dependency edges, so the true longest path is not
computed. The busiest GPU device timeline is used as a *proxy* for the
critical-path spine — a defensible approximation for single-GPU and
compute-dominated runs, but only that. The class is reported with a
confidence and refuses to commit on a near-tie or on too little on-path time.

* Take the busiest GPU device timeline as the spine of the critical path.
* At every instant on that spine, the time is attributed to exactly one of
  three on-path categories:

  - ``gpu-compute`` — a compute kernel is running. Following the HTA
    convention, communication that overlaps compute is *hidden* (off the
    critical path), so overlap time counts as compute.
  - ``comm`` — only communication (NCCL) is running with no compute to hide
    it. This is the *exposed* collective time.
  - ``cpu`` — the GPU is idle, i.e. waiting on the host (dispatch lag,
    synchronization, dataloader / GIL). GPU-idle gaps are the cpu-bound
    signal.

Because ``compute-only + nccl-only + overlap + idle == span``, the three
on-path buckets partition the critical-path time exactly. The dominant
bucket is the bound class; when the margin between the top bucket and the
runner-up is small the result is ``mixed`` with low confidence — a
deliberate refusal to emit a confident-but-wrong class. Confidence is that
margin.

This reuses :func:`nsys_ai.overlap.overlap_analysis` (compute/comm/idle
decomposition), :mod:`sync_cost_analysis` and :mod:`cpu_gpu_pipeline` (CPU
attribution of the idle bucket), so it does not duplicate overlap or idle
logic that already exists.
"""

import logging

from nsys_ai.connection import DB_ERRORS, is_safe_identifier, wrap_connection

from ..base import Skill, SkillParam

_log = logging.getLogger(__name__)

# Bound-class labels (the public vocabulary of this skill).
_CLASS_CPU = "cpu-bound"
_CLASS_GPU = "gpu-compute-bound"
_CLASS_COMM = "comm-bound"
_CLASS_MIXED = "mixed"
_CLASS_NA = "n/a"

# Internal category -> public bound-class label.
_CATEGORY_TO_CLASS = {
    "cpu": _CLASS_CPU,
    "gpu_compute": _CLASS_GPU,
    "comm": _CLASS_COMM,
}

# Internal category -> annotation.FindingCategory (step-time bucket).
_CATEGORY_TO_FINDING = {
    "cpu": "idle",
    "gpu_compute": "compute",
    "comm": "communication",
}

# Minimum margin (share of critical path) between the dominant category and
# the runner-up required to commit to a bound class. Below this the run is
# reported as ``mixed`` so a near-tie never forces a confident-but-wrong
# winner.
_MARGIN_THRESHOLD = 0.15

# The dominant category must also own at least this share of the critical path
# to commit to a class. With three buckets a bare margin gate alone can crown a
# ~43% plurality; requiring an outright majority keeps "gpu-compute-bound" from
# being emitted when compute is under half the path.
_MIN_DOMINANT_SHARE = 0.5

# Below this much on-path time there is too little signal to classify — a
# handful of microseconds of kernels should not yield a confident verdict.
_MIN_CRITICAL_PATH_MS = 1.0

# A committed class whose margin is only modestly above threshold is flagged
# tentative in the note (the class is still emitted, but not oversold).
_TENTATIVE_MARGIN = 0.25

# How many on-path kernels / collectives to surface.
_TOP_N = 5


def _busiest_device(adapter, kernel_table: str) -> int | None:
    """Return the device with the most total kernel busy time, or None."""
    if not is_safe_identifier(kernel_table):
        return None
    try:
        cur = adapter.execute(
            f"SELECT deviceId, SUM([end] - start) AS busy "  # noqa: S608 — validated identifier
            f"FROM {kernel_table} GROUP BY deviceId ORDER BY busy DESC LIMIT 1"
        )
        row = cur.fetchone()
    except DB_ERRORS:
        return None
    if not row or row[0] is None:
        return None
    return int(row[0])


def _top_on_path(adapter, kernel_table: str, device: int, trim):
    """Return (top_compute_kernels, top_collectives) lists with selections.

    Each entry aggregates a kernel by name and carries a representative time
    selection (the [min start, max end] window of that kernel's instances) so
    a caller can jump to it on the timeline.
    """
    from ...overlap import classify_kernel

    if not is_safe_identifier(kernel_table):
        return [], []

    conds = ["k.deviceId = ?"]
    params: list = [device]
    if trim is not None:
        conds.append("k.start >= ? AND k.[end] <= ?")
        params.extend([int(trim[0]), int(trim[1])])
    where_clause = " AND ".join(conds)

    sql = (
        f"SELECT s.value AS name, "  # noqa: S608 — validated identifier, bound params
        f"COUNT(*) AS instances, "
        f"SUM(k.[end] - k.start) AS total_ns, "
        f"MIN(k.start) AS start_ns, MAX(k.[end]) AS end_ns "
        f"FROM {kernel_table} k JOIN StringIds s ON k.shortName = s.id "
        f"WHERE {where_clause} "
        f"GROUP BY s.value"
    )
    try:
        cur = adapter.execute(sql, params)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    except DB_ERRORS as exc:
        _log.debug("critical_path top-on-path query failed: %s", exc, exc_info=True)
        return [], []

    compute: list[dict] = []
    collectives: list[dict] = []
    for r in rows:
        name = r["name"]
        entry = {
            "name": name,
            "instances": int(r["instances"] or 0),
            "total_ms": round((r["total_ns"] or 0) / 1e6, 3),
            "start_ns": int(r["start_ns"]) if r["start_ns"] is not None else None,
            "end_ns": int(r["end_ns"]) if r["end_ns"] is not None else None,
        }
        if classify_kernel(name).startswith("nccl_"):
            collectives.append(entry)
        else:
            compute.append(entry)

    compute.sort(key=lambda e: e["total_ms"], reverse=True)
    collectives.sort(key=lambda e: e["total_ms"], reverse=True)
    return compute[:_TOP_N], collectives[:_TOP_N]


def _cpu_attribution(conn, device: int, trim) -> dict:
    """Best-effort attribution of the GPU-idle (cpu) bucket to host causes.

    Pulls total host-side synchronization wall time and CPU->GPU dispatch
    starvation from the existing skills so the cpu-bound classification is
    grounded in why the GPU was idle, not just that it was. Degrades to an
    empty dict when the supporting tables are absent (legacy exports, mock
    fixtures) rather than raising.
    """
    attribution: dict = {}
    passthrough = {}
    if trim is not None:
        passthrough = {"trim_start_ns": int(trim[0]), "trim_end_ns": int(trim[1])}

    try:
        from ...skills.registry import get_skill

        sync_skill = get_skill("sync_cost_analysis")
        if sync_skill is not None:
            sync_rows = sync_skill.execute(conn, **passthrough)
            if sync_rows and "error" not in sync_rows[0]:
                attribution["sync_wall_ms"] = sync_rows[0].get("total_sync_wall_ms", 0.0)
    except Exception:
        _log.debug("critical_path sync attribution failed", exc_info=True)

    try:
        from ...skills.registry import get_skill

        pipe_skill = get_skill("cpu_gpu_pipeline")
        if pipe_skill is not None:
            pipe_rows = pipe_skill.execute(conn, device=device, **passthrough)
            starvation = sum(int(r.get("starvation_events", 0) or 0) for r in pipe_rows)
            attribution["dispatch_starvation_events"] = starvation
    except Exception:
        _log.debug("critical_path dispatch attribution failed", exc_info=True)

    return attribution


def _degenerate(note: str, device) -> list[dict]:
    """Return a single low-confidence result for a trace we cannot classify."""
    return [
        {
            "bound_class": _CLASS_NA,
            "confidence": 0.0,
            "critical_path_ms": 0.0,
            "device": device,
            "breakdown": {
                "gpu_compute_ms": 0.0,
                "comm_ms": 0.0,
                "cpu_ms": 0.0,
                "gpu_compute_share": 0.0,
                "comm_share": 0.0,
                "cpu_share": 0.0,
            },
            "top_compute_kernels": [],
            "top_collectives": [],
            "cpu_attribution": {},
            "note": note,
        }
    ]


def _execute(conn, **kwargs) -> list[dict]:
    from ...overlap import overlap_analysis
    from ...profile import Profile

    prof = Profile._from_conn(conn)
    adapter = wrap_connection(conn)
    kernel_table = prof.schema.kernel_table
    if not kernel_table:
        return _degenerate("No kernel table in profile — cannot estimate critical path.", None)

    # Resolve the device (the critical-path spine). When unspecified, pick the
    # busiest device so a single-GPU profile classifies without the caller
    # having to know the device id.
    device = kwargs.get("device")
    if device is None:
        device = _busiest_device(adapter, kernel_table)
        if device is None:
            return _degenerate("No GPU kernels found — nothing on the critical path.", None)
    else:
        device = int(device)

    trim = None
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    if trim_start is not None and trim_end is not None:
        trim = (int(trim_start), int(trim_end))

    ov = overlap_analysis(prof, device, trim=trim)
    if "error" in ov:
        return _degenerate(
            f"Overlap analysis unavailable for device {device}: {ov['error']}", device
        )

    # On-path buckets. Overlap counts as compute (HTA convention: hidden
    # communication is off the critical path); exposed NCCL is comm; GPU-idle
    # is the host-wait (cpu) bucket.
    gpu_compute_ms = float(ov.get("compute_only_ms", 0.0)) + float(ov.get("overlap_ms", 0.0))
    comm_ms = float(ov.get("nccl_only_ms", 0.0))
    cpu_ms = float(ov.get("idle_ms", 0.0))

    # `overlap_analysis` measures idle only *between* kernels, over the span of
    # the kernels themselves (MIN start .. MAX end). When a trim window is
    # given, GPU-idle before the first / after the last contained kernel is
    # real host-wait inside the iteration but falls outside that span, so it is
    # invisible to `idle_ms`. Fold that edge idle into the cpu bucket, or a step
    # that stalls on the host then computes back-to-back is misread as
    # compute-bound. (Without a trim there is no window bound, so edge idle is
    # genuinely undefined and left uncounted.)
    if trim is not None:
        span_start = ov.get("span_start_ns")
        span_end = ov.get("span_end_ns")
        if span_start is not None and span_end is not None:
            edge_idle_ms = ((trim[1] - trim[0]) - (int(span_end) - int(span_start))) / 1e6
            if edge_idle_ms > 0:
                cpu_ms += edge_idle_ms

    # Denominator is the sum of the on-path buckets so the shares always sum
    # to exactly 1.0 (avoids drift from independent per-bucket rounding and
    # from idle being clamped to >= 0 upstream).
    critical_path_ms = gpu_compute_ms + comm_ms + cpu_ms
    if critical_path_ms <= 0:
        return _degenerate(
            f"Device {device} has no measurable on-path time (empty or zero-length span).",
            device,
        )

    categories = {
        "gpu_compute": gpu_compute_ms,
        "comm": comm_ms,
        "cpu": cpu_ms,
    }
    shares = {k: v / critical_path_ms for k, v in categories.items()}

    ranked = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
    top_cat, top_share = ranked[0]
    second_share = ranked[1][1]
    margin = round(top_share - second_share, 4)

    # Commit to a class only when a single category both wins by a clear margin
    # *and* owns an outright majority of the path, and there is enough on-path
    # time to trust the split. Anything short of that is reported as ``mixed``
    # with the reason recorded, rather than a confident-but-weak verdict.
    if critical_path_ms < _MIN_CRITICAL_PATH_MS:
        bound_class = _CLASS_MIXED
        mixed_reason = "insufficient"
    elif margin >= _MARGIN_THRESHOLD and top_share >= _MIN_DOMINANT_SHARE:
        bound_class = _CATEGORY_TO_CLASS[top_cat]
        mixed_reason = None
    elif margin < _MARGIN_THRESHOLD:
        bound_class = _CLASS_MIXED
        mixed_reason = "near_tie"
    else:
        bound_class = _CLASS_MIXED
        mixed_reason = "no_majority"
    # Confidence is defined as the margin between the top bound and the
    # runner-up: a near-tie yields low confidence and a "mixed" verdict.
    confidence = max(0.0, margin)

    top_compute, top_collectives = _top_on_path(adapter, kernel_table, device, trim)
    cpu_attribution = _cpu_attribution(conn, device, trim)

    return [
        {
            "bound_class": bound_class,
            "confidence": round(confidence, 3),
            "margin": margin,
            "critical_path_ms": round(critical_path_ms, 3),
            "device": device,
            "breakdown": {
                "gpu_compute_ms": round(gpu_compute_ms, 3),
                "comm_ms": round(comm_ms, 3),
                "cpu_ms": round(cpu_ms, 3),
                "gpu_compute_share": round(shares["gpu_compute"], 4),
                "comm_share": round(shares["comm"], 4),
                "cpu_share": round(shares["cpu"], 4),
            },
            "top_compute_kernels": top_compute,
            "top_collectives": top_collectives,
            "cpu_attribution": cpu_attribution,
            "note": _verdict_note(
                bound_class, top_cat, top_share, second_share, margin, mixed_reason
            ),
        }
    ]


def _verdict_note(
    bound_class: str,
    top_cat: str,
    top_share: float,
    second_share: float,
    margin: float,
    mixed_reason: str | None,
) -> str:
    if bound_class == _CLASS_MIXED:
        if mixed_reason == "insufficient":
            return (
                "Insufficient on-path time to classify — too few kernels or too "
                "short a window to trust a bound class."
            )
        if mixed_reason == "no_majority":
            return (
                f"No single category owns a majority of the critical path "
                f"(top {top_share * 100:.0f}%); reported as mixed rather than "
                "forcing a plurality winner."
            )
        return (
            "Near-tie between on-path categories "
            f"({top_share * 100:.0f}% vs {second_share * 100:.0f}%); no single "
            "bottleneck dominates the critical path."
        )
    note = (
        f"{_CATEGORY_TO_CLASS[top_cat]} — {top_share * 100:.0f}% of the critical "
        f"path is {top_cat.replace('_', '-')} on-path time "
        f"(runner-up {second_share * 100:.0f}%)."
    )
    if margin < _TENTATIVE_MARGIN:
        note += " Low margin over the runner-up — treat as tentative."
    return note


def _format(rows: list[dict]) -> str:
    if not rows:
        return "(No critical-path result)"
    r = rows[0]
    b = r.get("breakdown", {})
    lines = ["== Critical-Path Bound Analysis =="]
    lines.append(f"  Device:          {r.get('device')}")
    lines.append(f"  Bound class:     {r.get('bound_class')}")
    lines.append(f"  Confidence:      {r.get('confidence')} (margin over runner-up)")
    lines.append(f"  Critical path:   {r.get('critical_path_ms', 0):.1f}ms")
    lines.append("")
    lines.append("  On-path breakdown:")
    lines.append(
        f"    gpu-compute:   {b.get('gpu_compute_ms', 0):9.1f}ms "
        f"({b.get('gpu_compute_share', 0) * 100:5.1f}%)"
    )
    lines.append(
        f"    comm (exposed):{b.get('comm_ms', 0):9.1f}ms "
        f"({b.get('comm_share', 0) * 100:5.1f}%)"
    )
    lines.append(
        f"    cpu (gpu-idle):{b.get('cpu_ms', 0):9.1f}ms "
        f"({b.get('cpu_share', 0) * 100:5.1f}%)"
    )

    top_compute = r.get("top_compute_kernels") or []
    if top_compute:
        lines.append("")
        lines.append("  Top on-path compute kernels:")
        for e in top_compute:
            lines.append(
                f"    - {e['name'][:48]:<48} {e['total_ms']:8.2f}ms x{e['instances']}"
            )

    top_collectives = r.get("top_collectives") or []
    if top_collectives:
        lines.append("")
        lines.append("  Top on-path collectives:")
        for e in top_collectives:
            lines.append(
                f"    - {e['name'][:48]:<48} {e['total_ms']:8.2f}ms x{e['instances']}"
            )

    attr = r.get("cpu_attribution") or {}
    if attr:
        parts = []
        if "sync_wall_ms" in attr:
            parts.append(f"host sync {attr['sync_wall_ms']:.1f}ms")
        if "dispatch_starvation_events" in attr:
            parts.append(f"{attr['dispatch_starvation_events']} dispatch-starvation events")
        if parts:
            lines.append("")
            lines.append("  CPU attribution: " + ", ".join(parts))

    if r.get("note"):
        lines.append("")
        lines.append(f"  {r['note']}")
    return "\n".join(lines)


_EXPLANATION = (
    "The critical path is estimated along the busiest GPU device timeline. "
    "Time is attributed to gpu-compute when a compute kernel runs (overlapping "
    "communication is hidden and counts as compute), to comm when only NCCL "
    "runs exposed, and to cpu when the GPU is idle waiting on the host. The "
    "dominant on-path bucket is the bound class. This is a proxy: the trace "
    "carries no dependency edges, so the true longest path is not computed."
)
_SUGGESTED_ACTIONS = [
    "For gpu-compute-bound: profile the top on-path kernels for occupancy / MFU headroom",
    "For cpu-bound: check dataloader throughput, host syncs, and kernel-launch overhead",
    "For comm-bound: improve compute/communication overlap or reduce exposed collectives",
    "Re-run on a single iteration (via --trim) to confirm the class is stable per step",
]
_FALSE_POSITIVE_NOTES = [
    "The trace carries no dependency edges; the busiest-device timeline is a proxy "
    "for the true longest path, not the path itself",
    "Selecting the busiest device by kernel time can under-rank a device that is "
    "bottlenecked by being idle or exposed (a straggler); in distributed runs the "
    "true critical device may differ",
    "Very short windows can misattribute time if they clip kernels or collectives",
    "GPU-idle attributed to cpu may include unavoidable launch overhead on tiny-kernel workloads",
    "Multi-GPU profiles are classified per device; the busiest device is used by default",
]


def _to_findings(rows: list[dict], *, context: dict | None = None) -> list:
    """Emit a structured Finding for a confident bound classification.

    Skipped for ``mixed`` / ``n/a`` results (no confident bottleneck to
    annotate). The Finding.category is a valid ``FindingCategory`` member.
    """
    from nsys_ai.annotation import EvidenceRow, Finding, TraceSelection

    if not rows:
        return []
    r = rows[0]
    bound_class = r.get("bound_class")
    if bound_class in (_CLASS_MIXED, _CLASS_NA):
        return []

    # Reverse-map the public class back to the internal category for the
    # FindingCategory lookup.
    top_cat = next((c for c, label in _CATEGORY_TO_CLASS.items() if label == bound_class), None)
    if top_cat is None:
        return []

    profile_id = (context or {}).get("profile_id", "unknown")
    device = r.get("device", 0)
    b = r.get("breakdown", {})

    # Span the finding across the union of the top on-path entries so it lands
    # on a real region of the trace.
    all_entries = (r.get("top_compute_kernels") or []) + (r.get("top_collectives") or [])
    starts = [e["start_ns"] for e in all_entries if e.get("start_ns") is not None]
    ends = [e["end_ns"] for e in all_entries if e.get("end_ns") is not None]
    start_ns = min(starts) if starts else 0
    end_ns = max(ends) if ends else 0

    finding_id = f"critical_path_gpu{device}_{bound_class.replace('-', '_')}"
    selection = TraceSelection(
        id=f"sel_{finding_id}",
        profile_id=profile_id,
        source="skill:critical_path",
        start_ns=start_ns,
        end_ns=end_ns,
        gpu_ids=[device],
        label=f"{bound_class} critical path",
    )
    ev_values = {
        "gpu_compute_ms": b.get("gpu_compute_ms", 0.0),
        "comm_ms": b.get("comm_ms", 0.0),
        "cpu_ms": b.get("cpu_ms", 0.0),
        "gpu_compute_share": b.get("gpu_compute_share", 0.0),
        "comm_share": b.get("comm_share", 0.0),
        "cpu_share": b.get("cpu_share", 0.0),
        "critical_path_ms": r.get("critical_path_ms", 0.0),
    }
    ev_units = {
        "gpu_compute_ms": "ms",
        "comm_ms": "ms",
        "cpu_ms": "ms",
        "gpu_compute_share": "fraction",
        "comm_share": "fraction",
        "cpu_share": "fraction",
        "critical_path_ms": "ms",
    }
    evidence_row = EvidenceRow(
        id=f"ev_{finding_id}",
        source_skill="critical_path",
        values=ev_values,
        units=ev_units,
        selection_id=selection.id,
        provenance={"row_kind": "bound_class", "device": device},
    )

    # No headroom, by design. This skill's contribution is the *verdict* —
    # which resource bounds the run — not a new pool of recoverable time: the
    # cpu bucket is the same GPU-idle that gpu_idle_gaps already claims, and the
    # comm bucket the same exposed NCCL that overlap_breakdown claims. Those
    # skills keep the claim because they can also localize it to specific gaps
    # and streams. The bucket sizes stay on the evidence row below, so the
    # measurement is still reported — only the double count is dropped.
    return [
        Finding(
            type="region",
            label=f"Critical path is {bound_class}",
            start_ns=start_ns,
            end_ns=end_ns,
            gpu_id=device,
            severity="info",
            note=r.get("note", ""),
            id=finding_id,
            category=_CATEGORY_TO_FINDING[top_cat],
            confidence=float(r.get("confidence", 0.0)),
            evidence=[evidence_row],
            selection=selection,
            explanation=_EXPLANATION,
            suggested_actions=list(_SUGGESTED_ACTIONS),
            false_positive_notes=list(_FALSE_POSITIVE_NOTES),
            provenance={"skill": "critical_path", "row_kind": "bound_class"},
        )
    ]


SKILL = Skill(
    name="critical_path",
    title="Critical-Path Bound Analysis",
    description=(
        "Classifies a run (or a trimmed iteration) as cpu-bound, "
        "gpu-compute-bound, comm-bound, or mixed by estimating the critical "
        "path along the busiest GPU device timeline. Reports the critical-path "
        "time, the on-path breakdown by category (gpu-compute / comm / cpu), "
        "and the top on-path kernels and collectives with time selections. "
        "Overlapping communication is hidden (counts as compute); only exposed "
        "NCCL is comm; GPU-idle is the host-wait (cpu) bucket. Confidence is "
        "the margin between the top bound and the runner-up, and a near-tie is "
        "reported as mixed rather than forcing a confident-but-wrong winner."
    ),
    category="system",
    execute_fn=_execute,
    format_fn=_format,
    to_findings_fn=_to_findings,
    params=[
        SkillParam(
            "device",
            "GPU device id to analyze. Omit to auto-select the busiest device.",
            "int",
            False,
            None,
        ),
    ],
    tags=[
        "critical-path", "bound", "cpu-bound", "gpu-bound", "comm-bound",
        "bottleneck", "classification", "roofline", "hta",
    ],
)
