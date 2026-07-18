"""Compute/Communication overlap breakdown.

Uses overlap_analysis() from overlap.py to quantify how much GPU compute
overlaps with NCCL communication.  This is a Python-level skill (execute_fn)
rather than a SQL skill because it needs interval merging and intersection
logic that can't be expressed in a single SQL query.
"""

import logging

from nsys_ai.connection import DB_ERRORS

from ..base import Skill, SkillParam

_log = logging.getLogger(__name__)

# Case-insensitive NCCL kernel-name match against StringIds.value, reused
# across three queries below. Keep the OR-of-LIKE form (rather than
# `LOWER(...) LIKE`) so SQLite/DuckDB can still use a string-prefix index
# if one exists on StringIds.value.
_NCCL_LIKE = "(s.value LIKE '%nccl%' OR s.value LIKE '%NCCL%')"

# A same-stream collision is only worth flagging when a meaningful fraction of
# BOTH compute and NCCL actually share a stream. A few stray cross-stream kernels
# (e.g. one eager kernel on the NCCL stream) score ~0% and must not trigger the
# diagnosis.
_SAME_STREAM_MIN_PCT = 5.0


def _execute(conn, **kwargs):
    from ...overlap import overlap_analysis
    from ...profile import Profile

    prof = Profile._from_conn(conn)
    device = int(kwargs.get("device", 0))
    # Support --trim passthrough from agent analyze
    trim = None
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    if trim_start is not None and trim_end is not None:
        trim = (int(trim_start), int(trim_end))
    result = overlap_analysis(prof, device, trim=trim)
    if "error" in result:
        return [result]

    # Same-stream diagnosis for evidence builder overlap check
    same_stream = []
    try:
        kernel_tbl = prof.schema.kernel_table
        if kernel_tbl:
            trim_clause = ""
            params = [device]
            if trim is not None:
                trim_clause = " AND k.[end] >= ? AND k.start <= ? "
                params.extend([trim[0], trim[1]])
            same_stream = prof._duckdb_query(
                f"""
                SELECT k.streamId,
                    SUM(CASE WHEN {_NCCL_LIKE} THEN 1 ELSE 0 END) AS nccl_count,
                    SUM(CASE WHEN NOT {_NCCL_LIKE} THEN 1 ELSE 0 END) AS compute_count
                FROM {kernel_tbl} k
                JOIN StringIds s ON k.shortName = s.id
                WHERE k.deviceId = ? {trim_clause}
                GROUP BY k.streamId
                HAVING nccl_count > 0 AND compute_count > 0
                """,
                params,
            )
            if same_stream:
                # Quantify the shared fraction BEFORE flagging. A near-clean
                # multi-stream layout with a few stray cross-stream kernels
                # reads identically to a 100% collision by stream id alone, so
                # gate the diagnosis on a meaningful share of both kinds rather
                # than firing on a single stray cross-stream kernel.
                totals = prof._duckdb_query(
                    f"""
                    SELECT
                        SUM(CASE WHEN {_NCCL_LIKE} THEN 1 ELSE 0 END) AS nccl_total,
                        SUM(CASE WHEN NOT {_NCCL_LIKE} THEN 1 ELSE 0 END) AS compute_total
                    FROM {kernel_tbl} k
                    JOIN StringIds s ON k.shortName = s.id
                    WHERE k.deviceId = ? {trim_clause}
                    """,
                    params,
                )
                compute_pct = nccl_pct = 0.0
                if totals:
                    nccl_total = totals[0]["nccl_total"] or 0
                    compute_total = totals[0]["compute_total"] or 0
                    nccl_same = sum((r["nccl_count"] or 0) for r in same_stream)
                    compute_same = sum((r["compute_count"] or 0) for r in same_stream)
                    if compute_total > 0:
                        compute_pct = round(100 * compute_same / compute_total, 1)
                    if nccl_total > 0:
                        nccl_pct = round(100 * nccl_same / nccl_total, 1)
                if (
                    compute_pct >= _SAME_STREAM_MIN_PCT
                    and nccl_pct >= _SAME_STREAM_MIN_PCT
                ):
                    result["same_stream_diagnosis"] = [
                        str(r["streamId"]) for r in same_stream
                    ]
                    result["same_stream_compute_pct"] = compute_pct
                    result["same_stream_nccl_pct"] = nccl_pct
    except DB_ERRORS:
        _log.debug("Failed to enrich stream compute/nccl overlap", exc_info=True)

    # Surface other GPUs that exist in the profile. Otherwise a multi-rank
    # single-node profile is invisible to the agent (which only sees
    # device=0 by default). Intentionally profile-wide — NOT window-scoped —
    # because "which devices exist" is a topology property, not a per-window
    # one. A trim window that excludes a rank's activity should not hide
    # that rank from the device list.
    try:
        kernel_tbl = prof.schema.kernel_table
        if kernel_tbl:
            devs = prof._duckdb_query(
                f"SELECT DISTINCT deviceId FROM {kernel_tbl} ORDER BY deviceId"
            )
            if devs:
                result["present_devices"] = [
                    int(r["deviceId"]) for r in devs if r["deviceId"] is not None
                ]
    except DB_ERRORS:
        _log.debug("Failed to enumerate present devices", exc_info=True)

    try:
        from ...skills.registry import get_skill

        sync_skill = get_skill("sync_cost_analysis")
        if sync_skill:
            sync_data = sync_skill.execute(conn, **kwargs)
            if sync_data and "error" not in sync_data[0]:
                result["sync_ms"] = sync_data[0].get("total_sync_wall_ms", 0)
    except Exception:
        _log.debug("Failed to enrich sync cost", exc_info=True)

    result["device_id"] = device
    return [result]


def _format(rows):
    if not rows:
        return "(No overlap data available)"
    r = rows[0]
    if "error" in r:
        return f"(Overlap analysis: {r['error']})"
    lines = [
        "── Compute/Communication Overlap ──",
        f"  Total span:    {r['total_ms']:.1f}ms",
        f"  Compute only:  {r['compute_only_ms']:.1f}ms",
        f"  NCCL only:     {r['nccl_only_ms']:.1f}ms",
        f"  Overlap:       {r['overlap_ms']:.1f}ms ({r['overlap_pct']}% of NCCL overlapped)",
        f"  Idle:          {r['idle_ms']:.1f}ms",
    ]
    if r.get("sync_ms"):
        lines.append(f"  ⚡ CPU Sync:     {r['sync_ms']:.1f}ms (global host-side blocking)")
    lines.append(f"  Kernels:       {r['compute_kernels']} compute + {r['nccl_kernels']} NCCL")

    # Same-stream serialization warning — surfaced in textual output so it's
    # visible without parsing JSON. Proportions added when available.
    streams = r.get("same_stream_diagnosis")
    if streams:
        compute_pct = r.get("same_stream_compute_pct")
        nccl_pct = r.get("same_stream_nccl_pct")
        suffix = ""
        if compute_pct is not None and nccl_pct is not None:
            suffix = f" — {compute_pct}% of compute + {nccl_pct}% of NCCL"
        lines.append(
            f"  ⚠️ Same-stream:  stream(s) [{', '.join(streams)}] carry both"
            f"{suffix} → overlap blocked"
        )

    # Multi-device hint — textual users miss the device list otherwise.
    devs = r.get("present_devices") or []
    if len(devs) > 1:
        analyzed = r.get("device_id", 0)
        others = [str(d) for d in devs if d != analyzed]
        lines.append(
            f"  Devices:       {analyzed} (analyzed); profile also has "
            f"[{', '.join(others)}] — re-run with -p device=N for each"
        )
    return "\n".join(lines)


_LOW_OVERLAP_EXPLANATION = (
    "NCCL communication is exposed on the timeline instead of being hidden "
    "under compute. Low overlap usually means communication is on the critical "
    "path for step time."
)
_COMM_DOMINATED_EXPLANATION = (
    "Communication time is larger than useful compute time in this window. "
    "This often points to exposed collectives, poor overlap, or an imbalanced "
    "parallelism strategy."
)
_SUGGESTED_ACTIONS = [
    "Check whether NCCL and compute run on separate streams with compatible dependencies",
    "Inspect whether collectives are launched early enough to overlap with backward compute",
    "Compare per-rank overlap to identify stragglers or imbalance",
    "Try narrowing to one iteration or NVTX phase to confirm where communication is exposed",
]
_FALSE_POSITIVE_NOTES = [
    "Short windows can understate overlap if they clip compute or communication intervals",
    "Profiles with very little NCCL time may not need optimization even when overlap is low",
]


def _overlap_confidence(overlap_pct: float, nccl_ms: float, total_ms: float) -> float:
    """Heuristic confidence for low communication-overlap findings."""
    if nccl_ms <= 0 or total_ms <= 0:
        return 0.4
    nccl_share = min(nccl_ms / total_ms, 1.0)
    overlap_signal = max(0.0, min((30.0 - overlap_pct) / 30.0, 1.0))
    return round(min(0.95, 0.55 + 0.25 * overlap_signal + 0.15 * nccl_share), 3)


def _ratio_confidence(ratio: float) -> float:
    """Heuristic confidence for communication-dominated findings."""
    if ratio < 0.25:
        return 0.95
    if ratio < 0.5:
        return 0.85
    return 0.7


def _common_evidence_values(r: dict, *, nccl_ms: float) -> tuple[dict, dict]:
    values = {
        "compute_only_ms": round(float(r.get("compute_only_ms", 0) or 0), 3),
        "nccl_only_ms": round(float(r.get("nccl_only_ms", 0) or 0), 3),
        "overlap_ms": round(float(r.get("overlap_ms", 0) or 0), 3),
        "nccl_total_ms": round(float(nccl_ms), 3),
        "idle_ms": round(float(r.get("idle_ms", 0) or 0), 3),
        "total_ms": round(float(r.get("total_ms", 0) or 0), 3),
        "overlap_pct": round(float(r.get("overlap_pct", 0) or 0), 3),
    }
    units = {
        "compute_only_ms": "ms",
        "nccl_only_ms": "ms",
        "overlap_ms": "ms",
        "nccl_total_ms": "ms",
        "idle_ms": "ms",
        "total_ms": "ms",
        "overlap_pct": "percent",
    }
    if r.get("sync_ms") is not None:
        values["sync_ms"] = round(float(r.get("sync_ms", 0) or 0), 3)
        units["sync_ms"] = "ms"
    if r.get("same_stream_diagnosis"):
        values["same_streams"] = list(r.get("same_stream_diagnosis") or [])
        if r.get("same_stream_compute_pct") is not None:
            values["same_stream_compute_pct"] = round(float(r["same_stream_compute_pct"]), 3)
            units["same_stream_compute_pct"] = "percent"
        if r.get("same_stream_nccl_pct") is not None:
            values["same_stream_nccl_pct"] = round(float(r["same_stream_nccl_pct"]), 3)
            units["same_stream_nccl_pct"] = "percent"
    return values, units


def _to_findings(rows: list[dict], *, context: dict | None = None) -> list:
    from nsys_ai.annotation import EvidenceRow, Finding, TraceSelection

    findings = []
    if not rows or "error" in rows[0]:
        return findings

    profile_id = (context or {}).get("profile_id", "unknown")
    r = rows[0]
    nccl_ms = r.get("nccl_only_ms", 0) + r.get("overlap_ms", 0)
    compute_ms = r.get("compute_only_ms", 0)
    overlap_pct = r.get("overlap_pct", 0)
    total_ms = r.get("total_ms", 1)
    start_ns = r.get("span_start_ns", 0)
    end_ns = r.get("span_end_ns", 0)
    device = r.get("device_id", 0)

    # Low overlap: NCCL not well hidden behind compute
    if nccl_ms > 0 and overlap_pct < 30:
        note = (
            f"Only {overlap_pct}% of NCCL time overlaps with compute. "
            f"NCCL-only: {r.get('nccl_only_ms', 0):.1f}ms out of "
            f"{total_ms:.1f}ms total span."
        )
        streams = r.get("same_stream_diagnosis")
        if streams:
            compute_pct = r.get("same_stream_compute_pct")
            nccl_pct = r.get("same_stream_nccl_pct")
            note += (
                f" Streams [{', '.join(streams)}] run both NCCL and "
                f"compute — overlap is impossible on same stream."
            )
            if compute_pct is not None and nccl_pct is not None:
                note += (
                    f" ({compute_pct}% of compute and {nccl_pct}% of NCCL "
                    f"share these streams.)"
                )

        finding_id = f"overlap_low_gpu{device}_{start_ns}"
        selection = TraceSelection(
            id=f"sel_{finding_id}",
            profile_id=profile_id,
            source="skill:overlap_breakdown",
            start_ns=start_ns,
            end_ns=end_ns,
            gpu_ids=[device],
            label=f"Low Compute/NCCL Overlap ({overlap_pct}%)",
        )
        ev_values, ev_units = _common_evidence_values(r, nccl_ms=nccl_ms)
        evidence_row = EvidenceRow(
            id=f"ev_{finding_id}",
            source_skill="overlap_breakdown",
            values=ev_values,
            units=ev_units,
            selection_id=selection.id,
            provenance={"row_kind": "low_overlap", "device": device},
        )

        findings.append(
            Finding(
                type="region",
                label=f"Low Compute/NCCL Overlap ({overlap_pct}%)",
                start_ns=start_ns,
                end_ns=end_ns,
                gpu_id=device,
                severity="warning",
                note=note,
                id=finding_id,
                category="communication",
                confidence=_overlap_confidence(float(overlap_pct), float(nccl_ms), float(total_ms)),
                # Exposed (non-overlapped) NCCL is the time recoverable if the
                # collective were hidden under compute.
                headroom_ms=r.get("nccl_only_ms", 0.0),
                evidence=[evidence_row],
                selection=selection,
                explanation=_LOW_OVERLAP_EXPLANATION,
                suggested_actions=list(_SUGGESTED_ACTIONS),
                false_positive_notes=list(_FALSE_POSITIVE_NOTES),
                provenance={"skill": "overlap_breakdown", "row_kind": "low_overlap"},
            )
        )

    # Communication dominated: NCCL > compute
    if nccl_ms > 0 and compute_ms > 0:
        ratio = compute_ms / nccl_ms
        if ratio < 0.5:
            finding_id = f"overlap_comm_dominated_gpu{device}_{start_ns}"
            selection = TraceSelection(
                id=f"sel_{finding_id}",
                profile_id=profile_id,
                source="skill:overlap_breakdown",
                start_ns=start_ns,
                end_ns=end_ns,
                gpu_ids=[device],
                label=f"Communication Dominated (ratio={ratio:.2f})",
            )
            ev_values, ev_units = _common_evidence_values(r, nccl_ms=nccl_ms)
            ev_values["compute_nccl_ratio"] = round(float(ratio), 3)
            ev_units["compute_nccl_ratio"] = "ratio"
            evidence_row = EvidenceRow(
                id=f"ev_{finding_id}",
                source_skill="overlap_breakdown",
                values=ev_values,
                units=ev_units,
                selection_id=selection.id,
                provenance={"row_kind": "communication_dominated", "device": device},
            )

            findings.append(
                Finding(
                    type="region",
                    label=f"Communication Dominated (ratio={ratio:.2f})",
                    start_ns=start_ns,
                    end_ns=end_ns,
                    gpu_id=device,
                    severity="critical",
                    note=(
                        f"Compute/Communication ratio is {ratio:.2f} "
                        f"(healthy > 2.0). Compute: {compute_ms:.1f}ms, "
                        f"NCCL: {nccl_ms:.1f}ms."
                    ),
                    id=finding_id,
                    category="communication",
                    confidence=_ratio_confidence(float(ratio)),
                    # Exposed (non-overlapped) NCCL is the recoverable time.
                    headroom_ms=r.get("nccl_only_ms", 0.0),
                    evidence=[evidence_row],
                    selection=selection,
                    explanation=_COMM_DOMINATED_EXPLANATION,
                    suggested_actions=list(_SUGGESTED_ACTIONS),
                    false_positive_notes=list(_FALSE_POSITIVE_NOTES),
                    provenance={
                        "skill": "overlap_breakdown",
                        "row_kind": "communication_dominated",
                    },
                )
            )

    return findings


SKILL = Skill(
    name="overlap_breakdown",
    title="Compute/Communication Overlap Breakdown",
    description=(
        "Quantifies how much GPU compute overlaps with NCCL communication. "
        "Shows compute-only, NCCL-only, overlap, and idle time. "
        "overlap_pct > 60% means NCCL is well-hidden behind compute. "
        "Also detects same-stream serialization (compute and NCCL sharing a "
        "single stream, which blocks overlap) and surfaces multi-GPU presence "
        "via present_devices for profiles with more than one rank."
    ),
    category="communication",
    execute_fn=_execute,
    format_fn=_format,
    to_findings_fn=_to_findings,
    params=[SkillParam("device", "GPU device ID", "int", False, 0)],
    tags=[
        "overlap", "nccl", "compute", "communication", "distributed",
        "multi-gpu", "same-stream", "stream-serialization",
    ],
)
