"""NCCL collective operation breakdown — per-stream.

Delegates to the unified ``nsys_ai.overlap.nccl_breakdown()`` engine so that
the Agent skill, CLI, and TUI chat tool all produce identical results.
"""

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    from ...overlap import nccl_breakdown
    from ...profile import Profile

    prof = Profile._from_conn(conn)
    device = int(kwargs.get("device", 0))

    trim = None
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    if trim_start is not None and trim_end is not None:
        trim = (int(trim_start), int(trim_end))

    rows = nccl_breakdown(prof, device, trim)
    span_start, span_end = trim if trim else prof.meta.time_range
    for r in rows:
        r["device_id"] = device
        r["span_start_ns"] = span_start
        r["span_end_ns"] = span_end
    return rows


def _format(rows):
    from ...overlap import format_nccl

    base = format_nccl(rows)

    # Add a brief diagnostic hint when no NCCL collectives are present.
    # This mirrors the previous, more actionable behavior while still
    # delegating core formatting to the shared overlap engine.
    if not rows:
        hint = (
            "\n\nHint: This usually means either the profile was captured on a "
            "single GPU, or that NCCL communication was not recorded for this run."
        )
        return f"{base}{hint}"

    return base



_SENDRECV_DOMINATED_EXPLANATION = (
    "SendRecv accounts for the majority of NCCL time, indicating pipeline "
    "parallelism (PP) is the dominant communication pattern. This is expected "
    "for PP workloads but means inter-stage latency is on the critical path."
)
_ALLGATHER_DOMINATED_EXPLANATION = (
    "AllGather accounts for a significant portion of NCCL time, indicating "
    "tensor parallelism (TP) or sequence parallelism gather operations dominate "
    "communication cost."
)
_ALLREDUCE_DOMINATED_EXPLANATION = (
    "AllReduce accounts for the majority of NCCL time, indicating data "
    "parallelism (DP) gradient synchronization dominates communication cost."
)
_HIGH_VARIABILITY_EXPLANATION = (
    "NCCL kernel durations vary widely (max >> avg), suggesting message-size "
    "bimodality or occasional stragglers. Single-rank proxy for Root Cause #8 "
    "(Compute-Communication Imbalance) — strict detection requires multi-rank "
    "comparison, but high variance on one rank often indicates this rank is "
    "intermittently waiting for other ranks at NCCL barriers."
)
_NCCL_SERIALIZATION_EXPLANATION = (
    "Per Root Cause #3 (NCCL Serialization): NCCL time consuming a large share "
    "of captured time suggests collective operations are not effectively overlapping "
    "with compute. Note: this uses captured time (full profile span or trim window); "
    "for strict per-iteration analysis use iteration_timing. Common causes: gradient "
    "bucketing misconfiguration, NCCL streams blocked by compute on same stream, or "
    "suboptimal network topology."
)
_NCCL_SERIALIZATION_ACTIONS = [
    "Tune DDP bucket sizes (e.g., bucket_cap_mb=25)",
    "Ensure NCCL runs on a separate stream from compute",
    "Use gradient compression or FSDP for large models",
    "Check NVLink/InfiniBand topology matches the collective algorithm",
]
_SUGGESTED_ACTIONS = [
    "Check whether NCCL streams are separate from compute streams to enable overlap",
    "Profile per-rank to identify stragglers causing variability",
    "Consider fusing small collectives to reduce launch overhead",
    "Verify the dominant collective matches your intended parallelism strategy",
]
_FALSE_POSITIVE_NOTES = [
    "Bimodal message sizes are expected for models with variable sequence lengths",
    "SendRecv dominance is normal for pipeline-parallel workloads (not a bug)",
    "AllReduce dominance is normal for data-parallel training (not a bug)",
    "AllGather dominance is normal for FSDP/TP workloads (not a bug)",
    "Short profiles may not capture the steady-state collective distribution",
    "Single-host inference will have no NCCL — empty result is a valid finding",
]

# Thresholds (calibrated to common workloads; some sourced from book.md root causes)
_DOMINATED_THRESHOLD_PCT = 70.0          # collective dominates NCCL when its pct > this
_VARIABILITY_RATIO_THRESHOLD = 2.0       # max/avg > this → high variability
_VARIABILITY_MIN_PCT = 1.0               # gate to suppress noise from negligible collectives
_NCCL_SERIALIZATION_THRESHOLD_PCT = 20.0  # per book.md Root Cause #3

def _nccl_confidence(pct: float, total_nccl_ms: float = 0.0) -> float:
    if total_nccl_ms > 0 and total_nccl_ms < 10.0:
        return 0.5
    if pct >= 95:
        return 0.95
    if pct >= 80:
        return 0.85
    return 0.7


def _variability_confidence(ratio: float, n_kernels: int = 0) -> float:
    if n_kernels > 0 and n_kernels < 10:
        return 0.5
    if ratio >= 5.0:
        return 0.95
    if ratio >= 3.0:
        return 0.85
    if ratio >= 2.0:
        return 0.70
    return 0.60


def _serialization_confidence(nccl_capture_pct: float, captured_ms: float = 0.0) -> float:
    if captured_ms > 0 and captured_ms < 100.0:
        return 0.5
    if nccl_capture_pct >= 40:
        return 0.95
    if nccl_capture_pct >= 30:
        return 0.85
    return 0.70

def _to_findings(rows: list[dict], *, context: dict | None = None)-> list:
    from nsys_ai.annotation import EvidenceRow, Finding, TraceSelection

    findings = []
    if not rows:
        return findings

    profile_id = (context or {}).get("profile_id", "unknown")
    device = rows[0].get("device_id", (context or {}).get("device", 0))
    start_ns = rows[0].get("span_start_ns", 0)
    end_ns = rows[0].get("span_end_ns", 0)

    pct_by_type: dict[str, float] = {}
    total_nccl_ms = sum(r.get("total_ms", 0.0) for r in rows)
    for r in rows:
        t = r["type"]
        pct_by_type[t] = pct_by_type.get(t, 0.0) + r["pct"]

    # Finding 1: sendrecv dominated
    sendrecv_pct = pct_by_type.get("sendrecv", 0.0)
    if sendrecv_pct > _DOMINATED_THRESHOLD_PCT:
        finding_id = "nccl_sendrecv_dominated"
        selection = TraceSelection(
            id=f"sel_{finding_id}",
            profile_id=profile_id,
            source="skill:nccl_breakdown",
            start_ns=start_ns,
            end_ns=end_ns,
            gpu_ids=[device],
            label=f"SendRecv Dominated ({sendrecv_pct:.1f}%)",
        )
        evidence_row = EvidenceRow(
            id=f"ev_{finding_id}",
            source_skill="nccl_breakdown",
            values={"sendrecv_pct": round(sendrecv_pct, 1)},
            units={"sendrecv_pct": "percent"},
            selection_id=selection.id,
            provenance={"row_kind": "sendrecv_dominated"},
        )
        findings.append(Finding(
            type="region",
            label=f"SendRecv Dominated ({sendrecv_pct:.1f}% of NCCL)",
            start_ns=start_ns,
            end_ns=end_ns,
            gpu_id=device,
            severity="info",
            note=f"SendRecv accounts for {sendrecv_pct:.1f}% of NCCL time — pipeline parallelism is the dominant communication pattern.",
            id=finding_id,
            category="communication",
            confidence=_nccl_confidence(sendrecv_pct, total_nccl_ms),
            evidence=[evidence_row],
            selection=selection,
            explanation=_SENDRECV_DOMINATED_EXPLANATION,
            suggested_actions=list(_SUGGESTED_ACTIONS),
            false_positive_notes=list(_FALSE_POSITIVE_NOTES),
            provenance={"skill": "nccl_breakdown", "row_kind": "sendrecv_dominated"},
        ))

    # Finding 2: allgather dominated
    allgather_pct = pct_by_type.get("allgather", 0.0)
    if allgather_pct > _DOMINATED_THRESHOLD_PCT:
        finding_id = "nccl_allgather_dominated"
        selection = TraceSelection(
            id=f"sel_{finding_id}",
            profile_id=profile_id,
            source="skill:nccl_breakdown",
            start_ns=start_ns,
            end_ns=end_ns,
            gpu_ids=[device],
            label=f"AllGather Dominated ({allgather_pct:.1f}%)",
        )
        evidence_row = EvidenceRow(
            id=f"ev_{finding_id}",
            source_skill="nccl_breakdown",
            values={"allgather_pct": round(allgather_pct, 1)},
            units={"allgather_pct": "percent"},
            selection_id=selection.id,
            provenance={"row_kind": "allgather_dominated"},
        )
        findings.append(Finding(
            type="region",
            label=f"AllGather Dominated ({allgather_pct:.1f}% of NCCL)",
            start_ns=start_ns,
            end_ns=end_ns,
            gpu_id=device,
            severity="info",
            note=f"AllGather accounts for {allgather_pct:.1f}% of NCCL time — TP/sequence parallelism gather dominates.",
            id=finding_id,
            category="communication",
            confidence=_nccl_confidence(allgather_pct, total_nccl_ms),
            evidence=[evidence_row],
            selection=selection,
            explanation=_ALLGATHER_DOMINATED_EXPLANATION,
            suggested_actions=list(_SUGGESTED_ACTIONS),
            false_positive_notes=list(_FALSE_POSITIVE_NOTES),
            provenance={"skill": "nccl_breakdown", "row_kind": "allgather_dominated"},
        ))

    # Finding 3: allreduce dominated
    allreduce_pct = pct_by_type.get("allreduce", 0.0)
    if allreduce_pct > _DOMINATED_THRESHOLD_PCT:
        finding_id = "nccl_allreduce_dominated"
        selection = TraceSelection(
            id=f"sel_{finding_id}",
            profile_id=profile_id,
            source="skill:nccl_breakdown",
            start_ns=start_ns,
            end_ns=end_ns,
            gpu_ids=[device],
            label=f"AllReduce Dominated ({allreduce_pct:.1f}%)",
        )
        evidence_row = EvidenceRow(
            id=f"ev_{finding_id}",
            source_skill="nccl_breakdown",
            values={"allreduce_pct": round(allreduce_pct, 1)},
            units={"allreduce_pct": "percent"},
            selection_id=selection.id,
            provenance={"row_kind": "allreduce_dominated"},
        )
        findings.append(Finding(
            type="region",
            label=f"AllReduce Dominated ({allreduce_pct:.1f}% of NCCL)",
            start_ns=start_ns,
            end_ns=end_ns,
            gpu_id=device,
            severity="info",
            note=f"AllReduce accounts for {allreduce_pct:.1f}% of NCCL time — data parallelism gradient sync dominates.",
            id=finding_id,
            category="communication",
            confidence=_nccl_confidence(allreduce_pct, total_nccl_ms),
            evidence=[evidence_row],
            selection=selection,
            explanation=_ALLREDUCE_DOMINATED_EXPLANATION,
            suggested_actions=list(_SUGGESTED_ACTIONS),
            false_positive_notes=list(_FALSE_POSITIVE_NOTES),
            provenance={"skill": "nccl_breakdown", "row_kind": "allreduce_dominated"},
        ))

    # Finding 4: high variability — single-rank proxy for book.md Root Cause #8 (Compute-Communication Imbalance)
    for r in rows:
        if r["avg_ms"] > 0 and r["max_ms"] / r["avg_ms"] > _VARIABILITY_RATIO_THRESHOLD and r["pct"] >= _VARIABILITY_MIN_PCT:
            ratio = round(r["max_ms"] / r["avg_ms"], 1)
            finding_id = f"nccl_high_variability_{r['type']}_stream{r['stream_id']}"
            selection = TraceSelection(
                id=f"sel_{finding_id}",
                profile_id=profile_id,
                source="skill:nccl_breakdown",
                start_ns=start_ns,
                end_ns=end_ns,
                gpu_ids=[device],
                label=f"High NCCL Variability ({r['type']}, {ratio}×)",
            )
            evidence_row = EvidenceRow(
                id=f"ev_{finding_id}",
                source_skill="nccl_breakdown",
                values={
                    "type": r["type"],
                    "avg_ms": round(r["avg_ms"], 3),
                    "max_ms": round(r["max_ms"], 3),
                    "max_avg_ratio": ratio,
                },
                units={"avg_ms": "ms", "max_ms": "ms", "max_avg_ratio": "ratio"},
                selection_id=selection.id,
                provenance={"row_kind": "high_variability", "collective_type": r["type"], "root_cause": "#8_proxy"},
            )
            findings.append(Finding(
                type="region",
                label=f"High NCCL Variability ({r['type']}, max/avg={ratio}×)",
                start_ns=start_ns,
                end_ns=end_ns,
                gpu_id=device,
                severity="warning",
                note=f"{r['type']} max={r['max_ms']:.1f}ms vs avg={r['avg_ms']:.1f}ms ({ratio}× ratio) — possible message-size bimodality or stragglers.",
                id=finding_id,
                category="communication",
                confidence=_variability_confidence(ratio, r.get("count", 0)),
                evidence=[evidence_row],
                selection=selection,
                explanation=_HIGH_VARIABILITY_EXPLANATION,
                suggested_actions=list(_SUGGESTED_ACTIONS),
                false_positive_notes=list(_FALSE_POSITIVE_NOTES),
                provenance={"skill": "nccl_breakdown", "row_kind": "high_variability", "root_cause": "#8_proxy"},
            ))

    # Finding 5: NCCL Serialization — book.md Root Cause #3
    # Signal: NCCL time > 20% of captured time.
    # Note: total_nccl_ms (computed at the top of this function) sums per-stream
    # times and may overcount wall-clock when collectives overlap across streams;
    # interval-union would be more accurate but is out of scope here (see
    # overlap_breakdown for the precise compute/comm overlap analysis).
    # Clip to 100% so the displayed value stays sane even with overlap.
    captured_ms = (end_ns - start_ns) / 1e6 if end_ns > start_ns else 0
    nccl_capture_pct = min(100.0, (total_nccl_ms / captured_ms * 100) if captured_ms > 0 else 0)

    if nccl_capture_pct > _NCCL_SERIALIZATION_THRESHOLD_PCT:
        finding_id = "nccl_serialization"
        selection = TraceSelection(
            id=f"sel_{finding_id}",
            profile_id=profile_id,
            source="skill:nccl_breakdown",
            start_ns=start_ns, end_ns=end_ns,
            gpu_ids=[device],
            label=f"NCCL Serialization ({nccl_capture_pct:.1f}% of captured time)",
        )
        evidence_row = EvidenceRow(
            id=f"ev_{finding_id}",
            source_skill="nccl_breakdown",
            values={
                "total_nccl_ms": round(total_nccl_ms, 2),
                "captured_ms": round(captured_ms, 2),
                "nccl_capture_pct": round(nccl_capture_pct, 1),
            },
            units={"total_nccl_ms": "ms", "captured_ms": "ms", "nccl_capture_pct": "percent"},
            selection_id=selection.id,
            provenance={"row_kind": "nccl_serialization", "root_cause": "#3"},
        )
        findings.append(Finding(
            type="region",
            label=f"NCCL Serialization ({nccl_capture_pct:.1f}% of captured time)",
            start_ns=start_ns, end_ns=end_ns,
            gpu_id=device,
            severity="warning",
            note=f"NCCL accounts for {nccl_capture_pct:.1f}% of captured time ({total_nccl_ms:.0f}ms / {captured_ms:.0f}ms) — may indicate poor compute/comm overlap.",
            id=finding_id,
            category="communication",
            confidence=_serialization_confidence(nccl_capture_pct, captured_ms),
            evidence=[evidence_row],
            selection=selection,
            explanation=_NCCL_SERIALIZATION_EXPLANATION,
            suggested_actions=list(_NCCL_SERIALIZATION_ACTIONS),
            false_positive_notes=list(_FALSE_POSITIVE_NOTES),
            provenance={"skill": "nccl_breakdown", "row_kind": "nccl_serialization", "root_cause": "#3"},
        ))

    return findings

SKILL = Skill(
    name="nccl_breakdown",
    title="NCCL Collective Breakdown",
    description=(
        "Summarizes NCCL collective operations (AllReduce, AllGather, ReduceScatter, etc.) "
        "per CUDA stream, showing count, total time, and variability. "
        "Per-stream grouping helps distinguish TP vs PP vs DP communication channels, "
        "since each parallelism dimension typically uses a dedicated stream."
    ),
    category="communication",
    execute_fn=_execute,
    format_fn=_format,
    to_findings_fn=_to_findings,
    params=[
        SkillParam("device", "GPU device ID", "int", False, 0),
    ],
    tags=["nccl", "collective", "allreduce", "communication", "distributed", "multi-gpu", "stream"],
)
