"""Kernel Instance Details — individual kernel instances with ns timestamps.

Returns the top N longest kernel execution instances with exact
nanosecond start/end timestamps. Essential for building findings.json
evidence overlays on the timeline viewer.
"""

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    from ...profile import Profile

    prof = Profile._from_conn(conn)
    if not prof.schema.kernel_table:
        return [{"error": "No kernel table found in profile"}]

    device = int(kwargs.get("device", 0))
    name_filter = kwargs.get("name", "")
    limit = int(kwargs.get("limit", 10))

    # Build WHERE clause safely with parameterized query
    params = [device]
    where_extra = ""
    if name_filter:
        where_extra = "AND d.value LIKE ?"
        params.append(f"%{name_filter}%")

    # Handle trim
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    if trim_start is not None and trim_end is not None:
        where_extra += " AND k.[end] >= ? AND k.start <= ?"
        params.extend([int(trim_start), int(trim_end)])

    params.append(limit)

    sql = f"""
        SELECT d.value AS kernel_name,
               s.value AS short_name,
               k.start AS start_ns,
               k.[end] AS end_ns,
               ROUND((k.[end] - k.start) / 1e6, 3) AS duration_ms,
               k.streamId AS stream_id,
               k.deviceId AS device_id
        FROM {prof.schema.kernel_table} k
        JOIN StringIds s ON k.shortName = s.id
        JOIN StringIds d ON k.demangledName = d.id
        WHERE k.deviceId = ?
          {where_extra}
        ORDER BY (k.[end] - k.start) DESC
        LIMIT ?
    """
    return prof._duckdb_query(sql, params)


def _format(rows):
    if not rows:
        return "(No kernel instances found)"
    if "error" in rows[0]:
        return rows[0]["error"]

    lines = ["── Kernel Instances ──"]
    for r in rows:
        name = r.get("kernel_name", "?")
        if len(name) > 60:
            name = name[:57] + "..."
        lines.append(
            f"  {name:<62s}  {r['duration_ms']:>8.3f}ms  "
            f"stream={r['stream_id']}  "
            f"[{r['start_ns']}..{r['end_ns']}]"
        )
    return "\n".join(lines)


_NCCL_EXPLANATION = (
    "This is a long NCCL communication kernel. Long communication kernels "
    "matter most when they are exposed on the critical path instead of "
    "overlapping with useful compute."
)
_HOTSPOT_EXPLANATION = (
    "This is one of the longest individual compute kernel instances in the "
    "selected window. Kernel hotspots are useful anchors for deeper NVTX, "
    "launch configuration, or instruction-level analysis."
)
_NCCL_ACTIONS = [
    "Check whether this NCCL kernel overlaps with compute in the timeline",
    "Compare the same collective across ranks to look for stragglers",
    "Inspect surrounding NVTX ranges to identify the distributed phase",
]
_HOTSPOT_ACTIONS = [
    "Map the kernel back to its enclosing NVTX range or model layer",
    "Compare repeated instances to see whether this is a persistent hotspot",
    "Use CUTracer or Nsight Compute for instruction-level analysis if needed",
]
_FALSE_POSITIVE_NOTES = [
    "A long individual kernel is not necessarily a bottleneck if it overlaps well",
    "Very short capture windows can over-emphasize one kernel instance",
]


def _confidence(duration_ms: float, *, is_nccl: bool) -> float:
    """Heuristic confidence for kernel instance findings."""
    if is_nccl:
        return 0.7
    if duration_ms > 50:
        return 0.9
    if duration_ms > 5:
        return 0.8
    return 0.65


def _to_findings(rows: list[dict], *, context: dict | None = None) -> list:
    from nsys_ai.annotation import EvidenceRow, Finding, TraceSelection

    findings = []
    profile_id = (context or {}).get("profile_id", "unknown")
    for r in rows:
        if "error" in r:
            continue

        name = r.get("short_name") or r.get("kernel_name", "?")
        is_nccl = "nccl" in name.lower()
        kernel_name = r.get("kernel_name", name)
        short_name = r.get("short_name", name)
        dur_ms = float(r.get("duration_ms", 0.0) or 0.0)
        start_ns = int(r["start_ns"])
        end_ns = int(r["end_ns"])
        duration_ns = max(0, end_ns - start_ns)
        device_id = int(r.get("device_id", 0) or 0)
        stream_id = int(r.get("stream_id", 0) or 0)

        if is_nccl:
            label = (
                f"Long NCCL ({dur_ms:.2f}ms)"
                if dur_ms > 5.0
                else f"NCCL kernel ({dur_ms:.2f}ms)"
            )
            sev = "warning"
            category = "communication"
            explanation = _NCCL_EXPLANATION
            suggested_actions = _NCCL_ACTIONS
            row_kind = "long_nccl"
        else:
            label = f"Hotspot: {name[:30]}"
            sev = "info"
            category = "compute"
            explanation = _HOTSPOT_EXPLANATION
            suggested_actions = _HOTSPOT_ACTIONS
            row_kind = "kernel_hotspot"

        finding_id = f"kernel_instance_gpu{device_id}_stream{stream_id}_{start_ns}"
        selection = TraceSelection(
            id=f"sel_{finding_id}",
            profile_id=profile_id,
            source="skill:kernel_instances",
            start_ns=start_ns,
            end_ns=end_ns,
            gpu_ids=[device_id],
            stream_ids=[int(stream_id)] if stream_id is not None else None,
            label=label,
        )
        evidence_row = EvidenceRow(
            id=f"ev_{finding_id}",
            source_skill="kernel_instances",
            values={
                "kernel_name": kernel_name,
                "short_name": short_name,
                "duration_ms": round(dur_ms, 3),
                "duration_ns": duration_ns,
                "device_id": device_id,
                "stream_id": stream_id,
                "is_nccl": is_nccl,
            },
            units={
                "duration_ms": "ms",
                "duration_ns": "ns",
            },
            selection_id=selection.id,
            provenance={"row_kind": row_kind, "device": device_id, "stream": stream_id},
        )

        findings.append(
            Finding(
                type="highlight",
                label=label,
                start_ns=start_ns,
                end_ns=end_ns,
                gpu_id=device_id,
                stream=str(stream_id),
                severity=sev,
                note=f"{kernel_name[:60]}: {dur_ms:.2f}ms",
                id=finding_id,
                category=category,
                confidence=_confidence(dur_ms, is_nccl=is_nccl),
                evidence=[evidence_row],
                selection=selection,
                explanation=explanation,
                suggested_actions=list(suggested_actions),
                false_positive_notes=list(_FALSE_POSITIVE_NOTES),
                provenance={"skill": "kernel_instances", "row_kind": row_kind},
            )
        )
    return findings


SKILL = Skill(
    name="kernel_instances",
    title="Kernel Instance Details",
    description=(
        "Returns individual kernel execution instances with exact nanosecond "
        "timestamps (start_ns, end_ns). Use to get precise time ranges for "
        "findings.json evidence overlay on the timeline viewer. "
        "Includes both demangled (kernel_name) and short (short_name) identifiers."
    ),
    category="kernels",
    execute_fn=_execute,
    format_fn=_format,
    to_findings_fn=_to_findings,
    params=[
        SkillParam("device", "GPU device ID", "int", False, 0),
        SkillParam("name", "Kernel name substring filter (demangled)", "str", False, ""),
        SkillParam("limit", "Max instances to return", "int", False, 10),
    ],
    tags=["kernel", "instance", "timestamp", "evidence", "finding", "nanosecond"],
)
