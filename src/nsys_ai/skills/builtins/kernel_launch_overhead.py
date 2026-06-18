"""CUDA API launch overhead — time between API call and kernel execution."""

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    from nsys_ai.connection import DB_ERRORS, wrap_connection

    limit = int(kwargs.get("limit", 20))
    min_launches = int(kwargs.get("min_launches", 100))
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")

    # Detect backend
    try:
        conn.execute("SELECT 1 FROM kernels LIMIT 1")
        kernel_tbl = "kernels"
        runtime_tbl = "runtime"
        string_tbl = "string_ids"
    except DB_ERRORS:
        adapter = wrap_connection(conn)
        tables = adapter.resolve_activity_tables()
        kernel_tbl = tables.get("kernel", "CUPTI_ACTIVITY_KIND_KERNEL")
        runtime_tbl = tables.get("runtime", "CUPTI_ACTIVITY_KIND_RUNTIME")
        string_tbl = "StringIds"

    trim_kernel = ""
    trim_runtime = ""
    params: list = []
    if trim_start is not None and trim_end is not None:
        trim_kernel = ' AND k.start >= ? AND k."end" <= ?'
        trim_runtime = ' AND r.start >= ? AND r."end" <= ?'
        params = [trim_start, trim_end, trim_start, trim_end]

    # The `launch_runtime` CTE filters runtime rows to just cuda{,Launch}Kernel*
    # APIs — this is the actual fix that prevents negative overheads (which
    # arose from joining non-launch runtime rows that happen to share a
    # correlationId). In observed profiles correlationId is 1:1 between launch
    # API and kernel, so the MAX(r.start) is effectively a single-row selector;
    # we still use MAX(...) GROUP BY (rather than a window function) to stay
    # compatible with DuckDB 1.5.0 (ROW_NUMBER crashes on BIGINT/INTEGER mix).
    sql = f"""
        WITH launch_runtime AS (
            SELECT r.correlationId, r.start, r."end"
            FROM {runtime_tbl} r
            JOIN {string_tbl} s_api ON r.nameId = s_api.id
            WHERE (s_api.value LIKE 'cudaLaunchKernel%'
                OR s_api.value LIKE 'cuLaunchKernel%')
                {trim_runtime}
        ),
        best_match AS (
            SELECT
                k.shortName,
                k.start AS k_start,
                k."end" AS k_end,
                MAX(r.start) AS r_start
            FROM {kernel_tbl} k
            JOIN launch_runtime r
              ON r.correlationId = k.correlationId
             AND r.start <= k.start
            WHERE 1=1 {trim_kernel}
            GROUP BY k.shortName, k.start, k."end"
        )
        SELECT s_k.value AS kernel_name,
               COUNT(*) AS launch_count,
               ROUND(SUM(k_start - r_start) / 1e6, 3) AS total_overhead_ms,
               ROUND(AVG(k_start - r_start) / 1e3, 1) AS avg_overhead_us,
               ROUND(MAX(k_start - r_start) / 1e3, 1) AS max_overhead_us,
               ROUND(MIN(k_start - r_start) / 1e3, 1) AS min_overhead_us,
               ROUND(SUM(k_end - k_start) / 1e6, 3) AS total_kernel_ms
        FROM best_match m
        JOIN {string_tbl} s_k ON m.shortName = s_k.id
        GROUP BY s_k.value
        HAVING launch_count >= {min_launches}
        ORDER BY total_overhead_ms DESC
        LIMIT {limit}
    """

    rows_raw = conn.execute(sql, params).fetchall()
    cols = ["kernel_name", "launch_count", "total_overhead_ms",
            "avg_overhead_us", "max_overhead_us", "min_overhead_us", "total_kernel_ms"]
    rows = [dict(zip(cols, r)) if isinstance(r, tuple) else dict(r) for r in rows_raw]

    # Derived field
    for r in rows:
        denom = r["total_overhead_ms"] + r["total_kernel_ms"]
        r["overhead_pct"] = round(100 * r["total_overhead_ms"] / denom, 1) if denom > 0 else 0.0

    # Inject metadata for findings — prefer trim window when provided so the
    # finding span matches the actually-analyzed range, not the whole profile.
    from ...profile import Profile
    prof = Profile._from_conn(conn)
    device = int(kwargs.get("device", 0))
    if trim_start is not None and trim_end is not None:
        span_start, span_end = int(trim_start), int(trim_end)
    else:
        span_start, span_end = prof.meta.time_range

    # If no kernel rows survived the min_launches filter, still emit a
    # metadata-only synthetic row so the independent Excessive Sync finding
    # (decoupled from kernel data) can still fire.
    if not rows:
        rows = [{
            "kernel_name": None,
            "launch_count": 0,
            "total_overhead_ms": 0.0,
            "avg_overhead_us": 0.0,
            "max_overhead_us": 0.0,
            "min_overhead_us": 0.0,
            "total_kernel_ms": 0.0,
            "overhead_pct": 0.0,
        }]

    for r in rows:
        r["device_id"] = device
        r["span_start_ns"] = span_start
        r["span_end_ns"] = span_end

    # src/nsys_ai/data/book.md Root Cause #10: count cudaDeviceSynchronize calls (within trim window if set,
    # so sync_count is consistent with the kernel/launch rows above).
    sync_trim = ""
    sync_params: list = []
    if trim_start is not None and trim_end is not None:
        sync_trim = ' AND r.start >= ? AND r."end" <= ?'
        sync_params = [trim_start, trim_end]
    sync_sql = f"""
        SELECT COUNT(*) AS cnt
        FROM {runtime_tbl} r
        JOIN {string_tbl} s ON r.nameId = s.id
        WHERE s.value LIKE 'cudaDeviceSynchronize%'{sync_trim}
    """
    sync_count = conn.execute(sync_sql, sync_params).fetchone()
    sync_count = sync_count[0] if sync_count else 0
    for r in rows:
        r["_global_sync_count"] = int(sync_count or 0)

    return rows

def _format(rows):
    # Filter out metadata-only synthetic rows (launch_count == 0).
    real_rows = [r for r in rows if r.get("launch_count", 0) > 0]
    if not real_rows:
        return "(No kernel launch overhead data found)"
    lines = [
        "── Kernel Launch Overhead (per-kernel aggregated) ──",
        f"{'Kernel':<50s}  {'Launches':>9s}  {'Avg(μs)':>9s}  {'Max(μs)':>10s}  {'Total(ms)':>10s}  {'Oh%':>5s}",
        "─" * 100,
    ]
    for r in real_rows:
        name = r["kernel_name"]
        if len(name) > 48:
            name = name[:45] + "..."
        lines.append(
            f"{name:<50s}  {r['launch_count']:>9d}  {r['avg_overhead_us']:>9.1f}  "
            f"{r['max_overhead_us']:>10.1f}  {r['total_overhead_ms']:>10.2f}  {r['overhead_pct']:>5.1f}"
        )
    return "\n".join(lines)

_SMALL_KERNEL_EXPLANATION = (
    "A kernel that runs <10μs on average and is launched 100+ times — the "
    "classic small-and-frequent pattern targeted by Root Cause #5 (Small Kernel "
    "Overhead). At this size the per-launch fixed cost (CPU-side dispatch, "
    "driver work, queue management) is non-trivial relative to actual kernel "
    "execution, making the kernel a strong candidate for fusion or CUDA Graphs."
)
_EXCESSIVE_SYNC_EXPLANATION = (
    "Profile contains many cudaDeviceSynchronize calls. Per src/nsys_ai/data/book.md Root Cause #10 "
    "(Excessive Synchronization): explicit syncs force the GPU pipeline to drain, "
    "preventing overlap and adding latency."
)
_SUGGESTED_ACTIONS = [
    "Use torch.compile() to fuse element-wise ops automatically",
    "Use CUDA Graphs for static, repeated kernel sequences",
    "Write fused Triton/CUDA kernels for hot small-kernel patterns",
    "Remove .item() / .cpu() calls from training loops (they force sync)",
    "Replace cudaDeviceSynchronize with event-based dependencies",
]
_FALSE_POSITIVE_NOTES = [
    "The reported overhead (k.start − r.start) is CPU-runs-ahead queue depth in any async workload, not pure dispatch latency — treat it as a directional signal, not an exact cost",
    "Initialization-phase kernels (one-time setup) may appear small-and-frequent in short profiles but are not steady-state",
    "Profiles with very few launches lack statistical confidence even if a kernel looks tiny",
    "A few cudaDeviceSynchronize calls are normal (e.g. at end of training step or for diagnostic logging)",
]

# Thresholds (sourced from book.md root causes where noted)
_SMALL_KERNEL_AVG_US_THRESHOLD = 10.0     # src/nsys_ai/data/book.md Root Cause #5: "tiny kernels < 10μs"
_MIN_LAUNCH_COUNT = 100                    # statistical confidence floor
_EXCESSIVE_SYNC_COUNT_THRESHOLD = 100      # book.md src/nsys_ai/data/book.md Root Cause #10 calibration


def _small_kernel_confidence(avg_kernel_us: float, launch_count: int) -> float:
    # Confidence ramps on the two physical signals we actually trust:
    # how small the kernel is, and how often it runs. The k.start - r.start
    # "overhead" metric is queue-depth-confounded in async workloads, so we
    # deliberately do not use it for gating or confidence.
    if launch_count < 100:
        return 0.5
    if avg_kernel_us < 5 and launch_count >= 1000:
        return 0.95
    if avg_kernel_us < 10 and launch_count >= 500:
        return 0.85
    return 0.70


def _sync_confidence(sync_count: int) -> float:
    if sync_count >= 1000:
        return 0.95
    if sync_count >= 100:
        return 0.80
    return 0.60


def _to_findings(rows: list[dict], *, context: dict | None = None) -> list:
    from nsys_ai.annotation import EvidenceRow, Finding, TraceSelection

    findings = []
    if not rows:
        return findings

    profile_id = (context or {}).get("profile_id", "unknown")
    device = rows[0].get("device_id", (context or {}).get("device", 0))
    start_ns = rows[0].get("span_start_ns", 0)
    end_ns = rows[0].get("span_end_ns", 0)
    sync_count = rows[0].get("_global_sync_count", 0)

    # Finding 1: Small Kernel Overhead — per src/nsys_ai/data/book.md Root Cause #5
    for r in rows:
        if r.get("launch_count", 0) == 0:
            continue  # skip metadata-only synthetic row
        avg_kernel_us = r["total_kernel_ms"] * 1000.0 / r["launch_count"] if r["launch_count"] else 0
        if (avg_kernel_us < _SMALL_KERNEL_AVG_US_THRESHOLD
                and r["launch_count"] >= _MIN_LAUNCH_COUNT):
            finding_id = f"klo_small_kernel_overhead_{r['kernel_name'][:30]}"
            selection = TraceSelection(
                id=f"sel_{finding_id}",
                profile_id=profile_id,
                source="skill:kernel_launch_overhead",
                start_ns=start_ns, end_ns=end_ns,
                gpu_ids=[device],
                label=f"Small Kernel Overhead ({r['kernel_name'][:40]}, {avg_kernel_us:.1f}μs avg exec)",
            )
            evidence_row = EvidenceRow(
                id=f"ev_{finding_id}",
                source_skill="kernel_launch_overhead",
                values={
                    "kernel_name": r["kernel_name"],
                    "avg_kernel_us": round(avg_kernel_us, 2),
                    "launch_count": r["launch_count"],
                    "total_kernel_ms": r["total_kernel_ms"],
                },
                units={"avg_kernel_us": "microseconds", "launch_count": "count", "total_kernel_ms": "ms"},
                selection_id=selection.id,
                provenance={"row_kind": "small_kernel_overhead", "kernel_name": r["kernel_name"], "root_cause": "src/nsys_ai/data/book.md#5"},
            )
            findings.append(Finding(
                type="region",
                label=f"Small Kernel Overhead: {r['kernel_name'][:40]} ({avg_kernel_us:.1f}μs avg, {r['launch_count']} launches)",
                start_ns=start_ns, end_ns=end_ns,
                gpu_id=device,
                severity="warning",
                note=f"{r['kernel_name']}: runs only {avg_kernel_us:.1f}μs per call and is launched {r['launch_count']} times — small-and-frequent pattern, candidate for fusion or CUDA Graphs.",
                id=finding_id,
                category="kernels",
                confidence=_small_kernel_confidence(avg_kernel_us, r["launch_count"]),
                evidence=[evidence_row],
                selection=selection,
                explanation=_SMALL_KERNEL_EXPLANATION,
                suggested_actions=list(_SUGGESTED_ACTIONS),
                false_positive_notes=list(_FALSE_POSITIVE_NOTES),
                provenance={"skill": "kernel_launch_overhead", "row_kind": "small_kernel_overhead", "root_cause": "src/nsys_ai/data/book.md#5"},
            ))

    # Finding 2: Excessive Synchronization — per src/nsys_ai/data/book.md Root Cause #10
    if sync_count > _EXCESSIVE_SYNC_COUNT_THRESHOLD:
        finding_id = "klo_excessive_sync"
        selection = TraceSelection(
            id=f"sel_{finding_id}",
            profile_id=profile_id,
            source="skill:kernel_launch_overhead",
            start_ns=start_ns, end_ns=end_ns,
            gpu_ids=[device],
            label=f"Excessive cudaDeviceSynchronize ({sync_count} calls)",
        )
        evidence_row = EvidenceRow(
            id=f"ev_{finding_id}",
            source_skill="kernel_launch_overhead",
            values={"sync_count": sync_count},
            units={"sync_count": "count"},
            selection_id=selection.id,
            provenance={"row_kind": "excessive_sync", "root_cause": "src/nsys_ai/data/book.md#10"},
        )
        findings.append(Finding(
            type="region",
            label=f"Excessive Synchronization ({sync_count} cudaDeviceSynchronize calls)",
            start_ns=start_ns, end_ns=end_ns,
            gpu_id=device,
            severity="warning",
            note=f"Profile contains {sync_count} cudaDeviceSynchronize calls — forces GPU pipeline drains.",
            id=finding_id,
            category="kernels",
            confidence=_sync_confidence(sync_count),
            evidence=[evidence_row],
            selection=selection,
            explanation=_EXCESSIVE_SYNC_EXPLANATION,
            suggested_actions=list(_SUGGESTED_ACTIONS),
            false_positive_notes=list(_FALSE_POSITIVE_NOTES),
            provenance={"skill": "kernel_launch_overhead", "row_kind": "excessive_sync", "root_cause": "src/nsys_ai/data/book.md#10"},
        ))

    return findings


SKILL = Skill(
    name="kernel_launch_overhead",
    title="Kernel Launch Overhead",
    description=(
        "Per-kernel aggregated gap between launch API call (cuda{,Launch}Kernel*) "
        "and GPU execution start. In async workloads this gap conflates true "
        "dispatch latency with CPU-runs-ahead queue depth, so treat it as a "
        "directional signal for small-and-frequent kernels rather than an exact "
        "dispatch cost. Joins kernels to launch-API runtime rows only (filtering "
        "out non-launch APIs that share correlationIds in some captures)."
    ),
    category="kernels",
    execute_fn=_execute,
    format_fn=_format,
    to_findings_fn=_to_findings,
    params=[
        SkillParam("limit", "Max number of kernels to return", "int", False, 20),
        SkillParam("min_launches", "Min launch count to include kernel", "int", False, 100),
        SkillParam("device", "GPU device ID", "int", False, 0),
    ],
    tags=["launch", "overhead", "latency", "cpu", "bottleneck", "kernel"],
)
