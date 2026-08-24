"""Small, frequent kernels and their CUDA launch API cost."""

from ..base import Skill, SkillParam, abstain, is_abstention_row

_SMALL_KERNEL_AVG_US_THRESHOLD = 10.0
_MIN_LAUNCH_COUNT = 100

_SMALL_KERNEL_EXPLANATION = (
    "A kernel that averages less than 10 microseconds and is launched at least "
    "100 times has the small-and-frequent pattern from Root Cause #5. Its CUDA "
    "launch API work can be material relative to the kernel execution itself."
)
_SMALL_KERNEL_ACTIONS = [
    "Use torch.compile() to fuse repeated element-wise operations",
    "Use CUDA Graphs for static, repeated kernel sequences",
    "Write a fused Triton or CUDA kernel for the hot small-kernel pattern",
]
_SMALL_KERNEL_FALSE_POSITIVES = [
    "Initialization kernels can repeat in short captures without representing steady state",
    "Queue time can indicate useful GPU saturation and is not an optimization target by itself",
]


def _safe_id(name: str) -> str:
    value = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return value[:64] or "unknown"


def _small_kernel_confidence(avg_kernel_us: float, launch_count: int) -> float:
    if avg_kernel_us < 5 and launch_count >= 1000:
        return 0.95
    if launch_count >= 500:
        return 0.85
    return 0.70


def _required_tables(conn):
    from nsys_ai.connection import wrap_connection

    adapter = wrap_connection(conn)
    activity = adapter.resolve_activity_tables()
    missing = [
        canonical
        for key, canonical in (
            ("kernel", "CUPTI_ACTIVITY_KIND_KERNEL"),
            ("runtime", "CUPTI_ACTIVITY_KIND_RUNTIME"),
        )
        if not activity.get(key)
    ]
    names = adapter.get_table_names()
    if "string_ids" in names:
        string_table = "string_ids"
    elif "StringIds" in names:
        string_table = "StringIds"
    else:
        string_table = None
        missing.append("StringIds")
    if missing:
        return adapter, None, None, None, abstain(
            "Kernel launch analysis needs CUDA kernel, CUDA runtime, and StringIds "
            "tables; this profile does not contain all required activity.",
            missing_tables=missing,
        )

    kernel_table = activity["kernel"]
    runtime_table = activity["runtime"]
    required_columns = {
        kernel_table: {
            "globalPid", "deviceId", "streamId", "correlationId",
            "start", "end", "shortName",
        },
        runtime_table: {"globalTid", "correlationId", "start", "end", "nameId"},
        string_table: {"id", "value"},
    }
    missing_columns = {}
    for table, columns in required_columns.items():
        absent = columns - set(adapter.get_table_columns(table))
        if absent:
            missing_columns[table] = sorted(absent)
    if missing_columns:
        return adapter, None, None, None, abstain(
            "Kernel launch analysis cannot establish device, process, and launch "
            "identity because required columns are absent.",
            missing_columns=missing_columns,
        )
    return adapter, kernel_table, runtime_table, string_table, None


def _execute(conn, **kwargs):
    limit = max(1, int(kwargs.get("limit", 20)))
    min_launches = max(1, int(kwargs.get("min_launches", _MIN_LAUNCH_COUNT)))
    device = int(kwargs.get("device", 0))
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")

    adapter, kernel_table, runtime_table, string_table, unavailable = _required_tables(conn)
    if unavailable is not None:
        return unavailable

    try:
        kernel_columns = {
            str(column).lower(): str(column)
            for column in adapter.get_table_columns(kernel_table)
        }
    except Exception:  # noqa: BLE001 - required-table validation already ran
        kernel_columns = {}
    graph_node_column = kernel_columns.get("graphnodeid") or kernel_columns.get("graph_node_id")
    graph_id_column = kernel_columns.get("graphid") or kernel_columns.get("graph_id")
    graph_node_expr = f'k."{graph_node_column}"' if graph_node_column else "NULL"
    graph_id_expr = f'k."{graph_id_column}"' if graph_id_column else "NULL"

    trim_clause = ""
    params: list[int] = [device]
    if trim_start is not None and trim_end is not None:
        trim_clause = ' AND k.start >= ? AND k."end" <= ?'
        params.extend([int(trim_start), int(trim_end)])
    params.append(min_launches)

    # Nsight encodes the process in the high bits of globalTid. Apply NVIDIA's
    # mask unconditionally; compact unencoded test IDs would weaken the same
    # process boundary this join is responsible for enforcing.
    sql = f"""
        WITH launch_runtime AS (
            SELECT
                r.globalTid & CAST(-16777216 AS BIGINT) AS process_id,
                r.globalTid AS global_tid,
                r.correlationId AS correlation_id,
                r.start AS api_start_ns,
                r."end" AS api_end_ns,
                r."end" - r.start AS api_duration_ns
            FROM {runtime_table} r
            JOIN {string_table} s_api ON r.nameId = s_api.id
            WHERE (s_api.value LIKE 'cudaLaunch%'
                   OR s_api.value LIKE 'cuLaunch%')
              AND r."end" >= r.start
        ),
        launch_candidates AS (
            SELECT
                k.globalPid AS process_id,
                k.deviceId AS device_id,
                k.streamId AS stream_id,
                k.correlationId AS correlation_id,
                k.start AS kernel_start_ns,
                k."end" AS kernel_end_ns,
                k.shortName AS short_name,
                {graph_node_expr} AS graph_node_id,
                {graph_id_expr} AS graph_id,
                r.global_tid,
                r.api_start_ns,
                r.api_end_ns,
                r.api_duration_ns,
                CASE
                    WHEN k.start - r.api_end_ns > 0 THEN k.start - r.api_end_ns
                    ELSE NULL
                END AS queue_duration_ns,
                k."end" - k.start AS kernel_duration_ns,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        k.globalPid, k.deviceId, k.streamId, k.correlationId,
                        k.start, k."end", k.shortName
                    ORDER BY
                        r.api_start_ns DESC,
                        r.api_end_ns DESC,
                        r.global_tid ASC
                ) AS match_rank
            FROM {kernel_table} k
            JOIN launch_runtime r
              ON k.correlationId = r.correlation_id
             AND k.globalPid = r.process_id
            WHERE k.deviceId = ?
              AND k."end" >= k.start
              AND r.api_start_ns <= k.start
              {trim_clause}
        ),
        launches AS (
            SELECT
                process_id, device_id, stream_id, correlation_id,
                kernel_start_ns, kernel_end_ns, short_name, global_tid,
                api_start_ns, api_end_ns, api_duration_ns,
                queue_duration_ns, kernel_duration_ns, graph_node_id, graph_id,
                CASE
                    WHEN ROW_NUMBER() OVER (
                        PARTITION BY global_tid, correlation_id, api_start_ns, api_end_ns
                        ORDER BY kernel_start_ns, kernel_end_ns, short_name
                    ) = 1
                    THEN api_duration_ns
                    ELSE 0
                END AS api_charge_ns
            FROM launch_candidates
            WHERE match_rank = 1
        )
        SELECT
            s_kernel.value AS kernel_name,
            l.device_id,
            COUNT(*) AS launch_count,
            ROUND(SUM(l.api_charge_ns) / 1e6, 3) AS total_api_ms,
            ROUND(AVG(NULLIF(l.api_charge_ns, 0)) / 1e3, 1) AS avg_api_us,
            ROUND(MAX(l.api_charge_ns) / 1e3, 1) AS max_api_us,
            COUNT(l.queue_duration_ns) AS queue_count,
            ROUND(COALESCE(SUM(l.queue_duration_ns), 0) / 1e6, 3) AS total_queue_ms,
            ROUND(COALESCE(AVG(l.queue_duration_ns), 0) / 1e3, 1) AS avg_queue_us,
            ROUND(SUM(l.kernel_duration_ns) / 1e6, 3) AS total_kernel_ms,
            ROUND(AVG(l.kernel_duration_ns) / 1e3, 3) AS avg_kernel_us,
            COUNT(DISTINCT l.graph_node_id) AS graph_node_count,
            COUNT(DISTINCT l.graph_id) AS graph_id_count,
            MIN(l.kernel_start_ns) AS span_start_ns,
            MAX(l.kernel_end_ns) AS span_end_ns
        FROM launches l
        JOIN {string_table} s_kernel ON l.short_name = s_kernel.id
        GROUP BY s_kernel.value, l.device_id
        HAVING COUNT(*) >= ?
           AND AVG(l.kernel_duration_ns) < {_SMALL_KERNEL_AVG_US_THRESHOLD * 1000}
        ORDER BY SUM(l.api_charge_ns) DESC,
                 COUNT(*) DESC,
                 s_kernel.value ASC,
                 l.device_id ASC
        LIMIT {limit}
    """
    cursor = adapter.execute(sql, params)
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _format(rows):
    real_rows = [row for row in rows if not is_abstention_row(row)]
    if not real_rows:
        return "(No small-and-frequent kernel launch data found)"
    lines = [
        "-- Small and Frequent Kernel Launches --",
        f"{'Kernel':<50s}  {'Launches':>9s}  {'API avg(us)':>11s}  "
        f"{'Queue avg(us)':>13s}  {'Kernel avg(us)':>14s}",
        "-" * 106,
    ]
    for row in real_rows:
        name = row["kernel_name"]
        if len(name) > 48:
            name = name[:45] + "..."
        lines.append(
            f"{name:<50s}  {row['launch_count']:>9d}  {row['avg_api_us']:>11.1f}  "
            f"{row['avg_queue_us']:>13.1f}  {row['avg_kernel_us']:>14.3f}"
        )
    return "\n".join(lines)


def _to_findings(rows: list[dict], *, context: dict | None = None) -> list:
    from nsys_ai.annotation import EvidenceRow, Finding, TraceSelection

    profile_id = (context or {}).get("profile_id", "unknown")
    findings = []
    for row in rows:
        if is_abstention_row(row):
            continue
        launch_count = int(row.get("launch_count", 0) or 0)
        avg_kernel_us = float(row.get("avg_kernel_us", 0) or 0)
        if launch_count < _MIN_LAUNCH_COUNT or avg_kernel_us >= _SMALL_KERNEL_AVG_US_THRESHOLD:
            continue
        kernel_name = str(row["kernel_name"])
        finding_id = f"klo_small_kernel_overhead_{_safe_id(kernel_name)}"
        selection = TraceSelection(
            id=f"sel_{finding_id}",
            profile_id=profile_id,
            source="skill:kernel_launch_overhead",
            start_ns=int(row["span_start_ns"]),
            end_ns=int(row["span_end_ns"]),
            gpu_ids=[int(row["device_id"])],
            label=f"Small kernel invocations: {kernel_name[:40]}",
        )
        evidence = EvidenceRow(
            id=f"ev_{finding_id}",
            source_skill="kernel_launch_overhead",
            values={
                "kernel_name": kernel_name,
                "launch_count": launch_count,
                "avg_kernel_us": round(avg_kernel_us, 3),
                "total_kernel_ms": float(row["total_kernel_ms"]),
                "avg_api_us": float(row["avg_api_us"]),
                "total_api_ms": float(row["total_api_ms"]),
                "queue_count": int(row["queue_count"]),
                "avg_queue_us": float(row["avg_queue_us"]),
                "total_queue_ms": float(row["total_queue_ms"]),
                "graph_node_count": int(row.get("graph_node_count", 0) or 0),
                "graph_id_count": int(row.get("graph_id_count", 0) or 0),
            },
            units={
                "launch_count": "count",
                "avg_kernel_us": "microseconds",
                "total_kernel_ms": "ms",
                "avg_api_us": "microseconds",
                "total_api_ms": "ms",
                "queue_count": "count",
                "avg_queue_us": "microseconds",
                "total_queue_ms": "ms",
                "graph_node_count": "count",
                "graph_id_count": "count",
            },
            selection_id=selection.id,
            provenance={
                "row_kind": "small_kernel_overhead",
                "kernel_name": kernel_name,
                "root_cause": "src/nsys_ai/data/book.md#5",
            },
        )
        findings.append(Finding(
            type="highlight",
            label=(
                f"Small Kernel Overhead: {kernel_name[:40]} "
                f"({avg_kernel_us:.1f}us avg, {launch_count} launches)"
            ),
            start_ns=0,
            end_ns=None,
            gpu_id=int(row["device_id"]),
            severity="warning",
            note=(
                f"{kernel_name} averages {avg_kernel_us:.1f}us across "
                f"{launch_count} launches; CUDA launch API time totals "
                f"{float(row['total_api_ms']):.1f}ms."
            ),
            id=finding_id,
            category="launch_overhead",
            confidence=_small_kernel_confidence(avg_kernel_us, launch_count),
            evidence=[evidence],
            selection=selection,
            explanation=_SMALL_KERNEL_EXPLANATION,
            suggested_actions=list(_SMALL_KERNEL_ACTIONS),
            false_positive_notes=list(_SMALL_KERNEL_FALSE_POSITIVES),
            provenance={
                "skill": "kernel_launch_overhead",
                "row_kind": "small_kernel_overhead",
                "root_cause": "src/nsys_ai/data/book.md#5",
            },
        ))
    return findings


SKILL = Skill(
    name="kernel_launch_overhead",
    title="Kernel Launch Overhead",
    description=(
        "Finds kernels averaging under 10 microseconds with at least 100 launches, "
        "and reports CUDA launch API duration separately from nonnegative queue time."
    ),
    category="kernels",
    execute_fn=_execute,
    format_fn=_format,
    to_findings_fn=_to_findings,
    params=[
        SkillParam("limit", "Max small-kernel candidates", "int", False, 20),
        SkillParam("min_launches", "Min launch count", "int", False, _MIN_LAUNCH_COUNT),
        SkillParam("device", "GPU device ID", "int", False, 0),
    ],
    tags=["launch", "overhead", "latency", "cpu", "small-kernel", "fusion"],
)
