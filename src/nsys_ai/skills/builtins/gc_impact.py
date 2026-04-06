"""Detect Python Garbage Collection and intensive memory free stalls."""

from nsys_ai.connection import DB_ERRORS, wrap_connection

from ..base import Skill


def _format(rows):
    if not rows:
        return "(No significant GC or memory allocation stalls detected)"
    lines = [
        "── GC & Memory Allocation Stalls ──",
        f"{'API / Event Name':<30s}  {'Count':>7s}  {'Total(ms)':>10s}  {'Max(ms)':>10s}  {'Avg(ms)':>10s}",
        "-" * 73,
    ]
    for r in rows:
        name = r["event_name"]
        if len(name) > 28:
            name = name[:25] + "..."
        lines.append(
            f"{name:<30s}  {r['occurrences']:>7d}  {r['total_ms']:>10.2f}  "
            f"{r['max_ms']:>10.2f}  {r['avg_ms']:>10.2f}"
        )
    return "\n".join(lines)


def _execute(conn, **kwargs):
    adapter = wrap_connection(conn)
    tables = adapter.resolve_activity_tables()
    runtime_table = tables.get("runtime", "CUPTI_ACTIVITY_KIND_RUNTIME")

    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")

    try:
        adapter.execute(f"SELECT 1 FROM {runtime_table} LIMIT 1")
    except DB_ERRORS:
        return []

    # --- Part 1: CUDA Memory APIs from Runtime table ---
    params_r = []
    trim_clause_r = ""
    if trim_start is not None and trim_end is not None:
        trim_clause_r = 'AND r.start >= ? AND r."end" <= ?'
        params_r.extend([trim_start, trim_end])

    sql_runtime = f"""
        SELECT
            s.value AS event_name,
            COUNT(*) AS occurrences,
            ROUND(SUM(r."end" - r.start) / 1e6, 2) AS total_ms,
            ROUND(MAX(r."end" - r.start) / 1e6, 2) AS max_ms,
            ROUND(AVG(r."end" - r.start) / 1e6, 2) AS avg_ms
        FROM {runtime_table} r
        JOIN StringIds s ON r.nameId = s.id
        WHERE (s.value LIKE '%cudaFree%'
           OR s.value LIKE '%cuMemFree%'
           OR s.value LIKE '%cudaMalloc%'
           OR s.value LIKE '%cuMemAlloc%')
          {trim_clause_r}
        GROUP BY s.value
        ORDER BY total_ms DESC
    """
    rows_runtime = adapter.execute(sql_runtime, params_r).fetchall()
    cols = ["event_name", "occurrences", "total_ms", "max_ms", "avg_ms"]
    results = [dict(zip(cols, r)) if isinstance(r, tuple) else dict(r) for r in rows_runtime]

    # --- Part 2: NVTX events mentioning GC ---
    nvtx_table = tables.get("nvtx")

    if nvtx_table:
        params_n = []
        trim_clause_n = ""
        if trim_start is not None and trim_end is not None:
            trim_clause_n = 'AND start >= ? AND "end" <= ?'
            params_n.extend([trim_start, trim_end])

        # Handle both text and textId schemas
        try:
            nvtx_cols = adapter.get_table_columns(nvtx_table)
        except DB_ERRORS:
            nvtx_cols = []

        if "textId" in nvtx_cols:
            text_expr = "COALESCE(n.text, sid.value)"
            text_join = "LEFT JOIN StringIds sid ON n.textId = sid.id"
            text_filter = f"(LOWER({text_expr}) LIKE '%gc%' OR LOWER({text_expr}) LIKE '%garbage%')"
            sql_nvtx = f"""
                SELECT
                    {text_expr} AS event_name,
                    COUNT(*) AS occurrences,
                    ROUND(SUM(n."end" - n.start) / 1e6, 2) AS total_ms,
                    ROUND(MAX(n."end" - n.start) / 1e6, 2) AS max_ms,
                    ROUND(AVG(n."end" - n.start) / 1e6, 2) AS avg_ms
                FROM {nvtx_table} n
                {text_join}
                WHERE {text_filter}
                  AND n.eventType IN (59, 60)
                  {trim_clause_n}
                GROUP BY {text_expr}
            """
        else:
            sql_nvtx = f"""
                SELECT
                    text AS event_name,
                    COUNT(*) AS occurrences,
                    ROUND(SUM("end" - start) / 1e6, 2) AS total_ms,
                    ROUND(MAX("end" - start) / 1e6, 2) AS max_ms,
                    ROUND(AVG("end" - start) / 1e6, 2) AS avg_ms
                FROM {nvtx_table}
                WHERE (LOWER(text) LIKE '%gc%' OR LOWER(text) LIKE '%garbage%')
                  AND eventType IN (59, 60)
                  {trim_clause_n}
                GROUP BY text
            """
        try:
            rows_nvtx = adapter.execute(sql_nvtx, params_n).fetchall()
            results.extend(
                dict(zip(cols, r)) if isinstance(r, tuple) else dict(r) for r in rows_nvtx
            )
        except DB_ERRORS:
            pass  # NVTX query is best-effort

    # Sort combined results by total_ms descending
    results.sort(key=lambda x: x.get("total_ms", 0), reverse=True)
    return results


SKILL = Skill(
    name="gc_impact",
    title="Garbage Collection & Memory Stalls",
    description=(
        "Quantifies CPU stalls caused by memory allocation (cudaMalloc), host-side memory freeing (cudaFree), "
        "and Python Garbage Collection (if annotated via NVTX). "
        "Frequent or long cudaFree calls usually indicate a bottleneck where the CPU is blocking."
    ),
    category="bottlenecks",
    execute_fn=_execute,
    format_fn=_format,
    tags=["gc", "memory", "cudafree", "cudamalloc", "stall", "bubble"],
)
