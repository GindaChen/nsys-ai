"""Detect JIT compilation and module loading stalls."""

from nsys_ai.connection import DB_ERRORS, wrap_connection

from ..base import Skill


def _format(rows):
    if not rows:
        return "(No JIT compilation or module loading stalls detected)"
    lines = [
        "── JIT Compilation & Module Loading Stalls ──",
        f"{'API Name':<30s}  {'Count':>7s}  {'Total(ms)':>10s}  {'Max(ms)':>10s}  {'Avg(ms)':>10s}",
        "-" * 73,
    ]
    for r in rows:
        name = r["api_name"]
        if len(name) > 28:
            name = name[:25] + "..."
        lines.append(
            f"{name:<30s}  {r['occurrences']:>7d}  {r['total_ms']:>10.2f}  "
            f"{r['max_ms']:>10.2f}  {r['avg_ms']:>10.2f}"
        )
    return "\n".join(lines)


def _execute(conn, **kwargs):
    tables = wrap_connection(conn).resolve_activity_tables()
    runtime_table = tables.get("runtime", "CUPTI_ACTIVITY_KIND_RUNTIME")

    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")

    try:
        conn.execute(f"SELECT 1 FROM {runtime_table} LIMIT 1")
    except DB_ERRORS:
        return []

    params = []
    trim_clause = ""
    if trim_start is not None and trim_end is not None:
        trim_clause = 'AND r.start >= ? AND r."end" <= ?'
        params.extend([trim_start, trim_end])

    # DuckDB / SQLite standard query using StringIds
    sql = f"""
        SELECT
            s.value AS api_name,
            COUNT(*) AS occurrences,
            ROUND(SUM(r."end" - r.start) / 1e6, 2) AS total_ms,
            ROUND(MAX(r."end" - r.start) / 1e6, 2) AS max_ms,
            ROUND(AVG(r."end" - r.start) / 1e6, 2) AS avg_ms
        FROM {runtime_table} r
        JOIN StringIds s ON r.nameId = s.id
        WHERE (s.value LIKE '%CompilePTX%'
           OR s.value LIKE '%cuModuleLoad%'
           OR s.value LIKE '%cudaModuleLoad%'
           OR s.value LIKE '%cuModuleGetLoadingMode%')
          {trim_clause}
        GROUP BY s.value
        ORDER BY total_ms DESC
    """
    rows = conn.execute(sql, params).fetchall()
    cols = ["api_name", "occurrences", "total_ms", "max_ms", "avg_ms"]
    return [dict(zip(cols, r)) if isinstance(r, tuple) else dict(r) for r in rows]


SKILL = Skill(
    name="module_loading",
    title="JIT Compilation & Module Loading Stalls",
    description=(
        "Detects JIT compilation (e.g., PTX) and module loading events via CUDA API. "
        "These operations usually block the CPU thread and can cause major pipeline bubbles "
        "if they happen during the main training loop instead of initialization."
    ),
    category="bottlenecks",
    execute_fn=_execute,
    format_fn=_format,
    tags=["jit", "compile", "module", "stall", "bubble", "cuda"],
)
