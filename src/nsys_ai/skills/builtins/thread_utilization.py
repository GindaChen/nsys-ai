"""CPU thread utilization analysis."""

import sqlite3

from ..base import Skill, SkillParam


def _execute(conn: sqlite3.Connection, *, limit: int = 10, **_kwargs):
    """Check for COMPOSITE_EVENTS table before querying."""
    import duckdb as _ddb

    if isinstance(conn, _ddb.DuckDBPyConnection):
        tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    else:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

    if "COMPOSITE_EVENTS" not in tables:
        return []

    from nsys_ai.sql_compat import sqlite_to_duckdb

    sql = f"""\
SELECT ce.globalTid % 0x1000000 AS tid,
       (SELECT s.value FROM StringIds s
        WHERE s.id = (SELECT tn.nameId FROM ThreadNames tn
                      WHERE tn.globalTid = ce.globalTid LIMIT 1)
       ) AS thread_name,
       ROUND(100.0 * SUM(ce.cpuCycles) / (
           SELECT MAX(1, SUM(cpuCycles)) FROM COMPOSITE_EVENTS
       ), 2) AS cpu_pct
FROM COMPOSITE_EVENTS ce
GROUP BY ce.globalTid
ORDER BY cpu_pct DESC
LIMIT {int(limit)}"""

    if isinstance(conn, _ddb.DuckDBPyConnection):
        sql = sqlite_to_duckdb(sql)
    cursor = conn.execute(sql)
    columns = [desc[0] for desc in cursor.description] if cursor.description else []
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _format(rows):
    if not rows:
        return "(No CPU utilization data found — COMPOSITE_EVENTS table may be missing)"
    lines = [
        "── CPU Thread Utilization ──",
        f"{'TID':>8s}  {'Thread Name':<40s}  {'CPU %':>7s}",
        "─" * 60,
    ]
    for r in rows:
        name = r["thread_name"] or "(unnamed)"
        if len(name) > 38:
            name = name[:35] + "..."
        cpu_pct = r["cpu_pct"] if r["cpu_pct"] is not None else 0
        lines.append(f"{r['tid']:>8d}  {name:<40s}  {cpu_pct:>7.2f}")
    return "\n".join(lines)


SKILL = Skill(
    name="thread_utilization",
    title="CPU Thread Utilization",
    description=(
        "Shows CPU utilization by thread — helps identify whether a CPU-bound "
        "thread is starving the GPU of work. Common in data loading, preprocessing, "
        "or Python GIL contention scenarios."
    ),
    category="system",
    execute_fn=_execute,
    params=[SkillParam("limit", "Max threads to show", "int", False, 10)],
    format_fn=_format,
    tags=["cpu", "thread", "utilization", "bottleneck", "GIL"],
)

