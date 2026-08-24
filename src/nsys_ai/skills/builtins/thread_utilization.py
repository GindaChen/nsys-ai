"""Built-in skill that reports CPU thread utilization from COMPOSITE_EVENTS."""

from __future__ import annotations

from typing import Any

from ..base import Skill, SkillParam, abstain


def _execute(conn: Any, *, limit: int = 10, **_kwargs):
    """Check for COMPOSITE_EVENTS table before querying."""
    from nsys_ai.connection import wrap_connection

    adapter = wrap_connection(conn)

    tables = adapter.get_table_names()

    if "COMPOSITE_EVENTS" not in tables:
        # Missing table means this skill cannot run, which is not the same as
        # running and finding every thread idle. Returning [] conflated the two.
        return abstain(
            "This profile has no COMPOSITE_EVENTS table, so CPU sampling was "
            "not captured. Thread utilization needs sampled CPU activity — "
            "re-capture with CPU sampling enabled to use this skill."
        )

    sql = f"""\
WITH total_cycles AS (
    SELECT CASE
               WHEN COALESCE(SUM(cpuCycles), 0) < 1 THEN 1
               ELSE SUM(cpuCycles)
           END AS total_cpu_cycles
    FROM COMPOSITE_EVENTS
), named_threads AS (
    SELECT tn.globalTid,
           s.value AS thread_name,
           ROW_NUMBER() OVER (
               PARTITION BY tn.globalTid ORDER BY tn.nameId ASC
           ) AS name_rank
    FROM ThreadNames tn
    LEFT JOIN StringIds s ON s.id = tn.nameId
)
SELECT ce.globalTid % 0x1000000 AS tid,
       nt.thread_name,
       ROUND(100.0 * SUM(ce.cpuCycles) / total_cycles.total_cpu_cycles, 2)
           AS cpu_pct
FROM COMPOSITE_EVENTS ce
CROSS JOIN total_cycles
LEFT JOIN named_threads nt
       ON nt.globalTid = ce.globalTid AND nt.name_rank = 1
GROUP BY ce.globalTid, nt.thread_name, total_cycles.total_cpu_cycles
-- globalTid, not the masked tid: the mask can collide across processes.
ORDER BY cpu_pct DESC, ce.globalTid ASC
LIMIT {int(limit)}"""

    cursor = adapter.execute(sql)
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
