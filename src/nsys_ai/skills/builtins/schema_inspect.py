"""Schema inspection — list all tables and their columns."""

from ..base import Skill


def _format(rows):
    if not rows:
        return "(No tables found in database)"
    lines = ["── Database Schema ──", ""]
    current_table = None
    for r in rows:
        if r["table_name"] != current_table:
            if current_table is not None:
                lines.append("")
            current_table = r["table_name"]
            lines.append(f"  {current_table}")
            lines.append(f"  {'─' * len(current_table)}")
        col_name = r["column_name"]
        col_type = r["column_type"] or ""
        pk = " (PK)" if r["is_pk"] else ""
        lines.append(f"    {col_name:<30s}  {col_type:<15s}{pk}")
    return "\n".join(lines)


SKILL = Skill(
    name="schema_inspect",
    title="Database Schema Inspector",
    description=(
        "Lists all tables and their columns in the Nsight SQLite database. "
        "Use this first to understand what data is available before running "
        "other skills. Different nsys versions may have different tables."
    ),
    category="utility",
    sql="""\
SELECT m.name AS table_name,
       p.name AS column_name,
       p.type AS column_type,
       p.pk AS is_pk
FROM sqlite_master m
JOIN pragma_table_info(m.name) p
WHERE m.type = 'table'
  AND m.name NOT LIKE 'sqlite_%'
ORDER BY m.name, p.cid""",
    format_fn=_format,
    tags=["schema", "tables", "columns", "inspect", "meta", "utility"],
)
