"""Schema inspection — list all tables and their columns."""

from ..base import Skill


def _execute(conn, **kwargs):
    from nsys_ai.connection import DuckDBAdapter, wrap_connection
    adapter = wrap_connection(conn)

    if isinstance(adapter, DuckDBAdapter):
        cur = adapter.execute(
            """
            SELECT table_name,
                   column_name,
                   data_type AS column_type,
                   0 AS is_pk
            FROM information_schema.columns
            WHERE table_schema='main'
               OR table_schema='src'
            ORDER BY table_name, ordinal_position
            """
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    else:
        sql = """
        SELECT m.name AS table_name,
               p.name AS column_name,
               p.type AS column_type,
               p.pk AS is_pk
        FROM sqlite_master m
        JOIN pragma_table_info(m.name) p
        WHERE m.type = 'table'
          AND m.name NOT LIKE 'sqlite_%'
        ORDER BY m.name, p.cid
        """
        cur = adapter.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


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
    execute_fn=_execute,
    format_fn=_format,
    tags=["schema", "tables", "columns", "inspect", "meta", "utility"],
)
