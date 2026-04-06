"""
base.py — Skill dataclass and execution helpers.

A Skill is the minimum analyzable unit: SQL template + parameters + formatter.
"""

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

from ..connection import DB_ERRORS

_log = logging.getLogger(__name__)


def _compute_interval_union(intervals: list[tuple[int, int]]) -> int:
    """Computes the total non-overlapping duration of a list of [start, end] intervals."""
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[0])
    total_ns = 0
    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total_ns += current_end - current_start
            current_start, current_end = start, end

    total_ns += current_end - current_start
    return total_ns

# Track connections that have already been indexed to avoid repeated work.
_indexed_connections: set[int] = set()

# Indexes to create on Nsight SQLite profiles for skill query performance.
# Uses ``_nsysai_`` prefix to avoid conflicts with upstream tables.
# Table names can vary between Nsight versions (e.g. *_KERNEL_V2/V3), so we
# resolve the actual table names from sqlite_master at runtime instead of
# hard-coding them here.


def ensure_indexes(conn: sqlite3.Connection) -> None:
    """Create performance indexes on the profile DB if they don't already exist.

    Delegates to the centralized :func:`~nsys_ai.indexing.ensure_performance_indexes`.
    """
    from ..indexing import ensure_performance_indexes

    ensure_performance_indexes(conn)


@dataclass
class SkillParam:
    """One parameter a skill accepts."""

    name: str
    description: str
    type: str = "str"  # str, int, float
    required: bool = False
    default: object = None


@dataclass
class Skill:
    """A self-contained GPU profile analysis skill.

    Attributes:
        name:        Short identifier (e.g. "top_kernels")
        title:       Human-readable title
        description: What this skill analyzes and why
        category:    One of: kernels, memory, nvtx, communication, system, utility
        sql:         SQL query template with {param} placeholders
        params:      Accepted parameters
        format_fn:   Optional function(rows) → formatted string
        tags:        Search tags for skill discovery
        execute_fn:  Optional Python callable(conn, **kwargs) → list[dict].
                     When set, used instead of sql for execution.
    """

    name: str
    title: str
    description: str
    category: str
    sql: str = ""
    params: list[SkillParam] = field(default_factory=list)
    format_fn: Callable | None = None
    tags: list[str] = field(default_factory=list)
    execute_fn: Callable | None = None
    to_findings_fn: Callable | None = None

    def execute(self, conn: sqlite3.Connection, **kwargs) -> list[dict]:
        """Run the skill against a connection.

        If ``execute_fn`` is set, delegates to it.  Otherwise runs the
        skill's SQL query against *conn*.

        Args:
            conn: SQLite connection to an Nsight profile database
            **kwargs: Parameter values (substituted into SQL template).
                      Special keys ``trim_start_ns`` and ``trim_end_ns``
                      trigger ``{trim_clause}`` substitution if present
                      in the SQL template.

        Returns:
            List of result rows as dicts
        """
        # Auto-create performance indexes (one-time per connection).
        ensure_indexes(conn)

        from ..connection import wrap_connection
        adapter = wrap_connection(conn)

        # Apply parameter defaults and required checks for all skill types.
        # Start from the provided kwargs so we preserve any extra arguments.
        resolved: dict[str, object] = dict(kwargs)
        for p in self.params:
            if p.name in resolved:
                # Caller-supplied value wins over default.
                continue
            if p.default is not None:
                resolved[p.name] = p.default
            elif p.required:
                raise ValueError(f"Skill '{self.name}' requires parameter '{p.name}'")

        # Handle {trim_clause} injection before execute_fn so {overhead_ns} can be computed correctly
        trim_start = resolved.get("trim_start_ns")
        trim_end = resolved.get("trim_end_ns")
        if trim_start is not None and trim_end is not None and "{trim_clause}" in self.sql:
            resolved["trim_clause"] = (
                f"AND k.start >= {int(trim_start)} AND k.[end] <= {int(trim_end)}"
            )
        elif "{trim_clause}" in self.sql:
            # No trim requested — replace with empty string
            resolved["trim_clause"] = ""

        # Inject resolved activity table names for versioned-table support.
        # SQL templates use {kernel_table} etc. instead of hardcoding
        # CUPTI_ACTIVITY_KIND_KERNEL which may be _KERNEL_V2/_V3 in
        # newer Nsight Systems versions.
        tables = adapter.resolve_activity_tables()
        resolved.setdefault(
            "kernel_table",
            tables.get("kernel", "CUPTI_ACTIVITY_KIND_KERNEL"),
        )
        resolved.setdefault(
            "runtime_table",
            tables.get("runtime", "CUPTI_ACTIVITY_KIND_RUNTIME"),
        )
        resolved.setdefault(
            "nvtx_table",
            tables.get("nvtx", "NVTX_EVENTS"),
        )
        resolved.setdefault(
            "memcpy_table",
            tables.get("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY"),
        )
        resolved.setdefault(
            "memset_table",
            tables.get("memset", "CUPTI_ACTIVITY_KIND_MEMSET"),
        )

        # Compute profiler overhead union duration dynamically.
        # Try the SELECT directly and catch DB_ERRORS if the table is absent,
        # avoiding a redundant catalog scan (resolve_activity_tables already did one).
        overhead_ns = 0
        for oh_table in ("profiler_overhead", "PROFILER_OVERHEAD"):
            try:
                conds = []
                params: list[object] = []
                if trim_start is not None:
                    conds.append("[end] >= ?")
                    params.append(int(trim_start))
                if trim_end is not None:
                    conds.append("start <= ?")
                    params.append(int(trim_end))

                where_clause = "WHERE " + " AND ".join(conds) if conds else ""
                cur = adapter.execute(
                    f"SELECT start, [end] FROM {oh_table} {where_clause}",
                    params,
                )
                rows = cur.fetchall()
                if rows:
                    intervals = []
                    for row in rows:
                        s, e = int(row[0]), int(row[1])
                        if trim_start is not None:
                            s = max(s, int(trim_start))
                        if trim_end is not None:
                            e = min(e, int(trim_end))
                        if s < e:
                            intervals.append((s, e))
                    overhead_ns = _compute_interval_union(intervals)
                break  # Table found and queried successfully
            except DB_ERRORS:
                continue  # Table doesn't exist under this name, try next
        if overhead_ns == 0:
            _log.debug("No profiler overhead data found (table absent or empty)")

        resolved["overhead_ns"] = overhead_ns

        # Python-level skill: delegate to execute_fn with resolved params.
        if self.execute_fn is not None:
            return self.execute_fn(conn, **resolved)

        # --- SQL Execution Path ---
        # NVTX text resolution: handle both legacy (text column only)
        # and modern schemas (textId → StringIds lookup).
        if "{nvtx_text_expr}" in self.sql or "{nvtx_text_join}" in self.sql:
            has_textid = adapter.detect_nvtx_text_id()
            if has_textid:
                resolved.setdefault("nvtx_text_expr", "COALESCE(n.text, s2.value)")
                resolved.setdefault("nvtx_text_join", "LEFT JOIN StringIds s2 ON n.textId = s2.id")
            else:
                resolved.setdefault("nvtx_text_expr", "n.text")
                resolved.setdefault("nvtx_text_join", "")

        sql = self.sql.format(**resolved) if resolved else self.sql
        try:
            cursor = adapter.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as exc:
            db_errors = (sqlite3.Error,)
            try:
                import duckdb

                db_errors += (duckdb.Error,)
            except ImportError:
                pass

            if not isinstance(exc, db_errors):
                raise

            from nsys_ai.exceptions import SkillExecutionError

            raise SkillExecutionError(f"SQL failed: {exc}", skill_name=self.name) from exc

    def format_rows(self, rows: list[dict]) -> str:
        """Format pre-computed rows as text (no re-execution)."""
        if self.format_fn:
            return self.format_fn(rows)
        return _default_format(self, rows)

    def run(self, conn: sqlite3.Connection, **kwargs) -> str:
        """Execute and format results as text."""
        return self.format_rows(self.execute(conn, **kwargs))

    def to_tool_description(self) -> str:
        """Return a one-paragraph description suitable for an LLM tool catalog."""
        params_desc = ""
        if self.params:
            params_desc = " Parameters: " + ", ".join(
                f"{p.name} ({p.type}, {'required' if p.required else 'optional'})"
                for p in self.params
            )
        return f"[{self.name}] {self.title}: {self.description}{params_desc}"


def _default_format(skill: Skill, rows: list[dict]) -> str:
    """Simple tabular format for skill results."""
    if not rows:
        return f"({skill.title}: no results)"

    cols = list(rows[0].keys())
    # Compute column widths
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("─" * widths[c] for c in cols)
    lines = [f"── {skill.title} ──", header, sep]
    for row in rows:
        lines.append("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))
    return "\n".join(lines)
