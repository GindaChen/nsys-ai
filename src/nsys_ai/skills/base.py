"""
base.py — Skill dataclass and execution helpers.

A Skill is the minimum analyzable unit: SQL template + parameters + formatter.
"""

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)

# Track connections that have already been indexed to avoid repeated work.
_indexed_connections: set[int] = set()

# Indexes to create on Nsight SQLite profiles for skill query performance.
# Uses ``_nsysai_`` prefix to avoid conflicts with upstream tables.
# Table names can vary between Nsight versions (e.g. *_KERNEL_V2/V3), so we
# resolve the actual table names from sqlite_master at runtime instead of
# hard-coding them here.


def _resolve_activity_tables(conn: sqlite3.Connection) -> dict[str, str]:
    """Resolve Nsight activity table names (kernel/runtime/NVTX) from sqlite_master.

    Nsight may emit versioned table names such as
    CUPTI_ACTIVITY_KIND_KERNEL_V2.
    This helper finds the first matching table for each logical kind.
    """
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    except Exception:
        return {}

    def _find_by_prefix(prefix: str) -> str | None:
        if prefix in tables:
            return prefix
        candidates = sorted(t for t in tables if t.startswith(prefix))
        return candidates[0] if candidates else None

    kernel_table = _find_by_prefix("CUPTI_ACTIVITY_KIND_KERNEL")
    runtime_table = _find_by_prefix("CUPTI_ACTIVITY_KIND_RUNTIME")
    memcpy_table = _find_by_prefix("CUPTI_ACTIVITY_KIND_MEMCPY")
    if "NVTX_EVENTS" in tables:
        nvtx_table = "NVTX_EVENTS"
    else:
        nvtx_table = _find_by_prefix("NVTX_EVENTS")

    resolved: dict[str, str] = {}
    if kernel_table:
        resolved["kernel"] = kernel_table
    if runtime_table:
        resolved["runtime"] = runtime_table
    if memcpy_table:
        resolved["memcpy"] = memcpy_table
    if nvtx_table:
        resolved["nvtx"] = nvtx_table

    return resolved


def ensure_indexes(conn: sqlite3.Connection) -> None:
    """Create performance indexes on the profile DB if they don't already exist.

    This is safe to call repeatedly — indexes are ``CREATE IF NOT EXISTS`` and
    the function tracks which connections have been processed.  Each index
    creation is wrapped in try/except so missing tables (common for profiles
    without NVTX or NCCL data) don't block the rest.
    """
    conn_id = id(conn)
    if conn_id in _indexed_connections:
        return

    tables = _resolve_activity_tables(conn)

    index_stmts: list[str] = []

    kernel_table = tables.get("kernel")
    if kernel_table:
        index_stmts.append(
            f"CREATE INDEX IF NOT EXISTS _nsysai_kernel_start ON {kernel_table}(start)"
        )
        index_stmts.append(
            f"CREATE INDEX IF NOT EXISTS _nsysai_kernel_corr  ON {kernel_table}(correlationId)"
        )

    runtime_table = tables.get("runtime")
    if runtime_table:
        index_stmts.append(
            f"CREATE INDEX IF NOT EXISTS _nsysai_runtime_corr ON {runtime_table}(correlationId)"
        )
        index_stmts.append(
            f"CREATE INDEX IF NOT EXISTS _nsysai_runtime_tid  ON {runtime_table}(globalTid, start)"
        )

    nvtx_table = tables.get("nvtx")
    if nvtx_table:
        index_stmts.append(
            f"CREATE INDEX IF NOT EXISTS _nsysai_nvtx_start   ON {nvtx_table}(start)"
        )
        index_stmts.append(
            f"CREATE INDEX IF NOT EXISTS _nsysai_nvtx_tid     ON {nvtx_table}(globalTid, start)"
        )
        # Compound index for NVTX join queries (nvtx_layer_breakdown, nvtx_kernel_map)
        index_stmts.append(
            f"CREATE INDEX IF NOT EXISTS _nsysai_nvtx_range   ON {nvtx_table}(globalTid, start, [end])"
        )

    # Streamwise kernel index for window-function skills
    # (gpu_idle_gaps, kernel_launch_pattern)
    if kernel_table:
        index_stmts.append(
            f"CREATE INDEX IF NOT EXISTS _nsysai_kernel_stream ON {kernel_table}(streamId, start)"
        )

    for stmt in index_stmts:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            # Table doesn't exist in this profile — skip silently.
            pass
        except Exception as exc:
            # Read-only filesystem, locked DB, etc.
            _log.debug("ensure_indexes: %s — %s", stmt.split("ON")[0].strip(), exc)

    try:
        conn.commit()
    except Exception:
        pass

    _indexed_connections.add(conn_id)


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
                raise ValueError(
                    f"Skill '{self.name}' requires parameter '{p.name}'"
                )

        # Python-level skill: delegate to execute_fn with resolved params.
        if self.execute_fn is not None:
            return self.execute_fn(conn, **resolved)

        # SQL skill: further augment resolved params and run query
        # Handle {trim_clause} injection
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
        tables = _resolve_activity_tables(conn)
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

        # NVTX text resolution: handle both legacy (text column only)
        # and modern schemas (textId → StringIds lookup).
        nvtx_tbl = resolved.get("nvtx_table", "NVTX_EVENTS")
        if "{nvtx_text_expr}" in self.sql or "{nvtx_text_join}" in self.sql:
            try:
                has_textid = conn.execute(
                    f"SELECT COUNT(*) FROM pragma_table_info('{nvtx_tbl}') WHERE name='textId'"
                ).fetchone()[0] > 0
            except Exception:
                has_textid = False
            if has_textid:
                resolved.setdefault("nvtx_text_expr", "COALESCE(n.text, s2.value)")
                resolved.setdefault("nvtx_text_join", "LEFT JOIN StringIds s2 ON n.textId = s2.id")
            else:
                resolved.setdefault("nvtx_text_expr", "n.text")
                resolved.setdefault("nvtx_text_join", "")

        sql = self.sql.format(**resolved) if resolved else self.sql
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

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
