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
_INDEX_STMTS = [
    "CREATE INDEX IF NOT EXISTS _nsysai_kernel_start ON CUPTI_ACTIVITY_KIND_KERNEL(start)",
    "CREATE INDEX IF NOT EXISTS _nsysai_kernel_corr  ON CUPTI_ACTIVITY_KIND_KERNEL(correlationId)",
    "CREATE INDEX IF NOT EXISTS _nsysai_runtime_corr ON CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)",
    "CREATE INDEX IF NOT EXISTS _nsysai_runtime_tid  ON CUPTI_ACTIVITY_KIND_RUNTIME(globalTid, start)",
    "CREATE INDEX IF NOT EXISTS _nsysai_nvtx_start   ON NVTX_EVENTS(start)",
    "CREATE INDEX IF NOT EXISTS _nsysai_nvtx_tid     ON NVTX_EVENTS(globalTid, start)",
]


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

    for stmt in _INDEX_STMTS:
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
    """

    name: str
    title: str
    description: str
    category: str
    sql: str
    params: list[SkillParam] = field(default_factory=list)
    format_fn: Callable | None = None
    tags: list[str] = field(default_factory=list)

    def execute(self, conn: sqlite3.Connection, **kwargs) -> list[dict]:
        """Run the skill's SQL against a connection.

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

        # Apply defaults
        resolved = {}
        for p in self.params:
            if p.name in kwargs:
                resolved[p.name] = kwargs[p.name]
            elif p.default is not None:
                resolved[p.name] = p.default
            elif p.required:
                raise ValueError(f"Skill '{self.name}' requires parameter '{p.name}'")

        # Handle {trim_clause} injection
        trim_start = kwargs.get("trim_start_ns")
        trim_end = kwargs.get("trim_end_ns")
        if trim_start is not None and trim_end is not None and "{trim_clause}" in self.sql:
            resolved["trim_clause"] = (
                f"AND k.start >= {int(trim_start)} AND k.[end] <= {int(trim_end)}"
            )
        elif "{trim_clause}" in self.sql:
            # No trim requested — replace with empty string
            resolved["trim_clause"] = ""

        sql = self.sql.format(**resolved) if resolved else self.sql
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def run(self, conn: sqlite3.Connection, **kwargs) -> str:
        """Execute and format results as text."""
        rows = self.execute(conn, **kwargs)
        if self.format_fn:
            return self.format_fn(rows)
        return _default_format(self, rows)

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
