"""
base.py — Skill dataclass and execution helpers.

A Skill is the minimum analyzable unit: SQL template + parameters + formatter.
"""
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class SkillParam:
    """One parameter a skill accepts."""
    name: str
    description: str
    type: str = "str"          # str, int, float
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

    def execute(self, conn: sqlite3.Connection,
                **kwargs) -> list[dict]:
        """Run the skill's SQL against a connection.

        Args:
            conn: SQLite connection to an Nsight profile database
            **kwargs: Parameter values (substituted into SQL template)

        Returns:
            List of result rows as dicts
        """
        # Apply defaults
        resolved = {}
        for p in self.params:
            if p.name in kwargs:
                resolved[p.name] = kwargs[p.name]
            elif p.default is not None:
                resolved[p.name] = p.default
            elif p.required:
                raise ValueError(f"Skill '{self.name}' requires parameter '{p.name}'")

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
