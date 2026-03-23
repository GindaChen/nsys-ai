"""sql_compat.py — SQLite-to-DuckDB SQL dialect translation.

Provides a lightweight query rewriter to handle the few syntax differences
between SQLite and DuckDB.  Today this covers:
  - ``[identifier]`` bracket escaping → ``"identifier"`` double-quote escaping
    (SQLite accepts both; DuckDB only accepts double-quotes or backticks.)

DuckDB natively supports GLOB, LIKE, COALESCE, CAST(... AS INT) and all
window/aggregate functions used in the codebase, so no translation is needed
for those.
"""

from __future__ import annotations

import re

# Matches SQLite bracket-escaped identifiers like [end], [start], [value].
# Requires the identifier to start with a letter or underscore — this avoids
# collision with DuckDB array indexing (e.g. arr[0]) and numeric literals.
_BRACKET_ID_RE = re.compile(r"\[([A-Za-z_]\w*)\]")

# Matches hex integer literals like 0x1000000 which DuckDB doesn't support.
_HEX_LITERAL_RE = re.compile(r"\b0x([0-9a-fA-F]+)\b")

# Tokenizer to split out string literals and comments, so we don't accidentally
# rewrite [end] or 0x1A if they appear inside a string or comment.
_TOKENS_RE = re.compile(
    r"""(
        '(?:[^']|'')*'     | # Single-quoted string literals
        "(?:[^"]|"")*"     | # Double-quoted identifiers
        --[^\n]*           | # Single-line comments
        /\*.*?\*/            # Multi-line comments
    )""",
    re.VERBOSE | re.DOTALL,
)


def _hex_to_decimal(match: re.Match) -> str:
    """Convert a hex literal match to its decimal string equivalent."""
    return str(int(match.group(1), 16))


def sqlite_to_duckdb(sql: str) -> str:
    """Translate SQLite-specific SQL to DuckDB dialect.

    Currently rewrites:
      ``[end]``  →  ``"end"``
      ``[start]`` → ``"start"``
      (any ``[identifier]`` → ``"identifier"``)
      ``0x1000000`` → ``16777216`` (hex literals to decimal)
    """
    if not sql:
        return sql

    parts = _TOKENS_RE.split(sql)
    for i in range(0, len(parts), 2):
        part = parts[i]
        if not part:
            continue
        part = _BRACKET_ID_RE.sub(r'"\1"', part)
        part = _HEX_LITERAL_RE.sub(_hex_to_decimal, part)
        parts[i] = part
    return "".join(parts)
