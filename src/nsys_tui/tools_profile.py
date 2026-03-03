"""
tools_profile.py - Safe read-only SQL tool and schema for the Text-to-SQL agent (v2).

Exposes query_profile_db(conn, sql_query) and get_profile_schema(conn) for use by
chat.py and test_agent.py. See docs/plan-brain-navigator.md
"""
import json
import logging
import re
import sqlite3
import threading

_log = logging.getLogger(__name__)

# Default max rows to prevent token explosion.
DEFAULT_MAX_LIMIT = 50
# Default maximum JSON length (characters) returned to the model.
DEFAULT_MAX_JSON_CHARS = 8000

# NVTX table name is stable; kernel table is detected by NsightSchema.
NVTX_TABLE = "NVTX_EVENTS"
# StringIds maps id -> value for kernel names (shortName, demangledName reference it).
STRING_IDS_TABLE = "StringIds"

# Regex: reject any mutating SQL.
_READ_ONLY_BLOCK = re.compile(
    r"(?i)\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE)\b"
)


def _adaptive_limit(sql_upper: str, base_limit: int) -> int:
    """
    Return a LIMIT adjusted by the projected column count.

    Narrow queries (1 column, e.g. SELECT name) get up to 2x the base limit;
    wide queries (4+ columns) get half the base limit (min 20) to keep JSON small.
    """
    select_idx = sql_upper.find("SELECT")
    if select_idx < 0:
        return base_limit
    from_idx = sql_upper.find(" FROM ", select_idx + 6)
    end = from_idx if from_idx > 0 else len(sql_upper)
    projection = sql_upper[select_idx + 6:end].strip()
    # Count top-level comma separators (commas inside function args don't split columns).
    depth, col_count = 0, 1
    for ch in projection:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            col_count += 1
    if col_count <= 1:
        return min(100, base_limit * 2)
    if col_count <= 3:
        return base_limit
    return max(20, base_limit // 2)


def query_profile_db(
    conn: sqlite3.Connection,
    sql_query: str,
    *,
    max_limit: int = DEFAULT_MAX_LIMIT,
) -> str:
    """
    Execute a read-only SELECT on the profile DB and return rows as a JSON string.

    - Guardrail 1: Rejects any query containing INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/etc.
    - Guardrail 2: If no LIMIT or LIMIT > max_limit, enforces LIMIT max_limit.
    - Guardrail 3: Rejects broad SELECT * queries to avoid token explosion.
    - Returns: "[{\"col\": \"val\"}, ...]" or an error string for the model.
    """
    if not (sql_query or "").strip():
        return "Error: Empty query."

    if _READ_ONLY_BLOCK.search(sql_query):
        return "Error: Read-only. Only SELECT queries are allowed."

    q = sql_query.strip().rstrip(";")
    upper = q.upper()

    # Reject broad SELECT * to keep responses small and focused.
    # This is conservative and applies to all tables; the model should learn
    # to select only the columns it needs instead of dumping entire rows.
    star_match = re.search(r"\bSELECT\s+\*", upper)
    if star_match:
        return (
            "Error: SELECT * is not allowed - it returns too many columns and wastes tokens. "
            "Please select only the columns you need, for example: "
            "SELECT start, [end], shortName FROM <table> WHERE ... LIMIT 20")

    # Adaptive LIMIT: narrow projections allow more rows; wide ones get fewer.
    effective_limit = _adaptive_limit(upper, max_limit)

    # Enforce LIMIT: if no LIMIT append; if LIMIT > effective_limit rewrite.
    limit_match = re.search(r"\bLIMIT\s+(\d+)", upper, re.IGNORECASE)
    if limit_match:
        n = int(limit_match.group(1))
        if n > effective_limit:
            q = re.sub(r"\bLIMIT\s+" + str(n) + r"\b", f"LIMIT {effective_limit}", q, count=1, flags=re.IGNORECASE)
    else:
        q = q + f" LIMIT {effective_limit}"

    try:
        cur = conn.execute(q)
        rows = cur.fetchall()
        if conn.row_factory is sqlite3.Row:
            out = [dict(r) for r in rows]
        else:
            names = [d[0] for d in cur.description] if cur.description else []
            out = [dict(zip(names, r)) for r in rows]
        # JSON-serializable values (sqlite3 can return bytes etc.)
        def _serialize(obj):
            if isinstance(obj, (bytes, bytearray)):
                return obj.decode("utf-8", errors="replace")
            if isinstance(obj, (int, float, str, type(None))):
                return obj
            return str(obj)

        for row in out:
            for k, v in list(row.items()):
                row[k] = _serialize(v)
        json_str = json.dumps(out, ensure_ascii=False)
        if len(json_str) > DEFAULT_MAX_JSON_CHARS:
            _log.info(
                "query_profile_db: result truncated (size=%d, max=%d)",
                len(json_str),
                DEFAULT_MAX_JSON_CHARS,
            )
            json_str = json_str[: DEFAULT_MAX_JSON_CHARS] + (
                f"...[Truncated - result exceeds {DEFAULT_MAX_JSON_CHARS} chars] "
                "Please refine your query: SELECT only essential columns "
                "(e.g. start, [end], shortName) or reduce the LIMIT."
            )
        return json_str
    except sqlite3.Error as e:
        return f"Error: Database error: {e}"


def get_profile_schema(
    conn: sqlite3.Connection,
    table_names: tuple[str, ...] | None = None,
) -> str:
    """
    Return a short schema description for injection into the system prompt.

    Uses NsightSchema to get the kernel table name, then fetches DDL from
    sqlite_master for that table and NVTX_EVENTS (if present).
    """
    try:
        from .profile import NsightSchema
        ns = NsightSchema(conn)
        kernel_table = ns.kernel_table
    except Exception:
        kernel_table = None

    want = list(table_names) if table_names else []
    if kernel_table and kernel_table not in want:
        want.append(kernel_table)
    if NVTX_TABLE not in want:
        want.append(NVTX_TABLE)
    if STRING_IDS_TABLE not in want:
        want.append(STRING_IDS_TABLE)

    if not want:
        return "(No tables specified.)"

    placeholders = ",".join("?" * len(want))
    try:
        cur = conn.execute(
            f"SELECT name, sql FROM sqlite_master WHERE type='table' AND name IN ({placeholders})",
            want,
        )
        rows = cur.fetchall()
    except sqlite3.Error:
        return "(Could not read schema.)"

    parts = []
    for name, ddl in rows:
        if ddl:
            parts.append(ddl.strip())
    return "\n\n".join(parts) if parts else "(Empty schema.)"


# Optional schema cache by profile path. Max entries to avoid unbounded growth.
_SCHEMA_CACHE_MAX = 16
_schema_cache: dict[str, str] = {}
_schema_cache_lock = threading.Lock()


def get_profile_schema_cached(conn: sqlite3.Connection, path: str | None = None) -> str:
    """
    Return schema for the given connection, using a cache keyed by path when provided.
    If path is None, always calls get_profile_schema(conn). Call with path when the
    same profile may be used repeatedly (e.g. Web/TUI per-session). Thread-safe.
    """
    if path is None:
        return get_profile_schema(conn)
    with _schema_cache_lock:
        if path in _schema_cache:
            return _schema_cache[path]
        schema = get_profile_schema(conn)
        if len(_schema_cache) >= _SCHEMA_CACHE_MAX:
            first = next(iter(_schema_cache))
            del _schema_cache[first]
        _schema_cache[path] = schema
    return schema


def open_profile_readonly(path: str) -> sqlite3.Connection:
    """Open a profile SQLite file in read-only mode (URI mode=ro)."""
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def open_profile_readonly_for_worker(path: str) -> sqlite3.Connection:
    """
    Open a read-only profile connection for use from a worker thread.

    Each worker (e.g. TUI chat worker) must open and close its own connection;
    do not share a connection across threads. This helper is equivalent to
    open_profile_readonly(path); use it when calling from a background thread
    to make the pattern explicit. stream_agent_loop opens the connection
    inside the same thread that runs the loop (worker or request handler).
    """
    return open_profile_readonly(path)


# OpenAI-style tool definition for query_profile_db.
TOOL_QUERY_PROFILE_DB = {
    "type": "function",
    "function": {
        "name": "query_profile_db",
        "description": (
            "Execute a read-only SELECT query on the Nsight Systems profile SQLite database. "
            "Use this to answer whole-profile questions (e.g. first kernel, slowest kernel, counts). "
            "Only SELECT is allowed. LIMIT is enforced (max 50 rows) if missing or too large. "
            "Use the table and column names from the schema provided in the system prompt."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "A single SELECT SQL statement (no INSERT/UPDATE/DELETE/DROP).",
                },
            },
            "required": ["sql_query"],
        },
    },
}
