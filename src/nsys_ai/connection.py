import logging
import re
import sqlite3
import weakref
from typing import Any, Protocol, runtime_checkable

# Narrow exception tuple for diagnostic query blocks.
DB_ERRORS: tuple[type[Exception], ...] = (sqlite3.Error, sqlite3.OperationalError)
try:
    import duckdb  # noqa: E402

    DB_ERRORS = DB_ERRORS + (duckdb.Error,)
except ImportError:
    pass

_log = logging.getLogger(__name__)

# Defence-in-depth: reject identifiers with special chars even though
# callers should only pass schema-derived names.
_SAFE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_safe_identifier(name: str) -> bool:
    """Check if the string is safe to be structurally spliced into a SQL query."""
    return bool(_SAFE_IDENT_RE.match(name))


# ── Per-connection probe cache ───────────────────────────────────────────
# - DuckDB: ``DuckDBPyConnection`` rejects arbitrary ``setattr`` but is
#   weak-referenceable → ``WeakKeyDictionary`` drops entries when the DB closes.
# - SQLite: ``sqlite3.Connection`` is *not* weak-referenceable → fall back to
#   ``id(conn)`` map (small; typical CLI uses one short-lived connection).

_PROBE_MISS = object()
_duck_probe_bags: weakref.WeakKeyDictionary[Any, dict[str, Any]] = weakref.WeakKeyDictionary()
_sqlite_probe_bags: dict[int, dict[str, Any]] = {}


def _probe_cache_get(conn, key: str):
    if isinstance(conn, sqlite3.Connection):
        bag = _sqlite_probe_bags.get(id(conn))
        if bag is None:
            return _PROBE_MISS
        return bag.get(key, _PROBE_MISS)
    bag = _duck_probe_bags.get(conn)
    if bag is None:
        return _PROBE_MISS
    return bag.get(key, _PROBE_MISS)


def _probe_cache_set(conn, key: str, value) -> None:
    if isinstance(conn, sqlite3.Connection):
        _sqlite_probe_bags.setdefault(id(conn), {})[key] = value
        return
    try:
        _duck_probe_bags.setdefault(conn, {})[key] = value
    except TypeError:
        pass


def cached_nvtx_map_uses_path_id(conn) -> bool:
    """True when ``nvtx_kernel_map`` + ``nvtx_path_dict`` expose ``path_id``."""
    cached = _probe_cache_get(conn, "nvtx_path_id")
    if cached is not _PROBE_MISS:
        return cached
    try:
        conn.execute("SELECT path_id FROM nvtx_kernel_map LIMIT 0")
        conn.execute("SELECT path_id FROM nvtx_path_dict LIMIT 0")
        ok = True
    except DB_ERRORS:
        ok = False
    _probe_cache_set(conn, "nvtx_path_id", ok)
    return ok


def cached_nvtx_map_has_embedded_tc(conn) -> bool:
    """True when ``nvtx_kernel_map`` includes Tensor Core columns (``is_tc_eligible``, ``uses_tc``)."""
    cached = _probe_cache_get(conn, "nvtx_embedded_tc")
    if cached is not _PROBE_MISS:
        return cached
    try:
        conn.execute("SELECT is_tc_eligible, uses_tc FROM nvtx_kernel_map LIMIT 0")
        ok = True
    except DB_ERRORS:
        ok = False
    _probe_cache_set(conn, "nvtx_embedded_tc", ok)
    return ok


@runtime_checkable
class ConnectionAdapter(Protocol):
    """Abstracts SQLite and DuckDB connections for cross-engine compatibility."""

    @property
    def raw_conn(self) -> Any: ...

    def execute(self, sql: str, parameters: tuple | list = ()) -> Any: ...

    def resolve_activity_tables(self) -> dict[str, str]: ...

    def detect_nvtx_text_id(self) -> bool: ...

    def get_table_columns(self, table_name: str) -> list[str]: ...

    def get_table_names(self) -> set[str]: ...


class SQLiteAdapter:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    @property
    def raw_conn(self) -> sqlite3.Connection:
        return self.conn

    def execute(self, sql: str, parameters: tuple | list = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, parameters)

    def resolve_activity_tables(self) -> dict[str, str]:
        try:
            cur = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cur.fetchall()}
        except sqlite3.Error as exc:
            _log.debug("Failed to resolve activity tables (sqlite): %s", exc, exc_info=True)
            return {}
        return _find_activity_tables(tables)

    def detect_nvtx_text_id(self) -> bool:
        try:
            nvtx_table = self.resolve_activity_tables().get("nvtx", "NVTX_EVENTS")
            cols = self.get_table_columns(nvtx_table)
            return "textId" in cols
        except (sqlite3.Error, ValueError) as exc:
            _log.debug("NVTX textId detection failed (sqlite): %s", exc, exc_info=True)
            return False

    def get_table_columns(self, table_name: str) -> list[str]:
        if not _SAFE_IDENT_RE.match(table_name):
            raise ValueError(f"Unsafe table name: {table_name!r}")
        cur = self.conn.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cur.fetchall()]

    def get_table_names(self) -> set[str]:
        try:
            cur = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            return {row[0] for row in cur.fetchall()}
        except sqlite3.Error:
            return set()


class DuckDBAdapter:
    def __init__(self, conn):
        self.conn = conn

    @property
    def raw_conn(self) -> Any:
        return self.conn

    def execute(self, sql: str, parameters: tuple | list = ()) -> Any:
        from .sql_compat import sqlite_to_duckdb

        rewritten_sql = sqlite_to_duckdb(sql)
        # duckdb parameters need to be passed directly to connection execute
        return self.conn.execute(rewritten_sql, list(parameters) if parameters else [])

    def resolve_activity_tables(self) -> dict[str, str]:
        try:
            tables = {row[0] for row in self.conn.execute("SHOW TABLES").fetchall()}
        except DB_ERRORS as exc:
            _log.debug("Failed to resolve activity tables (duckdb): %s", exc, exc_info=True)
            return {}
        return _find_activity_tables(tables)

    def detect_nvtx_text_id(self) -> bool:
        try:
            nvtx_table = self.resolve_activity_tables().get("nvtx")
            if not nvtx_table:
                return False

            cols = self.get_table_columns(nvtx_table)
            return "textId" in cols
        except DB_ERRORS + (ValueError,) as exc:
            _log.debug("NVTX textId detection failed (duckdb): %s", exc, exc_info=True)
            return False

    def get_table_columns(self, table_name: str) -> list[str]:
        if not is_safe_identifier(table_name):
            raise ValueError(f"Unsafe table name: {table_name!r}")
        cur = self.conn.execute(f"DESCRIBE {table_name}")
        return [row[0] for row in cur.fetchall()]

    def get_table_names(self) -> set[str]:
        try:
            return {row[0] for row in self.conn.execute("SHOW TABLES").fetchall()}
        except DB_ERRORS as exc:
            _log.debug("Failed to get table names (duckdb): %s", exc, exc_info=True)
            return set()


def _find_activity_tables(tables: set[str]) -> dict[str, str]:
    def _find_by_prefix(prefix: str) -> str | None:
        if prefix in tables:
            return prefix
        candidates = sorted(t for t in tables if t.startswith(prefix))
        return candidates[0] if candidates else None

    resolved: dict[str, str] = {}

    kernel_table = _find_by_prefix("CUPTI_ACTIVITY_KIND_KERNEL")
    if kernel_table:
        resolved["kernel"] = kernel_table

    runtime_table = _find_by_prefix("CUPTI_ACTIVITY_KIND_RUNTIME")
    if runtime_table:
        resolved["runtime"] = runtime_table

    memcpy_table = _find_by_prefix("CUPTI_ACTIVITY_KIND_MEMCPY")
    if memcpy_table:
        resolved["memcpy"] = memcpy_table

    memset_table = _find_by_prefix("CUPTI_ACTIVITY_KIND_MEMSET")
    if memset_table:
        resolved["memset"] = memset_table

    if "NVTX_EVENTS" in tables:
        nvtx_table = "NVTX_EVENTS"
    else:
        nvtx_table = _find_by_prefix("NVTX_EVENTS")
    if nvtx_table:
        resolved["nvtx"] = nvtx_table

    return resolved


def wrap_connection(conn) -> ConnectionAdapter:
    """Wraps a raw connection in the appropriate adapter."""
    if isinstance(conn, ConnectionAdapter):
        return conn

    # Attempt duckdb detection without forced import
    is_duckdb = False
    try:
        import duckdb

        if isinstance(conn, duckdb.DuckDBPyConnection):
            is_duckdb = True
    except ImportError:
        pass

    if is_duckdb:
        return DuckDBAdapter(conn)
    return SQLiteAdapter(conn)
