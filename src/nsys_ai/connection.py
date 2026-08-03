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
# - SQLite: ``sqlite3.Connection`` is *not* weak-referenceable.  We key the
#   dict by the connection object itself (identity hash).  This keeps a strong
#   reference for the process lifetime, which is fine for the short-lived CLI
#   and avoids the ``id()``-reuse hazard (recycled IDs after GC give false hits).

_PROBE_MISS = object()
_duck_probe_bags: weakref.WeakKeyDictionary[Any, dict[str, Any]] = weakref.WeakKeyDictionary()
# sqlite3.Connection is not weak-referenceable, so we key by the connection
# object itself (identity hash).  This keeps a strong reference for the
# lifetime of the process.  Using the object directly avoids the id()-reuse
# hazard (id() can be recycled after a connection is closed and GC'd, giving
# false cache hits).
#
# Size note, since this bag now holds more than probe booleans: memoized skill
# rows live here too, measured at ~570KB for one connection after a full
# EvidenceBuilder build (one nvtx_layer_breakdown entry is ~477KB of it), and
# they are not released on close().  That is bounded per connection rather than
# growing: the CLI is short-lived, and the web server holds a single Profile for
# the server's lifetime rather than opening one per request.  Caching across a
# connection's life is safe because a profile is an immutable capture.
_sqlite_probe_bags: dict[sqlite3.Connection, dict[str, Any]] = {}

# A DuckDB cursor is a second handle on the same database, and a thread that
# queries through one must not lose the cache the owning connection filled.
# Keying the bag on the handle would give each worker thread its own empty bag
# and re-execute every memoized skill once per thread -- measured as four extra
# executions across four threads for a skill the owner had already run.
#
# Sharing the bag is safe because it stores *results*, not handles: the same
# skill with the same resolved parameters over the same immutable capture has
# one answer whichever handle asks. Two threads can still both miss and both
# compute, which costs duplicate work and never correctness.
#
# DuckDB exposes no parent pointer on a cursor, and duckdb_databases() names
# every in-memory database "memory", so neither can identify the owner. The
# registry is populated where cursors are created, in Profile.query_conn().
_handle_owner: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()


def register_derived_handle(handle, owner) -> None:
    """Record that ``handle`` is a second handle on ``owner``'s database.

    Both are held weakly, so neither outlives its natural lifetime.
    """
    if handle is owner:
        return
    try:
        _handle_owner[handle] = owner
    except TypeError:  # not weak-referenceable; fall back to a private bag
        pass


def _cache_owner(conn):
    """The object whose bag ``conn`` should read and write."""
    return _handle_owner.get(conn, conn)


def _probe_cache_get(conn, key: str):
    if isinstance(conn, sqlite3.Connection):
        bag = _sqlite_probe_bags.get(conn)
        if bag is None:
            return _PROBE_MISS
        return bag.get(key, _PROBE_MISS)
    bag = _duck_probe_bags.get(_cache_owner(conn))
    if bag is None:
        return _PROBE_MISS
    return bag.get(key, _PROBE_MISS)


def _probe_cache_set(conn, key: str, value) -> None:
    if isinstance(conn, sqlite3.Connection):
        _sqlite_probe_bags.setdefault(conn, {})[key] = value
        return
    try:
        _duck_probe_bags.setdefault(_cache_owner(conn), {})[key] = value
    except TypeError:
        pass


SKILL_CACHE_MISS = _PROBE_MISS

# Key under which both adapters memoize resolve_activity_tables().
#
# What makes this safe is *not* that the catalog stops changing — it does not.
# ensure_nvtx_kernel_map materializes nvtx_kernel_map and nvtx_path_dict onto a
# connection well after it is handed out.  The invariant is narrower: the names
# these prefixes match are all created before the connection is returned, and
# nothing afterwards adds or drops one.
#
# Two halves to that.  Every DuckDB connection comes from open_cached_db /
# open_parquetdir_db / open_direct_sqlite, each of which creates its parquet
# views and its alias views before returning — this matters on DuckDB
# specifically, where SHOW TABLES lists views too, so caching a pre-alias
# catalog would pin the versioned name (or, in direct mode where the real tables
# live under src., pin nothing at all).  And the DDL that does run later names
# lowercase helper tables, which no CUPTI_ACTIVITY_KIND_* / NVTX_EVENTS prefix
# matches.  Both halves are asserted in tests/test_activity_table_memoization.py;
# a new post-open table that did collide would break the second one.
_ACTIVITY_TABLES_KEY = "activity_tables"


def cached_skill_rows(conn, key: str):
    """Rows a skill already produced for ``key`` on ``conn``, or SKILL_CACHE_MISS.

    Reuses the probe bag above rather than adding a second cache, so both share
    its connection-object keying — deliberately not ``id()``, which is recycled
    after a connection is closed and would give false hits across profiles.
    """
    return _probe_cache_get(conn, key)


def set_cached_skill_rows(conn, key: str, rows) -> None:
    _probe_cache_set(conn, key, rows)


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
        cached = _probe_cache_get(self.conn, _ACTIVITY_TABLES_KEY)
        if cached is not _PROBE_MISS:
            return dict(cached)
        try:
            cur = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cur.fetchall()}
        except sqlite3.Error as exc:
            _log.debug("Failed to resolve activity tables (sqlite): %s", exc, exc_info=True)
            # Deliberately not cached: a failure here should not pin an empty
            # answer for the rest of the connection's life.
            return {}
        resolved = _find_activity_tables(tables)
        _probe_cache_set(self.conn, _ACTIVITY_TABLES_KEY, resolved)
        return dict(resolved)

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
        cached = _probe_cache_get(self.conn, _ACTIVITY_TABLES_KEY)
        if cached is not _PROBE_MISS:
            return dict(cached)
        try:
            tables = {row[0] for row in self.conn.execute("SHOW TABLES").fetchall()}
        except DB_ERRORS as exc:
            _log.debug("Failed to resolve activity tables (duckdb): %s", exc, exc_info=True)
            # Deliberately not cached: a failure here should not pin an empty
            # answer for the rest of the connection's life.
            return {}
        resolved = _find_activity_tables(tables)
        _probe_cache_set(self.conn, _ACTIVITY_TABLES_KEY, resolved)
        return dict(resolved)

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


_VERSION_SUFFIX_RE = re.compile(r"_V(\d+)")


def resolve_table_variant(
    tables: set[str] | frozenset[str],
    prefix: str,
    *,
    allow_other_suffixes: bool = False,
) -> str | None:
    """Pick which of ``prefix``'s name variants a profile actually carries.

    Nsight Systems documents only unversioned ``CUPTI_ACTIVITY_KIND_*`` names,
    but exports in the wild have carried ``_V2``/``_V3`` suffixes, so the choice
    has to be made somewhere. The order is:

    1. the exact ``prefix``, when present — an unversioned table is the name the
       exporter documents, so it wins outright;
    2. otherwise the highest ``_V<n>``, compared as an *integer* so ``_V10``
       beats ``_V3`` (lexicographic order gets that backwards);
    3. otherwise, only when ``allow_other_suffixes``, the alphabetically first
       remaining name that starts with ``prefix``, so an unforeseen future
       suffix still resolves to something rather than to nothing.

    The ``_V<n>`` match is anchored (``fullmatch``) rather than a bare
    ``startswith``: ``CUPTI_ACTIVITY_KIND_MEMCPY2`` is a real CUPTI activity
    kind (peer-to-peer memcpy) and must never be mistaken for a newer variant
    of ``CUPTI_ACTIVITY_KIND_MEMCPY``.
    """
    if prefix in tables:
        return prefix
    versioned: list[tuple[int, str]] = []
    others: list[str] = []
    for name in tables:
        if not name.startswith(prefix):
            continue
        match = _VERSION_SUFFIX_RE.fullmatch(name[len(prefix) :])
        if match:
            versioned.append((int(match.group(1)), name))
        elif allow_other_suffixes:
            others.append(name)
    if versioned:
        return max(versioned)[1]
    return sorted(others)[0] if others else None


def _find_activity_tables(tables: set[str]) -> dict[str, str]:
    def _find_by_prefix(prefix: str) -> str | None:
        return resolve_table_variant(tables, prefix, allow_other_suffixes=True)

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

    sync_table = _find_by_prefix("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION")
    if sync_table:
        resolved["sync"] = sync_table

    sync_type_table = _find_by_prefix("ENUM_CUPTI_SYNC_TYPE")
    if sync_type_table:
        resolved["sync_type"] = sync_type_table

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
