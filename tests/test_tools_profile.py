"""Tests for tools_profile (query_profile_db guardrails, get_profile_schema)."""

import json
import sqlite3


def test_query_profile_db_readonly_guardrail():
    """query_profile_db rejects INSERT/UPDATE/DELETE/DROP/ALTER/CREATE."""
    from nsys_ai.tools_profile import query_profile_db

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t(id INT)")
    for bad in (
        "INSERT INTO t VALUES(1)",
        "UPDATE t SET id=1",
        "DELETE FROM t",
        "DROP TABLE t",
        "ALTER TABLE t RENAME TO x",
        "CREATE TABLE x(y INT)",
    ):
        out = query_profile_db(conn, bad)
        assert "Error" in out
    conn.close()


def test_query_profile_db_limit_enforced():
    """query_profile_db enforces LIMIT; adaptive limit applies based on column count."""
    from nsys_ai.tools_profile import DEFAULT_MAX_LIMIT, query_profile_db

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t(id INT, name TEXT, val REAL, extra TEXT)")
    conn.executemany(
        "INSERT INTO t VALUES(?,?,?,?)", [(i, f"k{i}", float(i), "x") for i in range(200)]
    )
    # Multi-column (4 cols → adaptive limit = max(20, 50//2) = 25): never return all 200 rows.
    out = query_profile_db(conn, "SELECT id, name, val, extra FROM t")
    rows = json.loads(out)
    assert len(rows) <= max(20, DEFAULT_MAX_LIMIT // 2)
    # Explicit LIMIT 200 on multi-column query is capped to adaptive limit.
    out2 = query_profile_db(conn, "SELECT id, name, val, extra FROM t LIMIT 200")
    rows2 = json.loads(out2)
    assert len(rows2) <= max(20, DEFAULT_MAX_LIMIT // 2)
    conn.close()


def test_query_profile_db_rejects_select_star():
    """query_profile_db rejects SELECT * with a clear error (§11.7.3, §11.8.4 Stage 2)."""
    from nsys_ai.tools_profile import query_profile_db

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t(a INT, b INT)")
    out = query_profile_db(conn, "SELECT * FROM t LIMIT 10")
    assert "Error" in out
    assert "SELECT *" in out or "select only the columns" in out.lower()
    conn.close()


def test_query_profile_db_valid_select():
    """query_profile_db returns JSON array of rows for valid SELECT."""
    from nsys_ai.tools_profile import query_profile_db

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE k(name TEXT, start INT)")
    conn.execute("INSERT INTO k VALUES('axpy', 100)")
    out = query_profile_db(conn, "SELECT name, start FROM k LIMIT 10")
    rows = json.loads(out)
    assert rows == [{"name": "axpy", "start": 100}]
    conn.close()


def test_get_profile_schema_in_memory():
    """get_profile_schema returns DDL for requested tables."""
    from nsys_ai.tools_profile import get_profile_schema

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INT, name TEXT)")
    conn.execute("CREATE TABLE NVTX_EVENTS(text TEXT)")
    schema = get_profile_schema(conn, table_names=("CUPTI_ACTIVITY_KIND_KERNEL", "NVTX_EVENTS"))
    assert "CUPTI_ACTIVITY_KIND_KERNEL" in schema
    assert "NVTX_EVENTS" in schema
    conn.close()


def test_get_profile_schema_cached_reuses_cache():
    """get_profile_schema_cached returns same result without querying DB again (§11.7.7)."""
    from nsys_ai.tools_profile import _schema_cache, get_profile_schema_cached

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE NVTX_EVENTS(text TEXT)")
    path = "/tmp/test_cache_path_unique_12345.sqlite"
    # Clear any stale cache entry
    _schema_cache.pop(path, None)

    schema1 = get_profile_schema_cached(conn, path)
    # Close the connection; a second call must use the cache, not the closed conn
    conn.close()
    schema2 = get_profile_schema_cached(None, path)  # conn unused when cache hits
    assert schema1 == schema2

    # Cleanup
    _schema_cache.pop(path, None)


def test_get_profile_schema_cached_no_path():
    """get_profile_schema_cached with path=None always calls get_profile_schema."""
    from nsys_ai.tools_profile import get_profile_schema_cached

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE NVTX_EVENTS(col INT)")
    schema = get_profile_schema_cached(conn, None)
    assert "NVTX_EVENTS" in schema
    conn.close()


def test_open_profile_readonly_for_worker_returns_connection(tmp_path):
    """open_profile_readonly_for_worker returns a usable read-only connection (§11.7.5)."""
    from nsys_ai.tools_profile import open_profile_readonly_for_worker

    db = tmp_path / "test.sqlite"
    # Create a DB with a recognised Nsight table so the Parquet cache exports it
    setup = sqlite3.connect(str(db))
    setup.execute("CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT)")
    setup.execute("INSERT INTO StringIds VALUES(1, 'hello')")
    setup.commit()
    setup.close()

    conn = open_profile_readonly_for_worker(str(db))
    rows = conn.execute("SELECT id, value FROM StringIds").fetchall()
    assert rows[0][0] == 1
    assert rows[0][1] == "hello"
    conn.close()


def test_query_profile_db_empty_query():
    """query_profile_db rejects empty or whitespace-only queries."""
    from nsys_ai.tools_profile import query_profile_db

    conn = sqlite3.connect(":memory:")
    assert "Error" in query_profile_db(conn, "")
    assert "Error" in query_profile_db(conn, "   ")
    conn.close()


def test_adaptive_limit_narrow_query():
    """Single-column queries get up to 2× base limit (§11.9 Phase 2.1)."""
    from nsys_ai.tools_profile import DEFAULT_MAX_LIMIT, query_profile_db

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t(name TEXT)")
    for i in range(200):
        conn.execute("INSERT INTO t VALUES(?)", (f"k{i}",))
    out = query_profile_db(conn, "SELECT name FROM t")
    rows = json.loads(out)
    # Single-column → effective limit = min(100, DEFAULT_MAX_LIMIT * 2)
    assert len(rows) <= min(100, DEFAULT_MAX_LIMIT * 2)
    assert len(rows) > DEFAULT_MAX_LIMIT  # must be higher than the default limit
    conn.close()


def test_adaptive_limit_wide_query():
    """Queries with 4+ columns get a reduced limit (§11.9 Phase 2.1)."""
    from nsys_ai.tools_profile import DEFAULT_MAX_LIMIT, query_profile_db

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t(a TEXT, b TEXT, c TEXT, d TEXT, e TEXT)")
    for i in range(100):
        conn.execute("INSERT INTO t VALUES(?,?,?,?,?)", (f"v{i}",) * 5)
    out = query_profile_db(conn, "SELECT a, b, c, d, e FROM t")
    rows = json.loads(out)
    # 5 columns → effective limit = max(20, DEFAULT_MAX_LIMIT // 2) = 25
    expected_max = max(20, DEFAULT_MAX_LIMIT // 2)
    assert len(rows) <= expected_max
    conn.close()


def test_query_profile_db_truncates_large_results():
    """query_profile_db truncates JSON output beyond DEFAULT_MAX_JSON_CHARS (§11.7.6)."""
    from nsys_ai.tools_profile import query_profile_db

    conn = sqlite3.connect(":memory:")
    # Create rows with long string values to exceed the JSON char limit
    conn.execute("CREATE TABLE t(v TEXT)")
    long_val = "x" * 500
    conn.executemany("INSERT INTO t VALUES(?)", [(long_val,)] * 50)
    out = query_profile_db(conn, "SELECT v FROM t", max_limit=50)
    # Should be truncated with actionable hint
    assert "Truncated" in out
    assert "refine your query" in out.lower() or "reduce the LIMIT" in out
    conn.close()
