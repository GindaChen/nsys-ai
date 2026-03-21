"""
test_duckdb_path.py — Verify that _duckdb_query() works via DuckDB, not SQLite fallback.

These tests use the ``duckdb_conn`` fixture (DuckDB in-memory) and construct a
Profile-like object with ``db=duckdb_conn`` and ``conn=None``.  Any code path
that accidentally touches ``self.conn`` will crash here — which is the point.
"""

import threading

import duckdb
import pytest

from nsys_ai.sql_compat import sqlite_to_duckdb


# ---------------------------------------------------------------------------
# Minimal Profile stub that only supports _duckdb_query (DuckDB path).
# ---------------------------------------------------------------------------


class _DuckDBOnlyProfile:
    """Profile-shaped stub with db=DuckDB and conn=None (no SQLite fallback)."""

    _log = __import__("logging").getLogger("test_duckdb_path")

    def __init__(self, db):
        self.db = db
        self.conn = None  # Forces DuckDB path — SQLite fallback will crash
        self._lock = threading.RLock()
        self._nvtx_has_text_id = True
        self._warned_sqlite_fallback = False

    def _duckdb_query(self, sql: str, params=None) -> list[dict]:
        conn = self.db if self.db is not None else self.conn
        if isinstance(conn, duckdb.DuckDBPyConnection):
            ddb_sql = sqlite_to_duckdb(sql)
            cur = conn.execute(ddb_sql, params or [])
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        else:
            raise RuntimeError("Should never reach SQLite fallback in this test")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDuckDBQueryPath:
    """Exercises _duckdb_query() through DuckDB — catches SQL translation bugs."""

    def test_select_with_bracket_end(self, duckdb_conn):
        """``[end]`` in SQL is translated to ``"end"`` for DuckDB."""
        p = _DuckDBOnlyProfile(duckdb_conn)
        rows = p._duckdb_query(
            "SELECT start, [end] FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start LIMIT 1"
        )
        assert len(rows) == 1
        assert "end" in rows[0]
        assert rows[0]["start"] == 1_000_000

    def test_kernel_query_with_join(self, duckdb_conn):
        """Typical kernel query with JOIN + bracket escaping works."""
        p = _DuckDBOnlyProfile(duckdb_conn)
        rows = p._duckdb_query(
            """
            SELECT k.start, k.[end], s.value AS name
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName = s.id
            WHERE k.deviceId = ?
            ORDER BY k.start
            """,
            (0,),
        )
        assert len(rows) == 5
        assert rows[0]["name"] == "kernel_A"

    def test_aggregate_with_bracket_end(self, duckdb_conn):
        """COUNT + SUM with ``[end]`` in expressions works."""
        p = _DuckDBOnlyProfile(duckdb_conn)
        rows = p._duckdb_query(
            """
            SELECT COUNT(*) AS cnt,
                   SUM(k.[end] - k.start) AS total_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            WHERE k.deviceId = ?
            """,
            (0,),
        )
        assert rows[0]["cnt"] == 5
        assert rows[0]["total_ns"] > 0

    def test_nvtx_query(self, duckdb_conn):
        """NVTX query with ``[end]`` works."""
        p = _DuckDBOnlyProfile(duckdb_conn)
        rows = p._duckdb_query(
            """
            SELECT text, start, [end]
            FROM NVTX_EVENTS
            WHERE text IS NOT NULL AND [end] > start
            ORDER BY start
            """
        )
        assert len(rows) == 2
        assert rows[0]["text"] == "train_step"

    def test_runtime_query(self, duckdb_conn):
        """CUDA Runtime query with ``[end]`` works."""
        p = _DuckDBOnlyProfile(duckdb_conn)
        rows = p._duckdb_query(
            """
            SELECT correlationId, start, [end]
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE globalTid = ? ORDER BY start
            """,
            (100,),
        )
        assert len(rows) == 11

    def test_memcpy_query(self, duckdb_conn):
        """Memcpy query with bracket escaping works."""
        p = _DuckDBOnlyProfile(duckdb_conn)
        rows = p._duckdb_query(
            """
            SELECT start, [end], copyKind, bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE deviceId = ?
            ORDER BY start
            """,
            (0,),
        )
        assert len(rows) == 5

    def test_window_function_with_bracket_end(self, duckdb_conn):
        """LAG() window function with ``[end]`` — used in GPU idle gap detection."""
        p = _DuckDBOnlyProfile(duckdb_conn)
        rows = p._duckdb_query(
            """
            WITH ordered AS (
                SELECT k.streamId, k.start, k.[end],
                       LAG(k.[end]) OVER (
                           PARTITION BY k.streamId ORDER BY k.start
                       ) AS prev_end
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                WHERE k.deviceId = ?
            )
            SELECT * FROM ordered WHERE prev_end IS NOT NULL
            """,
            (0,),
        )
        # Should have rows for kernels that have a predecessor on same stream
        assert len(rows) > 0

    def test_describe_works(self, duckdb_conn):
        """DESCRIBE instead of PRAGMA table_info works on DuckDB."""
        p = _DuckDBOnlyProfile(duckdb_conn)
        rows = p._duckdb_query("DESCRIBE CUPTI_ACTIVITY_KIND_KERNEL")
        col_names = [r.get("column_name") or r.get("Field") for r in rows]
        assert "start" in col_names
        assert "end" in col_names

    def test_show_tables(self, duckdb_conn):
        """SHOW TABLES works (replaces sqlite_master)."""
        p = _DuckDBOnlyProfile(duckdb_conn)
        rows = p._duckdb_query("SHOW TABLES")
        table_names = {r.get("name") for r in rows}
        assert "CUPTI_ACTIVITY_KIND_KERNEL" in table_names
        assert "NVTX_EVENTS" in table_names
        assert "StringIds" in table_names

    def test_thread_names_query(self, duckdb_conn):
        """ThreadNames table query works (used by nvtx_tree.py)."""
        p = _DuckDBOnlyProfile(duckdb_conn)
        rows = p._duckdb_query(
            """
            SELECT s.value FROM ThreadNames t
            JOIN StringIds s ON t.nameId = s.id
            WHERE t.globalTid = ?
            ORDER BY t.priority DESC LIMIT 1
            """,
            (100,),
        )
        assert len(rows) == 1
        assert rows[0]["value"] == "cudaLaunchKernel"
