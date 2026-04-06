"""
test_duckdb_path.py — Verify that _duckdb_query() works via DuckDB, not SQLite fallback.

These tests use the ``duckdb_conn`` fixture (DuckDB in-memory) and construct a
Profile-like object with ``db=duckdb_conn`` and ``conn=None``.  Any code path
that accidentally touches ``self.conn`` will crash here — which is the point.
"""

import threading

import duckdb

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


class TestDuckDBCompatibilityFixes:
    """Regression tests for DuckDB compatibility bugs (tuple vs Row, PRAGMA vs DESCRIBE)."""

    def test_gpu_idle_gaps_skill(self, duckdb_conn):
        """gpu_idle_gaps skill should not crash on DuckDB tuples or missing row_factory."""
        from nsys_ai.skills.registry import get_skill

        skill = get_skill("gpu_idle_gaps")
        # Should execute successfully and return rows
        rows = skill.execute(duckdb_conn)
        assert isinstance(rows, list)
        if rows:
            assert isinstance(rows[0], dict)

    def test_profile_from_conn_row_factory(self, duckdb_conn):
        """Profile._from_conn should not crash when assigning row_factory to DuckDB."""
        from nsys_ai.profile import Profile

        p = Profile._from_conn(duckdb_conn)
        assert p.conn is duckdb_conn
        assert p.db is duckdb_conn  # DuckDB connection is stored in p.db

    def test_root_cause_matcher_safe_execute(self, duckdb_conn):
        """root_cause_matcher should gracefully handle any internal skill exceptions."""
        from nsys_ai.skills.registry import get_skill

        skill = get_skill("root_cause_matcher")
        # Ensure it doesn't crash even if a sub-skill throws an error
        rows = skill.execute(duckdb_conn)
        assert isinstance(rows, list)

    def test_region_mfu_detect_nvtx_text_id(self, duckdb_conn):
        """ConnectionAdapter.detect_nvtx_text_id should use DESCRIBE for DuckDB."""
        from nsys_ai.connection import wrap_connection

        # Our mock duckdb_conn has an NVTX_EVENTS table with textId
        adapter = wrap_connection(duckdb_conn)
        has_text_id = adapter.detect_nvtx_text_id()
        assert has_text_id is True

    def test_profile_metadata_discovery(self, duckdb_conn):
        """Profile._detect_nvtx_text_id and NsightSchema should work with DuckDB."""
        from nsys_ai.profile import Profile

        p = Profile._from_conn(duckdb_conn)
        # Should have successfully detected version and NVTX text ID
        assert p._nvtx_has_text_id is True
        # DuckDB mocked schema doesn't have META_DATA_EXPORT so version is None,
        # but the method should have run without crashing on PRAGMA
        assert type(p.meta).__name__ == "ProfileMeta"

    def test_memory_transfers_duckdb_error(self, duckdb_conn):
        """memory_transfers H2D distribution should catch duckdb.Error on missing table."""
        import duckdb

        from nsys_ai.skills.registry import get_skill

        skill = get_skill("h2d_distribution")

        # Create a fresh DuckDB connection without the memcpy table
        bad_conn = duckdb.connect()
        # Should quietly return [] instead of raising duckdb.Error
        rows = skill.execute(bad_conn)
        assert rows == []
