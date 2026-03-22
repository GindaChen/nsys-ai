"""Tests for sql_compat module — SQLite-to-DuckDB SQL translation."""

from nsys_ai.sql_compat import sqlite_to_duckdb


class TestSqliteToDuckdb:
    """Test bracket-escaping rewrite."""

    def test_end_column(self):
        assert sqlite_to_duckdb("SELECT [end] FROM t") == 'SELECT "end" FROM t'

    def test_start_column(self):
        assert sqlite_to_duckdb("SELECT [start] FROM t") == 'SELECT "start" FROM t'

    def test_qualified_column(self):
        result = sqlite_to_duckdb("SELECT k.[end] FROM kernel k")
        assert result == 'SELECT k."end" FROM kernel k'

    def test_multiple_brackets(self):
        sql = "SELECT k.[start], k.[end], ([end] - [start]) AS dur FROM t"
        result = sqlite_to_duckdb(sql)
        assert result == 'SELECT k."start", k."end", ("end" - "start") AS dur FROM t'

    def test_no_brackets(self):
        sql = "SELECT name, start FROM t WHERE name LIKE ?"
        assert sqlite_to_duckdb(sql) == sql

    def test_preserves_array_indexing(self):
        """DuckDB array indexing (arr[0]) should not be bracket-escaped as an identifier."""
        # Note: arr[0] is preserved because the translation regex only rewrites bracketed
        # identifiers where the first character after '[' is a letter or underscore
        # (e.g., [end]), not a digit as in [0].
        sql = "SELECT arr[0] FROM t"
        assert sqlite_to_duckdb(sql) == sql

    def test_where_clause(self):
        sql = "WHERE k.start >= ? AND k.[end] <= ?"
        assert sqlite_to_duckdb(sql) == 'WHERE k.start >= ? AND k."end" <= ?'

    def test_window_function(self):
        sql = "LAG([end]) OVER (ORDER BY start)"
        assert sqlite_to_duckdb(sql) == 'LAG("end") OVER (ORDER BY start)'

    def test_hex_literal(self):
        """DuckDB does not support 0x... hex literals natively in standard SQL mode."""
        sql = "SELECT [end] - start FROM k WHERE start > 0x1000000"
        # 0x1000000 is 16777216
        assert sqlite_to_duckdb(sql) == 'SELECT "end" - start FROM k WHERE start > 16777216'
