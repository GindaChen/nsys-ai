"""Tests for Nsight export metadata detection (#235).

`META_DATA_*` lives only in the SQLite export, and its columns are lowercase
`name`/`value`. Both facts were missed, so every metadata field read as absent
on every real profile.
"""

import sqlite3

from nsys_ai.profile import NsightSchema


def _meta_db(rows, table="META_DATA_EXPORT", key_col="name", val_col="value"):
    conn = sqlite3.connect(":memory:")
    conn.execute(f"CREATE TABLE {table} ({key_col} TEXT, {val_col} TEXT)")
    conn.executemany(f"INSERT INTO {table} VALUES (?, ?)", rows)
    conn.commit()
    return conn


_REAL_ROWS = [
    ("EXPORT_PRODUCT_NAME", "NVIDIA Nsight Systems"),
    ("EXPORT_PRODUCT_VERSION", "2026.1.1.204"),
    ("EXPORT_SCHEMA_VERSION", "3.24.14"),
]


def test_reads_lowercase_name_column():
    """The real exports use `name`; only NAME/Name were probed, so the reader
    returned an empty mapping and every consumer degraded to None."""
    s = NsightSchema(_meta_db(_REAL_ROWS))
    kv = s._read_kv_table("META_DATA_EXPORT")
    assert kv["EXPORT_PRODUCT_VERSION"] == "2026.1.1.204"


def test_detects_product_version():
    s = NsightSchema(_meta_db(_REAL_ROWS))
    assert s.version == "2026.1.1.204"


def test_exposes_export_schema_version():
    """The schema version is what actually changes shape between releases, so
    it is the field compatibility handling should key off."""
    s = NsightSchema(_meta_db(_REAL_ROWS))
    assert s.schema_version == "3.24.14"


def test_product_name_is_never_mistaken_for_a_version():
    """The old value-substring fallback would return 'NVIDIA Nsight Systems'
    itself, since that string contains 'Nsight Systems'."""
    s = NsightSchema(_meta_db([("EXPORT_PRODUCT_NAME", "NVIDIA Nsight Systems")]))
    assert s.version is None


def test_uppercase_column_names_still_work():
    """Older exports used NAME/Value; matching is case-insensitive now."""
    s = NsightSchema(_meta_db(_REAL_ROWS, key_col="NAME", val_col="Value"))
    assert s.version == "2026.1.1.204"


def test_missing_metadata_is_absent_not_fabricated():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
    conn.commit()
    s = NsightSchema(conn)
    assert s.version is None
    assert s.schema_version is None


def test_metadata_falls_back_to_the_sqlite_connection():
    """The Parquet/DuckDB cache does not materialize META_DATA_*, so a schema
    built on the cache must consult the original SQLite export or metadata is
    invisible on the primary product path."""
    cache_like = sqlite3.connect(":memory:")  # stands in for the cache: no META_DATA_*
    cache_like.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
    cache_like.commit()

    s = NsightSchema(cache_like, meta_conn=_meta_db(_REAL_ROWS))
    assert "META_DATA_EXPORT" not in s.tables  # genuinely absent from the primary conn
    assert s.version == "2026.1.1.204"  # ...but recovered from the metadata source
    assert s.schema_version == "3.24.14"
