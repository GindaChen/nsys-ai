"""`resolve_activity_tables` rescanned the catalog on every call.

It is called from readers that run per NVTX thread and from `requires_nvtx` /
`resolve_nvtx_table`, which every NVTX skill goes through, so the scan multiplied
by whatever the caller was iterating. Measured on `h100_2gpu_1s.sqlite`: 1605 µs
per call against the DuckDB adapter, 20 µs against SQLite — the DuckDB catalog
scan is the one that hurts, and it is the adapter the cache path uses.

Caching it is only safe because the catalog is fixed before a caller sees the
connection: `open_cached_db`, `open_parquetdir_db` and `open_direct_sqlite_db`
each create their parquet views and alias views and only then return. That
precondition is what the tests below pin — not the timing, which no CI machine
can assert reliably.
"""

import sqlite3
from pathlib import Path

import pytest

from nsys_ai.connection import wrap_connection

REPO = Path(__file__).resolve().parent.parent
ANNOTATED = REPO / "tests" / "fixtures" / "h100_2gpu_1s.sqlite"


class _CountingSQLite(sqlite3.Connection):
    """A profile connection that records how often the catalog is scanned."""

    catalog_scans = 0

    def execute(self, sql, *args):  # noqa: D102
        if "sqlite_master" in sql:
            type(self).catalog_scans += 1
        return super().execute(sql, *args)


@pytest.fixture
def counting_conn():
    _CountingSQLite.catalog_scans = 0
    conn = sqlite3.connect(
        f"file:{ANNOTATED}?mode=ro", uri=True, factory=_CountingSQLite
    )
    try:
        yield conn
    finally:
        conn.close()


def test_the_catalog_is_scanned_once_per_connection(counting_conn):
    """The whole point: repeat calls must not re-scan.

    Each call goes through a *fresh* adapter, which is how callers actually use
    it — `resolve_nvtx_table` wraps the raw connection every time. So the cache
    has to hang off the connection, not off the adapter instance.
    """
    first = wrap_connection(counting_conn).resolve_activity_tables()
    assert _CountingSQLite.catalog_scans == 1, "premise: the first call must scan"

    for _ in range(5):
        again = wrap_connection(counting_conn).resolve_activity_tables()
        assert again == first

    assert _CountingSQLite.catalog_scans == 1, (
        f"the catalog was scanned {_CountingSQLite.catalog_scans} times for one connection"
    )


def test_the_answer_is_unchanged_by_caching(counting_conn):
    """A cache that returns something different from the uncached call is worse
    than no cache."""
    from nsys_ai.connection import _find_activity_tables

    rows = counting_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    expected = _find_activity_tables({row[0] for row in rows.fetchall()})

    assert wrap_connection(counting_conn).resolve_activity_tables() == expected
    assert expected.get("kernel"), "premise: the fixture must have a kernel table"


def test_a_caller_cannot_poison_the_cache(counting_conn):
    """The memoized value is mutable, so it is handed out as a copy.

    Without this every caller shares one dict, and a single `.pop()` or
    reassignment anywhere in the package silently changes what every later reader
    resolves.
    """
    first = wrap_connection(counting_conn).resolve_activity_tables()
    first["kernel"] = "SOMETHING_ELSE"
    first.pop("runtime", None)

    second = wrap_connection(counting_conn).resolve_activity_tables()
    assert second["kernel"] != "SOMETHING_ELSE"
    assert "runtime" in second


def test_two_connections_do_not_share_an_answer(tmp_path):
    """Keyed per connection, and the key survives one of them being closed.

    The bag is keyed by the connection object rather than `id()` precisely
    because `id()` is recycled after a close, which would let a second profile
    inherit the first's resolution.
    """
    import shutil

    renamed = tmp_path / "runtime_v3.sqlite"
    shutil.copy(ANNOTATED, renamed)
    conn = sqlite3.connect(renamed)
    try:
        conn.execute(
            "ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME "
            "RENAME TO CUPTI_ACTIVITY_KIND_RUNTIME_V3"
        )
        conn.commit()
    finally:
        conn.close()

    plain = sqlite3.connect(f"file:{ANNOTATED}?mode=ro", uri=True)
    versioned = sqlite3.connect(f"file:{renamed}?mode=ro", uri=True)
    try:
        assert wrap_connection(plain).resolve_activity_tables()["runtime"] == (
            "CUPTI_ACTIVITY_KIND_RUNTIME"
        )
        assert wrap_connection(versioned).resolve_activity_tables()["runtime"] == (
            "CUPTI_ACTIVITY_KIND_RUNTIME_V3"
        )
    finally:
        plain.close()
        versioned.close()


def test_a_failed_scan_is_not_cached():
    """A closed or broken connection must not pin `{}` for good.

    Caching the failure would turn a transient error into a profile that has no
    activity tables for the rest of its life, which reads as an empty profile
    rather than as an error.
    """
    conn = sqlite3.connect(f"file:{ANNOTATED}?mode=ro", uri=True)
    conn.close()

    assert wrap_connection(conn).resolve_activity_tables() == {}

    revived = sqlite3.connect(f"file:{ANNOTATED}?mode=ro", uri=True)
    try:
        assert wrap_connection(revived).resolve_activity_tables().get("kernel")
    finally:
        revived.close()


def test_every_duckdb_opener_creates_its_views_before_returning():
    """The precondition the cache rests on.

    On DuckDB `SHOW TABLES` lists views, so resolving before the alias views
    exist would cache the versioned name — or, in direct-SQLite mode where the
    real tables live under `src.`, cache nothing at all. Each opener must finish
    its view creation before a caller can resolve anything.
    """
    src = (REPO / "src" / "nsys_ai" / "parquet_cache.py").read_text()

    for opener, creator in (
        ("def open_cached_db", "_create_existing_alias_views(db)"),
        ("def open_parquetdir_db", "_create_existing_alias_views(db)"),
        ("def open_direct_sqlite", "_create_sqlite_alias_views(db)"),
    ):
        start = src.index(opener)
        body = src[start : src.index("\ndef ", start + 1)]
        assert creator in body, f"{opener} no longer creates its alias views"
        assert body.index(creator) < body.rindex("return db"), (
            f"{opener} returns the connection before creating its alias views — "
            "resolve_activity_tables may now cache a pre-alias catalog"
        )
