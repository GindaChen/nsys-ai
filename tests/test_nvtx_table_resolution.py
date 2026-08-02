"""The NVTX table name is versioned, and two more readers still named it literally.

Newer Nsight exports call the table NVTX_EVENTS_V2 or NVTX_EVENTS_V3. #285, #288
and #291 resolved it in the skills layer, `profile.py`, `nvtx_tree.py` and
`overlap.py`; `nccl_communicator.py` and `region_mfu.py` kept the literal.

The failure is quiet rather than loud. `region_mfu` returns no ranges, and
`nccl_communicator` is reached through `profile_health_manifest`'s
`_safe_skill_run`, which turns any database error into `[]`. A profile that has
NCCL communicator data is reported as having none.

That is what these tests are shaped around. #288 shipped a source-text assertion
that kept passing while the fix was inert and while these two files kept their
literals, so checking the code reads a resolved name proves nothing. Each test
below renames the table on a real annotated fixture and asserts on what the
reader returns, which is the only thing that separates resolved from
looks-resolved.
"""

import shutil
import sqlite3
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ANNOTATED = REPO / "tests" / "fixtures" / "h100_2gpu_1s.sqlite"

# The suffix newer exports use. _V2 and _V3 both occur; resolution is by prefix,
# so exercising one covers both.
RENAMED = "NVTX_EVENTS_V3"


def _nvtx_tables(path: Path) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'NVTX_EVENTS%'"
            )
        }
    finally:
        conn.close()


@pytest.fixture(scope="module")
def versioned_profile(tmp_path_factory) -> Path:
    """A copy of the annotated fixture with NVTX_EVENTS renamed to NVTX_EVENTS_V3.

    Renaming rather than hand-building keeps the row content real: 16k annotated
    ranges with the text/textId split an actual PyTorch capture produces.
    """
    dst = tmp_path_factory.mktemp("versioned") / "nvtx_v3.sqlite"
    shutil.copy(ANNOTATED, dst)
    conn = sqlite3.connect(dst)
    try:
        conn.execute(f"ALTER TABLE NVTX_EVENTS RENAME TO {RENAMED}")
        conn.commit()
    finally:
        conn.close()
    return dst


# ── Premises ────────────────────────────────────────────────────────────────
# Without these the tests below can pass vacuously: an empty fixture would make
# "no rows" the correct answer under both the literal and the resolved name.


def test_source_fixture_is_annotated():
    assert ANNOTATED.exists(), f"fixture missing: {ANNOTATED}"
    assert _nvtx_tables(ANNOTATED) == {"NVTX_EVENTS"}
    conn = sqlite3.connect(ANNOTATED)
    try:
        named = conn.execute(
            "SELECT COUNT(*) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id "
            "WHERE COALESCE(n.text, s.value) IS NOT NULL AND n.[end] > n.start"
        ).fetchone()[0]
    finally:
        conn.close()
    assert named > 0, "fixture has no named NVTX ranges — the tests below prove nothing"


def test_renamed_copy_has_only_the_versioned_table(versioned_profile):
    """The literal name must be genuinely absent, or resolution is untested."""
    assert _nvtx_tables(versioned_profile) == {RENAMED}


# ── region_mfu ──────────────────────────────────────────────────────────────


def _sample_range_name(path: Path) -> str:
    """A range name that really occurs, so a miss means resolution, not a typo."""
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT COALESCE(n.text, s.value) FROM NVTX_EVENTS n "
            "LEFT JOIN StringIds s ON n.textId = s.id "
            "WHERE COALESCE(n.text, s.value) IS NOT NULL AND n.[end] > n.start LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    # Range texts look like "aten::view, op_id = 314290"; match the stable head.
    return row[0].split(",")[0]


def test_find_nvtx_ranges_reads_the_versioned_table(versioned_profile):
    """The assertion that fails while the table name is a literal."""
    from nsys_ai.region_mfu import find_nvtx_ranges

    name = _sample_range_name(ANNOTATED)
    conn = sqlite3.connect(versioned_profile)
    try:
        rows = find_nvtx_ranges(conn, name)
    finally:
        conn.close()

    assert rows, (
        f"find_nvtx_ranges returned nothing for {name!r} on a profile whose NVTX "
        f"table is {RENAMED} — the table name is not being resolved"
    )


def test_find_nvtx_ranges_agrees_across_both_table_names(versioned_profile):
    """Resolution must not change the answer, only where it is read from."""
    from nsys_ai.region_mfu import find_nvtx_ranges

    name = _sample_range_name(ANNOTATED)

    original = sqlite3.connect(ANNOTATED)
    renamed = sqlite3.connect(versioned_profile)
    try:
        before = find_nvtx_ranges(original, name)
        after = find_nvtx_ranges(renamed, name)
    finally:
        original.close()
        renamed.close()

    assert before, "premise: the unrenamed profile must return ranges"
    assert len(after) == len(before)


# ── nccl_communicator ───────────────────────────────────────────────────────
# Only one of this module's two readers is reachable with the literal name.
#
# The primary reader goes through ``Profile._duckdb_query``, and
# ``parquet_cache`` creates an alias view named NVTX_EVENTS over whatever
# versioned table it finds — in cached and in direct-SQLite mode both. So that
# query answers correctly today despite the literal, and no test written against
# it can fail. It is still resolved below, because the alias layer is a
# coincidence of how the profile happens to be opened rather than a guarantee
# this module is entitled to.
#
# The blob fallback opens its own ``sqlite3`` connection, which has no alias
# layer. That one is genuinely broken, and it is what the test asserts on.


def test_duckdb_reader_is_not_the_broken_one(versioned_profile):
    """Records why the primary reader has no failing test: it cannot fail.

    If parquet_cache ever stops aliasing the unversioned name, this starts
    failing and points at the reader that would then be exposed.
    """
    from nsys_ai import profile as profile_mod

    with profile_mod.open(str(versioned_profile), cache_mode="direct") as prof:
        via_literal = prof._duckdb_query("SELECT COUNT(*) AS n FROM NVTX_EVENTS")
        via_actual = prof._duckdb_query(f"SELECT COUNT(*) AS n FROM {RENAMED}")

    assert via_literal[0]["n"] == via_actual[0]["n"] > 0


def test_sqlite_blob_fallback_reads_the_versioned_table(versioned_profile):
    """The assertion that fails while the fallback's table name is a literal."""
    from nsys_ai.nccl_communicator import _load_nvtx_payload_rows_sqlite

    _rows, diagnostics = _load_nvtx_payload_rows_sqlite(str(versioned_profile))

    assert not diagnostics, (
        f"the SQLite blob fallback failed against a profile whose NVTX table is "
        f"{RENAMED} — the table name is not being resolved: {diagnostics}"
    )
