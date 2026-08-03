"""One rule for choosing which variant of an Nsight table a profile carries.

Nsight Systems documents only unversioned ``CUPTI_ACTIVITY_KIND_*`` names — the
export as a whole is versioned through ``EXPORT_META_DATA.EXPORT_SCHEMA_VERSION``
— but this repo has met ``_V2``/``_V3`` suffixed exports, so it resolves names by
prefix. Until now it did so in three places with two different orderings:

* ``connection._find_activity_tables`` took ``sorted(...)[0]`` — the *oldest*
  variant — and never emitted a ``sync``/``sync_type`` key at all, so the
  ``{sync_table}`` placeholder that ``skills/base.py`` advertises always fell
  through to its unversioned literal;
* ``parquet_cache._find_table`` also took the oldest, deciding which table the
  cache builder copies;
* ``sync_cost_analysis`` had a private resolver taking ``ORDER BY name DESC`` —
  the *newest* — whose ``WHERE type='table'`` filter matched nothing on the
  Parquet-cache connection, where every catalog entry is a view.

These tests pin the single rule that replaced them (exact name first, then the
highest ``_V<n>`` by integer), the two newly-emitted keys, and the property that
made the three changes inseparable: with the resolvers unified but the ordering
left alone, the two engines pick different tables on a profile that carries
both variants.
"""

import shutil
import sqlite3
from pathlib import Path

import pytest

from nsys_ai.connection import _find_activity_tables, wrap_connection
from nsys_ai.skills.base import Skill
from nsys_ai.skills.registry import get_skill

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"

SYNC = "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"


# ── The ordering rule, in isolation ─────────────────────────────────────────


def test_the_unversioned_name_wins_over_any_suffixed_variant():
    """The documented name is the exporter's own answer; a suffixed table
    alongside it is a leftover, not an upgrade."""
    resolved = _find_activity_tables(
        {
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_KERNEL_V2",
            "CUPTI_ACTIVITY_KIND_KERNEL_V3",
        }
    )
    assert resolved["kernel"] == "CUPTI_ACTIVITY_KIND_KERNEL"


def test_versions_are_ordered_as_integers_not_as_strings():
    """``_V10`` is newer than ``_V3``. Lexicographic order says the opposite,
    and plain ``max()`` over the names would pick ``_V3``.

    Both sets are needed. The three-variant set alone does not discriminate
    against the ``sorted(...)[0]`` rule this replaced: ``_V10`` sorts first
    lexicographically, so the old rule returned the right answer here for the
    wrong reason. The two-variant set is where the old rule visibly picks the
    older table.
    """
    resolved = _find_activity_tables(
        {
            "CUPTI_ACTIVITY_KIND_KERNEL_V2",
            "CUPTI_ACTIVITY_KIND_KERNEL_V3",
            "CUPTI_ACTIVITY_KIND_KERNEL_V10",
        }
    )
    assert resolved["kernel"] == "CUPTI_ACTIVITY_KIND_KERNEL_V10"

    resolved = _find_activity_tables(
        {"CUPTI_ACTIVITY_KIND_KERNEL_V2", "CUPTI_ACTIVITY_KIND_KERNEL_V3"}
    )
    assert resolved["kernel"] == "CUPTI_ACTIVITY_KIND_KERNEL_V3"


def test_a_peer_to_peer_memcpy_table_is_not_mistaken_for_a_memcpy_variant():
    """``CUPTI_ACTIVITY_KIND_MEMCPY2`` is a real CUPTI activity kind (P2P
    memcpy), not a version of ``..._MEMCPY``. Anchoring the ``_V<n>`` match is
    what keeps the two apart."""
    resolved = _find_activity_tables(
        {"CUPTI_ACTIVITY_KIND_MEMCPY_V3", "CUPTI_ACTIVITY_KIND_MEMCPY2"}
    )
    assert resolved["memcpy"] == "CUPTI_ACTIVITY_KIND_MEMCPY_V3"


def test_an_unrecognised_suffix_still_resolves_to_something():
    """The query-side resolver keeps a permissive last resort so a future
    suffix shape degrades to "reads the odd table" rather than "reads nothing"."""
    resolved = _find_activity_tables({"CUPTI_ACTIVITY_KIND_KERNEL_NEXTGEN"})
    assert resolved["kernel"] == "CUPTI_ACTIVITY_KIND_KERNEL_NEXTGEN"


def test_the_sync_tables_are_resolved_at_all():
    """``skills/base.py`` offers ``{sync_table}`` and ``{sync_type_table}``;
    without these keys both placeholders always took the literal fallback."""
    resolved = _find_activity_tables({SYNC + "_V2", "ENUM_CUPTI_SYNC_TYPE_V2"})
    assert resolved["sync"] == SYNC + "_V2"
    assert resolved["sync_type"] == "ENUM_CUPTI_SYNC_TYPE_V2"


# ── The placeholder, against a real versioned profile ───────────────────────


@pytest.fixture(scope="module")
def sync_v2(tmp_path_factory) -> Path:
    """The annotated fixture with its synchronization table renamed to _V2."""
    dst = tmp_path_factory.mktemp("sync_v2") / "sync_v2.sqlite"
    shutil.copy(FIXTURE, dst)
    conn = sqlite3.connect(dst)
    try:
        conn.execute(f"ALTER TABLE {SYNC} RENAME TO {SYNC}_V2")
        conn.commit()
    finally:
        conn.close()
    return dst


def _counting_skill() -> Skill:
    """A skill whose whole body is the placeholder under test."""
    return Skill(
        name="_sync_table_placeholder_probe",
        title="probe",
        description="probe",
        category="system",
        sql="SELECT COUNT(*) AS n FROM {sync_table}",
    )


def test_the_fixture_has_sync_rows():
    """Premise: without sync rows the assertions below pass vacuously."""
    conn = sqlite3.connect(f"file:{FIXTURE}?mode=ro", uri=True)
    try:
        n = conn.execute(f"SELECT COUNT(*) FROM {SYNC}").fetchone()[0]
    finally:
        conn.close()
    assert n > 0


def test_sync_table_placeholder_reads_a_versioned_table_over_sqlite3(sync_v2):
    """The failure this PR removes: a plain ``sqlite3`` connection has no alias
    views, so an unresolved ``{sync_table}`` raises "no such table"."""
    conn = sqlite3.connect(sync_v2)
    try:
        rows = _counting_skill().execute(conn)
    finally:
        conn.close()

    assert rows and rows[0]["n"] > 0, f"{SYNC}_V2 produced no rows: {rows}"


def test_sync_table_placeholder_agrees_across_engines(sync_v2):
    """The cached path answered correctly all along, via the alias view for the
    unversioned name. Both engines must now give the same count."""
    from nsys_ai.profile import Profile

    pytest.importorskip("duckdb", reason="requires duckdb")

    conn = sqlite3.connect(sync_v2)
    try:
        via_sqlite = _counting_skill().execute(conn)
    finally:
        conn.close()

    with Profile(str(sync_v2)) as prof:
        if prof.db is None:  # pragma: no cover
            pytest.skip("requires duckdb")
        via_duckdb = _counting_skill().execute(prof.db)

    assert via_sqlite == via_duckdb


def test_sync_cost_analysis_reads_a_versioned_table_over_sqlite3(sync_v2):
    """The shipped skill, on the path with no alias layer."""
    conn = sqlite3.connect(sync_v2)
    try:
        rows = get_skill("sync_cost_analysis").execute(conn)
    finally:
        conn.close()

    assert "error" not in rows[0], rows[0]["error"]
    assert rows[0]["total_sync_wall_ms"] > 0


# ── The coupling: ordering and deduplication must land together ─────────────


@pytest.fixture(scope="module")
def dual_variant(tmp_path_factory) -> Path:
    """A profile carrying both ``_V2`` (truncated) and ``_V3`` (complete).

    Not something nsys emits — the exporter documents no table-name versioning
    at all — but it is the only shape in which the resolvers' orderings are
    distinguishable, so it is what pins them together. The two tables carry
    *different* row counts on purpose: picking the wrong one is then visible in
    the analysis result, not just in the resolved name.
    """
    dst = tmp_path_factory.mktemp("dual") / "dual.sqlite"
    shutil.copy(FIXTURE, dst)
    conn = sqlite3.connect(dst)
    try:
        conn.execute(f"ALTER TABLE {SYNC} RENAME TO {SYNC}_V3")
        total = conn.execute(f"SELECT COUNT(*) FROM {SYNC}_V3").fetchone()[0]
        conn.execute(
            f"CREATE TABLE {SYNC}_V2 AS SELECT * FROM {SYNC}_V3 LIMIT {total // 2}"
        )
        conn.commit()
        kept = conn.execute(f"SELECT COUNT(*) FROM {SYNC}_V2").fetchone()[0]
    finally:
        conn.close()
    if not 0 < kept < total:
        raise ValueError(f"dual-variant fixture is not distinguishable: {kept}/{total}")
    return dst


def test_the_newest_variant_is_the_one_resolved(dual_variant):
    conn = sqlite3.connect(dual_variant)
    try:
        tables = wrap_connection(conn).resolve_activity_tables()
    finally:
        conn.close()

    assert tables["sync"] == SYNC + "_V3"


def test_both_engines_pick_the_same_variant(dual_variant):
    """The reason the three fixes are one PR.

    Deleting ``sync_cost_analysis``'s private resolver moves the skill onto
    ``_find_activity_tables``. Left with its old ``sorted()[0]`` ordering that
    silently changes the skill's answer from the newest table to the oldest one;
    and if only the query-side ordering is fixed, the cache builder still copies
    the oldest, so the two engines disagree. This asserts the state in which
    neither is true.
    """
    from nsys_ai.profile import Profile

    pytest.importorskip("duckdb", reason="requires duckdb")

    skill = get_skill("sync_cost_analysis")

    conn = sqlite3.connect(dual_variant)
    try:
        via_sqlite = skill.execute(conn)
    finally:
        conn.close()

    with Profile(str(dual_variant)) as prof:
        if prof.db is None:  # pragma: no cover
            pytest.skip("requires duckdb")
        via_duckdb = skill.execute(prof.db)

    assert "error" not in via_sqlite[0], via_sqlite[0]["error"]
    assert via_sqlite == via_duckdb, "the two engines resolved different sync tables"


# ── The duplicate cache must not come back ──────────────────────────────────


def test_sync_cost_analysis_keeps_no_module_level_result_cache():
    """It used to hold a dict keyed on ``id(conn)`` that ``Profile.close()``
    never cleared, and it was the only skill cache handing out its stored list
    uncopied — ``profile_health_manifest`` embedded that dict by reference.
    ``Skill.execute``'s per-connection memo already does this job, with
    copy-on-store and copy-on-read.

    Asserted behaviourally rather than by attribute name: any per-connection
    memo, whatever it is called, has to grow a module-level container as it
    sees new connections. Two connections are held open at once so the second
    cannot reuse the first's ``id()``.
    """
    from nsys_ai.skills.builtins import sync_cost_analysis

    def container_sizes() -> dict[str, int]:
        return {
            name: len(value)
            for name, value in vars(sync_cost_analysis).items()
            if isinstance(value, (dict, list, set))
        }

    conn_a = sqlite3.connect(FIXTURE)
    conn_b = sqlite3.connect(FIXTURE)
    try:
        # Call the execute function directly: Skill.execute has its own
        # per-connection memo, which would mask a module-level one.
        sync_cost_analysis._execute_sync_analysis(conn_a)
        before = container_sizes()
        sync_cost_analysis._execute_sync_analysis(conn_b)
        after = container_sizes()
    finally:
        conn_a.close()
        conn_b.close()

    grew = {k: (before.get(k), v) for k, v in after.items() if before.get(k) != v}
    assert not grew, f"module-level result cache reintroduced: {grew}"
