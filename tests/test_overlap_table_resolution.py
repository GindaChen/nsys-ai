"""``overlap.py`` must resolve Nsight table variants the way everyone else does.

Three sites in ``overlap.py`` scanned ``prof.schema.tables`` themselves and took
``sorted(...)[0]`` — the *oldest* variant — which is the ordering the shared
resolver was introduced to replace. On a profile carrying both ``_V2`` and
``_V3`` the two engines then answered differently for the same skill on the same
profile:

* ``launch_overhead_ms`` picked ``CUPTI_ACTIVITY_KIND_RUNTIME_V2``;
* ``detect_iterations`` picked ``NVTX_EVENTS_V2`` *and* ``..._RUNTIME_V2``.

Two more sites had the same shape one layer out:
``profile_health_manifest``'s raw-SQLite NVTX fallback, and
``NsightSchema._detect_kernel_table`` — which supplies the *other* table in
``launch_overhead_ms``'s own join, so fixing only the three above left that one
statement reading a ``_V3`` runtime table against a ``_V2`` kernel table. The
rule these tests enforce is that no site outside the shared resolver orders
variants itself.

Only the raw-SQLite engine was affected. Both DuckDB paths (``open_cached_db``
and ``open_direct_sqlite``) create an unversioned alias view pointing at the
*resolver-chosen* table, so ``overlap.py``'s "prefer the unversioned name" first
branch landed on the right data there by accident. That is not a corner case:
thirteen builtin skill modules build a Profile with
``Profile._from_conn(<connection>)``, and ``Skill.execute(sqlite_conn)`` is a
first-class engine here — ``test_sync_table_resolution.py`` exercises it
directly.

The same commit that unified the resolver also claimed
``CUPTI_ACTIVITY_KIND_MEMCPY2`` (peer-to-peer memcpy — a distinct CUPTI activity
kind, not a version) could never be read as a memcpy variant. It could, twice
over: the query-side resolver's permissive tier 3 returned it when it was the
only memcpy-prefixed table, and two skills never called the resolver at all and
picked whichever memcpy-prefixed name came first out of a *set*. The tests at
the bottom pin both halves, and also pin what each of the three memcpy readers
then *does* with a profile the resolver declines to answer for — two abstain,
one carries on without a memcpy category. Declining to resolve is only half an
answer; the readers used to raise or go quiet, which is what the resolver's
docstring had claimed they did not do.

Every real capture available here carries exactly one variant per table and none
carries ``..._MEMCPY2``, so these cases have to be constructed rather than taken
from a fixture.
"""

import shutil
import sqlite3
from pathlib import Path

import pytest

from nsys_ai.connection import _find_activity_tables
from nsys_ai.overlap import detect_iterations, launch_overhead_ms
from nsys_ai.parquet_cache import _find_table
from nsys_ai.profile import Profile
from nsys_ai.skills.registry import get_skill

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"

NVTX = "NVTX_EVENTS"
RUNTIME = "CUPTI_ACTIVITY_KIND_RUNTIME"
MEMCPY = "CUPTI_ACTIVITY_KIND_MEMCPY"
KERNEL = "CUPTI_ACTIVITY_KIND_KERNEL"

# How much of each table the stale ``_V2`` leftover keeps. Deliberately not a
# half: at 50% (and above) this fixture's truncation no longer removes the rows
# that decide either answer, so a test built on it would pass whether or not the
# resolution is fixed. Measured on this fixture, the answers diverge from the
# complete table at 25% and coincide at 50%.
_KEEP_NUMERATOR, _KEEP_DENOMINATOR = 1, 4


def _truncated_count(conn: sqlite3.Connection, table: str) -> int:
    total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
    kept = total * _KEEP_NUMERATOR // _KEEP_DENOMINATOR
    if not 0 < kept < total:
        raise ValueError(f"{table} is not distinguishable when truncated: {kept}/{total}")
    return kept


def _build(dst: Path, mode: str, bases: tuple[str, ...] = (NVTX, RUNTIME)) -> Path:
    """Copy the fixture and rewrite each of ``bases`` into ``mode``.

    ``v3_only``   — the complete tables, renamed to ``_V3``. The control.
    ``dual``      — ``_V3`` complete, plus a truncated ``_V2`` alongside it.
    ``v3_stale``  — only ``_V3``, holding the *truncated* rows. The premise
                    guard: this is what a reader that picked ``_V2`` sees.

    ``bases`` defaults to the two tables ``overlap.py`` resolves for itself, so
    the tests of *its* three sites are not also exercising the kernel-table
    resolution one layer down; the fixture that versions the kernel table too
    passes ``bases`` explicitly.
    """
    shutil.copy(FIXTURE, dst)
    conn = sqlite3.connect(dst)
    try:
        for base in bases:
            kept = _truncated_count(conn, base)
            if mode == "v3_only":
                conn.execute(f"ALTER TABLE {base} RENAME TO {base}_V3")
            elif mode == "dual":
                conn.execute(f"ALTER TABLE {base} RENAME TO {base}_V3")
                conn.execute(
                    f"CREATE TABLE {base}_V2 AS "  # noqa: S608
                    f"SELECT * FROM {base}_V3 ORDER BY start LIMIT {kept}"
                )
            elif mode == "v3_stale":
                conn.execute(
                    f"CREATE TABLE {base}_stale AS "  # noqa: S608
                    f"SELECT * FROM {base} ORDER BY start LIMIT {kept}"
                )
                conn.execute(f"DROP TABLE {base}")
                conn.execute(f"ALTER TABLE {base}_stale RENAME TO {base}_V3")
            else:  # pragma: no cover - programming error
                raise ValueError(f"unknown mode {mode!r}")
        conn.commit()
    finally:
        conn.close()
    return dst


@pytest.fixture(scope="module")
def v3_only(tmp_path_factory) -> Path:
    return _build(tmp_path_factory.mktemp("v3_only") / "v3_only.sqlite", "v3_only")


@pytest.fixture(scope="module")
def dual_variant(tmp_path_factory) -> Path:
    return _build(tmp_path_factory.mktemp("dual") / "dual.sqlite", "dual")


@pytest.fixture(scope="module")
def v3_stale(tmp_path_factory) -> Path:
    return _build(tmp_path_factory.mktemp("stale") / "stale.sqlite", "v3_stale")


@pytest.fixture(scope="module")
def dual_variant_all(tmp_path_factory) -> Path:
    """Every activity table this file touches doubled, kernel table included."""
    return _build(
        tmp_path_factory.mktemp("dual_all") / "dual_all.sqlite", "dual", (KERNEL, NVTX, RUNTIME)
    )


@pytest.fixture(scope="module")
def v3_only_all(tmp_path_factory) -> Path:
    return _build(
        tmp_path_factory.mktemp("v3_all") / "v3_all.sqlite", "v3_only", (KERNEL, NVTX, RUNTIME)
    )


def _overlap_answers(path: Path) -> tuple[float, int]:
    """``launch_overhead_ms`` and the iteration count, over raw SQLite.

    ``Profile._from_conn`` on a ``sqlite3.Connection`` leaves ``prof.db`` None,
    which is the engine with no alias-view layer to paper over the resolution.
    """
    conn = sqlite3.connect(path)
    try:
        prof = Profile._from_conn(conn)
        return launch_overhead_ms(prof, 0), len(detect_iterations(prof, 0))
    finally:
        conn.close()


# ── The premise ─────────────────────────────────────────────────────────────


def test_the_two_variants_give_different_answers(v3_only, v3_stale):
    """Without this, the assertions below could pass on any resolution at all.

    The stale profile carries exactly what a reader landing on ``_V2`` would
    see, so the gap measured here is the size of the bug.
    """
    assert _overlap_answers(v3_stale) != _overlap_answers(v3_only)


# ── The fix, at the overlap.py API ──────────────────────────────────────────


def test_overlap_reads_the_newest_variant_not_the_oldest(dual_variant, v3_only):
    """All three sites at once: ``launch_overhead_ms`` resolves the runtime
    table, ``detect_iterations`` resolves both NVTX and runtime.

    A stale ``_V2`` sitting next to the real ``_V3`` must not change either
    answer. Before the fix this returned the truncated profile's numbers.
    """
    assert _overlap_answers(dual_variant) == _overlap_answers(v3_only)


def test_iteration_timing_agrees_across_engines(dual_variant):
    """The user-visible consequence, through a shipped skill.

    ``iteration_timing`` goes through ``detect_iterations``. The DuckDB path was
    always right — its alias views point at the resolved table — so before the
    fix the same skill on the same profile disagreed with itself depending on
    which connection it was handed (``is_real_iteration`` flipped).
    """
    pytest.importorskip("duckdb", reason="requires duckdb")

    skill = get_skill("iteration_timing")

    conn = sqlite3.connect(dual_variant)
    try:
        via_sqlite = skill.execute(conn)
    finally:
        conn.close()

    with Profile(str(dual_variant)) as prof:
        if prof.db is None:  # pragma: no cover
            pytest.skip("requires duckdb")
        via_duckdb = skill.execute(prof.db)

    assert via_sqlite == via_duckdb, "the two engines resolved different tables"


def test_the_kernel_table_is_resolved_the_same_way_as_everything_else(
    dual_variant_all, v3_only_all
):
    """A fifth site, and the one inside ``launch_overhead_ms``'s own FROM clause.

    ``NsightSchema._detect_kernel_table`` had the same "exact name, else
    ``sorted(candidates)[0]``" shape. Fixing the three ``overlap.py`` sites
    without it left the two halves of that function's join disagreeing:
    ``kernel_table`` came back ``..._KERNEL_V2`` while
    ``resolve_activity_tables()["kernel"]`` said ``_V3``. The result was not
    merely stale but empty — measured on this fixture, 0.0 ms and 0 iterations
    against the control's 0.05 ms and 28 — because a correlationId join across
    two differently-truncated tables matches almost nothing.
    """
    conn = sqlite3.connect(dual_variant_all)
    try:
        prof = Profile._from_conn(conn)
        assert prof.schema.kernel_table == KERNEL + "_V3"
        assert prof.adapter.resolve_activity_tables()["kernel"] == prof.schema.kernel_table
    finally:
        conn.close()

    assert _overlap_answers(dual_variant_all) == _overlap_answers(v3_only_all)


def test_the_manifest_auto_trim_reads_the_newest_nvtx_variant(dual_variant):
    """A fourth site with the same defect, one skill over.

    ``profile_health_manifest``'s raw-SQLite fallback had its own "exact name,
    else first prefix match in sorted order" scan, and its docstring listed
    ``_V2`` before ``_V3`` as if that were the preference. The rule is that no
    site outside the shared resolver orders variants itself.
    """
    from nsys_ai.skills.builtins.profile_health_manifest import (
        _resolve_nvtx_table_for_auto_trim,
    )

    conn = sqlite3.connect(dual_variant)
    try:
        prof = Profile._from_conn(conn)
        assert _resolve_nvtx_table_for_auto_trim(prof) == NVTX + "_V3"
    finally:
        conn.close()


# ── MEMCPY2 is not a memcpy variant, on every path that picks one ───────────


def test_a_memcpy2_only_profile_resolves_no_memcpy_table():
    """``CUPTI_ACTIVITY_KIND_MEMCPY2`` is peer-to-peer memcpy: a different
    activity kind with a different ``copyKind`` domain, not a newer
    ``..._MEMCPY``. Anchoring the ``_V<n>`` match kept the two apart only while
    a real memcpy table was also present; with ``..._MEMCPY2`` alone the
    query-side resolver's permissive tier 3 handed it back anyway.

    Resolving to nothing beats reporting P2P copies as ordinary H2D/D2H
    traffic. What each reader then *does* with "no memcpy table" is the
    reader's own contract, pinned by the two tests below — this one only says
    the resolver declines to answer.

    The tier-3 escape hatch for a non-numeric unknown suffix stays open; that
    is pinned by ``test_sync_table_resolution.py::
    test_an_unrecognised_suffix_still_resolves_to_something``, which asserts the
    same input this test would otherwise duplicate.
    """
    assert "memcpy" not in _find_activity_tables({MEMCPY + "2"})
    assert _find_table({MEMCPY + "2"}, MEMCPY) is None


@pytest.fixture(scope="module")
def memcpy2_only(tmp_path_factory) -> Path:
    """The fixture with its memcpy table renamed to ``..._MEMCPY2``.

    A profile that traced *only* peer-to-peer copies. Nothing here should read
    it as ordinary memcpy traffic.
    """
    dst = tmp_path_factory.mktemp("m2only") / "memcpy2_only.sqlite"
    shutil.copy(FIXTURE, dst)
    conn = sqlite3.connect(dst)
    try:
        conn.execute(f"ALTER TABLE {MEMCPY} RENAME TO {MEMCPY}2")
        conn.commit()
    finally:
        conn.close()
    return dst


@pytest.mark.parametrize("skill_name", ["memory_transfers", "memory_bandwidth"])
def test_the_memcpy_skills_abstain_on_a_memcpy2_only_profile(memcpy2_only, skill_name):
    """"Cannot run" has one spelling in this codebase, and it is ``abstain``.

    Declining to resolve ``..._MEMCPY2`` puts such a profile in the same class
    as one with no memcpy table at all, and neither of those two used to say so:
    ``memory_transfers`` is a SQL template, so the unresolved placeholder took
    the canonical literal and the query died on ``no such table`` — which
    ``EvidenceBuilder`` catches, so the skill vanished from the findings
    silently. ``memory_bandwidth`` returned ``[]``, which ``abstain``'s own
    docstring defines as "ran, nothing to report" and calls actively
    misleading. Measured on this profile before the guard: 1 raise, 1 ``[]``.
    """
    from nsys_ai.skills.base import is_abstention

    conn = sqlite3.connect(memcpy2_only)
    try:
        rows = get_skill(skill_name).execute(conn)
    finally:
        conn.close()

    assert is_abstention(rows), f"{skill_name} did not abstain: {rows[:1]}"
    assert MEMCPY in rows[0]["reason"], "the reason must name the table that is missing"


def test_the_overlap_matrix_reports_what_it_still_has(memcpy2_only, clean_copy):
    """The third memcpy reader, and the one that does *not* abstain.

    ``kernel_overlap_matrix`` takes memcpy as one input among several, so a
    profile without it is answerable from the kernel categories alone — an
    abstention would throw away the compute/comm matrix over a missing extra.
    It must still drop every ``memcpy_*`` category rather than fill them from
    the P2P table.
    """
    skill = get_skill("kernel_overlap_matrix")

    conn = sqlite3.connect(memcpy2_only)
    try:
        rows = skill.execute(conn)
    finally:
        conn.close()

    assert rows, "the matrix should still be computed from kernels alone"
    assert not any(
        r["category_a"].startswith("memcpy") or r["category_b"].startswith("memcpy") for r in rows
    ), "CUPTI_ACTIVITY_KIND_MEMCPY2 was read as memcpy"

    conn = sqlite3.connect(clean_copy)
    try:
        with_memcpy = skill.execute(conn)
    finally:
        conn.close()
    assert any(r["category_a"].startswith("memcpy") for r in with_memcpy), (
        "premise: the untouched fixture does produce memcpy categories, "
        "so their absence above means something"
    )


@pytest.fixture(scope="module")
def clean_copy(tmp_path_factory) -> Path:
    """The untouched fixture, copied. Analysis over a profile can leave indexes
    behind, so nothing here opens ``tests/fixtures`` writable."""
    dst = tmp_path_factory.mktemp("clean") / "clean.sqlite"
    shutil.copy(FIXTURE, dst)
    return dst


@pytest.fixture(scope="module")
def with_memcpy2(tmp_path_factory) -> Path:
    """The fixture plus a ``..._MEMCPY2`` table holding *different* rows.

    Its copies are relabelled P2P and confined to one device so a skill reading
    it instead of ``..._MEMCPY`` produces visibly different output rather than
    the same numbers by coincidence.
    """
    dst = tmp_path_factory.mktemp("memcpy2") / "memcpy2.sqlite"
    shutil.copy(FIXTURE, dst)
    conn = sqlite3.connect(dst)
    try:
        kept = _truncated_count(conn, MEMCPY)
        conn.execute(
            f"CREATE TABLE {MEMCPY}2 AS "  # noqa: S608
            f"SELECT * FROM {MEMCPY} ORDER BY start LIMIT {kept}"
        )
        conn.execute(f"UPDATE {MEMCPY}2 SET copyKind = 10, deviceId = 0")  # noqa: S608
        conn.commit()
    finally:
        conn.close()
    return dst


@pytest.mark.parametrize("skill_name", ["memory_bandwidth", "kernel_overlap_matrix"])
def test_memcpy_skills_ignore_the_p2p_table_when_the_real_one_is_present(
    with_memcpy2, clean_copy, skill_name
):
    """Both skills hand-rolled ``for t in prof.schema.tables: if
    t.startswith(...)``. ``NsightSchema.tables`` is ``list(<set>)``, so which of
    ``..._MEMCPY`` / ``..._MEMCPY2`` that loop found first was set-hash order —
    non-deterministic across processes, and wrong whenever it landed on P2P.
    """
    skill = get_skill(skill_name)

    conn = sqlite3.connect(with_memcpy2)
    try:
        polluted = skill.execute(conn)
    finally:
        conn.close()

    conn = sqlite3.connect(clean_copy)
    try:
        clean = skill.execute(conn)
    finally:
        conn.close()

    assert polluted == clean, f"{skill_name} read CUPTI_ACTIVITY_KIND_MEMCPY2"


def test_the_memcpy2_fixture_would_change_the_answer(with_memcpy2, clean_copy):
    """Premise for the two tests above: reading ``..._MEMCPY2`` really does
    produce a different result, so agreeing with the clean profile is evidence
    of the right table and not of an insensitive comparison."""
    skill = get_skill("memory_bandwidth")

    renamed = with_memcpy2.with_name("memcpy2_only.sqlite")
    shutil.copy(with_memcpy2, renamed)
    conn = sqlite3.connect(renamed)
    try:
        conn.execute(f"DROP TABLE {MEMCPY}")
        conn.execute(f"ALTER TABLE {MEMCPY}2 RENAME TO {MEMCPY}")
        conn.commit()
        via_p2p = skill.execute(conn)
    finally:
        conn.close()

    conn = sqlite3.connect(clean_copy)
    try:
        clean = skill.execute(conn)
    finally:
        conn.close()

    assert via_p2p != clean
