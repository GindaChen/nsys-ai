"""Every parameterless builtin skill must report the same numbers on both engines.

A profile can be read two ways in the shipped artifact: through the columnar
cache, or through a plain ``sqlite3`` connection when the cache is unavailable
(``Profile`` logs "DuckDB cache unavailable or not in use; falling back to
SQLite (slower)" and carries on). A skill's SQL is written once and runs on
both, so any expression whose meaning depends on the engine silently produces
two different answers for the same profile — and the user has no way to tell
which one they got.

Two such expressions are easy to write and impossible to spot by reading:

* ``a / b`` where both operands are integers. SQLite truncates (``7/2`` is
  ``3``); the columnar engine promotes to a real quotient (``3.5``). A
  bytes-per-nanosecond ratio below 1 therefore reads as a plausible ``0.0``.
* ``CAST(<fractional> AS INT)``. SQLite truncates (``0.6`` -> ``0``); the
  columnar engine rounds to nearest (``0.6`` -> ``1``). Used to bucket a
  timestamp, that puts the same events in different buckets.

This sweep runs every skill that needs no required parameter against both
engines on the same profile. It compares row counts, then the numeric cells.
Both halves matter: eight of the thirty-one skills emit no numeric column at
all on these fixtures (``root_cause_matcher`` is all strings), so for those the
row count is the entire assertion.
"""

import math
import shutil
import sqlite3
from pathlib import Path

import pytest

from nsys_ai import profile as _profile
from nsys_ai.skills.base import is_abstention
from nsys_ai.skills.registry import all_skills, get_skill

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURES = ["mfu_2gpu_before.sqlite", "h100_2gpu_1s.sqlite"]

# Reports the tables and columns of whatever database it is handed. The two
# engines are pointed at genuinely different databases -- the profile file
# versus the cache's derived views -- so a differing row set is the correct
# answer on both sides, not a disagreement.
ENGINE_DEPENDENT_SKILLS = {"schema_inspect"}

# Tensor Core attribution is a capability of the cache's ``kernels`` view,
# which synthesises ``is_tc_eligible`` / ``uses_tc`` by matching kernel names.
# A raw sqlite3 connection has no such columns, so ``tensor_core_usage``
# abstains outright there. Teaching the SQLite path to classify kernel names is
# a feature, not an arithmetic fix, so these cells stay out of the comparison.
#
# ``tc_achieved_pct`` joined that category rather than leaving it. It used to
# read 0.0 on sqlite3 against 100.0 on the cache -- a wrong answer, not an
# unavailable one, because the missing capability was summed as zero (#418).
# With that fixed the sqlite3 side reports None, and the divergence is now the
# honest one this list is for: one engine measures, the other says it cannot.
CAPABILITY_COLUMNS = frozenset(
    {
        "is_tc_eligible",
        "uses_tc",
        "tc_eligible",
        "tc_achieved_pct",
        "tc_calls_pct",
        "tc_active_calls",
        "tc_active_ms",
    }
)


def _parameterless_skill_names() -> list[str]:
    """Skills runnable with no arguments at all, in a stable order.

    ``cutracer_analysis`` is excluded separately: its ``trace_dir`` is not
    marked required, but it addresses an external trace directory rather than
    the profile, so there is nothing engine-specific to compare.
    """
    names = [
        s.name
        for s in all_skills()
        if not any(getattr(p, "required", False) for p in (s.params or []))
        and s.name != "cutracer_analysis"
    ]
    return sorted(names)


SKILL_NAMES = _parameterless_skill_names()


@pytest.fixture(scope="module")
def engines(tmp_path_factory):
    """One sqlite3 connection and one cache-backed connection per fixture.

    The profile is copied first: opening one builds a cache beside it and adds
    indexes to the file, and the fixtures are committed to the repository.

    Module-scoped because building the cache costs more than every skill in
    the sweep put together, and because skill results are memoized per
    connection -- reusing the connections is what keeps this test cheap.
    """
    work = tmp_path_factory.mktemp("engine_parity")
    opened = {}
    for name in FIXTURES:
        dst = work / name
        shutil.copy(FIXTURE_DIR / name, dst)
        prof = _profile.open(str(dst))
        cache_conn = prof.query_conn()
        if isinstance(cache_conn, sqlite3.Connection):
            # Skip, loudly, with a reason test_ci_coverage already accepts.
            # Without the check the sweep degrades to sqlite3-vs-sqlite3 and
            # still reports every case green: Profile falls back to self.conn
            # when both the cache build and the direct attach fail, and
            # query_conn hands that same connection back. Comparing a thing
            # with itself is the one result this file must never produce
            # quietly.
            pytest.skip("parquet cache unavailable")
        opened[name] = (sqlite3.connect(str(dst)), prof)
    yield {name: (lite, prof.query_conn()) for name, (lite, prof) in opened.items()}
    for lite, prof in opened.values():
        lite.close()
        prof.close()


def _declined(rows) -> bool:
    """True when a skill said it could not answer, by either encoding.

    ``is_abstention`` is the contract (``skills/base.py``). Three skills still
    decline with an untyped ``{"error": "..."}`` row that predates it --
    ``nccl_payload_breakdown``, ``tensor_core_usage`` and
    ``nccl_compile_context_breakdown``; see issue #400, which is converging
    those onto the marker. Recognising both here rather than re-deriving the
    marker keeps this sweep from comparing a refusal against a result.
    """
    if is_abstention(rows):
        return True
    return (
        isinstance(rows, list)
        and len(rows) == 1
        and isinstance(rows[0], dict)
        and isinstance(rows[0].get("error"), str)
        and bool(rows[0]["error"].strip())
    )


def _numeric_cells(rows) -> dict[tuple[int, str], float]:
    """Every comparable numeric cell, keyed by (row index, column).

    Booleans are excluded -- they are flags, not measurements, and Python
    treats them as ints. So are the capability columns above.
    """
    cells: dict[tuple[int, str], float] = {}
    if not isinstance(rows, list):
        return cells
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        for column, value in row.items():
            if column in CAPABILITY_COLUMNS:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            cells[(i, column)] = value
    return cells


@pytest.mark.parametrize("fixture_name", FIXTURES)
@pytest.mark.parametrize("skill_name", SKILL_NAMES)
def test_skill_agrees_across_engines(engines, fixture_name, skill_name):
    """The same skill on the same profile must give the same numbers either way."""
    if skill_name in ENGINE_DEPENDENT_SKILLS:
        # Return rather than skip. This is not coverage we are losing and would
        # like back -- these skills describe whichever database they are handed,
        # so cross-engine equality is not a property they are meant to have. A
        # skip would also need registering in test_ci_coverage's accepted list,
        # which exists to stop real coverage going quiet.
        return

    lite_conn, cache_conn = engines[fixture_name]
    skill = get_skill(skill_name)
    lite_rows = skill.execute(lite_conn)
    cache_rows = skill.execute(cache_conn)

    # A skill may be able to answer on one engine and not the other -- the
    # cache exposes derived columns a raw connection does not have. What it
    # must not do is answer anyway with a made-up number, so a stated refusal
    # on either side leaves nothing to compare.
    if _declined(lite_rows) or _declined(cache_rows):
        return

    # Before the cells, the rows. Eight skills emit no numeric column at all,
    # and for those every assertion below is vacuous -- a skill could return
    # fifty rows on one engine and seven on the other and still pass.
    assert len(lite_rows) == len(cache_rows), (
        f"{skill_name} on {fixture_name}: {len(lite_rows)} rows via sqlite3 "
        f"but {len(cache_rows)} via the cache"
    )

    lite_cells = _numeric_cells(lite_rows)
    cache_cells = _numeric_cells(cache_rows)

    missing = sorted(set(lite_cells) - set(cache_cells))
    extra = sorted(set(cache_cells) - set(lite_cells))
    assert not missing and not extra, (
        f"{skill_name} on {fixture_name}: the two engines produced different "
        f"result shapes ({len(lite_rows)} rows via sqlite3, {len(cache_rows)} "
        f"via the cache). Only in sqlite3: {missing[:8]}. "
        f"Only in the cache: {extra[:8]}."
    )

    disagreements = [
        (key, lite_cells[key], cache_cells[key])
        for key in sorted(lite_cells)
        if not math.isclose(lite_cells[key], cache_cells[key], rel_tol=1e-9, abs_tol=1e-12)
    ]
    assert not disagreements, (
        f"{skill_name} on {fixture_name}: {len(disagreements)} numeric cells "
        f"differ between engines (row, column, sqlite3, cache): "
        f"{disagreements[:8]}"
    )


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_h2d_distribution_reports_a_real_rate_on_sqlite(engines, fixture_name):
    """The H2D rate is a measurement on both engines, never a truncated zero.

    ``avg_gbps`` divides summed bytes by summed nanoseconds. Both sums are
    integers, so on SQLite an unpromoted division truncated every rate below
    1 GB/s to exactly 0.0 -- reporting "no throughput" for a transfer that ran.
    """
    lite_conn, cache_conn = engines[fixture_name]
    skill = get_skill("h2d_distribution")
    lite = [r for r in skill.execute(lite_conn) if not r.get("_pattern")]
    cache = [r for r in skill.execute(cache_conn) if not r.get("_pattern")]

    assert lite, "fixture has no H2D transfers to measure"
    assert len(lite) == len(cache), (
        f"engines bucketed the same transfers differently: "
        f"{[r['second'] for r in lite]} via sqlite3 vs "
        f"{[r['second'] for r in cache]} via the cache"
    )
    for lite_row, cache_row in zip(lite, cache):
        assert lite_row["second"] == cache_row["second"]
        assert lite_row["ops"] == cache_row["ops"]
        assert lite_row["avg_gbps"] > 0.0, (
            f"second {lite_row['second']} moved {lite_row['total_mb']} MB "
            f"but reported 0 GB/s on the sqlite3 path"
        )
        assert math.isclose(lite_row["avg_gbps"], cache_row["avg_gbps"], rel_tol=1e-9)
