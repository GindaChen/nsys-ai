"""`root_cause_matcher` must not treat "could not run" as evidence.

`_safe_execute` returns whatever the sub-skill returned, and every consumer
guards with `if <rows>:`. An abstention row is a non-empty list, so it passes
that guard and is read as data.

Nothing goes wrong today, and the reason is worth stating because it is not a
design: the only sub-skill that can abstain is `nvtx_layer_breakdown`, abstention
is exactly one row, and the layer analysers happen to sit behind
`len(layer_data) >= 2`. Add a second abstaining skill to a `len >= 1` path, or
give `abstain()` a detail row, and the accident stops holding.

`evidence_builder` already filters for this reason — its docstring names the same
hazard, that the safe skills are "safe only through unrelated guards ... which a
refactor could remove without anyone noticing". This puts `root_cause_matcher` on
the same footing.
"""

import sqlite3
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NO_NVTX = REPO / "tests" / "fixtures" / "healthy_1pct.sqlite"


@pytest.fixture
def conn():
    c = sqlite3.connect(NO_NVTX)
    try:
        yield c
    finally:
        c.close()


def test_the_fixture_really_has_no_nvtx(conn):
    """Guard the premise: without this the assertions below go vacuous."""
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert not any(t.startswith("NVTX_EVENTS") for t in tables)


def test_the_subskill_really_abstains(conn):
    """The other half of the premise: the skill must abstain, not return []."""
    from nsys_ai.skills.base import is_abstention
    from nsys_ai.skills.registry import get_skill

    rows = get_skill("nvtx_layer_breakdown").execute(conn)
    assert is_abstention(rows), f"expected an abstention row, got {rows[:1]}"


def test_safe_execute_does_not_pass_abstention_through(conn):
    """The assertion that fails without the guard."""
    from nsys_ai.skills.builtins.root_cause_matcher import _safe_execute

    rows = _safe_execute("nvtx_layer_breakdown", conn)

    assert rows == [], (
        f"_safe_execute returned an abstention row as if it were data; every "
        f"consumer guards with `if rows:` and a non-empty list passes that: {rows}"
    )


def test_matcher_reports_no_layer_findings_without_annotation(conn):
    """End to end: an unannotated profile must not yield layer-attributed causes."""
    from nsys_ai.skills.registry import get_skill

    findings = get_skill("root_cause_matcher").execute(conn)

    for f in findings:
        assert "_abstained" not in f, f"an abstention row was emitted as a finding: {f}"
        # Layer findings are the ones nvtx_layer_breakdown feeds; none can be
        # justified on a profile that carries no annotation at all.
        assert "layer" not in str(f.get("pattern", "")).lower(), (
            f"layer-attributed cause on an unannotated profile: {f}"
        )
