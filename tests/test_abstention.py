"""A skill that cannot run must say so, not vanish.

Silence has three possible meanings and callers could not tell them apart:
the skill ran and found nothing; the skill could not run because the profile
lacks a table it needs; or the skill raised and a caller swallowed it.

The last was the real state of affairs. `EvidenceBuilder` catches `Exception`
and logs, so on a profile captured without NVTX the four annotation-dependent
skills raised `no such table: NVTX_EVENTS` and simply disappeared from the
findings. A user reading that output sees no NVTX findings and no explanation,
which is indistinguishable from a clean bill of health — the failure mode a
verify-first tool can least afford.

`tests/fixtures/mock.sqlite` genuinely has no `NVTX_EVENTS` table, so it is the
no-NVTX negative fixture called for in #262; nothing had asserted against it.
"""

import sqlite3
from pathlib import Path

import pytest

from nsys_ai.skills.base import abstain
from nsys_ai.skills.registry import get_skill

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mock.sqlite"

# The skills that need NVTX annotation to say anything at all.
NVTX_DEPENDENT = [
    "iteration_timing",
    "nccl_compile_context_breakdown",
    "nvtx_kernel_map",
    "nvtx_layer_breakdown",
]


def test_the_fixture_really_has_no_nvtx():
    """Guard the premise: if NVTX_EVENTS ever appears here, these tests are vacuous."""
    conn = sqlite3.connect(FIXTURE)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert "NVTX_EVENTS" not in tables


@pytest.mark.parametrize("skill_name", NVTX_DEPENDENT)
def test_nvtx_skills_abstain_with_a_reason_instead_of_raising(skill_name):
    skill = get_skill(skill_name)
    assert skill is not None, f"{skill_name} is not registered"

    conn = sqlite3.connect(FIXTURE)
    try:
        rows = skill.execute(conn)  # must not raise
    finally:
        conn.close()

    assert rows, f"{skill_name} returned nothing — abstention must be visible"
    assert rows[0].get("_abstained") is True, (
        f"{skill_name} returned rows without the abstention marker: {rows[0]}"
    )
    reason = rows[0].get("reason", "")
    assert reason, f"{skill_name} abstained with no reason"
    # The reason has to be actionable, not merely present.
    assert "NVTX" in reason
    assert "re-capture" in reason.lower() or "annotate" in reason.lower()


@pytest.mark.parametrize("skill_name", NVTX_DEPENDENT)
def test_abstention_is_distinguishable_from_an_empty_result(skill_name):
    """The whole point: `[]` and "could not run" must not look alike."""
    conn = sqlite3.connect(FIXTURE)
    try:
        rows = skill.execute(conn) if (skill := get_skill(skill_name)) else []
    finally:
        conn.close()
    assert rows != [], f"{skill_name} is back to returning an ambiguous empty list"


def test_abstain_helper_shape():
    rows = abstain("because", table="NVTX_EVENTS")
    assert rows == [{"_abstained": True, "reason": "because", "table": "NVTX_EVENTS"}]


def test_evidence_builder_no_longer_loses_these_skills_to_an_exception():
    """The regression that motivated all of this.

    Before, these four raised and `EvidenceBuilder` caught and logged, so the
    skill left no trace. Abstaining keeps them in the pipeline, which is what
    lets a caller report "no NVTX annotation" instead of saying nothing.
    """
    conn = sqlite3.connect(FIXTURE)
    try:
        for name in NVTX_DEPENDENT:
            skill = get_skill(name)
            rows = skill.execute(conn)
            # A skill that abstains contributes no findings, but it does so
            # visibly — the rows survive to the caller.
            assert rows[0]["_abstained"] is True
            if skill.to_findings_fn:
                # Same invoker EvidenceBuilder uses: to_findings_fn signatures
                # vary across skills, and this is the path production takes.
                from nsys_ai.evidence_builder import _invoke_to_findings

                findings = _invoke_to_findings(
                    skill.to_findings_fn, rows, {"profile_id": "p"}
                )
                assert findings == [], (
                    f"{name} produced findings from an abstention row: {findings}"
                )
    finally:
        conn.close()
