"""Regression findings minted from a baseline diff."""

from __future__ import annotations

import io
import json
from pathlib import Path

from nsys_ai.annotation import EvidenceReport
from nsys_ai.diagnose_command import run_diagnose
from nsys_ai.diff import diff_profiles
from nsys_ai.diff_findings import findings_from_diff
from nsys_ai.profile import Profile
from nsys_ai.profile_runner import build_local_profile_reference
from nsys_ai.proposal import generate_proposal
from nsys_ai.runspec import RunSpec
from nsys_ai.session_cli import publish_session_findings, publish_session_proposal
from nsys_ai.session_store import SessionStore

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / "tests" / "fixtures" / "mfu_2gpu_before.sqlite"
AFTER = ROOT / "tests" / "fixtures" / "mfu_2gpu_after.sqlite"


def _summary(*, before: Path = AFTER, after: Path = BEFORE, gpu: int | None = None):
    with Profile(before) as before_prof, Profile(after) as after_prof:
        return diff_profiles(before_prof, after_prof, gpu=gpu)


def test_reversed_pair_mints_ranked_candidate_findings():
    summary = _summary()
    findings = findings_from_diff(summary)

    assert len(findings) == 15
    assert [finding.headroom_ms for finding in findings] == sorted(
        (finding.headroom_ms for finding in findings), reverse=True
    )
    assert all(finding.headroom_ms > 0 for finding in findings)
    assert all(finding.selection.profile_id == summary.after.profile_id for finding in findings)
    assert all(finding.diff_lineage.diff_id == summary.diff_id for finding in findings)
    assert all(
        finding.diff_lineage.baseline_profile_id == summary.before.profile_id
        for finding in findings
    )
    assert [finding.diff_lineage.rank for finding in findings] == list(range(15))
    assert len({finding.id for finding in findings}) == len(findings)
    assert all(finding.suggested_actions for finding in findings)
    assert all(finding.selection.source == "diff" for finding in findings)


def test_call_count_regression_action_describes_frequency_not_slowdown():
    findings = findings_from_diff(_summary())
    count_only = next(
        finding
        for finding in findings
        if "call count rose" in finding.suggested_actions[0]
    )

    assert "call count rose" in count_only.suggested_actions[0]
    assert "got slower" not in count_only.suggested_actions[0]
    proposal = generate_proposal(count_only, RunSpec(argv=("true",)))
    assert proposal.abstained is False
    assert proposal.expected_impact is not None
    assert proposal.expected_impact.headroom_basis == "baseline_delta"


def test_forward_gpu_pair_without_regression_returns_empty():
    summary = _summary(before=BEFORE, after=AFTER, gpu=0)

    assert summary.top_regressions == []
    assert findings_from_diff(summary) == []


def test_diff_findings_never_mint_a_whole_run_finding():
    findings = findings_from_diff(_summary())

    assert findings
    assert all(finding.selection is not None for finding in findings)
    assert all(finding.diff_lineage.role == "regression" for finding in findings)
    assert all(finding.diff_lineage.rank >= 0 for finding in findings)
    assert all(finding.headroom_basis == "baseline_delta" for finding in findings)
    assert all(finding.end_ns is None or finding.end_ns >= finding.start_ns for finding in findings)


def test_candidate_session_accepts_diff_finding_and_reaches_propose(tmp_path: Path):
    summary = _summary()
    finding = findings_from_diff(summary)[0]
    candidate = build_local_profile_reference(summary.after.path)
    report = EvidenceReport(
        title="Diff-seeded diagnosis",
        profile_path=candidate.path,
        profile_id=candidate.profile_id,
        findings=[finding],
    )
    sessions = tmp_path / "sessions"

    state = publish_session_findings(
        session_id="candidate-diff",
        report=report,
        before_profile=candidate,
        root=sessions,
    )
    assert state.phase == "diagnose"

    proposal = generate_proposal(finding, RunSpec(argv=("true",)))
    state = publish_session_proposal(
        session_id="candidate-diff",
        proposal=proposal,
        runspec=proposal.verification,
        root=sessions,
    )

    assert state.phase == "propose"
    snapshot = SessionStore(sessions).load("candidate-diff")
    assert snapshot.state.before_profile == candidate
    assert snapshot.findings is not None
    assert snapshot.findings.findings[0].selection.profile_id == candidate.profile_id


def test_diagnose_against_seeds_default_session_from_diff_id(tmp_path: Path):
    stdout = io.StringIO()
    result = run_diagnose(
        profile_path=BEFORE,
        against=AFTER,
        session_root=tmp_path / "sessions",
        format="json",
        stdout=stdout,
        open_browser=False,
    )

    assert result == 0
    payload = json.loads(stdout.getvalue())
    diff_findings = [item for item in payload["findings"] if "diff_lineage" in item]
    assert diff_findings
    diff_id = diff_findings[0]["diff_lineage"]["diff_id"]
    session_id = f"diff_{diff_id.replace(':', '_')}"
    session = SessionStore(tmp_path / "sessions").load(session_id)
    assert session.state.before_profile.profile_id == payload["profile_id"]
    assert session.findings is not None
    assert session.findings.findings[0].selection.profile_id == payload["profile_id"]
