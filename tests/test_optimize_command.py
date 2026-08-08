"""Behaviour of the ``nsys-ai optimize`` front door.

These tests drive the real session store, the real proposal generator and the
real diff against committed fixture profiles. Only the two things that need a
GPU are substituted: the nsys capability probe and the local runner. Nothing
here asserts on analysis numbers — that is covered where the analysis lives.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from nsys_ai.annotation import EvidenceReport, Finding, TraceSelection
from nsys_ai.optimize_command import (
    OptimizeCommandError,
    _decision_for_verdict,
    run_optimize,
)
from nsys_ai.profile_runner import (
    RunResult,
    RunStatus,
    RunTimings,
    build_local_profile_reference,
)
from nsys_ai.session_cli import publish_session_findings
from nsys_ai.session_store import SessionStore

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / "tests" / "fixtures" / "mfu_2gpu_before.sqlite"
AFTER = ROOT / "tests" / "fixtures" / "mfu_2gpu_after.sqlite"

SESSION_ID = "optimize-test"


def _session_root(tmp_path: Path) -> Path:
    return tmp_path / ".nsys-ai" / "sessions"


def _timings() -> RunTimings:
    return RunTimings(
        started_at_utc="2026-01-01T00:00:00+00:00",
        capture_seconds=0.0,
        export_seconds=0.0,
        validation_seconds=0.0,
        total_seconds=0.0,
    )


class _StubRunner:
    """Stands in for LocalProfileRunner: records the spec, returns a result."""

    def __init__(self, result_factory):
        self._result_factory = result_factory
        self.specs = []

    def __call__(self, artifact_dir, nsys_executable):
        self.artifact_dir = artifact_dir
        self.nsys_executable = nsys_executable
        return self

    def run(self, spec, progress_callback=None, cancellation=None):
        self.specs.append(spec)
        return self._result_factory()


def _succeeding_runner(after_reference):
    return _StubRunner(
        lambda: RunResult(
            status=RunStatus.SUCCEEDED,
            nsys_return_code=0,
            application_return_code=None,
            timings=_timings(),
            runspec_path=None,
            stdout_path=None,
            stderr_path=None,
            report_path=None,
            sqlite_path=after_reference.path,
            profile=after_reference,
        )
    )


def _failing_runner():
    return _StubRunner(
        lambda: RunResult(
            status=RunStatus.APPLICATION_FAILED,
            nsys_return_code=1,
            application_return_code=None,
            timings=_timings(),
            runspec_path=None,
            stdout_path=None,
            stderr_path=None,
            report_path=None,
            sqlite_path=None,
            profile=None,
            detail="workload exited 1",
        )
    )


def _exploding_runner():
    def _boom(artifact_dir, nsys_executable):
        raise AssertionError("optimize re-profiled after it should have stopped")

    return _boom


def _seed_findings(tmp_path: Path, *, actionable: bool = True, with_id: bool = True):
    """Publish one hand-built finding so the tests control the propose outcome."""
    reference = build_local_profile_reference(BEFORE)
    selection = TraceSelection(
        id="optimize-test-selection",
        profile_id=reference.profile_id,
        source="skill:top_kernels",
        start_ns=0,
        end_ns=1_000_000,
    )
    finding = Finding(
        type="highlight",
        label="Hotspot: fixture kernel",
        start_ns=0,
        end_ns=1_000_000,
        severity="warning",
        id="optimize-test-finding" if with_id else None,
        selection=selection,
        explanation="One fixture kernel dominates the selected window.",
        suggested_actions=["Fuse the kernel with its neighbour"] if actionable else None,
    )
    report = EvidenceReport(
        "optimize fixture",
        str(reference.path),
        [finding],
        profile_id=reference.profile_id,
    )
    publish_session_findings(
        session_id=SESSION_ID,
        report=report,
        before_profile=reference,
        root=_session_root(tmp_path),
    )
    return reference


@pytest.fixture(autouse=True)
def _stub_nsys_probe(monkeypatch):
    """Only the GPU-bound parts are stubbed; the RunSpec is built for real."""
    monkeypatch.setattr(
        "nsys_ai.optimize_command.resolve_nsys_executable", lambda selected: "/usr/bin/nsys"
    )
    monkeypatch.setattr(
        "nsys_ai.optimize_command.detect_supported_trace_domains",
        lambda executable: ("cuda", "nvtx", "nccl"),
    )


def _run(tmp_path, runner_factory, **overrides):
    stdout, stderr = io.StringIO(), io.StringIO()
    code = run_optimize(
        before_path=BEFORE,
        repo=tmp_path,
        workload=["./axpy", "20"],
        session_id=SESSION_ID,
        trim=None,
        session_root=_session_root(tmp_path),
        cwd=tmp_path,
        runner_factory=runner_factory,
        stdout=stdout,
        stderr=stderr,
        **overrides,
    )
    return code, stdout.getvalue(), stderr.getvalue()


def _snapshot(tmp_path: Path):
    return SessionStore(_session_root(tmp_path)).load(SESSION_ID)


def test_optimize_reaches_a_recorded_decision_without_a_supplied_after_profile(tmp_path):
    """The whole loop runs from a baseline profile and ends at a decision."""
    _seed_findings(tmp_path)
    after_reference = build_local_profile_reference(AFTER)
    runner = _succeeding_runner(after_reference)

    code, out, err = _run(tmp_path, runner)

    assert code == 0, err
    assert len(runner.specs) == 1
    # The executed RunSpec is the recorded one, not a fresh guess.
    assert runner.specs[0] == _snapshot(tmp_path).runspec
    # The plan is visible before the capture, and the verdict after it.
    assert "-- Verification RunSpec (about to execute) --" in out
    assert out.index("-- Proposal --") < out.index("Profile Diff")

    snapshot = _snapshot(tmp_path)
    assert snapshot.state.phase == "accept"
    assert snapshot.state.after_profile == after_reference
    decision = snapshot.diff["decision"]
    assert decision["status"] in {"accepted", "rejected"}
    assert decision["reason"].startswith("diff verdict ")
    payload = json.loads(
        (_session_root(tmp_path) / SESSION_ID / "diff.json").read_text(encoding="utf-8")
    )
    assert payload["decision"]["reason"] == decision["reason"]


def test_optimize_reports_an_already_decided_session_without_re_profiling(tmp_path):
    """Re-running a finished loop must report, not re-capture."""
    _seed_findings(tmp_path)
    after_reference = build_local_profile_reference(AFTER)
    first, _out, err = _run(tmp_path, _succeeding_runner(after_reference))
    assert first == 0, err

    code, out, _err = _run(tmp_path, _exploding_runner())

    assert code == 0
    assert "Loop already complete" in out
    assert "Decision: " in out


def test_optimize_stops_before_reprofile_when_the_proposal_abstains(tmp_path):
    """A finding with no suggested action cannot produce a proposal to verify."""
    _seed_findings(tmp_path, actionable=False)

    code, out, err = _run(tmp_path, _exploding_runner())

    assert code == 2
    assert "Abstained before re-profiling: source finding has no suggested action" in out
    assert "nothing was re-profiled and no verdict was reached" in err
    assert "Resume: nsys-ai optimize " in err

    snapshot = _snapshot(tmp_path)
    assert snapshot.proposal is not None
    assert snapshot.proposal.abstained is True
    assert snapshot.state.after_profile is None
    assert snapshot.diff is None
    # Nothing was captured, so no capture artifacts were created either.
    assert not (tmp_path / ".nsys-ai" / "profiles").exists()


def test_optimize_stops_when_no_finding_carries_an_id(tmp_path):
    """Without a finding id there is no input for a proposal at all."""
    _seed_findings(tmp_path, with_id=False)

    code, _out, err = _run(tmp_path, _exploding_runner())

    assert code == 2
    assert "no finding carries a stable id" in err
    assert _snapshot(tmp_path).proposal is None


def test_optimize_resumes_from_the_session_after_a_failed_capture(tmp_path):
    """A failed capture keeps diagnose and propose; the retry only re-captures."""
    _seed_findings(tmp_path)

    failed_code, _out, err = _run(tmp_path, _failing_runner())
    assert failed_code == 1
    assert "Verification capture failed: workload exited 1" in err
    interrupted = _snapshot(tmp_path)
    assert interrupted.proposal is not None
    assert interrupted.state.after_profile is None

    after_reference = build_local_profile_reference(AFTER)
    runner = _succeeding_runner(after_reference)
    code, _out2, err2 = _run(tmp_path, runner)

    assert code == 0, err2
    assert "Step 1/5 diagnose: reusing 1 findings" in err2
    assert "Step 2/5 propose" not in err2
    assert len(runner.specs) == 1
    assert runner.specs[0] == interrupted.runspec
    assert _snapshot(tmp_path).state.phase == "accept"


def test_optimize_refuses_a_session_opened_on_another_profile(tmp_path):
    """The session id is the loop's identity; silently rebinding it would lie."""
    _seed_findings(tmp_path)
    with pytest.raises(OptimizeCommandError, match="different before profile"):
        run_optimize(
            before_path=AFTER,
            repo=tmp_path,
            workload=["./axpy"],
            session_id=SESSION_ID,
            session_root=_session_root(tmp_path),
            cwd=tmp_path,
            runner_factory=_exploding_runner(),
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )


def test_optimize_rejects_a_repo_that_is_not_a_directory(tmp_path):
    """--repo names the directory the verification run executes in."""
    missing = tmp_path / "not-a-repo"
    with pytest.raises(OptimizeCommandError, match="--repo is not a directory"):
        run_optimize(
            before_path=BEFORE,
            repo=missing,
            workload=["./axpy"],
            session_id=SESSION_ID,
            session_root=_session_root(tmp_path),
            cwd=tmp_path,
            runner_factory=_exploding_runner(),
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("improvement_likely", "accept"),
        ("regression_likely", "reject"),
        ("neutral", "reject"),
        ("inconclusive", "reject"),
    ],
)
def test_decision_reason_names_the_verdict_it_came_from(verdict, expected):
    """Only a measured improvement is accepted, and the reason says why."""
    decision, reason = _decision_for_verdict(verdict, -3.5)
    assert decision == expected
    assert verdict in reason
    assert "-3.50%" in reason


def test_decision_reason_is_explicit_when_step_time_is_not_comparable():
    _decision, reason = _decision_for_verdict("inconclusive", None)
    assert "step time was not comparable" in reason
