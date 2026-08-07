"""CLI → SessionStore publishing, including cross-process restart coverage."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from nsys_ai.runspec import EnvironmentSpec, RunSpec
from nsys_ai.session_cli import session_dir, session_id_from_profile_id
from nsys_ai.session_store import SessionStore

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / "tests" / "fixtures" / "mfu_2gpu_before.sqlite"
AFTER = ROOT / "tests" / "fixtures" / "mfu_2gpu_after.sqlite"


def _subprocess_environment(**extra: str) -> dict[str, str]:
    environment = dict(os.environ)
    source = str(ROOT / "src")
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not current else f"{source}{os.pathsep}{current}"
    environment.update(extra)
    return environment


def _run_cli(cwd: Path, *args: str, timeout: float = 120.0, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "nsys_ai", *args],
        cwd=cwd,
        env=_subprocess_environment(**env),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _finding_id_from_evidence_stdout(stdout: str) -> str:
    payload = json.loads(stdout)
    return next(
        finding["id"]
        for finding in payload["findings"]
        if finding.get("id") and finding.get("suggested_actions")
    )


def test_cli_session_artifacts_survive_a_second_process(tmp_path: Path):
    """First process(es) publish via CLI; a new interpreter reloads the session."""
    if not BEFORE.is_file() or not AFTER.is_file():
        raise FileNotFoundError(f"missing fixture profiles: {BEFORE} / {AFTER}")
    before = BEFORE.resolve()
    after = AFTER.resolve()
    session_id = "cli-session-restart"
    runspec_path = tmp_path / "runspec.json"
    runspec_path.write_bytes(RunSpec(argv=("true",)).canonical_json_bytes())

    evidence = _run_cli(
        tmp_path,
        "evidence",
        "build",
        str(before),
        "--format",
        "json",
        "--gpu",
        "0",
        "--session",
        session_id,
        "--analyzers",
        "overlap_ratio",
    )
    assert evidence.returncode == 0, evidence.stderr
    finding_id = _finding_id_from_evidence_stdout(evidence.stdout)

    propose = _run_cli(
        tmp_path,
        "propose",
        "--session",
        session_id,
        "--finding-id",
        finding_id,
        "--runspec",
        str(runspec_path),
    )
    assert propose.returncode == 0, propose.stderr
    assert "Abstained:" not in propose.stdout

    diff = _run_cli(
        tmp_path,
        "diff",
        str(before),
        str(after),
        "--gpu",
        "0",
        "--format",
        "json",
        "--no-ai",
        "--session",
        session_id,
    )
    assert diff.returncode == 0, diff.stderr

    directory = session_dir(session_id, root=tmp_path / ".nsys-ai" / "sessions")
    assert directory.is_dir()
    names = {path.name for path in directory.iterdir()}
    assert names == {
        "session.json",
        "runspec.json",
        "findings.json",
        "proposal.json",
        "diff.json",
        "logs",
    }

    script = """
import json, sys
from nsys_ai.session_store import SessionStore
snapshot = SessionStore(sys.argv[1]).load(sys.argv[2])
print(json.dumps({
    "phase": snapshot.state.phase,
    "before_id": snapshot.state.before_profile.profile_id,
    "after_id": snapshot.state.after_profile.profile_id,
    "findings_count": len(snapshot.findings.findings),
    "proposal_id": snapshot.proposal.proposal_id,
    "diff_verdict": snapshot.diff["verdict"],
    "decision": snapshot.diff.get("decision"),
}))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path / ".nsys-ai" / "sessions"),
            session_id,
        ],
        cwd=tmp_path,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    reloaded = json.loads(result.stdout)
    assert reloaded["phase"] == "diff"
    assert reloaded["before_id"]
    assert reloaded["after_id"]
    assert reloaded["findings_count"] >= 1
    assert reloaded["proposal_id"]
    assert reloaded["diff_verdict"]
    assert reloaded["decision"] is None

    # Same before profile derives the same session id spelling the CLI would use.
    store = SessionStore(tmp_path / ".nsys-ai" / "sessions")
    snapshot = store.load(session_id)
    assert session_id_from_profile_id(snapshot.state.before_profile.profile_id).startswith(
        "nsys2_sha256_"
    )


def test_partial_session_layouts_after_evidence_and_propose(tmp_path: Path):
    """Acceptance: evidence-only and propose-only layouts are exact subsets."""
    if not BEFORE.is_file():
        raise FileNotFoundError(f"missing fixture profile: {BEFORE}")
    before = BEFORE.resolve()
    session_id = "partial-layout"
    runspec_path = tmp_path / "runspec.json"
    runspec_path.write_bytes(RunSpec(argv=("true",)).canonical_json_bytes())

    evidence = _run_cli(
        tmp_path,
        "evidence",
        "build",
        str(before),
        "--format",
        "json",
        "--gpu",
        "0",
        "--session",
        session_id,
        "--analyzers",
        "overlap_ratio",
    )
    assert evidence.returncode == 0, evidence.stderr
    directory = session_dir(session_id, root=tmp_path / ".nsys-ai" / "sessions")
    assert {path.name for path in directory.iterdir()} == {
        "session.json",
        "findings.json",
        "logs",
    }

    finding_id = _finding_id_from_evidence_stdout(evidence.stdout)
    propose = _run_cli(
        tmp_path,
        "propose",
        "--session",
        session_id,
        "--finding-id",
        finding_id,
        "--runspec",
        str(runspec_path),
    )
    assert propose.returncode == 0, propose.stderr
    assert {path.name for path in directory.iterdir()} == {
        "session.json",
        "runspec.json",
        "findings.json",
        "proposal.json",
        "logs",
    }


def test_derived_session_id_round_trip_without_explicit_id(tmp_path: Path):
    """C1: evidence --session, propose --session --profile, diff --session share one dir."""
    if not BEFORE.is_file() or not AFTER.is_file():
        raise FileNotFoundError(f"missing fixture profiles: {BEFORE} / {AFTER}")
    before = BEFORE.resolve()
    after = AFTER.resolve()
    runspec_path = tmp_path / "runspec.json"
    runspec_path.write_bytes(RunSpec(argv=("true",)).canonical_json_bytes())

    evidence = _run_cli(
        tmp_path,
        "evidence",
        "build",
        str(before),
        "--format",
        "json",
        "--gpu",
        "0",
        "--session",
        "--analyzers",
        "overlap_ratio",
    )
    assert evidence.returncode == 0, evidence.stderr
    assert "Findings published to session " in evidence.stderr
    finding_id = _finding_id_from_evidence_stdout(evidence.stdout)

    propose = _run_cli(
        tmp_path,
        "propose",
        "--session",
        "--profile",
        str(before),
        "--finding-id",
        finding_id,
        "--runspec",
        str(runspec_path),
    )
    assert propose.returncode == 0, propose.stderr
    assert "Proposal published to session " in propose.stdout

    diff = _run_cli(
        tmp_path,
        "diff",
        str(before),
        str(after),
        "--gpu",
        "0",
        "--format",
        "json",
        "--no-ai",
        "--session",
    )
    assert diff.returncode == 0, diff.stderr
    assert "Diff published to session " in diff.stderr

    sessions_root = tmp_path / ".nsys-ai" / "sessions"
    session_ids = [path.name for path in sessions_root.iterdir() if path.is_dir()]
    assert len(session_ids) == 1
    session_id = session_ids[0]
    assert session_id.startswith("nsys2_sha256_")
    names = {path.name for path in session_dir(session_id, root=sessions_root).iterdir()}
    assert names == {
        "session.json",
        "runspec.json",
        "findings.json",
        "proposal.json",
        "diff.json",
        "logs",
    }


def test_session_adopted_secret_runspec_succeeds(tmp_path: Path):
    """H1: adopting a secret-declaring runspec from the session resolves secrets."""
    if not BEFORE.is_file():
        raise FileNotFoundError(f"missing fixture profile: {BEFORE}")
    before = BEFORE.resolve()
    session_id = "secret-adopt"
    runspec_path = tmp_path / "rs.json"
    runspec_path.write_bytes(
        RunSpec(
            argv=("true",),
            environment=EnvironmentSpec(policy="clean", secrets=("MY_TOKEN",)),
        ).canonical_json_bytes()
    )

    evidence = _run_cli(
        tmp_path,
        "evidence",
        "build",
        str(before),
        "--format",
        "json",
        "--gpu",
        "0",
        "--session",
        session_id,
        "--analyzers",
        "overlap_ratio",
    )
    assert evidence.returncode == 0, evidence.stderr
    finding_id = _finding_id_from_evidence_stdout(evidence.stdout)

    first = _run_cli(
        tmp_path,
        "propose",
        "--session",
        session_id,
        "--finding-id",
        finding_id,
        "--runspec",
        str(runspec_path),
        MY_TOKEN="secret-value-for-test",
    )
    assert first.returncode == 0, first.stderr

    second = _run_cli(
        tmp_path,
        "propose",
        "--session",
        session_id,
        "--finding-id",
        finding_id,
        MY_TOKEN="secret-value-for-test",
    )
    assert second.returncode == 0, second.stderr
    assert "Proposal published to session " in second.stdout
    assert "MY_TOKEN" not in second.stderr
