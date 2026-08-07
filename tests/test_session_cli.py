"""CLI → SessionStore publishing, including cross-process restart coverage."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from nsys_ai.runspec import RunSpec
from nsys_ai.session_cli import session_id_from_profile_id
from nsys_ai.session_store import SessionStore

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / "tests" / "fixtures" / "mfu_2gpu_before.sqlite"
AFTER = ROOT / "tests" / "fixtures" / "mfu_2gpu_after.sqlite"


def _subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    source = str(ROOT / "src")
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not current else f"{source}{os.pathsep}{current}"
    return environment


def _run_cli(cwd: Path, *args: str, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "nsys_ai", *args],
        cwd=cwd,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
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
    payload = json.loads(evidence.stdout)
    finding_id = next(
        finding["id"]
        for finding in payload["findings"]
        if finding.get("id") and finding.get("suggested_actions")
    )

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

    session_dir = tmp_path / ".nsys-ai" / "sessions" / session_id
    assert session_dir.is_dir()
    names = {path.name for path in session_dir.iterdir()}
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
