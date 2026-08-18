"""Web session mode: open CLI-published artifacts; cross-process decision handoff."""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from nsys_ai import web
from nsys_ai.profile import Profile
from nsys_ai.runspec import RunSpec
from nsys_ai.session_cli import session_dir
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


def _run_cli(cwd: Path, *args: str, timeout: float = 180.0, **env: str) -> subprocess.CompletedProcess[str]:
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


def _post(port: int, path: str, payload: dict):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(
        "POST",
        path,
        body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    status, body = resp.status, resp.read()
    conn.close()
    return status, json.loads(body) if body else {}


def _get(port: int, path: str):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    status, body = resp.status, resp.read()
    conn.close()
    return status, json.loads(body) if body else {}


@contextmanager
def _session_web_server(tmp_path: Path, profile_path: Path, session_id: str):
    """Serve timeline against an existing session under tmp_path CWD."""
    handler = web._ViewerHandler
    saved = (
        handler.prof,
        handler._session_id,
        handler._session_root,
        handler._findings,
        handler.devices,
    )
    previous_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        with Profile(str(profile_path)) as prof:
            devices = prof.meta.devices if prof.meta.devices else [0]
            # Bind handler state the same way serve_timeline does in session mode,
            # without starting the blocking serve_forever used by serve_timeline.
            from nsys_ai.profile_runner import build_local_profile_reference
            from nsys_ai.session_cli import project_loop_state, resolve_session_id
            from nsys_ai.session_cli import session_dir as sdir
            from nsys_ai.session_store import SessionStore as Store

            handler.prof = prof
            handler.devices = list(devices)
            handler._tile_nvtx_cache = {}
            handler._session_root = ".nsys-ai/sessions"
            before_ref = build_local_profile_reference(profile_path)
            resolved = resolve_session_id(session_id, before=before_ref)
            snapshot = Store(handler._session_root).load(resolved)
            handler._session_id = resolved
            findings = []
            if snapshot.findings is not None:
                findings = [f.to_dict() for f in snapshot.findings.findings]
            handler._findings = findings
            # Touch projection once so missing artifacts fail before the server starts.
            project_loop_state(
                snapshot,
                session_dir_path=sdir(resolved, root=handler._session_root),
            )
            server = web._ThreadedHTTPServer(("127.0.0.1", 0), handler)
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                yield port, resolved, findings
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
    finally:
        os.chdir(previous_cwd)
        (
            handler.prof,
            handler._session_id,
            handler._session_root,
            handler._findings,
            handler.devices,
        ) = saved


def _publish_through_propose(tmp_path: Path, session_id: str) -> str:
    if not BEFORE.is_file():
        raise FileNotFoundError(f"missing fixture profile: {BEFORE}")
    before = BEFORE.resolve()
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
    return finding_id


def _publish_full_session(tmp_path: Path, session_id: str) -> None:
    _publish_through_propose(tmp_path, session_id)
    before = BEFORE.resolve()
    after = AFTER.resolve()
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


def test_cli_artifacts_open_in_the_browser(tmp_path: Path):
    """Acceptance: CLI-created findings, RunSpec and proposal open in the browser."""
    if not BEFORE.is_file():
        raise FileNotFoundError(f"missing fixture profile: {BEFORE}")
    session_id = "web-open-cli"
    _publish_through_propose(tmp_path, session_id)

    directory = session_dir(session_id, root=tmp_path / ".nsys-ai" / "sessions")
    assert (directory / "findings.json").is_file()
    assert (directory / "runspec.json").is_file()
    assert (directory / "proposal.json").is_file()

    with _session_web_server(tmp_path, BEFORE.resolve(), session_id) as (
        port,
        resolved,
        startup_findings,
    ):
        assert resolved == session_id
        assert startup_findings, "session findings should seed the overlay at open"

        status, state = _get(port, "/api/loop/state")
        assert status == 200
        assert state["session_mode"] is True
        assert state["session_id"] == session_id
        assert state["diagnose_ran"] is True
        assert state["diagnose_findings_count"] >= 1
        assert isinstance(state["proposal"], dict)
        assert state["proposal"].get("summary")
        assert state["proposal"].get("abstained") is False
        assert state["phase"] == "propose"
        assert state["diff_summary"] is None
        assert state["decision"] is None
        assert state["decision_path"] == ""

        status, findings = _get(port, "/api/findings")
        assert status == 200
        assert isinstance(findings, list)
        assert len(findings) == state["diagnose_findings_count"]

        status, limited = _post(port, "/api/loop/diagnose", {})
        assert status == 400
        assert limited.get("limitation") is True
        assert "nsys-ai evidence" in limited.get("cli", "")

        status, limited = _post(
            port, "/api/loop/proposal", {"proposal": "free text", "expected_impact": "x"}
        )
        assert status == 400
        assert limited.get("limitation") is True
        assert "nsys-ai propose" in limited.get("cli", "")


def test_browser_decision_visible_to_new_cli_process(tmp_path: Path):
    """Acceptance: browser decision survives a real second process (not same server)."""
    if not BEFORE.is_file() or not AFTER.is_file():
        raise FileNotFoundError(f"missing fixture profiles: {BEFORE} / {AFTER}")
    session_id = "web-decide-handoff"
    _publish_full_session(tmp_path, session_id)

    with _session_web_server(tmp_path, BEFORE.resolve(), session_id) as (port, _, _):
        status, state = _get(port, "/api/loop/state")
        assert status == 200
        assert state["phase"] == "diff"
        assert state["diff_summary"]
        assert state["decision"] is None
        assert state["decision_path"] == ""

        status, limited = _post(port, "/api/loop/diff", {})
        assert status == 200
        assert limited["diff"]["verdict"]
        assert limited["state"]["diff_summary"]

        status, decided = _post(
            port,
            "/api/loop/decision",
            {"decision": "accept", "reason": "browser verified improvement"},
        )
        assert status == 200, decided
        assert decided["decision"] == "accept"
        assert decided["decision_reason"] == "browser verified improvement"
        assert decided["decision_path"].endswith("diff.json")
        assert decided["phase"] == "accept"

    # New interpreter process — not a second request to the same server.
    script = """
import json, sys
from nsys_ai.session_store import SessionStore
snapshot = SessionStore(sys.argv[1]).load(sys.argv[2])
decision = snapshot.diff.get("decision") if snapshot.diff else None
print(json.dumps({
    "phase": snapshot.state.phase,
    "status": None if decision is None else decision.get("status"),
    "reason": None if decision is None else decision.get("reason"),
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
    assert reloaded["phase"] == "accept"
    assert reloaded["status"] == "accepted"
    assert reloaded["reason"] == "browser verified improvement"


def test_cli_decision_visible_to_browser(tmp_path: Path):
    """Acceptance reverse: CLI-recorded decision is visible when the browser opens."""
    if not BEFORE.is_file() or not AFTER.is_file():
        raise FileNotFoundError(f"missing fixture profiles: {BEFORE} / {AFTER}")
    session_id = "cli-decide-then-web"
    _publish_full_session(tmp_path, session_id)

    store = SessionStore(tmp_path / ".nsys-ai" / "sessions")
    with store.writer(session_id) as writer:
        writer.publish_decision("reject", "cli rejected before browser open")

    with _session_web_server(tmp_path, BEFORE.resolve(), session_id) as (port, _, _):
        status, state = _get(port, "/api/loop/state")
        assert status == 200
        assert state["decision"] == "reject"
        assert state["decision_reason"] == "cli rejected before browser open"
        assert state["decision_path"].endswith("decisions.json")
        assert state["phase"] == "propose"

        status, again = _post(
            port,
            "/api/loop/decision",
            {"decision": "accept", "reason": "should fail"},
        )
        assert status == 400
        assert "diff" in again.get("error", "").lower()


def test_abstained_proposal_diff_session_names_cause_and_fix(tmp_path: Path):
    """CLI defect fix: abstained propose must not surface the bare store error."""
    if not BEFORE.is_file() or not AFTER.is_file():
        raise FileNotFoundError(f"missing fixture profiles: {BEFORE} / {AFTER}")
    before = BEFORE.resolve()
    after = AFTER.resolve()
    session_id = "abstain-diff-msg"

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

    # No --runspec → abstained proposal (verified behaviour from the mission preamble).
    propose = _run_cli(
        tmp_path,
        "propose",
        "--session",
        session_id,
        "--finding-id",
        finding_id,
    )
    assert propose.returncode == 0, propose.stderr
    assert "Abstained:" in propose.stdout or "abstain" in propose.stdout.lower()

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
    assert diff.returncode != 0
    combined = (diff.stderr or "") + (diff.stdout or "")
    assert "abstained" in combined.lower()
    assert "runspec" in combined.lower()
    assert "after profile publication requires a non-abstained proposal" not in combined
