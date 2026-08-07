"""Web loop API: session-backed decision endpoints and route registration."""

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
from nsys_ai.profile_runner import build_local_profile_reference
from nsys_ai.runspec import RunSpec
from nsys_ai.session_cli import project_loop_state, resolve_session_id, session_dir
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


def _run_cli(cwd: Path, *args: str, timeout: float = 180.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "nsys_ai", *args],
        cwd=cwd,
        env=_subprocess_environment(),
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


def _publish_through_propose(tmp_path: Path, session_id: str) -> None:
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


@contextmanager
def _running_session_server(tmp_path: Path, session_id: str):
    """Start a real _ViewerHandler HTTP server bound to an ephemeral port."""
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
        handler.prof = None
        handler.devices = [0]
        handler._findings = []
        handler._session_root = ".nsys-ai/sessions"
        before_ref = build_local_profile_reference(BEFORE.resolve())
        resolved = resolve_session_id(session_id, before=before_ref)
        snapshot = SessionStore(handler._session_root).load(resolved)
        handler._session_id = resolved
        project_loop_state(
            snapshot,
            session_dir_path=session_dir(resolved, root=handler._session_root),
        )
        server = web._ThreadedHTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield port, resolved
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


def _post(port, path, payload):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", path, body=json.dumps(payload), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    status, body = resp.status, resp.read()
    conn.close()
    return status, json.loads(body) if body else {}


def _get(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    status, body = resp.status, resp.read()
    conn.close()
    return status, json.loads(body) if body else {}


def test_web_decision_writes_session_diff_json(tmp_path):
    if not BEFORE.is_file() or not AFTER.is_file():
        raise FileNotFoundError(f"missing fixture profiles: {BEFORE} / {AFTER}")
    session_id = "loop-api-decide"
    _publish_full_session(tmp_path, session_id)

    with _running_session_server(tmp_path, session_id) as (port, resolved):
        status, data = _post(
            port, "/api/loop/decision", {"decision": "accept", "reason": "faster on H100"}
        )

    assert status == 200, data
    assert data["decision"] == "accept"
    assert data["decision_reason"] == "faster on H100"
    assert data["decision_path"].endswith("diff.json")
    assert data["session_mode"] is True
    assert data["session_id"] == resolved

    written = tmp_path / ".nsys-ai" / "sessions" / session_id / "diff.json"
    assert written.exists()
    record = json.loads(written.read_text(encoding="utf-8"))
    assert record["decision"]["status"] == "accepted"
    assert record["decision"]["reason"] == "faster on H100"


def test_web_decision_survives_reload(tmp_path):
    if not BEFORE.is_file() or not AFTER.is_file():
        raise FileNotFoundError(f"missing fixture profiles: {BEFORE} / {AFTER}")
    session_id = "loop-api-reload"
    _publish_full_session(tmp_path, session_id)

    with _running_session_server(tmp_path, session_id) as (port, _):
        _post(port, "/api/loop/decision", {"decision": "reject", "reason": "regressed"})
        status, reloaded = _get(port, "/api/loop/state")

    assert status == 200
    assert reloaded["decision"] == "reject"
    assert reloaded["decision_reason"] == "regressed"
    assert reloaded["decision_path"].endswith("diff.json")
    assert (tmp_path / ".nsys-ai" / "sessions" / session_id / "diff.json").exists()


def test_web_decision_requires_reason(tmp_path):
    if not BEFORE.is_file() or not AFTER.is_file():
        raise FileNotFoundError(f"missing fixture profiles: {BEFORE} / {AFTER}")
    session_id = "loop-api-reason"
    _publish_full_session(tmp_path, session_id)

    with _running_session_server(tmp_path, session_id) as (port, _):
        status, data = _post(port, "/api/loop/decision", {"decision": "accept", "reason": "   "})

    assert status == 400
    assert "reason" in data.get("error", "").lower()


def test_web_decision_requires_diff_first(tmp_path):
    if not BEFORE.is_file():
        raise FileNotFoundError(f"missing fixture profile: {BEFORE}")
    session_id = "loop-api-no-diff"
    _publish_through_propose(tmp_path, session_id)

    with _running_session_server(tmp_path, session_id) as (port, _):
        status, data = _post(
            port, "/api/loop/decision", {"decision": "accept", "reason": "looks good"}
        )

    assert status == 400
    assert "diff" in data.get("error", "").lower()


def test_web_loop_endpoints_are_registered():
    web_py = Path("src/nsys_ai/web.py").read_text(encoding="utf-8")
    for route in (
        "/api/loop/state",
        "/api/loop/phase",
        "/api/loop/proposal",
        "/api/loop/reprofile",
        "/api/loop/diagnose",
        "/api/loop/diff",
        "/api/loop/decision",
    ):
        assert route in web_py


def test_timeline_template_has_loop_controls():
    html = Path("src/nsys_ai/templates/timeline.html").read_text(encoding="utf-8")
    assert 'id="loopBtn"' in html
    assert 'id="loopSidebar"' in html
    assert 'id="loopPrimaryBtn"' in html
    assert "loopRunPrimary()" in html
    assert 'id="loopDiffStats"' in html
    assert "LOOP_TRIM_NS" in html
