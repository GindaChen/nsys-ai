import http.client
import json
import threading
from contextlib import contextmanager
from pathlib import Path

from nsys_ai import web
from nsys_ai.diff_decision import write_diff_decision_json_from_diff_dict
from nsys_ai.loop_state import DiffLoopState


def _diff_summary(confidence: float = 0.9) -> dict:
    """Minimal to_diff_dict-shaped payload sufficient to record a decision."""
    return {
        "verdict": "neutral",
        "comparability_confidence": confidence,
        "warnings": [],
        "before": {"path": "/tmp/before.sqlite", "profile_id": "before-id"},
        "after": {"path": "/tmp/after.sqlite", "profile_id": "after-id"},
    }


@contextmanager
def _running_loop_server(loop_state):
    """Start a real _ViewerHandler HTTP server bound to an ephemeral port."""
    handler = web._ViewerHandler
    saved_state, saved_prof = handler._loop_state, handler.prof
    handler._loop_state, handler.prof = loop_state, None
    server = web._ThreadedHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        handler._loop_state, handler.prof = saved_state, saved_prof


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


def test_web_decision_writes_diff_json_and_matches_cli(tmp_path, monkeypatch):
    # The web writer targets Path("diff.json") in the process CWD (same as the CLI).
    monkeypatch.chdir(tmp_path)
    state = DiffLoopState()
    state.diff_summary = _diff_summary()

    with _running_loop_server(state) as port:
        status, data = _post(port, "/api/loop/decision", {"decision": "accept", "reason": "faster on H100"})

    assert status == 200
    assert data["decision"] == "accept"
    assert data["decision_reason"] == "faster on H100"
    assert data["decision_path"], "decision_path should be surfaced back to the client"

    written = tmp_path / "diff.json"
    assert written.exists()
    web_record = json.loads(written.read_text(encoding="utf-8"))
    assert web_record["decision"]["status"] == "accepted"
    assert web_record["decision"]["reason"] == "faster on H100"

    # Byte-shape parity with the CLI encoder for the same diff payload: everything
    # except the volatile identity/timestamp must be identical.
    write_diff_decision_json_from_diff_dict(
        _diff_summary(),
        decision="accepted",
        reason="faster on H100",
        path=tmp_path / "cli.json",
        decider="fixed@example",
        decided_at="2026-01-01T00:00:00Z",
    )
    cli_record = json.loads((tmp_path / "cli.json").read_text(encoding="utf-8"))
    web_record["decision"]["decider"] = "fixed@example"
    web_record["decision"]["decided_at"] = "2026-01-01T00:00:00Z"
    assert json.dumps(web_record, sort_keys=True) == json.dumps(cli_record, sort_keys=True)


def test_web_decision_survives_reload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = DiffLoopState()
    state.diff_summary = _diff_summary()

    with _running_loop_server(state) as port:
        _post(port, "/api/loop/decision", {"decision": "reject", "reason": "regressed"})
        # Simulate a page reload: re-fetch loop state from the server.
        status, reloaded = _get(port, "/api/loop/state")

    assert status == 200
    assert reloaded["decision"] == "reject"
    assert reloaded["decision_reason"] == "regressed"
    assert reloaded["decision_path"].endswith("diff.json")
    # The record is on disk independent of server memory, and rehydrates cleanly.
    assert (tmp_path / "diff.json").exists()
    restored = DiffLoopState.from_dict(reloaded)
    assert restored.decision == "reject"
    assert restored.decision_path.endswith("diff.json")


def test_web_decision_requires_reason(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = DiffLoopState()
    state.diff_summary = _diff_summary()

    with _running_loop_server(state) as port:
        status, data = _post(port, "/api/loop/decision", {"decision": "accept", "reason": "   "})

    assert status == 400
    assert "reason" in data.get("error", "").lower()
    assert not (tmp_path / "diff.json").exists()


def test_web_decision_requires_diff_first(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = DiffLoopState()  # no diff run yet

    with _running_loop_server(state) as port:
        status, data = _post(port, "/api/loop/decision", {"decision": "accept", "reason": "looks good"})

    assert status == 400
    assert "diff" in data.get("error", "").lower()
    assert not (tmp_path / "diff.json").exists()


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
