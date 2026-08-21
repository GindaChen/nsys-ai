from __future__ import annotations

import http.client
import json
import re
import threading
from types import SimpleNamespace

import pytest

from nsys_ai import web
from nsys_ai.diff_web import _DIFF_HTML, _DiffHandler
from nsys_ai.gpu_label import format_gpu_label, format_gpu_narrative_label
from nsys_ai.profile import Profile
from nsys_ai.summary import auto_commentary
from nsys_ai.viewer import generate_html
from nsys_ai.web import _slice_tree_nodes


def test_gpu_label_omits_missing_metadata():
    assert format_gpu_label(0) == "GPU 0"
    assert format_gpu_label(0, SimpleNamespace(name="", pci_bus="", sm_count=0, memory_bytes=0)) == "GPU 0"
    assert format_gpu_label(1, SimpleNamespace(name="H100", pci_bus="", sm_count=132, memory_bytes=0)) == "GPU 1 - H100, 132 SMs"
    assert format_gpu_narrative_label(0, {"name": ""}) == "GPU 0"
    assert format_gpu_narrative_label(1, {"name": "H100"}) == "GPU 1 (H100)"


def test_gpu_label_supports_summary_data():
    assert format_gpu_label(
        2,
        {"name": "H100", "pci_bus": "0000:01:00.0", "sm_count": 132, "memory_gb": 80},
    ) == "GPU 2 - H100 (0000:01:00.0), 132 SMs, 80GB"


def test_gpu_label_preserves_fractional_memory_precision():
    assert format_gpu_label(
        0,
        SimpleNamespace(name="H200", pci_bus="", sm_count=0, memory_bytes=150_100_000_000),
    ) == "GPU 0 - H200, 150.1GB"


def test_tree_web_shell_does_not_embed_profile_data(minimal_nsys_db_path):
    with Profile(minimal_nsys_db_path) as prof:
        html = generate_html(prof, 0, prof.meta.time_range, embed_data=False)

    assert len(html) < 200_000
    assert "let DATA = [];" in html
    assert "/api/tree" in html


def test_tree_web_binds_before_background_tree_build(monkeypatch):
    events = []

    class FakeServer:
        server_address = ("127.0.0.1", 9911)

    def bind(_port, _handler):
        events.append("bind")
        return FakeServer()

    def build(*_args):
        events.append("build")
        return []

    monkeypatch.setattr(web, "_bind_local_server", bind)
    monkeypatch.setattr(web, "_run_server", lambda *_args: events.append("serve"))
    monkeypatch.setattr(web, "generate_html", lambda *_args, **_kwargs: "<html>")
    monkeypatch.setattr("nsys_ai.nvtx_tree.build_nvtx_tree", build)

    web.serve(object(), 0, (0, 1), port=9911, open_browser=False)

    assert events.index("bind") < events.index("build")


def test_tree_web_endpoint_returns_bounded_slices():
    old_data = web._ViewerHandler._tree_data
    old_configured = web._ViewerHandler._tree_configured
    web._ViewerHandler._tree_data = [
        {
            "type": "nvtx",
            "name": "root",
            "path": "root",
            "children": [{"type": "kernel", "name": "child", "path": "root > child"}],
        }
    ]
    web._ViewerHandler._tree_configured = True
    try:
        status, body, content_type = _get(web._ViewerHandler, "/api/tree?depth=0")
    finally:
        web._ViewerHandler._tree_data = old_data
        web._ViewerHandler._tree_configured = old_configured

    payload = json.loads(body)
    assert status == 200
    assert content_type.startswith("application/json")
    assert payload["nodes"][0]["has_children"] is True
    assert "children" not in payload["nodes"][0] or payload["nodes"][0]["children"] == []


def test_tree_slice_preserves_children_marker_without_mutating_source():
    source = [{"path": "root", "children": [{"path": "child"}]}]

    result = _slice_tree_nodes(source, 0)

    assert result[0]["has_children"] is True
    assert source[0]["children"] == [{"path": "child"}]


def test_timeline_data_rejects_obsolete_resolution_parameter():
    status, body, content_type = _get(web._ViewerHandler, "/api/data?resolution=2000")

    assert status == 400
    assert content_type.startswith("application/json")
    assert b"use max_buckets" in body


@pytest.mark.parametrize("handler", [web._ViewerHandler, web._EvidenceHandler, _DiffHandler])
def test_web_surfaces_serve_shared_tokens_asset(handler):
    status, body, content_type = _get(handler, "/assets/tokens.css")

    assert status == 200
    assert content_type.startswith("text/css")
    assert b"--accent:" in body
    assert b"--heat-75:" in body


def test_summary_commentary_omits_empty_gpu_name():
    summary = {
        "device": 0,
        "hardware": {"name": ""},
        "timing": {"span_ms": 10.0, "utilization_pct": 25.0, "idle_ms": 0.0},
        "kernel_count": 3,
        "top_kernels": [],
    }

    commentary = auto_commentary(summary)

    assert commentary.startswith("GPU 0 ran 3 kernels")
    assert "()" not in commentary


def test_diff_web_declares_metric_polarity_for_overlap_summary():
    assert "const improvementDirection =" in _DIFF_HTML
    assert "overlap_ms: 1" in _DIFF_HTML
    assert "overlap_pct: 1" in _DIFF_HTML
    assert "const signedDelta = delta * improvementDirection[k];" in _DIFF_HTML
    assert "signedDelta > 0 ? 'delta-good'" in _DIFF_HTML
    assert "signedDelta < 0 ? 'delta-bad'" in _DIFF_HTML


def _get(handler, path: str, *, html: bytes = b"<html>"):
    if handler is web._ViewerHandler:
        old_html = handler.html_bytes
        handler.html_bytes = html
    else:
        old_html = None
    server = web._ThreadedHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    conn.request("GET", path)
    response = conn.getresponse()
    result = response.status, response.read(), response.getheader("Content-Type")
    conn.close()
    thread.join(timeout=5)
    server.server_close()
    if handler is web._ViewerHandler:
        handler.html_bytes = old_html
    return result


def _post(handler, path: str):
    server = web._ThreadedHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    conn.request("POST", path, body=b"{}", headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    result = response.status, response.read(), response.getheader("Content-Type")
    conn.close()
    thread.join(timeout=5)
    server.server_close()
    return result


def _head(handler, path: str, *, html: bytes = b"<html>"):
    if handler is web._ViewerHandler:
        old_html = handler.html_bytes
        handler.html_bytes = html
    else:
        old_html = None
    server = web._ThreadedHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    conn.request("HEAD", path)
    response = conn.getresponse()
    result = (
        response.status,
        response.read(),
        response.getheader("Content-Type"),
        response.getheader("Content-Length"),
    )
    conn.close()
    thread.join(timeout=5)
    server.server_close()
    if handler is web._ViewerHandler:
        handler.html_bytes = old_html
    return result


@pytest.mark.parametrize("handler", [web._ViewerHandler, web._EvidenceHandler, _DiffHandler])
def test_web_surfaces_head_matches_get_headers_without_body(handler):
    get_status, get_body, get_content_type = _get(handler, "/")
    head_status, head_body, head_content_type, head_length = _head(handler, "/")

    assert head_status == get_status == 200
    assert head_content_type == get_content_type
    assert head_length == str(len(get_body))
    assert head_body == b""


@pytest.mark.parametrize("handler", [web._ViewerHandler, web._EvidenceHandler, _DiffHandler])
def test_web_surfaces_head_preserves_json_404_contract(handler):
    get_status, get_body, get_content_type = _get(handler, "/api/does-not-exist")
    head_status, head_body, head_content_type, head_length = _head(
        handler, "/api/does-not-exist"
    )

    assert head_status == get_status == 404
    assert head_content_type == get_content_type
    assert head_length == str(len(get_body))
    assert head_body == b""


@pytest.mark.parametrize("handler", [web._ViewerHandler, web._EvidenceHandler, _DiffHandler])
def test_web_surfaces_return_404_for_unknown_paths(handler):
    status, body, content_type = _get(handler, "/api/definitely-not-real")

    assert status == 404
    assert content_type.startswith("application/json")
    assert b"not found" in body


@pytest.mark.parametrize("handler", [web._ViewerHandler, web._EvidenceHandler, _DiffHandler])
def test_web_surfaces_keep_root_page(handler):
    status, _body, _content_type = _get(handler, "/")

    assert status == 200


def test_viewer_returns_json_404_for_unknown_post_api():
    status, body, content_type = _post(web._ViewerHandler, "/api/definitely-not-real")

    assert status == 404
    assert content_type.startswith("application/json")
    assert b"not found" in body


def test_viewer_tree_reports_background_building_state():
    old_configured = web._ViewerHandler._tree_configured
    old_error = web._ViewerHandler._tree_build_error
    web._ViewerHandler._tree_configured = False
    web._ViewerHandler._tree_build_error = None
    try:
        status, body, content_type = _get(web._ViewerHandler, "/api/tree?depth=0")
    finally:
        web._ViewerHandler._tree_configured = old_configured
        web._ViewerHandler._tree_build_error = old_error

    assert status == 202
    assert content_type.startswith("application/json")
    assert json.loads(body) == {
        "status": "building",
        "message": "tree data is still building",
    }


def test_viewer_ask_route_delegates_to_shared_transport(monkeypatch):
    from nsys_ai import web

    monkeypatch.setattr(web, "_handle_ask_request", lambda body: {"answer": "ok"})
    old_session = web._ViewerHandler._session_id
    old_root = web._ViewerHandler._session_root
    web._ViewerHandler._session_id = None
    try:
        server = web._ThreadedHTTPServer(("127.0.0.1", 0), web._ViewerHandler)
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.request(
            "POST",
            "/api/ask",
            body=json.dumps({"question": "why?"}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        status, body = response.status, response.read()
        conn.close()
        thread.join(timeout=5)
        server.server_close()
    finally:
        web._ViewerHandler._session_id = old_session
        web._ViewerHandler._session_root = old_root

    assert status == 200
    assert json.loads(body) == {"answer": "ok"}


def test_viewer_ask_route_supports_shared_runner_sse(monkeypatch):
    from nsys_ai import web

    monkeypatch.setattr(
        web,
        "_handle_ask_stream_request",
        lambda body: iter(
            [
                b'event: text\ndata: {"chunk":"ok"}\n\n',
                b'event: done\ndata: {"selected_skills":[]}\n\n',
            ]
        ),
    )
    old_session = web._ViewerHandler._session_id
    web._ViewerHandler._session_id = None
    try:
        server = web._ThreadedHTTPServer(("127.0.0.1", 0), web._ViewerHandler)
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.request(
            "POST",
            "/api/ask",
            body=json.dumps({"question": "why?", "stream": True}),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        status, content_type, body = response.status, response.getheader("Content-Type"), response.read()
        conn.close()
        thread.join(timeout=5)
        server.server_close()
    finally:
        web._ViewerHandler._session_id = old_session

    assert status == 200
    assert content_type.startswith("text/event-stream")
    assert body.startswith(b"event: text\n")
    assert b'event: done\ndata: {"selected_skills":[]}' in body


def test_timeline_canvas_reads_the_shared_token_palette():
    javascript = open("src/nsys_ai/templates/timeline.js", encoding="utf-8").read()
    tokens = open("src/nsys_ai/templates/tokens.css", encoding="utf-8").read()

    assert "getComputedStyle(document.documentElement)" in javascript
    assert "const P = Object.freeze" in javascript
    assert "function nvtxColor(depth, identity)" in javascript
    assert "depth % NVTX_COLORS.length" not in javascript
    assert "background:${withAlpha(P.bg, 0.8)}" in javascript
    assert "withAlpha(P.selected, 0.5)" in javascript
    assert "P.laneMemory" in javascript
    assert "--cat-other" in tokens
    assert "--mag-5" in tokens
    assert not re.search(r"#[0-9a-fA-F]{6}", javascript)
