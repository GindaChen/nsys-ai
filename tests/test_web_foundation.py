from __future__ import annotations

import http.client
import threading
from types import SimpleNamespace

import pytest

from nsys_ai import web
from nsys_ai.diff_web import _DIFF_HTML, _DiffHandler
from nsys_ai.gpu_label import format_gpu_label, format_gpu_narrative_label
from nsys_ai.summary import auto_commentary


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
