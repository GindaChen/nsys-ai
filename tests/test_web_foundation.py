from __future__ import annotations

import http.client
import threading
from types import SimpleNamespace

import pytest

from nsys_ai import web
from nsys_ai.diff_web import _DiffHandler
from nsys_ai.gpu_label import format_gpu_label


def test_gpu_label_omits_missing_metadata():
    assert format_gpu_label(0) == "GPU 0"
    assert format_gpu_label(0, SimpleNamespace(name="", pci_bus="", sm_count=0, memory_bytes=0)) == "GPU 0"
    assert format_gpu_label(1, SimpleNamespace(name="H100", pci_bus="", sm_count=132, memory_bytes=0)) == "GPU 1 - H100, 132 SMs"


def test_gpu_label_supports_summary_data():
    assert format_gpu_label(
        2,
        {"name": "H100", "pci_bus": "0000:01:00.0", "sm_count": 132, "memory_gb": 80},
    ) == "GPU 2 - H100 (0000:01:00.0), 132 SMs, 80GB"


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
