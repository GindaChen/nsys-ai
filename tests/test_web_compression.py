"""HTTP responses are compressed when the client says it can read them (#UX).

A timeline window is ~1 MB of highly repetitive JSON and timeline.js is ~150 KB
of text, all of it paid on every cold load. The server prints an SSH
port-forward hint on startup, so it expects to be read over a link where those
bytes are the whole cost.
"""

from __future__ import annotations

import gzip
import json

from nsys_ai.web import _MIN_COMPRESS_BYTES, _send_body


class _FakeHandler:
    """Minimal stand-in for BaseHTTPRequestHandler's response surface."""

    def __init__(self, accept_encoding: str | None):
        self.headers = {} if accept_encoding is None else {"Accept-Encoding": accept_encoding}
        self.status: int | None = None
        self.sent: dict[str, str] = {}
        self.body = b""

    def send_response(self, status):
        self.status = status

    def send_header(self, name, value):
        self.sent[name] = value

    def end_headers(self):
        pass

    @property
    def wfile(self):
        return self

    def write(self, data):
        self.body += data


def _big_payload() -> bytes:
    rows = [{"name": "elementwise_kernel", "start_ns": 60111541307 + i, "stream": 7} for i in range(400)]
    body = json.dumps({"gpus": [{"id": 0, "kernels": rows}]}).encode()
    assert len(body) > _MIN_COMPRESS_BYTES
    return body


def test_gzip_is_used_when_offered_and_round_trips_exactly():
    body = _big_payload()
    handler = _FakeHandler("gzip, deflate, br")
    _send_body(handler, body, "application/json; charset=utf-8")

    assert handler.sent.get("Content-Encoding") == "gzip"
    assert len(handler.body) < len(body), "compression did not shrink the body"
    assert gzip.decompress(handler.body) == body, "decompressed body is not the original"
    assert handler.sent["Content-Length"] == str(len(handler.body))


def test_a_client_that_cannot_gzip_gets_the_plain_body():
    body = _big_payload()
    handler = _FakeHandler(None)
    _send_body(handler, body, "application/json; charset=utf-8")

    assert "Content-Encoding" not in handler.sent
    assert handler.body == body
    assert handler.sent["Content-Length"] == str(len(body))


def test_vary_is_always_set_so_caches_do_not_cross_serve():
    """Without Vary a cache can hand a gzipped body to a client that cannot read it."""
    for accept in ("gzip", None):
        handler = _FakeHandler(accept)
        _send_body(handler, _big_payload(), "application/json; charset=utf-8")
        assert handler.sent.get("Vary") == "Accept-Encoding"


def test_small_bodies_are_not_compressed():
    """Below the threshold the gzip header costs more than it saves."""
    body = b'{"ok":1}'
    handler = _FakeHandler("gzip")
    _send_body(handler, body, "application/json; charset=utf-8")

    assert "Content-Encoding" not in handler.sent
    assert handler.body == body


def test_extra_headers_survive_compression():
    """Assets carry Cache-Control; compressing them must not drop it."""
    handler = _FakeHandler("gzip")
    _send_body(
        handler,
        _big_payload(),
        "application/javascript; charset=utf-8",
        extra_headers={"Cache-Control": "no-cache, must-revalidate"},
    )
    assert handler.sent["Cache-Control"] == "no-cache, must-revalidate"
    assert handler.sent.get("Content-Encoding") == "gzip"


def test_incompressible_body_is_sent_as_is():
    """Refuse a "compression" that grew the payload."""
    body = gzip.compress(b"x" * 4096)  # already compressed, gzipping again grows it
    handler = _FakeHandler("gzip")
    _send_body(handler, body, "application/octet-stream")

    assert "Content-Encoding" not in handler.sent
    assert handler.body == body


def test_every_buffered_response_goes_through_the_shared_writer():
    """Pins the wiring, not just the helper.

    A new endpoint that writes its own body would silently opt out of
    compression, which is exactly how the timeline ended up shipping 2 MB per
    cold load. Streaming responses are the deliberate exception: SSE must not be
    buffered to be compressed.
    """
    import re
    from pathlib import Path

    import nsys_ai.web as web_module

    source = Path(web_module.__file__).read_text(encoding="utf-8")
    lines = source.splitlines()
    offenders = []
    for number, line in enumerate(lines, start=1):
        if not re.search(r"\bwfile\.write\(", line):
            continue
        # The writer itself, and the streaming chat path.
        context = "\n".join(lines[max(0, number - 40) : number])
        if "def _send_body" in context and "class " not in context.split("def _send_body")[-1]:
            continue
        if "stream" in context or "LLM not configured" in line or "text/event-stream" in context:
            continue
        offenders.append(f"{number}: {line.strip()}")
    assert not offenders, (
        "these response paths bypass _send_body and so are never compressed:\n  "
        + "\n  ".join(offenders)
    )
