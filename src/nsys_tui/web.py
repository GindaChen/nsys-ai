"""
web.py — Serve profiles via local HTTP servers.

Provides two modes:
  1. `serve`          — Serve the built-in interactive HTML viewer.
  2. `serve_perfetto` — Serve Perfetto JSON and open ui.perfetto.dev.

Usage:
    nsys-ai web      profile.sqlite --gpu 0 --trim 39 42
    nsys-ai perfetto profile.sqlite --gpu 0 --trim 39 42
"""
import json
import os
import queue
import socketserver
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import quote

# Bounded thread pool: fixed worker count, request queue with max size.
# Workers are released when each request finishes (finish_request + shutdown_request).
# See docs/chat-thread-pool.md.
CHAT_SERVER_POOL_SIZE = 8
CHAT_SERVER_QUEUE_SIZE = 16


class _ThreadPoolMixIn(socketserver.ThreadingMixIn):
    """Use a fixed-size thread pool instead of one thread per request. Prevents thread exhaustion."""

    daemon_threads = True
    _pool_size = CHAT_SERVER_POOL_SIZE
    _queue_maxsize = CHAT_SERVER_QUEUE_SIZE

    def process_request(self, request, client_address):
        """Enqueue request for a pool worker instead of spawning a new thread."""
        if not getattr(self, "_pool_ready", False):
            self._request_queue = queue.Queue(maxsize=self._queue_maxsize)
            for _ in range(self._pool_size):
                t = threading.Thread(target=self._pool_worker, daemon=True)
                t.start()
            self._pool_ready = True
        try:
            self._request_queue.put((request, client_address), block=True, timeout=30)
        except queue.Full:
            self.handle_error(request, client_address)

    def _pool_worker(self):
        """Worker loop: take (request, client_address) from queue and handle; thread is released when done."""
        while True:
            try:
                request, client_address = self._request_queue.get()
                if request is None:
                    break
                try:
                    self.process_request_thread(request, client_address)
                except Exception:
                    self.handle_error(request, client_address)
            except Exception:
                pass


class _ThreadedHTTPServer(_ThreadPoolMixIn, socketserver.ThreadingMixIn, HTTPServer):
    """Concurrent chat requests via bounded thread pool; workers released after each request."""
    daemon_threads = True

from .export import gpu_trace  # noqa: E402
from .viewer import generate_html, generate_timeline_html  # noqa: E402

# ── Shared helpers ───────────────────────────────────────────────

def _run_server(server, open_url, prof):
    """Run an HTTPServer with browser-open and graceful shutdown. Uses server.server_address[1] for URL/port."""
    actual_port = server.server_address[1]
    actual_url = f"http://127.0.0.1:{actual_port}"
    print(f"Serving at {actual_url}")
    pool_size = getattr(server, "_pool_size", None)
    if pool_size is not None:
        print(f"  (thread pool: {pool_size} workers, queue max {getattr(server, '_queue_maxsize', '?')})")
    if os.environ.get("SSH_CONNECTION"):
        print(f"  Remote/SSH: on your local machine run:  ssh -L {actual_port}:127.0.0.1:{actual_port} <host>  then open the URL in your local browser.")
    print("Press Ctrl-C to stop.")
    if open_url:
        # Use actual server URL when caller passed a localhost URL (so port=0 works).
        open_target = actual_url if (open_url and open_url.startswith("http://127.0.0.1:")) else open_url
        threading.Timer(0.3, webbrowser.open, args=(open_target,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
        prof.close()


# ── Mode 1: Built-in HTML viewer ────────────────────────────────

def _handle_chat_request(body_bytes: bytes) -> dict | None:
    """Handle POST /api/chat. Returns JSON-serializable dict or None for 501."""
    try:
        from . import chat
        return chat.chat_completion(body_bytes)
    except ImportError:
        pass
    # Phase 0 mock: no chat module or LLM not configured
    return {"content": "Mock reply. Configure an LLM endpoint (e.g. pip install nsys-ai[ai], set ANTHROPIC_API_KEY) for real analysis.", "actions": []}


def _handle_chat_stream(body_bytes: bytes):
    """Return generator yielding SSE bytes for stream=true, or None for 501."""
    try:
        from . import chat
        if hasattr(chat, "chat_completion_stream"):
            return chat.chat_completion_stream(body_bytes)
    except ImportError:
        pass
    return None


class _ViewerHandler(BaseHTTPRequestHandler):
    """Serve the pre-rendered HTML on GET; GET /api/models for model list; POST /api/chat for AI chat."""
    html_bytes: bytes = b""

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/models":
            try:
                import nsys_tui.chat as chat_mod
                options = chat_mod.get_available_models()
                default = chat_mod.get_default_model()
            except Exception:
                options = []
                default = None
            body = json.dumps({"default": default, "options": options}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(self.html_bytes)))
        self.end_headers()
        self.wfile.write(self.html_bytes)

    def do_POST(self):
        if self.path.split("?")[0] != "/api/chat":
            self.send_error(404)
            return
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b"{}"
        stream_requested = False
        try:
            payload = json.loads(body.decode("utf-8"))
            stream_requested = payload.get("stream") is True
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            pass
        try:
            if stream_requested:
                print("[chat] stream request received", flush=True)
                gen = _handle_chat_stream(body)
                if gen is None:
                    self.send_response(501)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(b'{"error":"LLM not configured"}')
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                for chunk in gen:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                return
            out = _handle_chat_request(body)
            if out is None:
                self.send_response(501)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(b'{"error":"LLM not configured"}')
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            resp = json.dumps(out).encode("utf-8")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def log_message(self, format, *args):
        pass


def serve(prof, device: int, trim: tuple[int, int], *,
          port: int = 8142, open_browser: bool = True):
    """Start a local HTTP server serving the interactive HTML viewer.
    If the requested port is in use, tries port 0 (system assigns a free port) and opens that URL.
    """
    html = generate_html(prof, device, trim)
    _ViewerHandler.html_bytes = html.encode("utf-8")

    try:
        server = _ThreadedHTTPServer(("127.0.0.1", port), _ViewerHandler)
    except OSError:
        if port == 0:
            raise
        server = _ThreadedHTTPServer(("127.0.0.1", 0), _ViewerHandler)
        print(f"Port {port} in use, using port {server.server_address[1]} instead.")
    open_url = f"http://127.0.0.1:{server.server_address[1]}" if open_browser else None
    _run_server(server, open_url, prof)


# ── Mode 2: Horizontal timeline viewer ──────────────────────────

def serve_timeline(prof, device: int, trim: tuple[int, int], *,
                   port: int = 8144, open_browser: bool = True):
    """Start a local HTTP server serving the horizontal timeline viewer."""
    html = generate_timeline_html(prof, device, trim)
    _ViewerHandler.html_bytes = html.encode("utf-8")

    server = _ThreadedHTTPServer(("127.0.0.1", port), _ViewerHandler)
    actual_url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Timeline viewer at {actual_url}")
    _run_server(server, actual_url if open_browser else None, prof)


# ── Mode 2: Perfetto UI ─────────────────────────────────────────

class _PerfettoHandler(BaseHTTPRequestHandler):
    """Serve Perfetto JSON trace with CORS so ui.perfetto.dev can fetch it."""
    trace_bytes: bytes = b""

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.trace_bytes)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(self.trace_bytes)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def log_message(self, format, *args):
        pass


def serve_perfetto(prof, device: int, trim: tuple[int, int], *,
                   port: int = 8143, open_browser: bool = True):
    """Generate Perfetto JSON, serve it locally, and open ui.perfetto.dev."""
    events = gpu_trace(prof, device, trim)
    trace = json.dumps({"traceEvents": events, "displayTimeUnit": "ms"})
    _PerfettoHandler.trace_bytes = trace.encode("utf-8")

    nk = sum(1 for e in events if e.get("cat") == "gpu_kernel")
    nn = sum(1 for e in events if e.get("cat") == "nvtx_projected")
    print(f"Trace: {nk} kernels, {nn} NVTX, {len(trace)//1024} KB")

    server = HTTPServer(("127.0.0.1", port), _PerfettoHandler)
    actual_port = server.server_address[1]
    trace_url = f"http://127.0.0.1:{actual_port}/trace.json"
    perfetto_url = f"https://ui.perfetto.dev/#!/?url={quote(trace_url, safe='')}"

    print(f"Perfetto UI: {perfetto_url}")
    _run_server(server, perfetto_url if open_browser else None, prof)
