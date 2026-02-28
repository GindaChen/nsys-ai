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
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import quote

from .viewer import generate_html, generate_timeline_html
from .export import gpu_trace


# ── Shared helpers ───────────────────────────────────────────────

def _run_server(server, open_url, prof):
    """Run an HTTPServer with browser-open and graceful shutdown. Uses server.server_address[1] for URL/port."""
    actual_port = server.server_address[1]
    actual_url = f"http://127.0.0.1:{actual_port}"
    print(f"Serving at {actual_url}")
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

class _ViewerHandler(BaseHTTPRequestHandler):
    """Serve the pre-rendered HTML on every GET request."""
    html_bytes: bytes = b""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(self.html_bytes)))
        self.end_headers()
        self.wfile.write(self.html_bytes)

    def log_message(self, format, *args):
        pass


def serve(prof, device: int, trim: tuple[int, int], *,
          port: int = 8142, open_browser: bool = True):
    """Start a local HTTP server serving the interactive HTML viewer."""
    html = generate_html(prof, device, trim)
    _ViewerHandler.html_bytes = html.encode("utf-8")

    server = HTTPServer(("127.0.0.1", port), _ViewerHandler)
    open_url = f"http://127.0.0.1:{server.server_address[1]}" if open_browser else None
    _run_server(server, open_url, prof)


# ── Mode 2: Horizontal timeline viewer ──────────────────────────

def serve_timeline(prof, device: int, trim: tuple[int, int], *,
                   port: int = 8144, open_browser: bool = True):
    """Start a local HTTP server serving the horizontal timeline viewer."""
    html = generate_timeline_html(prof, device, trim)
    _ViewerHandler.html_bytes = html.encode("utf-8")

    server = HTTPServer(("127.0.0.1", port), _ViewerHandler)
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
