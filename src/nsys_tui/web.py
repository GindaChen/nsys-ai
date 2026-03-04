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
import signal
import socketserver
import threading
import time as _time
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
    allow_reuse_address = True

from .export import gpu_trace  # noqa: E402
from .viewer import generate_html, generate_timeline_html  # noqa: E402

# ── Shared helpers ───────────────────────────────────────────────

def _run_server(server, open_url, prof):
    """Run an HTTPServer with browser-open and graceful shutdown."""
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
        open_target = actual_url if (open_url and open_url.startswith("http://127.0.0.1:")) else open_url
        threading.Timer(0.3, webbrowser.open, args=(open_target,)).start()
    # Ensure Ctrl-C always works
    def _sigint_handler(sig, frame):
        print("\nShutting down.")
        server.shutdown()
    signal.signal(signal.SIGINT, _sigint_handler)
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
    """Serve the pre-rendered HTML on GET; GET /api/models for model list;
    GET /api/data for on-demand tile data; GET /api/meta for profile metadata;
    POST /api/chat for AI chat."""
    html_bytes: bytes = b""
    prof = None           # set by serve_timeline
    devices: list = []    # set by serve_timeline
    _prebuilt_data: list = []  # pre-built full NVTX tree per GPU

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
            self._json_response({"default": default, "options": options})
            return
        if path == "/api/meta":
            self._handle_meta()
            return
        if path == "/api/data":
            self._handle_data()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(self.html_bytes)))
        self.end_headers()
        self.wfile.write(self.html_bytes)

    def _handle_meta(self):
        """Return profile metadata: time range, GPU list, device count."""
        prof = self.__class__.prof
        devices = self.__class__.devices
        if not prof:
            self._json_response({"error": "no profile"}, 500)
            return
        gpu_infos = []
        for dev in devices:
            info = prof.meta.gpu_info.get(dev)
            label = f"GPU {dev}"
            if info:
                label += f" - {info.name} ({info.pci_bus}), {info.sm_count} SMs, {info.memory_bytes/1e9:.0f}GB"
            gpu_infos.append({"id": dev, "label": label})
        # Get profile time range from kernel metadata (min_start_ns, max_end_ns)
        t_start, t_end = prof.meta.time_range
        self._json_response({
            "time_range_ns": [t_start, t_end],
            "gpus": gpu_infos,
            "device_ids": devices,
        })

    def _handle_data(self):
        """Return kernel/NVTX data for a requested time window (from pre-built cache)."""
        from urllib.parse import urlparse, parse_qs
        prebuilt = self.__class__._prebuilt_data
        if not prebuilt:
            self._json_response({"error": "no prebuilt data"}, 500)
            return
        qs = parse_qs(urlparse(self.path).query)
        try:
            start_s = float(qs.get("start_s", [0])[0])
            end_s = float(qs.get("end_s", [5])[0])
        except (ValueError, IndexError):
            start_s, end_s = 0, 5
        start_ns = int(start_s * 1e9)
        end_ns = int(end_s * 1e9)
        t0 = _time.monotonic()
        print(f"[tile] {start_s:.1f}s–{end_s:.1f}s  filtering...", flush=True)
        try:
            # Filter pre-built data by time window
            gpu_entries = []
            for gpu_data in prebuilt:
                filtered = _filter_nodes_by_time(gpu_data["data"], start_ns, end_ns)
                gpu_entries.append({"id": gpu_data["id"], "data": filtered})
            data_json = json.dumps({"gpus": gpu_entries})
            body = data_json.encode("utf-8")
            elapsed = _time.monotonic() - t0
            print(f"[tile] {start_s:.1f}s–{end_s:.1f}s  done in {elapsed:.3f}s  ({len(body)//1024}KB)", flush=True)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            elapsed = _time.monotonic() - t0
            print(f"[tile] {start_s:.1f}s–{end_s:.1f}s  ERROR in {elapsed:.2f}s: {e}", flush=True)
            self._json_response({"error": str(e)}, 500)

    def _json_response(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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

def _filter_nodes_by_time(nodes: list, start_ns: int, end_ns: int) -> list:
    """Filter a tree of nodes, keeping only those overlapping [start_ns, end_ns]."""
    result = []
    for node in nodes:
        ns = node.get("start_ns", 0)
        ne = node.get("end_ns", 0)
        # Skip if entirely outside the window
        if ne < start_ns or ns > end_ns:
            continue
        # Include this node; recursively filter children
        filtered = dict(node)
        if "children" in filtered and filtered["children"]:
            filtered["children"] = _filter_nodes_by_time(filtered["children"], start_ns, end_ns)
        result.append(filtered)
    return result


def serve_timeline(prof, device, trim: tuple[int, int] | None = None, *,
                   port: int = 8144, open_browser: bool = True):
    """Start a local HTTP server serving the horizontal timeline viewer.

    If *trim* is None, the initial view shows a default 5s window and
    the client can freely navigate via /api/data.
    """
    from typing import Sequence
    from .nvtx_tree import build_nvtx_tree, to_json
    devices: list[int] = list(device) if isinstance(device, Sequence) else [device]

    # Store prof + devices on handler for /api/meta queries
    _ViewerHandler.prof = prof
    _ViewerHandler.devices = devices

    if trim is not None:
        # Legacy: render full HTML with all data baked in
        html = generate_timeline_html(prof, devices, trim)
    else:
        # Progressive: generate shell HTML, data fetched via /api/data
        html = generate_timeline_html(prof, devices, None)

    _ViewerHandler.html_bytes = html.encode("utf-8")

    # Pre-build full NVTX tree for all GPUs (progressive mode)
    if trim is None:
        import os, hashlib
        db_path = prof.path if hasattr(prof, 'path') else ''
        cache_path = db_path + '.timeline-cache.json' if db_path else ''
        cache_valid = False

        # Try loading from disk cache
        if cache_path and os.path.exists(cache_path):
            try:
                src_mtime = os.path.getmtime(db_path)
                cache_mtime = os.path.getmtime(cache_path)
                if cache_mtime >= src_mtime:
                    t0 = _time.monotonic()
                    print(f"Loading cached NVTX tree from {os.path.basename(cache_path)}...", flush=True)
                    with open(cache_path, 'r') as f:
                        prebuilt = json.loads(f.read())
                    elapsed = _time.monotonic() - t0
                    print(f"Cache loaded in {elapsed:.2f}s ({os.path.getsize(cache_path) // 1024}KB)", flush=True)
                    _ViewerHandler._prebuilt_data = prebuilt
                    cache_valid = True
            except Exception as e:
                print(f"Cache load failed: {e}, rebuilding...", flush=True)

        if not cache_valid:
            t0 = _time.monotonic()
            full_range = prof.meta.time_range
            print(f"Pre-building NVTX tree for {len(devices)} GPU(s) "
                  f"({full_range[0]/1e9:.1f}s–{full_range[1]/1e9:.1f}s)...", flush=True)
            prebuilt = []
            for dev in devices:
                dt = _time.monotonic()
                roots = build_nvtx_tree(prof, dev, full_range)
                tree_json = to_json(roots)
                elapsed_dev = _time.monotonic() - dt
                print(f"  GPU {dev}: {len(tree_json)} roots, {elapsed_dev:.1f}s", flush=True)
                prebuilt.append({"id": dev, "data": tree_json})
            elapsed = _time.monotonic() - t0
            print(f"Pre-build complete in {elapsed:.1f}s", flush=True)
            _ViewerHandler._prebuilt_data = prebuilt

            # Save to disk cache
            if cache_path:
                try:
                    t0 = _time.monotonic()
                    with open(cache_path, 'w') as f:
                        f.write(json.dumps(prebuilt))
                    sz = os.path.getsize(cache_path)
                    print(f"Saved cache to {os.path.basename(cache_path)} ({sz // 1024}KB, {_time.monotonic() - t0:.1f}s)", flush=True)
                except Exception as e:
                    print(f"Cache save failed: {e}", flush=True)

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
