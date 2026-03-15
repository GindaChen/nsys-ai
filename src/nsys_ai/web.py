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

_FINDINGS_LOCK = threading.Lock()

# Bounded thread pool: fixed worker count, request queue with max size.
# Workers are released when each request finishes (finish_request + shutdown_request).
# See docs/chat-thread-pool.md.
CHAT_SERVER_POOL_SIZE = 8
CHAT_SERVER_QUEUE_SIZE = 16
_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


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
from .viewer import (  # noqa: E402
    build_timeline_gpu_data,
    generate_evidence_html,
    generate_html,
    generate_timeline_html,
)

# ── Shared helpers ───────────────────────────────────────────────


def _run_server(server, open_url, prof):
    """Run an HTTPServer with browser-open and graceful shutdown."""
    actual_port = server.server_address[1]
    actual_url = f"http://127.0.0.1:{actual_port}"
    print(f"Serving at {actual_url}")
    pool_size = getattr(server, "_pool_size", None)
    if pool_size is not None:
        print(
            f"  (thread pool: {pool_size} workers, queue max {getattr(server, '_queue_maxsize', '?')})"
        )
    if os.environ.get("SSH_CONNECTION"):
        print(
            f"  Remote/SSH: on your local machine run:  ssh -L {actual_port}:127.0.0.1:{actual_port} <host>  then open the URL in your local browser."
        )
    print("Press Ctrl-C to stop.")
    if open_url:
        open_target = (
            actual_url if (open_url and open_url.startswith("http://127.0.0.1:")) else open_url
        )
        threading.Timer(0.3, webbrowser.open, args=(open_target,)).start()
    # Ensure Ctrl-C works without deadlocking BaseServer.shutdown().
    # shutdown() must be called from a different thread than serve_forever().
    _stopping = False

    def _sigint_handler(sig, frame):
        nonlocal _stopping
        if _stopping:
            return
        _stopping = True
        print("\nShutting down.")
        threading.Thread(target=server.shutdown, daemon=True).start()

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
    return {
        "content": "Mock reply. Configure an LLM endpoint (e.g. pip install nsys-ai[ai], set ANTHROPIC_API_KEY) for real analysis.",
        "actions": [],
    }


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
    prof = None  # set by serve_timeline
    devices: list = []  # set by serve_timeline
    _prebuilt_data: list = []  # pre-built timeline payload per GPU
    _prebuilt_nvtx_mode: str = "full"  # "full" (prebuilt has NVTX) or "tile" (compute per tile)
    _tile_nvtx_cache: dict = {}  # (start_ns, end_ns, devices_tuple) -> {gpu_id: [nvtx_spans]}
    _asset_cache: dict[str, bytes] = {}
    _findings: list[dict] = []  # mutable findings state for evidence overlay

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/assets/timeline.css":
            self._serve_asset("timeline.css", "text/css; charset=utf-8")
            return
        if path == "/assets/timeline.js":
            self._serve_asset("timeline.js", "application/javascript; charset=utf-8")
            return
        if path == "/api/models":
            try:
                import nsys_ai.chat as chat_mod

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
        if path == "/api/findings":
            with _FINDINGS_LOCK:
                self._json_response(list(self._findings))
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(self.html_bytes)))
        self.end_headers()
        self.wfile.write(self.html_bytes)

    def _serve_asset(self, filename: str, content_type: str):
        """Serve static timeline assets from package templates directory."""
        body = self.__class__._asset_cache.get(filename)
        if body is None:
            try:
                path = os.path.join(_TEMPLATE_DIR, filename)
                with open(path, "rb") as f:
                    body = f.read()
                self.__class__._asset_cache[filename] = body
            except OSError:
                self.send_error(404)
                return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
                label += f" - {info.name} ({info.pci_bus}), {info.sm_count} SMs, {info.memory_bytes / 1e9:.0f}GB"
            gpu_infos.append({"id": dev, "label": label})
        # Get profile time range from kernel metadata (min_start_ns, max_end_ns)
        t_start, t_end = prof.meta.time_range
        self._json_response(
            {
                "time_range_ns": [t_start, t_end],
                "gpus": gpu_infos,
                "device_ids": devices,
            }
        )

    def _handle_data(self):
        """Return kernel/NVTX data for a requested time window (from pre-built cache)."""
        from urllib.parse import parse_qs, urlparse

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
        nvtx_requested = str(qs.get("nvtx", ["0"])[0]).lower() in ("1", "true", "yes")
        kernels_requested = str(qs.get("kernels", ["1"])[0]).lower() not in ("0", "false", "no")
        gpu_filter = None
        try:
            gpu_filter_raw = qs.get("gpu", [None])[0]
            if gpu_filter_raw is not None and str(gpu_filter_raw).strip() != "":
                gpu_filter = int(gpu_filter_raw)
        except (ValueError, TypeError):
            gpu_filter = None
        start_ns = int(start_s * 1e9)
        end_ns = int(end_s * 1e9)
        t0 = _time.monotonic()
        print(
            f"[tile] {start_s:.1f}s–{end_s:.1f}s  filtering "
            f"(kernels={'1' if kernels_requested else '0'}, nvtx={'1' if nvtx_requested else '0'}, "
            f"gpu={gpu_filter if gpu_filter is not None else 'all'})...",
            flush=True,
        )
        try:
            nvtx_spans_by_gpu = None
            if self.__class__._prebuilt_nvtx_mode == "tile" and nvtx_requested:
                annotate_devices = (
                    [gpu_filter]
                    if gpu_filter is not None and gpu_filter in self.__class__.devices
                    else self.__class__.devices
                )
                devices_key = tuple(annotate_devices)
                tile_key = (start_ns, end_ns, devices_key)
                nvtx_spans_by_gpu = self.__class__._tile_nvtx_cache.get(tile_key)
                if nvtx_spans_by_gpu is None:
                    t_nv = _time.monotonic()
                    print(f"[tile] {start_s:.1f}s–{end_s:.1f}s  NVTX annotate...", flush=True)
                    tile_nvtx_entries = build_timeline_gpu_data(
                        self.__class__.prof,
                        annotate_devices,
                        (start_ns, end_ns),
                        include_kernels=False,
                        include_nvtx=True,
                    )
                    nvtx_spans_by_gpu = {
                        e["id"]: e.get("nvtx_spans", []) for e in tile_nvtx_entries
                    }
                    self.__class__._tile_nvtx_cache[tile_key] = nvtx_spans_by_gpu
                    # Keep memory bounded (simple LRU via insertion-ordered dict semantics).
                    while len(self.__class__._tile_nvtx_cache) > 64:
                        self.__class__._tile_nvtx_cache.pop(
                            next(iter(self.__class__._tile_nvtx_cache))
                        )
                    print(
                        f"[tile] {start_s:.1f}s–{end_s:.1f}s  NVTX done in "
                        f"{_time.monotonic() - t_nv:.3f}s",
                        flush=True,
                    )

            # Filter pre-built data by time window
            gpu_entries = []
            for gpu_data in prebuilt:
                if "kernels" in gpu_data:
                    filtered = _filter_timeline_gpu_entry(
                        gpu_data,
                        start_ns,
                        end_ns,
                        filter_kernels=kernels_requested,
                        filter_nvtx=self.__class__._prebuilt_nvtx_mode == "full",
                    )
                    if nvtx_spans_by_gpu is not None:
                        filtered["nvtx_spans"] = nvtx_spans_by_gpu.get(filtered["id"], [])
                    gpu_entries.append(filtered)
                else:
                    # Backward-compatible fallback for older in-memory format.
                    filtered = _filter_nodes_by_time(gpu_data["data"], start_ns, end_ns)
                    gpu_entries.append({"id": gpu_data["id"], "data": filtered})
            data_json = json.dumps({"gpus": gpu_entries})
            body = data_json.encode("utf-8")
            elapsed = _time.monotonic() - t0
            print(
                f"[tile] {start_s:.1f}s–{end_s:.1f}s  done in {elapsed:.3f}s  ({len(body) // 1024}KB)",
                flush=True,
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
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

    def _handle_analyze(self):
        """POST /api/analyze — run EvidenceBuilder, replace all findings."""
        try:
            from .evidence_builder import EvidenceBuilder

            device = self.devices[0] if self.devices else 0
            builder = EvidenceBuilder(self.prof, device=device)
            report = builder.build()
            with _FINDINGS_LOCK:
                self.__class__._findings = [f.to_dict() for f in report.findings]
                findings = list(self.__class__._findings)
            print(
                f"[analyze] Generated {len(findings)} finding(s)",
                flush=True,
            )
            self._json_response(findings)
        except Exception as e:
            print(f"[analyze] Error: {e}", flush=True)
            self._json_response({"error": str(e)}, 500)

    def _handle_post_finding(self):
        """POST /api/findings — append a single finding (from chat agent)."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(content_length) if content_length else b"{}"
            finding_dict = json.loads(raw.decode("utf-8"))
            with _FINDINGS_LOCK:
                self.__class__._findings.append(finding_dict)
                idx = len(self.__class__._findings)
            self._json_response({"index": idx})
        except Exception as e:
            self._json_response({"error": str(e)}, 400)

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/api/analyze":
            self._handle_analyze()
            return
        if path == "/api/findings":
            self._handle_post_finding()
            return
        if path != "/api/chat":
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
                self.send_header("Connection", "close")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                for chunk in gen:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                self.close_connection = True
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


def serve(prof, device: int, trim: tuple[int, int], *, port: int = 8142, open_browser: bool = True):
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


def _filter_timeline_gpu_entry(
    gpu_entry: dict,
    start_ns: int,
    end_ns: int,
    *,
    filter_kernels: bool = True,
    filter_nvtx: bool = True,
) -> dict:
    """Filter kernel-first timeline payload to a time window."""
    if filter_kernels:
        kernels = [
            k
            for k in gpu_entry.get("kernels", [])
            if k.get("end_ns", 0) >= start_ns and k.get("start_ns", 0) <= end_ns
        ]
    else:
        kernels = []
    if filter_nvtx:
        nvtx_spans = [
            s
            for s in gpu_entry.get("nvtx_spans", [])
            if s.get("end", 0) >= start_ns and s.get("start", 0) <= end_ns
        ]
    else:
        nvtx_spans = []
    return {"id": gpu_entry.get("id"), "kernels": kernels, "nvtx_spans": nvtx_spans}


def serve_timeline(
    prof,
    device,
    trim: tuple[int, int] | None = None,
    *,
    port: int = 8144,
    open_browser: bool = True,
    findings_path: str | None = None,
    auto_findings: list[dict] | None = None,
):
    """Start a local HTTP server serving the horizontal timeline viewer.

    If *trim* is None, the initial view shows a default 5s window and
    the client can freely navigate via /api/data.
    If *findings_path* is given, findings are loaded and rendered as overlays.
    If *auto_findings* is given, they are used directly (from --auto-analyze).
    """
    from collections.abc import Sequence

    devices: list[int] = list(device) if isinstance(device, Sequence) else [device]

    # Load findings if provided
    findings_data = auto_findings  # from --auto-analyze
    if findings_path and not findings_data:
        from .annotation import load_findings

        report = load_findings(findings_path)
        findings_data = [f.to_dict() for f in report.findings]
        print(f"Loaded {len(findings_data)} finding(s) from {findings_path}", flush=True)

    # Store prof + devices on handler for /api/meta queries
    _ViewerHandler.prof = prof
    _ViewerHandler.devices = devices
    _ViewerHandler._tile_nvtx_cache = {}
    _ViewerHandler._findings = findings_data or []

    # Resolve profile path for chat agent DB access
    _profile_path = prof.path if hasattr(prof, "path") else ""

    if trim is not None:
        # Legacy: render full HTML with all data baked in
        html = generate_timeline_html(prof, devices, trim, findings_data=findings_data, profile_path=_profile_path)
        _ViewerHandler._prebuilt_nvtx_mode = "full"
    else:
        # Progressive: generate shell HTML, data fetched via /api/data
        html = generate_timeline_html(prof, devices, None, findings_data=findings_data, profile_path=_profile_path)
        _ViewerHandler._prebuilt_nvtx_mode = "tile"

    _ViewerHandler.html_bytes = html.encode("utf-8")

    # Pre-build full kernel-first timeline payload for all GPUs (progressive mode)
    if trim is None:
        import os

        db_path = prof.path if hasattr(prof, "path") else ""
        cache_path = db_path + ".timeline-cache-v3-kernels.json" if db_path else ""
        cache_valid = False

        # Try loading from disk cache
        if cache_path and os.path.exists(cache_path):
            try:
                src_mtime = os.path.getmtime(db_path)
                cache_mtime = os.path.getmtime(cache_path)
                if cache_mtime >= src_mtime:
                    t0 = _time.monotonic()
                    print(
                        f"Loading cached timeline payload from {os.path.basename(cache_path)}...",
                        flush=True,
                    )
                    with open(cache_path) as f:
                        prebuilt = json.loads(f.read())
                    if not (
                        isinstance(prebuilt, list)
                        and prebuilt
                        and isinstance(prebuilt[0], dict)
                        and "kernels" in prebuilt[0]
                    ):
                        raise ValueError("stale timeline cache format")
                    elapsed = _time.monotonic() - t0
                    print(
                        f"Cache loaded in {elapsed:.2f}s ({os.path.getsize(cache_path) // 1024}KB)",
                        flush=True,
                    )
                    _ViewerHandler._prebuilt_data = prebuilt
                    cache_valid = True
            except Exception as e:
                print(f"Cache load failed: {e}, rebuilding...", flush=True)

        if not cache_valid:
            t0 = _time.monotonic()
            full_range = prof.meta.time_range
            print(
                f"Pre-building kernels only for {len(devices)} GPU(s) "
                f"({full_range[0] / 1e9:.1f}s–{full_range[1] / 1e9:.1f}s)...",
                flush=True,
            )
            prebuilt = build_timeline_gpu_data(
                prof,
                devices,
                full_range,
                include_kernels=True,
                include_nvtx=False,
            )
            for gpu_entry in prebuilt:
                print(
                    f"  GPU {gpu_entry['id']}: {len(gpu_entry.get('kernels', []))} kernels, "
                    f"{len(gpu_entry.get('nvtx_spans', []))} NVTX spans",
                    flush=True,
                )
            elapsed = _time.monotonic() - t0
            print(f"Pre-build complete in {elapsed:.1f}s", flush=True)
            _ViewerHandler._prebuilt_data = prebuilt

            # Save to disk cache
            if cache_path:
                try:
                    t0 = _time.monotonic()
                    with open(cache_path, "w") as f:
                        f.write(json.dumps(prebuilt))
                    sz = os.path.getsize(cache_path)
                    print(
                        f"Saved cache to {os.path.basename(cache_path)} ({sz // 1024}KB, {_time.monotonic() - t0:.1f}s)",
                        flush=True,
                    )
                except Exception as e:
                    print(f"Cache save failed: {e}", flush=True)

    server = _ThreadedHTTPServer(("127.0.0.1", port), _ViewerHandler)
    actual_url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Timeline viewer at {actual_url}")
    _run_server(server, actual_url if open_browser else None, prof)


# ── Mode 3: Evidence View ────────────────────────────────────────


class _EvidenceHandler(BaseHTTPRequestHandler):
    """Serve the Evidence View HTML; GET /api/data for progressive kernel tiles."""

    html_bytes: bytes = b""
    prof = None
    devices: list = []
    _prebuilt_data: list = []
    _prebuilt_nvtx_mode: str = "full"
    _tile_nvtx_cache: dict = {}
    _asset_cache: dict[str, bytes] = {}

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/assets/evidence.css":
            # Reuse the existing timeline.css asset for evidence CSS.
            self._serve_asset("timeline.css", "text/css; charset=utf-8")
            return
        if path == "/assets/evidence.js":
            # Reuse the existing timeline.js asset for evidence JS.
            self._serve_asset("timeline.js", "application/javascript; charset=utf-8")
            return
        if path == "/api/data":
            self._handle_data()
            return
        # Default: serve evidence HTML
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(self.html_bytes)))
        self.end_headers()
        self.wfile.write(self.html_bytes)

    def _serve_asset(self, filename: str, content_type: str):
        body = self.__class__._asset_cache.get(filename)
        if body is None:
            try:
                path = os.path.join(_TEMPLATE_DIR, filename)
                with open(path, "rb") as f:
                    body = f.read()
                self.__class__._asset_cache[filename] = body
            except OSError:
                self.send_error(404)
                return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_response(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_data(self):
        """Return kernel data for a time window from this handler's prebuilt cache."""
        from urllib.parse import parse_qs, urlparse

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
        try:
            gpu_entries = []
            for gpu_data in prebuilt:
                if "kernels" in gpu_data:
                    filtered = _filter_timeline_gpu_entry(gpu_data, start_ns, end_ns)
                    gpu_entries.append(filtered)
            data_json = json.dumps({"gpus": gpu_entries})
            body = data_json.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def log_message(self, format, *args):
        pass


def serve_evidence(
    prof,
    device,
    findings_data: list[dict],
    title: str = "Evidence View",
    *,
    port: int = 8146,
    open_browser: bool = True,
):
    """Start a local HTTP server serving the Evidence View page.

    *findings_data* is a list of Finding dicts (from annotation.py).
    """
    from collections.abc import Sequence

    devices: list[int] = list(device) if isinstance(device, Sequence) else [device]

    html = generate_evidence_html(prof, devices, findings_data, title)
    _EvidenceHandler.html_bytes = html.encode("utf-8")

    # Set up progressive tile data (reuse _ViewerHandler's prebuilt data)
    _ViewerHandler.prof = prof
    _ViewerHandler.devices = devices
    _ViewerHandler._tile_nvtx_cache = {}

    # Pre-build kernel data for tile serving
    t0 = _time.monotonic()
    full_range = prof.meta.time_range
    print(f"Pre-building kernels for evidence view ({len(devices)} GPU(s))...", flush=True)
    prebuilt = build_timeline_gpu_data(
        prof, devices, full_range, include_kernels=True, include_nvtx=False
    )
    for gpu_entry in prebuilt:
        print(
            f"  GPU {gpu_entry['id']}: {len(gpu_entry.get('kernels', []))} kernels",
            flush=True,
        )
    elapsed = _time.monotonic() - t0
    print(f"Pre-build complete in {elapsed:.1f}s", flush=True)
    _EvidenceHandler._prebuilt_data = prebuilt
    _ViewerHandler._prebuilt_nvtx_mode = "full"

    server = _ThreadedHTTPServer(("127.0.0.1", port), _EvidenceHandler)
    actual_url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Evidence viewer at {actual_url}")
    print(f"  {len(findings_data)} finding(s): {title}")
    _run_server(server, actual_url if open_browser else None, prof)


# ── Mode 4: Perfetto UI ─────────────────────────────────────────



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


def serve_perfetto(
    prof, device: int, trim: tuple[int, int], *, port: int = 8143, open_browser: bool = True
):
    """Generate Perfetto JSON, serve it locally, and open ui.perfetto.dev."""
    events = gpu_trace(prof, device, trim)
    trace = json.dumps({"traceEvents": events, "displayTimeUnit": "ms"})
    _PerfettoHandler.trace_bytes = trace.encode("utf-8")

    nk = sum(1 for e in events if e.get("cat") == "gpu_kernel")
    nn = sum(1 for e in events if e.get("cat") == "nvtx_projected")
    print(f"Trace: {nk} kernels, {nn} NVTX, {len(trace) // 1024} KB")

    server = HTTPServer(("127.0.0.1", port), _PerfettoHandler)
    actual_port = server.server_address[1]
    trace_url = f"http://127.0.0.1:{actual_port}/trace.json"
    perfetto_url = f"https://ui.perfetto.dev/#!/?url={quote(trace_url, safe='')}"

    print(f"Perfetto UI: {perfetto_url}")
    _run_server(server, perfetto_url if open_browser else None, prof)
