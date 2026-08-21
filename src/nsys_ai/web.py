"""
web.py — Serve profiles via local HTTP servers.

Provides:
  1. `serve`          — Serve the built-in interactive HTML viewer.

Usage:
    nsys-ai web      profile.sqlite --gpu 0 --trim 39 42
"""

import gzip
import json
import logging
import os
import queue
import re
import signal
import socketserver
import threading
import time as _time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

from .gpu_label import format_gpu_label

_log = logging.getLogger(__name__)

_FINDINGS_LOCK = threading.Lock()
_LOOP_LOCK = threading.Lock()
# The progressive timeline starts serving kernels immediately.  NVTX is built
# in the background so the first interactive request does not make a cold
# load pay the full-profile annotation cost on the request thread.

# Bounded thread pool: fixed worker count, request queue with max size.
# Workers are released when each request finishes (finish_request + shutdown_request).
# See docs/chat-thread-pool.md.
CHAT_SERVER_POOL_SIZE = 8
CHAT_SERVER_QUEUE_SIZE = 16
_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _template_asset_version() -> str:
    """Cache-bust token from template mtimes (changes when timeline UI is edited)."""
    latest = 0.0
    for name in ("timeline.html", "timeline.css", "timeline.js", "tokens.css"):
        path = os.path.join(_TEMPLATE_DIR, name)
        try:
            latest = max(latest, os.path.getmtime(path))
        except OSError:
            pass
    return str(int(latest)) if latest else "0"


def _versioned_asset_url(path: str) -> str:
    return f"{path}?v={_template_asset_version()}"


def _find_tree_path(nodes: list[dict], path: str) -> dict | None:
    """Find a serialized NVTX node by its stable display path."""
    for node in nodes:
        if node.get("path") == path:
            return node
        children = node.get("children") or []
        found = _find_tree_path(children, path)
        if found is not None:
            return found
    return None


def _slice_tree_nodes(nodes: list[dict], depth: int | None) -> list[dict]:
    """Copy a tree to *depth*, preserving a marker for unloaded children."""
    result = []
    for node in nodes:
        copied = {key: value for key, value in node.items() if key != "children"}
        children = node.get("children") or []
        if children:
            if depth is None or depth > 0:
                copied["children"] = _slice_tree_nodes(
                    children, None if depth is None else depth - 1
                )
            else:
                copied["children"] = []
                copied["has_children"] = True
        result.append(copied)
    return result


#: Below this, the gzip header costs more than the compression saves.
_MIN_COMPRESS_BYTES = 1024

#: Level 1, not the default 9. Measured on a 932 KB kernel window: level 1 gives
#: 11.2% of the original in 2.9 ms, level 6 gives 7.9% in 6.3 ms, level 9 gives
#: 7.2% in 27.2 ms. The whole request currently takes ~7 ms, and the common case
#: is a browser on the same machine, so the 30 KB that level 6 saves does not pay
#: for the CPU it costs. Compression runs on the request thread.
_COMPRESS_LEVEL = 1


def _send_body(
    handler,
    body: bytes,
    content_type: str,
    status: int = 200,
    extra_headers: dict[str, str] | None = None,
) -> None:
    """Write a response body, compressed when the client said it can read it.

    Timeline payloads are highly repetitive JSON — a window of kernels compresses
    to roughly 7% of its size — and the server prints an SSH port-forward hint on
    startup, so it is expected to be read over a link where those bytes are the
    whole cost. Compression is negotiated, never assumed: a client that does not
    advertise gzip gets the plain body.
    """
    encoding = None
    accepted = handler.headers.get("Accept-Encoding", "") if handler.headers else ""
    if len(body) >= _MIN_COMPRESS_BYTES and "gzip" in accepted.lower():
        compressed = gzip.compress(body, compresslevel=_COMPRESS_LEVEL)
        # Refuse a "compression" that made it bigger (already-compressed content).
        if len(compressed) < len(body):
            body = compressed
            encoding = "gzip"
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    for name, value in (extra_headers or {}).items():
        handler.send_header(name, value)
    if encoding:
        handler.send_header("Content-Encoding", encoding)
    # Caches must not serve a gzipped body to a client that cannot read it.
    handler.send_header("Vary", "Accept-Encoding")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class _HeadBodySuppressor:
    """Preserve response headers while discarding a HEAD response body."""

    def __init__(self, wfile):
        self._wfile = wfile

    def write(self, body):
        return len(body)

    def __getattr__(self, name):
        return getattr(self._wfile, name)


class _HeadRequestMixin:
    """Route HEAD through GET while keeping its response body empty."""

    _head_only = False

    def do_HEAD(self):
        original_wfile = self.wfile
        self._head_only = True
        try:
            self.do_GET()
        finally:
            self.wfile = original_wfile
            self._head_only = False

    def end_headers(self):
        super().end_headers()
        if self._head_only:
            self.wfile = _HeadBodySuppressor(self.wfile)


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
            except OSError as exc:
                _log.debug("Pool worker OS error: %s", exc, exc_info=True)
            except Exception:
                _log.error("Unexpected pool worker error", exc_info=True)


class _ThreadedHTTPServer(_ThreadPoolMixIn, socketserver.ThreadingMixIn, HTTPServer):
    """Concurrent chat requests via bounded thread pool; workers released after each request."""

    daemon_threads = True
    allow_reuse_address = True


from .viewer import (  # noqa: E402
    build_timeline_gpu_data,
    build_timeline_gpu_data_lod,
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
    return None


def _handle_ask_request(body_bytes: bytes) -> dict | None:
    """Handle the shared-runner Web ask transport."""
    try:
        from . import chat

        if hasattr(chat, "ask_completion"):
            return chat.ask_completion(body_bytes)
    except ImportError:
        pass
    return None


def _handle_ask_stream_request(body_bytes: bytes):
    """Return the shared-runner ask SSE generator, or None when unavailable."""
    try:
        from . import chat

        if hasattr(chat, "ask_completion_stream"):
            return chat.ask_completion_stream(body_bytes)
    except ImportError:
        pass
    return None


def _handle_chat_stream(body_bytes: bytes):
    """Return generator yielding SSE bytes for stream=true, or None for 501."""
    try:
        from . import chat

        if hasattr(chat, "chat_completion_stream"):
            return chat.chat_completion_stream(body_bytes)
    except ImportError:
        pass
    return None


class _ViewerHandler(_HeadRequestMixin, BaseHTTPRequestHandler):
    """Serve the pre-rendered HTML on GET; GET /api/models for model list;
    GET /api/data for on-demand tile data (with optional max_buckets LOD);
    GET /api/meta for profile metadata;
    POST /api/chat for AI chat."""

    html_bytes: bytes = b""
    prof = None  # set by serve_timeline
    devices: list = []  # set by serve_timeline
    _prebuilt_data: list = []  # pre-built timeline payload per GPU
    _prebuilt_nvtx_mode: str = "full"  # "full" (baked) or "background" (progressive)
    _full_nvtx_by_gpu: dict = {}  # device id -> full-range spans, sliced per request
    _overview_bins: list[float] = []  # full-profile kernel activity for the overview strip
    _overview_kernel_count: int = 0
    _profile_id: str = ""
    _nvtx_prebuild_done: threading.Event | None = None
    _nvtx_prebuild_error: str | None = None
    _asset_cache: dict[str, tuple[float, bytes]] = {}  # filename -> (mtime, body)
    _findings: list[dict] = []  # mutable findings state for evidence overlay
    _session_id: str | None = None  # SessionStore id (always set by serve_timeline)
    _session_root: str = ".nsys-ai/sessions"
    _trim: tuple[int, int] | None = None
    _tree_data: list[dict] = []
    _tree_configured: bool = False
    _tree_build_done: threading.Event | None = None
    _tree_build_error: str | None = None
    _tree_device: int | None = None
    _tree_trim: tuple[int, int] | None = None

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/assets/timeline.css":
            self._serve_asset("timeline.css", "text/css; charset=utf-8")
            return
        if path == "/assets/tokens.css":
            self._serve_asset("tokens.css", "text/css; charset=utf-8")
            return
        if path == "/assets/timeline.js":
            self._serve_asset("timeline.js", "application/javascript; charset=utf-8")
            return
        if path == "/api/models":
            try:
                import nsys_ai.chat as chat_mod

                options = chat_mod.get_available_models()
                default = chat_mod.get_default_model()
            except Exception as exc:
                _log.debug("Model listing unavailable: %s", exc, exc_info=True)
                options = []
                default = None
            self._json_response({"default": default, "options": options})
            return
        if path == "/api/tree":
            self._handle_tree()
            return
        if path == "/api/meta":
            self._handle_meta()
            return
        if path == "/api/data":
            self._handle_data()
            return
        if path == "/api/search":
            self._handle_search()
            return
        if path == "/api/findings":
            if self.__class__._session_id is not None:
                try:
                    self._json_response(self._session_findings_payload())
                except Exception as e:
                    _log.exception("Error loading session findings")
                    self._json_response({"error": str(e)}, 500)
                return
            with _FINDINGS_LOCK:
                self._json_response(list(self._findings))
            return
        if path == "/api/loop/state":
            self._handle_loop_get()
            return
        if path.startswith("/api/"):
            self._json_response({"error": "not found", "path": path}, 404)
            return
        if path not in {"/", "/index.html"}:
            self.send_error(404)
            return
        _send_body(
            self,
            self.html_bytes,
            "text/html; charset=utf-8",
            extra_headers={"Cache-Control": "no-cache, must-revalidate"},
        )

    def _serve_asset(self, filename: str, content_type: str):
        """Serve static timeline assets; reload when template files change on disk."""
        path = os.path.join(_TEMPLATE_DIR, filename)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            self.send_error(404)
            return
        cache = self.__class__._asset_cache
        cached = cache.get(filename)
        if cached is None or cached[0] != mtime:
            with open(path, "rb") as f:
                body = f.read()
            cache[filename] = (mtime, body)
        else:
            body = cached[1]
        # timeline.js is ~150 KB and timeline.css ~40 KB of text; both are paid on
        # every cold load and compress to a fraction of that. Cache-Control stays
        # as it was: the ?v= token busts the cache, so the browser must revalidate.
        _send_body(
            self,
            body,
            content_type,
            extra_headers={"Cache-Control": "no-cache, must-revalidate"},
        )

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
            gpu_infos.append({"id": dev, "label": format_gpu_label(dev, info)})
        # Get profile time range from kernel metadata (min_start_ns, max_end_ns)
        t_start, t_end = self.__class__._trim or prof.meta.time_range
        self._json_response(
            {
                "time_range_ns": [t_start, t_end],
                "gpus": gpu_infos,
                "device_ids": devices,
                "overview_bins": list(self.__class__._overview_bins),
                "overview_kernel_count": self.__class__._overview_kernel_count,
                "profile_id": self.__class__._profile_id,
            }
        )

    def _handle_tree(self):
        """Return a bounded tree slice for the lazy NVTX tree viewer."""
        from urllib.parse import parse_qs, urlparse

        if self.__class__._tree_build_error:
            self._json_response({"error": "tree build failed"}, 500)
            return
        if not self.__class__._tree_configured:
            self._json_response(
                {"status": "building", "message": "tree data is still building"},
                202,
            )
            return
        qs = parse_qs(urlparse(self.path).query)
        path = str(qs.get("path", [""])[0])
        depth_raw = str(qs.get("depth", ["2"])[0]).lower()
        if depth_raw == "full":
            depth = None
        else:
            try:
                depth = max(0, min(8, int(depth_raw)))
            except ValueError:
                self._json_response({"error": "depth must be an integer or full"}, 400)
                return

        source = self.__class__._tree_data
        if path:
            source = _find_tree_path(source, path)
            if source is None:
                self._json_response({"error": "tree path not found", "path": path}, 404)
                return
            source = source.get("children") or []
        nodes = _slice_tree_nodes(source, depth)
        self._json_response({"nodes": nodes, "path": path, "depth": depth_raw})

    def _handle_search(self):
        """Search the complete pre-built profile, including unloaded tiles."""
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(self.path).query)
        query = str(qs.get("q", [""])[0]).strip()
        if len(query) > 256:
            self._json_response({"error": "search query is limited to 256 characters"}, 400)
            return
        try:
            limit = max(1, min(500, int(qs.get("limit", [100])[0])))
        except (TypeError, ValueError):
            limit = 100
        regex = str(qs.get("regex", ["0"])[0]).lower() in ("1", "true", "yes")
        if not query:
            self._json_response({"query": "", "matches": [], "count": 0, "truncated": False})
            return
        done = self.__class__._nvtx_prebuild_done
        if done is not None:
            done.wait()
        pattern = query
        if not regex and query.startswith("/") and query.rfind("/") > 0:
            slash = query.rfind("/")
            pattern = query[1:slash]
            regex = True
        try:
            matcher = re.compile(pattern, re.IGNORECASE) if regex else None
        except re.error as exc:
            self._json_response({"error": f"invalid regex: {exc}"}, 400)
            return
        needle = pattern.lower()
        matches = []
        total = 0
        for gpu_entry in self.__class__._prebuilt_data:
            gpu_id = gpu_entry.get("id")
            for kernel in gpu_entry.get("kernels", []):
                name = str(kernel.get("name", ""))
                if (matcher.search(name) if matcher else needle in name.lower()):
                    total += 1
                    if len(matches) < limit:
                        matches.append({
                            "kind": "kernel",
                            "gpu": gpu_id,
                            "name": name,
                            "start_ns": kernel.get("start_ns", 0),
                            "end_ns": kernel.get("end_ns", 0),
                            "stream": kernel.get("stream"),
                            "path": kernel.get("path", ""),
                        })
            spans = self.__class__._full_nvtx_by_gpu.get(gpu_id, gpu_entry.get("nvtx_spans", []))
            for span in spans:
                name = str(span.get("name", ""))
                if (matcher.search(name) if matcher else needle in name.lower()):
                    total += 1
                    if len(matches) < limit:
                        matches.append({
                            "kind": "nvtx",
                            "gpu": gpu_id,
                            "name": name,
                            "start_ns": span.get("start", 0),
                            "end_ns": span.get("end", 0),
                            "stream": None,
                            "path": span.get("path", ""),
                        })
        matches.sort(key=lambda item: (item["start_ns"], item["kind"], item["name"]))
        self._json_response({
            "query": query,
            "matches": matches,
            "count": total,
            "truncated": total > limit,
            "scope": "profile",
        })

    def _handle_data(self):
        """Return kernel/NVTX data for a requested time window (from pre-built cache)."""
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(self.path).query)
        if qs.get("resolution", [None])[0] not in (None, ""):
            self._json_response({"error": "resolution is obsolete; use max_buckets"}, 400)
            return
        prebuilt = self.__class__._prebuilt_data
        if not prebuilt:
            self._json_response({"error": "no prebuilt data"}, 500)
            return
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
        max_buckets = None
        if qs.get("max_buckets", [None])[0] not in (None, ""):
            try:
                max_buckets = max(1, min(100_000, int(qs["max_buckets"][0])))
            except (TypeError, ValueError):
                self._json_response({"error": "max_buckets must be a positive integer"}, 400)
                return
        try:
            nvtx_spans_by_gpu = None
            if self.__class__._prebuilt_nvtx_mode == "background" and nvtx_requested:
                annotate_devices = (
                    [gpu_filter]
                    if gpu_filter is not None and gpu_filter in self.__class__.devices
                    else self.__class__.devices
                )
                done = self.__class__._nvtx_prebuild_done
                if done is not None:
                    # A request arriving during the warm-up waits once for the
                    # shared result; subsequent pans and zooms only slice lists.
                    done.wait()
                if self.__class__._nvtx_prebuild_error:
                    raise RuntimeError(self.__class__._nvtx_prebuild_error)
                nvtx_spans_by_gpu = {
                    dev: _slice_nvtx_spans(
                        self.__class__._full_nvtx_by_gpu.get(dev, []), start_ns, end_ns
                    )
                    for dev in annotate_devices
                }

            lod_by_gpu = {}
            if max_buckets is not None and kernels_requested and self.__class__.prof is not None:
                try:
                    lod_by_gpu = {
                        entry["id"]: entry
                        for entry in build_timeline_gpu_data_lod(
                            self.__class__.prof,
                            self.__class__.devices,
                            (start_ns, end_ns),
                            max_buckets,
                        )
                    }
                except Exception:
                    _log.exception("DuckDB timeline LOD query failed; using cached payload")

            # Filter pre-built data by time window
            gpu_entries = []
            for gpu_data in prebuilt:
                if "kernels" in gpu_data:
                    filtered = lod_by_gpu.get(gpu_data.get("id"))
                    if filtered is None:
                        filtered = _filter_timeline_gpu_entry(
                            gpu_data,
                            start_ns,
                            end_ns,
                            filter_kernels=kernels_requested,
                            filter_nvtx=self.__class__._prebuilt_nvtx_mode == "full",
                        )
                    if nvtx_spans_by_gpu is not None:
                        filtered["nvtx_spans"] = nvtx_spans_by_gpu.get(filtered["id"], [])
                    if max_buckets is not None and kernels_requested and filtered.get("lod") is None:
                        filtered = _aggregate_timeline_gpu_entry(
                            filtered, start_ns, end_ns, max_buckets
                        )
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
            _send_body(self, body, "application/json; charset=utf-8")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception as e:
            _log.exception("Tile data error: %s", e)
            elapsed = _time.monotonic() - t0
            print(f"[tile] {start_s:.1f}s–{end_s:.1f}s  ERROR in {elapsed:.2f}s: {e}", flush=True)
            self._json_response({"error": str(e)}, 500)

    def _json_response(self, obj, status=200):
        _send_body(
            self, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8", status
        )

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
            _log.exception("Analyze error")
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
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError) as e:
            # Malformed JSON, invalid UTF-8, or bad payload fields — client error.
            self._json_response({"error": str(e)}, 400)
        except Exception as e:
            # Unexpected server-side error — log and return 500.
            _log.exception("Error handling POST /api/findings")
            self._json_response({"error": str(e)}, 500)

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length) if content_length else b"{}"
        payload = json.loads(raw.decode("utf-8")) if raw else {}
        if not isinstance(payload, dict):
            raise ValueError("JSON object required")
        return payload

    def _session_mode(self) -> bool:
        return self.__class__._session_id is not None

    def _session_store(self):
        from .session_store import SessionStore

        return SessionStore(self.__class__._session_root)

    def _session_dir_path(self):
        from .session_cli import session_dir

        return session_dir(
            self.__class__._session_id,
            root=self.__class__._session_root,
        )

    def _load_session_projection(self) -> dict:
        from .session_cli import project_loop_state

        snapshot = self._session_store().load(self.__class__._session_id)
        return project_loop_state(snapshot, session_dir_path=self._session_dir_path())

    def _session_findings_payload(self) -> list[dict]:
        snapshot = self._session_store().load(self.__class__._session_id)
        if snapshot.findings is None:
            return []
        return [f.to_dict() for f in snapshot.findings.findings]

    def _session_limitation(self, cli_command: str, detail: str) -> None:
        """Return a stated limitation: reduced capability, never a silent no-op.

        The body carries the canonical cannot-answer marker, built by
        ``cannot_answer`` so this route and a skill row satisfy one predicate —
        the decision is in ``docs/notes/cannot-answer.md``. ``limitation`` and
        ``error`` are retained aliases for browser code already reading them,
        and ``cli`` is retained because it carries what the marker cannot: the
        command that *can* do the job.

        The HTTP status stays 400 and is not the marker. A status answers a
        transport question — was this request serviceable on this route — while
        the body answers the analysis question. A consumer branches on the body.
        """
        from .cannot_answer import cannot_answer

        self._json_response(
            cannot_answer(
                detail,
                error=detail,
                limitation=True,
                cli=cli_command,
            ),
            400,
        )

    def _require_session(self) -> bool:
        if self.__class__._session_id is None:
            self._json_response({"error": "session not initialized"}, 500)
            return False
        return True

    def _handle_loop_get(self):
        with _LOOP_LOCK:
            if not self._require_session():
                return
            try:
                self._json_response(self._load_session_projection())
            except Exception as e:
                self._json_response({"error": str(e)}, 500)

    def _handle_loop_phase(self):
        with _LOOP_LOCK:
            if not self._require_session():
                return
            self._session_limitation(
                "nsys-ai evidence|propose|diff",
                "session mode does not set phase directly; CLI publishers "
                "advance the session phase (nsys-ai evidence, propose, diff)",
            )

    def _handle_loop_proposal(self):
        with _LOOP_LOCK:
            if not self._require_session():
                return
            self._session_limitation(
                "nsys-ai propose",
                "session mode does not save free-text proposals; use "
                "nsys-ai propose --session to publish a Proposal artifact",
            )

    def _handle_loop_reprofile(self):
        with _LOOP_LOCK:
            if not self._require_session():
                return
            self._session_limitation(
                "nsys-ai profile",
                "session mode does not register an after profile from the "
                "browser; capture/validate with nsys-ai profile, then "
                "publish via nsys-ai diff --session (which registers the "
                "after profile after a non-abstained proposal)",
            )

    def _handle_loop_diagnose(self):
        with _LOOP_LOCK:
            if not self._require_session():
                return
            self._session_limitation(
                "nsys-ai evidence",
                "session mode does not run diagnose in the browser; use "
                "nsys-ai evidence build <profile> --session to publish "
                "findings, then reload loop state",
            )

    def _handle_loop_diff(self):
        with _LOOP_LOCK:
            if not self._require_session():
                return
            # C6 recommendation: read-only reload of SessionSnapshot.diff.
            # No analysis and no publish_* on this endpoint.
            try:
                projected = self._load_session_projection()
            except Exception as e:
                self._json_response({"error": str(e)}, 500)
                return
            if projected.get("diff_summary") is None:
                self._session_limitation(
                    "nsys-ai diff",
                    "session has no published diff yet; use "
                    "nsys-ai diff <before> <after> --session to publish, "
                    "then reload",
                )
                return
            self._json_response(
                {"state": projected, "diff": projected["diff_summary"]}
            )

    def _handle_loop_decision(self):
        with _LOOP_LOCK:
            if not self._require_session():
                return
            payload = self._read_json_body()
            decision = str(payload.get("decision") or "").strip()
            reason = str(payload.get("reason") or "").strip()
            from .session_store import SessionConflictError

            try:
                store = self._session_store()
                with store.writer(self.__class__._session_id) as writer:
                    writer.publish_decision(decision, reason)
                self._json_response(self._load_session_projection())
            except SessionConflictError as e:
                self._json_response({"error": str(e)}, 409)
            except ValueError as e:
                self._json_response({"error": str(e)}, 400)
            except Exception as e:
                _log.exception("Error publishing session decision")
                self._json_response({"error": str(e)}, 500)

    def _handle_loop_server_error(self, path: str, exc: Exception):
        _log.exception("Error handling POST %s", path)
        with _LOOP_LOCK:
            try:
                state = (
                    self._load_session_projection()
                    if self.__class__._session_id is not None
                    else None
                )
            except Exception:
                state = None
            self._json_response({"error": str(exc), "state": state}, 500)

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/api/analyze":
            self._handle_analyze()
            return
        if path == "/api/findings":
            self._handle_post_finding()
            return
        if path == "/api/loop/phase":
            try:
                self._handle_loop_phase()
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError, TypeError) as e:
                self._json_response({"error": str(e)}, 400)
            return
        if path == "/api/loop/proposal":
            try:
                self._handle_loop_proposal()
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError, TypeError) as e:
                self._json_response({"error": str(e)}, 400)
            return
        if path == "/api/loop/reprofile":
            try:
                self._handle_loop_reprofile()
            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
                ValueError,
                KeyError,
                TypeError,
                FileNotFoundError,
            ) as e:
                self._json_response({"error": str(e)}, 400)
            return
        if path == "/api/loop/diagnose":
            try:
                self._handle_loop_diagnose()
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError, TypeError) as e:
                self._json_response({"error": str(e)}, 400)
            except Exception as e:
                self._handle_loop_server_error(path, e)
            return
        if path == "/api/loop/diff":
            try:
                self._handle_loop_diff()
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError, TypeError) as e:
                self._json_response({"error": str(e)}, 400)
            except Exception as e:
                self._handle_loop_server_error(path, e)
            return
        if path == "/api/loop/decision":
            try:
                self._handle_loop_decision()
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError, TypeError) as e:
                self._json_response({"error": str(e)}, 400)
            return
        if path == "/api/ask":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length else b"{}"
            stream_requested = False
            try:
                payload = json.loads(body.decode("utf-8"))
                if isinstance(payload, dict) and self.__class__._session_id is not None:
                    stream_requested = payload.get("stream") is True
                    payload["session_id"] = self.__class__._session_id
                    payload["session_root"] = self.__class__._session_root
                    body = json.dumps(payload).encode("utf-8")
                elif isinstance(payload, dict):
                    stream_requested = payload.get("stream") is True
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                pass
            if stream_requested:
                gen = _handle_ask_stream_request(body)
                if gen is None:
                    _send_body(
                        self,
                        b'{"error":"ask transport unavailable"}',
                        "application/json; charset=utf-8",
                        501,
                    )
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
            out = _handle_ask_request(body)
            if out is None:
                _send_body(
                    self,
                    b'{"error":"ask transport unavailable"}',
                    "application/json; charset=utf-8",
                    501,
                )
                return
            status = int(out.pop("_http_status", 400 if out.get("error") else 200))
            resp = json.dumps(out, default=str).encode("utf-8")
            _send_body(self, resp, "application/json; charset=utf-8", status)
            return
        if path != "/api/chat":
            if path.startswith("/api/"):
                self._json_response({"error": "not found", "path": path}, 404)
                return
            self.send_error(404)
            return
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b"{}"
        stream_requested = False
        try:
            payload = json.loads(body.decode("utf-8"))
            stream_requested = payload.get("stream") is True
            if isinstance(payload, dict) and self.__class__._session_id is not None:
                payload["session_id"] = self.__class__._session_id
                payload["session_root"] = self.__class__._session_root
                body = json.dumps(payload).encode("utf-8")
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
            _log.exception("Chat endpoint error")
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def log_message(self, format, *args):
        pass


def _bind_local_server(port: int, handler):
    """Bind 127.0.0.1:*port*, falling back to a free port when it is taken.

    A busy port is an ordinary condition — another viewer is already running —
    so every local server here answers it the same way instead of letting the
    bind error reach the user as a traceback.
    """
    try:
        return _ThreadedHTTPServer(("127.0.0.1", port), handler)
    except OSError:
        if port == 0:
            raise
        server = _ThreadedHTTPServer(("127.0.0.1", 0), handler)
        print(f"Port {port} in use, using port {server.server_address[1]} instead.")
        return server


def serve(prof, device: int, trim: tuple[int, int], *, port: int = 8142, open_browser: bool = True):
    """Start a local HTTP server serving the interactive HTML viewer.
    If the requested port is in use, tries port 0 (system assigns a free port) and opens that URL.
    """
    from .nvtx_tree import build_nvtx_tree
    from .tree import to_json

    _ViewerHandler.prof = prof
    _ViewerHandler._tree_device = device
    _ViewerHandler._tree_trim = trim
    _ViewerHandler._tree_data = []
    _ViewerHandler._tree_configured = False
    _ViewerHandler._tree_build_error = None
    html = generate_html(prof, device, trim, embed_data=False)
    _ViewerHandler.html_bytes = html.encode("utf-8")

    server = _bind_local_server(port, _ViewerHandler)
    open_url = f"http://127.0.0.1:{server.server_address[1]}" if open_browser else None
    tree_done = threading.Event()
    _ViewerHandler._tree_build_done = tree_done

    def _warm_tree() -> None:
        t0 = _time.monotonic()
        try:
            print("Building NVTX tree in background...", flush=True)
            _ViewerHandler._tree_data = to_json(build_nvtx_tree(prof, device, trim))
            print(
                f"NVTX tree ready in {_time.monotonic() - t0:.1f}s "
                f"({len(_ViewerHandler._tree_data)} roots)",
                flush=True,
            )
        except Exception as exc:
            _ViewerHandler._tree_build_error = str(exc)
            _log.exception("Background NVTX tree build failed")
        finally:
            _ViewerHandler._tree_configured = True
            tree_done.set()

    print(f"Web viewer at http://127.0.0.1:{server.server_address[1]}", flush=True)
    threading.Thread(target=_warm_tree, name="web-nvtx-tree-warmup", daemon=True).start()
    try:
        _run_server(server, open_url, prof)
    finally:
        # Keep the Profile alive until the worker has stopped using its DuckDB
        # connection, matching the timeline-web background build contract.
        tree_done.wait()


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


def _slice_nvtx_spans(spans: list[dict], start_ns: int, end_ns: int) -> list[dict]:
    """Return full-range NVTX spans overlapping a requested viewport."""
    return [
        span
        for span in spans
        if span.get("end", 0) >= start_ns and span.get("start", 0) <= end_ns
    ]


def _build_timeline_overview(
    prebuilt_data: list[dict], time_range_ns: tuple[int, int], bin_count: int = 240
) -> tuple[list[float], int]:
    """Build a compact, full-profile kernel activity histogram for the web overview."""
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    range_start, range_end = time_range_ns
    profile_span = max(range_end - range_start, 1)
    bins = [0.0] * bin_count
    kernel_count = 0
    for gpu_entry in prebuilt_data:
        for kernel in gpu_entry.get("kernels", []):
            start_ns = kernel.get("start_ns")
            end_ns = kernel.get("end_ns")
            if start_ns is None or end_ns is None or end_ns < range_start or start_ns > range_end:
                continue
            kernel_count += 1
            first = max(0, int((start_ns - range_start) / profile_span * bin_count))
            last = min(bin_count - 1, int((end_ns - range_start) / profile_span * bin_count))
            weight = max(1.0, min(8.0, float(kernel.get("duration_ms") or 1.0)))
            for index in range(first, last + 1):
                bins[index] += weight
    return bins, kernel_count


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


def _aggregate_timeline_gpu_entry(
    gpu_entry: dict,
    start_ns: int,
    end_ns: int,
    max_buckets: int,
) -> dict:
    """Summarise a dense kernel payload into at most ``max_buckets`` bins.

    This is deliberately a request-time representation.  An aggregate is not
    a kernel: it carries the number of records, busy-time union, longest
    record, and dominant categories so the UI can label it as derived data.
    Busy time is an interval union rather than a sum of durations; concurrent
    streams must not make a bucket look more than 100% occupied.
    """
    kernels = [
        kernel
        for kernel in gpu_entry.get("kernels", [])
        if kernel.get("end_ns", 0) >= start_ns and kernel.get("start_ns", 0) <= end_ns
    ]
    if not kernels or max_buckets <= 0 or end_ns <= start_ns:
        return {
            "id": gpu_entry.get("id"),
            "kernels": kernels,
            "nvtx_spans": gpu_entry.get("nvtx_spans", []),
        }
    if len(kernels) <= max_buckets:
        return {
            "id": gpu_entry.get("id"),
            "kernels": kernels,
            "nvtx_spans": gpu_entry.get("nvtx_spans", []),
            "lod": {"mode": "exact", "record_count": len(kernels)},
        }

    bucket_count = min(max_buckets, len(kernels))
    span_ns = end_ns - start_ns
    bucket_width = span_ns / bucket_count
    buckets: list[dict] = [
        {"intervals": [], "count": 0, "max_duration_ns": 0, "types": {}, "names": {}}
        for _ in range(bucket_count)
    ]

    for kernel in kernels:
        raw_start = int(kernel.get("start_ns", start_ns))
        raw_end = int(kernel.get("end_ns", raw_start))
        clipped_start = max(start_ns, raw_start)
        clipped_end = min(end_ns, raw_end)
        if clipped_end <= clipped_start:
            continue
        first = max(0, min(bucket_count - 1, int((clipped_start - start_ns) / bucket_width)))
        last = max(
            first,
            min(bucket_count - 1, int((max(clipped_start, clipped_end - 1) - start_ns) / bucket_width)),
        )
        duration_ns = max(0, raw_end - raw_start)
        kind = str(kernel.get("type") or "kernel")
        name = str(kernel.get("name") or "(unnamed)")
        for bucket_index in range(first, last + 1):
            bucket_start = int(start_ns + bucket_index * bucket_width)
            bucket_end = int(start_ns + (bucket_index + 1) * bucket_width)
            interval_start = max(clipped_start, bucket_start)
            interval_end = min(clipped_end, bucket_end)
            if interval_end <= interval_start:
                continue
            bucket = buckets[bucket_index]
            bucket["intervals"].append((interval_start, interval_end))
            bucket["count"] += 1
            bucket["max_duration_ns"] = max(bucket["max_duration_ns"], duration_ns)
            bucket["types"][kind] = bucket["types"].get(kind, 0) + 1
            bucket["names"][name] = bucket["names"].get(name, 0) + 1

    aggregate_rows = []
    for bucket_index, bucket in enumerate(buckets):
        if not bucket["count"]:
            continue
        intervals = sorted(bucket["intervals"])
        union_ns = 0
        union_start, union_end = intervals[0]
        for interval_start, interval_end in intervals[1:]:
            if interval_start > union_end:
                union_ns += union_end - union_start
                union_start, union_end = interval_start, interval_end
            else:
                union_end = max(union_end, interval_end)
        union_ns += union_end - union_start
        dominant_type = max(bucket["types"].items(), key=lambda item: (item[1], item[0]))[0]
        dominant_name = max(bucket["names"].items(), key=lambda item: (item[1], item[0]))[0]
        bucket_start = int(start_ns + bucket_index * bucket_width)
        bucket_end = int(start_ns + (bucket_index + 1) * bucket_width)
        aggregate_rows.append(
            {
                "type": "aggregate",
                "aggregate": True,
                "name": f"[Aggregate] {bucket['count']:,} records",
                "start_ns": bucket_start,
                "end_ns": bucket_end,
                "duration_ms": round(union_ns / 1e6, 3),
                "stream": "__aggregate__",
                "path": "",
                "record_count": bucket["count"],
                "busy_ns": union_ns,
                "occupancy": union_ns / max(bucket_end - bucket_start, 1),
                "max_duration_ms": round(bucket["max_duration_ns"] / 1e6, 3),
                "dominant_type": dominant_type,
                "dominant_name": dominant_name,
            }
        )
    return {
        "id": gpu_entry.get("id"),
        "kernels": aggregate_rows,
        "nvtx_spans": gpu_entry.get("nvtx_spans", []),
        "lod": {
            "mode": "aggregate",
            "record_count": len(kernels),
            "bucket_count": len(aggregate_rows),
            "max_buckets": max_buckets,
        },
    }


def serve_timeline(
    prof,
    device,
    trim: tuple[int, int] | None = None,
    *,
    port: int = 8144,
    open_browser: bool = True,
    findings_path: str | None = None,
    auto_findings: list[dict] | None = None,
    loop_before: str | None = None,
    loop_h100_preset: bool = False,
    session: str | None = None,
    session_root: str | os.PathLike[str] = ".nsys-ai/sessions",
):
    """Start a local HTTP server serving the horizontal timeline viewer.

    If *trim* is None, the initial view shows a default 5s window and
    the client can freely navigate via /api/data.
    If *findings_path* is given, findings are loaded and rendered as overlays.
    If *auto_findings* is given, they are used directly (from --auto-analyze).
    If *session* is given (including empty string), open that SessionStore
    session; empty or omitted derives the id from the before profile content id
    (C1). SessionStore is always used — there is no in-memory loop state.
    *session_root* is the SessionStore root the caller already published into;
    default is ``.nsys-ai/sessions`` under the process CWD. Callers that used a
    different root must pass it here so --web opens that same session.
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
    _ViewerHandler._full_nvtx_by_gpu = {}
    _ViewerHandler._overview_bins = []
    _ViewerHandler._overview_kernel_count = 0
    _ViewerHandler._profile_id = ""
    _ViewerHandler._nvtx_prebuild_done = None
    _ViewerHandler._nvtx_prebuild_error = None
    _ViewerHandler._findings = findings_data or []
    _ViewerHandler._session_id = None
    _ViewerHandler._session_root = os.fspath(session_root)
    _ViewerHandler._trim = trim
    from .loop_state import detect_h100_replay_preset

    raw_path = prof.path if hasattr(prof, "path") else ""
    _profile_path = os.fspath(raw_path) if raw_path else ""
    try:
        from .fingerprint import get_profile_id

        _ViewerHandler._profile_id = get_profile_id(
            prof.query_conn(), fallback_path=_profile_path or None
        )
    except Exception as exc:
        _log.debug("Could not derive profile id: %s", exc, exc_info=True)

    preset = detect_h100_replay_preset() if loop_h100_preset else None
    if preset:
        loop_before_path = preset["before_path"]
    else:
        loop_before_path = loop_before or _profile_path

    from .profile_runner import build_local_profile_reference
    from .session_cli import (
        project_loop_state,
        resolve_session_id,
        resolve_session_location,
        session_dir,
    )
    from .session_store import SessionExistsError, SessionStore

    before_for_id = loop_before_path or _profile_path
    if not before_for_id:
        raise ValueError(
            "session mode requires a before profile path to open or derive "
            "the session id"
        )
    before_ref = build_local_profile_reference(before_for_id)
    location = resolve_session_location(session or None, root=session_root)
    if location is not None:
        session_id = location.session_id
        session_root = location.root
        _ViewerHandler._session_root = os.fspath(session_root)
    else:
        session_id = resolve_session_id(None, before=before_ref)
    store = SessionStore(_ViewerHandler._session_root)
    try:
        store.create(session_id, before_profile=before_ref)
    except SessionExistsError:
        pass
    snapshot = store.load(session_id)
    _ViewerHandler._session_id = session_id
    if snapshot.findings is not None and not findings_data:
        findings_data = [f.to_dict() for f in snapshot.findings.findings]
        _ViewerHandler._findings = findings_data
        print(
            f"Loaded {len(findings_data)} finding(s) from session {session_id}",
            flush=True,
        )
    projected = project_loop_state(
        snapshot,
        session_dir_path=session_dir(
            session_id, root=_ViewerHandler._session_root
        ),
    )
    print(
        f"Opened session {session_id} (phase={projected['phase']})",
        flush=True,
    )
    _in_loop_mode = True

    _asset_v = _template_asset_version()
    _css_href = _versioned_asset_url("/assets/timeline.css")
    _js_src = _versioned_asset_url("/assets/timeline.js")
    # Always serve the progressive shell. A trim window is metadata for the
    # initial viewport and API range, not a reason to embed the selected data.
    html = generate_timeline_html(
        prof,
        devices,
        trim,
        findings_data=findings_data,
        profile_path=_profile_path,
        profile_id=_ViewerHandler._profile_id,
        loop_mode=_in_loop_mode,
        timeline_css_href=_css_href,
        timeline_js_src=_js_src,
        progressive_mode=True,
    )
    _ViewerHandler._prebuilt_nvtx_mode = "background"

    _ViewerHandler.html_bytes = html.encode("utf-8")
    if _in_loop_mode:
        print(f"Loop UI loaded (assets v{_asset_v}) — hard-refresh if the panel looks outdated", flush=True)

    nvtx_done: threading.Event | None = None

    # Pre-build the requested range only. A trim window must not switch back
    # to the legacy full-document path or force a full-profile scan.
    if _ViewerHandler._prebuilt_nvtx_mode == "background":
        db_path = _profile_path
        cache_path = (
            db_path + ".timeline-cache-v3-kernels.json"
            if db_path and trim is None
            else ""
        )
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
            except (ValueError, KeyError, json.JSONDecodeError, OSError) as e:
                _log.debug("Cache load failed: %s", e, exc_info=True)
                print(f"Cache load failed: {e}, rebuilding...", flush=True)

        if not cache_valid:
            t0 = _time.monotonic()
            full_range = trim or prof.meta.time_range
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

        _ViewerHandler._overview_bins, _ViewerHandler._overview_kernel_count = (
            _build_timeline_overview(prebuilt, trim or prof.meta.time_range)
        )

        # NVTX is much more expensive than kernel filtering.  Warm the full
        # profile after the cheap kernel payload is resident and before the
        # server starts accepting browser requests.  The server itself is
        # started immediately after this thread is launched, so a first NVTX
        # request either receives the warm result or waits on the one shared
        # build rather than rebuilding the requested tile.
        nvtx_done = threading.Event()
        _ViewerHandler._nvtx_prebuild_done = nvtx_done

        def _warm_nvtx() -> None:
            t_nv = _time.monotonic()
            full_range = trim or prof.meta.time_range
            try:
                print(
                    f"Pre-building NVTX in background for {len(devices)} GPU(s) "
                    f"({full_range[0] / 1e9:.1f}s–{full_range[1] / 1e9:.1f}s)...",
                    flush=True,
                )
                entries = build_timeline_gpu_data(
                    prof,
                    devices,
                    full_range,
                    include_kernels=False,
                    include_nvtx=True,
                )
                _ViewerHandler._full_nvtx_by_gpu = {
                    entry["id"]: entry.get("nvtx_spans", []) for entry in entries
                }
                total = sum(len(spans) for spans in _ViewerHandler._full_nvtx_by_gpu.values())
                print(
                    f"NVTX background pre-build complete in {_time.monotonic() - t_nv:.1f}s "
                    f"({total} spans)",
                    flush=True,
                )
            except Exception as exc:
                _ViewerHandler._nvtx_prebuild_error = str(exc)
                _log.exception("Background NVTX pre-build failed")
            finally:
                nvtx_done.set()

        threading.Thread(target=_warm_nvtx, name="timeline-nvtx-warmup", daemon=True).start()

    try:
        # Binding is inside the try: if it fails, the caller's `with` closes the
        # Profile, and closing a DuckDB connection out from under the background
        # worker is a segfault rather than an exception.
        server = _bind_local_server(port, _ViewerHandler)
        actual_url = f"http://127.0.0.1:{server.server_address[1]}"
        print(f"Timeline viewer at {actual_url}")
        _run_server(server, actual_url if open_browser else None, prof)
    finally:
        # Keep the Profile alive until the background worker has stopped using
        # its DuckDB connection.  This also covers callers that replace
        # _run_server in tests or embed the server lifecycle themselves.
        if nvtx_done is not None:
            nvtx_done.wait()


# ── Mode 3: Evidence View ────────────────────────────────────────


class _EvidenceHandler(_HeadRequestMixin, BaseHTTPRequestHandler):
    """Serve the Evidence View HTML; GET /api/data for progressive kernel tiles."""

    html_bytes: bytes = b""
    prof = None
    devices: list = []
    _prebuilt_data: list = []
    _prebuilt_nvtx_mode: str = "full"
    _tile_nvtx_cache: dict = {}
    _asset_cache: dict[str, tuple[float, bytes]] = {}

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/assets/evidence.css":
            # Reuse the existing timeline.css asset for evidence CSS.
            self._serve_asset("timeline.css", "text/css; charset=utf-8")
            return
        if path == "/assets/tokens.css":
            self._serve_asset("tokens.css", "text/css; charset=utf-8")
            return
        if path == "/assets/evidence.js":
            # Reuse the existing timeline.js asset for evidence JS.
            self._serve_asset("timeline.js", "application/javascript; charset=utf-8")
            return
        if path == "/api/data":
            self._handle_data()
            return
        if path.startswith("/api/"):
            self._json_response({"error": "not found", "path": path}, 404)
            return
        if path not in {"/", "/index.html"}:
            self.send_error(404)
            return
        # Default: serve evidence HTML
        _send_body(self, self.html_bytes, "text/html; charset=utf-8")

    def _serve_asset(self, filename: str, content_type: str):
        path = os.path.join(_TEMPLATE_DIR, filename)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            self.send_error(404)
            return
        cache = self.__class__._asset_cache
        cached = cache.get(filename)
        if cached is None or cached[0] != mtime:
            with open(path, "rb") as f:
                body = f.read()
            cache[filename] = (mtime, body)
        else:
            body = cached[1]
        # timeline.js is ~150 KB and timeline.css ~40 KB of text; both are paid on
        # every cold load and compress to a fraction of that. Cache-Control stays
        # as it was: the ?v= token busts the cache, so the browser must revalidate.
        _send_body(
            self,
            body,
            content_type,
            extra_headers={"Cache-Control": "no-cache, must-revalidate"},
        )

    def _json_response(self, obj, status=200):
        _send_body(
            self, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8", status
        )

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
            _send_body(self, body, "application/json; charset=utf-8")
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

    server = _bind_local_server(port, _EvidenceHandler)
    actual_url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Evidence viewer at {actual_url}")
    print(f"  {len(findings_data)} finding(s): {title}")
    _run_server(server, actual_url if open_browser else None, prof)
