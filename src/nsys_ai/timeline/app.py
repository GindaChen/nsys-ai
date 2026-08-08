"""
timeline/app.py — NsysTimelineApp: Textual App for the horizontal timeline TUI.

Replaces the curses TimelineTUI in tui_timeline.py.
Entry point: run_timeline(db_path, device, trim, min_ms=0)

Keybindings (matching original tui_timeline.py):
    ←/→          Pan through time
    Shift+←/→    Page pan (1/4 viewport)
    ↑/↓          Select stream
    +/=  -/_     Zoom in / out
    Tab          Snap to next kernel
    Shift+Tab    Snap to previous kernel
    a            Toggle absolute / relative time axis
    L            Toggle TIME / LOGICAL mode (hides NVTX rows)
    /            Filter by kernel name
    n            Clear filter
    m            Set min duration threshold (interactive)
    d            Toggle demangled names
    h            Toggle help overlay
    B            Save bookmark at cursor
    , / .        Cycle through bookmarks
    '            Show bookmark list (1-9 to jump)
    [            Set range bookmark start
    ]            Set range bookmark end (auto-saves range bookmark)
    `            Jump back to previous position
    T            Cycle tick density
    Home / End   Jump to start / end of trace
    A            Toggle AI chat panel
    C            Toggle config panel
    q            Quit

Layout:
    ┌─────────────────────────────────────────┐
    │  Header: title + cursor info            │
    ├─────────────────────────────────────────┤
    │  TimelineFilterBar  (hidden by default) │
    │  TimelineMinDurBar  (hidden by default) │
    │  TimelineCanvas (streams + NVTX + axis) │
    │  [ConfigPanel dock:right when open]     │
    │  [TimelineBookmarkPanel dock:right]     │
    ├─────────────────────────────────────────┤
    │  BottomPanel (kernel detail, 4 rows)    │
    ├─────────────────────────────────────────┤
    │  ChatPanel (12 rows, collapsible)       │
    └─────────────────────────────────────────┘
"""

from __future__ import annotations

from collections.abc import Callable

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input
from textual.worker import get_current_worker

from .. import tui_actions
from ..formatting import fmt_dur as _fmt_dur
from ..formatting import fmt_ns as _fmt_ns
from ..tree.chat import ChatPanel
from ..tui_cache import CacheBuildProgress, format_cache_progress, post_cache_progress
from ..tui_models import KernelEvent
from .canvas import TimelineCanvas
from .logic import (
    build_stream_kernels,
    center_viewport,
    collect_streams,
    extract_events,
    find_kernel_by_name,
    kernel_at_time,
    kernel_index_at_time,
    time_bounds,
    zoom_ns_per_col,
)
from .widgets import (
    BottomPanel,
    ConfigPanel,
    TimelineBookmarkPanel,
    TimelineFilterBar,
    TimelineMinDurBar,
)


class NsysTimelineApp(App):
    """Textual horizontal timeline browser — replaces curses TimelineTUI."""

    TITLE = "nsys-ai Timeline"

    CSS = """
    TimelineCanvas { height: 1fr; }
    ChatPanel { display: none; }
    ChatPanel.-active { display: block; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("A", "toggle_chat", "AI Chat", priority=True),
        Binding("f5", "loop_diagnose", "Loop diagnose", show=False),
        Binding("f6", "loop_propose", "Loop propose", show=False),
        Binding("f7", "loop_reprofile", "Loop reprofile", show=False),
        Binding("f8", "loop_diff", "Loop diff", show=False),
        Binding("f9", "loop_accept", "Loop accept", show=False),
        Binding("left", "pan_left", "←", show=False),
        Binding("right", "pan_right", "→", show=False),
        Binding("shift+left", "page_left", "⇐", show=False),
        Binding("shift+right", "page_right", "⇒", show=False),
        Binding("pageup", "page_left", "⇐", show=False),
        Binding("pagedown", "page_right", "⇒", show=False),
        Binding("up,k", "prev_stream", "↑ stream", show=False),
        Binding("down,j", "next_stream", "↓ stream", show=False),
        Binding("tab", "next_kernel", "Next kernel", priority=True),
        Binding("shift+tab", "prev_kernel", "Prev kernel", priority=True),
        Binding("plus,equals_sign", "zoom_in", "Zoom in"),
        Binding("minus,underscore", "zoom_out", "Zoom out"),
        Binding("a", "toggle_time_mode", "Abs/rel time"),
        Binding("L", "toggle_logical", "Logical mode"),
        Binding("d", "toggle_demangled", "Demangled"),
        Binding("h", "toggle_help", "Help"),
        Binding("C", "toggle_config", "Config"),
        Binding("P", "toggle_path_mode", "Path mode"),
        Binding("S", "toggle_merged", "Merged view"),
        Binding("I", "toggle_labels", "Labels"),
        Binding("T", "toggle_tick_density", "Ticks", show=False),
        Binding("home", "jump_start", "Start", show=False),
        Binding("end", "jump_end", "End", show=False),
        Binding("g", "jump_to_time", "Jump to ns", show=False),
        Binding("B", "toggle_bookmark", "Bookmark"),
        Binding("comma", "prev_bookmark", "← bookmark", show=False),
        Binding("full_stop", "next_bookmark", "→ bookmark", show=False),
        Binding("r", "set_range_start", "Range start", show=False),
        Binding("R", "set_range_end", "Range end / clear", show=False),
        Binding("apostrophe", "show_bookmark_list", "BM list", show=False),
        Binding("left_square_bracket", "range_start", "[range", show=False),
        Binding("right_square_bracket", "range_end", "range]", show=False),
        Binding("grave_accent", "jump_back", "Jump back", show=False),
        Binding("/", "open_filter", "Filter"),
        Binding("n", "clear_filter", "Clear filter"),
        Binding("m", "open_min_dur", "Min dur"),
        Binding("P", "toggle_path_mode", "Path mode"),
        Binding("T", "toggle_tick_density", "Ticks", show=False),
        Binding("home", "jump_start", "Start", show=False),
        Binding("end", "jump_end", "End", show=False),
    ]

    # -------------------------------------------------------------------------
    # Reactive state
    # -------------------------------------------------------------------------
    cursor_ns: reactive[int] = reactive(0)
    selected_stream_idx: reactive[int] = reactive(0)
    ns_per_col: reactive[int] = reactive(1_000_000)
    filter_text: reactive[str] = reactive("")
    min_dur_us: reactive[float] = reactive(0)
    analysis_phase: reactive[str] = reactive("diagnose")

    # -------------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        db_path: str,
        device: int | list[int] = 0,
        trim: tuple[int, int] | None = None,
        min_ms: float = 0,
        json_roots: list[dict] | None = None,
        loop_after: str | None = None,
        session: str | None = None,
    ) -> None:
        super().__init__()
        self._db_path = db_path
        # Support single device int or list of devices
        if isinstance(device, int):
            self._devices: list[int] = [device]
        else:
            self._devices = list(device)
        self._device = self._devices[0]  # backward compat for ChatPanel etc.
        self._trim = trim or (0, 0)
        self._session_id: str | None = None
        self._session_root = ".nsys-ai/sessions"
        self._session_projection: dict | None = None
        # Always open a SessionStore session when a before profile path is
        # available (C1). from_json tests may omit db_path and skip open.
        if self._db_path:
            self._open_session(session)

        # ALL plain attributes MUST be initialized before any reactive is set.

        # Per-GPU data — keyed by device ID
        self._gpu_kernels: dict[int, list[KernelEvent]] = {}
        self._gpu_streams: dict[int, list[str]] = {}
        self._gpu_stream_kernels: dict[int, dict[str, list[KernelEvent]]] = {}
        self._gpu_nvtx_spans: dict[int, list] = {}
        self._gpu_nvtx_max_depth: dict[int, int] = {}

        # Flattened views (for navigation / bottom panel)
        self._kernels: list[KernelEvent] = []
        self._stream_kernels: dict[str, list[KernelEvent]] = {}
        self._streams: list[str] = []  # flat list of "gpu:stream" keys
        self._stream_color_idx: dict[str, int] = {}
        self._stream_gpu: dict[str, int] = {}  # stream_key -> gpu_id
        self._time_start = 0
        self._time_end = 0
        self._time_span = 1
        self._is_mounted = False

        # Bookmarks and navigation state
        self._bookmarks: list[dict] = []
        self._bookmark_idx = -1
        self._range_start_ns: int | None = None
        self._prev_position: dict | None = None

        # Display toggles
        self._relative_time = False
        self._show_demangled = False
        self._logical_mode = False
        self._merged_mode = False  # S key: merged all-streams view
        self._label_mode = "gpu+stream"  # I key: "gpu+stream" -> "gpu" -> "none"

        # Tick density (cycles: 3 → 6 → 10 → 15)
        self._tick_density = 6

        # Now safe to set reactives
        self.min_dur_us = min_ms * 1000

        if json_roots:
            self._load_from_json(json_roots)

    @classmethod
    def from_json(cls, json_roots: list[dict], **kwargs: object) -> NsysTimelineApp:
        """Factory for tests — skip DB loading."""
        return cls(db_path="", device=0, trim=None, json_roots=json_roots, **kwargs)

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------
    def _load_from_json(self, json_roots: list[dict], gpu_id: int = 0) -> None:
        """Load JSON tree for a single GPU into per-GPU storage."""
        kernels, nvtx_spans = extract_events(json_roots)
        self._gpu_kernels[gpu_id] = kernels
        streams = collect_streams(kernels)
        self._gpu_streams[gpu_id] = streams if streams else ["?"]
        self._gpu_stream_kernels[gpu_id] = build_stream_kernels(kernels, self._gpu_streams[gpu_id])
        self._gpu_nvtx_spans[gpu_id] = nvtx_spans
        self._gpu_nvtx_max_depth[gpu_id] = max((s.depth for s in nvtx_spans), default=-1) + 1

    def _rebuild_flattened(self) -> None:
        """Rebuild flattened views from per-GPU data."""
        self._kernels = []
        self._streams = []
        self._stream_kernels = {}
        self._stream_color_idx = {}
        self._stream_gpu = {}

        color_idx = 0
        for gpu_id in sorted(self._gpu_kernels.keys()):
            gpu_kernels = self._gpu_kernels[gpu_id]
            self._kernels.extend(gpu_kernels)
            for s in self._gpu_streams.get(gpu_id, []):
                key = f"{gpu_id}:{s}"  # unique key "0:21", "1:52" etc.
                self._streams.append(key)
                self._stream_kernels[key] = self._gpu_stream_kernels.get(gpu_id, {}).get(s, [])
                self._stream_color_idx[key] = color_idx % 7
                self._stream_gpu[key] = gpu_id
                color_idx += 1

        self._time_start, self._time_end = time_bounds(self._kernels, self._trim)
        self._time_span = max(self._time_end - self._time_start, 1)
        # Merge all nvtx spans
        self._nvtx_spans = []
        for spans in self._gpu_nvtx_spans.values():
            self._nvtx_spans.extend(spans)
        self._nvtx_max_depth = max(self._gpu_nvtx_max_depth.values(), default=0)

    def _fetch_gpu_data_from_db(
        self, progress: Callable[[str, int, int], None]
    ) -> dict:
        """Open the profile and collect per-GPU data. Safe to call from a worker thread."""
        from .. import profile as _profile
        from ..nvtx_tree import build_nvtx_tree, to_json

        gpu_kernels: dict[int, list[KernelEvent]] = {}
        gpu_streams: dict[int, list[str]] = {}
        gpu_stream_kernels: dict[int, dict[str, list[KernelEvent]]] = {}
        gpu_nvtx_spans: dict[int, list] = {}
        gpu_nvtx_max_depth: dict[int, int] = {}
        devices = list(self._devices)

        with _profile.open(self._db_path, progress=progress) as prof:
            # Auto-detect devices if none specified or single default 0
            if not devices or devices == [0]:
                if (
                    hasattr(prof, "meta")
                    and hasattr(prof.meta, "devices")
                    and prof.meta.devices
                ):
                    devices = list(prof.meta.devices)
            for dev in devices:
                # All kernels from GPU table (not only those under NVTX)
                raw_kernels = prof.kernels(dev, self._trim)
                kernel_events = [
                    KernelEvent(
                        {
                            "name": k["name"],
                            "demangled": k.get("demangled", ""),
                            "start_ns": k["start"],
                            "end_ns": k["end"],
                            "duration_ms": (k["end"] - k["start"]) / 1e6,
                            "stream": str(k["streamId"]),
                        }
                    )
                    for k in raw_kernels
                ]
                # NVTX spans from tree for overlay rows
                roots = build_nvtx_tree(prof, dev, self._trim)
                json_roots = to_json(roots)
                _, nvtx_spans = extract_events(json_roots)
                gpu_kernels[dev] = kernel_events
                streams = collect_streams(kernel_events)
                gpu_streams[dev] = streams if streams else ["?"]
                gpu_stream_kernels[dev] = build_stream_kernels(
                    kernel_events, gpu_streams[dev]
                )
                gpu_nvtx_spans[dev] = nvtx_spans
                gpu_nvtx_max_depth[dev] = max((s.depth for s in nvtx_spans), default=-1) + 1

        return {
            "devices": devices,
            "gpu_kernels": gpu_kernels,
            "gpu_streams": gpu_streams,
            "gpu_stream_kernels": gpu_stream_kernels,
            "gpu_nvtx_spans": gpu_nvtx_spans,
            "gpu_nvtx_max_depth": gpu_nvtx_max_depth,
        }

    def _apply_gpu_data(self, payload: dict, error: str | None) -> None:
        """Main-thread apply after a DB load."""
        # See NsysTreeApp._apply_json_roots: covers the shutdown window where the
        # worker passed its is_cancelled check before quit cleared ``_running``.
        if not self.is_running:
            return
        self.sub_title = ""
        if error is not None:
            self.notify(f"Failed to load profile: {error}", severity="error")
            return
        self._devices = payload["devices"]
        self._gpu_kernels = payload["gpu_kernels"]
        self._gpu_streams = payload["gpu_streams"]
        self._gpu_stream_kernels = payload["gpu_stream_kernels"]
        self._gpu_nvtx_spans = payload["gpu_nvtx_spans"]
        self._gpu_nvtx_max_depth = payload["gpu_nvtx_max_depth"]
        self._rebuild_flattened()
        # initial zoom: fit full trace in ~100 columns
        self.ns_per_col = max(1, self._time_span // 100)
        self.cursor_ns = self._time_start
        if self._is_mounted:
            self.query_one("#bottom-panel", BottomPanel).set_nvtx_spans(
                getattr(self, "_nvtx_spans", [])
            )
        self._push_canvas_state()
        self._update_title()

    # -------------------------------------------------------------------------
    # Compose
    # -------------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        yield TimelineFilterBar(id="tl-filter-bar")
        yield TimelineMinDurBar(id="tl-mindur-bar")
        yield TimelineCanvas(id="canvas")
        yield ConfigPanel(id="config-panel")
        yield TimelineBookmarkPanel(id="bm-panel")
        yield BottomPanel(id="bottom-panel")
        yield ChatPanel(
            id="chat-panel",
            db_path=self._db_path,
            device=self._device,
            ui_context_fn=self._build_ui_context,
            on_action_fn=self._handle_ai_action,
        )
        yield Footer()

    # -------------------------------------------------------------------------
    # Mount
    # -------------------------------------------------------------------------
    def on_mount(self) -> None:
        self._is_mounted = True
        if not self._kernels and self._db_path:
            self._run_load_worker()
        else:
            self._rebuild_flattened()
            self.ns_per_col = max(1, self._time_span // 100)
            self.cursor_ns = self._time_start
            self.query_one("#bottom-panel", BottomPanel).set_nvtx_spans(
                getattr(self, "_nvtx_spans", [])
            )
            self._push_canvas_state()
            self._update_title()
        # Focus the canvas so App-level key bindings work.
        self.query_one("#canvas", TimelineCanvas).focus()

    def on_cache_build_progress(self, event: CacheBuildProgress) -> None:
        if not self.is_running:
            return
        self.sub_title = format_cache_progress(event.label, event.step, event.total)

    @work(thread=True, exclusive=True, group="load")
    def _run_load_worker(self) -> None:
        """Build/open the profile off the message loop; apply results on the UI thread."""
        worker = get_current_worker()
        try:
            payload = self._fetch_gpu_data_from_db(
                lambda label, step, total: post_cache_progress(self, label, step, total)
            )
            error: str | None = None
        except Exception as e:
            payload = {}
            error = str(e)
        if worker.is_cancelled:
            return
        self.call_from_thread(self._apply_gpu_data, payload, error)

    def _push_canvas_state(self) -> None:
        if not self._is_mounted:
            return  # DOM not ready yet — called from __init__ via reactive setter
        canvas = self.query_one("#canvas", TimelineCanvas)
        viewport_start = center_viewport(
            self.cursor_ns, self.ns_per_col, max(canvas.size.width - canvas.label_w, 1)
        )
        # Read live config values so panel adjustments take effect immediately.
        cfg = self.query_one("#config-panel", ConfigPanel)
        selected_rows = cfg.selected_rows
        default_rows = cfg.default_rows
        # In logical mode hide all NVTX rows; otherwise cap by ConfigPanel setting.
        raw_nvtx_depth = getattr(self, "_nvtx_max_depth", 0)
        if self._logical_mode:
            nvtx_depth = 0
        elif cfg.nvtx_depth == 0:
            nvtx_depth = 0
        else:
            nvtx_depth = min(raw_nvtx_depth, cfg.nvtx_depth)
        canvas.refresh_from_app(
            cursor_ns=self.cursor_ns,
            viewport_start_ns=viewport_start,
            ns_per_col=self.ns_per_col,
            selected_stream_idx=self.selected_stream_idx,
            streams=self._streams,
            stream_kernels=self._stream_kernels,
            stream_color_idx=self._stream_color_idx,
            nvtx_spans=getattr(self, "_nvtx_spans", []),
            nvtx_max_depth=nvtx_depth,
            filter_text=self.filter_text,
            min_dur_us=self.min_dur_us,
            relative_time=self._relative_time,
            show_demangled=self._show_demangled,
            tick_density=self._tick_density,
            selected_stream_rows=selected_rows,
            default_stream_rows=default_rows,
            merged_mode=self._merged_mode,
            label_mode=self._label_mode,
            stream_gpu=self._stream_gpu,
            gpu_kernels=self._gpu_kernels,
        )
        self._update_bottom_panel()

    def _update_title(self) -> None:
        trim_s = f"{self._trim[0] / 1e9:.1f}s–{self._trim[1] / 1e9:.1f}s"
        kernel_count = len(self._kernels)
        stream = self._streams[self.selected_stream_idx] if self._streams else "?"
        k = kernel_at_time(self._stream_kernels.get(stream, []), self.cursor_ns)
        k_info = f" [{k.name[:30]}]" if k else ""
        mode = "LOGICAL" if self._logical_mode else "TIME"
        bm_indicator = f"  📌{len(self._bookmarks)}" if self._bookmarks else ""
        filter_indicator = f"  /{self.filter_text}" if self.filter_text else ""
        min_dur_indicator = f"  ≥{int(self.min_dur_us)}μs" if self.min_dur_us else ""
        gpu_str = ",".join(str(d) for d in self._devices)
        self.title = (
            f"nsys-ai Timeline  GPU {gpu_str}  {trim_s}  "
            f"{kernel_count} kernels  @ {_fmt_ns(self.cursor_ns)}{k_info}  "
            f"[{mode}]{bm_indicator}{filter_indicator}{min_dur_indicator}"
            f"  [LOOP:{self.analysis_phase.upper()}]"
        )

    def _update_bottom_panel(self) -> None:
        stream = self._streams[self.selected_stream_idx] if self._streams else "?"
        k = kernel_at_time(self._stream_kernels.get(stream, []), self.cursor_ns)
        self.query_one("#bottom-panel", BottomPanel).update(
            k, stream, self.cursor_ns, self._time_start, self._time_end
        )

    # -------------------------------------------------------------------------
    # Watch reactives → push canvas
    # -------------------------------------------------------------------------
    def watch_cursor_ns(self) -> None:
        self._push_canvas_state()
        self._update_title()

    def watch_selected_stream_idx(self) -> None:
        self._push_canvas_state()
        if self._is_mounted:
            self._update_title()

    def watch_ns_per_col(self) -> None:
        self._push_canvas_state()

    def watch_filter_text(self) -> None:
        self._push_canvas_state()
        if self._is_mounted:
            self._update_title()

    def watch_min_dur_us(self) -> None:
        self._push_canvas_state()
        if self._is_mounted:
            self._update_title()

    # -------------------------------------------------------------------------
    # Navigation actions
    # -------------------------------------------------------------------------
    def _step(self) -> int:
        return self.ns_per_col

    def action_pan_left(self) -> None:
        self.cursor_ns = max(self._time_start, self.cursor_ns - self._step())

    def action_pan_right(self) -> None:
        self.cursor_ns = min(self._time_end, self.cursor_ns + self._step())

    def action_page_left(self) -> None:
        canvas = self.query_one("#canvas", TimelineCanvas)
        page = self._step() * max(1, (canvas.size.width - canvas.label_w) // 4)
        self.cursor_ns = max(self._time_start, self.cursor_ns - page)

    def action_page_right(self) -> None:
        canvas = self.query_one("#canvas", TimelineCanvas)
        page = self._step() * max(1, (canvas.size.width - canvas.label_w) // 4)
        self.cursor_ns = min(self._time_end, self.cursor_ns + page)

    def action_prev_stream(self) -> None:
        self.selected_stream_idx = max(0, self.selected_stream_idx - 1)

    def action_next_stream(self) -> None:
        self.selected_stream_idx = min(len(self._streams) - 1, self.selected_stream_idx + 1)

    def action_next_kernel(self) -> None:
        stream = self._streams[self.selected_stream_idx] if self._streams else "?"
        ks = self._stream_kernels.get(stream, [])
        if not ks:
            return
        ki = kernel_index_at_time(ks, self.cursor_ns)
        if ki < 0:
            return
        # If cursor is already on this kernel, go to next; otherwise snap to it first
        if ks[ki].start_ns <= self.cursor_ns <= ks[ki].end_ns:
            if ki + 1 < len(ks):
                self.cursor_ns = ks[ki + 1].start_ns
        else:
            # Cursor is between kernels — snap to the nearest one ahead
            if self.cursor_ns < ks[ki].start_ns:
                self.cursor_ns = ks[ki].start_ns
            elif ki + 1 < len(ks):
                self.cursor_ns = ks[ki + 1].start_ns

    def action_prev_kernel(self) -> None:
        stream = self._streams[self.selected_stream_idx] if self._streams else "?"
        ks = self._stream_kernels.get(stream, [])
        if not ks:
            return
        ki = kernel_index_at_time(ks, self.cursor_ns)
        if ki < 0:
            return
        # If cursor is on this kernel, go to previous; otherwise snap to it first
        if ks[ki].start_ns <= self.cursor_ns <= ks[ki].end_ns:
            if ki > 0:
                self.cursor_ns = ks[ki - 1].start_ns
        else:
            # Cursor is between kernels — snap to the nearest one behind
            if self.cursor_ns > ks[ki].end_ns:
                self.cursor_ns = ks[ki].start_ns
            elif ki > 0:
                self.cursor_ns = ks[ki - 1].start_ns

    def action_zoom_in(self) -> None:
        self.ns_per_col = zoom_ns_per_col(self.ns_per_col, -1, self._time_span)
        self.notify(f"Zoom: {_fmt_dur(self.ns_per_col / 1e6)}/col", timeout=2)

    def action_zoom_out(self) -> None:
        self.ns_per_col = zoom_ns_per_col(self.ns_per_col, +1, self._time_span)
        self.notify(f"Zoom: {_fmt_dur(self.ns_per_col / 1e6)}/col", timeout=2)

    def action_jump_start(self) -> None:
        self.cursor_ns = self._time_start

    def action_jump_end(self) -> None:
        self.cursor_ns = self._time_end

    # -------------------------------------------------------------------------
    # Toggle actions
    # -------------------------------------------------------------------------
    def action_toggle_time_mode(self) -> None:
        """Toggle absolute / relative time axis ('a' key — matches original)."""
        self._relative_time = not self._relative_time
        self._push_canvas_state()
        self.notify(f"Time: {'relative' if self._relative_time else 'absolute'}", timeout=2)

    def action_toggle_logical(self) -> None:
        """Toggle TIME / LOGICAL mode ('L' key — matches original)."""
        self._logical_mode = not self._logical_mode
        self._push_canvas_state()
        self._update_title()
        self.notify(f"Mode: {'LOGICAL (NVTX hidden)' if self._logical_mode else 'TIME'}", timeout=2)

    def action_toggle_path_mode(self) -> None:
        """Toggle NVTX path display between breadcrumb and hierarchy ('P' key)."""
        bp = self.query_one("#bottom-panel", BottomPanel)
        bp.toggle_path_mode()
        self._update_bottom_panel()
        mode = bp._path_mode
        self.notify(f"Path: {mode}", timeout=2)

    def action_toggle_merged(self) -> None:
        """Toggle merged all-streams view ('S' key)."""
        self._merged_mode = not self._merged_mode
        self._push_canvas_state()
        self.notify(f"View: {'merged' if self._merged_mode else 'per-stream'}", timeout=2)

    def action_toggle_labels(self) -> None:
        """Cycle label visibility: gpu+stream → gpu → none ('I' key)."""
        modes = ["gpu+stream", "gpu", "none"]
        idx = modes.index(self._label_mode)
        self._label_mode = modes[(idx + 1) % len(modes)]
        self._push_canvas_state()
        self.notify(f"Labels: {self._label_mode}", timeout=2)

    def action_toggle_demangled(self) -> None:
        self._show_demangled = not self._show_demangled
        self._push_canvas_state()
        self.notify(f"Demangled: {'on' if self._show_demangled else 'off'}", timeout=2)

    def action_toggle_help(self) -> None:
        self.notify(
            "←→:pan  Shift+←→:page  ↑↓:stream  Tab:kernel  +/-:zoom  "
            "a:time  L:logical  /:filter  m:mindur  d:demangled  "
            "B:bookmark  ,.:cycle  ':list  []:range  `:back  "
            "A:chat  C:config  T:ticks  Home/End  q:quit",
            timeout=6,
        )

    def action_toggle_config(self) -> None:
        self.query_one("#config-panel", ConfigPanel).toggle()

    @on(ConfigPanel.Changed)
    def _config_changed(self) -> None:
        """Re-render canvas whenever the user adjusts a ConfigPanel value."""
        self._push_canvas_state()

    def action_toggle_chat(self) -> None:
        cp = self.query_one("#chat-panel", ChatPanel)
        if "-active" in cp.classes:
            cp.remove_class("-active")
            self.query_one("#canvas", TimelineCanvas).focus()
        else:
            cp.add_class("-active")
            cp.query_one("#chat-input", Input).focus()

    def action_toggle_tick_density(self) -> None:
        """Cycle tick density: 3 → 6 → 10 → 15 → 3 ('T' key — matches original)."""
        densities = [3, 6, 10, 15]
        try:
            ni = (densities.index(self._tick_density) + 1) % len(densities)
        except ValueError:
            ni = 1
        self._tick_density = densities[ni]
        self._push_canvas_state()
        self.notify(f"Tick density: {self._tick_density}", timeout=2)

    # -------------------------------------------------------------------------
    # Filter
    # -------------------------------------------------------------------------
    def action_open_filter(self) -> None:
        fb = self.query_one("#tl-filter-bar", TimelineFilterBar)
        fb.show_bar(self.filter_text)

    def action_clear_filter(self) -> None:
        self.filter_text = ""
        self.notify("Filter cleared", timeout=2)

    @on(Input.Submitted, "#tl-filter-input")
    def _tl_filter_submitted(self, event: Input.Submitted) -> None:
        self.filter_text = event.value
        self.query_one("#tl-filter-bar", TimelineFilterBar).hide_bar()
        self.query_one("#canvas", TimelineCanvas).focus()

    @on(Input.Submitted, "#tl-mindur-input")
    def _tl_mindur_submitted(self, event: Input.Submitted) -> None:
        try:
            self.min_dur_us = max(0, float(event.value)) if event.value.strip() else 0
        except ValueError:
            self.notify("Invalid number", severity="warning", timeout=2)
        self.query_one("#tl-mindur-bar", TimelineMinDurBar).hide_bar()
        self.query_one("#canvas", TimelineCanvas).focus()

    # -------------------------------------------------------------------------
    # Min duration
    # -------------------------------------------------------------------------
    def action_open_min_dur(self) -> None:
        """Open interactive min-duration input ('m' key — matches original)."""
        mdb = self.query_one("#tl-mindur-bar", TimelineMinDurBar)
        mdb.show_bar(self.min_dur_us)

    # -------------------------------------------------------------------------
    # Bookmarks
    # -------------------------------------------------------------------------
    def _save_prev_position(self) -> None:
        self._prev_position = {
            "cursor_ns": self.cursor_ns,
            "stream": self.selected_stream_idx,
        }

    def action_save_bookmark(self) -> None:
        stream = self._streams[self.selected_stream_idx] if self._streams else "?"
        k = kernel_at_time(self._stream_kernels.get(stream, []), self.cursor_ns)
        name = f"#{len(self._bookmarks) + 1} @{_fmt_ns(self.cursor_ns)}"
        bm: dict = {
            "name": name,
            "cursor_ns": self.cursor_ns,
            "stream": self.selected_stream_idx,
        }
        if k:
            bm["kernel_name"] = k.name
            if k.nvtx_path:
                bm["nvtx_path"] = k.nvtx_path
        # Attach pending range if set
        if self._range_start_ns is not None:
            bm["range_start_ns"] = self._range_start_ns
            bm["range_end_ns"] = self.cursor_ns
            self._range_start_ns = None
        self._bookmarks.append(bm)
        self._bookmark_idx = len(self._bookmarks) - 1
        self.notify(f"📌 {name}", timeout=2)
        self._update_title()

    def action_prev_bookmark(self) -> None:
        if self._bookmarks:
            self._bookmark_idx = (self._bookmark_idx - 1) % len(self._bookmarks)
            self._jump_bookmark(self._bookmarks[self._bookmark_idx])

    def action_next_bookmark(self) -> None:
        if self._bookmarks:
            self._bookmark_idx = (self._bookmark_idx + 1) % len(self._bookmarks)
            self._jump_bookmark(self._bookmarks[self._bookmark_idx])

    def action_show_bookmark_list(self) -> None:
        """Show numbered bookmark list ('apostrophe' key — matches original)."""
        if not self._bookmarks:
            self.notify("No bookmarks saved", timeout=2)
            return
        panel = self.query_one("#bm-panel", TimelineBookmarkPanel)
        panel.show_panel(self._bookmarks)

    def action_range_start(self) -> None:
        """Set range bookmark start ('[' key — matches original)."""
        self._range_start_ns = self.cursor_ns
        self.notify(f"[range start: {_fmt_ns(self.cursor_ns)}", timeout=2)

    def action_range_end(self) -> None:
        """Set range bookmark end, auto-saves range bookmark (']' key — matches original)."""
        if self._range_start_ns is None:
            self.notify("Press [ first to set range start", timeout=2)
            return
        name = (
            f"#{len(self._bookmarks) + 1} {_fmt_ns(self._range_start_ns)}↔{_fmt_ns(self.cursor_ns)}"
        )
        bm: dict = {
            "name": name,
            "cursor_ns": self._range_start_ns,
            "stream": self.selected_stream_idx,
            "range_start_ns": self._range_start_ns,
            "range_end_ns": self.cursor_ns,
        }
        self._bookmarks.append(bm)
        self._bookmark_idx = len(self._bookmarks) - 1
        self._range_start_ns = None
        self.notify(f"📌 range saved: {name}", timeout=3)
        self._update_title()

    def action_jump_back(self) -> None:
        """Jump to previous cursor position ('`' backtick — matches original)."""
        if self._prev_position is None:
            self.notify("No previous position", timeout=2)
            return
        old = self._prev_position
        self._save_prev_position()
        self.cursor_ns = old["cursor_ns"]
        self.selected_stream_idx = old["stream"]
        self.notify("Jumped back", timeout=2)

    def jump_to_bookmark_n(self, n: int) -> None:
        """Public API: jump to bookmark by 1-based index (used by TimelineBookmarkPanel)."""
        if 1 <= n <= len(self._bookmarks):
            self._bookmark_idx = n - 1
            self._jump_bookmark(self._bookmarks[self._bookmark_idx])

    def _jump_bookmark(self, bm: dict) -> None:
        self._save_prev_position()
        self.cursor_ns = bm["cursor_ns"]
        self.selected_stream_idx = bm.get("stream", 0)
        if "range_start_ns" in bm and "range_end_ns" in bm:
            # Zoom to the range
            span = bm["range_end_ns"] - bm["range_start_ns"]
            canvas = self.query_one("#canvas", TimelineCanvas)
            timeline_w = max(canvas.size.width - canvas.label_w, 1)
            self.ns_per_col = max(1, span // timeline_w)
            self.cursor_ns = bm["range_start_ns"]
        self.notify(f"📌 {bm['name']}", timeout=2)

    # -------------------------------------------------------------------------
    # AI integration
    # -------------------------------------------------------------------------
    def _build_ui_context(self) -> dict:
        stream = self._streams[self.selected_stream_idx] if self._streams else "?"
        k = kernel_at_time(self._stream_kernels.get(stream, []), self.cursor_ns)
        return {
            "selected_kernel": {
                "name": k.name,
                "duration_ms": k.duration_ms,
                "stream": k.stream,
            }
            if k
            else None,
            "view_state": {
                "time_range_s": [self._trim[0] / 1e9, self._trim[1] / 1e9],
                "cursor_ns": self.cursor_ns,
                "ns_per_col": self.ns_per_col,
                "mode": "LOGICAL" if self._logical_mode else "TIME",
            },
            "stats": {"kernel_count": len(self._kernels)},
        }

    def _handle_ai_action(self, action_dict: dict) -> None:
        tui_actions.execute_tui_action(action_dict, self)

    # -------------------------------------------------------------------------
    # Public API (for tui_actions dispatcher)
    # -------------------------------------------------------------------------
    def scroll_to_kernel(self, target_name: str, occurrence_index: int = 1) -> None:
        result = find_kernel_by_name(self._stream_kernels, target_name, occurrence_index)
        if result is None:
            self.notify(f"AI: kernel not found: {target_name}", severity="warning", timeout=3)
            return
        stream, idx = result
        if stream in self._streams:
            self.selected_stream_idx = self._streams.index(stream)
        self.cursor_ns = self._stream_kernels[stream][idx].start_ns
        self.notify(f"AI → {target_name} (occ {occurrence_index})", timeout=3)

    def zoom_to_time_range(self, start_s: float, end_s: float) -> None:
        if end_s <= start_s:
            return
        start_ns = int(start_s * 1e9)
        end_ns = int(end_s * 1e9)
        canvas = self.query_one("#canvas", TimelineCanvas)
        timeline_w = max(canvas.size.width - canvas.label_w, 1)
        span_ns = end_ns - start_ns
        self.ns_per_col = max(1, span_ns // timeline_w)
        self.cursor_ns = start_ns
        self.notify(f"AI zoom → {start_s:.2f}s–{end_s:.2f}s", timeout=3)

    def _session_mode(self) -> bool:
        return self._session_id is not None

    def _open_session(self, session: str) -> None:
        from ..profile_runner import build_local_profile_reference
        from ..session_cli import project_loop_state, resolve_session_id, session_dir
        from ..session_store import SessionExistsError, SessionStore

        if not self._db_path:
            raise ValueError(
                "session mode requires a before profile path to open or derive "
                "the session id"
            )
        before_ref = build_local_profile_reference(self._db_path)
        session_id = resolve_session_id(session or None, before=before_ref)
        store = SessionStore(self._session_root)
        try:
            store.create(session_id, before_profile=before_ref)
        except SessionExistsError:
            pass
        snapshot = store.load(session_id)
        self._session_id = session_id
        self._session_projection = project_loop_state(
            snapshot,
            session_dir_path=session_dir(session_id, root=self._session_root),
        )
        self.analysis_phase = str(self._session_projection["phase"])

    def _load_session_projection(self) -> dict:
        from ..session_cli import project_loop_state, session_dir
        from ..session_store import SessionStore

        snapshot = SessionStore(self._session_root).load(self._session_id)
        projected = project_loop_state(
            snapshot,
            session_dir_path=session_dir(self._session_id, root=self._session_root),
        )
        self._session_projection = projected
        self.analysis_phase = str(projected["phase"])
        return projected

    def _session_limitation(self, cli_command: str, detail: str) -> None:
        """Stated limitation: reduced capability, never a silent no-op."""
        self.notify(
            f"{detail} (use {cli_command})",
            severity="warning",
            timeout=5,
        )

    def set_analysis_phase(self, phase: str) -> None:
        if not self._session_mode():
            self._session_limitation(
                "nsys-ai evidence|propose|diff",
                "loop actions require a SessionStore session (open with a profile path)",
            )
            return
        self._session_limitation(
            "nsys-ai evidence|propose|diff",
            "session mode does not set phase directly; CLI publishers "
            "advance the session phase",
        )

    def action_loop_diagnose(self) -> None:
        self._session_limitation(
            "nsys-ai evidence build <profile> --session",
            "session mode does not run diagnose in the timeline TUI",
        )

    def action_loop_propose(self) -> None:
        self._session_limitation(
            "nsys-ai propose --session",
            "session mode does not save free-text proposals in the timeline TUI",
        )

    def action_loop_reprofile(self) -> None:
        self._session_limitation(
            "nsys-ai profile / nsys-ai diff --session",
            "session mode does not register an after profile from the timeline TUI",
        )

    def action_loop_diff(self) -> None:
        if not self._session_mode():
            self._session_limitation(
                "nsys-ai diff <before> <after> --session",
                "loop diff requires a SessionStore session",
            )
            return
        # C6: read-only reload of SessionSnapshot.diff. No analysis, no publish_*.
        try:
            projected = self._load_session_projection()
        except Exception as e:
            self.notify(f"Loop diff failed: {e}", severity="warning", timeout=4)
            return
        self._update_title()
        if projected.get("diff_summary") is None:
            self._session_limitation(
                "nsys-ai diff <before> <after> --session",
                "session has no published diff yet",
            )
            return
        self.notify(
            f"Loop diff verdict: {projected.get('verdict', 'neutral')}",
            timeout=3,
        )

    def action_loop_accept(self) -> None:
        self._record_loop_decision("accept", "accepted in timeline TUI")

    def set_loop_decision(self, decision: str, reason: str = "") -> None:
        self._record_loop_decision(decision, reason)

    def _record_loop_decision(self, decision: str, reason: str) -> None:
        """Persist a loop accept/reject via SessionStore (C4)."""
        if not self._session_mode():
            self.notify(
                "Loop decision requires a SessionStore session",
                severity="warning",
                timeout=4,
            )
            return
        projected = self._session_projection or {}
        if projected.get("diff_summary") is None:
            try:
                projected = self._load_session_projection()
            except Exception as e:
                self.notify(
                    f"Loop decision failed: {e}", severity="warning", timeout=4
                )
                return
        if projected.get("diff_summary") is None:
            self.notify(
                "Run loop diff before recording a decision",
                severity="warning",
                timeout=4,
            )
            return
        reason = (reason or "").strip() or f"{decision}ed in timeline TUI"
        try:
            from ..session_store import SessionConflictError, SessionStore

            store = SessionStore(self._session_root)
            with store.writer(self._session_id) as writer:
                writer.publish_decision(decision, reason)
            projected = self._load_session_projection()
        except SessionConflictError as e:
            self.notify(f"Loop decision failed: {e}", severity="warning", timeout=4)
            return
        except ValueError as e:
            self.notify(f"Loop decision failed: {e}", severity="warning", timeout=4)
            return
        except Exception as e:
            self.notify(f"Loop decision failed: {e}", severity="warning", timeout=4)
            return
        self._update_title()
        path = projected.get("decision_path") or ""
        self.notify(f"Loop decision: {decision} -> {path}", timeout=3)


# ---------------------------------------------------------------------------
# Entry point (replaces tui_timeline.run_timeline)
# ---------------------------------------------------------------------------


def run_timeline(
    db_path: str,
    device: int,
    trim: tuple[int, int] | None,
    min_ms: float = 0,
    loop_after: str | None = None,
    session: str | None = None,
) -> None:
    """Launch the Textual horizontal timeline browser."""
    app = NsysTimelineApp(
        db_path, device, trim, min_ms=min_ms, loop_after=loop_after, session=session
    )
    app.run()
