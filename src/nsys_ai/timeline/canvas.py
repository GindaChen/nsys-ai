"""
timeline/canvas.py — TimelineCanvas: Textual widget for the horizontal timeline.

Uses render_line(y) → Strip for cell-level control over block characters,
matching the Perfetto-style rendering from the old curses TimelineTUI.
"""
from __future__ import annotations

from rich.color import Color
from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip
from textual.widget import Widget

from ..formatting import fmt_dur as _fmt_dur
from ..tui_models import KernelEvent, NvtxSpan
from .logic import filter_kernels

# ---------------------------------------------------------------------------
# Stream color palette (7 slots, cycles)
# ---------------------------------------------------------------------------
_STREAM_COLORS = [
    Color.parse("bright_green"),
    Color.parse("bright_cyan"),
    Color.parse("bright_yellow"),
    Color.parse("bright_magenta"),
    Color.parse("bright_blue"),
    Color.parse("bright_red"),
    Color.parse("white"),
]

_NCCL_COLOR = Color.parse("magenta")


def _stream_color(stream_idx: int, selected: bool, heat: float,
                   is_nccl: bool, parity: int = 0) -> Style:
    color = _NCCL_COLOR if is_nccl else _STREAM_COLORS[stream_idx % len(_STREAM_COLORS)]
    bold = selected or heat > 0.7
    # Alternate brightness for adjacent kernels: even = normal, odd = dimmed
    dim = (parity % 2 == 1 and not selected) or (heat < 0.2 and not selected)
    return Style(color=color, bold=bold, dim=dim)


class TimelineCanvas(Widget):
    """Horizontal timeline widget rendering kernel blocks with Strip-based cell control.

    Reactive inputs (set by NsysTimelineApp):
        cursor_ns, viewport_start_ns, ns_per_col, selected_stream_idx
        stream_kernels, streams, nvtx_spans, ...
    """

    # Accept focus so that App-level bindings fire (key events bubble to App when
    # no focusable child swallows them).
    can_focus = True

    DEFAULT_CSS = """
    TimelineCanvas {
        height: 1fr;
    }
    """

    def __init__(
        self,
        label_w: int = 8,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.label_w = label_w

        # Populated by the parent app before first render
        self.cursor_ns: int = 0
        self.viewport_start_ns: int = 0
        self.ns_per_col: int = 1_000_000
        self.selected_stream_idx: int = 0
        self.streams: list[str] = []
        self.stream_kernels: dict[str, list[KernelEvent]] = {}
        self.stream_color_idx: dict[str, int] = {}
        self.nvtx_spans: list[NvtxSpan] = []
        self.nvtx_max_depth: int = 0
        self.filter_text: str = ""
        self.min_dur_us: float = 0
        self.selected_stream_rows: int = 3
        self.default_stream_rows: int = 1
        # Display flags
        self.relative_time: bool = False
        self.show_demangled: bool = False
        self.tick_density: int = 6
        # Multi-GPU fields
        self.merged_mode: bool = False
        self.label_mode: str = "gpu+stream"
        self.stream_gpu: dict[str, int] = {}
        self.gpu_kernels: dict[int, list[KernelEvent]] = {}

    # ------------------------------------------------------------------
    # Textual rendering
    # ------------------------------------------------------------------

    def _effective_label_w(self) -> int:
        if self.label_mode == "none":
            return 0
        elif self.label_mode == "gpu":
            return 5  # "G0  " etc.
        return self.label_w  # "gpu+stream"

    def render_line(self, y: int) -> Strip:
        """Return a Strip for each terminal row."""
        width = self.size.width
        if width <= 0:
            return Strip.blank(width, Style())
        if not self.streams:
            return Strip([Segment(" " * width, Style())])

        label_w = self._effective_label_w()
        timeline_w = max(width - label_w, 1)

        # Layout: row 0 = time axis, rows 1..nvtx_max_depth = NVTX, then stream rows
        axis_row = 0
        nvtx_start = 1
        nvtx_end = nvtx_start + self.nvtx_max_depth
        stream_start = nvtx_end + 1  # +1 for separator row

        if y == axis_row:
            return self._render_time_axis(timeline_w, label_w)

        if nvtx_start <= y < nvtx_end:
            depth = y - nvtx_start
            return self._render_nvtx_row(depth, timeline_w, label_w, width)

        if y == nvtx_end:
            return Strip([Segment("─" * width, Style(dim=True))])

        # Stream rows — build layout with GPU separators
        row_y = y - stream_start
        cur_y = 0
        prev_gpu: int | None = None

        for si, stream in enumerate(self.streams):
            gpu_id = self.stream_gpu.get(stream, 0)

            # GPU separator row between different GPUs
            if prev_gpu is not None and gpu_id != prev_gpu:
                if row_y == cur_y:
                    sep_label = f"─ GPU {gpu_id} "
                    sep = sep_label + "─" * max(0, width - len(sep_label))
                    return Strip([Segment(sep, Style(color="bright_white", bold=True))])
                cur_y += 1
            prev_gpu = gpu_id

            row_h = self.selected_stream_rows if si == self.selected_stream_idx else self.default_stream_rows
            if cur_y <= row_y < cur_y + row_h:
                is_selected = (si == self.selected_stream_idx)
                within = row_y - cur_y
                return self._render_stream_row(stream, si, within, row_h, is_selected,
                                               timeline_w, label_w, width)
            cur_y += row_h
            if cur_y > row_y + 100:
                break

        return Strip.blank(width, Style())

    # ------------------------------------------------------------------
    # Per-row renderers
    # ------------------------------------------------------------------

    def _render_time_axis(self, timeline_w: int, label_w: int) -> Strip:
        from ..formatting import fmt_ns as _fmt_ns
        from ..formatting import fmt_relative as _fmt_relative
        from .logic import nice_tick_interval

        segments: list[Segment] = [Segment(" " * label_w, Style(dim=True))]
        tick_interval = nice_tick_interval(timeline_w, self.ns_per_col, self.tick_density)
        view_end = self.viewport_start_ns + self.ns_per_col * timeline_w

        axis = [" "] * timeline_w

        if self.relative_time:
            # Show absolute origin at far left, then +offset labels for ticks
            origin = self.viewport_start_ns
            left_label = _fmt_ns(origin)
            for ci, ch in enumerate(left_label):
                if ci < timeline_w:
                    axis[ci] = ch
            t = ((self.viewport_start_ns // tick_interval) + 1) * tick_interval
            while t < view_end:
                col = int((t - self.viewport_start_ns) / max(self.ns_per_col, 1))
                if 0 <= col < timeline_w:
                    label = _fmt_relative(t - origin)
                    for ci, ch in enumerate(label):
                        if col + ci < timeline_w:
                            axis[col + ci] = ch
                t += tick_interval
        else:
            t = ((self.viewport_start_ns // tick_interval) + 1) * tick_interval
            while t < view_end:
                col = int((t - self.viewport_start_ns) / max(self.ns_per_col, 1))
                if 0 <= col < timeline_w:
                    label = _fmt_ns(t)
                    for ci, ch in enumerate(label):
                        if col + ci < timeline_w:
                            axis[col + ci] = ch
                t += tick_interval

        segments.append(Segment("".join(axis), Style(dim=True)))
        return Strip(segments)

    def _render_nvtx_row(self, depth: int, timeline_w: int, label_w: int, width: int) -> Strip:
        view_end = self.viewport_start_ns + self.ns_per_col * timeline_w
        label_seg = Segment(f"N{depth}".ljust(label_w - 1), Style(color="blue", dim=True))
        cells = [" "] * timeline_w

        for span in self.nvtx_spans:
            if span.depth != depth:
                continue
            if span.end_ns < self.viewport_start_ns or span.start_ns >= view_end:
                continue
            s_col = max(0, int((span.start_ns - self.viewport_start_ns) / max(self.ns_per_col, 1)))
            e_col = min(timeline_w - 1, int((span.end_ns - self.viewport_start_ns) / max(self.ns_per_col, 1)))
            span_w = max(1, e_col - s_col + 1)
            if span_w >= len(span.name) + 2:
                content = f"[{span.name}]"
            elif span_w >= 5:
                # Truncated name with ellipsis: [na…]
                inner = span_w - 3  # room for [ … ]
                content = f"[{span.name[:inner]}…]"
            elif span_w >= 2:
                content = f"[{']' if span_w == 2 else span.name[:span_w - 2] + ']'}"
            else:
                content = "│"  # thin marker instead of fat block
            for ci, ch in enumerate(content[:span_w]):
                if s_col + ci < timeline_w:
                    cells[s_col + ci] = ch

        return Strip([label_seg, Segment("".join(cells), Style(color="blue"))])

    def _render_stream_row(
        self,
        stream: str,
        stream_idx: int,
        within: int,
        row_h: int,
        is_selected: bool,
        timeline_w: int,
        label_w: int,
        width: int,
    ) -> Strip:
        from ..tui_models import short_kernel_name as _short_name

        ci = self.stream_color_idx.get(stream, stream_idx % len(_STREAM_COLORS))
        # Check if this stream is predominantly NCCL — match label color to kernel color
        all_kernels = self.stream_kernels.get(stream, [])
        is_nccl_stream = all_kernels and all_kernels[0].is_nccl
        label_color = _NCCL_COLOR if is_nccl_stream else _STREAM_COLORS[ci % len(_STREAM_COLORS)]
        label_style = Style(
            color=label_color,
            bold=is_selected,
            dim=not is_selected,
        )
        # Build label based on label_mode
        gpu_id = self.stream_gpu.get(stream, 0)
        # stream key format is "gpu:stream_id"
        stream_id = stream.split(":", 1)[1] if ":" in stream else stream
        if self.label_mode == "none":
            label_text = ""
        elif self.label_mode == "gpu":
            label_text = f"G{gpu_id}"
        else:
            label_text = f"G{gpu_id}:S{stream_id}"

        # Only show stream label on the first sub-row
        if within == 0:
            label_seg = Segment(label_text.ljust(label_w - 1) if label_w > 0 else "", label_style)
        else:
            label_seg = Segment(" " * max(0, label_w - 1), Style())

        kernels = filter_kernels(
            self.stream_kernels.get(stream, []),
            self.filter_text,
            self.min_dur_us,
        )
        cells = [(" ", Style())] * timeline_w

        is_block_row = (within == row_h - 1)
        # How many label rows do we have above the block row?
        label_row_count = row_h - 1  # row_h=1 → 0 label rows, row_h=2 → 1, row_h=3 → 2, etc.

        kernel_idx = 0  # counter for color alternation
        for k in kernels:
            s_col = int((k.start_ns - self.viewport_start_ns) / max(self.ns_per_col, 1))
            e_col = int((k.end_ns - self.viewport_start_ns) / max(self.ns_per_col, 1))
            if e_col < 0 or s_col >= timeline_w:
                continue
            s_col = max(0, s_col)
            e_col = min(timeline_w - 1, max(s_col, e_col))
            block_w = e_col - s_col + 1

            is_at_cursor = is_selected and k.start_ns <= self.cursor_ns <= k.end_ns
            style = _stream_color(ci, is_at_cursor, k.heat, k.is_nccl, parity=kernel_idx)
            kernel_idx += 1

            if is_block_row:
                # Last row: block bars
                char = "█"
                for col in range(s_col, s_col + block_w):
                    if col < timeline_w:
                        cells[col] = (char, style)
            elif label_row_count > 0:
                # Determine which label row this kernel belongs to.
                # Row 0 (within=0): ONLY the focused/cursor kernel
                # Rows 1+ (within=1..): other kernels distributed to minimize overlap
                if is_at_cursor:
                    target_row = 0
                elif label_row_count >= 2:
                    # Distribute non-focused kernels across rows 1..label_row_count-1
                    # Use position-based assignment to spread them out
                    target_row = 1 + (s_col % max(1, label_row_count - 1))
                else:
                    # Only 1 label row total — everything shares row 0
                    target_row = 0

                if within != target_row:
                    continue

                # Render kernel label with overflow
                name_to_use = (k.demangled if self.show_demangled and k.demangled else k.name)
                short = _short_name(name_to_use)
                dur = _fmt_dur(k.duration_ms)
                if len(short) + len(dur) + 1 <= 40:
                    text = f"{short} {dur}"
                else:
                    text = short

                # Calculate available space: extend rightward into empty cells
                avail = block_w
                for probe in range(s_col + block_w, min(s_col + len(text), timeline_w)):
                    if cells[probe][0] == " ":
                        avail += 1
                    else:
                        break

                label_text = text[:avail] if avail >= 2 else ""
                for ci2, ch in enumerate(label_text):
                    if s_col + ci2 < timeline_w:
                        cells[s_col + ci2] = (ch, style)

        # Cursor highlight — preserve the existing character, just change its style
        cursor_col = int((self.cursor_ns - self.viewport_start_ns) / max(self.ns_per_col, 1))
        if 0 <= cursor_col < timeline_w:
            existing_ch, _ = cells[cursor_col]
            ch = existing_ch if existing_ch.strip() else "│"
            cells[cursor_col] = (ch, Style(color="black", bgcolor="yellow", bold=True))

        segments: list[Segment] = [label_seg]
        # Compress runs of same style
        if cells:
            cur_ch, cur_st = cells[0]
            run = cur_ch
            for ch, st in cells[1:]:
                if st == cur_st:
                    run += ch
                else:
                    segments.append(Segment(run, cur_st))
                    cur_ch, cur_st = ch, st
                    run = ch
            segments.append(Segment(run, cur_st))

        return Strip(segments)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def refresh_from_app(self, **kwargs: object) -> None:
        """Update canvas state from app and request re-render."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.refresh()
