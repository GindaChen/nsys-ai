"""
tui_timeline.py — Horizontal timeline TUI v2 (Perfetto-style).

TIME-CURSOR driven: ←/→ pans through time, ↑/↓ selects stream.
The cursor is a nanosecond timestamp; viewport auto-centers on it.

Keybindings:
    ←/→          Pan through time (1 column per press)
    Shift+←/→    Page pan (1/4 viewport)
    ↑/↓          Select stream (keeps time position)
    +/=  -/_     Zoom in / out
    Tab          Snap to next kernel on selected stream
    Shift+Tab    Snap to previous kernel
    a            Toggle absolute / relative time axis
    L            Toggle TIME / LOGICAL ordering
    /            Filter by kernel name
    n            Clear filter
    m            Set min duration threshold
    d            Toggle demangled names
    h            Toggle help overlay
    B            Save bookmark at cursor
    , / .        Cycle through bookmarks
    '            Show bookmark list (number to jump)
    [            Set range bookmark start
    ]            Set range bookmark end (saves range)
    Home / End   Jump to start / end of trace
    q            Quit
    Ctrl+C       Exit cleanly
"""
import curses
import os
from typing import Optional


def _fmt_dur(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    if ms >= 1:
        return f"{ms:.1f}ms"
    return f"{ms * 1000:.0f}μs"


def _fmt_ns(ns) -> str:
    if ns is None:
        return "?"
    return f"{ns / 1e9:.3f}s"


def _fmt_relative(ns_offset) -> str:
    """Format as +0.5s, +1.0s etc."""
    s = ns_offset / 1e9
    if s < 0.001:
        return "+0"
    return f"+{s:.1f}s" if s >= 0.1 else f"+{s * 1000:.0f}ms"


def _short_kernel_name(name: str) -> str:
    """Shorten kernel name for inline display."""
    for prefix in ('void ', 'at::native::', 'at::cuda::', 'cutlass::',
                   'cublasLt', 'cublas', 'sm90_', 'sm80_'):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if '<' in name:
        name = name[:name.index('<')]
    if name.endswith('_kernel'):
        name = name[:-7]
    return name if name else '?'


class KernelEvent:
    __slots__ = ('name', 'demangled', 'start_ns', 'end_ns', 'duration_ms',
                 'stream', 'heat', 'nvtx_path', 'is_nccl')

    def __init__(self, json_node: dict, path: str = ''):
        self.name = json_node.get('name', '?')
        self.demangled = json_node.get('demangled', '')
        self.start_ns = json_node.get('start_ns', 0)
        self.end_ns = json_node.get('end_ns', 0)
        self.duration_ms = json_node.get('duration_ms', 0)
        self.stream = str(json_node.get('stream', '?'))
        self.heat = json_node.get('heat', 0)
        self.nvtx_path = path
        self.is_nccl = 'nccl' in self.name.lower()


class NvtxSpan:
    __slots__ = ('name', 'start_ns', 'end_ns', 'depth', 'path')

    def __init__(self, name: str, start_ns: int, end_ns: int, depth: int, path: str):
        self.name = name
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.depth = depth
        self.path = path


class TimelineTUI:
    """Horizontal timeline viewer — time-cursor driven."""

    def __init__(self, json_roots: list[dict], title: str = "Timeline",
                 db_path: str = '', device: int = 0,
                 trim: tuple = (0, 0)):
        self.title = title
        self.db_path = db_path
        self.device = device
        self.trim = trim

        # Extract events
        self.kernels: list[KernelEvent] = []
        self.nvtx_spans: list[NvtxSpan] = []
        self._extract_events(json_roots, '', 0)
        self.kernels.sort(key=lambda k: k.start_ns)
        self.nvtx_spans.sort(key=lambda s: s.start_ns)

        # Streams
        stream_set = sorted(set(k.stream for k in self.kernels),
                           key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0))
        self.streams = stream_set if stream_set else ['?']

        # Per-stream sorted kernel lists
        self.stream_kernels: dict[str, list[KernelEvent]] = {}
        for s in self.streams:
            self.stream_kernels[s] = sorted(
                [k for k in self.kernels if k.stream == s],
                key=lambda k: k.start_ns)

        # Stream colors
        self.stream_color_idx: dict[str, int] = {}
        for i, s in enumerate(self.streams):
            self.stream_color_idx[s] = i % 7

        # Time bounds
        if self.kernels:
            self.time_start = min(k.start_ns for k in self.kernels)
            self.time_end = max(k.end_ns for k in self.kernels)
        else:
            self.time_start = trim[0]
            self.time_end = trim[1]
        self.time_span = max(self.time_end - self.time_start, 1)

        # NVTX depth
        self.nvtx_max_depth = min(4, max((s.depth for s in self.nvtx_spans), default=0) + 1)

        # ── Cursor state ──
        self.cursor_ns = self.time_start  # time position of cursor
        self.selected_stream = 0
        self.ns_per_col = max(1, self.time_span // 100)
        self.view_start = self.time_start

        # ── Options ──
        self.logical_mode = False
        self.relative_time = False
        self.show_demangled = False
        self.show_help = False
        self.show_config = False  # config panel toggle
        self.min_dur_us = 0
        self.filter_text = ''
        self.filter_mode = False
        self.filter_input = ''
        self.threshold_mode = False
        self.threshold_input = ''
        self.status_msg = ''
        self.tick_density = 6

        # Stream row heights
        self.selected_stream_rows = 2   # rows for selected stream
        self.default_stream_rows = 1    # rows for other streams

        # ── Bookmarks ──
        self.bookmarks: list[dict] = []
        self.bookmark_idx = -1
        self.range_start_ns: Optional[int] = None
        self.bookmark_mode = False
        self.bookmark_input = ''
        self.bookmark_list_mode = False
        self.prev_position: Optional[dict] = None  # {cursor_ns, stream} for jump-back

        # Config panel state
        self.config_items = [
            'selected_stream_rows', 'default_stream_rows',
            'tick_density', 'nvtx_max_depth', 'min_dur_us',
        ]
        self.config_cursor = 0

    def _extract_events(self, nodes: list[dict], path: str, depth: int):
        for node in nodes:
            node_path = f"{path} > {node['name']}" if path else node['name']
            ntype = node.get('type', '')
            if ntype == 'kernel':
                self.kernels.append(KernelEvent(node, path))
            elif ntype == 'nvtx':
                self.nvtx_spans.append(NvtxSpan(
                    node['name'],
                    node.get('start_ns', 0) or 0,
                    node.get('end_ns', 0) or 0,
                    depth, node_path))
            if node.get('children'):
                self._extract_events(node['children'], node_path, depth + 1)

    def _get_stream_kernels(self, stream: str) -> list[KernelEvent]:
        ks = self.stream_kernels.get(stream, [])
        if self.min_dur_us > 0:
            min_ms = self.min_dur_us / 1000.0
            ks = [k for k in ks if k.duration_ms >= min_ms]
        if self.filter_text:
            ft = self.filter_text.lower()
            ks = [k for k in ks if ft in k.name.lower() or
                  (k.demangled and ft in k.demangled.lower())]
        return ks

    def _kernel_at_time(self, stream: str, ns: int) -> Optional[KernelEvent]:
        """Find the kernel containing or nearest to the given timestamp."""
        ks = self._get_stream_kernels(stream)
        if not ks:
            return None
        # Check if cursor is inside a kernel
        for k in ks:
            if k.start_ns <= ns <= k.end_ns:
                return k
        # Find nearest
        best = min(ks, key=lambda k: min(abs(k.start_ns - ns), abs(k.end_ns - ns)))
        return best

    def _kernel_index_at_time(self, stream: str, ns: int) -> int:
        """Find index of kernel nearest to timestamp."""
        ks = self._get_stream_kernels(stream)
        if not ks:
            return -1
        best_i = 0
        best_dist = abs(ks[0].start_ns - ns)
        for i, k in enumerate(ks):
            d = min(abs(k.start_ns - ns), abs(k.end_ns - ns))
            if d < best_dist:
                best_dist = d
                best_i = i
        return best_i

    def _kernel_name(self, k: KernelEvent) -> str:
        if self.show_demangled and k.demangled:
            return k.demangled
        return k.name

    def _center_viewport(self, timeline_w: int):
        """Center viewport on cursor."""
        half = (self.ns_per_col * timeline_w) // 2
        self.view_start = self.cursor_ns - half

    def run(self, stdscr):
        curses.curs_set(0)
        curses.use_default_colors()

        # Stream palette: pairs 11-17 (fg), 21-27 (bg)
        palette = [curses.COLOR_GREEN, curses.COLOR_CYAN, curses.COLOR_YELLOW,
                   curses.COLOR_MAGENTA, curses.COLOR_BLUE, curses.COLOR_RED,
                   curses.COLOR_WHITE]
        for i, c in enumerate(palette):
            curses.init_pair(11 + i, c, -1)
            curses.init_pair(21 + i, curses.COLOR_BLACK, c)

        curses.init_pair(5, curses.COLOR_BLUE, -1)
        curses.init_pair(6, curses.COLOR_WHITE, -1)
        curses.init_pair(7, curses.COLOR_RED, -1)
        curses.init_pair(8, curses.COLOR_YELLOW, -1)
        curses.init_pair(9, curses.COLOR_CYAN, -1)

        try:
            self._main_loop(stdscr)
        except KeyboardInterrupt:
            pass

    def _main_loop(self, stdscr):
        label_w = 8

        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()
            timeline_w = max(width - label_w - 1, 20)

            stream = self.streams[self.selected_stream] if self.streams else '?'
            sk = self._get_stream_kernels(stream)

            # Auto-center viewport
            self._center_viewport(timeline_w)

            view_end = self.view_start + self.ns_per_col * timeline_w

            # Cursor column
            cursor_col = int((self.cursor_ns - self.view_start) / max(self.ns_per_col, 1))

            # ── Header ──
            mode_label = "LOGICAL" if self.logical_mode else "TIME"
            sel_k = self._kernel_at_time(stream, self.cursor_ns)
            k_info = f"  [{_short_kernel_name(sel_k.name)} {_fmt_dur(sel_k.duration_ms)}]" if sel_k else ""
            header = f" {self.title}  S{stream}{k_info}  [{mode_label}]"
            if self.filter_text:
                header += f"  /{self.filter_text}"
            if self.min_dur_us > 0:
                header += f"  ≥{self.min_dur_us}μs"
            if self.bookmarks:
                header += f"  📌{len(self.bookmarks)}"
            stdscr.addnstr(0, 0, header, width - 1, curses.A_BOLD)

            # ── Time axis (row 1) ──
            self._draw_time_axis(stdscr, 1, label_w, timeline_w, width)

            # ── NVTX rows (fixed height) ──
            nvtx_y = 2
            if not self.logical_mode and self.ns_per_col > 0:
                for depth in range(self.nvtx_max_depth):
                    row_y = nvtx_y + depth
                    if row_y >= height - 6:
                        break
                    spans = [s for s in self.nvtx_spans
                             if s.depth == depth and s.end_ns > self.view_start
                             and s.start_ns < view_end]

                    dlabel = f"N{depth}".ljust(label_w - 1)
                    try:
                        stdscr.addnstr(row_y, 0, dlabel, label_w - 1,
                                       curses.A_DIM | curses.color_pair(5))
                    except curses.error:
                        pass

                    for span in spans:
                        s_col = max(0, int((span.start_ns - self.view_start) / self.ns_per_col))
                        e_col = min(timeline_w - 1, int((span.end_ns - self.view_start) / self.ns_per_col))
                        span_w = e_col - s_col + 1
                        if span_w < 1:
                            continue

                        dur_ms = (span.end_ns - span.start_ns) / 1e6
                        time_suffix = f" {_fmt_dur(dur_ms)}"
                        name = span.name

                        if span_w >= len(name) + len(time_suffix) + 3:
                            fill = span_w - len(name) - len(time_suffix) - 2
                            content = f"[{name}{'─' * fill}{time_suffix}]"
                        elif span_w >= len(name) + 2:
                            content = f"[{name}{'─' * max(0, span_w - len(name) - 2)}]"
                        elif span_w >= 3:
                            content = f"[{name[:span_w - 2]}]"
                        else:
                            content = '█' * span_w

                        x = label_w + s_col
                        try:
                            stdscr.addnstr(row_y, x, content[:span_w], span_w,
                                           curses.color_pair(5))
                        except curses.error:
                            pass

            # ── Separator + streams ──
            sep_y = nvtx_y + self.nvtx_max_depth
            try:
                stdscr.addnstr(sep_y, 0, '─' * width, width - 1, curses.A_DIM)
            except curses.error:
                pass
            swim_y = sep_y + 1

            # Stream rows — variable height per stream
            available = height - swim_y - 6
            # Calculate total rows needed
            stream_y_map: list[tuple[int, int, int]] = []  # (start_y, row_h, stream_idx)
            cur_y = swim_y
            for si in range(len(self.streams)):
                rh = self.selected_stream_rows if si == self.selected_stream else self.default_stream_rows
                if cur_y + rh > swim_y + available:
                    break
                stream_y_map.append((cur_y, rh, si))
                cur_y += rh
            streams_show_count = len(stream_y_map)

            for start_y_s, row_h, si in stream_y_map:
                s = self.streams[si]
                is_sel = (si == self.selected_stream)
                ci = self.stream_color_idx.get(s, 0)
                s_kernels = self._get_stream_kernels(s)

                # Stream label
                label = f"S{s}".ljust(label_w - 1)
                try:
                    stdscr.addnstr(start_y_s, 0, label, label_w - 1,
                                   (curses.A_BOLD if is_sel else curses.A_DIM) | curses.color_pair(11 + ci))
                except curses.error:
                    pass

                # Draw kernel blocks: name row(s) on top, block row at bottom
                block_y = start_y_s + row_h - 1  # last row = block bar
                name_y = start_y_s  # first row = name/dur labels
                self._draw_stream_row(stdscr, name_y, block_y, label_w, timeline_w,
                                     s_kernels, is_sel, ci, width)

            # ── Cursor line through all stream rows ──
            total_stream_h = sum(rh for _, rh, _ in stream_y_map)
            if 0 <= cursor_col < timeline_w:
                cursor_x = label_w + cursor_col
                for row in range(swim_y, swim_y + total_stream_h):
                    if row >= height - 5:
                        break
                    try:
                        stdscr.addstr(row, cursor_x, '│',
                                     curses.color_pair(8) | curses.A_BOLD)
                    except curses.error:
                        pass

            # ── Bottom panel ──
            panel_y = swim_y + total_stream_h + 1
            self._draw_bottom_panel(stdscr, panel_y, width, height, stream, sel_k)

            # ── Config panel ──
            if self.show_config:
                self._draw_config_panel(stdscr, height, width)

            # ── Help overlay ──
            if self.show_help:
                self._draw_help(stdscr, height, width)

            # ── Bookmark list overlay ──
            if self.bookmark_list_mode and self.bookmarks:
                bm_lines = ["─── Bookmarks (1-9 to jump, Esc to cancel) ───"]
                for bi, bm in enumerate(self.bookmarks[:9]):
                    num = bi + 1
                    name = bm['name']
                    ts = _fmt_ns(bm['cursor_ns'])
                    extra = ""
                    if 'kernel_name' in bm:
                        extra = f"  [{_short_kernel_name(bm['kernel_name'])}]"
                    if 'range_start_ns' in bm:
                        extra += f"  ↔ range"
                    marker = " ◀" if bi == self.bookmark_idx else ""
                    bm_lines.append(f"  {num}  {name}  {ts}{extra}{marker}")
                start_y = max(height - len(bm_lines) - 2, 3)
                for i, line in enumerate(bm_lines):
                    y = start_y + i
                    if y >= height - 1:
                        break
                    try:
                        stdscr.addnstr(y, 0, line, width - 1,
                                       curses.color_pair(9) if i == 0 else curses.A_NORMAL)
                    except curses.error:
                        pass

            # ── Input prompts ──
            if self.filter_mode:
                try:
                    stdscr.addnstr(height - 2, 0, f" Filter: {self.filter_input}█",
                                   width - 1, curses.A_BOLD | curses.color_pair(7))
                except curses.error:
                    pass
            elif self.threshold_mode:
                try:
                    stdscr.addnstr(height - 2, 0, f" Min (μs): {self.threshold_input}█",
                                   width - 1, curses.A_BOLD | curses.color_pair(8))
                except curses.error:
                    pass
            elif self.bookmark_mode:
                try:
                    stdscr.addnstr(height - 2, 0, f" Bookmark name: {self.bookmark_input}█",
                                   width - 1, curses.A_BOLD | curses.color_pair(9))
                except curses.error:
                    pass
            elif self.status_msg:
                try:
                    stdscr.addnstr(height - 2, 0, f" {self.status_msg}",
                                   width - 1, curses.color_pair(8))
                except curses.error:
                    pass
                self.status_msg = ''

            # Help line
            help_short = " ←→:time ↑↓:stream Tab:kernel +/-:zoom B:bookmark a:axis h:help q:quit"
            try:
                stdscr.addnstr(height - 1, 0, help_short[:width - 1], width - 1, curses.A_DIM)
            except curses.error:
                pass

            stdscr.refresh()

            # ── Input ──
            key = stdscr.getch()

            if self.filter_mode:
                if key == 27:
                    self.filter_mode = False
                elif key in (10, 13):
                    self.filter_text = self.filter_input
                    self.filter_mode = False
                elif key in (8, 127, 263):
                    self.filter_input = self.filter_input[:-1]
                elif 32 <= key <= 126:
                    self.filter_input += chr(key)
                continue

            if self.threshold_mode:
                if key == 27:
                    self.threshold_mode = False
                elif key in (10, 13):
                    try:
                        self.min_dur_us = int(self.threshold_input) if self.threshold_input else 0
                    except ValueError:
                        self.min_dur_us = 0
                    self.threshold_mode = False
                    self.threshold_input = ''
                elif key in (8, 127, 263):
                    self.threshold_input = self.threshold_input[:-1]
                elif ord('0') <= key <= ord('9'):
                    self.threshold_input += chr(key)
                continue

            if self.bookmark_mode:
                if key == 27:
                    self.bookmark_mode = False
                elif key in (10, 13):
                    name = self.bookmark_input or f"#{len(self.bookmarks) + 1}"
                    bm: dict = {'name': name, 'cursor_ns': self.cursor_ns,
                           'stream': self.selected_stream}
                    # Store kernel context
                    sel_k = self._kernel_at_time(stream, self.cursor_ns)
                    if sel_k:
                        bm['kernel_name'] = sel_k.name
                        bm['nvtx_path'] = sel_k.nvtx_path
                        # Kernel sequence index on its stream
                        ki = self._kernel_index_at_time(stream, self.cursor_ns)
                        bm['kernel_seq'] = ki
                    if self.range_start_ns is not None:
                        bm['range_start_ns'] = self.range_start_ns
                        bm['range_end_ns'] = self.cursor_ns
                        self.range_start_ns = None
                    self.bookmarks.append(bm)
                    self.bookmark_idx = len(self.bookmarks) - 1
                    self.bookmark_mode = False
                    self.bookmark_input = ''
                    self.status_msg = f"📌 Saved: {name}"
                elif key in (8, 127, 263):
                    self.bookmark_input = self.bookmark_input[:-1]
                elif 32 <= key <= 126:
                    self.bookmark_input += chr(key)
                continue

            # ── Bookmark list mode ──
            if self.bookmark_list_mode:
                if key == 27:
                    self.bookmark_list_mode = False
                elif ord('1') <= key <= ord('9'):
                    idx = key - ord('1')
                    if idx < len(self.bookmarks):
                        self.bookmark_idx = idx
                        self._jump_to_bookmark(self.bookmarks[idx], timeline_w)
                    self.bookmark_list_mode = False
                elif key == ord('0') and len(self.bookmarks) >= 10:
                    self.bookmark_idx = 9
                    self._jump_to_bookmark(self.bookmarks[9], timeline_w)
                    self.bookmark_list_mode = False
                continue

            # ── Config panel mode ──
            if self.show_config:
                if key == 27 or key == ord('C'):
                    self.show_config = False
                    continue
                elif key in (curses.KEY_UP, ord('k')):
                    self.config_cursor = max(0, self.config_cursor - 1)
                    continue
                elif key in (curses.KEY_DOWN, ord('j')):
                    self.config_cursor = min(len(self.config_items) - 1, self.config_cursor + 1)
                    continue
                elif key in (curses.KEY_RIGHT, ord('l'), ord('+'), ord('=')):
                    self._adjust_config(1)
                    continue
                elif key in (curses.KEY_LEFT, ord('h'), ord('-'), ord('_')):
                    self._adjust_config(-1)
                    continue
                elif key == ord('q'):
                    break
                continue

            # ── Navigation ──
            step = self.ns_per_col  # 1 column

            if key == ord('q'):
                break
            elif key in (curses.KEY_LEFT,):
                self.cursor_ns -= step
                self.cursor_ns = max(self.time_start, self.cursor_ns)
            elif key in (curses.KEY_RIGHT,):
                self.cursor_ns += step
                self.cursor_ns = min(self.time_end, self.cursor_ns)
            elif key == curses.KEY_SLEFT or key == 393:  # Shift+Left
                self.cursor_ns -= step * (timeline_w // 4)
                self.cursor_ns = max(self.time_start, self.cursor_ns)
            elif key == curses.KEY_SRIGHT or key == 402:  # Shift+Right
                self.cursor_ns += step * (timeline_w // 4)
                self.cursor_ns = min(self.time_end, self.cursor_ns)
            elif key in (curses.KEY_UP, ord('k')):
                self.selected_stream = max(0, self.selected_stream - 1)
            elif key in (curses.KEY_DOWN, ord('j')):
                self.selected_stream = min(len(self.streams) - 1, self.selected_stream + 1)
            elif key == 9:  # Tab: next kernel
                ki = self._kernel_index_at_time(stream, self.cursor_ns)
                if ki >= 0:
                    ks = self._get_stream_kernels(stream)
                    if ki + 1 < len(ks):
                        self.cursor_ns = ks[ki + 1].start_ns
                    elif ks:
                        self.cursor_ns = ks[-1].start_ns
            elif key == 353:  # Shift+Tab: prev kernel
                ki = self._kernel_index_at_time(stream, self.cursor_ns)
                if ki > 0:
                    ks = self._get_stream_kernels(stream)
                    self.cursor_ns = ks[ki - 1].start_ns
            elif key in (ord('+'), ord('=')):
                self.ns_per_col = max(1, self.ns_per_col * 2 // 3)
                self.status_msg = f'Zoom: {_fmt_dur(self.ns_per_col / 1e6)}/col'
            elif key in (ord('-'), ord('_')):
                self.ns_per_col = min(self.time_span, self.ns_per_col * 3 // 2)
                self.status_msg = f'Zoom: {_fmt_dur(self.ns_per_col / 1e6)}/col'
            elif key == ord('a'):
                self.relative_time = not self.relative_time
                self.status_msg = f"Time: {'relative' if self.relative_time else 'absolute'}"
            elif key == ord('L'):
                self.logical_mode = not self.logical_mode
            elif key == ord('d'):
                self.show_demangled = not self.show_demangled
            elif key == ord('/'):
                self.filter_mode = True
                self.filter_input = self.filter_text
            elif key == ord('n'):
                self.filter_text = ''
            elif key == ord('m'):
                self.threshold_mode = True
                self.threshold_input = str(self.min_dur_us) if self.min_dur_us else ''
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif key == ord('B'):
                self.bookmark_mode = True
                self.bookmark_input = ''
            elif key == ord("'"):
                if self.bookmarks:
                    self.bookmark_list_mode = True
                else:
                    self.status_msg = 'No bookmarks saved'
            elif key == ord('T'):
                # Cycle tick density: 3 → 6 → 10 → 15 → 3
                densities = [3, 6, 10, 15]
                try:
                    ni = (densities.index(self.tick_density) + 1) % len(densities)
                except ValueError:
                    ni = 0
                self.tick_density = densities[ni]
                self.status_msg = f'Tick density: {self.tick_density}'
            elif key == ord(',') and self.bookmarks:
                self._save_prev_position()
                self.bookmark_idx = (self.bookmark_idx - 1) % len(self.bookmarks)
                self._jump_to_bookmark(self.bookmarks[self.bookmark_idx], timeline_w)
            elif key == ord('.') and self.bookmarks:
                self._save_prev_position()
                self.bookmark_idx = (self.bookmark_idx + 1) % len(self.bookmarks)
                self._jump_to_bookmark(self.bookmarks[self.bookmark_idx], timeline_w)
            elif key == ord('`'):
                # Jump back to previous position
                if self.prev_position:
                    old = self.prev_position
                    self._save_prev_position()  # so you can toggle back
                    self.cursor_ns = old['cursor_ns']
                    self.selected_stream = old['stream']
                    self.status_msg = 'Jumped back'
                else:
                    self.status_msg = 'No previous position'
            elif key == ord('C'):
                self.show_config = not self.show_config
            elif key == ord('['):
                self.range_start_ns = self.cursor_ns
                self.status_msg = f"Range start: {_fmt_ns(self.cursor_ns)}"
            elif key == ord(']'):
                if self.range_start_ns is not None:
                    self.bookmark_mode = True
                    self.bookmark_input = ''
                    self.status_msg = f"Range: {_fmt_ns(self.range_start_ns)}→{_fmt_ns(self.cursor_ns)}"
            elif key == curses.KEY_HOME:
                self.cursor_ns = self.time_start
            elif key == curses.KEY_END:
                self.cursor_ns = self.time_end
            elif key == curses.KEY_PPAGE:
                self.cursor_ns -= step * (timeline_w // 4)
                self.cursor_ns = max(self.time_start, self.cursor_ns)
            elif key == curses.KEY_NPAGE:
                self.cursor_ns += step * (timeline_w // 4)
                self.cursor_ns = min(self.time_end, self.cursor_ns)

    def _save_prev_position(self):
        self.prev_position = {
            'cursor_ns': self.cursor_ns,
            'stream': self.selected_stream,
        }

    def _jump_to_bookmark(self, bm: dict, timeline_w: int):
        self.cursor_ns = bm['cursor_ns']
        self.selected_stream = bm.get('stream', 0)
        if 'range_start_ns' in bm and 'range_end_ns' in bm:
            span = bm['range_end_ns'] - bm['range_start_ns']
            self.ns_per_col = max(1, span // timeline_w)
            self.cursor_ns = bm['range_start_ns']
        self.status_msg = f"📌 {bm['name']}"

    def _draw_time_axis(self, stdscr, y: int, label_w: int, timeline_w: int, width: int):
        """Draw time axis with absolute or relative markers."""
        if self.ns_per_col <= 0:
            return

        axis = [' '] * timeline_w
        tick_interval = self._nice_tick_interval(timeline_w)
        view_end = self.view_start + self.ns_per_col * timeline_w

        if self.relative_time:
            # Show absolute on left, then relative +offset
            origin = self.view_start
            left_label = _fmt_ns(origin)
            for ci, ch in enumerate(left_label):
                if ci < timeline_w:
                    axis[ci] = ch

            first_tick = ((self.view_start // tick_interval) + 1) * tick_interval
            t = first_tick
            while t < view_end:
                col = int((t - self.view_start) / self.ns_per_col)
                if 0 <= col < timeline_w:
                    label = _fmt_relative(t - origin)
                    for ci, ch in enumerate(label):
                        if col + ci < timeline_w:
                            axis[col + ci] = ch
                t += tick_interval
        else:
            first_tick = ((self.view_start // tick_interval) + 1) * tick_interval
            t = first_tick
            while t < view_end:
                col = int((t - self.view_start) / self.ns_per_col)
                if 0 <= col < timeline_w:
                    label = _fmt_ns(t)
                    for ci, ch in enumerate(label):
                        if col + ci < timeline_w:
                            axis[col + ci] = ch
                t += tick_interval

        line = ' ' * label_w + ''.join(axis)
        try:
            stdscr.addnstr(y, 0, line, width - 1, curses.A_DIM)
        except curses.error:
            pass

    def _draw_stream_row(self, stdscr, y1: int, y2: int, label_w: int,
                         timeline_w: int, kernels: list[KernelEvent],
                         is_selected: bool, ci: int, screen_w: int):
        """Draw 2-row stream: y1=name+dur labels, y2=kernel blocks."""
        if not kernels or self.ns_per_col <= 0:
            return

        for ki, k in enumerate(kernels):
            s_col = int((k.start_ns - self.view_start) / self.ns_per_col)
            e_col = int((k.end_ns - self.view_start) / self.ns_per_col)
            if e_col < 0 or s_col >= timeline_w:
                continue
            s_col = max(0, s_col)
            e_col = min(timeline_w - 1, max(s_col, e_col))
            block_w = e_col - s_col + 1

            is_at_cursor = (is_selected and
                           k.start_ns <= self.cursor_ns <= k.end_ns)

            # Colors
            if is_at_cursor:
                block_attr = curses.color_pair(21 + ci) | curses.A_BOLD
                name_attr = curses.color_pair(21 + ci) | curses.A_BOLD
            else:
                block_attr = curses.color_pair(11 + ci)
                name_attr = curses.color_pair(11 + ci) | curses.A_DIM
                if k.heat > 0.7:
                    block_attr |= curses.A_BOLD
                elif k.heat < 0.2:
                    block_attr |= curses.A_DIM

            x = label_w + s_col
            if x >= screen_w:
                continue

            # Row 1 (y1): name + duration label
            short = _short_kernel_name(self._kernel_name(k))
            dur = _fmt_dur(k.duration_ms)
            if block_w >= len(short) + len(dur) + 2:
                label = f"{short} {dur}"
            elif block_w >= len(short):
                label = short[:block_w]
            elif block_w >= len(dur):
                label = dur[:block_w]
            elif block_w >= 2:
                label = short[:block_w]
            else:
                label = ''

            if label:
                try:
                    stdscr.addnstr(y1, x, label, min(block_w, screen_w - x), name_attr)
                except curses.error:
                    pass

            # Row 2 (y2): solid block (always █, NCCL gets magenta)
            if k.is_nccl:
                nccl_attr = curses.color_pair(11 + 3)  # magenta = index 3
                if is_at_cursor:
                    nccl_attr = curses.color_pair(21 + 3) | curses.A_BOLD
                block_content = '█' * block_w
                try:
                    stdscr.addnstr(y2, x, block_content, min(block_w, screen_w - x), nccl_attr)
                except curses.error:
                    pass
            else:
                block_content = '█' * block_w
                try:
                    stdscr.addnstr(y2, x, block_content, min(block_w, screen_w - x), block_attr)
                except curses.error:
                    pass

    def _draw_bottom_panel(self, stdscr, panel_y: int, width: int, height: int,
                           stream: str, sel_k: Optional[KernelEvent]):
        """Draw detail bar + NVTX hierarchy in bottom panel."""
        if panel_y >= height - 2:
            return

        if sel_k:
            ci = self.stream_color_idx.get(sel_k.stream, 0)

            # Detail line
            time_col = f"{_fmt_ns(sel_k.start_ns)}→{_fmt_ns(sel_k.end_ns)}".ljust(22)
            dur_col = f"{_fmt_dur(sel_k.duration_ms)} [S{sel_k.stream}]".ljust(18)
            name = self._kernel_name(sel_k)
            detail = f" {time_col} │ {dur_col} │ {name}"
            try:
                stdscr.addnstr(panel_y, 0, detail[:width - 1], width - 1,
                               curses.A_BOLD | curses.color_pair(11 + ci))
            except curses.error:
                pass

            # NVTX hierarchy
            if sel_k.nvtx_path and panel_y + 2 < height - 1:
                parts = sel_k.nvtx_path.split(' > ')
                y = panel_y + 1
                try:
                    stdscr.addnstr(y, 0, '─' * min(50, width), width - 1, curses.A_DIM)
                except curses.error:
                    pass
                y += 1
                for pi, part in enumerate(parts):
                    if y >= height - 1:
                        break
                    indent = '  ' * pi
                    pfx = '└─ 📁 ' if pi < len(parts) - 1 else '└─ ▸ '
                    line = f"{indent}{pfx}{part}"
                    attr = curses.color_pair(5) if pi < len(parts) - 1 else curses.color_pair(11 + ci)
                    try:
                        stdscr.addnstr(y, 0, line[:width - 1], width - 1, attr)
                    except curses.error:
                        pass
                    y += 1
                # Show the kernel name at the leaf
                if y < height - 1:
                    indent = '  ' * len(parts)
                    k_line = f"{indent}▶ {name}  {_fmt_dur(sel_k.duration_ms)}"
                    try:
                        stdscr.addnstr(y, 0, k_line[:width - 1], width - 1,
                                       curses.A_BOLD | curses.color_pair(11 + ci))
                    except curses.error:
                        pass
        else:
            # No kernel at cursor — show cursor time
            try:
                stdscr.addnstr(panel_y, 0,
                               f" Cursor: {_fmt_ns(self.cursor_ns)}  (no kernel on S{stream})",
                               width - 1, curses.A_DIM)
            except curses.error:
                pass

    def _draw_help(self, stdscr, height: int, width: int):
        """Draw help overlay at bottom."""
        help_lines = [
            "─── Keybindings ───",
            "←/→         Pan through time (1 column)",
            "Shift+←/→   Page pan (1/4 viewport)",
            "PgUp/PgDn   Page pan",
            "↑/↓         Select stream",
            "Tab         Next kernel on stream",
            "Shift+Tab   Previous kernel",
            "+/-         Zoom in/out",
            "a           Toggle absolute/relative time",
            "T           Cycle time tick density",
            "/           Filter by name  (n: clear)",
            "m           Set min duration (μs)",
            "d           Toggle demangled names",
            "B           Save bookmark",
            "'           Show bookmarks (1-9 to jump)",
            ",/.         Cycle bookmarks",
            "`           Jump back to previous position",
            "[           Set range start  ] Save range",
            "C           Config panel (↑↓ select, ←→ adjust)",
            "Home/End    Jump to start/end",
            "h           Toggle this help",
            "q           Quit",
        ]
        start_y = max(height - len(help_lines) - 1, 3)
        for i, line in enumerate(help_lines):
            y = start_y + i
            if y >= height - 1:
                break
            try:
                stdscr.addnstr(y, 0, f"  {line}", width - 1,
                               curses.color_pair(8) if i == 0 else curses.A_NORMAL)
            except curses.error:
                pass

    def _draw_config_panel(self, stdscr, height: int, width: int):
        """Draw config panel on the right side."""
        panel_w = 40
        panel_x = max(width - panel_w - 2, 0)
        panel_y = 3

        labels = {
            'selected_stream_rows': 'Selected stream rows',
            'default_stream_rows': 'Other stream rows',
            'tick_density': 'Time tick density',
            'nvtx_max_depth': 'NVTX depth levels',
            'min_dur_us': 'Min kernel dur (μs)',
        }

        header = "─── Config (C or Esc to close) ───"
        try:
            stdscr.addnstr(panel_y, panel_x, header, panel_w, curses.color_pair(8) | curses.A_BOLD)
        except curses.error:
            pass

        for i, item in enumerate(self.config_items):
            y = panel_y + 1 + i
            if y >= height - 2:
                break
            val = getattr(self, item, 0)
            label = labels.get(item, item)
            is_sel = (i == self.config_cursor)

            arrow = '▶ ' if is_sel else '  '
            line = f"{arrow}{label}: ◀ {val} ▶"
            attr = curses.A_BOLD | curses.color_pair(8) if is_sel else curses.A_NORMAL
            try:
                stdscr.addnstr(y, panel_x, line.ljust(panel_w), panel_w, attr)
            except curses.error:
                pass

        hint_y = panel_y + len(self.config_items) + 1
        if hint_y < height - 1:
            try:
                stdscr.addnstr(hint_y, panel_x, "  ↑↓ select  ←→ adjust",
                               panel_w, curses.A_DIM)
            except curses.error:
                pass

    def _adjust_config(self, delta: int):
        """Adjust config item by delta."""
        item = self.config_items[self.config_cursor]
        val = getattr(self, item, 0)
        if item == 'selected_stream_rows':
            self.selected_stream_rows = max(1, min(6, val + delta))
        elif item == 'default_stream_rows':
            self.default_stream_rows = max(1, min(4, val + delta))
        elif item == 'tick_density':
            self.tick_density = max(2, min(20, val + delta))
        elif item == 'nvtx_max_depth':
            self.nvtx_max_depth = max(0, min(8, val + delta))
        elif item == 'min_dur_us':
            self.min_dur_us = max(0, val + delta * 10)

    def _nice_tick_interval(self, timeline_w: int) -> int:
        viewport_ns = self.ns_per_col * timeline_w
        raw = viewport_ns // max(self.tick_density, 1)
        for nice in [1_000, 5_000, 10_000, 50_000, 100_000, 500_000,
                     1_000_000, 5_000_000, 10_000_000, 50_000_000,
                     100_000_000, 500_000_000, 1_000_000_000]:
            if nice >= raw:
                return nice
        return max(raw, 1)


def run_timeline(db_path: str, device: int, trim: tuple[int, int],
                 max_depth: int = -1, min_ms: float = 0):
    """Entry point."""
    from . import profile as _profile
    from .tree import build_nvtx_tree, to_json

    prof = _profile.open(db_path)
    roots = build_nvtx_tree(prof, device, trim)
    json_roots = to_json(roots)

    gpu_name = f"GPU {device}"
    try:
        gpus = prof.gpus()
        for g in gpus:
            if g.get("id") == device or g.get("deviceId") == device:
                gpu_name = g.get("name", gpu_name)
                break
    except Exception:
        pass

    trim_label = f"{trim[0] / 1e9:.1f}s – {trim[1] / 1e9:.1f}s"
    title = f"{gpu_name}  |  {trim_label}"

    if not os.isatty(1):
        print(f"{title}")
        all_k: list[KernelEvent] = []
        _collect_kernels(json_roots, all_k)
        print(f"Kernels: {len(all_k)}")
        streams = sorted(set(k.stream for k in all_k))
        for s in streams:
            count = sum(1 for k in all_k if k.stream == s)
            print(f"  Stream {s}: {count} kernels")
        return

    tui = TimelineTUI(json_roots, title=title, db_path=db_path,
                      device=device, trim=trim)
    if min_ms > 0:
        tui.min_dur_us = int(min_ms * 1000)
    curses.wrapper(tui.run)


def _collect_kernels(roots: list[dict], out: list, path: str = ''):
    for n in roots:
        np = f"{path} > {n['name']}" if path else n['name']
        if n.get('type') == 'kernel':
            out.append(KernelEvent(n, path))
        if n.get('children'):
            _collect_kernels(n['children'], out, np)
