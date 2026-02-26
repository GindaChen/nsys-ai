"""
tui.py — Interactive Terminal UI for NVTX tree view.

Renders the NVTX tree in an interactive curses-based terminal UI.
Designed for use over SSH on remote servers where opening a browser is
not practical.

Usage:
    PYTHONPATH=lib/python python -m nsight tui <profile.sqlite> --gpu N --trim S E

Keybindings:
    ↑/↓ or j/k     Navigate
    Enter/Space     Toggle expand/collapse NVTX scope
    ← / h           Collapse current scope (or go to parent)
    → / l           Expand current scope
    e               Expand all
    c               Collapse all
    /               Start search filter (Enter to apply, Esc to cancel)
    F               Toggle dynamic filter (live update as you type)
    n               Clear filter
    1-5             Set max depth
    0               Unlimited depth
    d               Toggle demangled kernel names
    t               Toggle start/end timestamps
    q               Quit
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
    """Format nanoseconds as seconds with 3 decimal places."""
    if ns is None:
        return "?"
    return f"{ns / 1e9:.3f}s"


class TreeNode:
    """Flat representation of a tree node for display."""
    __slots__ = ('name', 'type', 'depth', 'duration_ms', 'heat', 'stream',
                 'relative_pct', 'path', 'demangled', 'children_count',
                 'kernel_count', 'nvtx_count', 'expanded', 'has_children',
                 'start_ns', 'end_ns', 'json_node', '_bubble_us')

    def __init__(self, json_node: dict, depth: int):
        self.json_node = json_node
        self.name = json_node.get('name', '?')
        self.type = json_node.get('type', '')
        self.depth = depth
        self.duration_ms = json_node.get('duration_ms', 0)
        self.heat = json_node.get('heat', 0)
        self.stream = json_node.get('stream', '?')
        self.relative_pct = json_node.get('relative_pct', 100)
        self.path = json_node.get('path', '')
        self.demangled = json_node.get('demangled', '')
        self.start_ns = json_node.get('start_ns')
        self.end_ns = json_node.get('end_ns')
        children = json_node.get('children', [])
        self.has_children = len(children) > 0
        self.kernel_count = sum(1 for c in children if c.get('type') == 'kernel')
        self.nvtx_count = sum(1 for c in children if c.get('type') == 'nvtx')
        self.children_count = len(children)
        self.expanded = depth < 2  # Default: expand top 2 levels
        self._bubble_us = 0


class InteractiveTUI:
    """Interactive curses-based tree viewer."""

    def __init__(self, json_roots: list[dict], title: str = "NVTX Tree",
                 db_path: str = '', device: int = 0,
                 trim: tuple = (0, 0)):
        self.json_roots = json_roots
        self.title = title
        self.db_path = db_path
        self.device = device
        self.trim = trim
        self.cursor = 0
        self.scroll_offset = 0
        self.filter_text = ''
        self.filter_mode = False
        self.filter_input = ''
        self.dynamic_filter = False   # live update filter as you type
        self.view_mode = 'tree'       # 'tree' or 'linear'
        self.max_depth = -1           # -1 = unlimited
        self.min_dur_us = 0           # min duration threshold in μs (0 = show all)
        self.show_demangled = False
        self.show_times = False       # toggle start/end timestamps
        self.show_bubbles = False     # show gaps between consecutive kernels
        self.bubble_threshold_us = 10 # only show bubbles >= this (μs)
        self.status_msg = ''
        self.threshold_mode = False   # entering threshold value
        self.threshold_input = ''
        self.threshold_target = 'min_dur'  # 'min_dur' or 'bubble'

        # Selection and bookmarks
        self.selection_anchor = -1    # -1 = no selection; >= 0 = anchor row
        self.bookmarks = []           # list of {name, start_ns, end_ns, cursor_path}
        self.show_bookmarks = False   # toggle right panel

        # Build all nodes in DFS order — this is the source of truth
        self.all_nodes: list[TreeNode] = []
        self._build_nodes(json_roots, 0)

        # Compute summary
        self.total_kernels = 0
        self.total_gpu_ms = 0.0
        self.total_nvtx = 0
        self._compute_summary(json_roots)

    def _build_nodes(self, nodes: list[dict], depth: int):
        """Build flat node list from JSON tree in DFS order."""
        for n in nodes:
            self.all_nodes.append(TreeNode(n, depth))
            if n.get('children'):
                self._build_nodes(n['children'], depth + 1)

    def _compute_summary(self, nodes: list[dict]):
        for n in nodes:
            if n['type'] == 'nvtx':
                self.total_nvtx += 1
            elif n['type'] == 'kernel':
                self.total_kernels += 1
                self.total_gpu_ms += n.get('duration_ms', 0)
            if n.get('children'):
                self._compute_summary(n['children'])

    def _visible_rows(self) -> list[TreeNode]:
        """Return visible nodes based on current view mode."""
        if self.view_mode == 'linear':
            return self._visible_rows_linear()
        return self._visible_rows_tree()

    def _visible_rows_linear(self) -> list[TreeNode]:
        """Flat list of all nodes sorted by start time."""
        ft = self.filter_text.lower() if self.filter_text else ''
        nodes = []
        for node in self.all_nodes:
            # Text filter
            if ft and ft not in node.name.lower():
                if not (node.demangled and ft in node.demangled.lower()):
                    continue
            # Duration filter
            if self.min_dur_us > 0 and node.type == 'kernel':
                if node.duration_ms < self.min_dur_us / 1000.0:
                    continue
            nodes.append(node)

        # Sort by start time
        nodes.sort(key=lambda n: n.start_ns if n.start_ns is not None else 0)

        # Compute bubbles if enabled
        if self.show_bubbles:
            last_end_by_stream = {}
            for node in nodes:
                if node.type == 'kernel' and node.end_ns is not None:
                    sid = str(node.stream)
                    if sid in last_end_by_stream and node.start_ns is not None:
                        gap_ns = node.start_ns - last_end_by_stream[sid]
                        gap_us = gap_ns / 1000.0
                        node._bubble_us = gap_us if gap_us >= self.bubble_threshold_us else 0
                    else:
                        node._bubble_us = 0
                    last_end_by_stream[sid] = node.end_ns
                else:
                    node._bubble_us = 0
        else:
            for node in nodes:
                node._bubble_us = 0

        return nodes

    def _visible_rows_tree(self) -> list[TreeNode]:
        """
        Walk all_nodes (DFS order) and return the visible subset.

        A node is visible if:
        1. All its NVTX ancestors are expanded
        2. It passes the depth filter
        3. It passes the text filter (or a descendant does)

        We track visibility via a depth stack: when we see an NVTX node
        that is collapsed or filtered out, we skip all subsequent nodes
        with greater depth until we return to the same or lesser depth.
        """
        visible = []
        # skip_below_depth: if >= 0, skip nodes deeper than this
        skip_below_depth = -1
        ft = self.filter_text.lower() if self.filter_text else ''

        for node in self.all_nodes:
            # If we're skipping children of a collapsed/hidden parent
            if skip_below_depth >= 0 and node.depth > skip_below_depth:
                continue

            # Reset skip once we're back to the same or lesser depth
            skip_below_depth = -1

            # Depth filter
            if self.max_depth >= 0 and node.depth > self.max_depth:
                skip_below_depth = node.depth - 1
                continue

            # Text filter: if filter is active, check if this node or descendants match
            if ft:
                if not self._node_matches_filter(node, ft):
                    # Don't skip children — they might match
                    continue

            visible.append(node)

            # If this is a collapsed NVTX, skip its children
            if node.type == 'nvtx' and not node.expanded:
                skip_below_depth = node.depth

        # Apply min duration filter (remove kernels below threshold, keep NVTX)
        if self.min_dur_us > 0:
            min_ms = self.min_dur_us / 1000.0
            visible = [n for n in visible
                       if n.type != 'kernel' or n.duration_ms >= min_ms]

        # Compute bubble gaps between consecutive kernels per stream
        if self.show_bubbles:
            last_end_by_stream = {}  # stream -> end_ns
            for node in visible:
                if node.type == 'kernel' and node.end_ns is not None:
                    sid = str(node.stream)
                    if sid in last_end_by_stream and node.start_ns is not None:
                        gap_ns = node.start_ns - last_end_by_stream[sid]
                        gap_us = gap_ns / 1000.0
                        node._bubble_us = gap_us if gap_us >= self.bubble_threshold_us else 0
                    else:
                        node._bubble_us = 0
                    last_end_by_stream[sid] = node.end_ns
                else:
                    node._bubble_us = 0
        else:
            for node in visible:
                node._bubble_us = 0

        return visible

    def _node_matches_filter(self, node: TreeNode, ft: str) -> bool:
        """Check if node name (or demangled) matches filter text."""
        if ft in node.name.lower():
            return True
        if node.demangled and ft in node.demangled.lower():
            return True
        # Also search descendants via json_node
        return self._json_descendant_matches(node.json_node, ft)

    def _json_descendant_matches(self, json_node: dict, ft: str) -> bool:
        """Check if any descendant of json_node matches filter."""
        name = json_node.get('name', '').lower()
        demangled = json_node.get('demangled', '').lower()
        if ft in name or ft in demangled:
            return True
        for c in json_node.get('children', []):
            if self._json_descendant_matches(c, ft):
                return True
        return False

    def _expand_all(self):
        for n in self.all_nodes:
            if n.type == 'nvtx':
                n.expanded = True
        self.status_msg = 'Expanded all'

    def _collapse_all(self):
        for n in self.all_nodes:
            if n.type == 'nvtx':
                n.expanded = False
        self.status_msg = 'Collapsed all'

    def _reload(self):
        """Reload profile from disk and rebuild tree."""
        from . import profile as _profile
        from .tree import build_nvtx_tree, to_json
        try:
            prof = _profile.open(self.db_path)
            roots = build_nvtx_tree(prof, self.device, self.trim)
            self.json_roots = to_json(roots)
            self.all_nodes = []
            self._build_nodes(self.json_roots, 0)
            self.total_kernels = 0
            self.total_gpu_ms = 0.0
            self.total_nvtx = 0
            self._compute_summary(self.json_roots)
            trim_label = f"{self.trim[0] / 1e9:.1f}s \u2013 {self.trim[1] / 1e9:.1f}s"
            self.title = f"GPU {self.device}  |  {trim_label}"
            self.cursor = 0
            self.scroll_offset = 0
            self.status_msg = 'Reloaded'
        except Exception as e:
            self.status_msg = f'Reload failed: {e}'

    def _find_parent(self, visible: list, idx: int) -> int:
        """Find the parent NVTX node index in visible list."""
        if idx < 0 or idx >= len(visible):
            return idx
        target_depth = visible[idx].depth
        for i in range(idx - 1, -1, -1):
            if visible[i].depth < target_depth and visible[i].type == 'nvtx':
                return i
        return idx

    def _heat_color(self, heat: float, is_nccl: bool) -> int:
        """Map heat to curses color pair number."""
        if is_nccl:
            return 5 if heat > 0.5 else 6
        if heat > 0.75:
            return 1
        if heat > 0.5:
            return 2
        if heat > 0.25:
            return 3
        return 4

    def run(self, stdscr):
        """Main curses loop."""
        curses.curs_set(0)
        curses.use_default_colors()

        # Init color pairs
        curses.init_pair(1, curses.COLOR_GREEN, -1)    # bright green
        curses.init_pair(2, curses.COLOR_GREEN, -1)    # green
        curses.init_pair(3, curses.COLOR_GREEN, -1)    # dim green
        curses.init_pair(4, curses.COLOR_WHITE, -1)    # dim
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)  # nccl bright
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # nccl dim
        curses.init_pair(7, curses.COLOR_CYAN, -1)     # duration/info
        curses.init_pair(8, curses.COLOR_BLUE, -1)     # nvtx scope
        curses.init_pair(9, curses.COLOR_YELLOW, -1)   # percentage
        curses.init_pair(10, curses.COLOR_RED, -1)     # filter highlight

        try:
            self._main_loop(stdscr)
        except KeyboardInterrupt:
            pass  # Ctrl+C exits cleanly

    def _main_loop(self, stdscr):

        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()

            # Header (2 lines)
            header = f" {self.title}  |  {self.total_nvtx} NVTX, {self.total_kernels} kernels, {_fmt_dur(self.total_gpu_ms)} GPU  [{self.view_mode.upper()}]"
            stdscr.addnstr(0, 0, header, width - 1, curses.A_BOLD)

            # Status bar
            bar = f" Depth: {'∞' if self.max_depth < 0 else self.max_depth}"
            if self.min_dur_us > 0:
                bar += f"  ≥{self.min_dur_us}μs"
            if self.filter_text:
                fmode = 'dynamic' if self.dynamic_filter else 'static'
                bar += f"  Filter[{fmode}]: \"{self.filter_text}\""
            if self.show_demangled:
                bar += "  [D]emangle"
            if self.show_times:
                bar += "  [T]imes"
            if self.show_bubbles:
                bar += f"  [B]ubbles≥{self.bubble_threshold_us}μs"
            if self.status_msg:
                bar += f"  [{self.status_msg}]"
            stdscr.addnstr(1, 0, bar, width - 1, curses.color_pair(7))

            # Help line
            help_text = " ↑↓jk:nav ←→:expand e/c:this E/C:all /:filter m:min S:save p:panel h:help q:quit"
            stdscr.addnstr(height - 1, 0, help_text[:width - 1], width - 1, curses.A_DIM)

            # Visible rows
            visible = self._visible_rows()
            display_height = height - 7  # header(1) + status(1) + breadcrumb(1) + tree area + detail(2) + help(1) + gap

            # Breadcrumb hierarchy — show NVTX path of cursor's position
            breadcrumb = ""
            if visible and 0 <= self.cursor < len(visible):
                cur_node = visible[self.cursor]
                # Build breadcrumb from node.path ("root > child > grandchild")
                if cur_node.path:
                    parts = [p.strip() for p in cur_node.path.split(' > ')]
                    # For kernels, show parent path. For NVTX, show own path.
                    breadcrumb = ' › '.join(parts)
            crumb_line = f" ⌂ {breadcrumb}" if breadcrumb else " ⌂ (root)"
            stdscr.addnstr(2, 0, crumb_line, width - 1, curses.color_pair(9) | curses.A_DIM)

            if not visible:
                stdscr.addnstr(4, 2, "(no visible nodes)", width - 3, curses.A_DIM)
            else:
                # Clamp cursor
                self.cursor = max(0, min(self.cursor, len(visible) - 1))

                # Adjust scroll
                if self.cursor < self.scroll_offset:
                    self.scroll_offset = self.cursor
                if self.cursor >= self.scroll_offset + display_height:
                    self.scroll_offset = self.cursor - display_height + 1

                for i in range(display_height):
                    row_idx = self.scroll_offset + i
                    if row_idx >= len(visible):
                        break

                    node = visible[row_idx]
                    y = i + 4  # offset for header + status + breadcrumb
                    is_selected = (row_idx == self.cursor)

                    indent = "  " * node.depth if self.view_mode == 'tree' else ""
                    time_prefix = f"{_fmt_ns(node.start_ns)} " if self.view_mode == 'linear' and node.start_ns is not None else ""

                    if node.type == 'nvtx':
                        arrow = '▼' if node.expanded else '▶'
                        name = node.name

                        info_parts = []
                        if node.kernel_count:
                            info_parts.append(f"{node.kernel_count}k")
                        if node.nvtx_count:
                            info_parts.append(f"{node.nvtx_count}s")
                        info = f" ({','.join(info_parts)})" if info_parts else ""

                        pct = f" {node.relative_pct:.0f}%" if node.relative_pct < 100 else ""
                        dur = f" {_fmt_dur(node.duration_ms)}"
                        time_str = ""
                        if self.show_times:
                            time_str = f" [{_fmt_ns(node.start_ns)}..{_fmt_ns(node.end_ns)}]"

                        suffix = f"{dur}{info}{pct}{time_str}"
                        max_name_len = width - len(indent) - len(suffix) - 6
                        if max_name_len > 0 and len(name) > max_name_len:
                            name = name[:max_name_len - 1] + '…'

                        line = f" {time_prefix}{indent}{arrow} {name}{suffix}"

                    elif node.type == 'kernel':
                        is_nccl = 'nccl' in node.name.lower()
                        icon = '⚡' if is_nccl else '▸'
                        name = node.demangled if (self.show_demangled and node.demangled) else node.name
                        dur = f" {_fmt_dur(node.duration_ms)}"
                        stream = f" [Stream {node.stream}]"
                        time_str = ""
                        if self.show_times:
                            time_str = f" [{_fmt_ns(node.start_ns)}..{_fmt_ns(node.end_ns)}]"

                        suffix = f"{dur}{stream}{time_str}"
                        max_name_len = width - len(indent) - len(suffix) - 6
                        if max_name_len > 0 and len(name) > max_name_len:
                            name = name[:max_name_len - 1] + '…'

                        line = f" {time_prefix}{indent}{icon} {name}{suffix}"

                        # Bubble annotation
                        bubble_us = getattr(node, '_bubble_us', 0)
                        if bubble_us > 0:
                            bubble_str = f" ⚠ {_fmt_dur(bubble_us / 1000)}gap"
                            line += bubble_str
                    else:
                        line = f" {time_prefix}{indent}? {node.name}"

                    # Apply styling
                    attr = curses.A_NORMAL
                    bubble_us = getattr(node, '_bubble_us', 0)

                    # Selection range highlight
                    in_selection = False
                    if self.selection_anchor >= 0:
                        sel_lo = min(self.selection_anchor, self.cursor)
                        sel_hi = max(self.selection_anchor, self.cursor)
                        if sel_lo <= row_idx <= sel_hi:
                            in_selection = True

                    if is_selected:
                        attr |= curses.A_REVERSE
                    elif in_selection:
                        attr |= curses.A_REVERSE | curses.A_DIM
                    elif bubble_us > 0:
                        attr |= curses.color_pair(9)

                    if node.type == 'nvtx':
                        attr |= curses.color_pair(8)
                    elif node.type == 'kernel':
                        is_nccl = 'nccl' in node.name.lower()
                        attr |= curses.color_pair(self._heat_color(node.heat, is_nccl))

                    try:
                        stdscr.addnstr(y, 0, line, width - 1, attr)
                    except curses.error:
                        pass

            # ── Bookmarks panel (right side) ──
            if self.show_bookmarks and self.bookmarks:
                panel_w = min(35, width // 3)
                panel_x = width - panel_w
                panel_y_start = 4
                panel_y_end = min(panel_y_start + len(self.bookmarks) + 2, height - 4)
                # Header
                try:
                    bm_header = f" ─ Bookmarks ({len(self.bookmarks)}) ─"
                    stdscr.addnstr(panel_y_start, panel_x, bm_header[:panel_w], panel_w, curses.A_BOLD | curses.color_pair(7))
                except curses.error:
                    pass
                # List
                for bi, bm in enumerate(self.bookmarks):
                    row_y = panel_y_start + 1 + bi
                    if row_y >= panel_y_end:
                        break
                    bm_line = f" {bi+1}. {bm['name'][:panel_w - 5]}"
                    try:
                        stdscr.addnstr(row_y, panel_x, bm_line[:panel_w], panel_w,
                                       curses.color_pair(9))
                    except curses.error:
                        pass

            # ── Detail bar: selected node info + mini timeline ──
            detail_y = height - 3
            if visible and 0 <= self.cursor < len(visible):
                sel = visible[self.cursor]
                # Line 1: fixed-width columns — time │ dur stream │ name
                # Col 1: time range (fixed 22 chars)
                if sel.start_ns is not None and sel.end_ns is not None:
                    time_col = f"{_fmt_ns(sel.start_ns)}→{_fmt_ns(sel.end_ns)}"
                else:
                    time_col = "?"
                time_col = time_col.ljust(22)
                # Col 2: duration + stream (fixed 20 chars)
                dur_col = _fmt_dur(sel.duration_ms)
                if sel.type == 'kernel':
                    dur_col += f" [S{sel.stream}]"
                if sel.relative_pct < 100:
                    dur_col += f" {sel.relative_pct:.0f}%"
                dur_col = dur_col.ljust(20)
                # Col 3: name (fills rest)
                info = f" {time_col} │ {dur_col} │ {sel.name}"
                try:
                    stdscr.addnstr(detail_y, 0, info[:width - 1], width - 1,
                                   curses.A_BOLD | curses.color_pair(7))
                except curses.error:
                    pass

                # Line 2: mini timeline bar
                trim_start, trim_end = self.trim
                trim_span = max(trim_end - trim_start, 1)
                bar_width = min(width - 6, 60)
                if bar_width > 10 and sel.start_ns is not None and sel.end_ns is not None:
                    # Compute positions
                    s_pos = max(0, min(bar_width - 1, int((sel.start_ns - trim_start) / trim_span * bar_width)))
                    e_pos = max(s_pos + 1, min(bar_width, int((sel.end_ns - trim_start) / trim_span * bar_width)))
                    bar_chars = ['─'] * bar_width
                    for i in range(s_pos, e_pos):
                        if i < bar_width:
                            bar_chars[i] = '█'
                    timeline = f" [{_fmt_ns(trim_start)}] {''.join(bar_chars)} [{_fmt_ns(trim_end)}]"
                    try:
                        stdscr.addnstr(detail_y + 1, 0, timeline[:width - 1], width - 1,
                                       curses.color_pair(9))
                    except curses.error:
                        pass

            # Filter input mode
            if self.filter_mode:
                fmode_label = " [LIVE]" if self.dynamic_filter else ""
                prompt = f" Filter{fmode_label}: {self.filter_input}█"
                try:
                    stdscr.addnstr(height - 2, 0, prompt, width - 1,
                                   curses.A_BOLD | curses.color_pair(10))
                except curses.error:
                    pass

            # Threshold input mode
            if self.threshold_mode:
                labels = {'min_dur': 'Min duration (μs)', 'bubble': 'Bubble threshold (μs)',
                          'trim': 'Trim (START_S END_S)', 'bookmark': 'Bookmark # to jump to'}
                label = labels.get(self.threshold_target, '?')
                prompt = f" {label}: {self.threshold_input}█"
                try:
                    stdscr.addnstr(height - 2, 0, prompt, width - 1,
                                   curses.A_BOLD | curses.color_pair(9))
                except curses.error:
                    pass

            stdscr.refresh()

            # Handle input
            key = stdscr.getch()

            if self.threshold_mode:
                if key == 27:  # Esc
                    self.threshold_mode = False
                    self.threshold_input = ''
                    self.status_msg = 'Threshold cancelled'
                elif key in (10, 13):  # Enter
                    raw = self.threshold_input.strip()
                    try:
                        val = int(raw) if raw else 0
                    except ValueError:
                        val = 0
                    self.threshold_mode = False
                    self.threshold_input = ''
                    self.cursor = 0
                    if self.threshold_target == 'trim':
                        parts = raw.split()
                        if len(parts) == 2:
                            try:
                                s, e = float(parts[0]), float(parts[1])
                                self.trim = (int(s * 1e9), int(e * 1e9))
                                self._reload()
                            except ValueError:
                                self.status_msg = 'Invalid trim (use: 39 42)'
                        else:
                            self.status_msg = 'Format: START_S END_S'
                    elif self.threshold_target == 'bookmark':
                        try:
                            idx = int(raw) - 1
                        except ValueError:
                            idx = -1
                        if 0 <= idx < len(self.bookmarks):
                            bm = self.bookmarks[idx]
                            # Find node closest to bookmark start_ns
                            target_ns = bm['start_ns']
                            best_i = 0
                            best_diff = float('inf')
                            for vi, vn in enumerate(visible):
                                if vn.start_ns is not None:
                                    diff = abs(vn.start_ns - target_ns)
                                    if diff < best_diff:
                                        best_diff = diff
                                        best_i = vi
                            self.cursor = best_i
                            self.status_msg = f'Jumped to bookmark #{idx+1}'
                        else:
                            self.status_msg = f'No bookmark #{idx+1}'
                    elif self.threshold_target == 'bubble':
                        self.bubble_threshold_us = max(0, val)
                        self.status_msg = f'Bubble threshold: {self.bubble_threshold_us}μs'
                    else:
                        self.min_dur_us = max(0, val)
                        self.status_msg = f'Min duration: {self.min_dur_us}μs' if self.min_dur_us else 'Min duration: off'
                elif key in (8, 127, 263):  # Backspace
                    self.threshold_input = self.threshold_input[:-1]
                elif ord('0') <= key <= ord('9') or key in (ord('.'), ord(' ')):
                    self.threshold_input += chr(key)
                continue

            if self.filter_mode:
                if key == 27:  # Esc
                    self.filter_mode = False
                    self.filter_input = ''
                    if not self.dynamic_filter:
                        self.status_msg = 'Filter cancelled'
                elif key in (10, 13):  # Enter
                    self.filter_text = self.filter_input
                    self.filter_mode = False
                    self.cursor = 0
                    self.scroll_offset = 0
                    self.status_msg = f'Filter: "{self.filter_text}"' if self.filter_text else ''
                elif key in (8, 127, 263):  # Backspace
                    self.filter_input = self.filter_input[:-1]
                    if self.dynamic_filter:
                        self.filter_text = self.filter_input
                        self.cursor = 0
                        self.scroll_offset = 0
                elif 32 <= key <= 126:
                    self.filter_input += chr(key)
                    if self.dynamic_filter:
                        self.filter_text = self.filter_input
                        self.cursor = 0
                        self.scroll_offset = 0
                continue

            if key == ord('q'):
                break
            elif key in (curses.KEY_UP, ord('k')):
                self.cursor = max(0, self.cursor - 1)
                self.selection_anchor = -1  # clear selection on normal nav
                self.status_msg = ''
            elif key in (curses.KEY_DOWN, ord('j')):
                self.cursor = min(len(visible) - 1, self.cursor + 1) if visible else 0
                self.selection_anchor = -1
                self.status_msg = ''
            elif key == curses.KEY_SR or key == 337:  # Shift+Up
                if self.selection_anchor < 0:
                    self.selection_anchor = self.cursor
                self.cursor = max(0, self.cursor - 1)
                sel_count = abs(self.cursor - self.selection_anchor) + 1
                self.status_msg = f'Selected: {sel_count} rows'
            elif key == curses.KEY_SF or key == 336:  # Shift+Down
                if self.selection_anchor < 0:
                    self.selection_anchor = self.cursor
                self.cursor = min(len(visible) - 1, self.cursor + 1) if visible else 0
                sel_count = abs(self.cursor - self.selection_anchor) + 1
                self.status_msg = f'Selected: {sel_count} rows'
            elif key in (curses.KEY_LEFT,):
                # Left = collapse current, or jump to parent
                if visible and 0 <= self.cursor < len(visible):
                    node = visible[self.cursor]
                    if node.type == 'nvtx' and node.expanded:
                        node.expanded = False
                        self.status_msg = f'Collapsed: {node.name}'
                    else:
                        # Jump to parent
                        parent_idx = self._find_parent(visible, self.cursor)
                        if parent_idx != self.cursor:
                            self.cursor = parent_idx
            elif key in (curses.KEY_RIGHT,):
                # Right = expand current NVTX
                if visible and 0 <= self.cursor < len(visible):
                    node = visible[self.cursor]
                    if node.type == 'nvtx' and node.has_children and not node.expanded:
                        node.expanded = True
                        self.status_msg = f'Expanded: {node.name}'
            elif key in (curses.KEY_PPAGE,):  # Page Up
                self.cursor = max(0, self.cursor - display_height)
            elif key in (curses.KEY_NPAGE,):  # Page Down
                self.cursor = min(len(visible) - 1, self.cursor + display_height) if visible else 0
            elif key == curses.KEY_HOME:
                self.cursor = 0
                self.scroll_offset = 0
            elif key == curses.KEY_END:
                self.cursor = len(visible) - 1 if visible else 0
            elif key in (10, 13, ord(' ')):  # Enter/Space = toggle
                if visible and 0 <= self.cursor < len(visible):
                    node = visible[self.cursor]
                    if node.type == 'nvtx' and node.has_children:
                        node.expanded = not node.expanded
                        self.status_msg = f"{'Expanded' if node.expanded else 'Collapsed'}: {node.name}"
            elif key == ord('e'):
                # Expand one level down from cursor
                if visible and 0 <= self.cursor < len(visible):
                    node = visible[self.cursor]
                    if node.type == 'nvtx' and node.has_children:
                        node.expanded = True
                        self.status_msg = f'Expanded: {node.name}'
                    else:
                        self.status_msg = 'Not an expandable scope'
            elif key == ord('c'):
                # Collapse selected level
                if visible and 0 <= self.cursor < len(visible):
                    node = visible[self.cursor]
                    if node.type == 'nvtx' and node.expanded:
                        node.expanded = False
                        self.status_msg = f'Collapsed: {node.name}'
                    else:
                        # Jump to parent and collapse it
                        parent_idx = self._find_parent(visible, self.cursor)
                        if parent_idx != self.cursor:
                            visible[parent_idx].expanded = False
                            self.cursor = parent_idx
                            self.status_msg = f'Collapsed: {visible[parent_idx].name}'
            elif key == ord('E'):
                self._expand_all()
            elif key == ord('C'):
                self._collapse_all()
            elif key == ord('h'):
                # Toggle help — just show a status message with key summary
                self.status_msg = 'e/c:expand/collapse E/C:all ←→:nav /:filter m:min S:bookmark p:panel v:view'
            elif key == ord('/'):
                self.filter_mode = True
                self.filter_input = self.filter_text
                self.status_msg = ''
            elif key == ord('F'):
                self.dynamic_filter = not self.dynamic_filter
                self.status_msg = f"Filter mode: {'dynamic' if self.dynamic_filter else 'static'}"
            elif key == ord('n'):
                self.filter_text = ''
                self.cursor = 0
                self.scroll_offset = 0
                self.status_msg = 'Filter cleared'
            elif key == ord('v'):
                self.view_mode = 'linear' if self.view_mode == 'tree' else 'tree'
                self.cursor = 0
                self.scroll_offset = 0
                self.selection_anchor = -1
                self.status_msg = f"View: {self.view_mode}"
            elif key == ord('S'):
                # Save bookmark from selection or current node
                if visible:
                    if self.selection_anchor >= 0:
                        lo = min(self.selection_anchor, self.cursor)
                        hi = max(self.selection_anchor, self.cursor)
                    else:
                        lo = hi = self.cursor
                    if 0 <= lo < len(visible) and 0 <= hi < len(visible):
                        n_lo, n_hi = visible[lo], visible[hi]
                        s_ns = n_lo.start_ns if n_lo.start_ns is not None else 0
                        e_ns = n_hi.end_ns if n_hi.end_ns is not None else 0
                        label = f"{_fmt_ns(s_ns)}-{_fmt_ns(e_ns)} {n_lo.name}"
                        if lo != hi:
                            label += f"..{n_hi.name}"
                        bm = {'name': label, 'start_ns': s_ns, 'end_ns': e_ns,
                              'path': n_lo.path}
                        self.bookmarks.append(bm)
                        self.selection_anchor = -1
                        self.status_msg = f'Bookmark #{len(self.bookmarks)} saved'
            elif key == ord('p'):
                self.show_bookmarks = not self.show_bookmarks
                self.status_msg = f"Bookmarks: {'ON' if self.show_bookmarks else 'OFF'}"
            elif key == ord('G'):
                # Jump to bookmark — show list and prompt
                if self.bookmarks:
                    self.threshold_mode = True
                    self.threshold_target = 'bookmark'
                    self.threshold_input = ''
                    bm_list = '  '.join(f'{i+1}:{b["name"][:20]}' for i, b in enumerate(self.bookmarks[:5]))
                    self.status_msg = f'Jump to: {bm_list}'
            elif key == ord('d'):
                self.show_demangled = not self.show_demangled
                self.status_msg = f"Demangled: {'ON' if self.show_demangled else 'OFF'}"
            elif key == ord('t'):
                self.show_times = not self.show_times
                self.status_msg = f"Timestamps: {'ON' if self.show_times else 'OFF'}"
            elif ord('1') <= key <= ord('5'):
                self.max_depth = key - ord('0')
                self.cursor = 0
                self.status_msg = f'Depth: {self.max_depth}'
            elif key == ord('0'):
                self.max_depth = -1
                self.status_msg = 'Depth: unlimited'
            elif key == ord('m'):
                self.threshold_mode = True
                self.threshold_target = 'min_dur'
                self.threshold_input = str(self.min_dur_us) if self.min_dur_us else ''
                self.status_msg = ''
            elif key == ord('+') or key == ord('='):
                # Increase min duration: step through 0 -> 1 -> 10 -> 100 -> 1000 -> 10000
                steps = [0, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
                cur = self.min_dur_us
                nxt = next((s for s in steps if s > cur), cur + 1000)
                self.min_dur_us = nxt
                self.cursor = 0
                self.status_msg = f'Min: {self.min_dur_us}μs'
            elif key == ord('-') or key == ord('_'):
                # Decrease min duration
                steps = [0, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
                cur = self.min_dur_us
                prev = next((s for s in reversed(steps) if s < cur), 0)
                self.min_dur_us = prev
                self.cursor = 0
                self.status_msg = f'Min: {self.min_dur_us}μs' if self.min_dur_us else 'Min: off'
            elif key == ord('b'):
                self.show_bubbles = not self.show_bubbles
                self.status_msg = f"Bubbles: {'ON' if self.show_bubbles else 'OFF'}"
            elif key == ord('B'):
                self.threshold_mode = True
                self.threshold_target = 'bubble'
                self.threshold_input = str(self.bubble_threshold_us)
                self.status_msg = 'Enter bubble threshold (μs)'
            elif key == ord('R'):
                self._reload()
            elif key == ord('T'):
                self.threshold_mode = True
                self.threshold_target = 'trim'
                self.threshold_input = f"{self.trim[0]/1e9:.1f} {self.trim[1]/1e9:.1f}"
                self.status_msg = 'Enter trim: START_S END_S'


def render_tree(roots: list[dict], title: str = "NVTX Tree",
                max_depth: int = -1, min_ms: float = 0,
                width: int = None):
    """Non-interactive render (fallback for piped output)."""
    from rich.console import Console
    from rich.tree import Tree
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table

    console = Console(width=width)
    total_nvtx = total_kernels = 0
    total_gpu_ms = 0.0
    kernel_by_name = {}

    def _count(nodes):
        nonlocal total_nvtx, total_kernels, total_gpu_ms
        for n in nodes:
            if n["type"] == "nvtx":
                total_nvtx += 1
            elif n["type"] == "kernel":
                total_kernels += 1
                total_gpu_ms += n.get("duration_ms", 0)
                kn = n["name"]
                kernel_by_name[kn] = kernel_by_name.get(kn, 0) + n.get("duration_ms", 0)
            if n.get("children"):
                _count(n["children"])

    _count(roots)

    def _add(parent, node, depth):
        dur = node.get("duration_ms", 0)
        if min_ms > 0 and dur < min_ms:
            return
        if 0 <= max_depth < depth:
            return
        name = node.get("name", "?")
        ntype = node.get("type", "")
        if ntype == "nvtx":
            label = Text()
            label.append("📁 ", style="dim")
            label.append(name, style="bold dodger_blue1")
            label.append(f"  {_fmt_dur(dur)}", style="dim cyan")
            branch = parent.add(label)
            for c in node.get("children", []):
                _add(branch, c, depth + 1)
        elif ntype == "kernel":
            is_nccl = "nccl" in name.lower()
            color = "bright_magenta" if is_nccl else "green"
            label = Text()
            label.append("⚡ " if is_nccl else "▸ ", style=color)
            label.append(name, style=color)
            label.append(f"  {_fmt_dur(dur)}", style="dim cyan")
            label.append(f"  S{node.get('stream', '?')}", style="dim")
            parent.add(label)

    tree = Tree(f"[bold]{title}[/bold]")
    for root in roots:
        _add(tree, root, 0)
    console.print()
    console.print(tree)
    console.print()

    top5 = sorted(kernel_by_name.items(), key=lambda x: -x[1])[:5]
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("NVTX scopes", str(total_nvtx))
    summary.add_row("GPU kernels", str(total_kernels))
    summary.add_row("Total GPU time", _fmt_dur(total_gpu_ms))
    if top5:
        summary.add_row("", "")
        summary.add_row("[bold]Top kernels", "[bold]Time")
        for kname, ms in top5:
            pct = ms / total_gpu_ms * 100 if total_gpu_ms > 0 else 0
            short = kname[:60] + "…" if len(kname) > 60 else kname
            summary.add_row(f"  {short}", f"{_fmt_dur(ms)} ({pct:.0f}%)")
    console.print(Panel(summary, title="Summary", border_style="dim"))
    console.print()


def run_tui(db_path: str, device: int, trim: tuple[int, int],
            max_depth: int = -1, min_ms: float = 0):
    """
    Main entry point: load profile, build tree, launch interactive TUI.
    Falls back to static rich rendering if stdout is not a terminal.
    """
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

    # If not a terminal (piped), fall back to static rich rendering
    if not os.isatty(1):
        render_tree(json_roots, title=title, max_depth=max_depth, min_ms=min_ms)
        return

    # Interactive mode
    tui = InteractiveTUI(json_roots, title=title, db_path=db_path,
                         device=device, trim=trim)
    if max_depth >= 0:
        tui.max_depth = max_depth
    curses.wrapper(tui.run)
