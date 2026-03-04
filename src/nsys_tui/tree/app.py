"""
tree/app.py — NsysTreeApp: Textual App for the NVTX tree TUI.

Replaces the curses InteractiveTUI in tui.py.
Entry point: run_tui(db_path, device, trim, max_depth=-1, min_ms=0)

Layout:
    ┌────────────────────────────────────┐
    │  Header (title + stats)            │
    ├────────────────────────────────────┤
    │  FilterBar (hidden by default)     │
    ├────────────────────────────────────┤
    │  TreeTable (main content, 1fr)     │
    │                   BookmarkPanel ─┐ │
    │                                  │ │
    ├────────────────────────────────────┤
    │  DetailBar (2 rows, dock bottom)   │
    ├────────────────────────────────────┤
    │  ChatPanel (12 rows, collapsible)  │
    └────────────────────────────────────┘
"""
from __future__ import annotations

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Input

from .. import tui_actions
from ..tui_models import TreeNode
from .chat import ChatPanel
from .logic import (
    build_nodes,
    compute_summary,
    find_kernel_occurrence,
    find_parent,
    node_index_in_visible,
    visible_rows_linear,
    visible_rows_tree,
)
from .widgets import BookmarkPanel, DetailBar, FilterBar, TreeTable


class NsysTreeApp(App):
    """Textual NVTX tree browser — replaces curses InteractiveTUI."""

    TITLE = "nsys-ai Tree"

    CSS = """
    #main-body {
        height: 1fr;
        layout: horizontal;
    }
    #tree-col {
        width: 1fr;
        layout: vertical;
    }
    ChatPanel {
        display: none;
    }
    ChatPanel.-active {
        display: block;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("A", "toggle_chat", "AI Chat", priority=True),
        Binding("v", "toggle_view", "View mode"),
        Binding("e", "expand_node", "Expand"),
        Binding("c", "collapse_node", "Collapse"),
        Binding("E", "expand_all", "Expand all"),
        Binding("C", "collapse_all", "Collapse all"),
        Binding("/", "open_filter", "Filter"),
        Binding("n", "clear_filter", "Clear filter"),
        Binding("F", "toggle_live_filter", "Live filter"),
        Binding("d", "toggle_demangled", "Demangled"),
        Binding("B", "toggle_bubbles", "Bubbles"),
        Binding("R", "reload", "Reload"),
        Binding("p", "toggle_bookmarks", "Bookmarks"),
        Binding("S", "save_bookmark", "Save bookmark"),
        Binding("left,h", "go_left", "Collapse/parent", show=False),
        Binding("right,l", "go_right", "Expand", show=False),
        Binding("0", "depth_unlimited", "Depth ∞", show=False),
        Binding("1", "depth_1", "Depth 1", show=False),
        Binding("2", "depth_2", "Depth 2", show=False),
        Binding("3", "depth_3", "Depth 3", show=False),
        Binding("4", "depth_4", "Depth 4", show=False),
        Binding("5", "depth_5", "Depth 5", show=False),
        Binding("plus,equals_sign", "min_dur_up", "Min dur+", show=False),
        Binding("minus,underscore", "min_dur_down", "Min dur-", show=False),
    ]

    # -------------------------------------------------------------------------
    # Reactive state
    # -------------------------------------------------------------------------
    filter_text: reactive[str] = reactive("", layout=False)
    view_mode: reactive[str] = reactive("tree", layout=False)
    max_depth: reactive[int] = reactive(-1, layout=False)
    min_dur_us: reactive[float] = reactive(0, layout=False)
    show_demangled: reactive[bool] = reactive(False, layout=False)
    show_bubbles: reactive[bool] = reactive(False, layout=False)
    bubble_threshold_us: reactive[float] = reactive(10, layout=False)
    live_filter: reactive[bool] = reactive(False, layout=False)

    # -------------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        db_path: str,
        device: int,
        trim: tuple[int, int] | None,
        max_depth: int = -1,
        min_ms: float = 0,
        json_roots: list[dict] | None = None,
    ) -> None:
        super().__init__()
        self._db_path = db_path
        self._device = device
        self._trim = trim or (0, 0)
        self.max_depth = max_depth
        self.min_dur_us = min_ms * 1000  # convert ms → µs

        # Tree data — may be pre-loaded (test fixture) or loaded on mount
        self._json_roots: list[dict] = json_roots or []
        self._all_nodes: list[TreeNode] = build_nodes(self._json_roots) if json_roots else []
        self._visible: list[TreeNode] = []

        # Summary
        self._total_kernels = 0
        self._total_gpu_ms = 0.0
        self._total_nvtx = 0
        if json_roots:
            self._total_kernels, self._total_gpu_ms, self._total_nvtx = compute_summary(json_roots)

        # Bookmarks
        self._bookmarks: list[dict] = []

    @classmethod
    def from_json(cls, json_roots: list[dict], **kwargs: object) -> NsysTreeApp:
        """Factory for tests — skip DB loading, use pre-built JSON."""
        return cls(db_path="", device=0, trim=None, json_roots=json_roots, **kwargs)

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        from textual.containers import Horizontal, Vertical
        yield Header()
        with Horizontal(id="main-body"):
            with Vertical(id="tree-col"):
                yield FilterBar(id="filter-bar")
                yield TreeTable(id="tree-table")
            yield BookmarkPanel(id="bookmark-panel")
        yield DetailBar(id="detail-bar")
        yield ChatPanel(
            id="chat-panel",
            db_path=self._db_path,
            device=self._device,
            ui_context_fn=self._build_ui_context,
            on_action_fn=self._handle_ai_action,
        )
        yield Footer()

    # -------------------------------------------------------------------------
    # Mount / data load
    # -------------------------------------------------------------------------
    def on_mount(self) -> None:
        if not self._json_roots and self._db_path:
            self._load_from_db()
        else:
            self._refresh_table()
        self._update_title()
        # Ensure the DataTable has focus so App-level key bindings work.
        # Without this, a hidden widget (e.g. RichLog inside ChatPanel) can grab
        # focus first and silently swallow key presses.
        self.query_one(DataTable).focus()

    def _load_from_db(self) -> None:
        from .. import profile as _profile
        from ..nvtx_tree import build_nvtx_tree, to_json

        try:
            with _profile.open(self._db_path) as prof:
                roots = build_nvtx_tree(prof, self._device, self._trim)
                self._json_roots = to_json(roots)
        except Exception as e:
            self.notify(f"Failed to load profile: {e}", severity="error")
            return
        self._all_nodes = build_nodes(self._json_roots)
        self._total_kernels, self._total_gpu_ms, self._total_nvtx = compute_summary(self._json_roots)
        self._refresh_table()
        self._update_title()

    # -------------------------------------------------------------------------
    # Table refresh
    # -------------------------------------------------------------------------
    def _get_visible(self) -> list[TreeNode]:
        if self.view_mode == "linear":
            return visible_rows_linear(
                self._all_nodes,
                filter_text=self.filter_text,
                min_dur_us=self.min_dur_us,
                show_bubbles=self.show_bubbles,
                bubble_threshold_us=self.bubble_threshold_us,
            )
        return visible_rows_tree(
            self._all_nodes,
            filter_text=self.filter_text,
            max_depth=self.max_depth,
            min_dur_us=self.min_dur_us,
            show_bubbles=self.show_bubbles,
            bubble_threshold_us=self.bubble_threshold_us,
        )

    def _refresh_table(self, preserve_cursor: bool = True) -> None:
        # Remember the currently selected node so we can restore position.
        prev_node: TreeNode | None = None
        if preserve_cursor and self._visible:
            table = self.query_one("#tree-table", TreeTable)
            row = table.cursor_row
            if 0 <= row < len(self._visible):
                prev_node = self._visible[row]

        self._visible = self._get_visible()
        table = self.query_one("#tree-table", TreeTable)
        table.populate(self._visible, self.view_mode, self.show_demangled)

        # Restore cursor to the same node (or closest position).
        if prev_node is not None:
            new_idx = node_index_in_visible(self._visible, prev_node)
            if new_idx is not None:
                self.query_one(DataTable).move_cursor(row=new_idx)

        self._update_detail_bar()

    def _update_title(self) -> None:
        trim_s = f"{self._trim[0]/1e9:.1f}s–{self._trim[1]/1e9:.1f}s" if self._trim != (0, 0) else "full"
        self.title = (
            f"nsys-ai  GPU {self._device}  {trim_s}  |  "
            f"{self._total_nvtx} NVTX  {self._total_kernels} kernels  "
            f"[{self.view_mode.upper()}]"
        )

    def _update_detail_bar(self) -> None:
        dt = self.query_one("#tree-table", TreeTable)
        row = dt.cursor_row
        node = self._visible[row] if self._visible and 0 <= row < len(self._visible) else None
        self.query_one("#detail-bar", DetailBar).update_node(node, self._trim)

    # -------------------------------------------------------------------------
    # DataTable events
    # -------------------------------------------------------------------------
    @on(DataTable.RowHighlighted, "#tree-dt")
    def _on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._update_detail_bar()

    @on(DataTable.RowSelected, "#tree-dt")
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter on a DataTable row toggles expand/collapse for NVTX nodes."""
        self.action_toggle_node()

    # -------------------------------------------------------------------------
    # Filter events
    # -------------------------------------------------------------------------
    @on(Input.Submitted, "#filter-input")
    def _filter_submitted(self, event: Input.Submitted) -> None:
        self.filter_text = event.value
        self.query_one("#filter-bar", FilterBar).hide_bar()
        self._refresh_table()

    @on(Input.Changed, "#filter-input")
    def _filter_changed(self, event: Input.Changed) -> None:
        if self.live_filter:
            self.filter_text = event.value
            self._refresh_table()

    # -------------------------------------------------------------------------
    # Actions — view
    # -------------------------------------------------------------------------
    def action_toggle_view(self) -> None:
        self.view_mode = "linear" if self.view_mode == "tree" else "tree"
        self._refresh_table()
        self._update_title()

    def action_toggle_demangled(self) -> None:
        self.show_demangled = not self.show_demangled
        self._refresh_table()

    def action_toggle_timestamps(self) -> None:
        self.notify("Timestamps: use DetailBar", timeout=2)

    def action_toggle_live_filter(self) -> None:
        self.live_filter = not self.live_filter
        self.notify(f"Live filter: {'ON' if self.live_filter else 'OFF'}", timeout=2)

    # Actions — filter
    def action_open_filter(self) -> None:
        fb = self.query_one("#filter-bar", FilterBar)
        fb.show_bar(self.filter_text)

    def action_clear_filter(self) -> None:
        self.filter_text = ""
        self._refresh_table()
        self.notify("Filter cleared", timeout=2)

    # Actions — toggle (Enter/Space: expand if collapsed, collapse if expanded)
    def action_toggle_node(self) -> None:
        node = self._current_node()
        if node and node.type == "nvtx" and node.has_children:
            node.expanded = not node.expanded
            self._refresh_table()

    # Actions — expand/collapse
    def _current_node(self) -> TreeNode | None:
        dt = self.query_one("#tree-table", TreeTable)
        row = dt.cursor_row
        if self._visible and 0 <= row < len(self._visible):
            return self._visible[row]
        return None

    def action_expand_node(self) -> None:
        node = self._current_node()
        if node and node.type == "nvtx" and node.has_children:
            node.expanded = True
            self._refresh_table()

    def action_collapse_node(self) -> None:
        node = self._current_node()
        if node and node.type == "nvtx" and node.expanded:
            node.expanded = False
            self._refresh_table()
        elif node:
            dt = self.query_one("#tree-table", TreeTable)
            parent_idx = find_parent(self._visible, dt.cursor_row)
            if parent_idx != dt.cursor_row:
                self._visible[parent_idx].expanded = False
                self._refresh_table()

    def action_expand_all(self) -> None:
        for n in self._all_nodes:
            if n.type == "nvtx":
                n.expanded = True
        self._refresh_table()

    def action_collapse_all(self) -> None:
        for n in self._all_nodes:
            if n.type == "nvtx":
                n.expanded = False
        self._refresh_table()

    def action_go_left(self) -> None:
        node = self._current_node()
        if node and node.type == "nvtx" and node.expanded:
            node.expanded = False
            self._refresh_table()
        else:
            dt = self.query_one("#tree-table", TreeTable)
            parent_idx = find_parent(self._visible, dt.cursor_row)
            if parent_idx != dt.cursor_row:
                self.query_one(DataTable).move_cursor(row=parent_idx)

    def action_go_right(self) -> None:
        node = self._current_node()
        if node and node.type == "nvtx" and node.has_children and not node.expanded:
            node.expanded = True
            self._refresh_table()

    # Actions — depth
    def action_depth_unlimited(self) -> None:
        self.max_depth = -1
        self._refresh_table()

    # "Depth N" means "show N levels" — max_depth is 0-indexed, so depth N = max_depth N-1.
    def action_depth_1(self) -> None:
        self.max_depth = 0
        self._refresh_table()

    def action_depth_2(self) -> None:
        self.max_depth = 1
        self._refresh_table()

    def action_depth_3(self) -> None:
        self.max_depth = 2
        self._refresh_table()

    def action_depth_4(self) -> None:
        self.max_depth = 3
        self._refresh_table()

    def action_depth_5(self) -> None:
        self.max_depth = 4
        self._refresh_table()

    # Actions — min duration
    _DUR_STEPS = [0, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]

    def action_min_dur_up(self) -> None:
        cur = self.min_dur_us
        nxt = next((s for s in self._DUR_STEPS if s > cur), cur + 1000)
        self.min_dur_us = nxt
        self._refresh_table()
        self.notify(f"Min dur: {int(nxt)}μs" if nxt else "Min dur: off", timeout=2)

    def action_min_dur_down(self) -> None:
        cur = self.min_dur_us
        prev = next((s for s in reversed(self._DUR_STEPS) if s < cur), 0)
        self.min_dur_us = prev
        self._refresh_table()
        self.notify(f"Min dur: {int(prev)}μs" if prev else "Min dur: off", timeout=2)

    # Actions — bubbles
    def action_toggle_bubbles(self) -> None:
        """Toggle bubble gap detection display ('B' key — matches original tui.py)."""
        self.show_bubbles = not self.show_bubbles
        self._refresh_table()
        threshold = int(self.bubble_threshold_us)
        self.notify(f"Bubbles: {'ON  ≥' + str(threshold) + 'μs' if self.show_bubbles else 'OFF'}", timeout=2)

    # Actions — reload
    def action_reload(self) -> None:
        if self._db_path:
            self._load_from_db()
            self.notify("Reloaded", timeout=2)

    # Actions — bookmarks
    def action_toggle_bookmarks(self) -> None:
        bp = self.query_one("#bookmark-panel", BookmarkPanel)
        if "-visible" in bp.classes:
            bp.hide_panel()
        else:
            bp.show_bookmarks(self._bookmarks)

    def action_save_bookmark(self) -> None:
        node = self._current_node()
        if node:
            s_ns = node.start_ns or 0
            e_ns = node.end_ns or 0
            label = f"{node.name[:30]} @{s_ns//1_000_000}ms"
            self._bookmarks.append({"name": label, "start_ns": s_ns, "end_ns": e_ns})
            self.notify(f"Bookmark #{len(self._bookmarks)} saved", timeout=2)

    # Actions — chat
    def action_toggle_chat(self) -> None:
        cp = self.query_one("#chat-panel", ChatPanel)
        if "-active" in cp.classes:
            cp.remove_class("-active")
        else:
            cp.add_class("-active")
            cp.query_one("#chat-input", Input).focus()

    # -------------------------------------------------------------------------
    # AI integration
    # -------------------------------------------------------------------------
    def _build_ui_context(self) -> dict:
        node = self._current_node()
        sel_kernel = None
        if node and node.type == "kernel":
            sel_kernel = {
                "name": node.name,
                "duration_ms": node.duration_ms,
                "stream": node.stream,
            }
        return {
            "selected_kernel": sel_kernel,
            "view_state": {
                "time_range_s": [self._trim[0] / 1e9, self._trim[1] / 1e9],
                "scope": node.path if node else "",
            },
            "stats": {
                "total_gpu_ms": self._total_gpu_ms,
                "kernel_count": self._total_kernels,
                "nvtx_count": self._total_nvtx,
            },
        }

    def _handle_ai_action(self, action_dict: dict) -> None:
        tui_actions.execute_tui_action(action_dict, self)

    # -------------------------------------------------------------------------
    # Public API (for tui_actions dispatcher)
    # -------------------------------------------------------------------------
    def scroll_to_kernel(self, target_name: str, occurrence_index: int = 1) -> None:
        """Navigate to the Nth kernel with the given name (1-indexed)."""
        self.action_expand_all()
        target = find_kernel_occurrence(self._all_nodes, target_name, occurrence_index)
        if target is None:
            self.notify(f"AI: kernel not found: {target_name}", severity="warning", timeout=3)
            return
        self._visible = self._get_visible()
        idx = node_index_in_visible(self._visible, target)
        if idx is None:
            self.notify(f"AI: kernel not visible: {target_name}", severity="warning", timeout=3)
            return
        # Rebuild the table first (this resets cursor), then set cursor position.
        self._refresh_table()
        self.query_one(DataTable).move_cursor(row=idx)
        n_total = sum(1 for n in self._all_nodes if n.type == "kernel" and n.name == target_name)
        self.notify(f"AI → {target_name} ({occurrence_index}/{n_total})", timeout=3)

    def zoom_to_time_range(self, start_s: float, end_s: float) -> None:
        """Re-trim tree to a specific time range (in seconds)."""
        if end_s <= start_s:
            return
        self._trim = (int(start_s * 1e9), int(end_s * 1e9))
        if self._db_path:
            self._load_from_db()
        self._update_title()
        self.notify(f"AI zoom → {start_s:.2f}s–{end_s:.2f}s", timeout=3)


# ---------------------------------------------------------------------------
# Entry point (replaces tui.run_tui)
# ---------------------------------------------------------------------------

def run_tui(
    db_path: str,
    device: int,
    trim: tuple[int, int] | None,
    max_depth: int = -1,
    min_ms: float = 0,
) -> None:
    """Launch the Textual NVTX tree browser."""
    app = NsysTreeApp(db_path, device, trim, max_depth=max_depth, min_ms=min_ms)
    app.run()
