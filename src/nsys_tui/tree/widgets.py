"""
tree/widgets.py — Textual widgets for the NVTX tree TUI.

Widgets:
    TreeTable     — renders a list of TreeNode objects in a DataTable
    FilterBar     — Input widget with live-filter toggle
    DetailBar     — one-row selected-node info + mini timeline bar
    BookmarkPanel — right-side bookmark list overlay
"""
from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.widget import Widget
from textual.widgets import DataTable, Input, Static

from ..formatting import fmt_dur as _fmt_dur
from ..formatting import fmt_ns as _fmt_ns
from ..tui_models import TreeNode

# Heat / NCCL color mapping (Rich markup colors)
_HEAT_COLORS = {
    "hot": "bright_green",
    "warm": "green",
    "cool": "dark_green",
    "cold": "grey50",
    "nccl_hot": "bright_magenta",
    "nccl_cold": "magenta",
}


def _heat_color(heat: float, is_nccl: bool) -> str:
    if is_nccl:
        return _HEAT_COLORS["nccl_hot"] if heat > 0.5 else _HEAT_COLORS["nccl_cold"]
    if heat > 0.75:
        return _HEAT_COLORS["hot"]
    if heat > 0.5:
        return _HEAT_COLORS["warm"]
    if heat > 0.25:
        return _HEAT_COLORS["cool"]
    return _HEAT_COLORS["cold"]


# ---------------------------------------------------------------------------
# TreeTable
# ---------------------------------------------------------------------------

class TreeTable(Widget):
    """Wraps a DataTable to display a list of TreeNode objects.

    Call ``populate(nodes)`` to refresh from a new visible-row list.
    The DataTable's row_key is the index in the provided list.
    """

    DEFAULT_CSS = """
    TreeTable {
        height: 1fr;
    }
    TreeTable > DataTable {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        dt = DataTable(id="tree-dt", cursor_type="row", show_header=True)
        dt.add_columns("Node", "Type", "Duration", "Heat%")
        yield dt

    def populate(self, nodes: list[TreeNode], view_mode: str = "tree",
                 show_demangled: bool = False) -> None:
        """Replace table rows with the given node list."""
        dt = self.query_one(DataTable)
        dt.clear()
        for i, node in enumerate(nodes):
            indent = "  " * node.depth if view_mode == "tree" else ""
            # In linear view, prefix with start timestamp (matches original tui.py behaviour)
            time_prefix = (
                f"{_fmt_ns(node.start_ns)} "
                if view_mode == "linear" and node.start_ns is not None
                else ""
            )
            if node.type == "nvtx":
                arrow = "▼" if node.expanded else "▶"
                label = Text(f"{time_prefix}{indent}{arrow} {node.name}", style="bold blue")
                type_col = Text("NVTX", style="blue")
                heat_col = Text(f"{node.relative_pct:.0f}%", style="yellow")
            elif node.type == "kernel":
                is_nccl = "nccl" in node.name.lower()
                icon = "⚡" if is_nccl else "▸"
                name = (node.demangled if show_demangled and node.demangled else node.name)
                color = _heat_color(node.heat, is_nccl)
                label = Text(f"{time_prefix}{indent}{icon} {name}", style=color)
                type_col = Text("nccl" if is_nccl else "kernel",
                                style="magenta" if is_nccl else "green")
                heat_col = Text(f"{node.heat * 100:.0f}%", style=color)
            else:
                label = Text(f"{time_prefix}{indent}? {node.name}")
                type_col = Text(node.type or "?")
                heat_col = Text("")

            dur_text = _fmt_dur(node.duration_ms)
            if node._bubble_us > 0:
                dur_text += f" ↕{node._bubble_us:.0f}μs"
            dur_col = Text(dur_text, style="cyan")
            dt.add_row(label, type_col, dur_col, heat_col, key=str(i))

    @property
    def row_count(self) -> int:
        return self.query_one(DataTable).row_count

    @property
    def cursor_row(self) -> int:
        return self.query_one(DataTable).cursor_row


# ---------------------------------------------------------------------------
# FilterBar
# ---------------------------------------------------------------------------

class FilterBar(Widget):
    """Input bar for text filter with live-mode indicator.

    Attributes:
        (live filter state is managed by NsysTreeApp; this widget is a thin view.)
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    FilterBar {
        height: 3;
        dock: top;
        display: none;
    }
    FilterBar.-visible {
        display: block;
    }
    """

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Filter nodes… (Enter apply, Esc cancel)", id="filter-input")

    def action_cancel(self) -> None:
        self.hide_bar()

    def show_bar(self, initial: str = "") -> None:
        self.add_class("-visible")
        inp = self.query_one(Input)
        inp.value = initial
        inp.focus()

    def hide_bar(self) -> None:
        self.remove_class("-visible")


# ---------------------------------------------------------------------------
# DetailBar
# ---------------------------------------------------------------------------

MINI_BAR_WIDTH = 50


class DetailBar(Widget):
    """Two-line panel showing the currently selected node's details."""

    DEFAULT_CSS = """
    DetailBar {
        height: 2;
        dock: bottom;
        background: $surface-darken-2;
        color: $text;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="detail-info")
        yield Static("", id="detail-timeline")

    def update_node(self, node: TreeNode | None, trim: tuple[int, int]) -> None:
        info_w = self.query_one("#detail-info", Static)
        bar_w = self.query_one("#detail-timeline", Static)

        if node is None:
            info_w.update("")
            bar_w.update("")
            return

        # Line 1: time range | duration | stream | name
        if node.start_ns is not None and node.end_ns is not None:
            time_col = f"{_fmt_ns(node.start_ns)}→{_fmt_ns(node.end_ns)}"
        else:
            time_col = "?"

        dur_col = _fmt_dur(node.duration_ms)
        if node.type == "kernel":
            dur_col += f" [S{node.stream}]"
        if node.relative_pct < 100:
            dur_col += f" {node.relative_pct:.0f}%"

        info_w.update(
            Text.assemble(
                (f" {time_col:<24}", "dim"),
                " │ ",
                (f"{dur_col:<20}", "cyan bold"),
                " │ ",
                node.name,
            )
        )

        # Line 2: mini timeline bar
        trim_start, trim_end = trim
        trim_span = max(trim_end - trim_start, 1)
        bar = ""
        if node.start_ns is not None and node.end_ns is not None:
            s_pos = max(0, int((node.start_ns - trim_start) / trim_span * MINI_BAR_WIDTH))
            e_pos = max(s_pos + 1, int((node.end_ns - trim_start) / trim_span * MINI_BAR_WIDTH))
            e_pos = min(e_pos, MINI_BAR_WIDTH)
            chars = ["─"] * MINI_BAR_WIDTH
            for i in range(s_pos, e_pos):
                chars[i] = "█"
            bar = f" [{_fmt_ns(trim_start)}] {''.join(chars)} [{_fmt_ns(trim_end)}]"
        bar_w.update(Text(bar, style="yellow"))


# ---------------------------------------------------------------------------
# BookmarkPanel
# ---------------------------------------------------------------------------

class BookmarkPanel(Widget):
    """Right-side overlay listing saved bookmarks."""

    DEFAULT_CSS = """
    BookmarkPanel {
        width: 35;
        height: auto;
        dock: right;
        background: $surface;
        border-left: solid $primary-darken-2;
        display: none;
    }
    BookmarkPanel.-visible {
        display: block;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("─ Bookmarks ─", id="bm-header")
        yield Static("", id="bm-list")

    def show_bookmarks(self, bookmarks: list[dict]) -> None:
        self.add_class("-visible")
        if not bookmarks:
            self.query_one("#bm-list", Static).update("(none saved)")
            return
        lines = "\n".join(
            f" {i+1}. {bm['name'][:28]}" for i, bm in enumerate(bookmarks)
        )
        self.query_one("#bm-list", Static).update(lines)

    def hide_panel(self) -> None:
        self.remove_class("-visible")
