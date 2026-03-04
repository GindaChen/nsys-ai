"""
timeline/widgets.py — Supporting Textual widgets for the timeline TUI.

Widgets:
    BottomPanel           — selected kernel detail + NVTX hierarchy path
    ConfigPanel           — adjustable config overlay
    TimelineFilterBar     — inline filter input (shown on '/')
    TimelineBookmarkPanel — numbered bookmark list overlay (shown on apostrophe)
"""
from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, Static

from ..formatting import fmt_dur as _fmt_dur
from ..formatting import fmt_ns as _fmt_ns
from ..tui_models import KernelEvent
from ..tui_models import short_kernel_name as _short_name


class BottomPanel(Widget):
    """Shows the selected kernel's detail and NVTX hierarchy."""

    DEFAULT_CSS = """
    BottomPanel {
        height: auto;
        min-height: 4;
        max-height: 12;
        dock: bottom;
        background: $surface-darken-2;
        border-top: solid $primary-darken-2;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._path_mode: str = "breadcrumb"  # "breadcrumb" or "hierarchy"
        self._nvtx_spans: list = []

    def compose(self) -> ComposeResult:
        yield Static("", id="bp-kernel")
        yield Static("", id="bp-nvtx")
        yield Static("", id="bp-stats")

    def toggle_path_mode(self) -> None:
        self._path_mode = "hierarchy" if self._path_mode == "breadcrumb" else "breadcrumb"

    def set_nvtx_spans(self, spans: list) -> None:
        self._nvtx_spans = spans

    def update(  # type: ignore[override]
        self,
        kernel: KernelEvent | None,
        stream: str,
        cursor_ns: int,
        time_start: int,
        time_end: int,
    ) -> None:
        k_w = self.query_one("#bp-kernel", Static)
        n_w = self.query_one("#bp-nvtx", Static)
        s_w = self.query_one("#bp-stats", Static)

        if kernel is None:
            k_w.update(f" Stream S{stream}  (no kernel at cursor {_fmt_ns(cursor_ns)})")
            n_w.update("")
            s_w.update("")
            return

        dur = _fmt_dur(kernel.duration_ms)
        short = _short_name(kernel.name)
        k_info = Text.assemble(
            (" ▸ ", "bold green"),
            (short, "bold"),
            f"  {dur}",
            f"  [{_fmt_ns(kernel.start_ns)} → {_fmt_ns(kernel.end_ns)}]",
        )
        k_w.update(k_info)

        if kernel.nvtx_path:
            if self._path_mode == "hierarchy":
                n_w.update(self._render_hierarchy(kernel.nvtx_path))
            else:
                n_w.update(Text(f" ⌂ {kernel.nvtx_path}", style="bright_cyan"))
        else:
            n_w.update("")

        # Mini timeline bar
        span = max(time_end - time_start, 1)
        bar_w = 50
        s_pos = max(0, int((kernel.start_ns - time_start) / span * bar_w))
        e_pos = min(bar_w, max(s_pos + 1, int((kernel.end_ns - time_start) / span * bar_w)))
        bar = "─" * s_pos + "█" * (e_pos - s_pos) + "─" * (bar_w - e_pos)
        s_w.update(Text(f" [{_fmt_ns(time_start)}] {bar} [{_fmt_ns(time_end)}]", style="yellow dim"))

    def _render_hierarchy(self, nvtx_path: str) -> Text:
        """Render NVTX path as indented hierarchy with timing from spans."""
        parts = [p.strip() for p in nvtx_path.split(" > ") if p.strip()]
        lines: list[str] = []
        for i, part in enumerate(parts):
            indent = "  " * (i + 1)
            # Try to find matching NVTX span for timing info
            timing = ""
            for s in self._nvtx_spans:
                if s.name == part and s.depth == i:
                    timing = f"  {_fmt_dur((s.end_ns - s.start_ns) / 1e6)}  [{_fmt_ns(s.start_ns)}→{_fmt_ns(s.end_ns)}]"
                    break
            lines.append(f"{indent}📂 {part}{timing}")
        return Text("\n".join(lines), style="bright_cyan") if lines else Text("")


class ConfigPanel(Widget):
    """Right-side config overlay."""

    class Changed(Message):
        """Posted when any config value is adjusted by the user."""

    can_focus = True

    BINDINGS = [
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("left", "value_dec", "Dec", show=False),
        Binding("right", "value_inc", "Inc", show=False),
        Binding("escape", "close_panel", "Close", show=False),
    ]

    DEFAULT_CSS = """
    ConfigPanel {
        width: 30;
        height: auto;
        dock: right;
        background: $surface;
        border-left: solid $primary-darken-2;
        display: none;
    }
    ConfigPanel.-visible {
        display: block;
    }
    """

    _ITEMS = [
        ("selected_rows", "Selected stream rows", 1, 5),
        ("default_rows", "Other stream rows", 1, 3),
        ("nvtx_depth", "NVTX depth (0=hide)", 0, 99),
    ]

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.selected_rows = 3
        self.default_rows = 1
        self.nvtx_depth = 99  # 99 = show all available levels
        self._cursor = 0

    def compose(self) -> ComposeResult:
        yield Static("─ Config (↑↓ select, ←→ adjust) ─", id="cfg-header")
        yield Static("", id="cfg-body")

    def on_mount(self) -> None:
        self._update_display()

    def _update_display(self) -> None:
        lines = []
        vals = [self.selected_rows, self.default_rows, self.nvtx_depth]
        for i, (_, label, mn, mx) in enumerate(self._ITEMS):
            marker = "▶ " if i == self._cursor else "  "
            lines.append(f"{marker}{label}: {vals[i]} [{mn}–{mx}]")
        self.query_one("#cfg-body", Static).update("\n".join(lines))

    def adjust(self, delta: int) -> None:
        attrs = ["selected_rows", "default_rows", "nvtx_depth"]
        attr = attrs[self._cursor]
        _, _, mn, mx = self._ITEMS[self._cursor]
        setattr(self, attr, max(mn, min(mx, getattr(self, attr) + delta)))
        self._update_display()
        self.post_message(self.Changed())

    def move_cursor(self, delta: int) -> None:
        self._cursor = max(0, min(len(self._ITEMS) - 1, self._cursor + delta))
        self._update_display()

    def toggle(self) -> None:
        if "-visible" in self.classes:
            self.remove_class("-visible")
        else:
            self.add_class("-visible")
            self.focus()

    def action_cursor_up(self) -> None:
        self.move_cursor(-1)

    def action_cursor_down(self) -> None:
        self.move_cursor(1)

    def action_value_dec(self) -> None:
        self.adjust(-1)

    def action_value_inc(self) -> None:
        self.adjust(1)

    def action_close_panel(self) -> None:
        self.remove_class("-visible")
        # Return focus to canvas
        try:
            from .canvas import TimelineCanvas
            self.app.query_one("#canvas", TimelineCanvas).focus()
        except Exception:
            pass


class TimelineFilterBar(Widget):
    """Inline filter input bar for the timeline — appears on '/', hides on Esc/Enter."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    TimelineFilterBar {
        height: 3;
        dock: top;
        display: none;
    }
    TimelineFilterBar.-visible {
        display: block;
    }
    """

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Filter kernels… (Enter apply, Esc cancel)", id="tl-filter-input")

    def action_cancel(self) -> None:
        self.hide_bar()

    def show_bar(self, initial: str = "") -> None:
        self.add_class("-visible")
        inp = self.query_one(Input)
        inp.value = initial
        inp.focus()

    def hide_bar(self) -> None:
        self.remove_class("-visible")


class TimelineMinDurBar(Widget):
    """Inline min-duration input — appears on 'm', hides on Esc/Enter."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    TimelineMinDurBar {
        height: 3;
        dock: top;
        display: none;
    }
    TimelineMinDurBar.-visible {
        display: block;
    }
    """

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Min duration μs (Enter apply, Esc cancel)", id="tl-mindur-input")

    def action_cancel(self) -> None:
        self.hide_bar()

    def show_bar(self, current_us: float) -> None:
        self.add_class("-visible")
        inp = self.query_one(Input)
        inp.value = str(int(current_us)) if current_us else ""
        inp.focus()

    def hide_bar(self) -> None:
        self.remove_class("-visible")


class TimelineBookmarkPanel(Widget):
    """Numbered bookmark list overlay — shown on apostrophe key.

    Pressing 1-9 while visible jumps to that bookmark.
    ESC or apostrophe again closes the panel.
    """

    BINDINGS = [
        Binding("escape", "close_panel", "Close", show=False),
        Binding("apostrophe", "close_panel", "Close", show=False),
        Binding("1", "jump_1", "", show=False),
        Binding("2", "jump_2", "", show=False),
        Binding("3", "jump_3", "", show=False),
        Binding("4", "jump_4", "", show=False),
        Binding("5", "jump_5", "", show=False),
        Binding("6", "jump_6", "", show=False),
        Binding("7", "jump_7", "", show=False),
        Binding("8", "jump_8", "", show=False),
        Binding("9", "jump_9", "", show=False),
    ]

    can_focus = True

    DEFAULT_CSS = """
    TimelineBookmarkPanel {
        width: 50;
        height: auto;
        dock: right;
        background: $surface;
        border-left: solid $primary-darken-2;
        display: none;
    }
    TimelineBookmarkPanel.-visible {
        display: block;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("─ Bookmarks (1-9 to jump, Esc cancel) ─", id="tbm-header")
        yield Static("", id="tbm-list")

    def show_panel(self, bookmarks: list[dict]) -> None:
        self.add_class("-visible")
        self._render_list(bookmarks)
        self.focus()

    def hide_panel(self) -> None:
        self.remove_class("-visible")

    def _render_list(self, bookmarks: list[dict]) -> None:
        from ..formatting import fmt_ns as _fmt_ns_local
        if not bookmarks:
            self.query_one("#tbm-list", Static).update("(no bookmarks saved)")
            return
        lines = []
        for i, bm in enumerate(bookmarks[:9]):
            ts = _fmt_ns_local(bm["cursor_ns"])
            extra = ""
            if "kernel_name" in bm:
                extra = f"  [{bm['kernel_name'][:20]}]"
            if "range_start_ns" in bm:
                extra += "  ↔range"
            lines.append(f"  {i+1}  {bm['name'][:24]}  {ts}{extra}")
        self.query_one("#tbm-list", Static).update("\n".join(lines))

    def _jump(self, n: int) -> None:
        """Jump to bookmark n (1-indexed) via the app."""
        from .app import NsysTimelineApp
        app = self.app
        if isinstance(app, NsysTimelineApp):
            app.jump_to_bookmark_n(n)
        self.hide_panel()

    def action_close_panel(self) -> None:
        self.hide_panel()
        # Return focus to canvas
        try:
            from .canvas import TimelineCanvas
            self.app.query_one("#canvas", TimelineCanvas).focus()
        except Exception:
            pass

    def action_jump_1(self) -> None:
        self._jump(1)

    def action_jump_2(self) -> None:
        self._jump(2)

    def action_jump_3(self) -> None:
        self._jump(3)

    def action_jump_4(self) -> None:
        self._jump(4)

    def action_jump_5(self) -> None:
        self._jump(5)

    def action_jump_6(self) -> None:
        self._jump(6)

    def action_jump_7(self) -> None:
        self._jump(7)

    def action_jump_8(self) -> None:
        self._jump(8)

    def action_jump_9(self) -> None:
        self._jump(9)
