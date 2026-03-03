"""
tui_textual.py — Textual AI chat TUI for nsys-ai (§11.3, §11.8).

Usage (registered as CLI command):
    nsys-ai chat <profile>

Layout:
    Left panel:  DataTable of top kernels by GPU time.
    Right panel: RichLog (chat history) + Static (streaming) + Input.

Threading:
    stream_agent_loop runs in a @work(thread=True) worker.
    UI updates are dispatched via self.call_from_thread().
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import DataTable, Footer, Header, Input, Label, RichLog, Static

# ---------------------------------------------------------------------------
# Profile helper — load top kernels without importing the full Profile class.
# ---------------------------------------------------------------------------

def _load_top_kernels(sqlite_path: str, limit: int = 30) -> list[dict]:
    """Return top kernels by total GPU duration as a list of dicts.

    Each dict has: name, count, total_ms, avg_ms.
    Opens the file in read-only URI mode; returns [] on any error.
    """
    try:
        conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except Exception:
        return []
    try:
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        kernel_table = next(
            (
                t for t in (
                    "CUPTI_ACTIVITY_KIND_KERNEL",
                    "CUPTI_ACTIVITY_KIND_KERNEL_V2",
                    "CUPTI_ACTIVITY_KIND_KERNEL_V3",
                )
                if t in tables
            ),
            None,
        )
        if not kernel_table:
            return []
        rows = conn.execute(
            f"""
            SELECT s.value AS name,
                   COUNT(*) AS count,
                   SUM(k.[end] - k.start) / 1e6 AS total_ms,
                   AVG(k.[end] - k.start) / 1e6 AS avg_ms
            FROM {kernel_table} k
            JOIN StringIds s ON k.shortName = s.id
            GROUP BY k.shortName
            ORDER BY total_ms DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Textual App
# ---------------------------------------------------------------------------

class NsysChatApp(App):
    """Textual AI chat TUI for Nsight Systems profiles.

    Left panel : DataTable of top kernels (navigate_to_kernel lands here).
    Right panel: RichLog (completed turns) + Static (streaming) + Input.
    """

    # ── Textual Messages ─────────────────────────────────────────────────────

    class NavigateToKernel(Message):
        """Posted when the agent calls navigate_to_kernel."""
        def __init__(self, target_name: str, reason: str | None = None) -> None:
            super().__init__()
            self.target_name = target_name
            self.reason = reason

    class ZoomToTimeRange(Message):
        """Posted when the agent calls zoom_to_time_range."""
        def __init__(self, start_s: float, end_s: float) -> None:
            super().__init__()
            self.start_s = start_s
            self.end_s = end_s

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }

    #main-area {
        layout: horizontal;
        height: 1fr;
    }

    #left-panel {
        width: 46;
        min-width: 30;
        border-right: tall $primary-darken-2;
        padding: 0 1;
        layout: vertical;
    }

    #left-panel Label {
        text-style: bold;
        color: $accent;
        width: 100%;
        margin-bottom: 1;
    }

    DataTable {
        height: 1fr;
    }

    #right-panel {
        width: 1fr;
        layout: vertical;
        padding: 0 1;
    }

    #chat-log {
        height: 1fr;
        border: round $surface-lighten-2;
        padding: 0 1;
    }

    #streaming-area {
        height: auto;
        max-height: 12;
        padding: 0 1;
        color: $accent;
    }

    #chat-input {
        margin-top: 1;
    }

    #chat-input:focus {
        border: tall $accent;
    }

    #kernel-info-bar {
        height: auto;
        max-height: 4;
        padding: 0 2;
        color: $accent;
        background: $surface-darken-2;
        border-bottom: tall $accent;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("ctrl+l", "clear_chat", "Clear Chat", show=True),
        Binding("ctrl+c", "cancel_generation", "Cancel", show=False),
        Binding("escape", "quit", "Quit", show=True),
    ]

    def __init__(self, profile_path: str) -> None:
        super().__init__()
        self.profile_path = profile_path
        self.profile_name = Path(profile_path).name
        # LLM conversation history (user + assistant turns only; system is built internally).
        self._chat_messages: list[dict] = []
        # Top kernels loaded from the profile DB.
        self._kernels: list[dict] = []
        # Maps lowercase kernel name → DataTable row index for navigation.
        self._kernel_name_to_row: dict[str, int] = {}
        # Protects against overlapping stream workers.
        self._is_generating: bool = False
        # Accumulates chunks for the current streaming response.
        self._streaming_buffer: list[str] = []
        # Last navigation message (set by _on_action_navigate, consumed by _on_stream_done).
        self._last_nav_message: str = ""

    # ── Compose ──────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="main-area"):
            with Vertical(id="left-panel"):
                yield Label("Top Kernels (by GPU time)")
                yield DataTable(id="kernel-table", cursor_type="row", show_cursor=True)
            with Vertical(id="right-panel"):
                yield Static("", id="kernel-info-bar")
                yield RichLog(id="chat-log", markup=True, highlight=False, wrap=True)
                yield Static("", id="streaming-area")
                yield Input(
                    placeholder="Ask about this profile… (Enter to send)",
                    id="chat-input",
                )
        yield Footer()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self.title = f"nsys-ai chat — {self.profile_name}"
        self._setup_table_columns()
        self._load_and_populate_kernels()
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[bold cyan]Profile:[/bold cyan] {self.profile_path}")
        if self._kernels:
            log.write(
                f"[bold cyan]{len(self._kernels)} kernels loaded.[/bold cyan] "
                "Ask anything about this GPU profile."
            )
        else:
            log.write("[yellow]No kernels found. Profile may be empty or incompatible.[/yellow]")
        log.write("[dim]─────────────────────────────────────────[/dim]")

    # ── Setup helpers ─────────────────────────────────────────────────────────

    def _setup_table_columns(self) -> None:
        table = self.query_one("#kernel-table", DataTable)
        table.add_columns("#", "Kernel", "ms (total)", "Calls")

    def _load_and_populate_kernels(self) -> None:
        """Synchronously load kernels from the profile DB and fill the DataTable."""
        try:
            from .profile import resolve_profile_path
            sqlite_path = resolve_profile_path(self.profile_path)
            self._kernels = _load_top_kernels(sqlite_path)
        except Exception as e:
            self.query_one("#chat-log", RichLog).write(
                f"[bold red]Error loading profile:[/bold red] {e}"
            )
            return

        table = self.query_one("#kernel-table", DataTable)
        self._kernel_name_to_row = {}
        for i, k in enumerate(self._kernels):
            name = k.get("name", "")
            display = name[:28] + "…" if len(name) > 29 else name
            table.add_row(
                str(i + 1),
                display,
                f"{k.get('total_ms', 0):.1f}",
                str(k.get("count", 0)),
            )
            self._kernel_name_to_row[name.lower()] = i

    # ── Input handler ─────────────────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_msg = event.value.strip()
        if not user_msg or self._is_generating:
            return
        event.input.value = ""
        self._add_user_turn(user_msg)
        self._is_generating = True
        self._streaming_buffer = []
        self._last_nav_message = ""
        # Show "AI: " prefix in RichLog; streaming chunks go to #streaming-area.
        self.query_one("#streaming-area", Static).update("[bold green]AI:[/bold green] …")
        self._run_stream_worker(user_msg)

    # ── Main-thread UI callbacks (called via call_from_thread) ────────────────

    def _add_user_turn(self, text: str) -> None:
        """Write user message to RichLog and append to history."""
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[bold yellow]You:[/bold yellow] {text}")
        self._chat_messages.append({"role": "user", "content": text})

    def _on_text_chunk(self, chunk: str) -> None:
        """Append streaming text chunk to the Static display area."""
        self._streaming_buffer.append(chunk)
        current = "".join(self._streaming_buffer)
        self.query_one("#streaming-area", Static).update(
            f"[bold green]AI:[/bold green] {current}"
        )

    def _on_system_event(self, content: str) -> None:
        """Display a system/status message in the chat log."""
        self.query_one("#chat-log", RichLog).write(f"[dim]{content}[/dim]")

    def _on_action_navigate(self, target_name: str, reason: str | None) -> None:
        """Move DataTable cursor to the kernel matching target_name.

        Does NOT write to the chat log directly — the message is stored in
        _last_nav_message and displayed by _on_stream_done to avoid duplication.
        """
        table = self.query_one("#kernel-table", DataTable)
        target_lower = target_name.lower()
        # Substring match — LLM names may differ slightly from short names.
        matched_row = next(
            (
                row
                for name, row in self._kernel_name_to_row.items()
                if target_lower in name or name in target_lower
            ),
            None,
        )
        if matched_row is not None:
            table.move_cursor(row=matched_row, animate=False)
            table.scroll_visible()
            kernel_info = self._kernels[matched_row] if matched_row < len(self._kernels) else {}
            gpu_ms = kernel_info.get("total_ms", 0)
            calls = kernel_info.get("count", 0)
            avg_ms = gpu_ms / calls if calls else 0
            msg = f"⚡ → [bold]{target_name}[/bold] (row {matched_row + 1}, {gpu_ms:.1f}ms, {calls} calls)"
            # Update the info bar with kernel details.
            self._update_kernel_info_bar(target_name, gpu_ms, calls, avg_ms)
        else:
            msg = f"⚠ Kernel not found: {target_name}"
        if reason:
            msg += f" — {reason}"
        self._last_nav_message = msg
        self.post_message(self.NavigateToKernel(target_name, reason))

    def _update_kernel_info_bar(self, name: str, total_ms: float, calls: int, avg_ms: float) -> None:
        """Update the selected kernel info bar below the DataTable."""
        bar = self.query_one("#kernel-info-bar", Static)
        bar.update(
            f"▶ [bold]{name}[/bold]  │  [magenta]{total_ms:,.1f}ms total[/magenta]  │  "
            f"[cyan]{calls} calls[/cyan]  │  avg [yellow]{avg_ms:.2f}ms[/yellow]"
        )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Update the info bar when the user manually selects a DataTable row."""
        row_idx = event.cursor_row
        if 0 <= row_idx < len(self._kernels):
            k = self._kernels[row_idx]
            name = k.get("name", "")
            total_ms = k.get("total_ms", 0)
            calls = k.get("count", 0)
            avg_ms = total_ms / calls if calls else 0
            self._update_kernel_info_bar(name, total_ms, calls, avg_ms)

    def _on_stream_done(self, final_content: str) -> None:
        """Finalize the streaming response: move to RichLog, unlock input.

        Display logic (mirrors web frontend):
        - text + nav  → AI bubble = text, plus a dim nav note below
        - nav only    → AI bubble = nav message
        - text only   → AI bubble = text
        """
        log = self.query_one("#chat-log", RichLog)
        if final_content and self._last_nav_message:
            # Both text and navigation
            log.write(f"[bold green]AI:[/bold green] {final_content}")
            log.write(f"[dim]{self._last_nav_message}[/dim]")
            self._chat_messages.append({"role": "assistant", "content": final_content})
        elif final_content:
            # Text only (e.g. "tell me the slowest kernel")
            log.write(f"[bold green]AI:[/bold green] {final_content}")
            self._chat_messages.append({"role": "assistant", "content": final_content})
        elif self._last_nav_message:
            # Pure navigation, no text — nav message becomes the response
            log.write(f"[bold green]AI:[/bold green] {self._last_nav_message}")
            self._chat_messages.append({"role": "assistant", "content": self._last_nav_message})
        # else: truly empty (rare edge case)
        log.write("")
        self.query_one("#streaming-area", Static).update("")
        self._streaming_buffer = []
        self._last_nav_message = ""
        self._is_generating = False

    # ── Background worker ─────────────────────────────────────────────────────

    @work(thread=True, exclusive=True)
    def _run_stream_worker(self, user_msg: str) -> None:
        """Background thread: calls stream_agent_loop, posts UI updates via call_from_thread.

        Imports chat module lazily so the rest of the app works without litellm installed.
        """
        try:
            from .chat import _get_model_and_key, distill_history, stream_agent_loop
        except ImportError as e:
            self.call_from_thread(
                self._on_system_event, f"chat module unavailable: {e}"
            )
            self.call_from_thread(self._on_stream_done, "")
            return

        model, _ = _get_model_and_key()
        if not model:
            self.call_from_thread(
                self._on_system_event,
                "No LLM configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY.",
            )
            self.call_from_thread(self._on_stream_done, "")
            return

        # Build a lightweight ui_context so the model knows which profile it's analyzing.
        ui_context: dict = {
            "profile": self.profile_path,
            "global_top_kernels": [
                {
                    "name": k.get("name", ""),
                    "total_ms": round(k.get("total_ms", 0), 2),
                    "count": k.get("count", 0),
                }
                for k in self._kernels[:10]
            ],
        }

        # Snapshot history so the worker sees a consistent view (read-only copy).
        messages = list(self._chat_messages)
        accumulated: list[str] = []

        try:
            for event in stream_agent_loop(
                model=model,
                messages=messages,
                ui_context=ui_context,
                profile_path=self.profile_path,
                max_turns=5,
            ):
                etype = event.get("type")
                if etype == "text":
                    chunk = event.get("content", "")
                    if chunk:
                        accumulated.append(chunk)
                        self.call_from_thread(self._on_text_chunk, chunk)
                elif etype == "system":
                    self.call_from_thread(
                        self._on_system_event, event.get("content", "")
                    )
                elif etype == "action":
                    action = event.get("action", {})
                    atype = action.get("type")
                    if atype == "navigate_to_kernel":
                        self.call_from_thread(
                            self._on_action_navigate,
                            action.get("target_name", ""),
                            action.get("reason"),
                        )
                    elif atype == "zoom_to_time_range":
                        start_s = action.get("start_s", 0)
                        end_s = action.get("end_s", 0)
                        self.call_from_thread(
                            self._on_system_event,
                            f"🔍 Zoom: {start_s:.3f}s – {end_s:.3f}s",
                        )
                # "done" type is handled by loop termination.
        except Exception as e:
            self.call_from_thread(self._on_system_event, f"Stream error: {e}")

        final_content = "".join(accumulated)
        self.call_from_thread(self._on_stream_done, final_content)

        # Distill history to compress tool call/result sequences for lean context (§11.7).
        try:
            self._chat_messages[:] = distill_history(self._chat_messages)
        except Exception:
            pass  # Non-critical; don't break the UI if distillation fails.

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_clear_chat(self) -> None:
        """Clear the chat history and the log display."""
        self._chat_messages.clear()
        self.query_one("#chat-log", RichLog).clear()
        self.query_one("#streaming-area", Static).update("")
        self._on_system_event("Chat cleared.")

    def action_cancel_generation(self) -> None:
        """Cancel the current AI generation if running (Ctrl+C)."""
        if self._is_generating:
            self.workers.cancel_all()
            self._is_generating = False
            self._streaming_buffer = []
            self.query_one("#streaming-area", Static).update("")
            self._on_system_event("⚠ Generation cancelled.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_chat_tui(profile_path: str) -> None:
    """Launch the Textual AI chat TUI for the given profile path."""
    NsysChatApp(profile_path).run()
