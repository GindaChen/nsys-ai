"""
tree/chat.py — Shared ChatPanel widget for Textual tree + timeline TUIs.

Encapsulates the AI streaming chat UI:
  - RichLog: completed conversation turns
  - Static: in-progress streaming token area
  - Input: user message box

Usage:
    class MyApp(App):
        def compose(self):
            yield ChatPanel(db_path=..., device=..., ui_context_fn=...)
"""
from __future__ import annotations

import threading
from collections.abc import Callable

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, RichLog, Static


class ChatPanel(Widget):
    """Reusable AI chat panel for Textual TUIs.

    Integration points:
      - ``db_path``         : path passed to chat backend for DB skills
      - ``device``          : GPU index
      - ``ui_context_fn``   : callable() -> dict  (snapshot of current UI state)
      - ``on_action_fn``    : callable(action_dict)  (navigate/zoom callbacks)
    """

    DEFAULT_CSS = """
    ChatPanel {
        height: 12;
        border-top: solid $primary-darken-2;
        background: $surface-darken-1;
    }
    ChatPanel > #chat-log {
        height: 1fr;
        background: $surface-darken-1;
        border: none;
    }
    ChatPanel > #chat-stream {
        height: 1;
        color: $text-muted;
        background: $surface-darken-1;
        padding: 0 1;
    }
    ChatPanel > #chat-input {
        dock: bottom;
        height: 3;
    }
    """

    is_running: reactive[bool] = reactive(False)

    def __init__(
        self,
        db_path: str = "",
        device: int = 0,
        ui_context_fn: Callable[[], dict] | None = None,
        on_action_fn: Callable[[dict], None] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._db_path = db_path
        self._device = device
        self._ui_context_fn = ui_context_fn or (lambda: {})
        self._on_action_fn = on_action_fn or (lambda _: None)
        self._history: list[dict] = []
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    BINDINGS = [("escape", "close_panel", "Close chat")]

    def compose(self) -> ComposeResult:
        yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)
        yield Static("", id="chat-stream")
        yield Input(placeholder="Ask AI… (Enter to send, Esc to close)", id="chat-input")

    def action_close_panel(self) -> None:
        """Close the chat panel and restore focus to the main widget."""
        self.remove_class("-active")
        # Try to re-focus a sensible main widget
        from textual.widgets import DataTable
        try:
            self.app.query_one(DataTable).focus()
        except Exception:
            self.app.set_focus(None)

    # Don't auto-focus the input on mount — ChatPanel starts hidden (display:none)
    # and auto-focusing would steal key presses from the rest of the app.
    # Focus is set explicitly in NsysTreeApp/NsysTimelineApp.action_toggle_chat().

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()
        if not text or self.is_running:
            return
        self._add_user_turn(text)
        self.is_running = True
        self._cancel_event.clear()
        t = threading.Thread(
            target=self._stream_worker, args=(text,), daemon=True
        )
        t.start()

    def action_cancel(self) -> None:
        """Cancel in-flight AI generation."""
        self._cancel_event.set()

    # ------------------------------------------------------------------
    # UI helpers (must be called from main thread)
    # ------------------------------------------------------------------

    def _add_user_turn(self, text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[bold cyan]You:[/bold cyan] {text}")
        with self._lock:
            self._history.append({"role": "user", "content": text})
            if len(self._history) > 50:
                self._history = self._history[-50:]

    def _on_text_chunk(self, chunk: str) -> None:
        stream_label = self.query_one("#chat-stream", Static)
        if not hasattr(self, "_stream_buffer"):
            self._stream_buffer = ""
        self._stream_buffer += chunk
        stream_label.update(self._stream_buffer)

    def _on_system_event(self, content: str) -> None:
        self.query_one("#chat-log", RichLog).write(
            f"[dim italic]{content}[/dim italic]"
        )

    def _on_stream_done(self, final_content: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        stream_label = self.query_one("#chat-stream", Static)
        if final_content:
            log.write(f"[bold green]AI:[/bold green] {final_content}")
        stream_label.update("")
        self._stream_buffer = ""
        with self._lock:
            if final_content:
                self._history.append({"role": "assistant", "content": final_content})
                if len(self._history) > 50:
                    self._history = self._history[-50:]
        self.is_running = False

    # ------------------------------------------------------------------
    # Background streaming worker
    # ------------------------------------------------------------------

    def _stream_worker(self, user_msg: str) -> None:
        """Run in background thread; posts UI updates via call_from_thread."""
        try:
            from .. import chat as chat_mod
        except ImportError:
            self.app.call_from_thread(
                self._on_system_event,
                "litellm not installed — pip install 'nsys-ai[chat]'",
            )
            self.app.call_from_thread(self._on_stream_done, "")
            return

        try:
            model = chat_mod.get_default_model()
        except Exception:
            model = None

        if not model:
            self.app.call_from_thread(
                self._on_system_event,
                "LLM not configured (no API key found).",
            )
            self.app.call_from_thread(self._on_stream_done, "")
            return

        with self._lock:
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in self._history[-10:]
            ]

        ui_context = self._ui_context_fn()

        content = ""
        actions: list[dict] = []
        try:
            stream = chat_mod.stream_agent_loop(
                model=model,
                messages=history,
                ui_context=ui_context,
                tools=chat_mod._tools_openai(),
                profile_path=self._db_path,
                max_turns=5,
            )
            for ev in stream:
                if self._cancel_event.is_set():
                    break
                t = ev.get("type")
                if t == "text":
                    chunk = ev.get("content", "")
                    content += chunk
                    self.app.call_from_thread(self._on_text_chunk, chunk)
                elif t == "system":
                    self.app.call_from_thread(
                        self._on_system_event, (ev.get("content") or "")[:80]
                    )
                elif t == "action":
                    act = ev.get("action")
                    if act:
                        actions.append(act)
                elif t == "done":
                    break
        except Exception as e:
            content = f"Error: {e}"

        # Distill history
        try:
            from .. import chat as chat_mod  # re-import for distill
            with self._lock:
                self._history[:] = chat_mod.distill_history(self._history)
        except Exception:
            pass

        self.app.call_from_thread(self._on_stream_done, content)

        # Dispatch navigation / zoom actions back onto the Textual app thread.
        for action in actions:
            try:
                self.app.call_from_thread(self._on_action_fn, action)
            except Exception:
                pass
