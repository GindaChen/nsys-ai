"""Session handoff tests for the shared tree/timeline ChatPanel."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from textual.app import App, ComposeResult

from nsys_ai.profile_runner import build_local_profile_reference
from nsys_ai.session_store import SessionStore
from nsys_ai.tree.chat import ChatPanel

PROFILE = Path(__file__).parent / "fixtures" / "h100_2gpu_1s.sqlite"


class _ChatHost(App):
    def __init__(self, panel: ChatPanel) -> None:
        super().__init__()
        self.panel = panel

    def compose(self) -> ComposeResult:
        yield self.panel


async def _until(pilot, predicate, limit: int = 300) -> bool:
    for _ in range(limit):
        if predicate():
            return True
        await pilot.pause()
    return predicate()


@pytest.mark.asyncio
async def test_chat_panel_persists_completed_runner_handoff(tmp_path, monkeypatch):
    """A completed TUI turn writes the same ask record as Web transports."""
    from nsys_ai import chat as chat_mod

    session_root = tmp_path / "sessions"
    SessionStore(session_root).create(
        "tui-chat-001",
        before_profile=build_local_profile_reference(str(PROFILE.absolute())),
    )

    captured: dict = {}

    def fake_stream(**kwargs):
        captured.update(kwargs)
        yield {"type": "text", "content": "Grounded answer"}
        yield {
            "type": "done",
            "selected_skills": ["root_cause_matcher", "top_kernels"],
            "evidence": {"top_kernels": [{"name": "gemm"}]},
        }

    monkeypatch.setattr(chat_mod, "get_default_model", lambda: "test/model")
    monkeypatch.setattr(chat_mod, "stream_agent_loop", fake_stream)

    panel = ChatPanel(
        db_path=str(PROFILE.absolute()),
        session_id="tui-chat-001",
        session_root=str(session_root),
    )
    app = _ChatHost(panel)

    async with app.run_test(size=(100, 20)) as pilot:
        field = panel.query_one("#chat-input")
        field.focus()
        field.value = "why is this slow?"
        await pilot.press("enter")
        assert await _until(pilot, lambda: bool(captured))
        assert await _until(pilot, lambda: not panel.is_running)

    record_path = session_root / "tui-chat-001" / "logs" / "ask.jsonl"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["kind"] == "ask"
    assert record["question"] == "why is this slow?"
    assert record["answer"] == "Grounded answer"
    assert record["selected_skills"] == ["root_cause_matcher", "top_kernels"]
    assert record["evidence"]["top_kernels"][0]["name"] == "gemm"
    assert captured["prefill_evidence"] is True


@pytest.mark.asyncio
async def test_chat_panel_does_not_persist_incomplete_turn(tmp_path, monkeypatch):
    """A cancelled/error turn must not create a misleading completed ask log."""
    from nsys_ai import chat as chat_mod

    session_root = tmp_path / "sessions"
    SessionStore(session_root).create(
        "tui-chat-002",
        before_profile=build_local_profile_reference(str(PROFILE.absolute())),
    )

    calls: list[dict] = []

    def incomplete_stream(**kwargs):
        calls.append(kwargs)
        yield {"type": "text", "content": "partial"}

    monkeypatch.setattr(chat_mod, "get_default_model", lambda: "test/model")
    monkeypatch.setattr(chat_mod, "stream_agent_loop", incomplete_stream)

    panel = ChatPanel(
        db_path=str(PROFILE.absolute()),
        session_id="tui-chat-002",
        session_root=str(session_root),
    )
    app = _ChatHost(panel)

    async with app.run_test(size=(100, 20)) as pilot:
        field = panel.query_one("#chat-input")
        field.focus()
        field.value = "why is this slow?"
        await pilot.press("enter")
        assert await _until(pilot, lambda: bool(calls))
        assert await _until(pilot, lambda: not panel.is_running)

    assert not (session_root / "tui-chat-002" / "logs" / "ask.jsonl").exists()
