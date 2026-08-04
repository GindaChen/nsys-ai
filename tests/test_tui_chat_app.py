"""Tests for the Textual chat TUI (src/nsys_ai/tui_textual.py).

Issue #311: Ctrl+C (and an exclusive supersession by a new submit) only *marks*
a Textual thread worker cancelled — it cannot interrupt the OS thread. The
worker has to stop itself. These tests drive a real NsysChatApp with a fake
stream and assert that a cancelled turn stops writing to the UI and to the
LLM history.

The fake stream is gate-driven, never sleep-driven, so the tests are
deterministic. Note that `_until` must pump the app with `await pilot.pause()`:
awaiting `asyncio.sleep` instead lets a whole turn's `call_from_thread`
callbacks arrive in one batch, which makes the assertions vacuous.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"


@pytest.fixture
def profile_copy(tmp_path):
    """A private copy of the profile — on_mount builds a .nsys-cache beside it."""
    dest = tmp_path / FIXTURE.name
    shutil.copy(FIXTURE, dest)
    return str(dest)


class _GatedStream:
    """Fake stream_agent_loop: yields one chunk, parks on a gate, then yields more.

    Labels every chunk with the user message that produced it (``<A0>``,
    ``<B3>``, …) so a turn's output can be attributed unambiguously.
    """

    def __init__(self) -> None:
        self.gates: dict[str, threading.Event] = {}
        self.parked: dict[str, threading.Event] = {}
        self.finished: dict[str, threading.Event] = {}
        # Chunks the generator actually produced, i.e. what the worker pulled.
        # Distinct from what reached the UI: this pins that the worker stopped
        # *reading* the stream, not merely that its UI writes were dropped.
        self.produced: list[str] = []

    def gate(self, label: str) -> threading.Event:
        return self.gates.setdefault(label, threading.Event())

    def parked_at(self, label: str) -> threading.Event:
        return self.parked.setdefault(label, threading.Event())

    def finished_at(self, label: str) -> threading.Event:
        """Set when the generator frame ends — exhausted, closed, or GC'd."""
        return self.finished.setdefault(label, threading.Event())

    def __call__(self, model, messages, ui_context, profile_path=None, max_turns=5, **kw):
        label = messages[-1]["content"] if messages else "?"
        try:
            self.produced.append(f"<{label}0>")
            yield {"type": "text", "content": f"<{label}0>"}
            self.parked_at(label).set()
            # Park until the test opens the gate. Generous timeout: this is a
            # deadlock backstop, not a synchronisation mechanism.
            self.gate(label).wait(10)
            for i in range(1, 6):
                self.produced.append(f"<{label}{i}>")
                yield {"type": "text", "content": f"<{label}{i}>"}
            yield {"type": "done"}
        finally:
            # Lets a test wait for the worker to be done with this stream instead
            # of sleeping. The worker's `finally: stream.close()` throws
            # GeneratorExit in here, so this fires on the cancelled path too.
            self.finished_at(label).set()


async def _until(pilot, pred, limit: int = 400) -> bool:
    """Pump the app until *pred* holds. Must use pilot.pause(), not asyncio.sleep."""
    for _ in range(limit):
        if pred():
            return True
        await pilot.pause()
    return pred()


def _spy_on_chunks(app, seen: list[str]) -> None:
    orig = app._on_text_chunk

    def spy(chunk):
        seen.append(chunk)
        return orig(chunk)

    app._on_text_chunk = spy


@pytest.fixture
def fake_stream(monkeypatch):
    import nsys_ai.chat as chat_mod

    stream = _GatedStream()
    monkeypatch.setattr(chat_mod, "stream_agent_loop", stream)
    monkeypatch.setattr(chat_mod, "_get_model_and_key", lambda *a, **k: ("fake/model", "key"))
    return stream


@pytest.mark.asyncio
async def test_cancelled_turn_stops_writing(profile_copy, fake_stream):
    """Ctrl+C must stop the worker writing text, history, and the submit guard."""
    from nsys_ai.tui_textual import NsysChatApp

    app = NsysChatApp(profile_copy)
    seen: list[str] = []
    _spy_on_chunks(app, seen)

    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#chat-input")
        inp.focus()
        inp.value = "A"
        await pilot.press("enter")
        assert await _until(pilot, lambda: "<A0>" in seen), "turn A never streamed"
        assert fake_stream.parked_at("A").wait(5), "turn A generator never parked"

        app.action_cancel_generation()
        await pilot.pause()
        fake_stream.gate("A").set()  # let the abandoned worker run on

        # Wait for the abandoned worker to be *done with the stream* before asserting
        # what it did not do. A bounded pump on "<A5>" in seen would pass vacuously on
        # a slow runner -- the worker simply may not have produced yet -- which is a
        # false green rather than a false red. finished_at fires on both paths (fixed:
        # stream.close() throws GeneratorExit into the generator's finally; unfixed:
        # exhaustion does), so this is the same terminal condition either way.
        # Pumped rather than a bare Event.wait(): the worker blocks inside
        # call_from_thread until the main thread runs the callback, so a blocking wait
        # here would deadlock the unfixed path against its own dispatches.
        assert await _until(pilot, lambda: fake_stream.finished_at("A").is_set()), (
            f"cancelled worker never let go of the stream; it pulled {fake_stream.produced}"
        )

        assert "<A1>" not in seen, f"cancelled turn A kept streaming: {seen}"
        # The worker must stop *pulling* the stream, not just stop writing to the
        # UI: draining it means real LLM spend and the profile DB connection held
        # open until the generator finishes. Pins the `if cancelled(): break` in
        # the stream loop specifically, which the UI-side assertions do not.
        assert "<A3>" not in fake_stream.produced, (
            f"worker kept pulling the cancelled stream: {fake_stream.produced}"
        )
        assert not any(m["role"] == "assistant" for m in app._chat_messages), (
            f"cancelled turn A wrote an assistant turn: {app._chat_messages}"
        )
        assert app._is_generating is False

        # A replacement turn must not be polluted by the cancelled one.
        inp.focus()
        inp.value = "B"
        await pilot.press("enter")
        assert await _until(pilot, lambda: "<B0>" in seen), "turn B never streamed"
        assert fake_stream.parked_at("B").wait(5), "turn B generator never parked"
        assert app._is_generating is True, "stale worker released the submit guard mid-turn B"
        fake_stream.gate("B").set()
        assert await _until(pilot, lambda: app._is_generating is False), "turn B never finished"

        assistant = [m["content"] for m in app._chat_messages if m["role"] == "assistant"]
        assert assistant == ["<B0><B1><B2><B3><B4><B5>"], f"history polluted: {app._chat_messages}"
        assert not any(c.startswith("<A") and c != "<A0>" for c in seen), seen

        # The decision this change actually makes, pinned: the cancelled turn's
        # *question* stays in the history with no assistant reply, so two consecutive
        # user messages reach the model. That shape is unchanged by the fix -- what
        # the fix removes is A's assistant text landing after B's question. Asserting
        # the full role sequence is what catches a regression on either side; the
        # role == "assistant" filters above are blind to the user messages.
        assert [m["role"] for m in app._chat_messages] == ["user", "user", "assistant"], (
            f"unexpected history shape: {app._chat_messages}"
        )


@pytest.mark.asyncio
async def test_stale_worker_early_return_leaves_the_live_turn_alone(profile_copy, monkeypatch):
    """A worker that starts after its turn was superseded must not touch the UI.

    The two early-return paths — no chat module, no LLM configured — never reach
    the stream loop, so ``worker.is_cancelled`` is never consulted on them. Only
    the generation id the *caller* hands the worker keeps them from clearing a
    later turn's submit guard. Drives the undecorated worker body on a plain
    thread, which is what a stale worker amounts to.
    """
    import nsys_ai.chat as chat_mod

    monkeypatch.setattr(chat_mod, "_get_model_and_key", lambda *a, **k: (None, None))

    from nsys_ai.tui_textual import NsysChatApp

    app = NsysChatApp(profile_copy)

    async with app.run_test(size=(120, 40)) as pilot:
        # Turn A was submitted at generation 1; it was then cancelled (2) and a
        # replacement turn B submitted (3). Only now does A's worker body run.
        app._generation_id = 3
        app._is_generating = True

        raw = NsysChatApp._run_stream_worker.__wrapped__
        errors: list[BaseException] = []

        def run() -> None:
            try:
                raw(app, "A", 1)
            except BaseException as exc:  # recorded, then asserted on below
                errors.append(exc)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        assert await _until(pilot, lambda: not t.is_alive()), "stale worker never finished"
        t.join(5)

        # Guards the assertion below against passing vacuously: a worker that
        # never ran (e.g. because it no longer takes a generation argument)
        # cannot clear anything either.
        assert not errors, f"stale worker raised instead of returning quietly: {errors}"
        assert app._is_generating is True, "stale worker cleared the live turn's submit guard"


@pytest.mark.asyncio
async def test_superseded_turn_stops_writing(profile_copy, fake_stream):
    """A new submit while one is in flight cancels via exclusive=True — same rule.

    The submit guard blocks this in the UI, so drive the worker directly to
    exercise the supersession path itself.
    """
    from nsys_ai.tui_textual import NsysChatApp

    app = NsysChatApp(profile_copy)
    seen: list[str] = []
    _spy_on_chunks(app, seen)

    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#chat-input")
        inp.focus()
        inp.value = "A"
        await pilot.press("enter")
        assert await _until(pilot, lambda: "<A0>" in seen), "turn A never streamed"
        assert fake_stream.parked_at("A").wait(5), "turn A generator never parked"

        # Supersede A without going through the guard: this is what exclusive=True
        # does to the in-flight worker.
        app._is_generating = False
        inp.value = "B"
        await pilot.press("enter")
        assert await _until(pilot, lambda: "<B0>" in seen), "turn B never streamed"
        assert fake_stream.parked_at("B").wait(5), "turn B generator never parked"

        fake_stream.gate("A").set()  # release the superseded worker
        # Terminal condition, not a bounded guess -- see test_cancelled_turn_stops_writing.
        assert await _until(pilot, lambda: fake_stream.finished_at("A").is_set()), (
            f"superseded worker never let go of the stream; it pulled {fake_stream.produced}"
        )
        assert "<A1>" not in seen, f"superseded turn A kept streaming: {seen}"
        assert "<A3>" not in fake_stream.produced, (
            f"worker kept pulling the superseded stream: {fake_stream.produced}"
        )

        fake_stream.gate("B").set()
        assert await _until(pilot, lambda: app._is_generating is False), "turn B never finished"
        assistant = [m["content"] for m in app._chat_messages if m["role"] == "assistant"]
        assert assistant == ["<B0><B1><B2><B3><B4><B5>"], f"history polluted: {app._chat_messages}"


@pytest.mark.asyncio
async def test_quitting_the_app_stops_the_worker(profile_copy, fake_stream):
    """Quitting mid-turn must stop the worker too — and only ``is_cancelled`` can.

    Textual calls ``workers.cancel_all()`` itself when the message loop ends
    (app.py, the ``finally`` around ``_process_messages_loop``). Nothing bumps
    ``_generation_id`` on that path, so the generation guard is blind to it and
    ``worker.is_cancelled`` is the only thing that stops the worker draining the
    LLM response — and releasing the profile DB connection — into a dead app.
    This is what pins that half of ``cancelled()`` on its own.
    """
    from nsys_ai.tui_textual import NsysChatApp

    app = NsysChatApp(profile_copy)
    seen: list[str] = []
    _spy_on_chunks(app, seen)

    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#chat-input")
        inp.focus()
        inp.value = "A"
        await pilot.press("enter")
        assert await _until(pilot, lambda: "<A0>" in seen), "turn A never streamed"
        assert fake_stream.parked_at("A").wait(5), "turn A generator never parked"
        app.exit()

    # The app is down and its workers cancelled; the worker thread is still
    # parked inside the generator. Release it and let it discover that.
    assert app._generation_id == 1, "quitting must not bump the generation id"
    fake_stream.gate("A").set()
    let_go = fake_stream.finished_at("A").wait(10)

    # Two ways an is_cancelled regression shows up here, so both are asserted.
    # Measured on the unfixed code: the worker pulls the next chunk and then
    # wedges in call_from_thread on the dead message loop — it never comes back,
    # so `let_go` is the one that fires and the stream is left open behind a
    # thread that will never finish.
    assert "<A3>" not in fake_stream.produced, (
        f"worker kept pulling the stream after the app quit: {fake_stream.produced}"
    )
    assert let_go, f"worker never let go of the stream; it pulled {fake_stream.produced}"
