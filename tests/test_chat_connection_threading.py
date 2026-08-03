"""The chat path's DuckDB handle belongs to the thread that drives the generator.

`Profile.query_conn()` exists because a shared DuckDB handle serves concurrent
queries wrong rows with nothing raised. The chat path does not use it and does
not need to: `_prepare_session` calls `open_profile_readonly`, which returns a
fresh `duckdb.connect()` — a private in-memory database, not a cursor on a
shared one — and `_prepare_session` runs inside the `stream_agent_loop`
generator body, so the connection is created on whichever thread advances the
generator and closed in that generator's `finally`.

Two ways a future change could break that, both of which these tests catch:
memoizing `open_profile_readonly` per path (one handle shared by every session),
or hoisting the session setup out of the generator (the connection then belongs
to the thread that *built* the generator, not the one that drains it).

Neither is hypothetical protection against an imaginary caller. A cancelled
Textual `@work(thread=True, exclusive=True)` worker keeps running — cancellation
sets a flag, it does not stop the OS thread — so two chat turns really do
overlap, and only the per-invocation connection makes that harmless.
"""

import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nsys_ai import chat as chat_mod

# Imported for its side effect: `stream_agent_loop` imports `tool_dispatch`
# lazily, and six threads racing to execute that import for the first time is
# not what these tests are measuring.
from nsys_ai import tool_dispatch as _tool_dispatch  # noqa: F401

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"

CONCURRENT_SESSIONS = 6


@pytest.fixture(scope="module")
def profile_copy(tmp_path_factory):
    """A private copy of the fixture, with its Parquet cache already built.

    The copy keeps `tests/fixtures/` free of `.nsys-cache` directories; the
    pre-warm keeps cache construction out of the concurrent section.
    """
    dst = tmp_path_factory.mktemp("chat_conn") / "profile.sqlite"
    shutil.copy(FIXTURE, dst)
    path = str(dst)
    chat_mod.open_profile_readonly(path).close()
    return path


@pytest.fixture
def fake_litellm(monkeypatch):
    """A litellm whose stream yields one text chunk and no tool calls.

    Installed once, on the main thread. The `patch.dict(sys.modules, ...)`
    idiom used elsewhere in the chat tests races with itself when entered from
    several threads and fails with a spurious ImportError.
    """
    fake = MagicMock()

    def completion(**_kwargs):
        delta = MagicMock()
        delta.content = "ok"
        delta.tool_calls = []
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=delta)]
        chunk.usage = None
        return iter([chunk])

    fake.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return fake


def _drain(profile_path):
    for _ in chat_mod.stream_agent_loop(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        ui_context={},
        profile_path=profile_path,
        max_turns=1,
    ):
        pass


def test_each_concurrent_session_gets_its_own_connection(
    profile_copy, fake_litellm, monkeypatch
):
    """Overlapping chat sessions must not share one DuckDB handle.

    Every connection is held open until all of them exist, so this measures
    simultaneous sessions rather than sequential ones that happen to reuse an
    address.
    """
    seen = []
    lock = threading.Lock()
    real_open = chat_mod.open_profile_readonly
    all_open = threading.Barrier(CONCURRENT_SESSIONS)

    def recording_open(path):
        conn = real_open(path)
        with lock:
            seen.append((threading.get_ident(), conn))
        try:
            all_open.wait(timeout=60)
        except threading.BrokenBarrierError:
            # A session never reached this point; let the assertions below say so
            # instead of leaking the connection this call already opened.
            pass
        return conn

    monkeypatch.setattr(chat_mod, "open_profile_readonly", recording_open)

    with ThreadPoolExecutor(max_workers=CONCURRENT_SESSIONS) as pool:
        for fut in [pool.submit(_drain, profile_copy) for _ in range(CONCURRENT_SESSIONS)]:
            fut.result()

    assert len(seen) == CONCURRENT_SESSIONS, seen
    assert len({id(conn) for _, conn in seen}) == CONCURRENT_SESSIONS, (
        "connection object shared across concurrent chat sessions"
    )
    assert len({tid for tid, _ in seen}) == CONCURRENT_SESSIONS, (
        "sessions did not actually run on distinct threads"
    )


def test_connection_opens_on_the_thread_that_iterates(profile_copy, fake_litellm, monkeypatch):
    """The session is set up in the generator body, not when the generator is built.

    Textual and the web server both construct the generator on one thread and
    drain it on another; the connection must belong to the draining thread.
    """
    opened_on = []
    real_open = chat_mod.open_profile_readonly

    def recording_open(path):
        opened_on.append(threading.get_ident())
        return real_open(path)

    monkeypatch.setattr(chat_mod, "open_profile_readonly", recording_open)

    gen = chat_mod.stream_agent_loop(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        ui_context={},
        profile_path=profile_copy,
        max_turns=1,
    )
    assert opened_on == [], "generator opened a connection before it was iterated"

    drained_on = []

    def consume():
        drained_on.append(threading.get_ident())
        for _ in gen:
            pass

    thread = threading.Thread(target=consume)
    thread.start()
    thread.join(timeout=60)

    assert not thread.is_alive()
    assert opened_on == drained_on
    assert opened_on[0] != threading.get_ident()
