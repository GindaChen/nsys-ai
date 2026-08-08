"""Headless tests: Textual apps stay responsive across a cache build (#332).

A callback that only fires is not enough — that passed when the build still
ran on the message loop. These assert the UI updates *between* progress
steps while the worker is still parked inside the build.
"""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

import pytest

from nsys_ai import parquet_cache
from nsys_ai.tui_cache import format_cache_progress

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"
# Tree/timeline treat (0, 0) as the UI "full" sentinel but Profile.kernels
# applies it as a real window; CLI always passes a concrete --trim range.
FULL_TRIM = (0, 10**18)


@pytest.fixture
def profile_copy(tmp_path, monkeypatch):
    """Cold profile copy in an isolated cwd (session store + cache stay local)."""
    dest = tmp_path / FIXTURE.name
    shutil.copy(FIXTURE, dest)
    monkeypatch.chdir(tmp_path)
    return str(dest)


async def _until(pilot, pred, limit: int = 400) -> bool:
    """Pump the app until *pred* holds. Must use pilot.pause(), not asyncio.sleep."""
    for _ in range(limit):
        if pred():
            return True
        await pilot.pause()
    return pred()


def _install_gated_build(monkeypatch, app):
    """Each progress step parks until the test releases it.

    Raises immediately if the build is still on the message-loop thread, so a
    synchronous ``on_mount`` cannot deadlock the headless driver.
    """
    parked: dict[int, threading.Event] = {}
    gates: dict[int, threading.Event] = {}
    build_done = threading.Event()
    real_build = parquet_cache.build_cache

    def gated_build(sqlite_path, *, progress=None, env_escape=True):
        try:
            if progress is not None:
                for step in (1, 2, 3):
                    progress(f"gate-{step}.parquet", step, 3)
                    parked.setdefault(step, threading.Event()).set()
                    if threading.get_ident() == app._thread_id:
                        raise AssertionError(
                            "cache build is running on the Textual message loop — "
                            "progress cannot be processed until it finishes"
                        )
                    gate = gates.setdefault(step, threading.Event())
                    if not gate.wait(timeout=10):
                        raise TimeoutError(f"test never released cache-build gate {step}")
            return real_build(sqlite_path, progress=progress, env_escape=env_escape)
        finally:
            build_done.set()

    monkeypatch.setattr(parquet_cache, "build_cache", gated_build)
    return parked, gates, build_done


def _release_gates(gates: dict[int, threading.Event], n: int = 8) -> None:
    for step in range(1, n + 1):
        gates.setdefault(step, threading.Event()).set()


async def _assert_responsive(pilot, app, parked, gates, build_done, loaded_pred):
    assert await _until(pilot, lambda: parked.get(1) is not None and parked[1].is_set()), (
        "worker never reached the first gated progress step"
    )
    expected = format_cache_progress("gate-1.parquet", 1, 3)
    assert await _until(pilot, lambda: app.sub_title == expected), (
        f"UI did not show step-1 progress while the build was parked; "
        f"sub_title={app.sub_title!r}"
    )
    assert not build_done.is_set(), (
        "build finished before the UI showed progress — still synchronous on the message loop"
    )
    _release_gates(gates)
    assert await _until(pilot, lambda: build_done.is_set())
    assert await _until(pilot, loaded_pred)


@pytest.mark.asyncio
async def test_chat_stays_responsive_across_cache_build(profile_copy, monkeypatch):
    from nsys_ai.tui_textual import NsysChatApp

    app = NsysChatApp(profile_copy)
    parked, gates, build_done = _install_gated_build(monkeypatch, app)
    async with app.run_test(size=(120, 40)) as pilot:
        await _assert_responsive(
            pilot, app, parked, gates, build_done, lambda: bool(app._kernels)
        )


@pytest.mark.asyncio
async def test_tree_stays_responsive_across_cache_build(profile_copy, monkeypatch):
    from nsys_ai.tree.app import NsysTreeApp

    app = NsysTreeApp(db_path=profile_copy, device=0, trim=FULL_TRIM)
    parked, gates, build_done = _install_gated_build(monkeypatch, app)
    async with app.run_test(size=(120, 40)) as pilot:
        await _assert_responsive(
            pilot, app, parked, gates, build_done, lambda: bool(app._json_roots)
        )


@pytest.mark.asyncio
async def test_timeline_stays_responsive_across_cache_build(profile_copy, monkeypatch):
    from nsys_ai.timeline.app import NsysTimelineApp

    app = NsysTimelineApp(db_path=profile_copy, device=0, trim=FULL_TRIM)
    parked, gates, build_done = _install_gated_build(monkeypatch, app)
    async with app.run_test(size=(120, 40)) as pilot:
        await _assert_responsive(
            pilot, app, parked, gates, build_done, lambda: bool(app._kernels)
        )


@pytest.mark.asyncio
async def test_tree_cache_build_writes_no_cr_to_piped_stderr(profile_copy, tmp_path):
    from nsys_ai.tree.app import NsysTreeApp

    err_path = tmp_path / "tree-stderr.bin"
    err_fd = os.open(str(err_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    saved = os.dup(2)
    try:
        os.dup2(err_fd, 2)
        os.close(err_fd)
        app = NsysTreeApp(db_path=profile_copy, device=0, trim=FULL_TRIM)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await _until(pilot, lambda: bool(app._json_roots))
    finally:
        os.dup2(saved, 2)
        os.close(saved)

    data = err_path.read_bytes()
    assert b"\r" not in data, (
        f"carriage returns reached piped stderr during a tree TUI cache build: {data!r}"
    )


@pytest.mark.asyncio
async def test_timeline_cache_build_writes_no_cr_to_piped_stderr(profile_copy, tmp_path):
    from nsys_ai.timeline.app import NsysTimelineApp

    err_path = tmp_path / "timeline-stderr.bin"
    err_fd = os.open(str(err_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    saved = os.dup(2)
    try:
        os.dup2(err_fd, 2)
        os.close(err_fd)
        app = NsysTimelineApp(db_path=profile_copy, device=0, trim=FULL_TRIM)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await _until(pilot, lambda: bool(app._kernels))
    finally:
        os.dup2(saved, 2)
        os.close(saved)

    data = err_path.read_bytes()
    assert b"\r" not in data, (
        f"carriage returns reached piped stderr during a timeline TUI cache build: {data!r}"
    )


async def _quit_while_build_parked(pilot, app, parked, gates, build_done):
    """Quit mid-load; the worker must stop without raising into the driver."""
    assert await _until(pilot, lambda: parked.get(1) is not None and parked[1].is_set()), (
        "worker never reached the gated progress step before quit"
    )
    assert not build_done.is_set(), "build finished before quit — still on the message loop"
    app.exit()
    _release_gates(gates)
    # Wait on build_done alone. ``app.exit()`` returns immediately but a thread
    # worker cannot be cancelled, so ``not app.is_running`` would let the test
    # return while the real DuckDB build is still writing parquet under a
    # tmp_path pytest is about to delete.
    assert await _until(pilot, lambda: build_done.is_set())


@pytest.mark.asyncio
async def test_chat_quit_mid_cache_build_does_not_raise(profile_copy, monkeypatch):
    from nsys_ai.tui_textual import NsysChatApp

    app = NsysChatApp(profile_copy)
    parked, gates, build_done = _install_gated_build(monkeypatch, app)
    async with app.run_test(size=(120, 40)) as pilot:
        await _quit_while_build_parked(pilot, app, parked, gates, build_done)


@pytest.mark.asyncio
async def test_tree_quit_mid_cache_build_does_not_raise(profile_copy, monkeypatch):
    from nsys_ai.tree.app import NsysTreeApp

    app = NsysTreeApp(db_path=profile_copy, device=0, trim=FULL_TRIM)
    parked, gates, build_done = _install_gated_build(monkeypatch, app)
    async with app.run_test(size=(120, 40)) as pilot:
        await _quit_while_build_parked(pilot, app, parked, gates, build_done)


@pytest.mark.asyncio
async def test_timeline_quit_mid_cache_build_does_not_raise(profile_copy, monkeypatch):
    from nsys_ai.timeline.app import NsysTimelineApp

    app = NsysTimelineApp(db_path=profile_copy, device=0, trim=FULL_TRIM)
    parked, gates, build_done = _install_gated_build(monkeypatch, app)
    async with app.run_test(size=(120, 40)) as pilot:
        await _quit_while_build_parked(pilot, app, parked, gates, build_done)


@pytest.mark.asyncio
async def test_tree_row_highlight_after_teardown_does_not_raise(profile_copy):
    """RowHighlighted dispatched once the app has stopped must not query the DOM.

    Populating the table posts a RowHighlighted that bubbles table → … → screen
    → app, one queue hop per pump. ``App._shutdown`` clears ``_running`` first,
    then prunes the screen while the app's loop keeps draining, so a highlight
    still in flight at quit reaches the handler with ``#tree-table`` gone. This
    was observed intermittently in the responsiveness tests above; asserting it
    on a real post-teardown app makes it deterministic.

    ``dt.is_mounted`` below is the reason this is gated on the *app*: Textual
    sets ``Widget._is_mounted`` True on mount and never back to False, so the
    control's own flag reads True here and cannot gate anything.
    """
    from textual.css.query import NoMatches
    from textual.widgets import DataTable

    from nsys_ai.tree.app import NsysTreeApp
    from nsys_ai.tree.widgets import TreeTable

    app = NsysTreeApp(db_path=profile_copy, device=0, trim=FULL_TRIM)
    async with app.run_test(size=(120, 40)) as pilot:
        assert await _until(pilot, lambda: bool(app._json_roots))
        dt = app.query_one("#tree-table", TreeTable).query_one(DataTable)

    # Teardown is done: the app has stopped and the tree is off the DOM.
    assert not app.is_running
    assert dt.is_mounted, "Widget.is_mounted never returns to False — it cannot gate teardown"
    with pytest.raises(NoMatches):
        app.query_one("#tree-table", TreeTable)

    # The handler must swallow this on the lifecycle check, not on a try/except
    # around query_one — a live #tree-table id regression still has to raise.
    app._on_row_highlighted(DataTable.RowHighlighted(dt, row_key=None, cursor_row=0))


@pytest.mark.asyncio
async def test_chat_row_highlight_after_teardown_does_not_raise(profile_copy):
    """Same teardown race in the chat app's kernel-info bar (seen under CPU load)."""
    from textual.css.query import NoMatches
    from textual.widgets import DataTable, Static

    from nsys_ai.tui_textual import NsysChatApp

    app = NsysChatApp(profile_copy)
    async with app.run_test(size=(120, 40)) as pilot:
        assert await _until(pilot, lambda: bool(app._kernels))
        dt = app.query_one(DataTable)

    assert not app.is_running
    assert dt.is_mounted, "Widget.is_mounted never returns to False — it cannot gate teardown"
    with pytest.raises(NoMatches):
        app.query_one("#kernel-info-bar", Static)

    app.on_data_table_row_highlighted(DataTable.RowHighlighted(dt, row_key=None, cursor_row=0))
