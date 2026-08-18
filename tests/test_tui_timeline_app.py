"""
tests/test_tui_timeline_app.py — Pilot-based headless tests for NsysTimelineApp.

Run with: pytest tests/test_tui_timeline_app.py -v
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nsys_ai.timeline.app import NsysTimelineApp

SAMPLE_JSON = [
    {
        "name": "forward",
        "type": "nvtx",
        "duration_ms": 100.0,
        "heat": 0.5,
        "stream": "0",
        "relative_pct": 100,
        "path": "forward",
        "demangled": "",
        "start_ns": 0,
        "end_ns": 100_000_000,
        "children": [
            {
                "name": "aten::mm",
                "type": "kernel",
                "duration_ms": 30.0,
                "heat": 0.9,
                "stream": "1",
                "relative_pct": 30,
                "path": "forward",
                "demangled": "at::native::matmul",
                "start_ns": 10_000_000,
                "end_ns": 40_000_000,
                "children": [],
            },
            {
                "name": "nccl_allreduce",
                "type": "kernel",
                "duration_ms": 20.0,
                "heat": 0.2,
                "stream": "2",
                "relative_pct": 20,
                "path": "forward",
                "demangled": "",
                "start_ns": 50_000_000,
                "end_ns": 70_000_000,
                "children": [],
            },
        ],
    }
]


@pytest.fixture
def timeline_app():
    return NsysTimelineApp.from_json(SAMPLE_JSON)


@pytest.mark.asyncio
async def test_timeline_app_mounts(timeline_app):
    """App mounts without error."""
    async with timeline_app.run_test(size=(120, 40)):
        assert timeline_app.is_running


@pytest.mark.asyncio
async def test_timeline_app_has_streams(timeline_app):
    """Streams are extracted from JSON."""
    async with timeline_app.run_test(size=(120, 40)):
        assert len(timeline_app._streams) == 2  # streams "1" and "2"


@pytest.mark.asyncio
async def test_zoom_in_decreases_ns_per_col(timeline_app):
    """Plus key zooms in (fewer ns per column)."""
    async with timeline_app.run_test(size=(120, 40)) as pilot:
        initial = timeline_app.ns_per_col
        await pilot.press("equals_sign")  # = key (Textual 8: "equals_sign", not "equal")
        await pilot.pause()
        assert timeline_app.ns_per_col < initial


@pytest.mark.asyncio
async def test_zoom_out_increases_ns_per_col(timeline_app):
    """Minus key zooms out (more ns per column)."""
    async with timeline_app.run_test(size=(120, 40)) as pilot:
        initial = timeline_app.ns_per_col
        await pilot.press("-")
        await pilot.pause()
        assert timeline_app.ns_per_col > initial


@pytest.mark.asyncio
async def test_pan_right_moves_cursor(timeline_app):
    """Right arrow advances cursor_ns."""
    async with timeline_app.run_test(size=(120, 40)) as pilot:
        initial_cursor = timeline_app.cursor_ns
        await pilot.press("right")
        await pilot.pause()
        assert timeline_app.cursor_ns > initial_cursor


@pytest.mark.asyncio
async def test_pan_left_clamps_at_start(timeline_app):
    """Left arrow does not go before time_start."""
    async with timeline_app.run_test(size=(120, 40)) as pilot:
        # Already at time_start; pressing left should stay clamped
        await pilot.press("left")
        await pilot.pause()
        assert timeline_app.cursor_ns >= timeline_app._time_start


@pytest.mark.asyncio
async def test_jump_start_end(timeline_app):
    """Home/End jump to trace boundaries."""
    async with timeline_app.run_test(size=(120, 40)) as pilot:
        await pilot.press("end")
        await pilot.pause()
        assert timeline_app.cursor_ns == timeline_app._time_end

        await pilot.press("home")
        await pilot.pause()
        assert timeline_app.cursor_ns == timeline_app._time_start


@pytest.mark.asyncio
async def test_stream_selection(timeline_app):
    """Down arrow increments selected stream; Up decrements."""
    async with timeline_app.run_test(size=(120, 40)) as pilot:
        initial_stream = timeline_app.selected_stream_idx
        await pilot.press("down")
        await pilot.pause()
        assert timeline_app.selected_stream_idx == initial_stream + 1

        await pilot.press("up")
        await pilot.pause()
        assert timeline_app.selected_stream_idx == initial_stream


@pytest.mark.asyncio
async def test_scroll_to_kernel_api(timeline_app):
    """scroll_to_kernel updates cursor to the kernel's start."""
    async with timeline_app.run_test(size=(120, 40)) as pilot:
        timeline_app.scroll_to_kernel("aten::mm", 1)
        await pilot.pause()
        assert timeline_app.cursor_ns == 10_000_000  # aten::mm start_ns


@pytest.mark.asyncio
async def test_zoom_to_time_range_api(timeline_app):
    """zoom_to_time_range updates cursor and ns_per_col."""
    async with timeline_app.run_test(size=(120, 40)) as pilot:
        timeline_app.zoom_to_time_range(0.01, 0.05)  # 10ms–50ms
        await pilot.pause()
        assert timeline_app.cursor_ns == 10_000_000  # start_s * 1e9


# ---------------------------------------------------------------------------
# Loop accept/reject decision recording (SessionStore path)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / "tests" / "fixtures" / "mfu_2gpu_before.sqlite"
AFTER = ROOT / "tests" / "fixtures" / "mfu_2gpu_after.sqlite"


def _subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    source = str(ROOT / "src")
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not current else f"{source}{os.pathsep}{current}"
    return environment


def _run_cli(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "nsys_ai", *args],
        cwd=cwd,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=180.0,
        check=False,
    )


def _finding_id_from_evidence_stdout(stdout: str) -> str:
    payload = json.loads(stdout)
    return next(
        finding["id"]
        for finding in payload["findings"]
        if finding.get("id") and finding.get("suggested_actions")
    )


def _publish_full_session(tmp_path: Path, session_id: str) -> None:
    from nsys_ai.runspec import RunSpec

    before = BEFORE.resolve()
    after = AFTER.resolve()
    runspec_path = tmp_path / "runspec.json"
    runspec_path.write_bytes(RunSpec(argv=("true",)).canonical_json_bytes())
    evidence = _run_cli(
        tmp_path,
        "evidence",
        "build",
        str(before),
        "--format",
        "json",
        "--gpu",
        "0",
        "--session",
        session_id,
        "--analyzers",
        "overlap_ratio",
    )
    assert evidence.returncode == 0, evidence.stderr
    finding_id = _finding_id_from_evidence_stdout(evidence.stdout)
    propose = _run_cli(
        tmp_path,
        "propose",
        "--session",
        session_id,
        "--finding-id",
        finding_id,
        "--runspec",
        str(runspec_path),
    )
    assert propose.returncode == 0, propose.stderr
    diff = _run_cli(
        tmp_path,
        "diff",
        str(before),
        str(after),
        "--gpu",
        "0",
        "--format",
        "json",
        "--no-ai",
        "--session",
        session_id,
    )
    assert diff.returncode == 0, diff.stderr


@pytest.mark.asyncio
async def test_loop_accept_before_diff_warns_without_crashing(timeline_app, tmp_path, monkeypatch):
    """Pressing accept without a session notifies instead of crashing."""
    monkeypatch.chdir(tmp_path)
    async with timeline_app.run_test(size=(120, 40)) as pilot:
        notes: list[tuple] = []
        timeline_app.notify = lambda *a, **k: notes.append((a, k))  # type: ignore[method-assign]
        timeline_app.action_loop_accept()
        await pilot.pause()
        assert timeline_app.is_running
        assert timeline_app._session_id is None
        assert not (tmp_path / "diff.json").exists()
        assert notes and any("session" in str(a).lower() for a, _ in notes)


@pytest.mark.asyncio
async def test_loop_accept_after_diff_writes_session_diff(tmp_path, monkeypatch):
    """Accept after a published session diff persists <session>/diff.json."""
    if not BEFORE.is_file() or not AFTER.is_file():
        raise FileNotFoundError(f"missing fixture profiles: {BEFORE} / {AFTER}")
    session_id = "timeline-loop-accept"
    _publish_full_session(tmp_path, session_id)
    monkeypatch.chdir(tmp_path)
    app = NsysTimelineApp(
        db_path=str(BEFORE.resolve()),
        device=0,
        trim=None,
        json_roots=SAMPLE_JSON,
        session=session_id,
    )
    async with app.run_test(size=(120, 40)) as pilot:
        app.action_loop_accept()
        await pilot.pause()
        assert app._session_projection is not None
        assert app._session_projection["decision"] == "accept"
        written = tmp_path / ".nsys-ai" / "sessions" / session_id / "diff.json"
        assert written.exists()
        assert app._session_projection["decision_path"] == str(written)


@pytest.mark.asyncio
async def test_loop_set_decision_empty_reason_uses_fallback(tmp_path, monkeypatch):
    """An agent set_decision action with the default empty reason does not crash."""
    if not BEFORE.is_file() or not AFTER.is_file():
        raise FileNotFoundError(f"missing fixture profiles: {BEFORE} / {AFTER}")
    session_id = "timeline-loop-reason"
    _publish_full_session(tmp_path, session_id)
    monkeypatch.chdir(tmp_path)
    app = NsysTimelineApp(
        db_path=str(BEFORE.resolve()),
        device=0,
        trim=None,
        json_roots=SAMPLE_JSON,
        session=session_id,
    )
    async with app.run_test(size=(120, 40)) as pilot:
        app.set_loop_decision("reject", "")
        await pilot.pause()
        assert app.is_running
        assert app._session_projection is not None
        assert app._session_projection["decision"] == "reject"
        assert app._session_projection["decision_reason"].strip()
        history = tmp_path / ".nsys-ai" / "sessions" / session_id / "decisions.json"
        assert history.exists()
        assert app._session_projection["decision_path"] == str(history)
