"""
tests/test_tui_timeline_app.py — Pilot-based headless tests for NsysTimelineApp.

Run with: pytest tests/test_tui_timeline_app.py -v
"""
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
        await pilot.press("equals_sign")   # = key (Textual 8: "equals_sign", not "equal")
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
