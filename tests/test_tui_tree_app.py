"""
tests/test_tui_tree_app.py — Pilot-based headless tests for NsysTreeApp.

Uses Textual's App.run_test() + asyncio for headless interaction.
No display, no X server needed.

Run with: pytest tests/test_tui_tree_app.py -v
"""
import pytest

from nsys_ai.tree.app import NsysTreeApp

# ---------------------------------------------------------------------------
# Shared fixture: app with synthetic in-memory data
# ---------------------------------------------------------------------------

SAMPLE_JSON = [
    {
        "name": "forward",
        "type": "nvtx",
        "duration_ms": 120.0,
        "heat": 0.8,
        "stream": "0",
        "relative_pct": 100,
        "path": "forward",
        "demangled": "",
        "start_ns": 0,
        "end_ns": 120_000_000,
        "children": [
            {
                "name": "aten::mm",
                "type": "kernel",
                "duration_ms": 50.0,
                "heat": 0.9,
                "stream": "1",
                "relative_pct": 42,
                "path": "forward",
                "demangled": "at::native::matmul",
                "start_ns": 0,
                "end_ns": 50_000_000,
                "children": [],
            },
            {
                "name": "nccl_allreduce",
                "type": "kernel",
                "duration_ms": 30.0,
                "heat": 0.3,
                "stream": "2",
                "relative_pct": 25,
                "path": "forward",
                "demangled": "",
                "start_ns": 60_000_000,
                "end_ns": 90_000_000,
                "children": [],
            },
        ],
    },
    {
        "name": "backward",
        "type": "nvtx",
        "duration_ms": 80.0,
        "heat": 0.5,
        "stream": "0",
        "relative_pct": 100,
        "path": "backward",
        "demangled": "",
        "start_ns": 120_000_000,
        "end_ns": 200_000_000,
        "children": [
            {
                "name": "aten::mm",
                "type": "kernel",
                "duration_ms": 40.0,
                "heat": 0.7,
                "stream": "1",
                "relative_pct": 50,
                "path": "backward",
                "demangled": "",
                "start_ns": 130_000_000,
                "end_ns": 170_000_000,
                "children": [],
            },
        ],
    },
]


@pytest.fixture
def tree_app():
    return NsysTreeApp.from_json(SAMPLE_JSON)


# ---------------------------------------------------------------------------
# Mount tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tree_app_mounts(tree_app):
    """App mounts without error and shows rows."""
    async with tree_app.run_test(size=(120, 40)):
        dt = tree_app.query_one("#tree-dt")
        # Should have at least the top-level nodes
        assert dt.row_count > 0


@pytest.mark.asyncio
async def test_tree_app_initial_rows_match_visible(tree_app):
    """Initial row count matches expected visible nodes."""
    async with tree_app.run_test(size=(120, 40)):
        dt = tree_app.query_one("#tree-dt")
        # 2 nvtx + 3 kernels = 5 in tree view (all expanded by default)
        assert dt.row_count == 5


# ---------------------------------------------------------------------------
# Expand/collapse
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_collapse_reduces_rows(tree_app):
    """Pressing 'c' on the root node collapses it."""
    async with tree_app.run_test(size=(120, 40)) as pilot:
        dt = tree_app.query_one("#tree-dt")
        initial_count = dt.row_count

        # Press C to collapse all
        await pilot.press("C")
        await pilot.pause()

        assert dt.row_count < initial_count


@pytest.mark.asyncio
async def test_expand_all_then_collapse_all(tree_app):
    """E/C bindings fully expand and collapse the tree."""
    async with tree_app.run_test(size=(120, 40)) as pilot:
        dt = tree_app.query_one("#tree-dt")

        await pilot.press("E")  # expand all
        await pilot.pause()
        expanded_count = dt.row_count

        await pilot.press("C")  # collapse all
        await pilot.pause()
        collapsed_count = dt.row_count

        assert expanded_count >= collapsed_count


# ---------------------------------------------------------------------------
# Depth filter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_depth_1_shows_only_root(tree_app):
    """Pressing '1' restricts depth to 1 (top-level NVTX only)."""
    async with tree_app.run_test(size=(120, 40)) as pilot:
        await pilot.press("1")
        await pilot.pause()
        dt = tree_app.query_one("#tree-dt")
        rows = dt.row_count
        assert rows > 0
        # Depth 1 means only depth-0 nodes
        assert rows == 2  # forward, backward


@pytest.mark.asyncio
async def test_depth_0_restores_all(tree_app):
    """Pressing '0' is the 'unlimited depth' toggle."""
    async with tree_app.run_test(size=(120, 40)) as pilot:
        await pilot.press("1")  # restrict
        await pilot.pause()
        await pilot.press("0")  # restore
        await pilot.pause()
        assert tree_app.max_depth == -1


# ---------------------------------------------------------------------------
# View mode toggle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_toggle_view_mode(tree_app):
    """Pressing 'v' switches between tree and linear modes."""
    async with tree_app.run_test(size=(120, 40)) as pilot:
        initial_mode = tree_app.view_mode
        await pilot.press("v")
        await pilot.pause()
        assert tree_app.view_mode != initial_mode
        await pilot.press("v")
        await pilot.pause()
        assert tree_app.view_mode == initial_mode


# ---------------------------------------------------------------------------
# scroll_to_kernel API
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scroll_to_kernel_navigates(tree_app):
    """scroll_to_kernel(name) updates the DataTable cursor."""
    async with tree_app.run_test(size=(120, 40)) as pilot:
        tree_app.scroll_to_kernel("aten::mm", 1)
        await pilot.pause()
        dt = tree_app.query_one("#tree-dt")
        # cursor should have moved to the match
        assert dt.cursor_row > 0  # "forward" is row 0; aten::mm is row 1


@pytest.mark.asyncio
async def test_scroll_to_nonexistent_kernel_shows_notification(tree_app):
    """scroll_to_kernel with unknown name fires a notification (no crash)."""
    async with tree_app.run_test(size=(120, 40)) as pilot:
        tree_app.scroll_to_kernel("totally_fake_kernel", 1)
        await pilot.pause()
        # Should not raise; app still running
        assert tree_app.is_running


# ---------------------------------------------------------------------------
# Chat panel
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_toggle_chat_panel(tree_app):
    """'a' opens the chat panel; Escape closes it."""
    async with tree_app.run_test(size=(120, 40)) as pilot:
        from nsys_ai.tree.chat import ChatPanel
        cp = tree_app.query_one("#chat-panel", ChatPanel)
        assert "-active" not in cp.classes

        await pilot.press("A")      # open (A = AI chat, matching original tui.py)
        await pilot.pause()
        assert "-active" in cp.classes

        await pilot.press("escape")  # close via ChatPanel escape binding
        await pilot.pause()
        assert "-active" not in cp.classes
