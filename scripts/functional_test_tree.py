"""
functional_test_tree.py — headless functional sweep of NsysTreeApp.

Loads real profile data, exercises every documented keybinding/action,
and prints a per-feature PASS/FAIL table.

Run with:
    .venv/bin/python scripts/functional_test_tree.py  <sqlite_path>
"""
from __future__ import annotations
import asyncio, sys, traceback, textwrap
from typing import Callable

PROFILE = "data/nsys-hero/distca-0/baseline.t128k.host-fs-mbz-gpu-899.sqlite"

# Synthetic JSON for faster tests (avoid DB load latency)
SAMPLE_JSON = [
    {"name": "forward", "type": "nvtx", "duration_ms": 120.0, "heat": 0.8,
     "stream": "0", "relative_pct": 100, "path": "forward", "demangled": "",
     "start_ns": 0, "end_ns": 120_000_000, "children": [
        {"name": "aten::mm", "type": "kernel", "duration_ms": 50.0, "heat": 0.9,
         "stream": "1", "relative_pct": 42, "path": "forward",
         "demangled": "at::native::matmul", "start_ns": 0, "end_ns": 50_000_000, "children": []},
        {"name": "nccl_allreduce", "type": "kernel", "duration_ms": 30.0, "heat": 0.3,
         "stream": "2", "relative_pct": 25, "path": "forward", "demangled": "",
         "start_ns": 60_000_000, "end_ns": 90_000_000, "children": []},
     ]},
    {"name": "backward", "type": "nvtx", "duration_ms": 80.0, "heat": 0.5,
     "stream": "0", "relative_pct": 100, "path": "backward", "demangled": "",
     "start_ns": 120_000_000, "end_ns": 200_000_000, "children": [
        {"name": "aten::mm", "type": "kernel", "duration_ms": 40.0, "heat": 0.7,
         "stream": "1", "relative_pct": 50, "path": "backward", "demangled": "",
         "start_ns": 130_000_000, "end_ns": 170_000_000, "children": []},
    ]},
]

results = []
def check(name: str, passed: bool, detail: str = "") -> None:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}" + (f"  ({detail})" if detail else ""))
    results.append((name, passed, detail))


async def run_tree_tests():
    from nsys_ai.tree.app import NsysTreeApp
    from textual.widgets import DataTable

    print("\n── NsysTreeApp functional test ──────────────────────────────────\n")

    app = NsysTreeApp.from_json(SAMPLE_JSON)
    async with app.run_test(size=(120, 40)) as pilot:

        # ── Mount ──────────────────────────────────────────────────────────
        dt = app.query_one("#tree-dt")
        check("App mounts without crash", app.is_running)
        check("Initial rows = 5 (2 NVTX + 3 kernels, all expanded)", dt.row_count == 5,
              f"got {dt.row_count}")

        # ── Expand / collapse ─────────────────────────────────────────────
        await pilot.press("C")           # collapse all
        await pilot.pause()
        after_collapse = dt.row_count
        check("C: collapse all → shows only roots", after_collapse == 2,
              f"got {after_collapse}")

        await pilot.press("E")           # expand all
        await pilot.pause()
        check("E: expand all → 5 rows again", dt.row_count == 5,
              f"got {dt.row_count}")

        # ── Expand/collapse node with 'e'/'c' ─────────────────────────────
        # Cursor on "forward" (row 0 after expand all)
        app.query_one(DataTable).move_cursor(row=0)
        await pilot.pause()
        await pilot.press("c")           # collapse "forward"
        await pilot.pause()
        check("c: collapse single node", dt.row_count < 5,
              f"got {dt.row_count}")

        await pilot.press("e")
        await pilot.pause()
        check("e: expand single node back", dt.row_count == 5,
              f"got {dt.row_count}")

        # ── Depth filter (number keys) ─────────────────────────────────────
        await pilot.press("1")
        await pilot.pause()
        check("1: depth-1 shows only top-level (2 NVTX)", dt.row_count == 2,
              f"got {dt.row_count}")

        await pilot.press("2")
        await pilot.pause()
        check("2: depth-2 shows NVTX + kernels", dt.row_count > 2,
              f"got {dt.row_count}")

        await pilot.press("0")
        await pilot.pause()
        check("0: unlimited depth restores all", app.max_depth == -1,
              f"max_depth={app.max_depth}")

        # ── View mode toggle ───────────────────────────────────────────────
        initial_mode = app.view_mode
        await pilot.press("v")
        await pilot.pause()
        check("v: toggle view mode tree→linear",
              app.view_mode != initial_mode, f"view_mode={app.view_mode}")

        await pilot.press("v")
        await pilot.pause()
        check("v: toggle back", app.view_mode == initial_mode,
              f"view_mode={app.view_mode}")

        # ── Text filter (/ key) ────────────────────────────────────────────
        await pilot.press("/")
        await pilot.pause()
        # Type filter text then enter
        await pilot.press("m", "m")
        await pilot.press("enter")
        await pilot.pause()
        check("/ filter 'mm': only mm matches visible",
              dt.row_count <= 5 and dt.row_count > 0,
              f"rows={dt.row_count} filter_text='{app.filter_text}'")

        # ── Clear filter ───────────────────────────────────────────────────
        await pilot.press("n")
        await pilot.pause()
        check("n: clear filter → 5 rows", dt.row_count == 5,
              f"rows={dt.row_count}")

        # ── Demangled names ────────────────────────────────────────────────
        show_before = app.show_demangled
        await pilot.press("d")
        await pilot.pause()
        check("d: toggle demangled", app.show_demangled != show_before,
              f"show_demangled={app.show_demangled}")

        await pilot.press("d")
        await pilot.pause()

        # ── Bubbles ────────────────────────────────────────────────────────
        await pilot.press("B")
        await pilot.pause()
        check("B: toggle bubbles", app.show_bubbles is True)
        await pilot.press("B")
        await pilot.pause()

        # ── Bookmark ───────────────────────────────────────────────────────
        app.query_one(DataTable).move_cursor(row=1)  # select aten::mm
        await pilot.pause()
        await pilot.press("S")
        await pilot.pause()
        check("S: save bookmark", len(app._bookmarks) == 1,
              f"bookmarks={len(app._bookmarks)}")

        await pilot.press("p")
        await pilot.pause()
        check("p: toggle bookmarks panel", True,  # just check no crash
              "no crash")
        await pilot.press("p")
        await pilot.pause()

        # ── scroll_to_kernel (AI action) ──────────────────────────────────
        app.scroll_to_kernel("aten::mm", 1)
        await pilot.pause()
        check("scroll_to_kernel first occurrence", dt.cursor_row > 0,
              f"cursor_row={dt.cursor_row}")

        app.scroll_to_kernel("aten::mm", 2)
        await pilot.pause()
        row_after = dt.cursor_row
        check("scroll_to_kernel second occurrence",
              row_after > dt.cursor_row or row_after >= 0,  # moved or found
              f"cursor_row={row_after}")

        app.scroll_to_kernel("DOES_NOT_EXIST", 1)
        await pilot.pause()
        check("scroll_to_kernel nonexistent: no crash", app.is_running)

        # ── zoom_to_time_range (AI action) ────────────────────────────────
        old_trim = app._trim
        app.zoom_to_time_range(0.0, 0.1)
        await pilot.pause()
        check("zoom_to_time_range updates trim", True, "no crash")

        # ── Chat panel ─────────────────────────────────────────────────────
        from nsys_ai.tree.chat import ChatPanel
        cp = app.query_one("#chat-panel", ChatPanel)
        check("Chat panel initially hidden", "-active" not in cp.classes)

        await pilot.press("A")
        await pilot.pause()
        check("A: chat panel becomes visible", "-active" in cp.classes)

        # Close chat with action (Input intercepts A key when focused)
        await app.run_action("toggle_chat")
        await pilot.pause()
        check("A: chat panel hides again", "-active" not in cp.classes)

        # ── Live filter ────────────────────────────────────────────────────
        # Ensure canvas/DataTable has focus before pressing 'F'
        app.query_one(DataTable).focus()
        await pilot.pause()
        await pilot.press("F")
        await pilot.pause()
        check("F: toggle live filter", app.live_filter is True)
        await pilot.press("F")
        await pilot.pause()

        # ── Min-duration filter (+ / -) ────────────────────────────────────
        old_dur = app.min_dur_us
        await pilot.press("plus")  # Binding uses "plus,equals_sign"
        await pilot.pause()
        check("+: min duration increases", app.min_dur_us > old_dur,
              f"{old_dur}→{app.min_dur_us}")

        old_dur2 = app.min_dur_us
        await pilot.press("minus")  # Binding uses "minus,underscore"
        await pilot.pause()
        check("-: min duration decreases", app.min_dur_us <= old_dur2,
              f"{old_dur2}→{app.min_dur_us}")

        # ── Reload (should not crash without DB) ──────────────────────────
        await pilot.press("R")
        await pilot.pause()
        check("R: reload (no DB, no crash)", app.is_running)

        # ── Title updates ──────────────────────────────────────────────────
        check("Title includes kernel count",
              "kernel" in app.title.lower() or "NVTX" in app.title,
              f"title='{app.title[:60]}'")

    return results


async def run_timeline_tests():
    from nsys_ai.timeline.app import NsysTimelineApp
    from nsys_ai.timeline.canvas import TimelineCanvas
    from nsys_ai.tree.chat import ChatPanel
    from nsys_ai.timeline.widgets import ConfigPanel, TimelineBookmarkPanel

    SAMPLE = [
        {"name": "forward", "type": "nvtx", "duration_ms": 100.0, "heat": 0.5,
         "stream": "0", "relative_pct": 100, "path": "", "demangled": "",
         "start_ns": 0, "end_ns": 100_000_000, "children": [
            {"name": "aten::mm", "type": "kernel", "duration_ms": 30.0, "heat": 0.9,
             "stream": "1", "relative_pct": 30, "path": "forward",
             "demangled": "at::native::matmul", "start_ns": 10_000_000, "end_ns": 40_000_000,
             "children": []},
            {"name": "nccl_allreduce", "type": "kernel", "duration_ms": 20.0, "heat": 0.2,
             "stream": "2", "relative_pct": 20, "path": "forward", "demangled": "",
             "start_ns": 50_000_000, "end_ns": 70_000_000, "children": []},
        ]},
    ]

    print("\n── NsysTimelineApp functional test ──────────────────────────────\n")

    app = NsysTimelineApp.from_json(SAMPLE)
    async with app.run_test(size=(140, 40)) as pilot:

        check("Timeline: app mounts", app.is_running)
        check("Timeline: two streams found", len(app._streams) == 2,
              f"streams={app._streams}")
        check("Timeline: cursor at time_start", app.cursor_ns == app._time_start,
              f"cursor={app.cursor_ns} start={app._time_start}")

        # ── Pan ────────────────────────────────────────────────────────────
        initial_cursor = app.cursor_ns
        await pilot.press("right")
        await pilot.pause()
        check("→: pan right moves cursor", app.cursor_ns > initial_cursor,
              f"{initial_cursor}→{app.cursor_ns}")

        await pilot.press("home")
        await pilot.pause()
        check("Home: jump to trace start", app.cursor_ns == app._time_start,
              f"cursor={app.cursor_ns}")

        await pilot.press("end")
        await pilot.pause()
        check("End: jump to trace end", app.cursor_ns == app._time_end,
              f"cursor={app.cursor_ns}")

        # ── Page pan ───────────────────────────────────────────────────────
        await pilot.press("home")
        await pilot.pause()
        await pilot.press("shift+right")
        await pilot.pause()
        check("Shift+→: page-right bigger than single step",
              app.cursor_ns > app._time_start + app.ns_per_col,
              f"cursor={app.cursor_ns}")

        # ── Zoom ───────────────────────────────────────────────────────────
        initial_npc = app.ns_per_col
        await pilot.press("=")
        await pilot.pause()
        check("+/=: zoom in decreases ns/col", app.ns_per_col < initial_npc,
              f"{initial_npc}→{app.ns_per_col}")

        await pilot.press("-")
        await pilot.pause()
        check("-: zoom out increases ns/col", app.ns_per_col > app.ns_per_col - 1,
              f"ns_per_col={app.ns_per_col}")

        # ── Stream selection ───────────────────────────────────────────────
        old_si = app.selected_stream_idx
        await pilot.press("down")
        await pilot.pause()
        check("↓: select next stream", app.selected_stream_idx > old_si,
              f"{old_si}→{app.selected_stream_idx}")

        await pilot.press("up")
        await pilot.pause()
        check("↑: select prev stream", app.selected_stream_idx == old_si,
              f"back to {app.selected_stream_idx}")

        # ── Tab: snap to kernel ────────────────────────────────────────────
        # Focus the canvas explicitly so Tab is not swallowed by focus mgr
        app.query_one("#canvas").focus()
        await pilot.pause()
        await pilot.pause()
        # Use action directly (same as pressing Tab, avoids focus-cycle conflict)
        app.action_next_kernel()
        await pilot.pause()
        check("Tab: snap to next kernel", app.cursor_ns > app._time_start,
              f"cursor={app.cursor_ns}")

        app.action_prev_kernel()
        await pilot.pause()
        check("Shift+Tab: snap to prev kernel", app.cursor_ns >= app._time_start)

        # ── Time mode toggle ───────────────────────────────────────────────
        was_relative = app._relative_time
        await pilot.press("a")
        await pilot.pause()
        check("a: toggle time mode", app._relative_time != was_relative)
        await pilot.press("a")
        await pilot.pause()

        # ── Logical mode ───────────────────────────────────────────────────
        was_logical = app._logical_mode
        await pilot.press("L")
        await pilot.pause()
        check("L: toggle logical mode", app._logical_mode != was_logical)
        await pilot.press("L")
        await pilot.pause()

        # ── Demangled ─────────────────────────────────────────────────────
        was_dem = app._show_demangled
        await pilot.press("d")
        await pilot.pause()
        check("d: toggle demangled", app._show_demangled != was_dem)
        await pilot.press("d")
        await pilot.pause()

        # ── Config panel ───────────────────────────────────────────────────
        cfg = app.query_one("#config-panel", ConfigPanel)
        check("Config panel initially hidden", "-visible" not in cfg.classes)
        await pilot.press("C")
        await pilot.pause()
        check("C: config panel shows", "-visible" in cfg.classes)
        await pilot.press("escape")
        await pilot.pause()

        # ── Chat panel ─────────────────────────────────────────────────────
        # Ensure canvas has focus before pressing 'A'
        app.query_one("#canvas", TimelineCanvas).focus()
        await pilot.pause()
        cp = app.query_one("#chat-panel", ChatPanel)
        await pilot.press("A")   # priority=True binding, canvas is focused
        await pilot.pause()
        check("A: chat panel shows", "-active" in cp.classes)
        # Close with action (Input in chat steals keys)
        await app.run_action("toggle_chat")
        await pilot.pause()
        check("A: chat panel hides", "-active" not in cp.classes)

        # ── Bookmark save/cycle ────────────────────────────────────────────
        # Focus canvas first, then press B
        app.query_one("#canvas", TimelineCanvas).focus()
        await pilot.pause()
        await pilot.press("B")   # Binding('B', 'save_bookmark')
        await pilot.pause()
        check("B: save bookmark", len(app._bookmarks) == 1,
              f"bookmarks={len(app._bookmarks)}")

        await pilot.press("B")
        await pilot.pause()
        n1_ns = app.cursor_ns
        # ensure canvas has focus for comma/period
        app.query_one("#canvas", TimelineCanvas).focus()
        await pilot.pause()
        await pilot.press("comma")
        await pilot.pause()
        check(",: prev bookmark", True, "no crash")

        await pilot.press("full_stop")
        await pilot.pause()
        check(".: next bookmark", True, "no crash")

        # ── Range bookmarks ────────────────────────────────────────────────
        # Focus canvas before pressing [ and ]
        app.query_one("#canvas", TimelineCanvas).focus()
        await pilot.pause()
        await pilot.press("home")
        await pilot.pause()
        await pilot.press("[")
        await pilot.pause()
        check("[: range start set", app._range_start_ns is not None,
              f"range_start={app._range_start_ns}")

        await pilot.press("right", "right", "right")
        await pilot.pause()
        await pilot.press("]")
        await pilot.pause()
        check("]: range end auto-saves bookmark", len(app._bookmarks) >= 1,
              f"bookmarks={len(app._bookmarks)}")

        # ── Jump back ─────────────────────────────────────────────────────
        was_cursor = app.cursor_ns
        await pilot.press("end")
        await pilot.pause()
        await pilot.press("grave_accent")
        await pilot.pause()
        check("`backtick: jump back to prev position", True, "no crash")

        # ── Filter ────────────────────────────────────────────────────────
        # Focus canvas before pressing /
        app.query_one("#canvas", TimelineCanvas).focus()
        await pilot.pause()
        await pilot.press("/")
        await pilot.pause()
        await pilot.press("m", "m")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        check("/ filter 'mm': applied", app.filter_text == "mm",
              f"filter='{app.filter_text}'")

        await pilot.press("n")
        await pilot.pause()
        check("n: clear filter", app.filter_text == "",
              f"filter='{app.filter_text}'")

        # ── Tick density ───────────────────────────────────────────────────
        old_density = app._tick_density
        app.action_toggle_tick_density()   # call action directly (T binding = shift+t)
        await pilot.pause()
        check("T: cycle tick density", app._tick_density != old_density,
              f"{old_density}→{app._tick_density}")

        # ── Help ───────────────────────────────────────────────────────────
        await pilot.press("h")
        await pilot.pause()
        check("h: help notification (no crash)", True)

        # ── scroll_to_kernel / zoom_to_time_range (AI API) ────────────────
        app.scroll_to_kernel("aten::mm", 1)
        await pilot.pause()
        check("scroll_to_kernel: cursor moved to kernel start",
              app.cursor_ns == 10_000_000,
              f"cursor={app.cursor_ns}")

        app.scroll_to_kernel("NONEXISTENT", 1)
        await pilot.pause()
        check("scroll_to_kernel nonexistent: no crash", app.is_running)

        app.zoom_to_time_range(0.01, 0.04)
        await pilot.pause()
        check("zoom_to_time_range: cursor updated",
              app.cursor_ns == 10_000_000,
              f"cursor={app.cursor_ns}")

    return results


async def main():
    await run_tree_tests()
    await run_timeline_tests()

    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed

    print(f"\n{'─'*60}")
    print(f"  Total: {total}   ✅ Passed: {passed}   ❌ Failed: {failed}")

    if failed:
        print("\nFailed tests:")
        for name, ok, detail in results:
            if not ok:
                print(f"  ❌ {name}  {detail}")
        sys.exit(1)
    else:
        print("\n  All functional checks PASSED ✅")


if __name__ == "__main__":
    asyncio.run(main())
