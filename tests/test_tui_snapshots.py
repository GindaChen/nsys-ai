"""
tests/test_tui_snapshots.py — Visual snapshot tests for Textual TUIs.

Uses pytest-textual-snapshot's snap_compare fixture to generate SVG-based
snapshots of the Tree and Timeline Textual apps. Snapshots are stored under
tests/snapshots/ and should be updated with:

    pytest tests/test_tui_snapshots.py --snapshot-update -v
"""
from __future__ import annotations

from nsys_tui.timeline.app import NsysTimelineApp
from nsys_tui.tree.app import NsysTreeApp


def test_tree_snapshot(snap_compare, minimal_nsys_db_path):
    """Snapshot of the NVTX tree Textual app on a minimal Nsight profile."""
    app = NsysTreeApp(db_path=minimal_nsys_db_path, device=0, trim=None)
    assert snap_compare(app, terminal_size=(120, 40))


def test_timeline_snapshot(snap_compare, minimal_nsys_db_path):
    """Snapshot of the horizontal timeline Textual app on a minimal Nsight profile."""
    app = NsysTimelineApp(db_path=minimal_nsys_db_path, device=0, trim=None)
    assert snap_compare(app, terminal_size=(120, 40))

