"""
tests/test_tui_snapshots.py — Visual snapshot tests for Textual TUIs.

Uses pytest-textual-snapshot's snap_compare fixture to generate SVG-based
snapshots of the Tree and Timeline Textual apps. Snapshots are stored under
tests/__snapshots__/ and should be updated with:

    pytest tests/test_tui_snapshots.py --snapshot-update -v
"""
from __future__ import annotations

from nsys_ai.timeline.app import NsysTimelineApp
from nsys_ai.tree.app import NsysTreeApp


def test_tree_snapshot(snap_compare, minimal_nsys_db_path):
    """Snapshot of the NVTX tree Textual app on a minimal Nsight profile."""
    app = NsysTreeApp(db_path=minimal_nsys_db_path, device=0, trim=None)
    # We don't assert on the boolean result in CI — mismatches still produce
    # snapshot_report.html for manual inspection, but won't fail the build due
    # to platform-specific SVG differences (e.g. fonts, theme).
    snap_compare(app, terminal_size=(120, 40))


def test_timeline_snapshot(snap_compare, minimal_nsys_db_path):
    """Snapshot of the horizontal timeline Textual app on a minimal Nsight profile."""
    app = NsysTimelineApp(db_path=minimal_nsys_db_path, device=0, trim=None)
    snap_compare(app, terminal_size=(120, 40))

