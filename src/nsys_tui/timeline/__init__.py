"""
timeline/ — Textual-based horizontal timeline TUI package.

Replaces the monolithic curses-based tui_timeline.py.

Public API:
    run_timeline(db_path, device, trim, min_ms=0)

Note: Textual is imported lazily inside run_timeline to avoid stalling at
package import time when Textual is not installed.
"""
from __future__ import annotations


def run_timeline(
    db_path: str,
    device: int,
    trim: tuple[int, int] | None,
    max_depth: int = -1,
    min_ms: float = 0,
) -> None:
    """Launch the Textual horizontal timeline browser (lazy-imports Textual)."""
    from .app import run_timeline as _run
    _run(db_path, device, trim, min_ms=min_ms)


__all__ = ["run_timeline"]
