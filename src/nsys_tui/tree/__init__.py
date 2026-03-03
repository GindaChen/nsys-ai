"""
tree/ — Textual-based NVTX tree TUI package.

This package replaces the old ``tree.py`` module AND the curses ``tui.py``.

Backward-compat re-exports (other src modules do ``from .tree import X``):
    build_nvtx_tree, to_json, format_text, format_markdown, _find_primary_thread

TUI entry point (lazy-imports Textual to avoid import-time stalls):
    run_tui(db_path, device, trim, max_depth=-1, min_ms=0)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Backward-compat re-exports from nvtx_tree.py.
# Other existing modules do: from .tree import build_nvtx_tree
# (They previously imported from the old tree.py module; now this package
# shadows that module, so we must re-export the same names.)
# ---------------------------------------------------------------------------
from ..nvtx_tree import (
    _find_primary_thread,
    build_nvtx_tree,
    format_markdown,
    format_text,
    to_json,
)

__all__ = [
    "build_nvtx_tree",
    "to_json",
    "format_text",
    "format_markdown",
    "_find_primary_thread",
    "run_tui",
]


# ---------------------------------------------------------------------------
# TUI entry point (lazy-imports Textual)
# ---------------------------------------------------------------------------

def run_tui(
    db_path: str,
    device: int,
    trim: tuple[int, int] | None,
    max_depth: int = -1,
    min_ms: float = 0,
) -> None:
    """Launch the Textual NVTX tree browser (imports Textual lazily)."""
    from .app import run_tui as _run
    _run(db_path, device, trim, max_depth=max_depth, min_ms=min_ms)
