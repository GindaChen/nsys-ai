"""
timeline/ — Textual-based horizontal timeline TUI package.

Replaces the monolithic curses-based tui_timeline.py.

Public API:
    run_timeline(db_path, device, trim, min_ms=0)

Note: Textual is imported lazily inside run_timeline to avoid stalling at
package import time when Textual is not installed.
"""
from __future__ import annotations

import sys


def run_timeline(
    db_path: str,
    device: int,
    trim: tuple[int, int] | None,
    min_ms: float = 0,
) -> None:
    """Launch the Textual horizontal timeline browser.

    Falls back to a text kernel summary when stdout is not a TTY (e.g. piped).
    """
    if not sys.stdout.isatty():
        _print_static_summary(db_path, device, trim, min_ms)
        return
    from .app import run_timeline as _run
    _run(db_path, device, trim, min_ms=min_ms)


def _print_static_summary(
    db_path: str,
    device: int,
    trim: tuple[int, int] | None,
    min_ms: float,
) -> None:
    """Text summary fallback for piped / non-TTY output.

    Uses the GPU kernel table directly (Profile.kernels) so the summary
    includes all kernels in the trim window, not only those under NVTX.
    """
    from collections import defaultdict

    from .. import profile as _profile
    from ..formatting import fmt_dur as _fmt_dur
    try:
        with _profile.open(db_path) as prof:
            trim_ns = trim or (prof.meta.time_range[0], prof.meta.time_range[1])
            raw = prof.kernels(device, trim_ns)
            # Filter by min_ms (duration in ms)
            min_ns = int(min_ms * 1e6)
            kernels = [k for k in raw if (k["end"] - k["start"]) >= min_ns]
            print(f"Timeline summary: GPU {device}  {len(kernels)} kernels")
            stream_count: dict[str, int] = defaultdict(int)
            stream_dur: dict[str, float] = defaultdict(float)
            for k in kernels:
                sid = str(k.get("streamId", ""))
                dur_ms = (k["end"] - k["start"]) / 1e6
                stream_count[sid] += 1
                stream_dur[sid] += dur_ms
            for sid in sorted(stream_count, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s)):
                print(f"  Stream {sid}: {stream_count[sid]} kernels  {_fmt_dur(stream_dur[sid])} total")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)


__all__ = ["run_timeline"]
