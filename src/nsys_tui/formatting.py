"""
formatting.py — Shared time/duration formatting helpers.

These utilities are used by both the tree TUI (tui.py) and the timeline
TUI (tui_timeline.py). Centralising them here eliminates duplication and
provides a single place to adjust display precision.
"""


def fmt_dur(ms: float) -> str:
    """Format a duration given in milliseconds to a human-readable string.

    Examples:
        fmt_dur(0.5)    -> "500μs"
        fmt_dur(1.5)    -> "1.5ms"
        fmt_dur(1500.0) -> "1.50s"
    """
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    if ms >= 1:
        return f"{ms:.1f}ms"
    return f"{ms * 1000:.0f}μs"


def fmt_ns(ns: float | None) -> str:
    """Format a nanosecond timestamp as seconds with 3 decimal places.

    Returns "?" when *ns* is None (unknown / missing).

    Examples:
        fmt_ns(1_500_000_000) -> "1.500s"
        fmt_ns(None)          -> "?"
    """
    if ns is None:
        return "?"
    return f"{ns / 1e9:.3f}s"


def fmt_relative(ns_offset: float) -> str:
    """Format a nanosecond offset as a relative time string (+Xs / +Xms).

    Examples:
        fmt_relative(0)           -> "+0"
        fmt_relative(500_000_000) -> "+0.5s"
        fmt_relative(50_000_000)  -> "+50ms"
    """
    s = ns_offset / 1e9
    if s < 0.001:
        return "+0"
    return f"+{s:.1f}s" if s >= 0.1 else f"+{s * 1000:.0f}ms"
