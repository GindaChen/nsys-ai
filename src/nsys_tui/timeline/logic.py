"""
timeline/logic.py — Pure functions for the horizontal timeline TUI.

No Textual, no curses, no I/O — fully unit-testable with plain pytest.
All viewport / kernel lookup math lives here.
"""
from __future__ import annotations

import bisect

from ..tui_models import KernelEvent, NvtxSpan

# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------

def extract_events(
    nodes: list[dict],
    path: str = "",
    depth: int = 0,
) -> tuple[list[KernelEvent], list[NvtxSpan]]:
    """Recursively walk JSON tree and collect kernel events + NVTX spans."""
    kernels: list[KernelEvent] = []
    spans: list[NvtxSpan] = []
    _extract(nodes, path, depth, kernels, spans)
    kernels.sort(key=lambda k: k.start_ns)
    spans.sort(key=lambda s: s.start_ns)
    return kernels, spans


def _extract(
    nodes: list[dict],
    path: str,
    depth: int,
    kernels: list[KernelEvent],
    spans: list[NvtxSpan],
) -> None:
    for node in nodes:
        node_path = f"{path} > {node['name']}" if path else node["name"]
        ntype = node.get("type", "")
        if ntype == "kernel":
            kernels.append(KernelEvent(node, path))
        elif ntype == "nvtx":
            spans.append(NvtxSpan(
                node["name"],
                node.get("start_ns", 0) or 0,
                node.get("end_ns", 0) or 0,
                depth,
                node_path,
            ))
        if node.get("children"):
            _extract(node["children"], node_path, depth + 1, kernels, spans)


# ---------------------------------------------------------------------------
# Stream helpers
# ---------------------------------------------------------------------------

def collect_streams(kernels: list[KernelEvent]) -> list[str]:
    """Return sorted unique stream IDs (numeric-first)."""
    stream_set = set(k.stream for k in kernels)
    return sorted(stream_set, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s))


def build_stream_kernels(
    kernels: list[KernelEvent],
    streams: list[str],
) -> dict[str, list[KernelEvent]]:
    """Partition kernels by stream, each list sorted by start_ns."""
    result: dict[str, list[KernelEvent]] = {}
    for s in streams:
        result[s] = sorted([k for k in kernels if k.stream == s], key=lambda k: k.start_ns)
    return result


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_kernels(
    kernels: list[KernelEvent],
    filter_text: str = "",
    min_dur_us: float = 0,
) -> list[KernelEvent]:
    """Return kernels matching filter + duration threshold."""
    out = kernels
    if min_dur_us > 0:
        min_ms = min_dur_us / 1000.0
        out = [k for k in out if k.duration_ms >= min_ms]
    if filter_text:
        ft = filter_text.lower()
        out = [k for k in out
               if ft in k.name.lower() or (k.demangled and ft in k.demangled.lower())]
    return out


# ---------------------------------------------------------------------------
# Kernel lookup
# ---------------------------------------------------------------------------

def kernel_at_time(kernels: list[KernelEvent], ns: int) -> KernelEvent | None:
    """Return the kernel containing ns, or the nearest one."""
    if not kernels:
        return None
    for k in kernels:
        if k.start_ns <= ns <= k.end_ns:
            return k
    return min(kernels, key=lambda k: min(abs(k.start_ns - ns), abs(k.end_ns - ns)))


def kernel_index_at_time(kernels: list[KernelEvent], ns: int) -> int:
    """Return index of kernel nearest to ns (-1 if empty)."""
    if not kernels:
        return -1
    starts = [k.start_ns for k in kernels]
    idx = bisect.bisect_left(starts, ns)
    if idx == 0:
        return 0
    if idx >= len(kernels):
        return len(kernels) - 1
    # Compare idx-1 and idx
    dist_before = min(abs(kernels[idx - 1].start_ns - ns), abs(kernels[idx - 1].end_ns - ns))
    dist_at = min(abs(kernels[idx].start_ns - ns), abs(kernels[idx].end_ns - ns))
    return idx - 1 if dist_before <= dist_at else idx


def find_kernel_by_name(
    stream_kernels: dict[str, list[KernelEvent]],
    target_name: str,
    occurrence: int = 1,
) -> tuple[str, int] | None:
    """Return (stream, index) of the N-th kernel named target_name, or None."""
    count = 0
    for stream, kernels in stream_kernels.items():
        for idx, k in enumerate(kernels):
            if k.name == target_name:
                count += 1
                if count == occurrence:
                    return stream, idx
    return None


# ---------------------------------------------------------------------------
# Viewport math
# ---------------------------------------------------------------------------

def center_viewport(cursor_ns: int, ns_per_col: int, timeline_w: int) -> int:
    """Return the viewport start_ns so cursor_ns is centered."""
    half = (ns_per_col * timeline_w) // 2
    return cursor_ns - half


def nice_tick_interval(timeline_w: int, ns_per_col: int, tick_density: int = 6) -> int:
    """Choose a 'nice' tick interval such that ticks aren't too dense.

    Returns a nanosecond interval that results in roughly `tick_density`
    ticks across the visible viewport.
    """
    total_ns = ns_per_col * timeline_w
    rough = total_ns // tick_density

    # Round to nice power-of-10 multiples: 1, 2, 5, 10, 20, 50, …
    if rough <= 0:
        return 1
    magnitude = 10 ** (len(str(rough)) - 1)
    for factor in (1, 2, 5, 10):
        candidate = factor * magnitude
        if candidate >= rough:
            return candidate
    return rough


def zoom_ns_per_col(
    current: int,
    direction: int,
    time_span: int,
) -> int:
    """Zoom in (direction=-1) or out (+1), clamped to [1, time_span]."""
    if direction > 0:
        new = current * 3 // 2
    else:
        new = max(1, current * 2 // 3)
    return max(1, min(time_span, new))


# ---------------------------------------------------------------------------
# Time bounds
# ---------------------------------------------------------------------------

def time_bounds(kernels: list[KernelEvent], trim: tuple[int, int]) -> tuple[int, int]:
    """Return (time_start, time_end) from kernel list or trim fallback."""
    if kernels:
        return min(k.start_ns for k in kernels), max(k.end_ns for k in kernels)
    return trim
