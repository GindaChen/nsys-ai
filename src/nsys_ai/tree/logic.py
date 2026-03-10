"""
tree/logic.py — Pure functions for the NVTX tree TUI.

No Textual, no curses, no I/O — fully unit-testable with plain pytest.
All state is passed as arguments; the only in-place changes are view
annotations on TreeNode instances (for example, bubble_us flags).
"""

from __future__ import annotations

from ..tui_models import TreeNode

# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------


def build_nodes(json_roots: list[dict]) -> list[TreeNode]:
    """Build a depth-first flat list of TreeNode objects from JSON tree."""
    result: list[TreeNode] = []
    _walk(json_roots, 0, result)
    return result


def _walk(nodes: list[dict], depth: int, out: list[TreeNode]) -> None:
    for n in nodes:
        out.append(TreeNode(n, depth))
        if n.get("children"):
            _walk(n["children"], depth + 1, out)


def compute_summary(json_roots: list[dict]) -> tuple[int, float, int]:
    """Return (total_kernels, total_gpu_ms, total_nvtx) for the root list."""
    kernels = 0
    gpu_ms = 0.0
    nvtx = 0

    def _recurse(nodes: list[dict]) -> None:
        nonlocal kernels, gpu_ms, nvtx
        for n in nodes:
            t = n.get("type", "")
            if t == "kernel":
                kernels += 1
                gpu_ms += n.get("duration_ms", 0)
            elif t == "nvtx":
                nvtx += 1
            if n.get("children"):
                _recurse(n["children"])

    _recurse(json_roots)
    return kernels, gpu_ms, nvtx


# ---------------------------------------------------------------------------
# Visibility: tree mode
# ---------------------------------------------------------------------------


def visible_rows_tree(
    all_nodes: list[TreeNode],
    filter_text: str = "",
    max_depth: int = -1,
    min_dur_us: float = 0,
    show_bubbles: bool = False,
    bubble_threshold_us: float = 10,
) -> list[TreeNode]:
    """Return the visible subset of nodes in tree mode.

    A node is visible if:
    1. All NVTX ancestors are expanded.
    2. It passes the depth filter (max_depth).
    3. It passes the text filter (or a descendant does).
    """
    visible: list[TreeNode] = []
    skip_below_depth = -1
    ft = filter_text.lower() if filter_text else ""

    for node in all_nodes:
        if skip_below_depth >= 0 and node.depth > skip_below_depth:
            continue
        skip_below_depth = -1

        if max_depth >= 0 and node.depth > max_depth:
            skip_below_depth = node.depth - 1
            continue

        if ft and not _node_matches_filter(node, ft):
            continue

        visible.append(node)

        if node.type == "nvtx" and not node.expanded:
            skip_below_depth = node.depth

    if min_dur_us > 0:
        min_ms = min_dur_us / 1000.0
        visible = [n for n in visible if n.type != "kernel" or n.duration_ms >= min_ms]

    _annotate_bubbles(visible, show_bubbles, bubble_threshold_us)
    return visible


# ---------------------------------------------------------------------------
# Visibility: linear mode
# ---------------------------------------------------------------------------


def visible_rows_linear(
    all_nodes: list[TreeNode],
    filter_text: str = "",
    min_dur_us: float = 0,
    show_bubbles: bool = False,
    bubble_threshold_us: float = 10,
) -> list[TreeNode]:
    """Return nodes sorted by start time (flat/linear view)."""
    ft = filter_text.lower() if filter_text else ""
    nodes: list[TreeNode] = []
    for node in all_nodes:
        if ft and ft not in node.name.lower():
            if not (node.demangled and ft in node.demangled.lower()):
                continue
        if min_dur_us > 0 and node.type == "kernel":
            if node.duration_ms < min_dur_us / 1000.0:
                continue
        nodes.append(node)

    nodes.sort(key=lambda n: n.start_ns if n.start_ns is not None else 0)
    _annotate_bubbles(nodes, show_bubbles, bubble_threshold_us)
    return nodes


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------


def _node_matches_filter(node: TreeNode, ft: str) -> bool:
    """True if node name / demangled / any descendant matches filter term."""
    if ft in node.name.lower():
        return True
    if node.demangled and ft in node.demangled.lower():
        return True
    return _json_descendant_matches(node.json_node, ft)


def _json_descendant_matches(json_node: dict, ft: str) -> bool:
    if ft in json_node.get("name", "").lower():
        return True
    if ft in json_node.get("demangled", "").lower():
        return True
    for child in json_node.get("children", []):
        if _json_descendant_matches(child, ft):
            return True
    return False


# ---------------------------------------------------------------------------
# Bubble annotation
# ---------------------------------------------------------------------------


def _annotate_bubbles(
    nodes: list[TreeNode],
    enabled: bool,
    threshold_us: float,
) -> None:
    """Mutate _bubble_us on each node in-place."""
    if not enabled:
        for n in nodes:
            n._bubble_us = 0
        return

    last_end: dict[str, int] = {}
    for node in nodes:
        if node.type == "kernel" and node.end_ns is not None:
            sid = str(node.stream)
            if sid in last_end and node.start_ns is not None:
                gap_us = (node.start_ns - last_end[sid]) / 1000.0
                node._bubble_us = gap_us if gap_us >= threshold_us else 0
            else:
                node._bubble_us = 0
            last_end[sid] = node.end_ns
        else:
            node._bubble_us = 0


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------


def find_parent(visible: list[TreeNode], idx: int) -> int:
    """Return index of the nearest NVTX parent in visible list, or idx itself."""
    if idx <= 0 or idx >= len(visible):
        return idx
    target_depth = visible[idx].depth
    for i in range(idx - 1, -1, -1):
        if visible[i].depth < target_depth and visible[i].type == "nvtx":
            return i
    return idx


def find_kernel_occurrence(
    all_nodes: list[TreeNode],
    target_name: str,
    occurrence_index: int = 1,
) -> TreeNode | None:
    """Return the N-th kernel node matching target_name (1-indexed)."""
    matches = [n for n in all_nodes if n.type == "kernel" and n.name == target_name]
    if not matches:
        return None
    idx = max(0, min(occurrence_index - 1, len(matches) - 1))
    return matches[idx]


def node_index_in_visible(visible: list[TreeNode], node: TreeNode) -> int | None:
    """Return the index of a node in the visible list (identity check), or None."""
    for i, n in enumerate(visible):
        if n is node:
            return i
    return None
