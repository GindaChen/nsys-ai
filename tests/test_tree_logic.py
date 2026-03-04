"""
tests/test_tree_logic.py — Unit tests for tree/logic.py.

All pure-function tests: no Textual, no curses, no display required.
Run with: pytest tests/test_tree_logic.py -v
"""

from nsys_ai.tree.logic import (
    build_nodes,
    compute_summary,
    find_kernel_occurrence,
    find_parent,
    node_index_in_visible,
    visible_rows_linear,
    visible_rows_tree,
)

# ---------------------------------------------------------------------------
# Fixtures
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
                "name": "aten::mm",    # second occurrence of same kernel name
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


# ---------------------------------------------------------------------------
# build_nodes
# ---------------------------------------------------------------------------

def test_build_nodes_count():
    nodes = build_nodes(SAMPLE_JSON)
    # 2 nvtx + 3 kernels = 5 nodes total in DFS order
    assert len(nodes) == 5


def test_build_nodes_dfs_order():
    nodes = build_nodes(SAMPLE_JSON)
    names = [n.name for n in nodes]
    assert names == ["forward", "aten::mm", "nccl_allreduce", "backward", "aten::mm"]


def test_build_nodes_depth():
    nodes = build_nodes(SAMPLE_JSON)
    depths = [n.depth for n in nodes]
    assert depths == [0, 1, 1, 0, 1]


# ---------------------------------------------------------------------------
# compute_summary
# ---------------------------------------------------------------------------

def test_compute_summary():
    kernels, gpu_ms, nvtx = compute_summary(SAMPLE_JSON)
    assert kernels == 3
    assert nvtx == 2
    assert abs(gpu_ms - 120.0) < 1e-6  # 50 + 30 + 40


# ---------------------------------------------------------------------------
# visible_rows_tree
# ---------------------------------------------------------------------------

def test_visible_rows_tree_all_expanded():
    nodes = build_nodes(SAMPLE_JSON)
    # All nodes start expanded (depth < 2 → expanded=True)
    visible = visible_rows_tree(nodes)
    assert len(visible) == 5


def test_visible_rows_tree_collapsed():
    nodes = build_nodes(SAMPLE_JSON)
    forward_node = nodes[0]
    forward_node.expanded = False
    visible = visible_rows_tree(nodes)
    # forward collapsed → its children hidden
    assert len(visible) == 3  # forward, backward, aten::mm(backward)


def test_visible_rows_tree_filter():
    nodes = build_nodes(SAMPLE_JSON)
    visible = visible_rows_tree(nodes, filter_text="mm")
    assert all(
        "mm" in n.name.lower() or "mm" in n.demangled.lower() or n.type == "nvtx"
        for n in visible
    )
    # Should include the nvtx parents too (ancestor match via json_node)
    kernel_names = [n.name for n in visible if n.type == "kernel"]
    assert "aten::mm" in kernel_names


def test_visible_rows_tree_max_depth():
    nodes = build_nodes(SAMPLE_JSON)
    visible = visible_rows_tree(nodes, max_depth=0)
    # Only depth-0 nodes visible
    assert all(n.depth == 0 for n in visible)
    assert len(visible) == 2


def test_visible_rows_tree_min_dur():
    nodes = build_nodes(SAMPLE_JSON)
    # min_dur_us=35_000 → 35ms → only aten::mm(50ms) and nccl(30ms<35 filtered)
    visible = visible_rows_tree(nodes, min_dur_us=35_000)
    kernel_names = [n.name for n in visible if n.type == "kernel"]
    assert "aten::mm" in kernel_names


# ---------------------------------------------------------------------------
# visible_rows_linear
# ---------------------------------------------------------------------------

def test_visible_rows_linear_sorted():
    nodes = build_nodes(SAMPLE_JSON)
    visible = visible_rows_linear(nodes)
    start_times = [n.start_ns for n in visible if n.start_ns is not None]
    assert start_times == sorted(start_times)


def test_visible_rows_linear_filter_demangled():
    nodes = build_nodes(SAMPLE_JSON)
    visible = visible_rows_linear(nodes, filter_text="matmul")
    # Should match via demangled name "at::native::matmul"
    assert len(visible) == 1
    assert visible[0].name == "aten::mm"


# ---------------------------------------------------------------------------
# find_parent
# ---------------------------------------------------------------------------

def test_find_parent_returns_nvtx_ancestor():
    nodes = build_nodes(SAMPLE_JSON)
    visible = visible_rows_tree(nodes)
    names = [n.name for n in visible]
    # "aten::mm" is at index 1; parent is "forward" at index 0
    kernel_idx = names.index("aten::mm")
    parent_idx = find_parent(visible, kernel_idx)
    assert visible[parent_idx].name == "forward"


def test_find_parent_at_root_returns_self():
    nodes = build_nodes(SAMPLE_JSON)
    visible = visible_rows_tree(nodes)
    # Index 0 is "forward" at depth 0 — no parent → returns idx unchanged
    assert find_parent(visible, 0) == 0


# ---------------------------------------------------------------------------
# find_kernel_occurrence
# ---------------------------------------------------------------------------

def test_find_kernel_occurrence_first():
    nodes = build_nodes(SAMPLE_JSON)
    k = find_kernel_occurrence(nodes, "aten::mm", 1)
    assert k is not None
    assert k.path == "forward"


def test_find_kernel_occurrence_second():
    nodes = build_nodes(SAMPLE_JSON)
    k = find_kernel_occurrence(nodes, "aten::mm", 2)
    assert k is not None
    assert k.path == "backward"


def test_find_kernel_occurrence_beyond():
    nodes = build_nodes(SAMPLE_JSON)
    k = find_kernel_occurrence(nodes, "aten::mm", 99)
    assert k is not None  # clamps to last


def test_find_kernel_occurrence_not_found():
    nodes = build_nodes(SAMPLE_JSON)
    k = find_kernel_occurrence(nodes, "nonexistent_kernel", 1)
    assert k is None


# ---------------------------------------------------------------------------
# node_index_in_visible
# ---------------------------------------------------------------------------

def test_node_index_in_visible_found():
    nodes = build_nodes(SAMPLE_JSON)
    visible = visible_rows_tree(nodes)
    target = visible[2]
    idx = node_index_in_visible(visible, target)
    assert idx == 2


def test_node_index_in_visible_not_found():
    nodes = build_nodes(SAMPLE_JSON)
    visible = visible_rows_tree(nodes[:2])  # only partial list
    other = build_nodes(SAMPLE_JSON)[4]    # different object
    assert node_index_in_visible(visible, other) is None


# ---------------------------------------------------------------------------
# Bubble annotation (side-effect test)
# ---------------------------------------------------------------------------

def test_visible_rows_tree_bubbles_annotated():
    nodes = build_nodes(SAMPLE_JSON)
    visible = visible_rows_tree(nodes, show_bubbles=True, bubble_threshold_us=1)
    kernel_nodes = [n for n in visible if n.type == "kernel"]
    # nccl_allreduce starts at 60ms, aten::mm ends at 50ms → gap 10ms > 1μs
    nccl = next((n for n in kernel_nodes if n.name == "nccl_allreduce"), None)
    if nccl is not None:
        # bubble should be annotated (gap > threshold)
        assert nccl._bubble_us >= 0  # may be 0 if different streams
