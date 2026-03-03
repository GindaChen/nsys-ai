"""
tui_models.py — Data model classes shared by tui.py and tui_timeline.py.

Separating these from the rendering code keeps the classes testable without
importing curses, and serves as the type-safe foundation before further
splitting the large TUI modules.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Tree TUI data model
# ---------------------------------------------------------------------------

class TreeNode:
    """Flat representation of a single NVTX/kernel node for tree display.

    Constructed from a JSON dict produced by ``tree.to_json()`` and a nesting
    *depth*.  All attributes are declared in ``__slots__`` for memory
    efficiency — the TUI may hold thousands of nodes simultaneously.
    """
    __slots__ = (
        'name', 'type', 'depth', 'duration_ms', 'heat', 'stream',
        'relative_pct', 'path', 'demangled', 'children_count',
        'kernel_count', 'nvtx_count', 'expanded', 'has_children',
        'start_ns', 'end_ns', 'json_node', '_bubble_us',
    )

    # Type annotations for static analysis tools (mypy / pyright).
    name: str
    type: str
    depth: int
    duration_ms: float
    heat: float
    stream: str
    relative_pct: float
    path: str
    demangled: str
    children_count: int
    kernel_count: int
    nvtx_count: int
    expanded: bool
    has_children: bool
    start_ns: int | None
    end_ns: int | None
    json_node: dict
    _bubble_us: int

    def __init__(self, json_node: dict, depth: int) -> None:
        self.json_node = json_node
        self.name = json_node.get('name', '?')
        self.type = json_node.get('type', '')
        self.depth = depth
        self.duration_ms = json_node.get('duration_ms', 0)
        self.heat = json_node.get('heat', 0)
        self.stream = json_node.get('stream', '?')
        self.relative_pct = json_node.get('relative_pct', 100)
        self.path = json_node.get('path', '')
        self.demangled = json_node.get('demangled', '')
        self.start_ns = json_node.get('start_ns')
        self.end_ns = json_node.get('end_ns')
        children = json_node.get('children', [])
        self.has_children = len(children) > 0
        self.kernel_count = sum(1 for c in children if c.get('type') == 'kernel')
        self.nvtx_count = sum(1 for c in children if c.get('type') == 'nvtx')
        self.children_count = len(children)
        self.expanded = depth < 2  # Default: expand top 2 levels
        self._bubble_us = 0


# ---------------------------------------------------------------------------
# Timeline TUI data models
# ---------------------------------------------------------------------------

class KernelEvent:
    """A single GPU kernel execution event extracted from the JSON tree.

    Used by ``TimelineTUI`` to populate the timeline view.  Memory layout is
    optimised with ``__slots__`` because large profiles may have hundreds of
    thousands of kernels.
    """
    __slots__ = (
        'name', 'demangled', 'start_ns', 'end_ns', 'duration_ms',
        'stream', 'heat', 'nvtx_path', 'is_nccl',
    )

    name: str
    demangled: str
    start_ns: int
    end_ns: int
    duration_ms: float
    stream: str
    heat: float
    nvtx_path: str
    is_nccl: bool

    def __init__(self, json_node: dict, path: str = '') -> None:
        self.name = json_node.get('name', '?')
        self.demangled = json_node.get('demangled', '')
        self.start_ns = json_node.get('start_ns', 0)
        self.end_ns = json_node.get('end_ns', 0)
        self.duration_ms = json_node.get('duration_ms', 0)
        self.stream = str(json_node.get('stream', '?'))
        self.heat = json_node.get('heat', 0)
        self.nvtx_path = path
        self.is_nccl = 'nccl' in self.name.lower()


class NvtxSpan:
    """A single NVTX push/pop range shown as a coloured band on the timeline."""
    __slots__ = ('name', 'start_ns', 'end_ns', 'depth', 'path')

    name: str
    start_ns: int
    end_ns: int
    depth: int
    path: str

    def __init__(self, name: str, start_ns: int, end_ns: int,
                 depth: int, path: str) -> None:
        self.name = name
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.depth = depth
        self.path = path


# ---------------------------------------------------------------------------
# Timeline helpers
# ---------------------------------------------------------------------------

_KERNEL_NAME_PREFIXES = (
    'void ', 'at::native::', 'at::cuda::', 'cutlass::',
    'cublasLt', 'cublas', 'sm90_', 'sm80_',
)


def short_kernel_name(name: str) -> str:
    """Strip common namespace/qualifier prefixes for inline timeline display.

    Examples::

        short_kernel_name("void at::native::elementwise_kernel<...>")
        # -> "elementwise"
    """
    for prefix in _KERNEL_NAME_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
    if '<' in name:
        name = name[:name.index('<')]
    if name.endswith('_kernel'):
        name = name[:-7]
    return name if name else '?'


def collect_kernels(roots: list[dict], out: list[KernelEvent],
                    path: str = '') -> None:
    """Recursively walk the JSON tree and append :class:`KernelEvent` objects.

    Args:
        roots: JSON node list from ``tree.to_json()``.
        out:   Accumulator list — kernel events are appended here.
        path:  Dot-separated NVTX ancestor path (built during recursion).
    """
    for n in roots:
        node_path = f"{path} > {n['name']}" if path else n['name']
        if n.get('type') == 'kernel':
            out.append(KernelEvent(n, path))
        if n.get('children'):
            collect_kernels(n['children'], out, node_path)
