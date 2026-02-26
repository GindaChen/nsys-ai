"""
gate.py — Environment-variable gated NVTX annotations.

Provides a context manager that wraps torch.cuda.nvtx.range() but is a
complete no-op when the NSIGHT_AI environment variable is not set.
This makes it safe to leave annotations in source code permanently —
zero overhead when not profiling.

Usage:
    from nsys_tui.ai.gate import nsight_range

    with nsight_range("MyModel.forward.attention"):
        y = self.attention(x)

Enable profiling:
    NSIGHT_AI=1 nsys profile python my_script.py
"""
import os
from contextlib import contextmanager

_ENABLED = bool(os.environ.get("NSIGHT_AI"))


@contextmanager
def nsight_range(name: str):
    """
    NVTX range that only activates when NSIGHT_AI=1.

    When disabled: zero overhead (no torch import, no function calls).
    When enabled: wraps torch.cuda.nvtx.range_push/range_pop.
    """
    if not _ENABLED:
        yield
        return

    import torch.cuda.nvtx as nvtx
    nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()


def is_enabled() -> bool:
    """Check if Nsight AI annotations are active."""
    return _ENABLED
