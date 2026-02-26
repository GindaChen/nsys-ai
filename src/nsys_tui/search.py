"""
search.py — Search kernels and NVTX annotations in Nsight profiles.

Supports:
  - Full-text search on kernel names (demangled)
  - NVTX text search
  - Hierarchical search: find kernels under a specific NVTX path
    (e.g. "sample_0 > flash_attention_forward")
"""
import re
from typing import Optional
from .profile import Profile
from .tree import build_nvtx_tree


def search_kernels(prof: Profile, query: str,
                   device: Optional[int] = None,
                   trim: Optional[tuple[int, int]] = None,
                   limit: int = 200) -> list[dict]:
    """
    Search kernel names (case-insensitive substring match).

    Returns: [{name, start, end, duration_ms, device, stream}]
    """
    q = query.lower()
    devices = [device] if device is not None else prof.meta.devices
    results = []

    for dev in devices:
        for k in prof.kernels(dev, trim):
            if q in k["name"].lower():
                results.append(dict(
                    name=k["name"],
                    start=k["start"], end=k["end"],
                    duration_ms=round((k["end"] - k["start"]) / 1e6, 3),
                    device=dev, stream=k["streamId"],
                ))
                if len(results) >= limit:
                    return results
    return results


def search_nvtx(prof: Profile, query: str,
                device: Optional[int] = None,
                trim: Optional[tuple[int, int]] = None,
                limit: int = 200) -> list[dict]:
    """
    Search NVTX annotation text (case-insensitive substring match).

    Returns: [{text, start, end, duration_ms, thread}]
    """
    q = query.lower()
    devices = [device] if device is not None else prof.meta.devices
    results = []

    for dev in devices:
        threads = prof.gpu_threads(dev)
        if not threads:
            continue
        window = trim or prof.meta.time_range
        nvtx = prof.nvtx_events(threads, window)
        for n in nvtx:
            text = n["text"]
            if text and q in text.lower():
                results.append(dict(
                    text=text,
                    start=n["start"], end=n["end"],
                    duration_ms=round((n["end"] - n["start"]) / 1e6, 3),
                    thread=n["globalTid"],
                ))
                if len(results) >= limit:
                    return results
    return results


def search_hierarchy(prof: Profile, parent_pattern: str,
                     child_pattern: str, device: int,
                     trim: tuple[int, int]) -> list[dict]:
    """
    Find kernels under a specific NVTX path.

    Example: search_hierarchy(prof, "sample_0", "flash", 4, trim)
    Returns kernels whose name matches child_pattern, under NVTX nodes
    matching parent_pattern.

    Returns: [{name, start, end, duration_ms, stream, nvtx_path}]
    """
    roots = build_nvtx_tree(prof, device, trim)
    results = []
    _walk_hierarchy(roots, parent_pattern.lower(), child_pattern.lower(),
                    [], results)
    return results


def _walk_hierarchy(nodes: list, parent_pat: str, child_pat: str,
                    path: list[str], results: list):
    """Recursively walk tree looking for parent→child matches."""
    for node in nodes:
        current_path = path + [node["name"]]

        # Check if any ancestor matches parent_pattern
        in_parent = any(parent_pat in p.lower() for p in current_path)

        if node["type"] == "kernel":
            if in_parent and child_pat in node["name"].lower():
                results.append(dict(
                    name=node["name"],
                    start=node["start"], end=node["end"],
                    duration_ms=round((node["end"] - node["start"]) / 1e6, 3),
                    stream=node.get("stream", -1),
                    nvtx_path=" > ".join(current_path[:-1]),
                ))
        elif node.get("children"):
            _walk_hierarchy(node["children"], parent_pat, child_pat,
                            current_path, results)


def format_results(results: list[dict], kind: str = "kernel") -> str:
    """Format search results as readable text."""
    if not results:
        return "No results found."

    lines = [f"Found {len(results)} result(s):", ""]

    if kind == "hierarchy":
        for r in results:
            lines.append(f"  ⚡ {r['name']}  ({r['duration_ms']:.3f}ms)  "
                         f"[stream {r['stream']}]")
            lines.append(f"    path: {r['nvtx_path']}")
    elif kind == "nvtx":
        for r in results:
            lines.append(f"  📦 {r['text']}  ({r['duration_ms']:.3f}ms)")
    else:
        for r in results:
            lines.append(f"  ⚡ {r['name']}  ({r['duration_ms']:.3f}ms)  "
                         f"[GPU {r['device']}, stream {r['stream']}]")

    return "\n".join(lines)
