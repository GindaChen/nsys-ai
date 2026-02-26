"""
analyzer.py — Analyze NVTX trees to find multi-kernel regions for refinement.

After profiling with NVTX annotations, this module examines the resulting
tree structure to determine which NVTX regions need finer-grained scoping.

The convergence criterion: each innermost NVTX should contain exactly one
GPU kernel. Regions with multiple kernels are candidates for refinement.
"""
from typing import Optional


def find_refinement_targets(tree_roots: list[dict],
                            max_kernels: int = 1) -> list[dict]:
    """
    Find NVTX nodes that contain more than `max_kernels` kernel children.

    These are the regions where the agent should insert finer-grained
    NVTX annotations to narrow down kernel-to-source mapping.

    Args:
        tree_roots: Output from tree.build_nvtx_tree() converted via to_json()
        max_kernels: Maximum allowed kernels per NVTX leaf (default: 1)

    Returns:
        List of dicts with: name, kernel_count, kernels, depth, duration_ms
    """
    targets = []
    _walk(tree_roots, targets, max_kernels, depth=0)
    return sorted(targets, key=lambda t: -t["kernel_count"])


def _walk(nodes, targets, max_kernels, depth):
    """Recursively find NVTX nodes with too many kernel children."""
    for node in nodes:
        if node["type"] != "nvtx":
            continue

        children = node.get("children", [])
        nvtx_children = [c for c in children if c["type"] == "nvtx"]
        kern_children = [c for c in children if c["type"] == "kernel"]

        # If this NVTX has no NVTX children (it's a leaf NVTX)
        # and has more than max_kernels kernel children → refinement target
        if not nvtx_children and len(kern_children) > max_kernels:
            targets.append({
                "name": node["name"],
                "depth": depth,
                "duration_ms": node.get("duration_ms", 0),
                "kernel_count": len(kern_children),
                "kernels": [k["name"] for k in kern_children],
            })

        # Also check NVTX nodes that have BOTH nvtx and kernel children
        # (mixed — the kernels at this level aren't covered by sub-NVTX)
        if nvtx_children and len(kern_children) > max_kernels:
            targets.append({
                "name": node["name"],
                "depth": depth,
                "duration_ms": node.get("duration_ms", 0),
                "kernel_count": len(kern_children),
                "kernels": [k["name"] for k in kern_children],
                "note": "mixed: has both NVTX sub-ranges and uncovered kernels",
            })

        # Recurse into NVTX children
        _walk(nvtx_children, targets, max_kernels, depth + 1)


def convergence_report(tree_roots: list[dict]) -> dict:
    """
    Report on how close the NVTX tree is to full convergence.

    Returns:
        total_nvtx: Total NVTX nodes
        total_kernels: Total kernel leaves
        converged_nvtx: NVTX leaves with exactly 1 kernel
        unconverged_nvtx: NVTX leaves with >1 kernel
        coverage_pct: Percentage of kernels mapped to a single NVTX
        unmapped_kernels: Kernels under multi-kernel NVTX (still ambiguous)
    """
    stats = {"total_nvtx": 0, "total_kernels": 0,
             "converged": 0, "unconverged": 0,
             "mapped_kernels": 0, "unmapped_kernels": 0}
    _convergence_walk(tree_roots, stats)

    total = stats["mapped_kernels"] + stats["unmapped_kernels"]
    return {
        "total_nvtx": stats["total_nvtx"],
        "total_kernels": total,
        "converged_nvtx": stats["converged"],
        "unconverged_nvtx": stats["unconverged"],
        "coverage_pct": round(100 * stats["mapped_kernels"] / total, 1) if total else 0,
        "mapped_kernels": stats["mapped_kernels"],
        "unmapped_kernels": stats["unmapped_kernels"],
    }


def _convergence_walk(nodes, stats):
    for node in nodes:
        if node["type"] == "kernel":
            continue

        stats["total_nvtx"] += 1
        children = node.get("children", [])
        nvtx_children = [c for c in children if c["type"] == "nvtx"]
        kern_children = [c for c in children if c["type"] == "kernel"]

        if not nvtx_children:
            # Leaf NVTX
            if len(kern_children) == 1:
                stats["converged"] += 1
                stats["mapped_kernels"] += 1
            elif len(kern_children) > 1:
                stats["unconverged"] += 1
                stats["unmapped_kernels"] += len(kern_children)
            # 0 kernels = annotation with no GPU work (skip)

        _convergence_walk(nvtx_children, stats)


def format_report(report: dict) -> str:
    """Format convergence report as readable text."""
    return (
        f"Convergence Report\n"
        f"  NVTX nodes:        {report['total_nvtx']}\n"
        f"  Total kernels:     {report['total_kernels']}\n"
        f"  Converged NVTX:    {report['converged_nvtx']} (1 kernel → 1 source location)\n"
        f"  Unconverged NVTX:  {report['unconverged_nvtx']} (need finer scoping)\n"
        f"  Mapped kernels:    {report['mapped_kernels']} ({report['coverage_pct']}%)\n"
        f"  Unmapped kernels:  {report['unmapped_kernels']}"
    )
