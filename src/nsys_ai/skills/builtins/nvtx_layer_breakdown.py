"""Per-NVTX-region GPU time breakdown.

Attributes GPU kernels to their parent NVTX regions via the efficient
nvtx_attribution module (nsys recipe primary, sort-merge fallback),
producing a flat table of "which code region spent the most GPU time".

This enables the agent to say "Layer 12 Attention backward has 15ms
NCCL stall" instead of "some stall at timestamp X".
"""

from collections import defaultdict

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    """Execute NVTX region GPU time breakdown via attribution module."""
    from ...nvtx_attribution import attribute_kernels_to_nvtx
    from ...overlap import classify_kernel

    limit = int(kwargs.get("limit", 20))
    depth = kwargs.get("depth")
    if depth is not None:
        depth = int(depth)
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    trim = (trim_start, trim_end) if trim_start is not None and trim_end is not None else None

    sqlite_path = kwargs.get("_sqlite_path")

    rows = attribute_kernels_to_nvtx(conn, sqlite_path=sqlite_path, trim=trim)

    if not rows:
        return []

    # Optional depth filtering: only keep rows at exactly the requested depth
    if depth is not None:
        if depth < 0:
            return [{"error": "Invalid depth <0 requested. Depth must be >= 0."}]
        rows = [r for r in rows if r.get("nvtx_depth") == depth]
        if not rows:
            return []

    # Python GROUP BY on nvtx_text with kernel classification for
    # compute/NCCL split.
    groups: dict[str, dict] = defaultdict(
        lambda: {
            "total_ns": 0,
            "compute_ns": 0,
            "nccl_ns": 0,
            "count": 0,
            "max_ns": 0,
            "nvtx_depth": -1,
            "nvtx_path": "",
            "nvtx_region": "",
        }
    )
    _class_cache = {}
    for r in rows:
        text = r["nvtx_text"]
        if not text:
            continue
        dur_ns = r["k_dur_ns"]
        k_name = r["kernel_name"]
        if k_name not in _class_cache:
            _class_cache[k_name] = classify_kernel(k_name)
        kernel_class = _class_cache[k_name]

        path = r.get("nvtx_path", text)

        stats = groups[path]
        stats["total_ns"] += dur_ns
        stats["count"] += 1
        if dur_ns > stats["max_ns"]:
            stats["max_ns"] = dur_ns
        # Compute/NCCL split
        if kernel_class.startswith("nccl_"):
            stats["nccl_ns"] += dur_ns
        else:
            stats["compute_ns"] += dur_ns
        # Capture depth/path from first seen entry
        if stats["nvtx_depth"] < 0:
            stats["nvtx_depth"] = r.get("nvtx_depth", 0)
            stats["nvtx_path"] = path
            stats["nvtx_region"] = text

    # Build aggregated results
    results = []
    for path_key, stats in groups.items():
        total_ns = stats["total_ns"]
        count = stats["count"]
        max_ns = stats["max_ns"]
        compute_ns = stats["compute_ns"]
        nccl_ns = stats["nccl_ns"]
        results.append(
            {
                "_raw_total_ns": total_ns,
                "nvtx_region": stats["nvtx_region"],
                "nvtx_depth": stats["nvtx_depth"],
                "nvtx_path": stats["nvtx_path"],
                "kernel_count": count,
                "total_gpu_ms": round(total_ns / 1e6, 2),
                "compute_ms": round(compute_ns / 1e6, 2),
                "nccl_ms": round(nccl_ns / 1e6, 2),
                "nccl_pct": round(100 * nccl_ns / total_ns, 1) if total_ns > 0 else 0,
                "avg_kernel_ms": round(total_ns / count / 1e6, 3),
                "max_kernel_ms": round(max_ns / 1e6, 3),
            }
        )

    # Sort by total GPU time descending, apply limit
    results.sort(key=lambda r: -r["_raw_total_ns"])
    for r in results:
        r.pop("_raw_total_ns", None)
    return results[:limit]


def _format(rows):
    if not rows:
        return "(No NVTX regions with attributed kernels found)"
    if "error" in rows[0]:
        return f"Error: {rows[0]['error']}"

    lines = [
        "── NVTX Region GPU Time Breakdown ──",
        f"{'NVTX Region':<40s}  {'Depth':>5s}  {'Kernels':>7s}  {'Total(ms)':>10s}"
        f"  {'Compute':>9s}  {'NCCL':>9s}  {'NCCL%':>6s}",
        "─" * 96,
    ]
    for r in rows:
        # Favor nvtx_path over nvtx_region for disambiguation
        name = r.get("nvtx_path") or r.get("nvtx_region") or "(unnamed)"
        if len(name) > 38:
            name = "..." + name[-35:]
        lines.append(
            f"{name:<40s}  {r['nvtx_depth']:>5d}  {r['kernel_count']:>7d}  {r['total_gpu_ms']:>10.2f}"
            f"  {r['compute_ms']:>9.2f}  {r['nccl_ms']:>9.2f}  {r['nccl_pct']:>5.1f}%"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="nvtx_layer_breakdown",
    title="NVTX Region GPU Time Breakdown",
    description=(
        "Attributes GPU kernels to their parent NVTX regions (e.g. layers, "
        "forward/backward passes) and ranks them by total GPU time. "
        "Shows compute vs NCCL split per region. "
        "Use to identify which code region is the bottleneck."
    ),
    category="nvtx",
    execute_fn=_execute,
    params=[
        SkillParam("limit", "Max number of NVTX regions to return", "int", False, 20),
        SkillParam(
            "depth", "Filter to specific NVTX nesting depth (0=top-level). Only supported when using Tier 2 sqlite attribution.", "int", False, None
        ),
    ],
    format_fn=_format,
    tags=["nvtx", "layer", "breakdown", "attribution", "region", "nccl", "compute"],
)
