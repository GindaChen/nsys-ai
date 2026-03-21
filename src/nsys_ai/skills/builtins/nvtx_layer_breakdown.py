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

    limit = int(kwargs.get("limit", 20))
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    trim = (trim_start, trim_end) if trim_start is not None and trim_end is not None else None

    sqlite_path = kwargs.get("_sqlite_path")

    rows = attribute_kernels_to_nvtx(conn, sqlite_path=sqlite_path, trim=trim)

    if not rows:
        return []

    # Python GROUP BY on nvtx_text, with incremental aggregation to avoid
    # storing all individual kernel durations in memory.
    groups: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total_ns": 0, "count": 0, "max_ns": 0}
    )
    for r in rows:
        text = r["nvtx_text"]
        if not text:
            continue
        dur_ns = r["k_dur_ns"]
        stats = groups[text]
        stats["total_ns"] += dur_ns
        stats["count"] += 1
        if dur_ns > stats["max_ns"]:
            stats["max_ns"] = dur_ns

    # Build aggregated results
    results = []
    for nvtx_text, stats in groups.items():
        total_ns = stats["total_ns"]
        count = stats["count"]
        max_ns = stats["max_ns"]
        results.append(
            {
                "nvtx_region": nvtx_text,
                "kernel_count": count,
                "total_gpu_ms": round(total_ns / 1e6, 2),
                "avg_kernel_ms": round(total_ns / count / 1e6, 3),
                "max_kernel_ms": round(max_ns / 1e6, 3),
            }
        )

    # Sort by total GPU time descending, apply limit
    results.sort(key=lambda r: -r["total_gpu_ms"])
    return results[:limit]


def _format(rows):
    if not rows:
        return "(No NVTX regions with attributed kernels found)"
    lines = [
        "── NVTX Region GPU Time Breakdown ──",
        f"{'NVTX Region':<50s}  {'Kernels':>7s}  {'Total(ms)':>10s}"
        f"  {'Avg(ms)':>9s}  {'Max(ms)':>9s}",
        "─" * 92,
    ]
    for r in rows:
        name = r["nvtx_region"] or "(unnamed)"
        if len(name) > 48:
            name = name[:45] + "..."
        lines.append(
            f"{name:<50s}  {r['kernel_count']:>7d}  {r['total_gpu_ms']:>10.2f}"
            f"  {r['avg_kernel_ms']:>9.3f}  {r['max_kernel_ms']:>9.3f}"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="nvtx_layer_breakdown",
    title="NVTX Region GPU Time Breakdown",
    description=(
        "Attributes GPU kernels to their parent NVTX regions (e.g. layers, "
        "forward/backward passes) and ranks them by total GPU time. "
        "Use to identify which code region is the bottleneck."
    ),
    category="nvtx",
    execute_fn=_execute,
    params=[SkillParam("limit", "Max number of NVTX regions to return", "int", False, 20)],
    format_fn=_format,
    tags=["nvtx", "layer", "breakdown", "attribution", "region"],
)
