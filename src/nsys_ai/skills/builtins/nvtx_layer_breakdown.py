"""Per-NVTX-region GPU time breakdown.

Attributes GPU kernels to their parent NVTX regions via the runtime
correlation chain (NVTX → Runtime → Kernel), producing a flat table
of "which code region spent the most GPU time".

This enables the agent to say "Layer 12 Attention backward has 15ms
NCCL stall" instead of "some stall at timestamp X".
"""

from ..base import Skill, SkillParam


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
    sql="""\
SELECT
    COALESCE(n.text, s2.value) AS nvtx_region,
    COUNT(DISTINCT k.correlationId) AS kernel_count,
    ROUND(SUM(k.[end] - k.start) / 1e6, 2) AS total_gpu_ms,
    ROUND(AVG(k.[end] - k.start) / 1e6, 3) AS avg_kernel_ms,
    ROUND(MAX(k.[end] - k.start) / 1e6, 3) AS max_kernel_ms
FROM {nvtx_table} n
JOIN {runtime_table} r
    ON r.globalTid = n.globalTid
    AND r.start >= n.start AND r.[end] <= n.[end]
JOIN {kernel_table} k
    ON k.correlationId = r.correlationId
LEFT JOIN StringIds s2 ON n.textId = s2.id
WHERE n.[end] > n.start
    AND COALESCE(n.text, s2.value) IS NOT NULL
    {trim_clause}
GROUP BY COALESCE(n.text, s2.value)
ORDER BY total_gpu_ms DESC
LIMIT {limit}""",
    params=[SkillParam("limit", "Max number of NVTX regions to return", "int", False, 20)],
    format_fn=_format,
    tags=["nvtx", "layer", "breakdown", "attribution", "region"],
)
