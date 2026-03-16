"""CUDA API launch overhead — time between API call and kernel execution."""

from ..base import Skill, SkillParam


def _format(rows):
    if not rows:
        return "(No kernel launch overhead data found)"
    lines = [
        "── Kernel Launch Overhead ──",
        f"{'Kernel':<50s}  {'API(ms)':>8s}  {'Kern(ms)':>9s}  {'Overhead(μs)':>13s}",
        "─" * 86,
    ]
    for r in rows:
        name = r["kernel_name"]
        if len(name) > 48:
            name = name[:45] + "..."
        lines.append(
            f"{name:<50s}  {r['api_ms']:>8.3f}  {r['kernel_ms']:>9.3f}  {r['overhead_us']:>13.1f}"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="kernel_launch_overhead",
    title="Kernel Launch Overhead",
    description=(
        "Measures the gap between a CUDA Runtime API call (e.g. cudaLaunchKernel) "
        "and the actual GPU kernel execution. High overhead indicates CPU-side "
        "bottlenecks or excessive kernel launch latency."
    ),
    category="kernels",
    sql="""\
SELECT s.value AS kernel_name,
       ROUND((r.[end] - r.start) / 1e6, 3) AS api_ms,
       ROUND((k.[end] - k.start) / 1e6, 3) AS kernel_ms,
       ROUND((k.start - r.start) / 1e3, 1) AS overhead_us
FROM {runtime_table} r
JOIN {kernel_table} k ON r.correlationId = k.correlationId
JOIN StringIds s ON k.shortName = s.id
WHERE 1=1 {trim_clause}
ORDER BY overhead_us DESC
LIMIT {limit}""",
    params=[SkillParam("limit", "Max results", "int", False, 20)],
    format_fn=_format,
    tags=["launch", "overhead", "latency", "cpu", "bottleneck"],
)
