"""Top GPU kernels by total execution time."""

from ..base import Skill, SkillParam


def _format(rows):
    if not rows:
        return "(No kernels found)"
    lines = [
        "── Top GPU Kernels by Total Time ──",
        f"{'Kernel':<60s}  {'Count':>7s}  {'Total(ms)':>10s}  {'Avg(ms)':>9s}  {'Min(ms)':>9s}  {'Max(ms)':>9s}",
        "─" * 112,
    ]
    for r in rows:
        name = r["kernel_name"]
        if len(name) > 58:
            name = name[:55] + "..."
        lines.append(
            f"{name:<60s}  {r['invocations']:>7d}  {r['total_ms']:>10.2f}  "
            f"{r['avg_ms']:>9.2f}  {r['min_ms']:>9.2f}  {r['max_ms']:>9.2f}"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="top_kernels",
    title="Top GPU Kernels by Total Time",
    description=(
        "Lists the heaviest GPU kernels ranked by cumulative execution time. "
        "Use this to identify hotspots — the kernels that dominate total GPU time."
    ),
    category="kernels",
    sql="""\
SELECT s.value AS kernel_name,
       COUNT(*) AS invocations,
       ROUND(SUM(k.[end] - k.start) / 1e6, 2) AS total_ms,
       ROUND(AVG(k.[end] - k.start) / 1e6, 2) AS avg_ms,
       ROUND(MIN(k.[end] - k.start) / 1e6, 2) AS min_ms,
       ROUND(MAX(k.[end] - k.start) / 1e6, 2) AS max_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
WHERE 1=1 {trim_clause}
GROUP BY s.value
ORDER BY total_ms DESC
LIMIT {limit}""",
    params=[SkillParam("limit", "Max number of kernels to return", "int", False, 15)],
    format_fn=_format,
    tags=["hotspot", "kernel", "duration", "performance", "top"],
)
