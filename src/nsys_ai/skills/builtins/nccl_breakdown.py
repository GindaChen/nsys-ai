"""NCCL collective operation breakdown."""

from ..base import Skill


def _format(rows):
    if not rows:
        return "(No NCCL operations found — is this a multi-GPU profile?)"
    lines = [
        "── NCCL Collective Breakdown ──",
        f"{'Operation':<40s}  {'Count':>7s}  {'Total(ms)':>10s}  {'Avg(ms)':>9s}  {'Max(ms)':>9s}",
        "─" * 82,
    ]
    for r in rows:
        name = r["kernel_name"]
        if len(name) > 38:
            name = name[:35] + "..."
        lines.append(
            f"{name:<40s}  {r['count']:>7d}  {r['total_ms']:>10.2f}  "
            f"{r['avg_ms']:>9.2f}  {r['max_ms']:>9.2f}"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="nccl_breakdown",
    title="NCCL Collective Breakdown",
    description=(
        "Summarizes NCCL collective operations (AllReduce, AllGather, ReduceScatter, etc.) "
        "by type, showing count, total time, and variability. Use this to assess whether "
        "communication is a bottleneck in distributed training."
    ),
    category="communication",
    sql="""\
SELECT s.value AS kernel_name,
       COUNT(*) AS count,
       ROUND(SUM(k.[end] - k.start) / 1e6, 2) AS total_ms,
       ROUND(AVG(k.[end] - k.start) / 1e6, 2) AS avg_ms,
       ROUND(MAX(k.[end] - k.start) / 1e6, 2) AS max_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
WHERE (s.value LIKE '%nccl%' OR s.value LIKE '%NCCL%') {trim_clause}
GROUP BY s.value
ORDER BY total_ms DESC""",
    format_fn=_format,
    tags=["nccl", "collective", "allreduce", "communication", "distributed", "multi-gpu"],
)
