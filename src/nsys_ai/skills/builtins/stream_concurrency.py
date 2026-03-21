"""Stream concurrency analysis.

Measures how many GPU streams are executing kernels simultaneously.
True multi-stream concurrency means the GPU can overlap compute
with memory copies or different compute workloads.

This skill answers: "Is the GPU saturated, or are kernels serialized
on a single stream?"
"""

from ..base import Skill, SkillParam

SKILL = Skill(
    name="stream_concurrency",
    title="Stream Concurrency Analysis",
    description=(
        "Analyzes multi-stream GPU concurrency: how many streams are active "
        "simultaneously, whether kernels are packed or sparse, and whether "
        "compute overlaps with memory operations. "
        "Identifies serialization bottlenecks."
    ),
    category="kernels",
    sql="""\
WITH stream_summary AS (
    SELECT
        k.streamId,
        COUNT(*) AS kernel_count,
        MIN(k.start) AS first_start,
        MAX(k.[end]) AS last_end,
        ROUND(SUM(k.[end] - k.start) / 1e6, 2) AS total_gpu_ms,
        ROUND(MAX(k.[end] - k.start) / 1e6, 3) AS max_kernel_ms,
        ROUND(AVG(k.[end] - k.start) / 1e3, 1) AS avg_kernel_us
    FROM {kernel_table} k
    WHERE 1=1
        {trim_clause}
    GROUP BY k.streamId
),
global_stats AS (
    SELECT
        COUNT(DISTINCT streamId) AS active_streams,
        SUM(kernel_count) AS total_kernels,
        MIN(first_start) AS global_start,
        MAX(last_end) AS global_end,
        SUM(total_gpu_ms) AS sum_gpu_ms
    FROM stream_summary
)
SELECT
    s.streamId,
    s.kernel_count,
    s.total_gpu_ms,
    s.avg_kernel_us,
    s.max_kernel_ms,
    ROUND((s.last_end - s.first_start) / 1e6, 2) AS stream_span_ms,
    ROUND(s.total_gpu_ms / NULLIF((s.last_end - s.first_start) / 1e6, 0) * 100, 1)
        AS stream_util_pct,
    g.active_streams,
    g.total_kernels,
    ROUND((g.global_end - g.global_start) / 1e6, 2) AS global_span_ms,
    ROUND(g.sum_gpu_ms / NULLIF((g.global_end - g.global_start) / 1e6, 0) * 100, 1)
        AS sum_util_pct
FROM stream_summary s, global_stats g
ORDER BY s.total_gpu_ms DESC
LIMIT {limit}""",
    format_fn=lambda rows: _format(rows),
    params=[
        SkillParam("limit", "Max streams to show", "int", False, 10),
    ],
    tags=["stream", "concurrency", "parallel", "serialization", "utilization"],
)


def _format(rows):
    if not rows:
        return "(No kernel activity found)"
    r0 = rows[0]
    lines = [
        "── Stream Concurrency Analysis ──",
        f"  Active streams: {r0['active_streams']}",
        f"  Total kernels:  {r0['total_kernels']}",
        f"  Global span:    {r0['global_span_ms']:.2f}ms",
        f"  Sum GPU time:   {r0['sum_util_pct']:.1f}% of span (>100% = true concurrency)",
        "",
        f"{'Stream':>7s} {'Kernels':>8s} {'GPU(ms)':>10s} {'AvgKern':>10s} {'Util%':>7s}",
        "─" * 50,
    ]
    for r in rows:
        lines.append(
            f"s{r['streamId']:>5d} {r['kernel_count']:>8d} "
            f"{r['total_gpu_ms']:>10.2f} {r['avg_kernel_us']:>8.1f}µs "
            f"{r['stream_util_pct']:>6.1f}%"
        )
    return "\n".join(lines)
