"""Kernel launch pattern analysis.

Goes beyond kernel_launch_overhead (which only measures individual gaps)
to analyze the pattern of kernel dispatches:
- Burst vs trickle launch patterns
- Sync points where GPU goes idle
- Average dispatch rate (kernels/ms)
- Longest gap between launches per stream
"""

from ..base import Skill, SkillParam

SKILL = Skill(
    name="kernel_launch_pattern",
    title="Kernel Launch Pattern Analysis",
    description=(
        "Analyzes kernel dispatch patterns per stream: dispatch rate, "
        "burst density, inter-launch gaps, and sync-stall detection. "
        "Identifies whether the CPU is keeping the GPU fed or starving it."
    ),
    category="kernels",
    sql="""\
WITH launch_gaps AS (
    SELECT
        k.streamId,
        k.start,
        k.[end],
        (k.[end] - k.start) AS dur_ns,
        LAG(k.[end]) OVER (
            PARTITION BY k.streamId ORDER BY k.start, k.correlationId
        ) AS prev_end,
        LEAD(k.start) OVER (
            PARTITION BY k.streamId ORDER BY k.start, k.correlationId
        ) AS next_start
    FROM {kernel_table} k
    WHERE 1=1
        {trim_clause}
),
stream_stats AS (
    SELECT
        streamId,
        COUNT(*) AS kernel_count,
        MIN(start) AS first_start,
        MAX([end]) AS last_end,
        ROUND(SUM(dur_ns) / 1e6, 2) AS total_kernel_ms,
        ROUND(AVG(dur_ns) / 1e3, 1) AS avg_kernel_us,
        ROUND(MAX(CASE WHEN prev_end IS NOT NULL THEN start - prev_end ELSE 0 END) / 1e6, 3)
            AS max_gap_ms,
        ROUND(AVG(CASE WHEN prev_end IS NOT NULL AND start > prev_end THEN start - prev_end ELSE NULL END) / 1e3, 1)
            AS avg_gap_us,
        SUM(CASE WHEN prev_end IS NOT NULL AND (start - prev_end) > 1000000 THEN 1 ELSE 0 END)
            AS sync_stalls
    FROM launch_gaps
    GROUP BY streamId
)
SELECT
    streamId,
    kernel_count,
    ROUND((last_end - first_start) / 1e6, 2) AS span_ms,
    total_kernel_ms,
    ROUND(CAST(kernel_count AS DOUBLE) / NULLIF((last_end - first_start) / 1e6, 0), 1) AS dispatch_rate_per_ms,
    ROUND(total_kernel_ms / NULLIF((last_end - first_start) / 1e6, 0) * 100, 1) AS occupancy_pct,
    avg_kernel_us,
    max_gap_ms,
    avg_gap_us,
    sync_stalls
FROM stream_stats
ORDER BY kernel_count DESC, streamId ASC
LIMIT {limit}""",
    format_fn=lambda rows: _format(rows),
    params=[
        SkillParam("limit", "Max streams to show", "int", False, 10),
    ],
    tags=["kernel", "launch", "pattern", "dispatch", "sync", "stall", "rate"],
)


def _format(rows):
    if not rows:
        return "(No kernel launch data found)"
    lines = [
        "── Kernel Launch Patterns ──",
        f"{'Stream':>7s} {'Kernels':>8s} {'Span(ms)':>10s} "
        f"{'Rate/ms':>8s} {'Occ%':>6s} {'MaxGap':>10s} {'Stalls':>7s}",
        "─" * 65,
    ]
    for r in rows:
        lines.append(
            f"s{r['streamId']:>5d} {r['kernel_count']:>8d} "
            f"{r['span_ms']:>10.2f} {r['dispatch_rate_per_ms']:>8.1f} "
            f"{r['occupancy_pct']:>5.1f}% {r['max_gap_ms']:>8.3f}ms "
            f"{r['sync_stalls']:>7d}"
        )
    return "\n".join(lines)
