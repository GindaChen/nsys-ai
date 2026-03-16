"""CPU-GPU pipeline analysis.

Analyzes the relationship between CPU-side kernel dispatch and GPU-side
kernel execution to determine if the CPU is keeping the GPU fed.

Key metrics:
- CPU dispatch lead time: how far ahead is CPU vs GPU
- GPU starvation events: when GPU runs out of queued work
- Per-thread launch contribution: which threads feed the GPU
"""

from ..base import Skill, SkillParam

SKILL = Skill(
    name="cpu_gpu_pipeline",
    title="CPU-GPU Pipeline Analysis",
    description=(
        "Analyzes the CPU-to-GPU dispatch pipeline: measures how far ahead "
        "CPU dispatch is relative to GPU execution, identifies GPU starvation "
        "events (where GPU runs out of queued work), and per-thread launch "
        "contribution. Detects CPU bottlenecks and pipeline bubbles."
    ),
    category="system",
    sql="""\
WITH runtime_kernel AS (
    SELECT
        r.globalTid AS cpu_tid,
        r.start AS cpu_dispatch_start,
        r.[end] AS cpu_dispatch_end,
        k.start AS gpu_start,
        k.[end] AS gpu_end,
        (k.start - r.[end]) AS queue_delay_ns,
        (k.[end] - k.start) AS gpu_dur_ns
    FROM {runtime_table} r
    JOIN {kernel_table} k ON r.correlationId = k.correlationId
    WHERE r.[end] < k.start  -- CPU dispatch finishes before GPU starts
        {trim_clause}
),
per_thread AS (
    SELECT
        cpu_tid,
        COUNT(*) AS dispatches,
        ROUND(AVG(queue_delay_ns) / 1e3, 1) AS avg_queue_delay_us,
        ROUND(MAX(queue_delay_ns) / 1e3, 1) AS max_queue_delay_us,
        ROUND(MIN(queue_delay_ns) / 1e3, 1) AS min_queue_delay_us,
        SUM(CASE WHEN queue_delay_ns > 1000000 THEN 1 ELSE 0 END) AS starvation_events,
        ROUND(AVG(gpu_dur_ns) / 1e3, 1) AS avg_kernel_us
    FROM runtime_kernel
    GROUP BY cpu_tid
)
SELECT
    cpu_tid,
    dispatches,
    avg_queue_delay_us,
    min_queue_delay_us,
    max_queue_delay_us,
    starvation_events,
    avg_kernel_us,
    ROUND(CAST(dispatches AS REAL) / (SELECT SUM(dispatches) FROM per_thread) * 100, 1)
        AS pct_of_dispatches
FROM per_thread
ORDER BY dispatches DESC
LIMIT {limit}""",
    format_fn=lambda rows: _format(rows),
    params=[
        SkillParam("limit", "Max threads to show", "int", False, 10),
    ],
    tags=["cpu", "gpu", "pipeline", "dispatch", "starvation", "bottleneck", "queue"],
)


def _format(rows):
    if not rows:
        return "(No CPU-GPU dispatch data — requires Runtime + Kernel correlation)"
    lines = [
        "── CPU-GPU Pipeline Analysis ──",
        f"{'Thread':>12s} {'Dispatches':>11s} {'AvgQueue':>10s} "
        f"{'MaxQueue':>10s} {'Starve':>7s} {'%Total':>7s}",
        "─" * 65,
    ]
    total_starvation = 0
    for r in rows:
        total_starvation += r["starvation_events"]
        lines.append(
            f"{r['cpu_tid']:>12d} {r['dispatches']:>11d} "
            f"{r['avg_queue_delay_us']:>8.1f}µs "
            f"{r['max_queue_delay_us']:>8.1f}µs "
            f"{r['starvation_events']:>7d} "
            f"{r['pct_of_dispatches']:>6.1f}%"
        )
    lines.append(f"\n  Total GPU starvation events (queue > 1ms): {total_starvation}")
    if total_starvation > 10:
        lines.append(
            "  ⚠ High starvation count — CPU may not be able to keep GPU fed. "
            "Check for Python GIL contention, heavy preprocessing, or explicit syncs."
        )
    return "\n".join(lines)
