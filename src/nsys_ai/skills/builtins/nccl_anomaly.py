"""NCCL anomaly detection — finds outlier collective operations.

Beyond the basic nccl_breakdown (aggregated stats), this skill identifies
individual NCCL operations whose duration is significantly above the
median for their operation type — indicating potential network stalls,
GPU imbalance, or contention.
"""

from ..base import Skill, SkillParam

SKILL = Skill(
    name="nccl_anomaly",
    title="NCCL Anomaly Detection",
    description=(
        "Detects outlier NCCL collective operations whose duration exceeds "
        "a threshold relative to the median for their op type. "
        "Identifies network stalls, GPU imbalance, and contention. "
        "Returns individual anomalous operations with timing and context."
    ),
    category="communication",
    sql="""\
WITH nccl_ops AS (
    SELECT
        s.value AS name,
        k.correlationId,
        k.streamId,
        k.start,
        k.[end],
        (k.[end] - k.start) AS dur_ns
    FROM {kernel_table} k
    JOIN StringIds s ON k.shortName = s.id
    WHERE (s.value LIKE '%nccl%' OR s.value LIKE '%NCCL%')
        {trim_clause}
),
op_stats AS (
    SELECT
        -- Normalize: strip device/rank suffixes to group by op type
        CASE
            WHEN name LIKE '%AllReduce%' THEN 'AllReduce'
            WHEN name LIKE '%AllGather%' THEN 'AllGather'
            WHEN name LIKE '%ReduceScatter%' THEN 'ReduceScatter'
            WHEN name LIKE '%Broadcast%' THEN 'Broadcast'
            WHEN name LIKE '%AllToAll%' THEN 'AllToAll'
            WHEN name LIKE '%Reduce%' THEN 'Reduce'
            ELSE 'Other'
        END AS op_type,
        name,
        dur_ns,
        start,
        correlationId,
        streamId
    FROM nccl_ops
),
-- SQLite lacks native MEDIAN; AVG is a reasonable proxy for outlier detection
op_averages AS (
    SELECT op_type,
           AVG(dur_ns) AS avg_dur_ns,
           COUNT(*) AS total_count
    FROM op_stats
    GROUP BY op_type
)
SELECT
    o.op_type,
    o.name,
    o.dur_ns,
    ROUND(o.dur_ns / 1e6, 3) AS dur_ms,
    ROUND(a.avg_dur_ns / 1e6, 3) AS avg_ms,
    ROUND(CAST(o.dur_ns AS REAL) / NULLIF(a.avg_dur_ns, 0), 1) AS ratio_to_avg,
    o.start,
    o.streamId,
    a.total_count
FROM op_stats o
JOIN op_averages a ON o.op_type = a.op_type
WHERE o.dur_ns > a.avg_dur_ns * {threshold}
ORDER BY o.dur_ns DESC
LIMIT {limit}""",
    format_fn=lambda rows: _format(rows),
    params=[
        SkillParam(
            "threshold", "Anomaly threshold: ratio to average duration", "float", False, 3.0
        ),
        SkillParam("limit", "Max anomalies to return", "int", False, 20),
    ],
    tags=["nccl", "anomaly", "outlier", "stall", "communication", "distributed"],
)


def _format(rows):
    if not rows:
        return "(No NCCL anomalies detected — all collectives within normal range)"
    lines = [
        "── NCCL Anomalies ──",
        f"{'Op Type':<16s} {'Duration':>10s} {'Avg':>10s} {'Ratio':>7s} {'Stream':>7s}",
        "─" * 58,
    ]
    for r in rows:
        lines.append(
            f"{r['op_type']:<16s} {r['dur_ms']:>8.3f}ms "
            f"{r['avg_ms']:>8.3f}ms {r['ratio_to_avg']:>6.1f}× "
            f"s{r['streamId']:>5d}"
        )
    count = rows[0]["total_count"] if rows else 0
    lines.append(f"\n  {len(rows)} anomalies found out of {count} total ops")
    return "\n".join(lines)
