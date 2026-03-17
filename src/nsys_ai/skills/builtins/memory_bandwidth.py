"""Memory bandwidth and utilization analysis.

Analyzes CUDA memory copy operations to compute:
- Per-direction bandwidth (H2D, D2H, D2D, P2P)
- Peak vs sustained bandwidth
- Large transfer identification
- Total bytes moved

Goes beyond the basic memory_transfers skill which only does direction aggregation.
"""

from ..base import Skill

_COPY_KIND_NAMES = {
    0: "Unknown",
    1: "H2D",
    2: "D2H",
    4: "H2H",
    8: "D2D",
    10: "P2P",
}


SKILL = Skill(
    name="memory_bandwidth",
    title="Memory Bandwidth & Utilization Analysis",
    description=(
        "Analyzes CUDA memory operations with bandwidth computation: "
        "per-direction throughput (GB/s), peak vs sustained bandwidth, "
        "large-transfer identification, and total bytes moved. "
        "Identifies memory bottlenecks and inefficient transfers."
    ),
    category="memory",
    sql="""\
WITH ranked AS (
    SELECT
        copyKind,
        bytes,
        (k.[end] - k.start) AS dur_ns
    FROM {memcpy_table} k
    WHERE 1=1
        {trim_clause}
)
SELECT
    r.copyKind,
    COUNT(*) AS op_count,
    ROUND(SUM(r.bytes) / 1e6, 2) AS total_mb,
    ROUND(AVG(r.bytes) / 1e3, 1) AS avg_kb,
    ROUND(MAX(r.bytes) / 1e6, 2) AS max_mb,
    ROUND(SUM(r.dur_ns) / 1e6, 2) AS total_dur_ms,
    ROUND(AVG(r.dur_ns) / 1e3, 1) AS avg_dur_us,
    CASE
        WHEN SUM(r.dur_ns) > 0
        THEN ROUND(SUM(r.bytes) / (SUM(r.dur_ns) / 1e9) / 1e9, 2)
        ELSE 0
    END AS avg_bandwidth_gbps,
    COALESCE(
        ROUND(
            MAX(
                CASE
                    WHEN r.dur_ns > 0
                    THEN r.bytes / (r.dur_ns / 1e9) / 1e9
                END
            ),
            2
        ),
        0
    ) AS peak_bandwidth_gbps
FROM ranked r
GROUP BY r.copyKind
ORDER BY total_mb DESC""",
    format_fn=lambda rows: _format(rows),
    params=[],
    tags=["memory", "bandwidth", "memcpy", "transfer", "H2D", "D2H", "utilization"],
)


def _format(rows):
    if not rows:
        return "(No memory copy operations found)"
    lines = [
        "── Memory Bandwidth Analysis ──",
        f"{'Direction':<10s} {'Count':>7s} {'Total MB':>10s} "
        f"{'Avg KB':>8s} {'Avg BW':>10s} {'Peak BW':>10s}",
        "─" * 65,
    ]
    for r in rows:
        kind = _COPY_KIND_NAMES.get(r["copyKind"], f"Kind{r['copyKind']}")
        lines.append(
            f"{kind:<10s} {r['op_count']:>7d} {r['total_mb']:>10.2f} "
            f"{r['avg_kb']:>8.1f} {r['avg_bandwidth_gbps']:>8.2f}GB/s "
            f"{r['peak_bandwidth_gbps']:>8.2f}GB/s"
        )
    return "\n".join(lines)
