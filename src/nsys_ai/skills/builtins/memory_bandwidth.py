"""Memory bandwidth and utilization analysis.

Analyzes CUDA memory copy operations to compute:
- Per-direction bandwidth (H2D, D2H, D2D, P2P)
- Peak vs sustained bandwidth
- Large transfer identification
- Total bytes moved

Goes beyond the basic memory_transfers skill which only does direction aggregation.
"""

from ..base import Skill, SkillParam

_COPY_KIND_NAMES = {
    0: "Unknown",
    1: "H2D",
    2: "D2H",
    4: "H2H",
    8: "D2D",
    10: "P2P",
}


def _execute(conn, **kwargs):
    import sqlite3

    import duckdb

    from ...profile import Profile

    prof = Profile._from_conn(conn)
    device = int(kwargs.get("device", 0))
    limit = int(kwargs.get("limit", 5))

    memcpy_table = None
    for t in prof.schema.tables:
        if t == "CUPTI_ACTIVITY_KIND_MEMCPY" or t.startswith("CUPTI_ACTIVITY_KIND_MEMCPY"):
            memcpy_table = t
            break
    if not memcpy_table:
        return []

    trim_clause = ""
    trim = None
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    trim_params = []
    if trim_start is not None and trim_end is not None:
        trim_clause = " AND k.[end] >= ? AND k.start <= ? "
        trim_params = [int(trim_start), int(trim_end)]
        trim = (int(trim_start), int(trim_end))

    # 1. Bandwidth aggregation
    agg_sql = f"""\
WITH ranked AS (
    SELECT copyKind, bytes, (k.[end] - k.start) AS dur_ns
    FROM {memcpy_table} k
    WHERE k.deviceId = ? {trim_clause}
)
SELECT r.copyKind, COUNT(*) AS op_count,
       ROUND(SUM(r.bytes) / 1e6, 2) AS total_mb,
       ROUND(AVG(r.bytes) / 1e3, 1) AS avg_kb,
       ROUND(MAX(r.bytes) / 1e6, 2) AS max_mb,
       ROUND(SUM(r.dur_ns) / 1e6, 2) AS total_dur_ms,
       ROUND(AVG(r.dur_ns) / 1e3, 1) AS avg_dur_us,
       CASE WHEN SUM(r.dur_ns) > 0 THEN ROUND(SUM(r.bytes) / (SUM(r.dur_ns) / 1e9) / 1e9, 2) ELSE 0 END AS avg_bandwidth_gbps,
       COALESCE(ROUND(MAX(CASE WHEN r.dur_ns > 0 THEN r.bytes / (r.dur_ns / 1e9) / 1e9 END), 2), 0) AS peak_bandwidth_gbps
FROM ranked r
GROUP BY r.copyKind
ORDER BY total_mb DESC"""

    try:
        rows = prof._duckdb_query(agg_sql, [device] + trim_params)
    except (sqlite3.Error, duckdb.Error):
        return []

    if not rows:
        return []

    # 2. Large transfer anomalies (for evidence/findings)
    anomaly_sql = f"""\
SELECT copyKind, bytes, start, [end], ([end] - start) AS dur_ns
FROM {memcpy_table}
WHERE deviceId = ? AND bytes > 10000000
""" + (" AND [end] >= ? AND start <= ? " if trim else "") + f"""
ORDER BY dur_ns DESC
LIMIT {limit}"""

    anomalies = []
    try:
        anomalies = prof._duckdb_query(anomaly_sql, [device] + (trim_params if trim_params else []))
    except (sqlite3.Error, duckdb.Error):
        pass

    rows[0]["_anomalies"] = anomalies
    rows[0]["device_id"] = device
    return rows


def _to_findings(rows: list[dict]) -> list:
    from nsys_ai.annotation import Finding
    findings = []
    if not rows:
        return findings

    anomalies = rows[0].get("_anomalies", [])
    device = rows[0].get("device_id", 0)

    for r in anomalies:
        kind = _COPY_KIND_NAMES.get(r["copyKind"], f"Kind{r['copyKind']}")
        mb = r["bytes"] / 1e6
        dur_ms = r["dur_ns"] / 1e6
        gbps = (r["bytes"] / max(r["dur_ns"], 1) * 1e9) / 1e9
        findings.append(
            Finding(
                type="highlight",
                label=f"Large {kind} Transfer ({mb:.1f}MB)",
                start_ns=int(r["start"]),
                end_ns=int(r["end"]),
                gpu_id=device,
                severity="warning" if dur_ms > 1.0 else "info",
                note=f"{kind}: {mb:.1f}MB in {dur_ms:.2f}ms ({gbps:.1f}GB/s)",
            )
        )
    return findings


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
    execute_fn=_execute,
    format_fn=lambda rows: _format(rows),
    to_findings_fn=_to_findings,
    params=[
        SkillParam("device", "GPU device ID", "int", False, 0),
        SkillParam("limit", "Num anomalies to report", "int", False, 5)
    ],
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
