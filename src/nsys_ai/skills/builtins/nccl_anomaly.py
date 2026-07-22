"""NCCL anomaly detection — finds outlier collective operations.

Beyond the basic nccl_breakdown (aggregated stats), this skill identifies
individual NCCL operations whose duration is significantly above the
average for their operation type — indicating potential network stalls,
GPU imbalance, or contention.

The per-op-type average and outlier selection are computed in Python from a
single flat fetch rather than a multi-CTE self-join. The join a SQL version
needs (every op against its type's average) re-materialises the NCCL scan and
crashes / hangs DuckDB's ``sqlite_scanner`` on a direct-attached profile with
no parquet cache (issue #251) — the same failure mode as host_sync (#248). The
flat fetch runs fine on every path.
"""

from ...connection import DB_ERRORS
from ..base import Skill, SkillParam

# Checked in order: more specific names first, so "AllReduce" and
# "ReduceScatter" are classified before the bare "Reduce" substring.
_OP_TYPES = ("AllReduce", "AllGather", "ReduceScatter", "Broadcast", "AllToAll", "Reduce")


def _op_type(name: str) -> str:
    for op in _OP_TYPES:
        if op in name:
            return op
    return "Other"


def _execute(conn, **kwargs):
    from ...connection import wrap_connection

    try:
        threshold = float(kwargs.get("threshold", 3.0))
    except (TypeError, ValueError):
        threshold = 3.0
    try:
        limit = int(kwargs.get("limit", 20))
    except (TypeError, ValueError):
        limit = 20

    adapter = wrap_connection(conn)
    kernel_table = adapter.resolve_activity_tables().get("kernel", "CUPTI_ACTIVITY_KIND_KERNEL")

    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    trim_clause = ""
    params: list = []
    if trim_start is not None and trim_end is not None:
        trim_clause = "AND k.start >= ? AND k.[end] <= ?"
        params = [int(trim_start), int(trim_end)]

    # Flat fetch of every NCCL collective. LIKE is lower-cased-agnostic on
    # SQLite but case-sensitive on DuckDB, so match both spellings.
    sql = f"""
        SELECT s.value AS name, k.streamId AS stream_id, k.start AS start,
               (k.[end] - k.start) AS dur_ns
        FROM {kernel_table} k
        JOIN StringIds s ON k.shortName = s.id
        WHERE (s.value LIKE '%nccl%' OR s.value LIKE '%NCCL%') {trim_clause}
    """
    try:
        rows = adapter.execute(sql, params).fetchall()
    except DB_ERRORS as exc:
        return [
            {
                "error": (
                    f"nccl_anomaly query failed: {exc}. "
                    "The kernel table or StringIds may be absent."
                )
            }
        ]

    # Group by op type: running sum + count for the average, and keep the ops.
    ops = []
    sum_ns: dict[str, int] = {}
    count: dict[str, int] = {}
    for name, stream_id, start, dur_ns in rows:
        if name is None:
            continue
        ot = _op_type(name)
        dur_ns = int(dur_ns or 0)
        ops.append((ot, name, stream_id, start, dur_ns))
        sum_ns[ot] = sum_ns.get(ot, 0) + dur_ns
        count[ot] = count.get(ot, 0) + 1

    out = []
    for ot, name, stream_id, start, dur_ns in ops:
        n = count[ot]
        avg = sum_ns[ot] / n if n else 0.0
        if avg <= 0 or dur_ns <= avg * threshold:
            continue
        out.append(
            {
                "op_type": ot,
                "name": name,
                "dur_ns": dur_ns,
                "dur_ms": round(dur_ns / 1e6, 3),
                "avg_ms": round(avg / 1e6, 3),
                "ratio_to_avg": round(dur_ns / avg, 1),
                "start": start,
                "streamId": stream_id,
                "total_count": n,
            }
        )

    # Slowest first; a deterministic secondary key keeps ties reproducible
    # (the SQL left tie order unspecified).
    out.sort(key=lambda r: (-r["dur_ns"], r["start"]))
    return out[:limit]


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


SKILL = Skill(
    name="nccl_anomaly",
    title="NCCL Anomaly Detection",
    description=(
        "Detects outlier NCCL collective operations whose duration exceeds "
        "a threshold relative to the average for their op type. "
        "Identifies network stalls, GPU imbalance, and contention. "
        "Returns individual anomalous operations with timing and context."
    ),
    category="communication",
    execute_fn=_execute,
    format_fn=_format,
    params=[
        SkillParam(
            "threshold", "Anomaly threshold: ratio to average duration", "float", False, 3.0
        ),
        SkillParam("limit", "Max anomalies to return", "int", False, 20),
    ],
    tags=["nccl", "anomaly", "outlier", "stall", "communication", "distributed"],
)
