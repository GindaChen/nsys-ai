"""Detect GPU idle gaps (bubbles) between consecutive kernel executions.

Enhanced with:
- Aggregation stats: total idle time, % of profile, distribution buckets
- CPU attribution: what CUDA Runtime APIs were active during each gap
"""

import logging
import sqlite3

from ..base import Skill, SkillParam, _resolve_activity_tables, ensure_indexes

_log = logging.getLogger(__name__)

# Gap classification rules based on dominant CUDA Runtime API during the gap
_GAP_CLASSIFICATIONS = [
    # (api_substring, category, description)
    ("cudaDeviceSynchronize", "synchronization", "Explicit GPU sync stall"),
    ("cudaStreamSynchronize", "synchronization", "Stream sync stall"),
    ("cudaEventSynchronize", "synchronization", "Event sync stall"),
    ("cudaMemcpy", "memory_transfer", "Blocked on memory transfer"),
    ("cudaMemset", "memory_transfer", "Blocked on memory set"),
    ("cudaLaunchKernel", "kernel_launch", "Kernel launch overhead"),
]


def _classify_gap_apis(api_names: list[str]) -> tuple[str, str]:
    """Classify a gap based on the dominant CUDA Runtime APIs observed.

    Returns (category, description).
    """
    for api_sub, category, desc in _GAP_CLASSIFICATIONS:
        for api in api_names:
            if api_sub in api:
                return category, desc
    if not api_names:
        return "cpu_stall", "No CUDA API activity — possible DataLoader / GIL / I/O wait"
    return "unknown", "Unclassified CUDA API activity"


def _execute(conn: sqlite3.Connection, **kwargs):
    """Execute GPU idle gaps analysis with aggregation and CPU attribution."""
    # Ensure row_factory is set for dict-like access from raw SQL
    old_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        return _execute_inner(conn, **kwargs)
    finally:
        conn.row_factory = old_factory


def _execute_inner(conn: sqlite3.Connection, **kwargs):
    """Inner implementation (row_factory managed by _execute)."""
    ensure_indexes(conn)

    tables = _resolve_activity_tables(conn)
    kernel_tbl = tables.get("kernel")
    runtime_tbl = tables.get("runtime")
    if not kernel_tbl:
        return []

    min_gap_ns = int(kwargs.get("min_gap_ns", 1_000_000))
    limit = int(kwargs.get("limit", 20))

    # Build trim clause
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    trim_clause = ""
    trim_params: list = []
    if trim_start is not None and trim_end is not None:
        trim_clause = "AND k.start >= ? AND k.[end] <= ?"
        trim_params = [int(trim_start), int(trim_end)]

    # --- Phase 1: Find all gaps ---
    gap_sql = f"""\
WITH ordered AS (
    SELECT k.streamId,
           k.start, k.[end],
           s.value AS kernel_name,
           LAG(k.[end]) OVER (PARTITION BY k.streamId ORDER BY k.start) AS prev_end,
           LAG(s.value) OVER (PARTITION BY k.streamId ORDER BY k.start) AS prev_kernel
    FROM {kernel_tbl} k
    JOIN StringIds s ON k.shortName = s.id
    WHERE 1=1 {trim_clause}
)
SELECT streamId,
       prev_end AS start_ns,
       start AS end_ns,
       (start - prev_end) AS gap_ns,
       prev_kernel AS before_kernel,
       kernel_name AS after_kernel
FROM ordered
WHERE prev_end IS NOT NULL AND (start - prev_end) > ?
ORDER BY gap_ns DESC
LIMIT ?"""
    try:
        rows = [
            dict(r) for r in conn.execute(gap_sql, trim_params + [min_gap_ns, limit]).fetchall()
        ]
    except sqlite3.OperationalError as e:
        _log.debug("gpu_idle_gaps: %s", e)
        return []

    if not rows:
        return []

    # --- Phase 2: Aggregation stats ---
    # Re-query without LIMIT to get full aggregation
    agg_sql = f"""\
WITH ordered AS (
    SELECT k.streamId,
           k.start, k.[end],
           LAG(k.[end]) OVER (PARTITION BY k.streamId ORDER BY k.start) AS prev_end
    FROM {kernel_tbl} k
    WHERE 1=1 {trim_clause}
)
SELECT COUNT(*) AS gap_count,
       SUM(start - prev_end) AS total_gap_ns,
       SUM(CASE WHEN (start - prev_end) BETWEEN ? AND 5000000 THEN 1 ELSE 0 END) AS gaps_1_5ms,
       SUM(CASE WHEN (start - prev_end) BETWEEN 5000001 AND 50000000 THEN 1 ELSE 0 END) AS gaps_5_50ms,
       SUM(CASE WHEN (start - prev_end) > 50000000 THEN 1 ELSE 0 END) AS gaps_gt50ms
FROM ordered
WHERE prev_end IS NOT NULL AND (start - prev_end) > ?"""
    try:
        agg = dict(conn.execute(agg_sql, trim_params + [min_gap_ns, min_gap_ns]).fetchone())
    except sqlite3.OperationalError:
        agg = {}

    # Profile time range for percentage calculation
    # Gaps are summed across ALL streams, so we need to normalize by
    # the number of active streams to avoid pct > 100%.
    try:
        time_range = conn.execute(
            f"SELECT MIN(start), MAX([end]) FROM {kernel_tbl} WHERE 1=1 {trim_clause}", trim_params
        ).fetchone()
        profile_span_ns = (time_range[1] or 0) - (time_range[0] or 0)
        stream_count_row = conn.execute(
            f"SELECT COUNT(DISTINCT streamId) AS n FROM {kernel_tbl} WHERE 1=1 {trim_clause}", trim_params
        ).fetchone()
        n_streams = stream_count_row["n"] if stream_count_row else 1
    except sqlite3.OperationalError:
        profile_span_ns = 0
        n_streams = 1

    total_gap_ns = agg.get("total_gap_ns") or 0
    # Normalize: total_gap across N streams / (span × N streams)
    effective_span = profile_span_ns * max(n_streams, 1)
    pct_of_profile = (
        min(round(100 * total_gap_ns / effective_span, 1), 100.0) if effective_span > 0 else 0
    )

    summary = {
        "_summary": True,
        "gap_count": agg.get("gap_count") or 0,
        "total_idle_ms": round(total_gap_ns / 1e6, 2),
        "pct_of_profile": pct_of_profile,
        "gaps_1_5ms": agg.get("gaps_1_5ms") or 0,
        "gaps_5_50ms": agg.get("gaps_5_50ms") or 0,
        "gaps_gt50ms": agg.get("gaps_gt50ms") or 0,
    }

    # --- Phase 3: CPU attribution for top 5 gaps ---
    if runtime_tbl:
        top_n = min(5, len(rows))
        for gap in rows[:top_n]:
            gap_start = gap["start_ns"]
            gap_end = gap["end_ns"]
            try:
                api_rows = conn.execute(
                    f"""\
SELECT s.value AS api_name, COUNT(*) AS call_count,
       SUM(r.[end] - r.start) AS total_ns
FROM {runtime_tbl} r
JOIN StringIds s ON r.nameId = s.id
WHERE r.start < ? AND r.[end] > ?
GROUP BY s.value
ORDER BY total_ns DESC
LIMIT 5""",
                    (gap_end, gap_start),
                ).fetchall()
                apis = [dict(r) for r in api_rows]
            except sqlite3.OperationalError:
                apis = []

            api_names = [a["api_name"] for a in apis]
            category, description = _classify_gap_apis(api_names)
            gap["attribution"] = {
                "category": category,
                "description": description,
                "top_apis": [
                    {"name": a["api_name"], "total_ms": round(a["total_ns"] / 1e6, 2)}
                    for a in apis[:3]
                ],
            }

    # Return: gap rows + summary as last element
    return rows + [summary]


def _format(rows):
    if not rows:
        return "(No significant GPU idle gaps found — GPU is well-utilized)"

    # Separate data rows from summary
    data_rows = [r for r in rows if not r.get("_summary")]
    summary = next((r for r in rows if r.get("_summary")), None)

    if not data_rows:
        return "(No significant GPU idle gaps found — GPU is well-utilized)"

    lines = ["── GPU Idle Gaps (Bubbles) ──"]

    # Summary header
    if summary:
        lines.append(
            f"  Total: {summary['gap_count']} gaps, "
            f"{summary['total_idle_ms']:.1f}ms idle "
            f"({summary['pct_of_profile']}% of profile)"
        )
        lines.append(
            f"  Distribution: "
            f"{summary['gaps_1_5ms']} × 1-5ms, "
            f"{summary['gaps_5_50ms']} × 5-50ms, "
            f"{summary['gaps_gt50ms']} × >50ms"
        )
        lines.append("")

    lines.append(f"{'Stream':>7s}  {'Gap(ms)':>9s}  {'Before Kernel':<40s}  {'Attribution':<30s}")
    lines.append("─" * 92)

    for r in data_rows:
        before = r.get("before_kernel") or "(start of stream)"
        if len(before) > 38:
            before = before[:35] + "..."

        attr = r.get("attribution", {})
        attr_text = attr.get("category", "") if attr else ""

        lines.append(
            f"{r['streamId']:>7d}  {r['gap_ns'] / 1e6:>9.3f}  {before:<40s}  {attr_text:<30s}"
        )

        # Show top API if attribution exists
        top_apis = attr.get("top_apis", []) if attr else []
        for api in top_apis[:2]:
            lines.append(f"{'':>7s}  {'':>9s}  └─ {api['name']}: {api['total_ms']:.2f}ms")

    lines.append("\n  💡 TIP: Large gaps (>1ms) often indicate the GPU is starved waiting for CPU")
    lines.append(
        "     (e.g., DataLoader blocking, explicit synchronization) or waiting on PCIe transfers."
    )
    return "\n".join(lines)


SKILL = Skill(
    name="gpu_idle_gaps",
    title="GPU Idle Gaps (Bubbles)",
    description=(
        "Finds idle gaps between consecutive GPU kernels on each stream — "
        "the 'bubbles' in the pipeline. Includes aggregation stats (total idle time, "
        "% of profile, distribution) and CPU attribution for top gaps "
        "(identifies what CUDA APIs were active during the gap). "
        "These are prime optimization targets."
    ),
    category="kernels",
    execute_fn=_execute,
    params=[
        SkillParam("min_gap_ns", "Minimum gap in nanoseconds to report", "int", False, 1000000),
        SkillParam("limit", "Max results", "int", False, 20),
    ],
    format_fn=_format,
    tags=["bubble", "idle", "gap", "pipeline", "stall", "utilization", "attribution"],
)
