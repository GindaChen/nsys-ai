"""Sync Cost Analysis skill.

Examines CUPTI_ACTIVITY_KIND_SYNCHRONIZATION to compute the true wall-clock
duration that the CPU was blocked by synchronizations, distinguishing explicit pipeline
stalls from naturally overlapping communications.
"""

from collections import defaultdict

from ...connection import DB_ERRORS, wrap_connection
from ..base import Skill, _compute_interval_union


def _resolve_table_name(conn, candidate: str) -> str:
    """Resolve an Nsight table name, allowing for version-suffixed variants."""
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name LIKE ?
            ORDER BY name DESC LIMIT 1
            """,
            (candidate + "%",),
        )
        row = cursor.fetchone()
        if row:
            return row[0]
    except Exception:
        pass
    return candidate


_sync_result_cache: dict[tuple, list[dict]] = {}


def _execute_sync_analysis(conn, **kwargs) -> list[dict]:
    # Module-level cache keyed by (connection identity, trim window).
    # Avoids redundant re-execution when called by multiple consumer skills
    # (manifest, root_cause, overlap, bubble) within the same profile session.
    cache_key = (
        id(conn),
        kwargs.get("trim_start_ns"),
        kwargs.get("trim_end_ns"),
    )
    cached = _sync_result_cache.get(cache_key)
    if cached is not None:
        return cached

    result = _execute_sync_analysis_impl(conn, **kwargs)
    _sync_result_cache[cache_key] = result
    return result


def _execute_sync_analysis_impl(conn, **kwargs) -> list[dict]:
    adapter = wrap_connection(conn)
    sync_table = _resolve_table_name(conn, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION")
    type_table = _resolve_table_name(conn, "ENUM_CUPTI_SYNC_TYPE")

    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")

    if trim_start is None or trim_end is None:
        try:
            # Attempt to retrieve true profile bounds to clamp host events
            from ...profile import Profile
            prof = Profile._from_conn(conn)
            p_start, p_end = prof.meta.time_range
            trim_start = trim_start if trim_start is not None else p_start
            trim_end = trim_end if trim_end is not None else p_end
        except Exception:
            pass

    conds = ["1=1"]
    params = []

    # We must explicitly clamp within SQL or Python. We'll do it in Python,
    # but we can filter eagerly in SQL to limit data loading overhead.
    if trim_start is not None:
        conds.append("s.[end] >= ?")
        params.append(int(trim_start))
    if trim_end is not None:
        conds.append("s.start <= ?")
        params.append(int(trim_end))

    where_clause = " AND ".join(conds)

    query = f"""
    SELECT
        s.start,
        s.[end],
        s.globalPid,
        COALESCE(e.name, 'Unknown') AS sync_type_name
    FROM {sync_table} s
    LEFT JOIN {type_table} e ON s.syncType = e.id
    WHERE {where_clause}
    """

    try:
        cur = adapter.execute(query, params)
        rows = cur.fetchall()
    except DB_ERRORS:
        # Table might not exist in old profiles
        return [{
            "total_sync_wall_ms": 0.0,
            "sync_by_type_ms": {},
            "profile_span_ms": 0.0,
            "sync_density_pct": 0.0,
            "error": "Synchronization tables not found in profile"
        }]

    # Filter and clamp intervals
    global_intervals = []
    type_intervals = defaultdict(list)

    for row in rows:
        s_start, s_end = int(row[0]), int(row[1])
        sync_type_name = str(row[3])

        # Explicit clamping to trim window
        if trim_start is not None:
            s_start = max(s_start, int(trim_start))
        if trim_end is not None:
            s_end = min(s_end, int(trim_end))

        if s_start < s_end:
            global_intervals.append((s_start, s_end))
            type_intervals[sync_type_name].append((s_start, s_end))

    total_sync_wall_ns = _compute_interval_union(global_intervals)
    sync_by_type_ms = {}
    for t_name, intervals in type_intervals.items():
        type_ns = _compute_interval_union(intervals)
        sync_by_type_ms[t_name] = round(type_ns / 1e6, 3)

    profile_span_ns = 0
    if trim_start is not None and trim_end is not None:
        profile_span_ns = max(0, int(trim_end) - int(trim_start))

    # Fallback to absolute min/max if no span provided
    if profile_span_ns == 0 and global_intervals:
        min_s = min([ival[0] for ival in global_intervals])
        max_e = max([ival[1] for ival in global_intervals])
        profile_span_ns = max_e - min_s

    total_sync_wall_ms = total_sync_wall_ns / 1e6
    profile_span_ms = profile_span_ns / 1e6

    sync_density_pct = 0.0
    if profile_span_ms > 0:
        sync_density_pct = (total_sync_wall_ms / profile_span_ms) * 100

    return [{
        "total_sync_wall_ms": round(total_sync_wall_ms, 3),
        "sync_by_type_ms": sync_by_type_ms,
        "profile_span_ms": round(profile_span_ms, 3),
        "sync_density_pct": round(sync_density_pct, 2)
    }]


def _format_sync_analysis(rows: list[dict]) -> str:
    if not rows:
        return "(No synchronization data)"

    m = rows[0]
    if "error" in m:
        return f"Sync Cost Analysis Error: {m['error']}"

    lines = ["══ Sync Cost Analysis ══"]
    lines.append(f"  Total Wall-Clock Sync Delay: {m['total_sync_wall_ms']:.1f}ms")
    lines.append(f"  Profile Span Evaluated:      {m['profile_span_ms']:.1f}ms")
    lines.append(f"  Sync Density (Stall %):      {m['sync_density_pct']:.1f}%")

    lines.append("\n  Breakdown by Sync Type:")
    if m["sync_by_type_ms"]:
        for t_name, ms_val in sorted(m["sync_by_type_ms"].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"    - {t_name:<30}: {ms_val:8.1f}ms")
    else:
        lines.append("    (None)")

    return "\n".join(lines)


SKILL = Skill(
    name="sync_cost_analysis",
    title="Sync Cost Analysis",
    description="Measures total wall-clock time the host CPU was blocked by CUPTI synchronization calls (e.g. cudaDeviceSynchronize, cudaStreamSynchronize). Reports sync density as a percentage of the profile span.",
    category="system",
    sql="",
    execute_fn=_execute_sync_analysis,
    format_fn=_format_sync_analysis
)
