"""Sync Cost Analysis skill.

Examines CUPTI_ACTIVITY_KIND_SYNCHRONIZATION to compute the true wall-clock
duration that the CPU was blocked by synchronizations, distinguishing explicit pipeline
stalls from naturally overlapping communications.
"""

from collections import defaultdict

from ...connection import DB_ERRORS, is_safe_identifier, wrap_connection
from ..base import Skill, SkillParam, _compute_interval_union


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
_CACHE_MAX_SIZE = 8  # bounded to prevent unbounded growth / id() reuse


def _execute_sync_analysis(conn, **kwargs) -> list[dict]:
    # Module-level cache keyed by (connection identity, trim window, device).
    # Avoids redundant re-execution when called by multiple consumer skills
    # (manifest, root_cause, overlap, bubble) within the same profile session.
    cache_key = (
        id(conn),
        kwargs.get("trim_start_ns"),
        kwargs.get("trim_end_ns"),
        kwargs.get("device"),
    )
    cached = _sync_result_cache.get(cache_key)
    if cached is not None:
        return cached

    result = _execute_sync_analysis_impl(conn, **kwargs)

    if len(_sync_result_cache) >= _CACHE_MAX_SIZE:
        _sync_result_cache.clear()
    _sync_result_cache[cache_key] = result
    return result


def _has_device_id(adapter, sync_table: str) -> bool:
    """Probe whether the sync table has a `deviceId` column.

    Nsight Systems' `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION` schema exposes a
    `deviceId INT NOT NULL` column (nsys exporter docs), but older profiles
    or stripped mock fixtures may lack it. Falling back to a global-only
    result when absent keeps backward compatibility.
    """
    try:
        adapter.execute(f"SELECT deviceId FROM {sync_table} LIMIT 0")
        return True
    except DB_ERRORS:
        return False


def _execute_sync_analysis_impl(conn, **kwargs) -> list[dict]:
    adapter = wrap_connection(conn)
    sync_table = _resolve_table_name(conn, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION")
    type_table = _resolve_table_name(conn, "ENUM_CUPTI_SYNC_TYPE")

    # Guard against SQL injection from maliciously crafted table names
    if not is_safe_identifier(sync_table) or not is_safe_identifier(type_table):
        return [
            {
                "total_sync_wall_ms": 0.0,
                "sync_by_type_ms": {},
                "sync_by_device": None,
                "profile_span_ms": 0.0,
                "sync_density_pct": 0.0,
                "error": f"Unsafe table name resolved: {sync_table!r} / {type_table!r}",
            }
        ]

    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    device_filter = kwargs.get("device")
    if device_filter is not None:
        try:
            device_filter = int(device_filter)
        except (TypeError, ValueError):
            return [{"error": f"`device` must be an integer (got {device_filter!r})"}]

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

    # Probe table existence before column existence so a missing sync table
    # reports the right error (rather than "no deviceId column") when the user
    # supplied `-p device=N`.
    try:
        adapter.execute(f"SELECT 1 FROM {sync_table} LIMIT 0")
        sync_table_exists = True
    except DB_ERRORS:
        sync_table_exists = False

    if not sync_table_exists:
        return [
            {
                "total_sync_wall_ms": 0.0,
                "sync_by_type_ms": {},
                "sync_by_device": None,
                "profile_span_ms": 0.0,
                "sync_density_pct": 0.0,
                "error": "Synchronization tables not found in profile",
            }
        ]

    has_device = _has_device_id(adapter, sync_table)
    if device_filter is not None and not has_device:
        return [
            {
                "error": (
                    f"`device={device_filter}` requested but the sync table has no "
                    "deviceId column (legacy Nsight export)."
                )
            }
        ]

    conds = ["1=1"]
    params: list = []
    if trim_start is not None:
        conds.append("s.[end] >= ?")
        params.append(int(trim_start))
    if trim_end is not None:
        conds.append("s.start <= ?")
        params.append(int(trim_end))
    if device_filter is not None:
        conds.append("s.deviceId = ?")
        params.append(device_filter)
    where_clause = " AND ".join(conds)

    device_expr = "s.deviceId AS device" if has_device else "NULL AS device"
    query = f"""
    SELECT
        s.start,
        s.[end],
        {device_expr},
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
        return [
            {
                "total_sync_wall_ms": 0.0,
                "sync_by_type_ms": {},
                "sync_by_device": None,
                "profile_span_ms": 0.0,
                "sync_density_pct": 0.0,
                "error": "Synchronization tables not found in profile",
            }
        ]

    # Filter and clamp intervals; accumulate global + per-type + per-device buckets.
    global_intervals: list[tuple[int, int]] = []
    type_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    device_intervals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    device_type_intervals: dict[int, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in rows:
        s_start, s_end = int(row[0]), int(row[1])
        dev = row[2]
        sync_type_name = str(row[3])

        if trim_start is not None:
            s_start = max(s_start, int(trim_start))
        if trim_end is not None:
            s_end = min(s_end, int(trim_end))

        if s_start >= s_end:
            continue
        global_intervals.append((s_start, s_end))
        type_intervals[sync_type_name].append((s_start, s_end))
        if dev is not None:
            dev_i = int(dev)
            device_intervals[dev_i].append((s_start, s_end))
            device_type_intervals[dev_i][sync_type_name].append((s_start, s_end))

    total_sync_wall_ns = _compute_interval_union(global_intervals)
    sync_by_type_ms = {
        t: round(_compute_interval_union(ivs) / 1e6, 3)
        for t, ivs in type_intervals.items()
    }

    profile_span_ns = 0
    if trim_start is not None and trim_end is not None:
        profile_span_ns = max(0, int(trim_end) - int(trim_start))

    total_sync_wall_ms = total_sync_wall_ns / 1e6
    profile_span_ms = profile_span_ns / 1e6
    sync_density_pct = (
        (total_sync_wall_ms / profile_span_ms) * 100 if profile_span_ms > 0 else 0.0
    )

    # Per-device breakdown. Always emitted when deviceId is available so
    # callers can detect multi-rank asymmetry without a second skill call.
    if has_device:
        sync_by_device = []
        for dev_i in sorted(device_intervals.keys()):
            dev_ns = _compute_interval_union(device_intervals[dev_i])
            dev_ms = dev_ns / 1e6
            dev_density = (dev_ms / profile_span_ms) * 100 if profile_span_ms > 0 else 0.0
            dev_by_type = {
                t: round(_compute_interval_union(ivs) / 1e6, 3)
                for t, ivs in device_type_intervals[dev_i].items()
            }
            sync_by_device.append(
                {
                    "device": dev_i,
                    "total_sync_wall_ms": round(dev_ms, 3),
                    "sync_density_pct": round(dev_density, 2),
                    "sync_by_type_ms": dev_by_type,
                }
            )
    else:
        sync_by_device = None

    return [
        {
            "total_sync_wall_ms": round(total_sync_wall_ms, 3),
            "sync_by_type_ms": sync_by_type_ms,
            "sync_by_device": sync_by_device,
            "profile_span_ms": round(profile_span_ms, 3),
            "sync_density_pct": round(sync_density_pct, 2),
        }
    ]


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
        for t_name, ms_val in sorted(
            m["sync_by_type_ms"].items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"    - {t_name:<30}: {ms_val:8.1f}ms")
    else:
        lines.append("    (None)")

    by_device = m.get("sync_by_device")
    if by_device:
        lines.append("\n  Breakdown by Device:")
        for d in sorted(by_device, key=lambda x: -x["total_sync_wall_ms"]):
            lines.append(
                f"    - device {d['device']}:"
                f" {d['total_sync_wall_ms']:8.1f}ms"
                f" ({d['sync_density_pct']:.1f}% density)"
            )

    return "\n".join(lines)


SKILL = Skill(
    name="sync_cost_analysis",
    title="Sync Cost Analysis",
    description=(
        "Measures total wall-clock time the host CPU was blocked by CUPTI "
        "synchronization calls (cudaDeviceSynchronize, cudaStreamSynchronize, "
        "cudaEventSynchronize). Reports sync density as a percentage of the "
        "profile span, with an optional per-device breakdown (`sync_by_device`) "
        "for detecting multi-rank asymmetry in distributed training."
    ),
    category="system",
    sql="",
    execute_fn=_execute_sync_analysis,
    format_fn=_format_sync_analysis,
    params=[
        SkillParam(
            "device",
            "Filter to a single GPU device id (e.g. 0). Omit for all-devices "
            "aggregate (sync_by_device still included).",
            "int",
            False,
            None,
        ),
    ],
)
