"""Quantify exact true pipeline bubble / idle time percentage on GPUs.

Uses Python-level interval merging (O(n log n) sort + O(n) sweep) instead
of SQL window functions, which are prohibitively slow on large profiles.
"""

import logging
import sqlite3

from nsys_ai.connection import wrap_connection

from ..base import Skill

logger = logging.getLogger(__name__)


try:
    import duckdb

    _DB_ERRORS = (sqlite3.Error, duckdb.Error)
except ImportError:
    _DB_ERRORS = (sqlite3.Error,)


def _format(rows):
    if not rows:
        return "(No kernel or memory operations detected for bubble analysis)"
    lines = [
        "── Pipeline Bubble (Idle Time) Metrics ──",
        f"{'GPU':<4s}  {'Total Span(ms)':>15s}  {'Active(ms)':>12s}  {'Bubble(ms)':>12s}  {'Bubble %':>10s}",
        "-" * 61,
    ]
    for r in rows:
        gpu_str = str(r["deviceId"])
        lines.append(
            f"{gpu_str:<4s}  {r['total_span_ms']:>15.2f}  {r['active_ms']:>12.2f}  "
            f"{r['bubble_ms']:>12.2f}  {r['bubble_pct']:>9.1f}%"
        )
    return "\n".join(lines)


def _execute(conn, **kwargs):
    tables = wrap_connection(conn).resolve_activity_tables()
    kernel_table = tables.get("kernel", "CUPTI_ACTIVITY_KIND_KERNEL")
    memcpy_table = tables.get("memcpy")
    memset_table = tables.get("memset")

    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")

    # --- Fetch kernel intervals per device (if available) ---
    kernel_rows = []
    if kernel_table:
        try:
            params_k = []
            trim_clause_k = ""
            if trim_start is not None and trim_end is not None:
                trim_clause_k = 'WHERE "end" > ? AND start < ?'
                params_k = [trim_start, trim_end]

            kernel_rows = conn.execute(
                f'SELECT deviceId, start, "end" FROM {kernel_table} {trim_clause_k}',
                params_k,
            ).fetchall()
        except _DB_ERRORS as e:
            logger.debug(f"Failed to fetch kernel intervals from {kernel_table}: {e}")

    # --- Fetch memcpy intervals per device (if available) ---
    memcpy_rows = []
    if memcpy_table:
        try:
            params_m = []
            trim_clause_m = ""
            if trim_start is not None and trim_end is not None:
                trim_clause_m = 'WHERE "end" > ? AND start < ?'
                params_m = [trim_start, trim_end]
            memcpy_rows = conn.execute(
                f'SELECT deviceId, start, "end" FROM {memcpy_table} {trim_clause_m}',
                params_m,
            ).fetchall()
        except _DB_ERRORS as e:
            logger.debug(f"Failed to fetch memcpy intervals from {memcpy_table}: {e}")

    # --- Fetch memset intervals per device (if available) ---
    memset_rows = []
    if memset_table:
        try:
            params_s = []
            trim_clause_s = ""
            if trim_start is not None and trim_end is not None:
                trim_clause_s = 'WHERE "end" > ? AND start < ?'
                params_s = [trim_start, trim_end]
            memset_rows = conn.execute(
                f'SELECT deviceId, start, "end" FROM {memset_table} {trim_clause_s}',
                params_s,
            ).fetchall()
        except _DB_ERRORS as e:
            logger.debug(f"Failed to fetch memset intervals from {memset_table}: {e}")

    # --- Group intervals by deviceId ---
    from collections import defaultdict

    by_device: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for rows in (kernel_rows, memcpy_rows, memset_rows):
        for dev, s, e in rows:
            if trim_start is not None:
                s = max(s, trim_start)
            if trim_end is not None:
                e = min(e, trim_end)
            if s < e:
                by_device[dev].append((s, e))

    if not by_device:
        return []

    from nsys_ai.overlap import merge_intervals, total_covered

    # --- Merge intervals and compute bubble metrics per device ---
    results = []
    for dev in sorted(by_device):
        intervals = by_device[dev]
        merged = merge_intervals(intervals)
        if not merged:
            continue

        active_ns = total_covered(merged)
        global_start = merged[0][0]
        global_end = merged[-1][1]
        total_span = global_end - global_start
        bubble_ns = total_span - active_ns
        bubble_pct = 100.0 * bubble_ns / total_span if total_span > 0 else 0.0

        results.append(
            {
                "deviceId": dev,
                "total_span_ms": round(total_span / 1e6, 2),
                "active_ms": round(active_ns / 1e6, 2),
                "bubble_ms": round(bubble_ns / 1e6, 2),
                "bubble_pct": round(bubble_pct, 2),
            }
        )

    return results


SKILL = Skill(
    name="pipeline_bubble_metrics",
    title="Pipeline Bubble Metrics (True Idle Percentage)",
    description=(
        "Quantifies exact true pipeline bubble (idle time percentage) on GPUs. "
        "It merges all overlapping compute kernels, memory transfers, and memory sets (memset) to find the "
        "actual sum of time the GPU was doing work, and reports the remaining time "
        "as a pure bubble metric (Bubble %)."
    ),
    category="utilization",
    execute_fn=_execute,
    format_fn=_format,
    tags=["bubble", "idle", "mfu", "utilization", "gap", "pipeline"],
)
