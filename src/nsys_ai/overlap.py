"""
overlap.py — Compute/communication overlap analysis and NCCL breakdown.

Quantifies how much GPU compute overlaps with NCCL communication,
detects training iterations, and breaks down collective operations.
"""

import logging
from collections import defaultdict

from .profile import Profile

log = logging.getLogger(__name__)

# ── NCCL kernel classification ──────────────────────────────────────

NCCL_TYPES = {
    "AllGather": "allgather",
    "ReduceScatter": "reducescatter",
    "AllReduce": "allreduce",
    "Broadcast": "broadcast",
    "SendRecv": "sendrecv",
    "Reduce": "reduce",
}


def classify_kernel(name: str) -> str:
    """Classify a kernel as 'compute', 'nccl_<type>', or 'other'."""
    if "nccl" in name.lower():
        for key, label in NCCL_TYPES.items():
            if key.lower() in name.lower():
                return f"nccl_{label}"
        return "nccl_other"
    return "compute"


# ── Compute/Communication overlap ──────────────────────────────────


def overlap_analysis(prof: Profile, device: int, trim: tuple[int, int] | None = None) -> dict:
    """
    Quantify compute vs communication overlap on a GPU.

    Returns:
        compute_only_ms:  Time only compute kernels are running
        nccl_only_ms:     Time only NCCL kernels are running
        overlap_ms:       Time both compute and NCCL run concurrently
        idle_ms:          Time no kernels are running
        total_ms:         Wall-clock span
    """
    if _has_duckdb(prof):
        from .connection import DB_ERRORS

        try:
            sql_result = _overlap_analysis_sql(prof, device, trim)
            if sql_result is not None:
                return sql_result
            return _no_kernels_diag(prof, device, trim)
        except DB_ERRORS as e:
            # Narrow to DB_ERRORS so genuine Python bugs in the SQL path
            # surface instead of silently falling back. Engine-level
            # failures (DuckDB / SQLite errors) still degrade gracefully.
            log.warning(
                "overlap_analysis SQL path failed (%s); falling back to Python interval merge",
                e,
            )

    return _overlap_analysis_python(prof, device, trim)


def launch_overhead_ms(
    prof: Profile, device: int | None, trim: tuple[int, int] | None = None
) -> float:
    """Exposed CUDA kernel-launch overhead on a device, in milliseconds.

    This is the slice of GPU-idle time during which the CPU was still inside a
    kernel-launch API call — the dispatch latency the GPU actually *waited on*.
    It is deliberately not the full ``kernel_start - api_start`` gap, which is
    mostly hidden by pipelined launches and would double-count compute (the
    cartesian-overhead trap). Being a strict subset of ``overlap_analysis``'s
    ``idle_ms``, it is meant to be *carved out* of idle in the step-time
    attribution so the four buckets still sum to total.

    Definition: walk kernels on the device by start time, tracking
    ``prev_end`` = the latest end of any earlier kernel (the GPU-busy
    boundary). The idle gap before kernel K is ``(prev_end, K.start)``; the part
    of K's launch window ``(api_start, api_end)`` that falls inside that gap is
    exposed launch overhead. Per-kernel gaps are disjoint, so nothing is counted
    twice. Returns 0.0 when the kernel or runtime table is unavailable.
    """
    kernel_table = prof.schema.kernel_table
    if not kernel_table:
        return 0.0
    tables = prof.schema.tables
    runtime_table: str | None = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if runtime_table not in tables:
        runtime_table = next(
            (t for t in sorted(tables) if t.startswith("CUPTI_ACTIVITY_KIND_RUNTIME")),
            None,
        )
    if not runtime_table:
        return 0.0

    # All kernels define the GPU-busy boundary (so idle gaps are correct); the
    # LEFT JOIN below — not a WHERE filter — gates which kernels have a launch
    # window to attribute. Seeding with 1=1 keeps the SQL valid when no device/
    # trim is given.
    where = ["1=1"]
    params: list = []
    if device is not None:
        where.append("k.deviceId = ?")
        params.append(device)
    if trim:
        where.append("k.start >= ? AND k.[end] <= ?")
        params.extend(trim)
    where_sql = " AND ".join(where)

    sql = f"""
        SELECT k.start AS ks, k.[end] AS ke, r.start AS rs, r.[end] AS re
        FROM {kernel_table} k
        LEFT JOIN {runtime_table} r ON k.correlationId = r.correlationId
        WHERE {where_sql}
        ORDER BY k.start
    """  # noqa: S608 — table names are validated schema identifiers, values are bound
    try:
        rows = prof.adapter.execute(sql, params).fetchall()
    except Exception:  # noqa: BLE001 — launch overhead is best-effort enrichment
        log.debug("launch_overhead_ms query failed", exc_info=True)
        return 0.0

    launch_ns = 0
    prev_end: int | None = None
    for row in rows:
        ks, ke, rs, re = row[0], row[1], row[2], row[3]
        if ks is None or ke is None:
            continue
        if prev_end is not None and rs is not None and re is not None and ks > prev_end:
            # exposed = | (prev_end, ks) ∩ (rs, re) |
            lo = prev_end if prev_end > rs else rs
            hi = ks if ks < re else re
            if hi > lo:
                launch_ns += hi - lo
        if prev_end is None or ke > prev_end:
            prev_end = ke
    return round(launch_ns / 1e6, 2)


def _has_duckdb(prof: Profile) -> bool:
    from .connection import DuckDBAdapter, wrap_connection

    conn = prof.db if prof.db is not None else prof.conn
    return isinstance(wrap_connection(conn), DuckDBAdapter)


def _overlap_analysis_sql(
    prof: Profile, device: int | None, trim: tuple[int, int] | None
) -> dict | None:
    """SQL sweep-line implementation of overlap_analysis.

    Pushes the entire interval-merge + intersection into a single DuckDB
    query — converting every kernel into a (+1/-1) start/end event,
    cumulatively summing per category to track concurrent counts, and
    aggregating duration buckets via LEAD-based windowing. Returns ``None``
    when no kernels match so the caller can produce the standard diagnostic
    payload; raises on SQL errors so the caller can fall back to Python.
    """
    kernel_table = prof.schema.kernel_table
    if not kernel_table:
        return None

    where = ["s.value IS NOT NULL"]
    params: list = []
    if device is not None:
        where.append("k.deviceId = ?")
        params.append(device)
    if trim:
        where.append("k.start >= ? AND k.[end] <= ?")
        params.extend(trim)
    where_sql = " AND ".join(where)

    sql = f"""
        WITH classified AS (
            SELECT
                k.start AS ts_in,
                k.[end] AS ts_out,
                CASE WHEN lower(s.value) LIKE '%nccl%' THEN 1 ELSE 0 END AS is_nccl
            FROM {kernel_table} k
            JOIN StringIds s ON k.shortName = s.id
            WHERE {where_sql}
        ),
        counts AS (
            SELECT
                COALESCE(SUM(CASE WHEN is_nccl = 0 THEN 1 ELSE 0 END), 0) AS compute_kernels,
                COALESCE(SUM(CASE WHEN is_nccl = 1 THEN 1 ELSE 0 END), 0) AS nccl_kernels,
                MIN(ts_in)  AS span_start,
                MAX(ts_out) AS span_end
            FROM classified
        ),
        events AS (
            SELECT ts_in  AS ts,  (1 - is_nccl) AS d_compute,  is_nccl  AS d_nccl FROM classified
            UNION ALL
            SELECT ts_out AS ts, -(1 - is_nccl) AS d_compute, -is_nccl AS d_nccl FROM classified
        ),
        agg AS (
            SELECT ts, SUM(d_compute) AS d_compute, SUM(d_nccl) AS d_nccl
            FROM events
            GROUP BY ts
        ),
        sweep AS (
            SELECT
                ts,
                LEAD(ts) OVER (ORDER BY ts) AS next_ts,
                SUM(d_compute) OVER (ORDER BY ts ROWS UNBOUNDED PRECEDING) AS compute_count,
                SUM(d_nccl)    OVER (ORDER BY ts ROWS UNBOUNDED PRECEDING) AS nccl_count
            FROM agg
        ),
        times AS (
            SELECT
                COALESCE(SUM(CASE WHEN compute_count > 0 THEN (next_ts - ts) ELSE 0 END), 0) AS compute_ns,
                COALESCE(SUM(CASE WHEN nccl_count    > 0 THEN (next_ts - ts) ELSE 0 END), 0) AS nccl_ns,
                COALESCE(SUM(CASE WHEN compute_count > 0 AND nccl_count > 0
                                  THEN (next_ts - ts) ELSE 0 END), 0) AS overlap_ns
            FROM sweep
            WHERE next_ts IS NOT NULL
        )
        SELECT c.compute_kernels, c.nccl_kernels, c.span_start, c.span_end,
               t.compute_ns, t.nccl_ns, t.overlap_ns
        FROM counts c, times t
    """
    rows = prof._duckdb_query(sql, params)
    if not rows:
        return None
    r = rows[0]
    if r["compute_kernels"] == 0 and r["nccl_kernels"] == 0:
        return None
    if r["span_start"] is None or r["span_end"] is None:
        return None

    span_start = int(r["span_start"])
    span_end = int(r["span_end"])
    compute_ns = int(r["compute_ns"] or 0)
    nccl_ns = int(r["nccl_ns"] or 0)
    overlap_ns = int(r["overlap_ns"] or 0)
    total_ns = span_end - span_start
    compute_only_ns = compute_ns - overlap_ns
    nccl_only_ns = nccl_ns - overlap_ns
    idle_ns = total_ns - compute_only_ns - nccl_only_ns - overlap_ns

    return {
        "compute_only_ms": round(compute_only_ns / 1e6, 2),
        "nccl_only_ms": round(nccl_only_ns / 1e6, 2),
        "overlap_ms": round(overlap_ns / 1e6, 2),
        "idle_ms": round(max(0, idle_ns) / 1e6, 2),
        "total_ms": round(total_ns / 1e6, 2),
        "overlap_pct": round(100 * overlap_ns / nccl_ns, 1) if nccl_ns else 0,
        "compute_kernels": int(r["compute_kernels"]),
        "nccl_kernels": int(r["nccl_kernels"]),
        "span_start_ns": span_start,
        "span_end_ns": span_end,
    }


def _overlap_analysis_python(
    prof: Profile, device: int | None, trim: tuple[int, int] | None
) -> dict:
    """Original interval-merge implementation. Retained as fallback for
    SQLite-only profiles (no DuckDB) and as a backstop if the SQL path
    raises unexpectedly."""
    kernels = prof.kernels(device, trim)
    if not kernels:
        return _no_kernels_diag(prof, device, trim)

    compute_intervals = []
    nccl_intervals = []
    for k in kernels:
        cls = classify_kernel(k["name"])
        interval = (k["start"], k["end"])
        if cls.startswith("nccl_"):
            nccl_intervals.append(interval)
        else:
            compute_intervals.append(interval)

    span_start = min(k["start"] for k in kernels)
    span_end = max(k["end"] for k in kernels)
    total_ns = span_end - span_start

    compute_merged = merge_intervals(compute_intervals)
    nccl_merged = merge_intervals(nccl_intervals)

    compute_ns = total_covered(compute_merged)
    nccl_ns = total_covered(nccl_merged)
    overlap_ns = intersection_coverage(compute_merged, nccl_merged)

    compute_only_ns = compute_ns - overlap_ns
    nccl_only_ns = nccl_ns - overlap_ns
    idle_ns = total_ns - compute_only_ns - nccl_only_ns - overlap_ns

    return {
        "compute_only_ms": round(compute_only_ns / 1e6, 2),
        "nccl_only_ms": round(nccl_only_ns / 1e6, 2),
        "overlap_ms": round(overlap_ns / 1e6, 2),
        "idle_ms": round(max(0, idle_ns) / 1e6, 2),
        "total_ms": round(total_ns / 1e6, 2),
        "overlap_pct": round(100 * overlap_ns / nccl_ns, 1) if nccl_ns else 0,
        "compute_kernels": len(compute_intervals),
        "nccl_kernels": len(nccl_intervals),
        "span_start_ns": span_start,
        "span_end_ns": span_end,
    }


def _no_kernels_diag(
    prof: Profile, device: int | None, trim: tuple[int, int] | None
) -> dict:
    """Build the diagnostic dict returned when no kernels match the
    requested device/trim combination."""
    diag: dict = {"error": "no kernels found", "requested_device": device}
    if trim:
        diag["requested_trim_ns"] = list(trim)
    try:
        kernel_tbl = prof.schema.kernel_table
        dev_rows = prof._duckdb_query(
            f"SELECT deviceId, COUNT(*) AS cnt FROM {kernel_tbl} "
            "GROUP BY deviceId ORDER BY deviceId"
        )
        diag["available_devices"] = {r["deviceId"]: r["cnt"] for r in dev_rows}
        total = sum(r["cnt"] for r in dev_rows)
        if total > 0 and device not in diag["available_devices"]:
            diag["hint"] = (
                f"Device {device} has no kernels. "
                f"Try: {', '.join(f'-p device={d}' for d in sorted(diag['available_devices']))}"
            )
        elif total > 0 and trim:
            diag["hint"] = (
                f"Device {device} has {diag['available_devices'].get(device, 0)} kernels "
                f"but none in the requested trim window. Try without --trim."
            )
        else:
            diag["hint"] = "Profile contains no GPU kernel data."
    except Exception:
        diag["hint"] = "Could not query device info."
    return diag


# ── NCCL collective breakdown ──────────────────────────────────────


def nccl_breakdown(prof: Profile, device: int, trim: tuple[int, int] | None = None) -> list[dict]:
    """
    Break down NCCL operations by stream and collective type.

    Returns a list sorted by stream_id (ascending) then total time (descending), each:
        {stream_id, type, count, total_ms, avg_ms, min_ms, max_ms, pct}

    The ``pct`` field is relative to total NCCL time across all streams,
    allowing direct cross-stream comparison to identify which parallelism
    dimension (TP/PP/DP) dominates communication cost.
    """
    kernels = prof.kernels(device, trim)

    nccl_data = []
    total_nccl_ns = 0
    for k in kernels:
        ctype = classify_kernel(k["name"])
        if ctype.startswith("nccl_"):
            dur_ns = k["end"] - k["start"]
            nccl_data.append((k["streamId"], ctype, dur_ns))
            total_nccl_ns += dur_ns

    if not nccl_data:
        return []

    # Group by (stream_id, collective_type)
    by_stream_type: dict[tuple[int, str], list[int]] = defaultdict(list)
    for stream_id, ctype, dur_ns in nccl_data:
        by_stream_type[(stream_id, ctype)].append(dur_ns)

    # Precompute totals to avoid redundant sum() during sort
    computed_groups = [
        (stream_id, ctype, sum(durs), durs) for (stream_id, ctype), durs in by_stream_type.items()
    ]

    result = []
    for stream_id, ctype, total, durs in sorted(
        computed_groups,
        key=lambda x: (x[0], -x[2]),  # stream asc, total desc
    ):
        result.append(
            {
                "stream_id": stream_id,
                "type": ctype.replace("nccl_", ""),
                "count": len(durs),
                "total_ms": round(total / 1e6, 2),
                "avg_ms": round(total / len(durs) / 1e6, 3),
                "min_ms": round(min(durs) / 1e6, 3),
                "max_ms": round(max(durs) / 1e6, 3),
                "pct": round(100 * total / total_nccl_ns, 1),
            }
        )
    return result


# ── Iteration detection ────────────────────────────────────────────


def detect_iterations(
    prof: Profile, device: int, trim: tuple[int, int] | None = None, marker: str = "sample_0"
) -> list[dict]:
    """
    Detect repeating training iterations using a top-level NVTX marker.

    Args:
        marker: NVTX text pattern to match as iteration boundary (default: 'sample_0')

    Returns list of iterations with timing and kernel counts.
    """
    kmap = prof.kernel_map(device)

    if not kmap:
        return []

    pad = int(5e9)
    time_range = trim or prof.meta.time_range

    # Find the primary thread
    from .tree import _find_primary_thread

    primary_tid = _find_primary_thread(prof, device)

    # Resolve table names dynamically (versioned-table support)
    # Prefer exact canonical name first, then sorted prefix fallback
    # (consistent with base._resolve_activity_tables).
    tables = prof.schema.tables

    nvtx_table = "NVTX_EVENTS"
    if nvtx_table not in tables:
        for t in sorted(tables):
            if t.startswith("NVTX_EVENTS"):
                nvtx_table = t
                break

    runtime_table = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if runtime_table not in tables:
        for t in sorted(tables):
            if t.startswith("CUPTI_ACTIVITY_KIND_RUNTIME"):
                runtime_table = t
                break

    # Filter to primary thread's top-level iterations
    # Use COALESCE to handle newer schemas where text is NULL and textId is used
    has_textid = prof._nvtx_has_text_id

    if has_textid:
        text_expr = "COALESCE(n.text, s.value)"
        text_join = "LEFT JOIN StringIds s ON n.textId = s.id"
    else:
        text_expr = "n.text"
        text_join = ""

    pri_nvtx = prof._duckdb_query(
        f"""
            SELECT {text_expr} AS text, n.start, n.[end] FROM {nvtx_table} n
            {text_join}
            WHERE {text_expr} LIKE ? AND n.[end] > n.start AND n.globalTid = ?
              AND n.start >= ? AND n.start <= ?
            ORDER BY n.start
        """,
        (f"%{marker}%", primary_tid, time_range[0] - pad, time_range[1]),
    )

    # Filter to non-overlapping (top-level only)
    iterations = []
    last_end = 0
    for n in pri_nvtx:
        if n["start"] >= last_end:
            iterations.append(n)
            last_end = n["end"]

    # --- Heuristic Fallback ---
    # If no NVTX markers match, fall back to detecting iterations by finding
    # large gaps in kernel execution on the primary CPU thread across all streams.
    if not iterations:
        rt_all = prof._duckdb_query(
            f"""
            SELECT correlationId, start, [end] FROM {runtime_table}
            WHERE globalTid = ? ORDER BY start
            """,
            (primary_tid,),
        )

        if not rt_all:
            return []

        # Keep both kernel and runtime timestamps so that heuristic
        # boundaries can be expressed in runtime (CPU) time domain,
        # matching the downstream rt-based iteration correlation.
        kernel_entries = []
        for rt in rt_all:
            k = kmap.get(rt["correlationId"])
            if not k:
                continue
            # Apply the same time window constraints as the NVTX path
            if time_range is not None:
                if k["end"] < time_range[0] or k["start"] > time_range[1]:
                    continue
            kernel_entries.append(
                {
                    "kernel": k,
                    "rt_start": rt["start"],
                    "rt_end": rt["end"],
                }
            )

        kernel_entries.sort(key=lambda x: x["kernel"]["start"])

        if not kernel_entries:
            return []

        # Find gaps > 2ms (2,000,000 ns) between kernels to denote step
        # boundaries.  Gap detection uses kernel timestamps (GPU domain),
        # but boundaries are recorded in runtime timestamps (CPU domain)
        # so the downstream rt-based filter works correctly.
        GAP_THRESHOLD_NS = 2_000_000
        boundaries = [kernel_entries[0]["rt_start"]]

        last_k_end = kernel_entries[0]["kernel"]["end"]
        for entry in kernel_entries[1:]:
            k = entry["kernel"]
            if k["start"] - last_k_end > GAP_THRESHOLD_NS:
                # Boundary detected: new step starts at this kernel's
                # runtime start so the downstream filter includes it.
                boundaries.append(entry["rt_start"])
            last_k_end = max(last_k_end, k["end"])

        boundaries.append(kernel_entries[-1]["rt_end"])

        # Construct synthetic iterations from these boundaries
        for i in range(len(boundaries) - 1):
            iterations.append(
                {
                    "start": boundaries[i],
                    "end": boundaries[i + 1],
                    "text": f"heuristic_step_{i}",
                }
            )

    if not iterations:
        return []

    # For each iteration, count kernels and compute GPU time
    rt_all = prof._duckdb_query(
        f"""
        SELECT start, [end], correlationId FROM {runtime_table}
        WHERE globalTid = ? ORDER BY start
        """,
        (primary_tid,),
    )

    results = []
    for i, it in enumerate(iterations):
        cpu_start, cpu_end = it["start"], it["end"]

        # Find correlated kernels
        kernels_in_iter = []
        for rt in rt_all:
            if rt["start"] > cpu_end:
                break
            if rt["start"] >= cpu_start and rt["end"] <= cpu_end:
                k = kmap.get(rt["correlationId"])
                if k:
                    kernels_in_iter.append(k)

        if not kernels_in_iter:
            continue

        gpu_start = min(k["start"] for k in kernels_in_iter)
        gpu_end = max(k["end"] for k in kernels_in_iter)
        compute_ns = sum(k["end"] - k["start"] for k in kernels_in_iter)
        nccl_count = sum(1 for k in kernels_in_iter if "nccl" in k["name"].lower())

        results.append(
            {
                "iteration": i,
                "text": it["text"] if "text" in it else "",
                "gpu_start_ns": gpu_start,
                "gpu_end_ns": gpu_end,
                "gpu_start_s": round(gpu_start / 1e9, 4),
                "gpu_end_s": round(gpu_end / 1e9, 4),
                "duration_ms": round((gpu_end - gpu_start) / 1e6, 2),
                "compute_ms": round(compute_ns / 1e6, 2),
                "kernel_count": len(kernels_in_iter),
                "nccl_count": nccl_count,
            }
        )

    return results


# ── Interval math helpers ──────────────────────────────────────────


def merge_intervals(intervals):
    """Merge overlapping intervals into non-overlapping set."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals)
    merged = [sorted_iv[0]]
    for start, end in sorted_iv[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def total_covered(merged):
    """Total time covered by merged intervals."""
    return sum(end - start for start, end in merged)


def intersection_coverage(a, b):
    """Total time covered by the intersection of two merged interval sets."""
    if not a or not b:
        return 0
    total = 0
    j = 0
    for a_start, a_end in a:
        while j < len(b) and b[j][1] <= a_start:
            j += 1
        k = j
        while k < len(b) and b[k][0] < a_end:
            overlap_start = max(a_start, b[k][0])
            overlap_end = min(a_end, b[k][1])
            if overlap_start < overlap_end:
                total += overlap_end - overlap_start
            k += 1
    return total


# ── Text formatting ────────────────────────────────────────────────


def format_overlap(result: dict) -> str:
    """Format overlap analysis as readable text."""
    if "error" in result:
        return f"Overlap: {result['error']}"
    return (
        f"Compute/Communication Overlap Analysis\n"
        f"  Total span:    {result['total_ms']:.1f}ms\n"
        f"  Compute only:  {result['compute_only_ms']:.1f}ms\n"
        f"  NCCL only:     {result['nccl_only_ms']:.1f}ms\n"
        f"  Overlap:       {result['overlap_ms']:.1f}ms ({result['overlap_pct']}% of NCCL overlapped)\n"
        f"  Idle:          {result['idle_ms']:.1f}ms\n"
        f"  Kernels:       {result['compute_kernels']} compute + {result['nccl_kernels']} NCCL"
    )


def format_nccl(breakdown: list[dict]) -> str:
    """Format NCCL breakdown as readable text, grouped by stream."""
    if not breakdown:
        return "No NCCL collectives found"
    lines = ["NCCL Collective Breakdown"]

    # Group by stream_id (if present)
    has_streams = "stream_id" in breakdown[0]
    if has_streams:
        from itertools import groupby

        for stream_id, group in groupby(breakdown, key=lambda b: b["stream_id"]):
            lines.append(f"  [Stream {stream_id}]")
            for b in group:
                lines.append(
                    f"    {b['type']:20s}  {b['pct']:5.1f}%  "
                    f"{b['total_ms']:8.1f}ms  ×{b['count']:<3d}  "
                    f"avg={b['avg_ms']:.1f}ms  [{b['min_ms']:.1f}–{b['max_ms']:.1f}ms]"
                )
    else:
        for b in breakdown:
            lines.append(
                f"  {b['type']:20s}  {b['pct']:5.1f}%  "
                f"{b['total_ms']:8.1f}ms  ×{b['count']:<3d}  "
                f"avg={b['avg_ms']:.1f}ms  [{b['min_ms']:.1f}–{b['max_ms']:.1f}ms]"
            )
    return "\n".join(lines)


def format_iterations(iters: list[dict]) -> str:
    """Format iteration timings as readable text."""
    if not iters:
        return "No iterations detected"
    lines = ["Iteration Timings"]
    for it in iters:
        lines.append(
            f"  iter {it['iteration']:2d}  "
            f"{it['duration_ms']:8.1f}ms  "
            f"({it['kernel_count']} kernels, {it['nccl_count']} NCCL)  "
            f"compute={it['compute_ms']:.1f}ms"
        )
    if len(iters) > 1:
        durs = [it["duration_ms"] for it in iters]
        avg = sum(durs) / len(durs)
        lines.append(f"\n  Average: {avg:.1f}ms  Min: {min(durs):.1f}ms  Max: {max(durs):.1f}ms")
    return "\n".join(lines)
