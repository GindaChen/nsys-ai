"""NVTX → Kernel attribution module.

Provides efficient NVTX-to-kernel mapping. Two strategies are used:

1. **DuckDB Parquet cache** (primary): When the profile has been cached,
   the ``nvtx_kernel_map`` + ``nvtx_path_dict`` views (or legacy map-only rows)
   provide a precomputed attribution table
   that is queried directly in ``nvtx_layer_breakdown``. This path is fast
   and avoids any Python-level sweep.

2. **Python sort-merge fallback**: For ``.sqlite``-only scenarios (no cache),
   load Kernel→Runtime (via correlationId index) and NVTX ranges, then sweep
   per-thread with a stack to find the innermost enclosing NVTX for each
   runtime call.  Complexity: O(N+M) after sorting.
"""

import logging
import sqlite3
from collections import defaultdict

_log = logging.getLogger(__name__)

# ── Python sort-merge fallback ───────────────────────────────────────


def _sort_merge_attribute(
    conn: sqlite3.Connection,
    trim: tuple[int, int] | None = None,
) -> list[dict]:
    """Sort-merge style attribute of kernels to NVTX ranges.

    Algorithm (high level):
    1. Load Kernel→Runtime via correlationId (fast indexed join).
    2. Load NVTX ranges sorted by (globalTid, start).
    3. For each thread, do a single forward sweep maintaining a stack of
       "currently open" NVTX ranges.  For each runtime call, search this
       stack (from top to bottom) to find the innermost NVTX that fully
       encloses the call, if any.

    Overall complexity is O(N+M) per thread (each NVTX is pushed and
    popped at most once; each runtime call is processed once).
    """
    from .connection import wrap_connection

    adapter = wrap_connection(conn)
    resolved_tables = adapter.resolve_activity_tables()

    kernel_table = resolved_tables.get("kernel", "CUPTI_ACTIVITY_KIND_KERNEL")
    runtime_table = resolved_tables.get("runtime", "CUPTI_ACTIVITY_KIND_RUNTIME")
    nvtx_table = resolved_tables.get("nvtx", "NVTX_EVENTS")

    # Trim clause for SQL queries
    trim_sql = ""
    trim_params: tuple = ()
    if trim:
        trim_sql = "AND k.start >= ? AND k.[end] <= ?"
        trim_params = (trim[0], trim[1])

    kr_rows = adapter.execute(
        f"""
        SELECT r.globalTid, r.start, r.[end],
               k.start AS ks, k.[end] AS ke, k.shortName
        FROM {kernel_table} k
        JOIN {runtime_table} r ON r.correlationId = k.correlationId
        WHERE 1=1 {trim_sql}
        """,
        trim_params,
    ).fetchall()

    if not kr_rows:
        return []

    # Phase 2: Load NVTX ranges (eventType 59 = NVTX push/pop range)
    has_textid = adapter.detect_nvtx_text_id()

    if has_textid:
        text_expr = "COALESCE(n.text, s.value)"
        text_join = "LEFT JOIN StringIds s ON n.textId = s.id"
    else:
        text_expr = "n.text"
        text_join = ""

    # Restrict NVTX scan to relevant tids and runtime window. This prevents
    # loading the full NVTX table on very large profiles.
    tids = sorted({int(r[0]) for r in kr_rows})
    if not tids:
        return []
    min_r_start = min(int(r[1]) for r in kr_rows)
    max_r_end = max(int(r[2]) for r in kr_rows)
    # SQLite has a default limit of 999 bound parameters; if we exceed it,
    # drop the TID whitelist and rely solely on the time-window filter.
    if len(tids) <= 900:
        tid_clause = f"AND n.globalTid IN ({','.join('?' for _ in tids)})"
        nvtx_params: tuple = tuple(tids) + (max_r_end, min_r_start)
    else:
        tid_clause = ""
        nvtx_params = (max_r_end, min_r_start)
    nvtx_rows = adapter.execute(
        f"""
        SELECT n.globalTid, n.start, n.[end], {text_expr} AS text
        FROM {nvtx_table} n
        {text_join}
        WHERE n.eventType = 59
          AND n.[end] > n.start
          {tid_clause}
          AND n.start <= ?
          AND n.[end] >= ?
        ORDER BY n.globalTid, n.start
        """,
        nvtx_params,
    ).fetchall()

    # StringIds lookup for kernel names — only fetch the IDs we need
    short_name_ids = {r[5] for r in kr_rows if r[5] is not None}
    if short_name_ids:
        placeholders = ",".join("?" for _ in short_name_ids)
        sid_rows = adapter.execute(
            f"SELECT id, value FROM StringIds WHERE id IN ({placeholders})",
            tuple(short_name_ids),
        ).fetchall()
        sid_map = dict(sid_rows)
    else:
        sid_map = {}

    # Phase 3: Group by globalTid, then sweep
    nvtx_by_tid: dict[int, list[tuple]] = defaultdict(list)
    for n in nvtx_rows:
        nvtx_by_tid[n[0]].append((n[1], n[2], n[3]))  # start, end, text

    kr_by_tid: dict[int, list[tuple]] = defaultdict(list)
    for r in kr_rows:
        kr_by_tid[r[0]].append((r[1], r[2], r[3], r[4], r[5]))
        # r_start, r_end, k_start, k_end, shortName

    results = []

    for tid in kr_by_tid:
        if tid not in nvtx_by_tid:
            continue

        # NVTX ranges for this thread, sorted by start time
        nvtx_list = nvtx_by_tid[tid]

        # Ensure runtime records for this thread are processed in start-time order
        kr_by_tid[tid].sort(key=lambda x: x[0])

        nvtx_idx = 0
        open_stack: list[tuple[int, int, str]] = []  # (start, end, text)

        for r_start, r_end, k_start, k_end, short_name in kr_by_tid[tid]:
            # 1. Pop NVTX ranges that have already closed before this runtime starts
            # Because NVTX ranges are assumed strictly nested per thread, O(1) amortized
            while open_stack and open_stack[-1][1] < r_start:
                open_stack.pop()

            # 2. Advance NVTX pointer, pushing any ranges that have opened by r_start
            # but ONLY if they are still active.
            while nvtx_idx < len(nvtx_list) and nvtx_list[nvtx_idx][0] <= r_start:
                if nvtx_list[nvtx_idx][1] >= r_start:
                    open_stack.append(nvtx_list[nvtx_idx])
                nvtx_idx += 1

            # Find innermost enclosing NVTX (scan stack from top)
            best_nvtx = None
            best_idx = -1
            for i in range(len(open_stack) - 1, -1, -1):
                ns, ne, nt = open_stack[i]
                if ns <= r_start and ne >= r_end:
                    best_nvtx = nt
                    best_idx = i
                    break

            if best_nvtx is not None:
                # Build path only from NVTX ranges that actually enclose [r_start, r_end]
                enclosing_ranges = [
                    entry
                    for entry in open_stack[: best_idx + 1]
                    if entry[0] <= r_start and entry[1] >= r_end
                ]
                # Derive depth from the number of enclosing ranges (0-based)
                nvtx_depth = len(enclosing_ranges) - 1
                path_parts = [entry[2] for entry in enclosing_ranges]
                results.append(
                    {
                        "nvtx_text": best_nvtx,
                        "nvtx_depth": nvtx_depth,
                        "nvtx_path": " > ".join(path_parts),
                        "kernel_name": sid_map.get(short_name, f"kernel_{short_name}"),
                        "k_start": k_start,
                        "k_end": k_end,
                        "k_dur_ns": k_end - k_start,
                    }
                )

    return results


# ── Public API ──────────────────────────────────────────────────────


def attribute_kernels_to_nvtx(
    conn,
    sqlite_path: str | None = None,
    trim: tuple[int, int] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Attribute GPU kernels to their enclosing NVTX ranges.

    Uses the DuckDB Parquet cache (``nvtx_kernel_map`` view) if available,
    falling back to a Python sort-merge O(N+M) sweep on the raw SQLite data.

    Returns list of dicts with keys:
      nvtx_text, nvtx_depth, nvtx_path,
      kernel_name, k_start, k_end, k_dur_ns
    """

    # DuckDB: try reading from precomputed nvtx_kernel_map first.
    try:
        from .connection import (
                DB_ERRORS,
                DuckDBAdapter,
                cached_nvtx_map_uses_path_id,
                wrap_connection,
            )

        adapter = wrap_connection(conn)

        if isinstance(adapter, DuckDBAdapter):
            trim_sql = ""
            params = []
            if trim:
                trim_sql = "WHERE k_start >= ? AND k_end <= ?"
                params = [trim[0], trim[1]]
            limit_sql = f" LIMIT {int(limit)}" if limit and int(limit) > 0 else ""

            try:
                uses_path_id = cached_nvtx_map_uses_path_id(adapter.raw_conn)

                if uses_path_id:
                    trim_sql_m = ""
                    if trim:
                        trim_sql_m = "WHERE m.k_start >= ? AND m.k_end <= ?"
                    cur = adapter.execute(
                        f"""
                        SELECT m.nvtx_text, m.nvtx_depth, d.nvtx_path, m.kernel_name,
                               m.k_start, m.k_end, m.k_dur_ns
                        FROM nvtx_kernel_map m
                        JOIN nvtx_path_dict d ON m.path_id = d.path_id
                        {trim_sql_m}
                        ORDER BY m.k_start{limit_sql}
                        """,
                        params,
                    )
                else:
                    cur = adapter.execute(
                        f"SELECT * FROM nvtx_kernel_map {trim_sql} ORDER BY k_start{limit_sql}",
                        params,
                    )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
            except DB_ERRORS:
                _log.debug("nvtx_kernel_map unavailable; using on-demand DuckDB attribution")

            # On-demand DuckDB SQL attribution for large profiles where the
            # cache intentionally skips nvtx_kernel_map precomputation.
            tables = adapter.resolve_activity_tables()
            kernel_table = tables.get("kernel", "CUPTI_ACTIVITY_KIND_KERNEL")
            runtime_table = tables.get("runtime", "CUPTI_ACTIVITY_KIND_RUNTIME")
            nvtx_table = tables.get("nvtx", "NVTX_EVENTS")
            has_textid = adapter.detect_nvtx_text_id()
            if has_textid:
                text_expr = "COALESCE(n.text, ns.value)"
                text_join = "LEFT JOIN StringIds ns ON n.textId = ns.id"
            else:
                text_expr = "n.text"
                text_join = ""

            trim_clause = ""
            trim_params: list[int] = []
            kr_where = ""
            nvtx_where = ""
            if trim:
                trim_clause = "WHERE k_start >= ? AND k_end <= ?"
                trim_start = int(trim[0])
                trim_end = int(trim[1])
                kr_where = 'WHERE k.start >= ? AND k."end" <= ?'
                nvtx_where = 'AND n.start <= ? AND n."end" >= ?'
                trim_params = [trim_start, trim_end, trim_end, trim_start, trim_start, trim_end]
            limit_sql = f" LIMIT {int(limit)}" if limit and int(limit) > 0 else ""

            cur = adapter.execute(
                f"""
                WITH kr AS (
                    SELECT r.globalTid, r.start AS r_start, r."end" AS r_end,
                           k.start AS k_start, k."end" AS k_end,
                           COALESCE(kd.value, ks.value, 'kernel_' || CAST(k.shortName AS VARCHAR)) AS kernel_name,
                           r.correlationId
                    FROM {kernel_table} k
                    JOIN {runtime_table} r ON r.correlationId = k.correlationId
                    LEFT JOIN StringIds ks ON k.shortName = ks.id
                    LEFT JOIN StringIds kd ON k.demangledName = kd.id
                    {kr_where}
                ),
                enclosing AS (
                    SELECT kr.k_start, kr.k_end, kr.kernel_name,
                           kr.r_start, kr.r_end, kr.globalTid, kr.correlationId,
                           {text_expr} AS nvtx_text,
                           (n."end" - n.start) AS n_dur, n.start AS n_start
                    FROM kr
                    JOIN {nvtx_table} n
                      ON n.globalTid = kr.globalTid
                      AND n.eventType = 59
                      AND n."end" > n.start
                      AND n.start <= kr.r_start
                      AND n."end" >= kr.r_end
                      {nvtx_where}
                    {text_join}
                    WHERE {text_expr} IS NOT NULL
                ),
                grouped AS (
                    SELECT
                        FIRST(nvtx_text ORDER BY n_dur ASC, n_start ASC) AS nvtx_text,
                        CAST(COUNT(*) - 1 AS INTEGER) AS nvtx_depth,
                        string_agg(nvtx_text, ' > ' ORDER BY n_dur DESC, n_start ASC) AS nvtx_path,
                        kernel_name,
                        k_start,
                        k_end,
                        (k_end - k_start) AS k_dur_ns
                    FROM enclosing
                    GROUP BY k_start, k_end, globalTid, kernel_name, correlationId
                )
                SELECT nvtx_text, nvtx_depth, nvtx_path, kernel_name, k_start, k_end, k_dur_ns
                FROM grouped
                {trim_clause}
                ORDER BY k_start
                {limit_sql}
                """,
                trim_params,
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception:
        _log.debug("DuckDB nvtx_kernel_map query failed, fallback to Python sweep", exc_info=True)

    # Tier 2: Python sort-merge fallback on SQLite
    rows = _sort_merge_attribute(conn, trim)
    if limit and int(limit) > 0:
        import heapq

        return heapq.nsmallest(int(limit), rows, key=lambda r: int(r["k_start"]))
    return rows
