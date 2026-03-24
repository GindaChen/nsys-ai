"""NVTX → Kernel attribution module.

Provides efficient NVTX-to-kernel mapping using a two-tier strategy:

1. **Tier 1 (nsys recipe)**: If ``nsys`` CLI is available and the ``.nsys-rep``
   file exists, delegate to ``nsys stats -r nvtx_gpu_proj_trace`` which handles
   the temporal join natively in C++ — fast and correct.

2. **Tier 2 (Python sort-merge)**: For ``.sqlite``-only scenarios, load
   Kernel→Runtime (via correlationId index) and NVTX ranges, then sweep
   per-thread with a stack to find the innermost enclosing NVTX for each
   runtime call.  Complexity: O(N+M) after sorting.
"""

import csv
import io
import logging
import os
import shutil
import sqlite3
import subprocess  # nosec B404 — list args, no shell
from collections import defaultdict

_log = logging.getLogger(__name__)

# ── Tier 1: nsys recipe ─────────────────────────────────────────────


def _find_nsys_rep(sqlite_path: str) -> str | None:
    """Derive the .nsys-rep path from a .sqlite path, if it exists."""
    # Common pattern: profile.sqlite → profile.nsys-rep
    base = sqlite_path
    for suffix in (".sqlite", ".sqlite3"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    rep = base + ".nsys-rep"
    return rep if os.path.isfile(rep) else None


def _run_nsys_recipe(nsys_rep_path: str, trim: tuple[int, int] | None = None) -> list[dict] | None:
    """Run nvtx_gpu_proj_trace recipe and parse CSV output.

    Returns list of dicts or None if nsys is unavailable or recipe fails.
    """
    nsys_exe = shutil.which("nsys")
    if not nsys_exe:
        return None

    cmd = [
        nsys_exe,
        "stats",
        "-r",
        "nvtx_gpu_proj_trace",
        "--format",
        "csv",
        "--force-export=true",
        nsys_rep_path,
    ]

    try:
        result = subprocess.run(  # nosec B603
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None

    # Parse the CSV output
    stdout = result.stdout
    if not stdout.strip():
        return None

    rows = []
    reader = csv.DictReader(io.StringIO(stdout))
    for row in reader:
        try:
            start_ns = int(float(row.get("Start (ns)", 0)))
            end_ns = int(float(row.get("End (ns)", 0)))
            dur_ns = end_ns - start_ns
        except (ValueError, TypeError):
            continue

        # Apply trim if specified: containment semantics (same as Tier 2)
        if trim:
            if start_ns < trim[0] or end_ns > trim[1]:
                continue

        rows.append(
            {
                "nvtx_text": row.get("NVTX Range", row.get("Name", "")),
                "nvtx_depth": 0,
                "nvtx_path": row.get("NVTX Range", row.get("Name", "")),
                "kernel_name": row.get("Kernel Name", row.get("Operation", "")),
                "k_start": start_ns,
                "k_end": end_ns,
                "k_dur_ns": dur_ns,
            }
        )

    return rows if rows else None


# ── Tier 2: Python sort-merge ───────────────────────────────────────


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
    # Detect versioned table names
    from .skills.base import _resolve_activity_tables
    resolved_tables = _resolve_activity_tables(conn)

    kernel_table = resolved_tables.get("kernel", "CUPTI_ACTIVITY_KIND_KERNEL")
    runtime_table = resolved_tables.get("runtime", "CUPTI_ACTIVITY_KIND_RUNTIME")
    nvtx_table = resolved_tables.get("nvtx", "NVTX_EVENTS")

    # Trim clause for SQL queries
    trim_sql = ""
    trim_params: tuple = ()
    if trim:
        trim_sql = "AND k.start >= ? AND k.[end] <= ?"
        trim_params = (trim[0], trim[1])

    # Phase 1: Kernel → Runtime via correlationId (indexed, fast)
    from .sql_compat import sqlite_to_duckdb
    kr_rows = conn.execute(sqlite_to_duckdb(
        f"""
        SELECT r.globalTid, r.start, r.[end],
               k.start AS ks, k.[end] AS ke, k.shortName
        FROM {kernel_table} k
        JOIN {runtime_table} r ON r.correlationId = k.correlationId
        WHERE 1=1 {trim_sql}
        ORDER BY r.globalTid, r.start
        """),
        trim_params,
    ).fetchall()

    if not kr_rows:
        return []

    # Phase 2: Load NVTX ranges (eventType 59 = NVTX push/pop range)
    # Resolve text expression (handles textId vs text column)
    has_textid = False
    try:
        import duckdb as _ddb
        if isinstance(conn, _ddb.DuckDBPyConnection):
            cols = [c[0] for c in conn.execute(f"DESCRIBE {nvtx_table}").fetchall()]
            has_textid = "textId" in cols
        else:
            has_textid = (
                conn.execute(
                    f"SELECT COUNT(*) FROM pragma_table_info('{nvtx_table}') WHERE name='textId'"
                ).fetchone()[0]
                > 0
            )
    except Exception:
        _log.debug("NVTX textId detection failed", exc_info=True)

    if has_textid:
        text_expr = "COALESCE(n.text, s.value)"
        text_join = "LEFT JOIN StringIds s ON n.textId = s.id"
    else:
        text_expr = "n.text"
        text_join = ""

    nvtx_rows = conn.execute(sqlite_to_duckdb(
        f"""
        SELECT n.globalTid, n.start, n.[end], {text_expr} AS text
        FROM {nvtx_table} n
        {text_join}
        WHERE n.eventType = 59 AND n.[end] > n.start
        ORDER BY n.globalTid, n.start
        """
    )).fetchall()

    # StringIds lookup for kernel names — only fetch the IDs we need
    short_name_ids = {r[5] for r in kr_rows if r[5] is not None}
    if short_name_ids:
        placeholders = ",".join("?" for _ in short_name_ids)
        sid_rows = conn.execute(
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
                    entry for entry in open_stack[: best_idx + 1]
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
) -> list[dict]:
    """Attribute GPU kernels to their enclosing NVTX ranges.

    Uses a two-tier strategy:
      - Tier 1: ``nsys stats -r nvtx_gpu_proj_trace`` if CLI + .nsys-rep available
      - Tier 2: Python sort-merge O(N+M) sweep on .sqlite data

    Returns list of dicts with keys:
      nvtx_text, nvtx_depth, nvtx_path,
      kernel_name, k_start, k_end, k_dur_ns
    """
    # Tier 1: Try nsys recipe if we can find the .nsys-rep file
    if sqlite_path:
        nsys_rep = _find_nsys_rep(sqlite_path)
        if nsys_rep:
            result = _run_nsys_recipe(nsys_rep, trim)
            if result:
                return result

    # DuckDB: Try reading from purely precomputed Parquet bounds!
    try:
        import duckdb as _ddb
        if isinstance(conn, _ddb.DuckDBPyConnection):
            trim_sql = ""
            params = []
            if trim:
                trim_sql = "WHERE k_start >= ? AND k_end <= ?"
                params = [trim[0], trim[1]]

            cur = conn.execute(f"SELECT * FROM nvtx_kernel_map {trim_sql} ORDER BY k_start", params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception:
        _log.debug("DuckDB nvtx_kernel_map query failed, fallback to Python sweep", exc_info=True)

    # Tier 2: Python sort-merge fallback on SQLite
    return _sort_merge_attribute(conn, trim)
