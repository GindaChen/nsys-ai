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
import os
import shutil
import sqlite3
import subprocess  # nosec B404 — list args, no shell
from collections import defaultdict

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
    all_tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]

    def _find(prefix: str) -> str:
        # Prefer exact match (e.g. CUPTI_ACTIVITY_KIND_KERNEL)
        if prefix in all_tables:
            return prefix
        # Otherwise pick the first sorted versioned variant, aligning
        # with the resolution used by ensure_indexes() / _resolve_activity_tables().
        candidates = sorted(t for t in all_tables if t.startswith(prefix + "_V"))
        if candidates:
            return candidates[0]
        return prefix

    kernel_table = _find("CUPTI_ACTIVITY_KIND_KERNEL")
    runtime_table = _find("CUPTI_ACTIVITY_KIND_RUNTIME")
    nvtx_table = _find("NVTX_EVENTS")

    # Trim clause for SQL queries
    trim_sql = ""
    trim_params: tuple = ()
    if trim:
        trim_sql = "AND k.start >= ? AND k.[end] <= ?"
        trim_params = (trim[0], trim[1])

    # Phase 1: Kernel → Runtime via correlationId (indexed, fast)
    kr_rows = conn.execute(
        f"""
        SELECT r.globalTid, r.start, r.[end],
               k.start AS ks, k.[end] AS ke, k.shortName
        FROM {kernel_table} k
        JOIN {runtime_table} r ON r.correlationId = k.correlationId
        WHERE 1=1 {trim_sql}
        ORDER BY r.globalTid, r.start
        """,
        trim_params,
    ).fetchall()

    if not kr_rows:
        return []

    # Phase 2: Load NVTX ranges (eventType 59 = NVTX push/pop range)
    # Resolve text expression (handles textId vs text column)
    has_textid = False
    try:
        has_textid = (
            conn.execute(
                f"SELECT COUNT(*) FROM pragma_table_info('{nvtx_table}') WHERE name='textId'"
            ).fetchone()[0]
            > 0
        )
    except Exception:
        pass

    if has_textid:
        text_expr = "COALESCE(n.text, s.value)"
        text_join = "LEFT JOIN StringIds s ON n.textId = s.id"
    else:
        text_expr = "n.text"
        text_join = ""

    nvtx_rows = conn.execute(
        f"""
        SELECT n.globalTid, n.start, n.[end], {text_expr} AS text
        FROM {nvtx_table} n
        {text_join}
        WHERE n.eventType = 59 AND n.[end] > n.start
        ORDER BY n.globalTid, n.start
        """
    ).fetchall()

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
            # Advance NVTX pointer, pushing any ranges that have opened by r_start
            while nvtx_idx < len(nvtx_list) and nvtx_list[nvtx_idx][0] <= r_start:
                open_stack.append(nvtx_list[nvtx_idx])
                nvtx_idx += 1

            # Pop NVTX ranges that have already closed before this runtime starts
            while open_stack and open_stack[-1][1] < r_start:
                open_stack.pop()

            # Find innermost enclosing NVTX (scan stack from top)
            best_nvtx = None
            for i in range(len(open_stack) - 1, -1, -1):
                ns, ne, nt = open_stack[i]
                if ns <= r_start and ne >= r_end:
                    best_nvtx = nt
                    break

            if best_nvtx is not None:
                results.append(
                    {
                        "nvtx_text": best_nvtx,
                        "kernel_name": sid_map.get(short_name, f"kernel_{short_name}"),
                        "k_start": k_start,
                        "k_end": k_end,
                        "k_dur_ns": k_end - k_start,
                    }
                )

    return results


# ── Public API ──────────────────────────────────────────────────────


def attribute_kernels_to_nvtx(
    conn: sqlite3.Connection,
    sqlite_path: str | None = None,
    trim: tuple[int, int] | None = None,
) -> list[dict]:
    """Attribute GPU kernels to their enclosing NVTX ranges.

    Uses a two-tier strategy:
      - Tier 1: ``nsys stats -r nvtx_gpu_proj_trace`` if CLI + .nsys-rep available
      - Tier 2: Python sort-merge O(N+M) sweep on .sqlite data

    Returns list of dicts with keys:
      nvtx_text, kernel_name, k_start, k_end, k_dur_ns
    """
    # Tier 1: Try nsys recipe if we can find the .nsys-rep file
    if sqlite_path:
        nsys_rep = _find_nsys_rep(sqlite_path)
        if nsys_rep:
            result = _run_nsys_recipe(nsys_rep, trim)
            if result:
                return result

    # Tier 2: Python sort-merge fallback
    return _sort_merge_attribute(conn, trim)
