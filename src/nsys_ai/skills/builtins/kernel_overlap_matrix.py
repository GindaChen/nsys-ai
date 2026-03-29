"""Kernel overlap matrix — pairwise overlap between kernel categories.

Computes comm×comm and comm×compute overlap matrices.
Includes memcpy (H2D/D2H/D2D/P2P) for data-movement contention analysis.

This is a Python-level skill (execute_fn) because it needs interval
merge and intersection logic that can't be expressed in a single SQL query.
"""

from collections import defaultdict

from ..base import Skill, SkillParam

# Memcpy copyKind → category mapping (from CUPTI docs)
_COPY_KINDS = {
    1: "memcpy_h2d",
    2: "memcpy_d2h",
    8: "memcpy_d2d",
    10: "memcpy_p2p",
}


def _query_memcpy(prof, device, trim):
    """Query memcpy intervals from CUPTI_ACTIVITY_KIND_MEMCPY."""
    memcpy_table = None
    for t in prof.schema.tables:
        if t == "CUPTI_ACTIVITY_KIND_MEMCPY" or t.startswith(
            "CUPTI_ACTIVITY_KIND_MEMCPY"
        ):
            memcpy_table = t
            break
    import re
    if memcpy_table is None or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", memcpy_table):
        return []

    sql = f"""\
SELECT copyKind, start, [end]
FROM {memcpy_table}
WHERE deviceId = ?"""
    params = [device]
    if trim:
        sql += " AND start >= ? AND [end] <= ?"
        params.extend(trim)
    sql += " ORDER BY start"
    return prof._duckdb_query(sql, params)


def _execute(conn, **kwargs):
    from ...overlap import (
        classify_kernel,
        intersection_coverage,
        merge_intervals,
        total_covered,
    )
    from ...profile import Profile

    prof = Profile._from_conn(conn)
    device = int(kwargs.get("device", 0))

    # Parse trim
    trim = None
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    if trim_start is not None and trim_end is not None:
        trim = (int(trim_start), int(trim_end))

    # 1. Query and classify kernels
    kernels = prof.kernels(device, trim)
    if not kernels:
        return [{"error": "no kernels found", "device": device}]

    categories: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for k in kernels:
        cat = classify_kernel(k["name"])
        categories[cat].append((k["start"], k["end"]))

    # 2. Query and classify memcpy
    memcpy_rows = _query_memcpy(prof, device, trim)
    for r in memcpy_rows:
        cat = _COPY_KINDS.get(r["copyKind"], f"memcpy_{r['copyKind']}")
        categories[cat].append((r["start"], r["end"]))

    # 3. Merge intervals within each category
    merged = {cat: merge_intervals(ivs) for cat, ivs in categories.items()}

    # 4. Compute diagonal (self-time) and pairwise overlap
    cats = sorted(merged.keys())
    self_time = {cat: total_covered(merged[cat]) for cat in cats}

    # Upper triangle pairwise overlap
    pairwise = {}
    for i, a in enumerate(cats):
        for j in range(i + 1, len(cats)):
            b = cats[j]
            overlap_ns = intersection_coverage(merged[a], merged[b])
            pairwise[(a, b)] = overlap_ns

    # 5. Build result rows (upper triangle including diagonal)
    pairs = []
    for a in cats:
        for b in cats:
            if a > b:
                continue
            if a == b:
                # Diagonal: self-time
                val_ns = self_time[a]
                pairs.append({
                    "category_a": a,
                    "category_b": b,
                    "overlap_ns": val_ns,
                    "overlap_ms": round(val_ns / 1e6, 2),
                    "is_diagonal": True,
                    "pct_of_a": None,
                    "pct_of_b": None,
                })
            else:
                # Off-diagonal: pairwise overlap
                val_ns = pairwise.get((a, b), 0)
                a_total = self_time.get(a, 0)
                b_total = self_time.get(b, 0)
                pairs.append({
                    "category_a": a,
                    "category_b": b,
                    "overlap_ns": val_ns,
                    "overlap_ms": round(val_ns / 1e6, 2),
                    "is_diagonal": False,
                    "pct_of_a": (
                        round(100 * val_ns / a_total, 1) if a_total else 0
                    ),
                    "pct_of_b": (
                        round(100 * val_ns / b_total, 1) if b_total else 0
                    ),
                })

    return pairs


def _format(rows):
    if not rows:
        return "(No kernel activity found)"
    if rows and "error" in rows[0]:
        return f"(kernel_overlap_matrix: {rows[0]['error']})"

    # Collect categories and build lookup
    cats_set = set()
    lookup = {}
    for r in rows:
        a, b = r["category_a"], r["category_b"]
        cats_set.add(a)
        cats_set.add(b)
        lookup[(a, b)] = r
        lookup[(b, a)] = r  # symmetric

    # Order: compute first, then nccl_* sorted, then memcpy_* sorted
    def _sort_key(cat):
        if cat == "compute":
            return (0, cat)
        if cat.startswith("nccl_"):
            return (1, cat)
        if cat.startswith("memcpy_"):
            return (2, cat)
        return (3, cat)

    cats = sorted(cats_set, key=_sort_key)

    # Filter out categories with zero total time
    cats = [c for c in cats if lookup.get((c, c), {}).get("overlap_ns", 0) > 0]

    if not cats:
        return "(No active kernel categories)"

    # Build ASCII matrix
    # Abbreviate long category names
    abbrev = {}
    for c in cats:
        if c == "compute":
            abbrev[c] = "compute"
        elif c.startswith("nccl_"):
            abbrev[c] = c[5:]  # strip "nccl_" prefix
        elif c.startswith("memcpy_"):
            abbrev[c] = c  # keep full name for clarity
        else:
            abbrev[c] = c

    col_w = max(10, max(len(abbrev[c]) for c in cats) + 2)
    label_w = max(len(abbrev[c]) for c in cats) + 2

    lines = ["── Kernel Overlap Matrix ──"]
    # Header row
    header = " " * label_w + "".join(abbrev[c].rjust(col_w) for c in cats)
    lines.append(header)

    # Data rows
    for a in cats:
        row_parts = [abbrev[a].rjust(label_w)]
        for b in cats:
            pair = lookup.get((a, b))
            if pair:
                val_ms = pair["overlap_ms"]
                row_parts.append(f"{val_ms:.1f}ms".rjust(col_w))
            else:
                row_parts.append("—".rjust(col_w))
        lines.append("".join(row_parts))

    # Summary: compute overall NCCL overlap efficiency
    total_nccl_ns = 0
    nccl_overlapped_ns = 0
    for c in cats:
        if c.startswith("nccl_"):
            diag = lookup.get((c, c))
            if diag:
                total_nccl_ns += diag["overlap_ns"]
            # Overlap with compute
            pair = lookup.get(("compute", c)) or lookup.get((c, "compute"))
            if pair and not pair["is_diagonal"]:
                nccl_overlapped_ns += pair["overlap_ns"]

    if total_nccl_ns > 0:
        eff = round(100 * nccl_overlapped_ns / total_nccl_ns, 1)
        lines.append(f"\n  NCCL overlap efficiency: {eff}% of NCCL time hidden behind compute")

    return "\n".join(lines)


SKILL = Skill(
    name="kernel_overlap_matrix",
    title="Kernel Overlap Matrix",
    description=(
        "Computes pairwise overlap between kernel categories: "
        "comm×comm, comm×compute, and memcpy. "
        "Shows how much NCCL communication is hidden behind compute, "
        "whether different collectives contend with each other, "
        "and whether data transfers overlap with GPU work. "
        "Critical for multi-parallelism debugging (TP+PP+DP)."
    ),
    category="communication",
    execute_fn=_execute,
    format_fn=_format,
    params=[
        SkillParam("device", "GPU device ID", "int", False, 0),
    ],
    tags=[
        "overlap", "matrix", "nccl", "compute", "communication",
        "contention", "distributed", "multi-gpu", "memcpy",
    ],
)
