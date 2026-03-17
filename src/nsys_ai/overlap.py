"""
overlap.py — Compute/communication overlap analysis and NCCL breakdown.

Quantifies how much GPU compute overlaps with NCCL communication,
detects training iterations, and breaks down collective operations.
"""

from collections import defaultdict

from .profile import Profile

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
    kernels = prof.kernels(device, trim)
    if not kernels:
        return {"error": "no kernels"}

    # Separate compute vs NCCL intervals
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

    # Merge overlapping intervals within each category
    compute_merged = _merge_intervals(compute_intervals)
    nccl_merged = _merge_intervals(nccl_intervals)

    # Calculate coverage
    compute_ns = _total_covered(compute_merged)
    nccl_ns = _total_covered(nccl_merged)

    # Overlap = time covered by BOTH compute and NCCL
    overlap_ns = _intersection_coverage(compute_merged, nccl_merged)

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
    }


# ── NCCL collective breakdown ──────────────────────────────────────


def nccl_breakdown(prof: Profile, device: int, trim: tuple[int, int] | None = None) -> list[dict]:
    """
    Break down NCCL operations by collective type.

    Returns a list sorted by total time, each:
        {type, count, total_ms, avg_ms, min_ms, max_ms, pct}
    """
    kernels = prof.kernels(device, trim)
    nccl_kernels = [k for k in kernels if classify_kernel(k["name"]).startswith("nccl_")]

    if not nccl_kernels:
        return []

    total_nccl_ns = sum(k["end"] - k["start"] for k in nccl_kernels)

    by_type = defaultdict(list)
    for k in nccl_kernels:
        ctype = classify_kernel(k["name"])
        dur_ns = k["end"] - k["start"]
        by_type[ctype].append(dur_ns)

    result = []
    for ctype, durs in sorted(by_type.items(), key=lambda x: -sum(x[1])):
        total = sum(durs)
        result.append(
            {
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

    with prof._lock:
        pri_nvtx = prof.conn.execute(
            f"""
            SELECT {text_expr} AS text, n.start, n.[end] FROM {nvtx_table} n
            {text_join}
            WHERE {text_expr} LIKE ? AND n.[end] > n.start AND n.globalTid = ?
              AND n.start >= ? AND n.start <= ?
            ORDER BY n.start
        """,
            (f"%{marker}%", primary_tid, time_range[0] - pad, time_range[1]),
        ).fetchall()

    # Filter to non-overlapping (top-level only)
    iterations = []
    last_end = 0
    for n in pri_nvtx:
        if n["start"] >= last_end:
            iterations.append(n)
            last_end = n["end"]

    if not iterations:
        return []

    # For each iteration, count kernels and compute GPU time
    with prof._lock:
        rt_all = prof.conn.execute(
            f"""
            SELECT start, [end], correlationId FROM {runtime_table}
            WHERE globalTid = ? ORDER BY start
        """,
            (primary_tid,),
        ).fetchall()

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


def _merge_intervals(intervals):
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


def _total_covered(merged):
    """Total time covered by merged intervals."""
    return sum(end - start for start, end in merged)


def _intersection_coverage(a, b):
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
    """Format NCCL breakdown as readable text."""
    if not breakdown:
        return "No NCCL collectives found"
    lines = ["NCCL Collective Breakdown"]
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
