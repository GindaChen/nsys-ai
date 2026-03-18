"""Root cause pattern matcher.

Programmatically detects known GPU performance anti-patterns from
the Book of Root Causes using existing skill data and direct SQL.

Each pattern has:
  - pattern: canonical root cause name (the unique identifier)
  - check: function that examines skill outputs and returns match info
  - severity: critical / warning / info
  - recommendation: actionable fix suggestion

This is a Python-level skill that runs other skills internally
to gather evidence, then matches against known patterns.
"""


import logging
import sqlite3

from ..base import Skill, SkillParam, _resolve_activity_tables

_log = logging.getLogger(__name__)


def _execute(conn: sqlite3.Connection, **kwargs):
    """Run all pattern matchers against the profile."""
    findings = []

    # Gather evidence from skills (forward trim kwargs)

    # Top kernels (request a larger slice to reduce bias in hotspot detection)
    top_kernels_data = _safe_execute("top_kernels", conn, limit=1000, **kwargs)
    # GPU idle gaps
    idle_gaps_data = _safe_execute("gpu_idle_gaps", conn, **kwargs)
    # Overlap
    overlap_data = _safe_execute("overlap_breakdown", conn, **kwargs)
    # Kernel launch overhead
    launch_data = _safe_execute("kernel_launch_overhead", conn, **kwargs)

    # --- GPU Bubbles (Pipeline Stalls) ---
    if idle_gaps_data:
        large_gaps = [g for g in idle_gaps_data
                      if g.get("gap_ms", 0) > 1.0]
        if len(large_gaps) >= 3:
            total_gap_ms = sum(g.get("gap_ms", 0) for g in large_gaps)
            findings.append({
                "pattern": "GPU Bubbles (Pipeline Stalls)",
                "severity": "warning",
                "evidence": f"{len(large_gaps)} gaps > 1ms detected, totaling {total_gap_ms:.1f}ms of idle time",
                "recommendation": (
                    "Use CUDA graphs, overlap data loading with compute, "
                    "or replace explicit cudaDeviceSynchronize with events."
                ),
            })

    # --- NCCL Serialization ---
    if overlap_data and len(overlap_data) > 0:
        ov = overlap_data[0]
        if "error" not in ov:
            overlap_pct = ov.get("overlap_pct", 100)
            nccl_only = ov.get("nccl_only_ms", 0)
            total = ov.get("total_ms", 1)
            if nccl_only > 0 and overlap_pct < 30:
                rec = (
                    "Tune DDP bucket sizes (bucket_cap_mb), "
                    "ensure NCCL runs on separate stream, "
                    "consider gradient compression or FSDP."
                )
                if overlap_pct == 0.0:
                    rec = (
                        "Overlap is EXACTLY 0.0%. NCCL is completely synchronous! "
                        "Ensure your script is not calling `torch.cuda.synchronize()` "
                        "after every step, or that your framework supports background "
                        "communication streams (e.g. PyTorch DDP with `find_unused_parameters=False`)."
                    )
                findings.append({
                    "pattern": "NCCL Serialization",
                    "severity": "critical",
                    "evidence": (
                        f"NCCL overlap only {overlap_pct}%, "
                        f"NCCL-only time: {nccl_only:.1f}ms / {total:.1f}ms"
                    ),
                    "recommendation": rec,
                })

    # --- Excessive H2D Transfers ---
    mem_data = _safe_execute("memory_bandwidth", conn, **kwargs)
    if mem_data:
        h2d = [r for r in mem_data if r.get("copyKind") == 1]
        if h2d:
            h2d_ms = h2d[0].get("total_dur_ms", 0)
            if h2d_ms > 50:  # > 50ms of H2D is suspicious
                findings.append({
                    "pattern": "Excessive H2D Transfers",
                    "severity": "warning",
                    "evidence": (
                        f"H2D transfers: {h2d_ms:.1f}ms total, "
                        f"{h2d[0].get('total_mb', 0):.1f}MB, "
                        f"{h2d[0].get('op_count', 0)} ops, "
                        f"avg bandwidth {h2d[0].get('avg_bandwidth_gbps', 0):.1f} GB/s"
                    ),
                    "recommendation": (
                        "Use pin_memory=True in DataLoader, keep "
                        "model params on GPU, accumulate metrics on GPU."
                    ),
                })

    # --- Small Kernel Overhead ---
    if launch_data:
        # overhead_us > kernel_ms*1000 means overhead > kernel duration
        high_overhead = [e for e in launch_data
                         if e.get("kernel_ms", 0) > 0
                         and e.get("overhead_us", 0) > e["kernel_ms"] * 1000]
        if len(high_overhead) >= 5:
            findings.append({
                "pattern": "Small Kernel Overhead",
                "severity": "warning",
                "evidence": f"{len(high_overhead)} kernels with launch overhead > kernel duration",
                "recommendation": (
                    "Use torch.compile() to fuse element-wise ops, "
                    "enable cudnn.benchmark, or use CUDA graphs."
                ),
            })

    # --- Kernel Hotspot ---
    if top_kernels_data and len(top_kernels_data) >= 2:
        # Compute percentage from total_ms since top_kernels doesn't have pct
        total_all_ms = sum(k.get("total_ms", 0) for k in top_kernels_data)
        if total_all_ms > 0:
            top_k = top_kernels_data[0]
            pct = (top_k.get("total_ms", 0) / total_all_ms) * 100
            if pct > 50:
                findings.append({
                    "pattern": "Kernel Hotspot",
                    "severity": "info",
                    "evidence": (
                        f"'{top_k.get('kernel_name', '?')}' accounts for {pct:.0f}% "
                        f"of time in the profiled top kernels "
                        f"({top_k.get('total_ms', 0):.1f}ms)"
                    ),
                    "recommendation": (
                        "Ensure shapes are multiples of 128 (H100) / 64 (A100), "
                        "use FlashAttention, or profile with NCU for details."
                    ),
                })

    # --- Compute-Communication Imbalance ---
    if overlap_data and len(overlap_data) > 0:
        ov = overlap_data[0]
        if "error" not in ov:
            compute_ms = ov.get("compute_only_ms", 0)
            nccl_ms_total = ov.get("nccl_only_ms", 0) + ov.get("overlap_ms", 0)
            if nccl_ms_total > 0 and compute_ms > 0:
                ratio = compute_ms / nccl_ms_total
                if ratio < 0.5:
                    findings.append({
                        "pattern": "Compute-Communication Imbalance",
                        "severity": "critical",
                        "evidence": (
                            f"Compute/NCCL ratio = {ratio:.2f} (healthy > 2.0). "
                            f"Compute: {compute_ms:.1f}ms, NCCL: {nccl_ms_total:.1f}ms"
                        ),
                        "recommendation": (
                            "Reduce tensor parallel degree (e.g. TP=4 → TP=1 "
                            "if model fits on one GPU), rebalance pipeline stages, "
                            "or pad sequences to uniform length."
                        ),
                    })

    # --- nsys anti-pattern checks (direct SQL) ---
    # These cover the 4 expert-rule recipes from nsys:
    # cuda_api_sync, cuda_memcpy_sync, cuda_memcpy_async, cuda_memset_sync
    findings += _check_sync_apis(conn, **kwargs)
    findings += _check_sync_memcpy(conn, **kwargs)
    findings += _check_pageable_memcpy(conn, **kwargs)
    findings += _check_sync_memset(conn, **kwargs)

    if not findings:
        findings.append({
            "pattern": "No Known Anti-Patterns Detected",
            "severity": "info",
            "evidence": "All checks passed — profile looks healthy",
            "recommendation": "Consider deep-diving with NCU for fine-grained analysis.",
        })

    return findings


# -----------------------------------------------------------------------
# nsys anti-pattern checkers — inline SQL for expert-rule recipe parity
# -----------------------------------------------------------------------

def _check_sync_apis(conn: sqlite3.Connection, **kwargs):
    """Detect excessive cuda*Synchronize calls.

    Uses a percentage-based threshold: sync time must exceed 2% of total
    GPU kernel time to avoid false positives from initialization phases.

    Note: Nsight exports use versioned API names (e.g. cudaDeviceSynchronize_v3020),
    so we use LIKE prefix matching via a two-step nameId resolution.
    """
    tables = _resolve_activity_tables(conn)
    runtime_tbl = tables.get("runtime")
    kernel_tbl = tables.get("kernel")
    if not runtime_tbl:
        return []

    try:
        # Step 1: resolve nameIds from StringIds (fast, tiny table)
        sync_names = conn.execute("""
            SELECT id, value FROM StringIds
            WHERE value LIKE 'cudaDeviceSynchronize%'
               OR value LIKE 'cudaStreamSynchronize%'
               OR value LIKE 'cudaEventSynchronize%'
               OR value LIKE 'cudaStreamWaitEvent%'
        """).fetchall()
        if not sync_names:
            return []

        name_ids = [r[0] for r in sync_names]
        placeholders = ",".join(str(nid) for nid in name_ids)

        # Step 2: count sync calls by nameId (fast with index)
        rows = conn.execute(f"""
            SELECT nameId, COUNT(*) AS call_count,
                   SUM([end] - start) AS total_ns
            FROM {runtime_tbl}
            WHERE nameId IN ({placeholders})
            GROUP BY nameId
        """).fetchall()
        if not rows:
            return []

        # Map nameId back to name
        id_to_name = {r[0]: r[1] for r in sync_names}
        total_sync_ns = sum(r[2] for r in rows)
        total_sync_ms = total_sync_ns / 1e6
        call_count = sum(r[1] for r in rows)
        # Strip version suffixes for cleaner display (cudaDeviceSynchronize_v3020 → cudaDeviceSynchronize)
        api_names = ", ".join(
            sorted({id_to_name[r[0]].split("_v")[0] for r in rows})
        )

        # Total GPU kernel time as baseline for percentage threshold
        total_gpu_ns = 0
        if kernel_tbl:
            gpu_row = conn.execute(
                f"SELECT SUM([end] - start) FROM {kernel_tbl}"
            ).fetchone()
            total_gpu_ns = gpu_row[0] or 0 if gpu_row else 0

        # Percentage-based threshold: sync time > 2% of total GPU time
        # Also require absolute minimum of 1ms to filter trivial cases
        sync_pct = (total_sync_ns / total_gpu_ns * 100) if total_gpu_ns > 0 else 100
        if total_sync_ms >= 1.0 and sync_pct >= 2.0:
            return [{
                "pattern": "Excessive Synchronization",
                "severity": "warning",
                "evidence": (
                    f"{call_count} sync calls totalling {total_sync_ms:.1f}ms "
                    f"({sync_pct:.1f}% of GPU time). APIs: {api_names}"
                ),
                "recommendation": (
                    "Remove .item()/.cpu() from the training loop, "
                    "use torch.cuda.set_sync_debug_mode(1) to find hidden syncs, "
                    "replace cudaDeviceSynchronize with event-based dependencies. "
                    "Run `nsys recipe cuda_api_sync <profile.nsys-rep>` for a detailed breakdown."
                ),
            }]
    except sqlite3.OperationalError as e:
        _log.debug("root_cause_matcher (_check_sync_apis): %s", e)
    return []


def _check_sync_memcpy(conn: sqlite3.Connection, **kwargs):
    """Detect synchronous cudaMemcpy (not cudaMemcpyAsync).

    Synchronous memcpy blocks the host until the transfer completes,
    preventing CPU/GPU overlap.

    Note: Nsight exports use versioned API names (e.g. cudaMemcpy_v3020).
    We match any name starting with 'cudaMemcpy' but NOT 'cudaMemcpyAsync'.
    """
    tables = _resolve_activity_tables(conn)
    runtime_tbl = tables.get("runtime")
    memcpy_tbl = tables.get("memcpy")
    if not runtime_tbl or not memcpy_tbl:
        return []

    try:
        # Step 1: find nameIds for sync cudaMemcpy (NOT async)
        sync_names = conn.execute("""
            SELECT id, value FROM StringIds
            WHERE value LIKE 'cudaMemcpy%'
              AND value NOT LIKE 'cudaMemcpyAsync%'
        """).fetchall()
        if not sync_names:
            return []

        name_ids = [r[0] for r in sync_names]
        placeholders = ",".join(str(nid) for nid in name_ids)

        # Step 2: find memcpy ops correlated with sync runtime calls
        row = conn.execute(f"""
            SELECT COUNT(*) AS count,
                   COALESCE(SUM(m.bytes), 0) AS total_bytes,
                   COALESCE(SUM(m.[end] - m.start), 0) AS total_ns
            FROM {runtime_tbl} r
            JOIN {memcpy_tbl} m ON r.correlationId = m.correlationId
            WHERE r.nameId IN ({placeholders})
        """).fetchone()
        if not row or row[0] == 0:
            return []

        count, total_bytes, total_ns = row
        total_ms = total_ns / 1e6
        total_mb = total_bytes / 1e6

        return [{
            "pattern": "Synchronous Memcpy",
            "severity": "warning",
            "evidence": (
                f"{count} sync cudaMemcpy calls: {total_mb:.1f}MB in {total_ms:.1f}ms. "
                f"These block the host thread."
            ),
            "recommendation": (
                "Replace cudaMemcpy with cudaMemcpyAsync + pinned memory. "
                "Use pin_memory=True in DataLoader and non_blocking=True in .to(device). "
                "Run `nsys recipe cuda_memcpy_sync <profile.nsys-rep>` for a detailed breakdown."
            ),
        }]
    except sqlite3.OperationalError as e:
        _log.debug("root_cause_matcher (_check_sync_memcpy): %s", e)
    return []


def _check_pageable_memcpy(conn: sqlite3.Connection, **kwargs):
    """Detect async memcpy using pageable (non-pinned) memory.

    When cudaMemcpyAsync is called with pageable memory, the driver silently
    falls back to a synchronous copy, defeating the purpose of async.

    Memory kind values (Nsight CUPTI schema):
      0 = Unknown, 1 = Pageable, 2 = Device, 3 = Array,
      4 = Managed, 5 = Device Static, 6 = Managed Static, 7 = Pinned
    Source: CUPTI_ACTIVITY_KIND_MEMCPY table schema, Nsight Systems export.
    """
    tables = _resolve_activity_tables(conn)
    memcpy_tbl = tables.get("memcpy")
    if not memcpy_tbl:
        return []

    try:
        row = conn.execute(f"""
            SELECT COUNT(*) AS pageable_count,
                   COALESCE(SUM(bytes), 0) AS total_bytes,
                   COALESCE(SUM([end] - start), 0) AS total_ns
            FROM {memcpy_tbl}
            WHERE srcKind = 1 OR dstKind = 1
        """).fetchone()
        if not row or row[0] == 0:
            return []

        count, total_bytes, total_ns = row
        total_ms = total_ns / 1e6
        total_mb = total_bytes / 1e6

        return [{
            "pattern": "Pageable Memory in Async Memcpy",
            "severity": "warning",
            "evidence": (
                f"{count} memcpy ops using pageable memory: {total_mb:.1f}MB in "
                f"{total_ms:.1f}ms. Pageable → async memcpy silently becomes sync."
            ),
            "recommendation": (
                "Use pinned (page-locked) memory: cudaMallocHost() / "
                "pin_memory=True in DataLoader. This enables true async H2D overlap. "
                "Run `nsys recipe cuda_memcpy_async <profile.nsys-rep>` for details on pageable fallback."
            ),
        }]
    except sqlite3.OperationalError as e:
        _log.debug("root_cause_matcher (_check_pageable_memcpy): %s", e)
    return []


def _check_sync_memset(conn: sqlite3.Connection, **kwargs):
    """Detect synchronous cudaMemset (not cudaMemsetAsync).

    Synchronous memset blocks the host. Usually a minor issue but
    can add up in tight loops.

    Note: Nsight exports use versioned API names (e.g. cudaMemset_v3020).
    We match any name starting with 'cudaMemset' but NOT 'cudaMemsetAsync'.
    """
    tables = _resolve_activity_tables(conn)
    runtime_tbl = tables.get("runtime")
    memset_tbl = tables.get("memset")
    if not runtime_tbl or not memset_tbl:
        return []

    try:
        # Step 1: find nameIds for sync cudaMemset (NOT async)
        sync_names = conn.execute("""
            SELECT id, value FROM StringIds
            WHERE value LIKE 'cudaMemset%'
              AND value NOT LIKE 'cudaMemsetAsync%'
        """).fetchall()
        if not sync_names:
            return []

        name_ids = [r[0] for r in sync_names]
        placeholders = ",".join(str(nid) for nid in name_ids)

        # Step 2: find memset ops correlated with sync runtime calls
        row = conn.execute(f"""
            SELECT COUNT(*) AS count,
                   COALESCE(SUM(ms.[end] - ms.start), 0) AS total_ns
            FROM {runtime_tbl} r
            JOIN {memset_tbl} ms ON r.correlationId = ms.correlationId
            WHERE r.nameId IN ({placeholders})
        """).fetchone()
        if not row or row[0] == 0:
            return []

        count, total_ns = row
        total_ms = total_ns / 1e6

        return [{
            "pattern": "Synchronous Memset",
            "severity": "info",
            "evidence": (
                f"{count} sync cudaMemset calls: {total_ms:.2f}ms total. "
                f"These block the host thread."
            ),
            "recommendation": (
                "Replace cudaMemset with cudaMemsetAsync on the appropriate stream. "
                "Run `nsys recipe cuda_memset_sync <profile.nsys-rep>` for a detailed breakdown."
            ),
        }]
    except sqlite3.OperationalError as e:
        _log.debug("root_cause_matcher (_check_sync_memset): %s", e)
    return []


# -----------------------------------------------------------------------

def _safe_execute(skill_name, conn: sqlite3.Connection, **kwargs):
    """Execute a skill, returning [] on any error."""
    from ...skills.registry import get_skill
    try:
        skill = get_skill(skill_name)
        if skill is None:
            return []
        return skill.execute(conn, **kwargs)
    except sqlite3.OperationalError as e:
        _log.debug("root_cause_matcher (%s): %s", skill_name, e)
        return []


def _format(rows):
    if not rows:
        return "(No patterns checked)"
    lines = ["── Root Cause Pattern Analysis ──"]
    for f in rows:
        icon = {"critical": "🔴", "warning": "🟡", "info": "🟢"}.get(f["severity"], "⚪")
        lines.append(f"\n{icon} {f['pattern']}")
        lines.append(f"  Evidence: {f['evidence']}")
        lines.append(f"  Fix: {f['recommendation']}")
    return "\n".join(lines)


SKILL = Skill(
    name="root_cause_matcher",
    title="Root Cause Pattern Matcher",
    description=(
        "Automatically detects known GPU performance anti-patterns from the "
        "Book of Root Causes: GPU bubbles, NCCL serialization, kernel hotspots, "
        "small kernel overhead, compute-communication imbalance, excessive "
        "synchronization, synchronous memcpy/memset, pageable memory in async "
        "transfers. Returns matched patterns with evidence and fix recommendations."
    ),
    category="analysis",
    execute_fn=_execute,
    format_fn=_format,
    params=[SkillParam("device", "GPU device ID", "int", False, 0)],
    tags=["root-cause", "pattern", "diagnosis", "analysis", "recommendation",
          "sync", "memcpy", "memset", "anti-pattern"],
)

