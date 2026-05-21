"""Classify NCCL kernels by their leaf NVTX label (call mode).

Buckets each NCCL kernel into one of three call modes based on the
leaf NVTX scope open at launch time. The dominant bucket selects the
fix path:

  - eager             → caller-side: ``async_op=True`` / dedicated stream
  - inductor_captured → ``torch._inductor.config`` (functional
                        collectives, ``reorder_for_compute_comm_overlap``)
  - temporal_only     → leaf NVTX uninformative; reach for
                        ``nccl_payload_breakdown`` / ``overlap_breakdown``

Classifies by **leaf** label, not ancestor-path containment. See
``nvtx_kernel_map``'s module docstring for why (temporal vs lexical
containment).
"""

from ..base import Skill

_INDUCTOR_LEAF_MARKERS = ("## Call CompiledFxGraph",)
_EAGER_LEAF_PREFIXES = ("c10d::", "nccl")


def _classify_leaf(leaf: str) -> str:
    if not leaf:
        return "temporal_only"
    if any(leaf.startswith(p) for p in _EAGER_LEAF_PREFIXES):
        return "eager"
    if any(m in leaf for m in _INDUCTOR_LEAF_MARKERS):
        return "inductor_captured"
    return "temporal_only"


def _is_nccl_kernel(name: str) -> bool:
    return bool(name) and "nccl" in name.lower()


def _execute(conn, **kwargs):
    from ...nvtx_attribution import attribute_kernels_to_nvtx

    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    trim = (
        (int(trim_start), int(trim_end))
        if trim_start is not None and trim_end is not None
        else None
    )
    sqlite_path = kwargs.get("_sqlite_path")

    # kernel_name_substring is the SQL-pushdown hint (advisory); the
    # _is_nccl_kernel loop below covers backends that ignore it.
    rows = attribute_kernels_to_nvtx(
        conn, sqlite_path=sqlite_path, trim=trim, limit=None,
        kernel_name_substring="nccl",
    )

    buckets: dict[str, dict[str, int]] = {
        "eager": {"count": 0, "ns": 0},
        "inductor_captured": {"count": 0, "ns": 0},
        "temporal_only": {"count": 0, "ns": 0},
    }
    for r in rows:
        if not _is_nccl_kernel(r.get("kernel_name", "")):
            continue
        bucket = _classify_leaf(r.get("nvtx_text", "") or "")
        buckets[bucket]["count"] += 1
        buckets[bucket]["ns"] += int(r.get("k_dur_ns") or 0)

    total_count = sum(b["count"] for b in buckets.values())
    total_ns = sum(b["ns"] for b in buckets.values())

    if total_count == 0:
        return [{"error": "No NCCL kernels found (single-GPU or no NCCL tracing)."}]

    return [
        {
            "_summary": True,
            "total_nccl_kernels": total_count,
            "total_nccl_ms": round(total_ns / 1e6, 3),
        },
        *[
            {
                "bucket": name,
                "count": b["count"],
                "ms": round(b["ns"] / 1e6, 3),
                "pct": round(b["count"] / total_count * 100, 1),
                "ms_pct": round(b["ns"] / total_ns * 100, 1) if total_ns > 0 else 0.0,
            }
            for name, b in buckets.items()
        ],
    ]


def _format(rows):
    if not rows or "error" in rows[0]:
        return f"(NCCL compile context: {rows[0].get('error', 'no data') if rows else 'no data'})"
    s = rows[0]
    lines = [
        "── NCCL Call-Mode Breakdown (by leaf NVTX) ──",
        f"  Total NCCL kernels: {s['total_nccl_kernels']:,}  ({s['total_nccl_ms']:.3f} ms)",
        "",
        f"  {'bucket':<20}  {'count':>8}  {'ms':>12}  {'count_pct':>10}  {'ms_pct':>10}",
        "  " + "─" * 66,
    ]
    for r in rows[1:]:
        lines.append(
            f"  {r['bucket']:<20}  {r['count']:>8}  {r['ms']:>12.3f}  "
            f"{r['pct']:>9.1f}%  {r['ms_pct']:>9.1f}%"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="nccl_compile_context_breakdown",
    title="NCCL Call-Mode Breakdown (eager vs inductor-captured vs temporal-only)",
    description=(
        "Classifies NCCL kernels by leaf NVTX label into eager / "
        "inductor_captured / temporal_only buckets. Decides whether a "
        "collective perf fix lives in user code or in torch._inductor.config."
    ),
    category="communication",
    execute_fn=_execute,
    format_fn=_format,
    tags=["nccl", "communication", "distributed", "torch-compile", "nvtx"],
)
