"""Classify NCCL kernels by their leaf NVTX label (call mode).

Buckets each NCCL kernel into one of three call modes based on the
leaf NVTX scope open at launch time. The dominant bucket selects the
fix path:

  - eager             → caller-side: ``async_op=True`` / dedicated stream
  - inductor_captured → ``torch._inductor.config`` (functional
                        collectives, ``reorder_for_compute_comm_overlap``)
  - temporal_only     → leaf NVTX uninformative; reach for
                        ``nccl_payload_breakdown`` / ``overlap_breakdown``

Exact ties are reported as mixed and do not select a bucket-specific fix path.

Classifies by **leaf** label, not ancestor-path containment. See
``nvtx_kernel_map``'s module docstring for why (temporal vs lexical
containment).
"""

from ..base import Skill, requires_pushpop_nvtx

_INDUCTOR_LEAF_MARKERS = ("## Call CompiledFxGraph",)
_EAGER_LEAF_PREFIXES = ("c10d::", "nccl")
_BUCKET_ORDER = ("eager", "inductor_captured", "temporal_only")
_LIMITATIONS = [
    "Classification covers only NCCL kernels attributed to a Push/Pop NVTX leaf; "
    "NCCL kernels without an attributable leaf are excluded.",
    "Classification aggregates attributable NCCL kernels across all captured devices; "
    "the EvidenceBuilder device selection is not applied by this skill.",
    "temporal_only means the leaf label did not identify eager or compiled context; "
    "it does not prove that torch.compile was absent.",
    "Call mode identifies where to investigate, not overlap quality or recoverable time.",
]
_ACTIONS_BY_BUCKET = {
    "eager": [
        "For eager NCCL calls, use async_op=True and a dedicated CUDA stream to enable overlap."
    ],
    "inductor_captured": [
        "For inductor-captured NCCL calls, use functional collectives and evaluate "
        "torch._inductor.config.reorder_for_compute_comm_overlap."
    ],
    "temporal_only": [
        "For temporal-only attribution, run nccl_payload_breakdown and overlap_breakdown "
        "to identify the collective and exposed communication."
    ],
}


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
    # A profile captured without NVTX ranges cannot be attributed to regions.
    # Say so rather than raising: callers catch and log, so an exception here
    # removes the skill from the output with no trace that it was even asked.
    from ...nvtx_attribution import attribute_kernels_to_nvtx

    guard = requires_pushpop_nvtx(conn, needs="Call-mode classification")
    if guard:
        return guard

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
        name: {"count": 0, "ns": 0} for name in _BUCKET_ORDER
    }
    nccl_rows = []
    for r in rows:
        if not _is_nccl_kernel(r.get("kernel_name", "")):
            continue
        nccl_rows.append(r)
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
            "total_nccl_ns": total_ns,
            "total_nccl_ms": round(total_ns / 1e6, 3),
            "span_start_ns": min(int(r["k_start"]) for r in nccl_rows),
            "span_end_ns": max(int(r["k_end"]) for r in nccl_rows),
            "device_scope": "all_captured_devices",
        },
        *[
            {
                "bucket": name,
                "count": b["count"],
                "ns": b["ns"],
                "ms": round(b["ns"] / 1e6, 3),
                "pct": round(b["count"] / total_count * 100, 1),
                "ms_pct": round(b["ns"] / total_ns * 100, 1) if total_ns > 0 else 0.0,
            }
            for name, b in buckets.items()
        ],
    ]


def _to_findings(rows: list[dict], *, context: dict | None = None) -> list:
    from nsys_ai.annotation import EvidenceRow, Finding, TraceSelection

    if not rows or "error" in rows[0]:
        return []

    summary = rows[0]
    buckets = {r.get("bucket"): r for r in rows[1:] if r.get("bucket") in _BUCKET_ORDER}
    if not buckets or not summary.get("total_nccl_kernels"):
        return []

    total_ns = sum(int(buckets.get(name, {}).get("ns") or 0) for name in _BUCKET_ORDER)
    dominant_basis = "duration" if total_ns > 0 else "count"
    dominant_key = "ns" if dominant_basis == "duration" else "count"
    max_value = max(int(buckets.get(name, {}).get(dominant_key) or 0) for name in _BUCKET_ORDER)
    dominant_buckets = [
        name
        for name in _BUCKET_ORDER
        if int(buckets.get(name, {}).get(dominant_key) or 0) == max_value
    ]
    dominant = dominant_buckets[0] if len(dominant_buckets) == 1 else "mixed"
    start_ns = int(summary.get("span_start_ns") or 0)
    end_ns = int(summary.get("span_end_ns") or start_ns)
    finding_id = f"nccl_compile_context_{start_ns}"
    profile_id = (context or {}).get("profile_id", "unknown")
    selection = TraceSelection(
        id=f"sel_{finding_id}",
        profile_id=profile_id,
        source="skill:nccl_compile_context_breakdown",
        start_ns=start_ns,
        end_ns=end_ns,
        label="NCCL kernels classified by leaf NVTX call mode across all captured devices",
    )

    values = {
        "total_nccl_kernels": int(summary["total_nccl_kernels"]),
        "total_nccl_ns": total_ns,
        "total_nccl_ms": float(summary.get("total_nccl_ms") or 0.0),
        "dominant_bucket": dominant,
        "dominant_buckets": dominant_buckets,
        "dominant_basis": dominant_basis,
        "device_scope": "all_captured_devices",
    }
    units = {
        "total_nccl_kernels": "count",
        "total_nccl_ns": "ns",
        "total_nccl_ms": "ms",
    }
    for name in _BUCKET_ORDER:
        bucket = buckets.get(name, {})
        values.update(
            {
                f"{name}_count": int(bucket.get("count") or 0),
                f"{name}_ns": int(bucket.get("ns") or 0),
                f"{name}_ms": float(bucket.get("ms") or 0.0),
                f"{name}_count_pct": float(bucket.get("pct") or 0.0),
                f"{name}_ms_pct": float(bucket.get("ms_pct") or 0.0),
            }
        )
        units.update(
            {
                f"{name}_count": "count",
                f"{name}_ns": "ns",
                f"{name}_ms": "ms",
                f"{name}_count_pct": "percent",
                f"{name}_ms_pct": "percent",
            }
        )

    evidence_row = EvidenceRow(
        id=f"ev_{finding_id}",
        source_skill="nccl_compile_context_breakdown",
        values=values,
        units=units,
        selection_id=selection.id,
        provenance={
            "row_kind": "call_mode_classification",
            "classification_basis": "leaf_pushpop_nvtx",
            "dominant_basis": dominant_basis,
            "dominant_buckets": dominant_buckets,
            "device_scope": "all_captured_devices",
            "limitations": list(_LIMITATIONS),
        },
    )
    bucket_summary = ", ".join(
        f"{name}={values[f'{name}_count']} ({values[f'{name}_count_pct']:.1f}% of kernels, "
        f"{values[f'{name}_ms_pct']:.1f}% of time)"
        for name in _BUCKET_ORDER
    )
    if dominant == "mixed":
        tied = ", ".join(dominant_buckets)
        dominance_explanation = f"The {dominant_basis} signal is tied across {tied}."
        suggested_actions = [
            f"{dominant_basis.capitalize()} is tied across {tied}; investigate those call-mode "
            "buckets separately with overlap_breakdown before selecting a fix path."
        ]
    else:
        dominance_explanation = (
            f"The dominant bucket by exact {dominant_basis} is {dominant}."
        )
        suggested_actions = list(_ACTIONS_BY_BUCKET[dominant])
    return [
        Finding(
            type="region",
            label=f"NCCL Call Mode (dominant: {dominant})",
            start_ns=start_ns,
            end_ns=end_ns,
            severity="info",
            note=bucket_summary,
            id=finding_id,
            category="communication",
            evidence=[evidence_row],
            selection=selection,
            explanation=(
                "Leaf Push/Pop NVTX labels classify the observed NCCL calls across all "
                "captured devices as eager, inductor-captured, or temporally attributed "
                f"only. {dominance_explanation}"
            ),
            suggested_actions=suggested_actions,
            false_positive_notes=list(_LIMITATIONS),
            provenance={
                "skill": "nccl_compile_context_breakdown",
                "row_kind": "call_mode_classification",
                "classification_basis": "leaf_pushpop_nvtx",
                "dominant_basis": dominant_basis,
                "dominant_buckets": dominant_buckets,
                "device_scope": "all_captured_devices",
            },
            headroom_ms=None,
            headroom_basis=None,
        )
    ]


def _format(rows):
    if not rows or "error" in rows[0]:
        return f"(NCCL compile context: {rows[0].get('error', 'no data') if rows else 'no data'})"
    s = rows[0]
    lines = [
        "── NCCL Call-Mode Breakdown (by leaf NVTX) ──",
        f"  Total NCCL kernels: {s['total_nccl_kernels']:,}  ({s['total_nccl_ms']:.3f} ms)",
        "  Scope: all captured devices",
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
    to_findings_fn=_to_findings,
    tags=["nccl", "communication", "distributed", "torch-compile", "nvtx"],
)
