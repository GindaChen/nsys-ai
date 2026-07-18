"""Region-level MFU (Model FLOPs Utilization) computation.

Exposes the full region_mfu.py analytical pipeline as a Skill.
This is a Python-level skill (execute_fn) because it needs NVTX range
resolution, kernel attribution, and interval math.

Requires user-provided theoretical_flops (model FLOPs per step).
"""

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    from ...region_mfu import compute_region_mfu_from_conn

    name = kwargs.get("name", "")
    theoretical_flops = float(kwargs.get("theoretical_flops", 0))
    source = kwargs.get("source", "nvtx")
    peak_tflops = kwargs.get("peak_tflops")
    if peak_tflops is not None:
        peak_tflops = float(peak_tflops)
    num_gpus = int(kwargs.get("num_gpus", 1))
    occurrence_index = int(kwargs.get("occurrence_index", 1))
    device_id = kwargs.get("device_id")
    if device_id is not None:
        device_id = int(device_id)
    match_mode = kwargs.get("match_mode", "contains")

    result = compute_region_mfu_from_conn(
        conn,
        profile_path=None,
        name=name,
        theoretical_flops=theoretical_flops,
        source=source,
        peak_tflops=peak_tflops,
        num_gpus=num_gpus,
        occurrence_index=occurrence_index,
        device_id=device_id,
        match_mode=match_mode,
    )
    return [result]


def _sol_headroom_ms(r: dict) -> float | None:
    """Speed-of-light headroom in ms: time recoverable if the region ran at peak.

    Uses the kernel-union basis — real GPU-busy wall time in the region, no
    double-counting of concurrent kernels — paired with its matching MFU, so
    numerator and denominator are the same physical time:
    ``union_ms * (1 - mfu_pct_kernel_union / 100)``. Returns ``None`` when the
    inputs are missing or non-positive (e.g. no theoretical_flops supplied),
    so the region contributes no headroom rather than a bogus zero.
    """
    union_s = r.get("gpu_kernel_union_s")
    mfu = r.get("mfu_pct_kernel_union")
    if not union_s or union_s <= 0 or mfu is None or mfu <= 0:
        return None
    recoverable = union_s * 1000.0 * (1.0 - min(float(mfu), 100.0) / 100.0)
    return round(recoverable, 3) if recoverable > 0 else None


def _format(rows):
    if not rows:
        return "(No MFU data)"
    r = rows[0]
    if "error" in r:
        err = r["error"]
        return f"(MFU error: {err.get('code', '?')}: {err.get('message', '')})"
    lines = ["── Region MFU ──"]
    lines.append(f"  Region: {r.get('matched_text') or r.get('name', '?')}")
    lines.append(f"  Source: {r.get('source', '?')}")
    # Timings are stored in seconds on the result; render as ms.
    lines.append(f"  Wall time:     {r.get('wall_time_s', 0) * 1e3:.2f}ms")
    lines.append(f"  Kernel sum:    {r.get('gpu_kernel_sum_s', 0) * 1e3:.2f}ms")
    lines.append(f"  Kernel union:  {r.get('gpu_kernel_union_s', 0) * 1e3:.2f}ms")
    lines.append(f"  MFU (union):   {r.get('mfu_pct_kernel_union', 0):.1f}%")
    lines.append(f"  Achieved:      {r.get('achieved_tflops_kernel_union', 0):.1f} TFLOPS")
    lines.append(f"  Peak:          {r.get('peak_tflops', 0):.0f} TFLOPS")
    headroom = _sol_headroom_ms(r)
    if headroom is not None:
        lines.append(f"  SOL headroom:  {headroom:.2f}ms (recoverable at peak)")
    return "\n".join(lines)


def _to_findings(rows, *, context: dict | None = None) -> list:
    """Emit a speed-of-light finding carrying the region's MFU headroom.

    Only emitted when a headroom can be computed (a theoretical_flops was
    supplied and MFU is below 100%); otherwise no finding, so behaviour is
    unchanged for callers that don't request MFU. The finding is region-level:
    region_mfu does not expose per-instance timestamps, so the overlay is not
    time-anchored — its value is the ranked opportunity, not a timeline span.
    """
    from nsys_ai.annotation import EvidenceRow, Finding, TraceSelection

    if not rows or "error" in rows[0]:
        return []
    r = rows[0]
    headroom = _sol_headroom_ms(r)
    if headroom is None:
        return []

    mfu = float(r.get("mfu_pct_kernel_union", 0.0))
    region = r.get("matched_text") or r.get("name") or "region"
    device_ids = r.get("device_ids") or ([r["device_id"]] if r.get("device_id") is not None else [])
    device = int(device_ids[0]) if device_ids else 0
    slug = "".join(c if c.isalnum() else "_" for c in str(region))[:48]
    finding_id = f"region_mfu_sol_{slug}"

    selection = TraceSelection(
        id=f"sel_{finding_id}",
        profile_id=(context or {}).get("profile_id", "unknown"),
        source="skill:region_mfu",
        gpu_ids=device_ids or None,
        label=f"{region} MFU {mfu:.0f}%",
    )
    evidence_row = EvidenceRow(
        id=f"ev_{finding_id}",
        source_skill="region_mfu",
        values={
            "mfu_pct_kernel_union": round(mfu, 2),
            "kernel_union_ms": round(float(r.get("gpu_kernel_union_s", 0.0)) * 1e3, 3),
            "achieved_tflops": round(float(r.get("achieved_tflops_kernel_union", 0.0)), 2),
            "peak_tflops": round(float(r.get("peak_tflops", 0.0)), 1),
            "sol_headroom_ms": headroom,
        },
        units={
            "mfu_pct_kernel_union": "percent",
            "kernel_union_ms": "ms",
            "achieved_tflops": "tflops",
            "peak_tflops": "tflops",
            "sol_headroom_ms": "ms",
        },
        selection_id=selection.id,
        provenance={"row_kind": "region_sol", "region": region},
    )

    return [
        Finding(
            type="region",
            label=f"{region} at {mfu:.0f}% MFU ({headroom:.0f}ms below peak)",
            start_ns=0,
            end_ns=0,
            gpu_id=device,
            severity="warning" if mfu < 40 else "info",
            note=(
                f"Region '{region}' runs at {mfu:.1f}% MFU; up to {headroom:.1f}ms is "
                "recoverable if it reached the hardware speed-of-light."
            ),
            id=finding_id,
            category="compute",
            confidence=None,
            headroom_ms=headroom,
            evidence=[evidence_row],
            selection=selection,
            provenance={"skill": "region_mfu", "row_kind": "region_sol"},
        )
    ]


SKILL = Skill(
    name="region_mfu",
    title="Region-Level MFU (Model FLOPs Utilization)",
    description=(
        "Computes MFU for an NVTX region or kernel. Finds the named region, "
        "attributes GPU kernels to it, and calculates MFU = achieved / peak TFLOPS. "
        "Requires theoretical_flops (model FLOPs per step) from the user."
    ),
    category="kernels",
    execute_fn=_execute,
    format_fn=_format,
    to_findings_fn=_to_findings,
    params=[
        SkillParam("name", "NVTX region or kernel name to analyze", "str", True, None),
        SkillParam(
            "theoretical_flops",
            "Model FLOPs per step (must be provided by user)",
            "float",
            True,
            None,
        ),
        SkillParam("source", "Match source: 'nvtx' or 'kernel'", "str", False, "nvtx"),
        SkillParam(
            "peak_tflops", "GPU peak TFLOPS (auto-detected if omitted)", "float", False, None
        ),
        SkillParam("num_gpus", "Number of GPUs (for DP/TP adjustment)", "int", False, 1),
        SkillParam("occurrence_index", "Which occurrence to analyze (1-based)", "int", False, 1),
        SkillParam("device_id", "GPU device ID filter", "int", False, None),
        SkillParam(
            "match_mode",
            "Name matching: 'contains', 'exact', 'startswith'",
            "str",
            False,
            "contains",
        ),
    ],
    tags=["mfu", "flops", "utilization", "efficiency", "region", "nvtx", "kernel"],
)
