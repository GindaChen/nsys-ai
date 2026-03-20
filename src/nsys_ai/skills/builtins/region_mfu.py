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


def _format(rows):
    if not rows:
        return "(No MFU data)"
    r = rows[0]
    if "error" in r:
        err = r["error"]
        return f"(MFU error: {err.get('code', '?')}: {err.get('message', '')})"
    lines = ["── Region MFU ──"]
    lines.append(f"  Region: {r.get('matched_name', '?')}")
    lines.append(f"  Source: {r.get('source', '?')}")
    timing = r.get("timing", {})
    if timing:
        lines.append(f"  Wall time:     {timing.get('wall_time_ms', 0):.2f}ms")
        lines.append(f"  Kernel sum:    {timing.get('kernel_sum_ms', 0):.2f}ms")
        lines.append(f"  Kernel union:  {timing.get('kernel_union_ms', 0):.2f}ms")
    mfu = r.get("mfu", {})
    if mfu:
        lines.append(f"  MFU: {mfu.get('mfu_pct', 0):.1f}%")
        lines.append(f"  Achieved: {mfu.get('achieved_tflops', 0):.1f} TFLOPS")
        lines.append(f"  Peak: {mfu.get('peak_tflops', 0):.0f} TFLOPS")
    return "\n".join(lines)


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
