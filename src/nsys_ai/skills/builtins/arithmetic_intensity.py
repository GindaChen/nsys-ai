"""Arithmetic intensity vs. GPU peak assessment (Roofline Model).

Combines GPU hardware specs with kernel execution time and user-provided
theoretical FLOPs to classify workloads as compute-bound or memory-bound.

Since Nsight Systems .sqlite does NOT contain per-kernel FLOPs or bytes-moved
(only NCU has that), this skill performs an **aggregate roofline estimation**.
"""

from nsys_ai.hardware import get_peak_tflops

from ..base import Skill, SkillParam, _resolve_activity_tables


def _execute(conn, **kwargs):
    theoretical_flops = float(kwargs["theoretical_flops"])
    device = int(kwargs.get("device", 0))

    tables = _resolve_activity_tables(conn)
    kernel_table = tables.get("kernel", "CUPTI_ACTIVITY_KIND_KERNEL")

    # --- Get GPU hardware spec ---
    gpu_name = "Unknown GPU"
    chip_name = ""
    hbm_bw_raw = 0

    try:
        row = conn.execute(
            "SELECT name, chipName, memoryBandwidth FROM TARGET_INFO_GPU WHERE id = ?",
            (device,),
        ).fetchone()
        if row:
            gpu_name = row[0] or "Unknown GPU"
            chip_name = row[1] or ""
            hbm_bw_raw = row[2] or 0
    except Exception:
        pass

    # Lookup from centralized hardware table, fallback to DB value
    spec1 = get_peak_tflops(chip_name)
    spec2 = get_peak_tflops(gpu_name)

    if "error" not in spec1:
        peak_tflops = spec1.get("peak_tflops", 312.0)
        hbm_bw_gbps = spec1.get("hbm_bw_gbps", 2039.0)
    elif "error" not in spec2:
        peak_tflops = spec2.get("peak_tflops", 312.0)
        hbm_bw_gbps = spec2.get("hbm_bw_gbps", 2039.0)
    else:
        # Fallback: use memoryBandwidth from DB (bytes/s → GB/s)
        peak_tflops = float(kwargs.get("peak_tflops", 312.0))  # A100 default
        hbm_bw_gbps = hbm_bw_raw / 1e9 if hbm_bw_raw > 0 else 2039.0

    # Allow user override
    if kwargs.get("peak_tflops") is not None:
        peak_tflops = float(kwargs["peak_tflops"])
    if kwargs.get("hbm_bw_gbps") is not None:
        hbm_bw_gbps = float(kwargs["hbm_bw_gbps"])

    bytes_moved = kwargs.get("bytes_moved")
    if bytes_moved is not None:
        bytes_moved = float(bytes_moved)

    # --- Compute total kernel time on device ---
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    params = [device]
    trim_clause = ""
    if trim_start is not None and trim_end is not None:
        trim_clause = 'AND "end" > ? AND start < ?'
        params.extend([trim_start, trim_end])

    try:
        cursor = conn.execute(
            f'SELECT start, "end" FROM {kernel_table} '
            f"WHERE deviceId = ? {trim_clause} ORDER BY start",
            params,
        )

        total_kernel_ns = 0
        kernel_count = 0
        cur_s, cur_e = None, None

        for row in cursor:
            kernel_count += 1
            s, e = row[0], row[1]
            if cur_s is None:
                cur_s, cur_e = s, e
            elif s <= cur_e:
                if e > cur_e:
                    cur_e = e
            else:
                total_kernel_ns += cur_e - cur_s
                cur_s, cur_e = s, e

        if cur_s is not None:
            total_kernel_ns += cur_e - cur_s

    except Exception:
        total_kernel_ns = 0
        kernel_count = 0

    if total_kernel_ns == 0 or kernel_count == 0:
        return [
            {
                "error": "No kernel data found on the specified device.",
                "gpu_name": gpu_name,
            }
        ]

    total_kernel_s = total_kernel_ns / 1e9
    total_kernel_ms = total_kernel_ns / 1e6

    # --- Roofline calculations ---
    achieved_tflops = theoretical_flops / total_kernel_s / 1e12
    mfu_pct = (achieved_tflops / peak_tflops) * 100.0 if peak_tflops > 0 else 0.0
    ridge_point = (peak_tflops * 1e12) / (hbm_bw_gbps * 1e9) if hbm_bw_gbps > 0 else 0.0

    op_intensity = None
    if bytes_moved is not None and bytes_moved > 0:
        op_intensity = theoretical_flops / bytes_moved

    # Classification
    if op_intensity is not None and ridge_point > 0:
        if op_intensity < ridge_point:
            classification = f"Memory-bound (AI={op_intensity:.1f} < Ridge={ridge_point:.1f})"
            severity = "warning"
            recommendation = (
                "Workload is mathematically memory-bound (Arithmetic Intensity < Ridge Point). "
                "Increase batch size, use operator fusion, or verify memory access patterns."
            )
        else:
            classification = f"Compute-bound (AI={op_intensity:.1f} >= Ridge={ridge_point:.1f})"
            severity = "info"
            recommendation = (
                "Workload is mathematically compute-bound. "
                "Optimize kernel occupancy, warp efficiency, and Tensor Core usage."
            )
    else:
        # Fallback heuristic based solely on MFU
        if mfu_pct >= 50:
            classification = "High utilization (likely compute-bound)"
            severity = "info"
            recommendation = (
                "Workload has good utilization. "
                "For further gains, consider kernel-level optimization with NCU "
                "(occupancy, warp efficiency, instruction mix)."
            )
        elif mfu_pct >= 15:
            classification = "Moderate utilization (mixed bound)"
            severity = "warning"
            recommendation = (
                "Workload is in a transition zone. "
                "Consider increasing batch size to raise arithmetic intensity, "
                "using FlashAttention for attention kernels, or fusing small ops with torch.compile()."
            )
        elif mfu_pct >= 5:
            classification = "Low utilization (likely memory-bound)"
            severity = "warning"
            recommendation = (
                "Kernels are likely bottlenecked by HBM bandwidth rather than compute. "
                "Increase batch size, use operator fusion (torch.compile), "
                "enable FlashAttention, or check for excessive memory-bound element-wise ops."
            )
        else:
            classification = "Severely under-utilized"
            severity = "critical"
            recommendation = (
                "GPU is severely under-utilized. Common causes: excessive CPU overhead, "
                "pipeline bubbles, small batch sizes, or profiling during warmup. "
                "Run gpu_idle_gaps and root_cause_matcher to diagnose."
            )

    return [
        {
            "gpu_name": gpu_name,
            "chip_name": chip_name,
            "peak_fp16_tflops": round(peak_tflops, 1),
            "hbm_bw_gbps": round(hbm_bw_gbps, 1),
            "ridge_point_flop_per_byte": round(ridge_point, 1),
            "total_kernel_ms": round(total_kernel_ms, 2),
            "kernel_count": kernel_count,
            "theoretical_flops": theoretical_flops,
            "achieved_tflops": round(achieved_tflops, 1),
            "mfu_pct": round(mfu_pct, 1),
            "classification": classification,
            "severity": severity,
            "recommendation": recommendation,
        }
    ]


def _format(rows):
    if not rows:
        return "(No data for arithmetic intensity assessment)"
    r = rows[0]
    if "error" in r:
        return f"(Error: {r['error']})"

    lines = [
        "── Arithmetic Intensity Assessment (Roofline) ──",
        f"  GPU:              {r['gpu_name']}",
        f"  Peak FP16:        {r['peak_fp16_tflops']} TFLOPS",
        f"  HBM Bandwidth:    {r['hbm_bw_gbps']} GB/s",
        f"  Ridge Point:      {r['ridge_point_flop_per_byte']} FLOP/Byte",
        "",
        f"  Total kernel time:  {r['total_kernel_ms']:.2f} ms  ({r['kernel_count']} kernels)",
        f"  Achieved TFLOPS:    {r['achieved_tflops']}",
        f"  MFU:                {r['mfu_pct']:.1f}%",
        "",
        f"  Classification:     {r['classification']}",
        f"  Recommendation:     {r['recommendation']}",
    ]
    return "\n".join(lines)


SKILL = Skill(
    name="arithmetic_intensity",
    title="Arithmetic Intensity vs. GPU Peak (Roofline)",
    description=(
        "Performs an aggregate roofline assessment by combining GPU hardware specs "
        "(peak TFLOPS, HBM bandwidth) with total kernel execution time and "
        "user-provided theoretical FLOPs. Classifies the workload as compute-bound "
        "or memory-bound and reports MFU (Model FLOPs Utilization). "
        "Requires theoretical_flops from the user or from the theoretical_flops skill."
    ),
    category="kernels",
    execute_fn=_execute,
    format_fn=_format,
    params=[
        SkillParam(
            "theoretical_flops",
            "Total FLOPs for the profiled workload (use theoretical_flops skill to compute)",
            "float",
            True,
            None,
        ),
        SkillParam(
            "bytes_moved",
            "Total bytes moved to/from HBM. If provided, computes true arithmetic intensity.",
            "float",
            False,
            None,
        ),
        SkillParam("device", "GPU device ID", "int", False, 0),
        SkillParam(
            "peak_tflops",
            "Override GPU peak FP16 TFLOPS (auto-detected from chipName if omitted)",
            "float",
            False,
            None,
        ),
        SkillParam(
            "hbm_bw_gbps",
            "Override HBM bandwidth in GB/s (auto-detected if omitted)",
            "float",
            False,
            None,
        ),
    ],
    tags=[
        "roofline",
        "arithmetic_intensity",
        "mfu",
        "compute_bound",
        "memory_bound",
        "utilization",
    ],
)
