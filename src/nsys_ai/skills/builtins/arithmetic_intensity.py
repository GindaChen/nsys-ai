"""Arithmetic intensity vs. GPU peak assessment (Roofline Model).

Combines GPU hardware specs with kernel execution time and user-provided
theoretical FLOPs to classify workloads as compute-bound or memory-bound.

Since Nsight Systems .sqlite does NOT contain per-kernel FLOPs or bytes-moved
(only NCU has that), this skill performs an **aggregate roofline estimation**.
"""

from ..base import Skill, SkillParam, _resolve_activity_tables

# ---------------------------------------------------------------------------
# GPU spec lookup table: chipName → (peak FP16 TFLOPS, HBM BW GB/s)
# Sources: NVIDIA datasheets, Wikipedia, techpowerup.com
# ---------------------------------------------------------------------------
_GPU_SPECS: dict[str, tuple[float, float]] = {
    # ── Hopper ────────────────────────────────────────────────────
    # H100 SXM: FP16/BF16 TC dense 989 TFLOPS, HBM3 3350 GB/s
    # Note: 1979 TFLOPS is the SPARSE (2:4) figure — MFU uses dense.
    "GH100": (989.0, 3350),
    "H100": (989.0, 3350),
    "H100_SXM": (989.0, 3350),
    # H100 PCIe: FP16/BF16 TC dense 756 TFLOPS, HBM2e 2000 GB/s
    "H100_PCIE": (756.0, 2000),
    "H100_NVL": (835.0, 2000),
    # H200: same GH100 die → same 989 TFLOPS dense, HBM3e 4800 GB/s
    "H200": (989.0, 4800),
    # ── Ampere (data center) ──────────────────────────────────────
    # A100 SXM: FP16 TC dense 312 TFLOPS, HBM2e 2039 GB/s (80GB)
    "GA100": (312.0, 2039),
    "A100": (312.0, 2039),
    "A100_SXM": (312.0, 2039),
    "A100_PCIE": (312.0, 1555),
    "A100_80GB": (312.0, 2039),
    # A10: FP16 TC 125 TFLOPS, GDDR6 600 GB/s
    "GA102": (125.0, 600),
    "A10": (125.0, 600),
    # A30: FP16 TC 165 TFLOPS, HBM2e 933 GB/s
    "A30": (165.0, 933),
    # A40: FP16 TC 149.7 TFLOPS, GDDR6 696 GB/s
    "A40": (149.7, 696),
    # ── Volta ─────────────────────────────────────────────────────
    # V100 SXM2: FP16 TC 125 TFLOPS, HBM2 900 GB/s
    "GV100": (125.0, 900),
    "V100": (125.0, 900),
    "V100_SXM2": (125.0, 900),
    "V100_PCIE": (112.0, 900),
    # ── Ada Lovelace ──────────────────────────────────────────────
    # L40S: FP16 TC dense 362 TFLOPS, GDDR6 864 GB/s
    "AD102": (362.0, 864),
    "L40S": (362.0, 864),
    # L40: FP16 TC dense 181 TFLOPS, GDDR6 864 GB/s
    "L40": (181.0, 864),
    # L4: FP16 TC dense 121 TFLOPS, GDDR6 300 GB/s
    "L4": (121.0, 300),
    # RTX 4090: FP16 TC dense 165.2 TFLOPS, GDDR6X 1008 GB/s
    "RTX4090": (165.2, 1008),
    # RTX 3090: FP16 TC dense 142 TFLOPS, GDDR6X 936 GB/s
    "RTX3090": (142.0, 936),
    # ── Blackwell ─────────────────────────────────────────────────
    # B100: FP16/BF16 TC dense 1750 TFLOPS, HBM3e 8000 GB/s
    "GB100": (1750.0, 8000),
    "B100": (1750.0, 8000),
    # B200: FP16/BF16 TC dense 2250 TFLOPS, HBM3e 8000 GB/s
    "B200": (2250.0, 8000),
}


def _lookup_gpu_spec(chip_name: str) -> tuple[float, float] | None:
    """Match chipName against the spec table using prefix/substring matching.

    Returns (peak_fp16_tflops, hbm_bw_gbps) or None.
    """
    if not chip_name:
        return None

    # Exact match first
    upper = chip_name.upper().replace(" ", "_").replace("-", "_")
    for key, val in _GPU_SPECS.items():
        if key.upper() == upper:
            return val

    # Substring match (e.g. "NVIDIA H100 80GB HBM3" → matches "H100")
    for key, val in sorted(_GPU_SPECS.items(), key=lambda kv: -len(kv[0])):
        if key.upper() in upper:
            return val

    return None


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

    # Lookup from table, fallback to DB value
    spec = _lookup_gpu_spec(chip_name) or _lookup_gpu_spec(gpu_name)
    if spec:
        peak_tflops, hbm_bw_gbps = spec
    else:
        # Fallback: use memoryBandwidth from DB (bytes/s → GB/s)
        peak_tflops = float(kwargs.get("peak_tflops", 312.0))  # A100 default
        hbm_bw_gbps = hbm_bw_raw / 1e9 if hbm_bw_raw > 0 else 2039.0

    # Allow user override
    if kwargs.get("peak_tflops") is not None:
        peak_tflops = float(kwargs["peak_tflops"])
    if kwargs.get("hbm_bw_gbps") is not None:
        hbm_bw_gbps = float(kwargs["hbm_bw_gbps"])

    # --- Compute total kernel time on device ---
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    params = [device]
    trim_clause = ""
    if trim_start is not None and trim_end is not None:
        trim_clause = 'AND "end" > ? AND start < ?'
        params.extend([trim_start, trim_end])

    try:
        rows = conn.execute(
            f'SELECT start, "end" FROM {kernel_table} '
            f"WHERE deviceId = ? {trim_clause}",
            params,
        ).fetchall()

        from nsys_ai.region_mfu import _merge_intervals
        total_kernel_ns = _merge_intervals([(row[0], row[1]) for row in rows])
        kernel_count = len(rows)
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

    # Classification
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
