"""CUTracer instruction-level drill-down skill.

Parses CUTracer ``proton_instr_histogram`` output, correlates kernel names
with NVTX attribution from the nsys profile, and produces a combined
instruction-level + system-level analysis.

Required parameter: ``trace_dir`` — path to the CUTracer output directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..base import Skill, SkillParam

_log = logging.getLogger(__name__)


def _execute(conn, **kwargs) -> list[dict]:
    trace_dir_raw = kwargs.get("trace_dir", "")
    if not trace_dir_raw:
        return [{"error": "trace_dir parameter is required"}]

    trace_dir = Path(str(trace_dir_raw))
    if not trace_dir.exists():
        return [{"error": f"trace_dir not found: {trace_dir}"}]

    from nsys_ai.cutracer.correlator import build_nsys_kernel_list, match_kernels
    from nsys_ai.cutracer.parser import parse_histogram_dir
    from nsys_ai.cutracer.report import KernelReport, compute_mix, to_dict

    # 1. Parse CUTracer traces
    histograms = parse_histogram_dir(trace_dir)
    if not histograms:
        return [{"error": f"No *_hist.csv files found in {trace_dir}"}]

    # 2. Fetch nsys kernel names for correlation
    nsys_kernels = build_nsys_kernel_list(conn)

    # 3. Match CUTracer kernel names → nsys kernel names
    cutracer_names = list(histograms.keys())
    # match_kernels takes nsys→cutracer; we also want the reverse
    nsys_to_ct = match_kernels(nsys_kernels, cutracer_names)
    ct_to_nsys: dict[str, str | None] = {}
    for nsys_k, ct_k in nsys_to_ct.items():
        if ct_k and ct_k not in ct_to_nsys:
            ct_to_nsys[ct_k] = nsys_k

    # 4. Fetch NVTX attribution if available
    nvtx_map: dict[str, dict] = {}
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    has_trim = trim_start is not None and trim_end is not None
    try:
        from nsys_ai.nvtx_attribution import attribute_kernels_to_nvtx

        attr_rows = attribute_kernels_to_nvtx(
            conn,
            trim=(trim_start, trim_end) if has_trim else None,
        )
        for row in attr_rows:
            k = row.get("kernel_name", "")
            if k and k not in nvtx_map:
                nvtx_map[k] = row
    except Exception as exc:
        _log.debug("Failed to fetch NVTX attribution for cutracer_analysis: %s", exc)

    # 5. Fetch per-kernel GPU time from nsys
    gpu_time_map: dict[str, float] = {}
    try:
        from nsys_ai.connection import wrap_connection

        adapter = wrap_connection(conn)
        tables = adapter.get_table_names()
        if "kernels" in tables:
            trim_clause = ""
            params = []
            if has_trim:
                trim_clause = 'WHERE start >= ? AND "end" <= ?'
                params = [trim_start, trim_end]
            rows = adapter.execute(
                f"""
                SELECT name, ROUND(SUM("end" - start) / 1e6, 3) AS total_ms
                FROM kernels {trim_clause}
                GROUP BY name
                """,
                params,
            ).fetchall()
            total_gpu = sum(r[1] for r in rows) or 1.0
            for kernel_name, total_ms in rows:
                gpu_time_map[kernel_name] = total_ms
    except Exception as exc:
        _log.debug("Failed to fetch per-kernel GPU time for cutracer_analysis: %s", exc)
        total_gpu = 1.0

    # 6. Build reports
    reports: list[dict] = []
    for ct_name, hist in histograms.items():
        mix = compute_mix(hist)
        nsys_name = ct_to_nsys.get(ct_name)
        nvtx_row = nvtx_map.get(nsys_name or ct_name, {})
        gpu_ms = gpu_time_map.get(nsys_name or ct_name)
        pct = round(gpu_ms / (total_gpu or 1) * 100, 1) if gpu_ms else None

        kr = KernelReport(
            mix=mix,
            nsys_kernel_name=nsys_name,
            nvtx_path=nvtx_row.get("nvtx_path"),
            total_ms=gpu_ms,
            pct_of_gpu=pct,
            achieved_warps=hist.warp_count if hist.warp_count > 0 else None,
        )
        reports.append(to_dict(kr))

    # Sort by GPU time descending (kernels with nsys data first)
    reports.sort(key=lambda r: -(r.get("total_ms") or 0))
    return reports


def _format(rows: list[dict]) -> str:
    if not rows:
        return "(No CUTracer data)"

    if rows and "error" in rows[0]:
        return f"Error: {rows[0]['error']}"

    from nsys_ai.cutracer.report import InstrMix, KernelReport, format_kernel_report

    lines = []
    for r in rows:
        mix = InstrMix(
            kernel_name=r.get("cutracer_kernel_name", r.get("kernel_name", "")),
            total_count=r.get("total_instructions", 0),
            category_pct=r.get("instruction_mix_pct", {}),
            top_stalls=[(s["opcode"], s["stall_score"]) for s in r.get("top_stalls", [])],
            tc_active=r.get("tensor_core_active", False),
            cycles_per_instr={
                s["opcode"]: s["cycles_per_instr"]
                for s in r.get("top_stalls", [])
                if s.get("cycles_per_instr")
            },
            bottleneck=r.get("bottleneck", "unknown"),
            bank_conflict_hint=r.get("bank_conflict_hint", False),
        )
        ct_name = r.get("cutracer_kernel_name", "")
        nsys_name = r.get("kernel_name", "")
        kr = KernelReport(
            mix=mix,
            nsys_kernel_name=nsys_name if nsys_name != ct_name else None,
            nvtx_path=r.get("nvtx_path"),
            total_ms=r.get("total_ms"),
            pct_of_gpu=r.get("pct_of_gpu"),
            achieved_warps=r.get("achieved_warps"),
        )
        lines.append(format_kernel_report(kr))

    # Summary footer
    total = len(rows)
    matched = sum(1 for r in rows if r.get("kernel_name") != r.get("cutracer_kernel_name"))
    unmatched = total - matched
    footer_parts = [f"{total} kernel(s) analyzed"]
    if matched:
        footer_parts.append(f"{matched} matched in nsys profile")
    if unmatched:
        footer_parts.append(f"{unmatched} not found in profile (instruction mix only)")
    lines.append("Summary: " + ", ".join(footer_parts) + ".")

    return "\n\n".join(lines)


SKILL = Skill(
    name="cutracer_analysis",
    title="CUTracer Instruction-Level Drill-Down",
    description=(
        "Parses CUTracer proton_instr_histogram traces and correlates them with "
        "NVTX attribution from the nsys profile. Reports SASS instruction mix, "
        "stall scores, warp occupancy, and bank conflict hints. "
        "Classifies each kernel as COMPUTE-BOUND, MEMORY-BOUND, or SYNC-BOUND. "
        "HIGH VALUE for: custom GEMMs (nvjet_*, cutlass_*), unknown kernels, "
        "TC-eligible kernels that may be falling back to FP32. "
        "LOW VALUE for: NCCL collectives, Flash Attention, elementwise ops "
        "(these are bandwidth-bound by construction — prefer nccl_breakdown or "
        "tensor_core_usage for those). "
        "Requires trace_dir from a prior `nsys-ai cutracer plan` + run."
    ),
    category="kernels",
    execute_fn=_execute,
    params=[
        SkillParam("trace_dir", "Path to CUTracer output directory (*_hist.csv or ndjson+cubin)", "str", True),
    ],
    format_fn=_format,
    tags=["cutracer", "instruction", "memory-bound", "compute-bound", "stall", "sass",
          "bank-conflict", "warp-occupancy", "tensor-core"],
)
