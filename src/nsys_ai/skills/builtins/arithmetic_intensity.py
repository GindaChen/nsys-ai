"""Arithmetic intensity vs. GPU peak assessment (Roofline Model).

Combines GPU hardware specs with kernel execution time and user-provided
theoretical FLOPs to classify workloads as compute-bound or memory-bound.

Since Nsight Systems .sqlite does NOT contain per-kernel FLOPs or bytes-moved
(only NCU has that), this skill performs an **aggregate roofline estimation**.
"""

import logging
import sqlite3

try:
    import duckdb

    _DB_ERRORS = (sqlite3.Error, duckdb.Error)
except ImportError:
    _DB_ERRORS = (sqlite3.Error,)

from nsys_ai.connection import wrap_connection
from nsys_ai.hardware import get_peak_tflops

from ..base import Skill, SkillParam, abstain

logger = logging.getLogger(__name__)


def _execute(conn, **kwargs):
    theoretical_flops = float(kwargs["theoretical_flops"])
    device = int(kwargs.get("device", 0))

    tables = wrap_connection(conn).resolve_activity_tables()
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
    except _DB_ERRORS as e:
        logger.debug(f"Failed to fetch GPU info from TARGET_INFO_GPU: {e}")

    # Lookup from centralized hardware table, fallback to DB value
    spec1 = get_peak_tflops(chip_name)
    spec2 = get_peak_tflops(gpu_name)

    peak_tflops = kwargs.get("peak_tflops")
    hbm_bw_gbps = kwargs.get("hbm_bw_gbps")

    # Where the peak came from, captured before the spec-table fallback
    # overwrites the local. The two refusals that name a peak quote it, because
    # "989 TFLOPS" is a different claim depending on whether the caller asserted
    # it or this package looked it up, and the reader can only act on the one
    # they own.
    peak_origin = (
        "which you provided"
        if peak_tflops is not None
        else "from this package's hardware spec table"
    )

    if peak_tflops is None or hbm_bw_gbps is None:
        if "error" not in spec1:
            peak_tflops = peak_tflops if peak_tflops is not None else spec1.get("peak_tflops")
            hbm_bw_gbps = hbm_bw_gbps if hbm_bw_gbps is not None else spec1.get("hbm_bw_gbps")
        elif "error" not in spec2:
            peak_tflops = peak_tflops if peak_tflops is not None else spec2.get("peak_tflops")
            hbm_bw_gbps = hbm_bw_gbps if hbm_bw_gbps is not None else spec2.get("hbm_bw_gbps")

    # If hbm_bw_gbps is still missing, attempt DB fallback
    if hbm_bw_gbps is None and hbm_bw_raw > 0:
        hbm_bw_gbps = hbm_bw_raw / 1e9

    # No peak, no roofline: this is the "cannot run" case, so it abstains rather
    # than returning a row with an ``error`` key. Consumers distinguish the two —
    # see :func:`nsys_ai.skills.base.abstain` — and an error row is data, so
    # ``format_rows`` would hand it to ``_format`` and ``to_findings_fn`` would be
    # entitled to mint a finding from it.
    #
    # This is also the whole of the "should an unknown GPU model block
    # classification?" question, and the answer is: it already does, here, when
    # the model is the only source of the peak. ``doctor`` refuses on this same
    # profile with "GPU model missing from CUPTI TARGET_INFO; MFU / efficiency
    # cannot be computed"; this branch covers that state and one more, since it
    # also fires when the model is reported but absent from the spec table.
    # What it must not do is refuse when the caller has supplied ``peak_tflops``
    # themselves: the number ``doctor`` is missing is then present, from the one
    # source that outranks a lookup table, and refusing anyway would make the
    # documented override useless on precisely the profiles that need it — an
    # unlisted GPU, or one whose export carries no chipName. So the header keeps
    # saying "Unknown GPU", the arithmetic proceeds on the caller's peak, and the
    # guards below check the *result* for consistency rather than the label.
    if peak_tflops is None:
        return abstain(
            f"This profile's GPU reports as {gpu_name}"
            f"{f' ({chip_name})' if chip_name else ' with no chipName'}, which is not in this "
            f"package's hardware spec table, so its peak throughput is unknown and MFU has no "
            f"denominator. Re-run with 'peak_tflops' set to the peak for the GPU this profile "
            f"was captured on (and optionally 'hbm_bw_gbps' for the ridge point).",
            gpu_name=gpu_name,
            chip_name=chip_name,
        )

    if hbm_bw_gbps is None:
        # Cannot determine HBM bandwidth — skip roofline classification,
        # fall through to the MFU-only heuristic (ridge_point will be 0).
        logger.warning(
            "HBM bandwidth not detected for %s (%s); "
            "roofline classification unavailable. "
            "Provide 'hbm_bw_gbps' explicitly for full analysis.",
            gpu_name,
            chip_name,
        )
        hbm_bw_gbps = 0.0

    peak_tflops = float(peak_tflops)
    hbm_bw_gbps = float(hbm_bw_gbps)

    # A non-positive peak is not a slow GPU, it is a broken denominator: every
    # ratio below divides by it. This must stay ahead of the achieved-above-peak
    # check, which divides by ``peak_tflops`` inside its own message and would
    # raise ZeroDivisionError instead of explaining itself.
    if peak_tflops <= 0:
        return abstain(
            f"peak_tflops is {peak_tflops:g}, {peak_origin}. MFU is a fraction of peak, so a "
            f"peak of zero or less gives it no denominator and any classification would be an "
            f"artefact of the input. Re-run with the peak throughput of the GPU this profile "
            f"was captured on, in TFLOPS.",
            gpu_name=gpu_name,
            peak_fp16_tflops=peak_tflops,
        )

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

        # O(1) streaming interval union: the query returns rows ORDER BY start,
        # so we can compute the merged union in a single pass without materialising
        # the full interval list.  This is deliberately inlined rather than using
        # nsys_ai.overlap.merge_intervals (which requires O(N) memory).
        total_kernel_ns = 0
        kernel_count = 0
        current_start = -1
        current_end = -1

        while True:
            row = cursor.fetchone()
            if row is None:
                break

            s, e = row[0], row[1]
            if trim_start is not None:
                s = max(s, trim_start)
            if trim_end is not None:
                e = min(e, trim_end)
            if s >= e:
                continue

            kernel_count += 1
            if current_start == -1:
                current_start = s
                current_end = e
            elif s <= current_end:
                current_end = max(current_end, e)
            else:
                total_kernel_ns += current_end - current_start
                current_start = s
                current_end = e

        if current_start != -1:
            total_kernel_ns += current_end - current_start

    except _DB_ERRORS as e:
        logger.debug(f"Failed to fetch kernel intervals: {e}")
        total_kernel_ns = 0
        kernel_count = 0

    if total_kernel_ns == 0 or kernel_count == 0:
        # Same contract as the peak branch above: no kernel time is "cannot run",
        # not a result. It is also the divisor two lines down, so continuing here
        # is a ZeroDivisionError, never a number.
        window = (
            f" within the requested window ({trim_start}-{trim_end} ns)"
            if trim_start is not None and trim_end is not None
            else ""
        )
        return abstain(
            f"{kernel_table} records no kernel activity on device {device}{window}, so there is "
            f"no kernel time to divide the FLOPs by and no throughput to compare against peak. "
            f"Check the device ID, or the trim range if one was given.",
            gpu_name=gpu_name,
            device=device,
        )

    # And the numerator. theoretical_flops is the caller-supplied total FLOPs of
    # the profiled work; zero or negative is not a slow workload, it is a number
    # that cannot describe work that ran. Left to run it reported
    # "Severely low kernel throughput" at critical severity for 0, and a negative
    # achieved throughput for a negative input — the same confident verdict drawn
    # from a mistyped number as the peak guard above, one sign away.
    if theoretical_flops <= 0:
        return abstain(
            f"theoretical_flops is {theoretical_flops:g}. It is read as the total FLOPs of the "
            f"profiled work, so zero or less cannot describe work that ran, and every figure "
            f"below it — achieved throughput, MFU, arithmetic intensity — would be an artefact "
            f"of the input rather than a measurement. Re-run with the FLOP count for the work "
            f"this profile captured.",
            gpu_name=gpu_name,
            theoretical_flops=theoretical_flops,
        )

    total_kernel_s = total_kernel_ns / 1e9
    total_kernel_ms = total_kernel_ns / 1e6

    # --- Roofline calculations ---
    achieved_tflops = theoretical_flops / total_kernel_s / 1e12

    # Achieved above peak is not a borderline reading, it is arithmetically
    # impossible: the hardware cannot issue more FLOPs per second than its peak,
    # so the only thing an MFU over 100% can mean is that the inputs do not
    # support this calculation. Left to run, it took the number at face value and
    # classified 369% as "High kernel throughput (likely compute-bound)", then
    # sent the reader to NCU to tune a kernel on the strength of it. Both
    # classification arms below are poisoned by it, not just the MFU heuristic:
    # ``op_intensity`` divides the same ``theoretical_flops``, so the roofline
    # verdict inherits whatever is wrong with it. Hence the check sits ahead of
    # both.
    #
    # No tolerance band. A GPU running 1% over its own peak is as impossible as
    # one running 300% over, and a band would only decide how much nonsense to
    # publish.
    if achieved_tflops > peak_tflops:
        return abstain(
            f"Refusing to report MFU. This calculation puts achieved throughput at "
            f"{achieved_tflops:,.1f} TFLOPS, {achieved_tflops / peak_tflops:.1f}x the peak it is "
            f"measured against ({peak_tflops:,.1f} TFLOPS, {peak_origin}) — an MFU of "
            f"{(achieved_tflops / peak_tflops) * 100.0:,.1f}%. Hardware cannot exceed its own "
            f"peak, so this is not a fast workload; one of the two inputs is not what the "
            f"calculation assumes. theoretical_flops={theoretical_flops:.4g} is read as the "
            f"total FLOPs of the profiled work, divided here by {total_kernel_ms:,.2f} ms of "
            f"kernel time on device {device} ({kernel_count} kernels): passing a per-second "
            f"rate, or a whole job's FLOPs against a profile that captured one slice of it, "
            f"lands exactly here. Check theoretical_flops first, then peak_tflops — a peak "
            f"quoted for a precision or a sparsity mode the kernels did not use is the other "
            f"way in.",
            gpu_name=gpu_name,
            # Prefixed, deliberately. These are the repudiated numbers, kept
            # because they are what the caller needs to see to find their input
            # error — but a consumer reading rows[0]["mfu_pct"] must not receive
            # the very figure this branch exists to refuse. The success row at
            # the bottom of this function owns those key names.
            implied_achieved_tflops=round(achieved_tflops, 1),
            peak_fp16_tflops=round(peak_tflops, 1),
            implied_mfu_pct=round((achieved_tflops / peak_tflops) * 100.0, 1),
            theoretical_flops=theoretical_flops,
            kernel_union_ms=round(total_kernel_ms, 2),
            kernel_count=kernel_count,
        )

    mfu_pct = (achieved_tflops / peak_tflops) * 100.0
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
            classification = "High kernel throughput (likely compute-bound)"
            severity = "info"
            recommendation = (
                "Workload has good kernel throughput. "
                "For further gains, consider kernel-level optimization with NCU "
                "(occupancy, warp efficiency, instruction mix)."
            )
        elif mfu_pct >= 15:
            classification = "Moderate kernel throughput (mixed bound)"
            severity = "warning"
            recommendation = (
                "Workload is in a transition zone. "
                "Consider increasing batch size to raise arithmetic intensity, "
                "using FlashAttention for attention kernels, or fusing small ops with torch.compile()."
            )
        elif mfu_pct >= 5:
            classification = "Low kernel throughput (likely memory-bound)"
            severity = "warning"
            recommendation = (
                "Kernels are likely bottlenecked by HBM bandwidth rather than compute. "
                "Increase batch size, use operator fusion (torch.compile), "
                "enable FlashAttention, or check for excessive memory-bound element-wise ops."
            )
        else:
            classification = "Severely low kernel throughput"
            severity = "critical"
            recommendation = (
                "GPU has severely low kernel throughput vs peak. Common causes: excessive CPU overhead, "
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
            "kernel_union_ms": round(total_kernel_ms, 2),
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
    # No ``if "error" in r`` arm: every failure this function decides on now
    # abstains, and ``Skill.format_rows`` renders an abstention with its reason
    # before a ``format_fn`` is ever called. A missing required parameter still
    # raises out of ``Skill.execute`` before this runs — that is issue #386, and
    # it is the CLI's to catch, not this skill's.

    lines = [
        "── Arithmetic Intensity Assessment (Roofline) ──",
        f"  GPU:              {r['gpu_name']}",
        f"  Peak FP16:        {r['peak_fp16_tflops']} TFLOPS",
        f"  HBM Bandwidth:    {r['hbm_bw_gbps']} GB/s",
        f"  Ridge Point:      {r['ridge_point_flop_per_byte']} FLOP/Byte",
        "",
        f"  Kernel Union Time:  {r['kernel_union_ms']:.2f} ms  ({r['kernel_count']} kernels)",
        f"  Achieved TFLOPS:    {r['achieved_tflops']} TFLOPS",
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
