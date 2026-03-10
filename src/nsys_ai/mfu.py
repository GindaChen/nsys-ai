"""
mfu.py — Model FLOPs Utilization (MFU) computation.

Shared by single-profile chat and scripts. Does not depend on diff;
use this module for any MFU calculation (one profile or compare two).
Formula: MFU = achieved_model_TFLOPS / peak_TFLOPS
         achieved_model_TFLOPS = (model_flops_per_step / step_time_s) / 1e12
"""

from __future__ import annotations


def compute_mfu_single(
    step_time_s: float,
    model_flops_per_step: float,
    peak_tflops: float,
) -> dict:
    """
    Compute MFU for a single profile / step.

    step_time_s: Wall-clock time per step in seconds (from profile span or iteration).
    model_flops_per_step: MUST be provided by user (nsys does not store model FLOPs).
    peak_tflops: GPU peak TFLOPS for precision (e.g. 989 for H100 FP16, 312 for A100 FP16).
    """
    if model_flops_per_step <= 0 or peak_tflops <= 0:
        return {
            "error": "model_flops_per_step and peak_tflops must be positive. User must provide model FLOPs (e.g. from fvcore, 6*N_params*tokens_per_step for Transformer).",
            "formula": "MFU = (model_flops_per_step / step_time_s) / 1e12 / peak_tflops",
        }
    if step_time_s <= 0:
        return {
            "error": "step_time_s must be positive.",
            "formula": "MFU = (model_flops_per_step / step_time_s) / 1e12 / peak_tflops",
        }
    achieved = (model_flops_per_step / step_time_s) / 1e12
    mfu_pct = 100.0 * achieved / peak_tflops
    return {
        "step_time_s": round(step_time_s, 4),
        "model_flops_per_step": model_flops_per_step,
        "peak_tflops": peak_tflops,
        "achieved_model_TFLOPS": round(achieved, 2),
        "MFU_pct": round(mfu_pct, 2),
    }


def compute_mfu_from_args(args: dict) -> dict:
    """Parse tool-call args (step_time_s, model_flops_per_step, peak_tflops) and return MFU result. Shared by chat and diff_tools."""
    return compute_mfu_single(
        float(args.get("step_time_s", 0)),
        float(args.get("model_flops_per_step", 0)),
        float(args.get("peak_tflops", 0)),
    )


def compute_mfu_compare(
    step_time_before_s: float,
    step_time_after_s: float,
    model_flops_per_step: float,
    peak_tflops: float,
) -> dict:
    """
    Compute MFU for before/after (e.g. diff or script comparison).
    Delegates to compute_mfu_single twice; no duplicate formula logic.
    """
    before = compute_mfu_single(step_time_before_s, model_flops_per_step, peak_tflops)
    after = compute_mfu_single(step_time_after_s, model_flops_per_step, peak_tflops)
    if "error" in before:
        return before
    if "error" in after:
        return after
    return {
        "step_time_before_s": before["step_time_s"],
        "step_time_after_s": after["step_time_s"],
        "model_flops_per_step": model_flops_per_step,
        "peak_tflops": peak_tflops,
        "achieved_model_TFLOPS": {
            "before": before["achieved_model_TFLOPS"],
            "after": after["achieved_model_TFLOPS"],
        },
        "MFU_pct": {"before": before["MFU_pct"], "after": after["MFU_pct"]},
        "delta_MFU_pct": round(after["MFU_pct"] - before["MFU_pct"], 2),
    }
