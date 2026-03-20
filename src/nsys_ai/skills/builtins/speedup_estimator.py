"""Speedup estimation framework.

Given profile metrics, estimates the potential speedup from common
optimizations. This is a pure-computation skill (no DB access needed)
that provides the "if you do X, you'll save Y ms" reasoning.

The agent or user provides current metrics; this skill computes
projected improvements.
"""

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    """Compute speedup estimates from provided metrics."""
    results = []

    iteration_ms = float(kwargs.get("iteration_ms", 0))
    # compute_ms available via kwargs but not used in current estimates
    nccl_ms = float(kwargs.get("nccl_ms", 0))
    idle_ms = float(kwargs.get("idle_ms", 0))
    overlap_pct = float(kwargs.get("overlap_pct", 0))
    tp_degree = int(kwargs.get("tp_degree", 1))
    model_params_b = float(kwargs.get("model_params_b", 0))
    gpu_memory_gb = float(kwargs.get("gpu_memory_gb", 80))

    if iteration_ms <= 0:
        return [{"error": "iteration_ms must be provided and > 0"}]

    # --- Estimate 1: Eliminate idle gaps ---
    if idle_ms > 0:
        saved = idle_ms
        new_iter = iteration_ms - saved
        speedup = iteration_ms / new_iter if new_iter > 0 else 1
        results.append(
            {
                "optimization": "Eliminate GPU Idle Gaps",
                "method": "CUDA graphs / async data loading / remove explicit syncs",
                "current_ms": idle_ms,
                "saved_ms": round(saved, 1),
                "new_iteration_ms": round(new_iter, 1),
                "speedup": round(speedup, 3),
                "confidence": "medium",
            }
        )

    # --- Estimate 2: Perfect NCCL overlap ---
    if nccl_ms > 0 and overlap_pct < 100:
        nccl_exposed = nccl_ms * (1 - overlap_pct / 100)
        if nccl_exposed > 0:
            new_iter = iteration_ms - nccl_exposed
            speedup = iteration_ms / new_iter if new_iter > 0 else 1
            results.append(
                {
                    "optimization": "Perfect Compute/NCCL Overlap",
                    "method": "Overlap all NCCL with compute via stream pipelining",
                    "current_exposed_nccl_ms": round(nccl_exposed, 1),
                    "saved_ms": round(nccl_exposed, 1),
                    "new_iteration_ms": round(new_iter, 1),
                    "speedup": round(speedup, 3),
                    "confidence": "medium",
                }
            )

    # --- Estimate 3: Reduce TP degree ---
    if tp_degree > 1 and nccl_ms > 0:
        # Rough model: reducing TP from N to 1 eliminates ~90% of NCCL
        # (some AllReduce may remain for DDP grad sync)
        # But only feasible if model fits on one GPU
        bf16_size_gb = model_params_b * 2  # BF16 = 2 bytes/param
        optimizer_size_gb = bf16_size_gb * 4  # Adam states ≈ 4× params
        total_needed_gb = bf16_size_gb + optimizer_size_gb
        fits_on_one = total_needed_gb < gpu_memory_gb * 0.85  # 85% safe margin

        if fits_on_one:
            nccl_savings = nccl_ms * 0.9
            new_iter = iteration_ms - nccl_savings
            speedup = iteration_ms / new_iter if new_iter > 0 else 1
            results.append(
                {
                    "optimization": f"Reduce TP={tp_degree} → TP=1",
                    "method": f"Model ({model_params_b:.1f}B params, ~{total_needed_gb:.0f}GB) fits on {gpu_memory_gb:.0f}GB GPU",
                    "nccl_savings_ms": round(nccl_savings, 1),
                    "saved_ms": round(nccl_savings, 1),
                    "new_iteration_ms": round(new_iter, 1),
                    "speedup": round(speedup, 3),
                    "confidence": "high",
                    "feasibility": "HIGH — model fits on single GPU",
                }
            )
        else:
            half_tp = max(1, tp_degree // 2)
            nccl_savings = nccl_ms * 0.4  # Rough: halving TP saves ~40% NCCL
            new_iter = iteration_ms - nccl_savings
            speedup = iteration_ms / new_iter if new_iter > 0 else 1
            results.append(
                {
                    "optimization": f"Reduce TP={tp_degree} → TP={half_tp}",
                    "method": f"Model too large for 1 GPU ({total_needed_gb:.0f}GB > {gpu_memory_gb:.0f}GB), halve TP instead",
                    "nccl_savings_ms": round(nccl_savings, 1),
                    "saved_ms": round(nccl_savings, 1),
                    "new_iteration_ms": round(new_iter, 1),
                    "speedup": round(speedup, 3),
                    "confidence": "low",
                    "feasibility": "MEDIUM — requires memory optimization",
                }
            )

    if not results:
        results.append(
            {
                "optimization": "No clear optimization path",
                "method": "Profile looks well-optimized; consider NCU for kernel-level tuning",
                "saved_ms": 0,
                "speedup": 1.0,
                "confidence": "n/a",
            }
        )

    return results


def _format(rows):
    if not rows:
        return "(No estimates)"
    if "error" in rows[0]:
        return f"(Error: {rows[0]['error']})"
    lines = ["── Speedup Estimates ──"]
    for r in rows:
        saved = r.get("saved_ms", 0)
        speedup = r.get("speedup", 1)
        lines.append(f"\n  📊 {r['optimization']}")
        lines.append(f"     Method: {r['method']}")
        lines.append(f"     Potential savings: {saved:.1f}ms ({speedup:.3f}× speedup)")
        lines.append(f"     Confidence: {r.get('confidence', '?')}")
    return "\n".join(lines)


SKILL = Skill(
    name="speedup_estimator",
    title="Speedup Estimation Framework",
    description=(
        "Estimates potential speedup from common optimizations given profile metrics. "
        "Computes projected improvements for: eliminating idle gaps, perfect NCCL overlap, "
        "reducing tensor parallel degree. Requires iteration_ms and relevant metrics."
    ),
    category="analysis",
    execute_fn=_execute,
    format_fn=_format,
    params=[
        SkillParam("iteration_ms", "Current iteration time in ms", "float", True, None),
        SkillParam("compute_ms", "Total compute time in ms", "float", False, 0),
        SkillParam("nccl_ms", "Total NCCL time in ms", "float", False, 0),
        SkillParam("idle_ms", "Total GPU idle time in ms", "float", False, 0),
        SkillParam("overlap_pct", "Current compute/NCCL overlap %", "float", False, 0),
        SkillParam("tp_degree", "Current tensor parallel degree", "int", False, 1),
        SkillParam("model_params_b", "Model size in billions of params", "float", False, 0),
        SkillParam("gpu_memory_gb", "GPU memory in GB", "float", False, 80),
    ],
    tags=["speedup", "estimation", "optimization", "projection", "analysis"],
)
