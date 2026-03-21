"""Theoretical FLOPs calculator for transformer operations.

Exposes the exact-arithmetic FLOP calculator from region_mfu.py as a Skill.
LLMs cannot reliably multiply large numbers like 131072² × 4096 × 32 —
this tool does the exact computation.
"""

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    from ...region_mfu import compute_theoretical_flops

    operation = kwargs.get("operation", "")
    result = compute_theoretical_flops(
        operation,
        hidden_dim=int(kwargs.get("hidden_dim", 0)),
        seq_len=int(kwargs.get("seq_len", 0)),
        num_layers=int(kwargs.get("num_layers", 1)),
        ffn_dim=int(kwargs["ffn_dim"]) if kwargs.get("ffn_dim") is not None else None,
        batch_size=int(kwargs.get("batch_size", 1)),
        multiplier=int(kwargs.get("multiplier", 1)),
        M=int(kwargs.get("M", 0)),
        N=int(kwargs.get("N", 0)),
        K=int(kwargs.get("K", 0)),
    )
    return [result]


def _format(rows):
    if not rows:
        return "(No FLOPs data)"
    r = rows[0]
    if "error" in r:
        err = r["error"]
        return f"(FLOPs error: {err.get('code', '?')}: {err.get('message', '')})"
    lines = ["── Theoretical FLOPs ──"]
    lines.append(f"  Operation: {r.get('operation', '?')}")
    lines.append(f"  Total FLOPs: {r.get('theoretical_flops', 0):,.0f}")
    lines.append(f"  Formula: {r.get('formula', '?')}")
    breakdown = r.get("breakdown", {})
    if breakdown:
        for k, v in breakdown.items():
            if isinstance(v, (int, float)):
                lines.append(f"    {k}: {v:,.0f}")
            else:
                lines.append(f"    {k}: {v}")
    return "\n".join(lines)


SKILL = Skill(
    name="theoretical_flops",
    title="Theoretical FLOPs Calculator",
    description=(
        "Computes exact theoretical FLOPs for transformer operations "
        "(attention, qkv_proj, output_proj, mlp, full_layer, full_model, linear). "
        "LLMs should use this instead of trying to multiply large numbers."
    ),
    category="kernels",
    execute_fn=_execute,
    format_fn=_format,
    params=[
        SkillParam(
            "operation",
            "Operation: attention/qkv_proj/output_proj/mlp/full_layer/full_model/linear",
            "str",
            True,
            None,
        ),
        SkillParam("hidden_dim", "Model hidden dimension (H)", "int", False, 0),
        SkillParam("seq_len", "Sequence length (S)", "int", False, 0),
        SkillParam("num_layers", "Number of transformer layers", "int", False, 1),
        SkillParam("ffn_dim", "FFN intermediate dimension (defaults to 4*H)", "int", False, None),
        SkillParam("batch_size", "Batch size", "int", False, 1),
        SkillParam("multiplier", "1=fwd, 3=fwd+bwd, 4=fwd+bwd+ckpt", "int", False, 1),
        SkillParam("M", "For 'linear': dimension M", "int", False, 0),
        SkillParam("N", "For 'linear': dimension N", "int", False, 0),
        SkillParam("K", "For 'linear': dimension K", "int", False, 0),
    ],
    tags=["flops", "theoretical", "mfu", "transformer", "computation"],
)
