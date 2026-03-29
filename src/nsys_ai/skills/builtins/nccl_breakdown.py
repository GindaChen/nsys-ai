"""NCCL collective operation breakdown — per-stream.

Delegates to the unified ``nsys_ai.overlap.nccl_breakdown()`` engine so that
the Agent skill, CLI, and TUI chat tool all produce identical results.
"""

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    from ...overlap import nccl_breakdown
    from ...profile import Profile

    prof = Profile._from_conn(conn)
    device = int(kwargs.get("device", 0))

    trim = None
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    if trim_start is not None and trim_end is not None:
        trim = (int(trim_start), int(trim_end))

    return nccl_breakdown(prof, device, trim)


def _format(rows):
    from ...overlap import format_nccl

    base = format_nccl(rows)

    # Add a brief diagnostic hint when no NCCL collectives are present.
    # This mirrors the previous, more actionable behavior while still
    # delegating core formatting to the shared overlap engine.
    if not rows:
        hint = (
            "\n\nHint: This usually means either the profile was captured on a "
            "single GPU, or that NCCL communication was not recorded for this run."
        )
        return f"{base}{hint}"

    return base


SKILL = Skill(
    name="nccl_breakdown",
    title="NCCL Collective Breakdown",
    description=(
        "Summarizes NCCL collective operations (AllReduce, AllGather, ReduceScatter, etc.) "
        "per CUDA stream, showing count, total time, and variability. "
        "Per-stream grouping helps distinguish TP vs PP vs DP communication channels, "
        "since each parallelism dimension typically uses a dedicated stream."
    ),
    category="communication",
    execute_fn=_execute,
    format_fn=_format,
    params=[
        SkillParam("device", "GPU device ID", "int", False, 0),
    ],
    tags=["nccl", "collective", "allreduce", "communication", "distributed", "multi-gpu", "stream"],
)
