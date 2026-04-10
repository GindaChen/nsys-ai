"""Communicator-aware NCCL analysis skill."""

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    from ...nccl_communicator import analyze_nccl_communicators
    from ...profile import Profile

    prof = Profile._from_conn(conn)
    device = kwargs.get("device")
    if device is not None:
        device = int(device)

    trim = None
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    if trim_start is not None and trim_end is not None:
        trim = (int(trim_start), int(trim_end))

    return analyze_nccl_communicators(prof, device=device, trim=trim)


def _format(rows):
    from ...nccl_communicator import format_nccl_communicator

    return format_nccl_communicator(rows)


SKILL = Skill(
    name="nccl_communicator_analysis",
    title="NCCL Communicator-Aware Analysis",
    description=(
        "Decodes NVTX extended payload blobs from enriched Nsight exports to group NCCL "
        "communication by communicator ID and collective type. Reports rank-count hints, "
        "message sizes, and effective bandwidth when the payload contains full byte counts."
    ),
    category="communication",
    execute_fn=_execute,
    format_fn=_format,
    params=[SkillParam("device", "Optional GPU device ID for drill-down", "int", False, None)],
    tags=[
        "nccl",
        "communicator",
        "allreduce",
        "allgather",
        "broadcast",
        "distributed",
        "payload",
    ],
)
