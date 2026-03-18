"""Per-iteration timing analysis.

Uses detect_iterations() from overlap.py to find repeating training iterations
via a top-level NVTX marker and report per-iteration GPU timing and kernel counts.
This is a Python-level skill (execute_fn).
"""

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    from ...overlap import detect_iterations
    from ...profile import Profile

    prof = Profile._from_conn(conn)
    device = int(kwargs.get("device", 0))
    marker = kwargs.get("marker", "sample_0")
    # Support --trim passthrough from agent analyze
    trim = None
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    if trim_start is not None and trim_end is not None:
        trim = (int(trim_start), int(trim_end))
    return detect_iterations(prof, device, trim=trim, marker=marker)


def _format(rows):
    if not rows:
        return "(No iterations detected — NVTX marker not found and no large gaps found)"
    is_heuristic = any(it.get('text', '').startswith('heuristic') for it in rows)
    title = "── Iteration Timings (Heuristic Fallback) ──" if is_heuristic else "── Iteration Timings ──"
    lines = [title]
    for it in rows:
        lines.append(
            f"  iter {it['iteration']:2d}  "
            f"{it['duration_ms']:8.1f}ms  "
            f"({it['kernel_count']} kernels, {it['nccl_count']} NCCL)  "
            f"compute={it['compute_ms']:.1f}ms"
        )
    if len(rows) > 1:
        durs = [it["duration_ms"] for it in rows]
        avg = sum(durs) / len(durs)
        lines.append(
            f"\n  Average: {avg:.1f}ms  Min: {min(durs):.1f}ms  Max: {max(durs):.1f}ms"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="iteration_timing",
    title="Per-Iteration Timing Analysis",
    description=(
        "Detects repeating training iterations using a top-level NVTX marker "
        "and reports per-iteration GPU timing, kernel counts, and NCCL counts. "
        "Use to identify slow iterations and iteration variance."
    ),
    category="nvtx",
    execute_fn=_execute,
    format_fn=_format,
    params=[
        SkillParam("device", "GPU device ID", "int", False, 0),
        SkillParam("marker", "NVTX text pattern for iteration boundary", "str", False, "sample_0"),
    ],
    tags=["iteration", "timing", "variance", "nvtx", "training"],
)
