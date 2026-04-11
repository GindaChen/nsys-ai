"""Map NVTX annotation ranges to their GPU kernel children."""

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    """Execute NVTX→Kernel mapping via efficient attribution module."""
    from ...nvtx_attribution import attribute_kernels_to_nvtx

    limit = int(kwargs.get("limit", 50))
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    trim = (trim_start, trim_end) if trim_start is not None and trim_end is not None else None

    # Get the sqlite path for Tier 1 (nsys recipe) attempt
    sqlite_path = kwargs.get("_sqlite_path")

    rows = attribute_kernels_to_nvtx(conn, sqlite_path=sqlite_path, trim=trim, limit=limit)

    # Select the earliest kernels by start time without fully sorting
    import heapq
    if len(rows) > limit:
        rows = heapq.nsmallest(limit, rows, key=lambda r: r["k_start"])

    # Format output to match expected schema
    return [
        {
            "nvtx_text": r["nvtx_text"],
            "kernel_name": r["kernel_name"],
            "start_ms": round(r["k_start"] / 1e6, 3),
            "end_ms": round(r["k_end"] / 1e6, 3),
        }
        for r in rows
    ]


def _format(rows):
    if not rows:
        return "(No NVTX-to-kernel mappings found — are NVTX annotations present?)"
    lines = [
        "── NVTX → Kernel Mapping ──",
        f"{'NVTX Range':<50s}  {'Kernel':<50s}  {'Start(ms)':>10s}  {'End(ms)':>10s}",
        "─" * 126,
    ]
    for r in rows:
        nvtx = r["nvtx_text"] or "(unnamed)"
        if len(nvtx) > 48:
            nvtx = nvtx[:45] + "..."
        kern = r["kernel_name"]
        if len(kern) > 48:
            kern = kern[:45] + "..."
        lines.append(f"{nvtx:<50s}  {kern:<50s}  {r['start_ms']:>10.3f}  {r['end_ms']:>10.3f}")
    return "\n".join(lines)


SKILL = Skill(
    name="nvtx_kernel_map",
    title="NVTX → Kernel Mapping",
    description=(
        "Maps NVTX annotation ranges to the GPU kernels that execute within them. "
        "This is the core of source-code attribution: each NVTX range tells you "
        "which code region launched which kernels."
    ),
    category="nvtx",
    execute_fn=_execute,
    params=[SkillParam("limit", "Max results", "int", False, 50)],
    format_fn=_format,
    tags=["nvtx", "kernel", "source", "attribution", "mapping"],
)
