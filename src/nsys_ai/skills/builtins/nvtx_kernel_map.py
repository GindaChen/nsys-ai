"""Map NVTX annotation ranges to their GPU kernel children."""

from ..base import Skill, SkillParam


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
    sql="""\
SELECT n.text AS nvtx_text,
       s.value AS kernel_name,
       ROUND(k.start / 1e6, 3) AS start_ms,
       ROUND(k.[end] / 1e6, 3) AS end_ms
FROM NVTX_EVENTS n
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
  ON n.eventType = 59
  AND n.globalTid = r.globalTid
  AND n.start <= r.start
  AND n.[end] >= r.[end]
JOIN CUPTI_ACTIVITY_KIND_KERNEL k
  ON r.correlationId = k.correlationId
JOIN StringIds s ON k.shortName = s.id
WHERE 1=1 {trim_clause}
ORDER BY k.start
LIMIT {limit}""",
    params=[SkillParam("limit", "Max results", "int", False, 50)],
    format_fn=_format,
    tags=["nvtx", "kernel", "source", "attribution", "mapping"],
)
