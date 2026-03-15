"""Detect GPU idle gaps (bubbles) between consecutive kernel executions."""

from ..base import Skill, SkillParam


def _format(rows):
    if not rows:
        return "(No significant GPU idle gaps found — GPU is well-utilized)"
    lines = [
        "── GPU Idle Gaps (Bubbles) ──",
        f"{'Stream':>7s}  {'Gap(ms)':>9s}  {'Before Kernel':<50s}  {'After Kernel':<50s}",
        "─" * 122,
    ]
    for r in rows:
        before = r["before_kernel"] or "?"
        after = r["after_kernel"] or "?"
        if len(before) > 48:
            before = before[:45] + "..."
        if len(after) > 48:
            after = after[:45] + "..."
        lines.append(f"{r['streamId']:>7d}  {r['gap_ms']:>9.3f}  {before:<50s}  {after:<50s}")
    return "\n".join(lines)


SKILL = Skill(
    name="gpu_idle_gaps",
    title="GPU Idle Gaps (Bubbles)",
    description=(
        "Finds idle gaps between consecutive GPU kernels on each stream — "
        "the 'bubbles' in the pipeline. Large gaps indicate the GPU is waiting "
        "for CPU, data transfer, or synchronization. These are prime optimization targets."
    ),
    category="kernels",
    sql="""\
WITH ordered AS (
    SELECT k.streamId,
           k.start, k.[end],
           s.value AS kernel_name,
           LAG(k.[end]) OVER (PARTITION BY k.streamId ORDER BY k.start) AS prev_end,
           LAG(s.value) OVER (PARTITION BY k.streamId ORDER BY k.start) AS prev_kernel
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.shortName = s.id
    WHERE 1=1 {trim_clause}
)
SELECT streamId,
       prev_end AS start_ns,
       start AS end_ns,
       ROUND((start - prev_end) / 1e6, 3) AS gap_ms,
       prev_kernel AS before_kernel,
       kernel_name AS after_kernel
FROM ordered
WHERE prev_end IS NOT NULL AND (start - prev_end) > {min_gap_ns}
ORDER BY gap_ms DESC
LIMIT {limit}""",
    params=[
        SkillParam("min_gap_ns", "Minimum gap in nanoseconds to report", "int", False, 1000000),
        SkillParam("limit", "Max results", "int", False, 20),
    ],
    format_fn=_format,
    tags=["bubble", "idle", "gap", "pipeline", "stall", "utilization"],
)
