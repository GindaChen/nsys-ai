"""Memory transfer summary — H2D, D2H, D2D, P2P breakdown."""

from ..base import Skill

_COPY_KINDS = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}


def _format(rows):
    if not rows:
        return "(No memory transfers found)"
    lines = [
        "── Memory Transfers Summary ──",
        f"{'Direction':<10s}  {'Count':>7s}  {'Total(MB)':>10s}  {'Total(ms)':>10s}",
        "─" * 44,
    ]
    for r in rows:
        direction = _COPY_KINDS.get(r["copyKind"], f"kind={r['copyKind']}")
        lines.append(
            f"{direction:<10s}  {r['count']:>7d}  {r['total_mb']:>10.2f}  {r['total_ms']:>10.2f}"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="memory_transfers",
    title="Memory Transfer Summary",
    description=(
        "Breaks down memory copy operations by direction (Host→Device, Device→Host, "
        "Device→Device, Peer-to-Peer). Excessive H2D transfers in the critical path "
        "often indicate data not being pre-staged on GPU."
    ),
    category="memory",
    sql="""\
SELECT k.copyKind,
       COUNT(*) AS count,
       ROUND(SUM(k.bytes) / 1e6, 2) AS total_mb,
       ROUND(SUM(k.[end] - k.start) / 1e6, 2) AS total_ms
FROM {memcpy_table} k
WHERE 1=1 {trim_clause}
GROUP BY k.copyKind
ORDER BY total_ms DESC""",
    format_fn=_format,
    tags=["memory", "transfer", "H2D", "D2H", "copy", "bandwidth"],
)


def _format_dist(rows):
    if not rows:
        return "(No H2D memory transfers found)"
    lines = [
        "── H2D Time Distribution ──",
        f"{'Second':<8s}  {'Count':>7s}  {'Total(MB)':>10s}  {'Avg GB/s':>10s}",
        "─" * 41,
    ]
    for r in rows:
        lines.append(
            f"{r['second']:<8d}  {r['ops']:>7d}  {r['total_mb']:>10.2f}  {r['avg_gbps']:>10.2f}"
        )
    return "\n".join(lines)


H2D_DIST_SKILL = Skill(
    name="h2d_distribution",
    title="H2D Transfer Time Distribution",
    description=(
        "Groups Host-to-Device (H2D) memory transfers by second. "
        "Useful for distinguishing between initial model loading (concentrated at start) "
        "and continuous data feeding in the training/inference loop."
    ),
    category="memory",
    sql="""\
WITH baseline AS (
    SELECT MIN(start) AS min_start FROM {memcpy_table}
)
SELECT 
    CAST((k.start - b.min_start) / 1000000000.0 AS INT) AS second,
    COUNT(*) AS ops,
    SUM(k.bytes) / 1e6 AS total_mb,
    (SUM(k.bytes) / NULLIF(SUM(k.[end] - k.start), 0)) * 1e9 / 1e9 AS avg_gbps
FROM {memcpy_table} k CROSS JOIN baseline b
WHERE k.copyKind = 1 {trim_clause}
GROUP BY 1
ORDER BY 1""",
    format_fn=_format_dist,
    tags=["memory", "transfer", "H2D", "distribution", "time", "leak"],
)

# Replace the direct SQL with a safe execute_fn for the module
def _execute_h2d_dist(conn, **kwargs):
    # Temporarily remove execute_fn to use the base SQL machinery
    fn = H2D_DIST_SKILL.execute_fn
    H2D_DIST_SKILL.execute_fn = None
    try:
        res = H2D_DIST_SKILL.execute(conn, **kwargs)
    except Exception:
        res = []
    finally:
        H2D_DIST_SKILL.execute_fn = fn
    return res

H2D_DIST_SKILL.execute_fn = _execute_h2d_dist



SKILLS = [SKILL, H2D_DIST_SKILL]

