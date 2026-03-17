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
