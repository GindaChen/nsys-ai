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


def _classify_h2d_pattern(rows: list[dict]) -> dict:
    """Classify H2D transfer distribution into init_heavy / spread_out / spike."""
    if not rows:
        return {"_pattern": True, "type": "none", "detail": "No H2D transfers"}

    total_mb = sum(r["total_mb"] for r in rows)
    if total_mb <= 0:
        return {"_pattern": True, "type": "none", "detail": "No bytes transferred"}

    # Init-heavy: first 2 seconds account for >80% of total bytes
    first_2s_mb = sum(r["total_mb"] for r in rows if r["second"] <= 1)
    if first_2s_mb / total_mb > 0.8:
        return {
            "_pattern": True,
            "type": "init_heavy",
            "detail": (
                f"H2D concentrated in first 2 seconds "
                f"({first_2s_mb:.1f}/{total_mb:.1f} MB = "
                f"{100 * first_2s_mb / total_mb:.0f}%). "
                f"Likely model weight loading — normal behavior."
            ),
        }

    # Spike detection: any second exceeds 3× median
    sorted_mb = sorted(r["total_mb"] for r in rows)
    median_mb = sorted_mb[len(sorted_mb) // 2]
    if median_mb > 0:
        spikes = [r for r in rows if r["total_mb"] > 3 * median_mb]
        if spikes:
            spike_secs = ", ".join(str(r["second"]) for r in spikes[:5])
            return {
                "_pattern": True,
                "type": "spike",
                "detail": (
                    f"{len(spikes)} spike(s) detected at second(s) [{spike_secs}] "
                    f"(>{3 * median_mb:.1f} MB vs median {median_mb:.1f} MB). "
                    f"Check timeline at those timestamps for unexpected data movement."
                ),
                "spike_seconds": [r["second"] for r in spikes],
            }

    # Spread-out: transfers happen across many seconds
    if len(rows) >= 3:
        return {
            "_pattern": True,
            "type": "spread_out",
            "detail": (
                f"H2D transfers spread across {len(rows)} seconds "
                f"({total_mb:.1f} MB total). Every step has H2D — "
                f"consider pin_memory=True and increasing DataLoader num_workers."
            ),
        }

    return {"_pattern": True, "type": "unknown", "detail": "Too few data points to classify"}


def _format_dist(rows):
    if not rows:
        return "(No H2D memory transfers found)"

    # Separate data rows from pattern metadata
    data_rows = [r for r in rows if not r.get("_pattern")]
    pattern = next((r for r in rows if r.get("_pattern")), None)

    if not data_rows:
        return "(No H2D memory transfers found)"

    lines = [
        "── H2D Time Distribution ──",
        f"{'Second':<8s}  {'Count':>7s}  {'Total(MB)':>10s}  {'Avg GB/s':>10s}",
        "─" * 41,
    ]
    for r in data_rows:
        lines.append(
            f"{r['second']:<8d}  {r['ops']:>7d}  {r['total_mb']:>10.2f}  {r['avg_gbps']:>10.2f}"
        )

    if pattern:
        ptype = pattern.get("type", "unknown")
        icon = {"init_heavy": "✅", "spread_out": "⚠️", "spike": "🔴"}.get(ptype, "ℹ️")
        lines.append(f"\n  {icon} Pattern: {ptype}")
        lines.append(f"     {pattern.get('detail', '')}")

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
    SELECT MIN(k.start) AS min_start
    FROM {memcpy_table} k
    WHERE k.copyKind = 1 {trim_clause}
)
SELECT
    CAST((k.start - b.min_start) / 1000000000.0 AS INT) AS second,
    COUNT(*) AS ops,
    SUM(k.bytes) / 1e6 AS total_mb,
    COALESCE(SUM(k.bytes) / NULLIF(SUM(k.[end] - k.start), 0), 0) * 1e9 / 1e9 AS avg_gbps
FROM {memcpy_table} k CROSS JOIN baseline b
WHERE k.copyKind = 1 {trim_clause}
GROUP BY 1
ORDER BY 1""",
    format_fn=_format_dist,
    tags=["memory", "transfer", "H2D", "distribution", "time", "leak"],
)


# Replace the direct SQL with a safe execute_fn for the module
def _execute_h2d_dist(conn, **kwargs):
    """Execute the H2D distribution query safely with pattern classification.

    This function is assigned as ``H2D_DIST_SKILL.execute_fn`` and wraps the
    underlying SQL execution so that if the memcpy table does not exist, it
    returns an empty result instead of propagating an sqlite3.OperationalError.

    After executing the SQL, appends a pattern classification dict as the
    last element of the result list.
    """
    import sqlite3

    # Create a temporary Skill that uses the same SQL but no custom execute_fn
    temp_skill = Skill(
        name=H2D_DIST_SKILL.name,
        title=H2D_DIST_SKILL.title,
        description=H2D_DIST_SKILL.description,
        category=H2D_DIST_SKILL.category,
        sql=H2D_DIST_SKILL.sql,
        format_fn=H2D_DIST_SKILL.format_fn,
        tags=getattr(H2D_DIST_SKILL, "tags", None),
    )
    import duckdb

    try:
        rows = temp_skill.execute(conn, **kwargs)
    except (sqlite3.OperationalError, duckdb.Error) as exc:
        err_msg = str(exc).lower()
        if "no such table" in err_msg or "does not exist" in err_msg:
            return []
        raise

    # Append pattern classification as metadata
    if rows:
        rows.append(_classify_h2d_pattern(rows))
    return rows


H2D_DIST_SKILL.execute_fn = _execute_h2d_dist


SKILLS = [SKILL, H2D_DIST_SKILL]
