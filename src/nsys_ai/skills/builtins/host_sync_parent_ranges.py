"""Host-sync parent-range attribution.

For each NVTX range that contains host-GPU sync events
(`aten::item`, `aten::_local_scalar_dense`, `cudaStreamSynchronize`, ...),
report the total sync time and event count. Used by Mode 6 to localize
*which* training phase owns the sync cost before correlating with source
code via `grep` (PRINCIPLES.md §5.7 Step 1).
"""

from ...connection import DB_ERRORS, wrap_connection
from ..base import Skill, SkillParam

# Substring patterns matched via SQL `LIKE '%<pattern>%'`. Intentionally narrow:
# host-GPU sync signatures only — not every NVTX event. Additions should be
# justified by an observed bottleneck class, not speculative.
DEFAULT_PATTERNS = "item,_local_scalar_dense,cudaStreamSynchronize"

# Cap on `limit` — guards the O(n²) ancestry self-join on pathological input.
_MAX_LIMIT = 1000


def _execute(conn, **kwargs):
    try:
        limit = int(kwargs.get("limit", 5))
    except (TypeError, ValueError):
        return [{"error": "`limit` must be a positive integer"}]
    if limit < 1:
        return [{"error": f"`limit` must be >= 1 (got {limit})"}]
    if limit > _MAX_LIMIT:
        limit = _MAX_LIMIT

    raw_patterns = str(kwargs.get("patterns") or DEFAULT_PATTERNS)
    patterns = [p.strip() for p in raw_patterns.split(",") if p.strip()]
    if not patterns:
        patterns = [p.strip() for p in DEFAULT_PATTERNS.split(",")]

    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")

    adapter = wrap_connection(conn)
    tables = adapter.resolve_activity_tables()
    nvtx_table = tables.get("nvtx", "NVTX_EVENTS")

    has_textid = adapter.detect_nvtx_text_id()
    if has_textid:
        label_expr = "COALESCE(n.text, s.value)"
        label_join = "LEFT JOIN StringIds s ON n.textId = s.id"
    else:
        label_expr = "n.text"
        label_join = ""

    trim_where = ""
    trim_params: list[int] = []
    if trim_start is not None and trim_end is not None:
        trim_where = "AND n.start >= ? AND n.[end] <= ?"
        trim_params = [int(trim_start), int(trim_end)]

    # Bound the LIKE patterns — user-supplied input flows through `patterns` param.
    like_parts = []
    like_params: list[str] = []
    for p in patterns:
        like_parts.append("child.label LIKE ?")
        like_params.append(f"%{p}%")
    like_clause = " OR ".join(like_parts)

    # Self-exclusion predicate: parent must cover child strictly (either a
    # different start or a different end). This keeps same-labeled but
    # distinct ancestors — e.g. nested `train_step` scopes — while still
    # preventing a range from matching itself.
    sql = f"""
        WITH resolved AS (
            SELECT {label_expr} AS label,
                   n.globalTid   AS tid,
                   n.start       AS start,
                   n.[end]       AS end_ns
            FROM {nvtx_table} n
            {label_join}
            WHERE n.[end] > n.start {trim_where}
        ),
        matched AS (
            SELECT parent.label                 AS parent_range,
                   child.label                  AS child_label,
                   child.end_ns - child.start   AS sync_ns
            FROM resolved child
            JOIN resolved parent
              ON parent.tid = child.tid
             AND parent.start <= child.start
             AND parent.end_ns >= child.end_ns
             AND (parent.start != child.start OR parent.end_ns != child.end_ns)
            WHERE ({like_clause}) AND parent.label IS NOT NULL
        ),
        parent_totals AS (
            SELECT parent_range,
                   COUNT(*)     AS n_syncs,
                   SUM(sync_ns) AS sync_ns
            FROM matched
            GROUP BY parent_range
        ),
        child_totals AS (
            SELECT parent_range,
                   child_label,
                   COUNT(*)     AS child_n_syncs,
                   SUM(sync_ns) AS child_sync_ns
            FROM matched
            GROUP BY parent_range, child_label
        ),
        ranked_children AS (
            SELECT parent_range,
                   child_label,
                   ROW_NUMBER() OVER (
                       PARTITION BY parent_range
                       ORDER BY child_sync_ns DESC, child_n_syncs DESC, child_label ASC
                   ) AS rn
            FROM child_totals
        )
        SELECT pt.parent_range,
               pt.n_syncs,
               pt.sync_ns,
               rc.child_label AS top_child_label
        FROM parent_totals pt
        LEFT JOIN ranked_children rc
          ON rc.parent_range = pt.parent_range
         AND rc.rn = 1
        ORDER BY pt.sync_ns DESC
        LIMIT {int(limit)}
    """

    try:
        cur = adapter.execute(sql, trim_params + like_params)
        rows = cur.fetchall()
    except DB_ERRORS as exc:
        return [
            {
                "error": (
                    f"host_sync_parent_ranges query failed: {exc}. "
                    "NVTX_EVENTS may be absent, or the profile schema is unexpected."
                ),
            }
        ]

    cols = ["parent_range", "n_syncs", "sync_ns", "top_child_label"]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        d["sync_ns"] = int(d["sync_ns"] or 0)
        d["sync_ms"] = round(d["sync_ns"] / 1e6, 3)
        out.append(d)
    return out


def _format(rows):
    if not rows:
        return "(No host-sync events found under any NVTX range)"
    if "error" in rows[0]:
        return f"Error: {rows[0]['error']}"

    lines = [
        "── Host-Sync Parent NVTX Ranges ──",
        f"{'Parent Range':<50s}  {'n':>6s}  {'Sync (ms)':>10s}  {'Top Child':<28s}",
        "─" * 102,
    ]
    for r in rows:
        parent = (r.get("parent_range") or "(unnamed)")[:48]
        child = (r.get("top_child_label") or "(unknown)")[:28]
        lines.append(
            f"{parent:<50s}  {r['n_syncs']:>6d}  {r['sync_ms']:>10.3f}  {child:<28s}"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="host_sync_parent_ranges",
    title="Host-Sync Parent NVTX Ranges",
    description=(
        "For each NVTX range that contains host-GPU sync events "
        "(`aten::item`, `_local_scalar_dense`, `cudaStreamSynchronize`, …), "
        "report total sync time and count. Localizes which training phase owns "
        "the sync cost so the plugin can grep the user's repo for the exact "
        "call site (PRINCIPLES.md §5.7 Step 1)."
    ),
    category="nvtx",
    execute_fn=_execute,
    format_fn=_format,
    params=[
        SkillParam("limit", "Max parent ranges to return", "int", False, 5),
        SkillParam(
            "patterns",
            "Comma-separated substrings matched via LIKE "
            "(default: 'item,_local_scalar_dense,cudaStreamSynchronize')",
            "str",
            False,
            DEFAULT_PATTERNS,
        ),
    ],
    tags=["nvtx", "sync", "host-sync", "attribution", "mode-6"],
)
