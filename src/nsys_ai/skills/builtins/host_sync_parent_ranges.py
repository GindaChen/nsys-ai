"""Host-sync parent-range attribution.

For each NVTX range that contains host-GPU sync events
(`aten::item`, `aten::_local_scalar_dense`, `cudaStreamSynchronize`, ...),
report the total sync time and event count. Used by Mode 6 to localize
*which* training phase owns the sync cost before correlating with source
code via `grep` (PRINCIPLES.md §5.7 Step 1).
"""

from collections import defaultdict

from ...connection import DB_ERRORS, wrap_connection
from ..base import Skill, SkillParam

# Substring patterns matched via SQL `LIKE '%<pattern>%'`. Intentionally narrow:
# host-GPU sync signatures only — not every NVTX event. Additions should be
# justified by an observed bottleneck class, not speculative.
DEFAULT_PATTERNS = "item,_local_scalar_dense,cudaStreamSynchronize"

# Cap on `limit` — the sweep credits every enclosing ancestor, so a deeply
# nested profile can produce many parent rows; bound what we return.
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

    # Bound the LIKE patterns — user-supplied input flows through `patterns`
    # param. Lower-case both sides so matching is consistent across DuckDB
    # (case-sensitive LIKE) and SQLite (case-insensitive LIKE by default).
    like_parts = []
    like_params: list[str] = []
    for p in patterns:
        like_parts.append("LOWER(label) LIKE ?")
        like_params.append(f"%{p.lower()}%")
    like_clause = " OR ".join(like_parts)

    # The attribution is a two-sided interval-containment join: a sync child is
    # credited to every range that covers it (parent.start <= child.start AND
    # parent.end >= child.end) on the same tid. Neither engine can run that as
    # SQL here: SQLite's nested-loop-only planner has no range-join operator, so
    # on an NVTX-heavy profile (~1.25M ranges, a handful of tids) it degrades to
    # ~n*m and never completes; and DuckDB's sqlite_scanner crashes optimizing
    # the CTE when the profile is direct-attached (no parquet cache). Both are
    # the no-cache paths issue #245 is about. So we do the containment ourselves
    # with a per-tid sweep, fed by two simple pattern/label queries that both
    # engines run without trouble.
    base = f"""
        SELECT {label_expr} AS label, n.globalTid AS tid,
               n.start AS start, n.[end] AS end_ns
        FROM {nvtx_table} n
        {label_join}
        WHERE n.[end] > n.start AND n.eventType IN (59, 60) {trim_where}
    """
    try:
        parents = adapter.execute(
            f"SELECT label, tid, start, end_ns FROM ({base}) WHERE label IS NOT NULL",
            trim_params,
        ).fetchall()
        children = adapter.execute(
            f"SELECT label, tid, start, end_ns FROM ({base}) "
            f"WHERE ({like_clause}) AND label IS NOT NULL",
            trim_params + like_params,
        ).fetchall()
    except DB_ERRORS as exc:
        return [
            {
                "error": (
                    f"host_sync_parent_ranges query failed: {exc}. "
                    "NVTX_EVENTS may be absent, or the profile schema is unexpected."
                ),
            }
        ]

    return _aggregate_matched(_sweep_containment(parents, children), limit)


def _sweep_containment(parents, children):
    """Yield ``(parent_label, child_label, sync_ns)`` for every strict containment.

    A child is credited to *every* range that contains it on the same tid
    (``parent.start <= child.start`` and ``parent.end >= child.end``), matching
    the SQL's multi-ancestor crediting — a child nested three deep credits all
    three enclosing ranges. The identical-coordinate pair is excluded, which is
    how the SQL drops a range matching itself (and, as in the SQL, a distinct
    range sharing the child's exact ``[start, end]``).

    An active set keyed on ``end`` — not a stack. eventType-60 (StartEnd) ranges
    are not guaranteed to nest, so ranges can cross; a stack-pop would evict the
    wrong range and misattribute a later child. Sorted by start, the active set
    holds ranges that have opened and not yet ended, so it is bounded by NVTX
    nesting depth and the sweep is O(n log n + matches).

    ``parents`` is every range in the profile and is held in memory (labels are
    near-unique per instance, so interning does not shrink it): ~1.25M rows is
    ~200MB on the largest profile seen. Acceptable — the in-engine SQL this
    replaces could not process that profile at all — but the reason the input is
    fetched as plain rows rather than streamed.
    """
    p_by_tid = defaultdict(list)
    c_by_tid = defaultdict(list)
    for label, tid, start, end_ns in parents:
        p_by_tid[tid].append((start, end_ns, label))
    for label, tid, start, end_ns in children:
        c_by_tid[tid].append((start, end_ns, label))

    matched = []
    for tid, clist in c_by_tid.items():
        plist = p_by_tid.get(tid)
        if not plist:
            continue
        plist.sort(key=lambda r: r[0])
        clist.sort(key=lambda r: r[0])
        active = []
        pi = 0
        npar = len(plist)
        for cs, ce, clabel in clist:
            while pi < npar and plist[pi][0] <= cs:
                active.append(plist[pi])
                pi += 1
            if active:
                # A range with end < the current (and every later) child.start
                # can contain none of them, so drop it for good.
                active = [p for p in active if p[1] >= cs]
            sync_ns = ce - cs
            for ps, pe, plabel in active:
                if pe >= ce and not (ps == cs and pe == ce):
                    matched.append((plabel, clabel, sync_ns))
    return matched


def _aggregate_matched(matched, limit):
    """Roll matched pairs up into the skill's output rows.

    Same contract as the SQL's parent_totals / child_totals / ranked_children:
    group by parent label, pick each parent's top child by aggregated sync time
    (tie-break: count desc, then label asc), order parents by sync_ns desc then
    label asc, and take the first ``limit``.
    """
    parent_n = defaultdict(int)
    parent_ns = defaultdict(int)
    child_n = defaultdict(int)
    child_ns = defaultdict(int)
    for plabel, clabel, sync_ns in matched:
        parent_n[plabel] += 1
        parent_ns[plabel] += sync_ns
        child_n[(plabel, clabel)] += 1
        child_ns[(plabel, clabel)] += sync_ns

    # Top child per parent: smallest of (-sync_ns, -count, label) == largest
    # sync_ns, largest count, alphabetically first — the SQL's rn=1 ordering.
    top_children = defaultdict(list)
    for (plabel, clabel), n in child_n.items():
        top_children[plabel].append((-child_ns[(plabel, clabel)], -n, clabel))
    top_child = {p: min(cands)[2] for p, cands in top_children.items()}

    ordered = sorted(parent_ns, key=lambda p: (-parent_ns[p], p))
    out = []
    for plabel in ordered[:limit]:
        sync_ns = int(parent_ns[plabel])
        out.append(
            {
                "parent_range": plabel,
                "n_syncs": parent_n[plabel],
                "sync_ns": sync_ns,
                "sync_ms": round(sync_ns / 1e6, 3),
                "top_child_label": top_child.get(plabel),
            }
        )
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
