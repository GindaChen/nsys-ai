"""
base.py — Skill dataclass and execution helpers.

A Skill is the minimum analyzable unit: SQL template + parameters + formatter.
"""

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

from ..connection import DB_ERRORS, SKILL_CACHE_MISS, cached_skill_rows, set_cached_skill_rows

_log = logging.getLogger(__name__)


def _copy_rows(rows):
    """Copy the row dicts so cache and caller do not share the top level.

    Shallow on purpose, and the limit is worth stating: nested values are still
    shared. Twelve of the twenty-one entries a build caches carry them —
    `gpu_idle_gaps.attribution`, `critical_path.breakdown`, the manifest's
    sections — so a consumer mutating one of those *in place* would reach the
    cache. No skill or consumer in this repo does that, which is why this is not
    a deep copy: deep-copying the manifest on every hit would cost more than the
    memo saves. If a consumer ever does mutate nested content, this is the line
    that has to change.
    """
    return [dict(r) if isinstance(r, dict) else r for r in rows]


def _params_key(resolved: dict) -> str:
    """Stable text for a parameter set, for use in a cache key.

    Values may be unhashable (a list of diagnostic dicts is passed through as a
    parameter), so this serialises rather than hashing a tuple. Measured at
    0.48ms across all 29 calls in a build, against 69ms saved.
    """
    import json

    return json.dumps(resolved, sort_keys=True, default=str)


# The activity-table placeholders a SQL template may use, mapped to the key
# ``ConnectionAdapter.resolve_activity_tables`` files the resolved name under
# and to the canonical (unversioned) name. Declared once here so the injection
# in ``Skill.execute`` and the "cannot run" check over it read the same list;
# they used to be one hand-written ``setdefault`` per entry, which made it easy
# to add a placeholder to the injection and forget the guard.
#
# The guard protects a subset, and saying which is the point of this note. It
# fires only for a placeholder that appears in a ``Skill.sql`` template, because
# that is what ``Skill.execute`` inspects — and only when the caller did not
# name the table itself, which ``Skill.execute`` skips as the caller's business.
# Measured over the registry: three of the seven — ``kernel_table``,
# ``runtime_table`` and ``memcpy_table`` — occur in a template; ``nvtx_table``,
# ``memset_table``, ``sync_table`` and ``sync_type_table`` occur in none, so for
# those four the guard never fires, at any profile, ever. Their uses are Python
# f-strings inside ``execute_fn`` bodies, which the guard has no way to read.
#
# The execute_fn paths now declare hard requirements on ``Skill.required_tables``
# where the whole skill cannot answer without them: ``sync_cost_analysis``
# requires both sync tables and ``host_sync_parent_ranges`` requires NVTX. The
# NVTX half of ``gc_impact`` remains optional because its runtime-derived rows
# are still useful without annotation. Likewise ``pipeline_bubble_metrics``
# keeps its partial result when memset is absent, and the memset checker inside
# ``root_cause_matcher`` returns an abstention that its parent filters rather
# than turning into a finding. This distinction is intentional: partial
# coverage is not the same as a skill-level abstention.
#
# ``test_the_table_guard_covers_only_sql_templates`` pins the split above, and
# the execute_fn abstention tests pin the declared requirements and the two
# legitimate partial-result cases.
ACTIVITY_TABLE_PLACEHOLDERS: dict[str, tuple[str, str]] = {
    "kernel_table": ("kernel", "CUPTI_ACTIVITY_KIND_KERNEL"),
    "runtime_table": ("runtime", "CUPTI_ACTIVITY_KIND_RUNTIME"),
    "nvtx_table": ("nvtx", "NVTX_EVENTS"),
    "memcpy_table": ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY"),
    "memset_table": ("memset", "CUPTI_ACTIVITY_KIND_MEMSET"),
    "sync_table": ("sync", "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"),
    "sync_type_table": ("sync_type", "ENUM_CUPTI_SYNC_TYPE"),
}


def abstain(reason: str, **detail) -> list[dict]:
    """Return the row a skill emits when it *cannot run*, with the reason why.

    An empty list is ambiguous: it reads identically whether the skill ran and
    found nothing worth reporting, or could not run at all because the profile
    lacks a table it needs. Raising is worse — callers such as
    ``EvidenceBuilder`` catch and log, so the failure leaves nothing a reader of
    the output can see.

    Where abstaining is visible is a caller that formats the rows rather than
    one that turns them into findings, and there is exactly one place that
    formats them: :meth:`Skill.format_rows` renders the reason as "not
    applicable to this profile". Both callers of it show the abstention for
    free — ``skill run`` (via :meth:`Skill.run`) and ``Agent.analyze``. It is
    *not* ``EvidenceBuilder``: its ``_invoke_to_findings`` drops abstention rows
    before ``to_findings_fn``, so an abstaining skill contributes no finding
    either. The measurement behind that claim is recorded once, beside the guard
    in :meth:`Skill.execute`, rather than copied here.

    Both cases matter for grounding. "This profile has no NVTX annotation, so
    layer attribution is unavailable" is a useful answer; silence is not, and
    silence that looks like a clean bill of health is actively misleading.

    Callers distinguish the three states by the ``_abstained`` marker:

    * ``[]``                      -> ran, nothing to report
    * ``[{"_abstained": True}]``  -> could not run, ``reason`` says why
    * anything else               -> ran, here are the rows
    """
    return [{"_abstained": True, "reason": reason, **detail}]


def is_abstention_row(row) -> bool:
    """True when a single row is an abstention marker.

    The list-level predicate below is the usual one; this exists for code that
    iterates rows, which otherwise re-implements the isinstance-and-key check
    inline and drifts.
    """
    return isinstance(row, dict) and row.get("_abstained") is True


def resolve_nvtx_table(conn) -> str | None:
    """The profile's NVTX table name, or None when it has none.

    Companion to :func:`requires_nvtx` for the skills that need the *name* and
    not only the guard — the payload decoders query the table directly. Both
    resolve through here so ``skills/builtins`` never re-inlines the lookup,
    which is what ``test_the_nvtx_guard_is_defined_once`` enforces: an earlier
    inlined copy used an exact name match and missed ``NVTX_EVENTS_V2``.
    """
    from ..connection import wrap_connection

    return wrap_connection(conn).resolve_activity_tables().get("nvtx")


def requires_nvtx(conn, *, needs: str = "Region attribution") -> list[dict] | None:
    """Abstain if ``conn`` has no NVTX table, else None so the caller proceeds.

    Six skills need this and each had its own copy of the resolve-and-abstain
    block. Keeping it in one place also keeps the *resolution* right: an exact
    name match misses versioned variants such as NVTX_EVENTS_V2 and the parquet
    backend's views, which is a bug that already shipped once.

    ``needs`` names what the caller cannot do, so the message says something
    true for that skill rather than a copy of a sibling's wording.
    """
    if resolve_nvtx_table(conn):
        return None
    return abstain(
        f"This profile has no NVTX_EVENTS table, so it carries no NVTX "
        f"annotation. {needs} needs annotated ranges — re-capture with NVTX "
        f"enabled, or annotate the workload, to use this skill."
    )


def requires_pushpop_nvtx(conn, *, needs: str = "Region attribution") -> list[dict] | None:
    """Abstain unless ``conn`` carries Push/Pop NVTX ranges, else None.

    Kernel attribution in this package is a per-thread nesting sweep
    (``nvtx_attribution._sort_merge_attribute`` and the cache's equivalent in
    ``parquet_cache``). That sweep is only valid for eventType 59
    (``NvtxPushPopRange``): NVIDIA maintains the push/pop range stack per CPU
    thread, so a thread's ranges are strictly nested by construction. eventType
    60 (``NvtxStartEndRange``) carries no such guarantee — NVIDIA documents that
    Start/End ranges "expose arbitrary concurrency (not just nesting)" and that
    "the start of a range can occur on a different thread than the end", which
    Nsight materialises as a single row with ``globalTid`` and ``endGlobalTid``
    disagreeing. Widening the ``eventType = 59`` filter would therefore not fix
    anything; it would feed non-nested intervals to a stack.

    So Start/End ranges are deliberately not attributed, and this guard makes
    that visible. Without it a 60-only profile produced an empty result and the
    formatter asked "are NVTX annotations present?" — a question whose answer is
    yes, and which ``profile_health_manifest`` answers with the region names on
    the very same profile, because ``Profile.aggregate_nvtx_ranges`` does not
    filter eventType.

    Known limitation, stated rather than half-solved: on a *mixed* profile the
    59-probe hits, this guard passes, and the 60 ranges are still dropped from
    attribution with no signal. Rank-ordering every range kind is not worth
    building for a case with no observed users: no committed profile fixture and
    no real capture available here holds a single eventType-60 row — a 3.5GB
    trace with 15.9M NVTX rows is 100% eventType 59 — and PyTorch annotates with
    push/pop. The Start/End fixtures the tests build are synthetic on purpose.

    NVTXT ranges (eventType 70/71, the text-import variants) are excluded too —
    the sweep filters ``eventType = 59`` and nothing else — so every message
    here says "Push/Pop (eventType 59)" rather than implying 59 is the only
    range kind Nsight records. Those profiles reach the final branch, which for
    that reason names the eventTypes it measured instead of describing them.

    Fails open. Several test databases create ``NVTX_EVENTS`` with no
    ``eventType`` column at all, and a schema surprise must not be reported to a
    user as "your annotation is the wrong kind".
    """
    missing = requires_nvtx(conn, needs=needs)
    if missing:
        return missing

    from ..connection import wrap_connection

    adapter = wrap_connection(conn)
    nvtx_table = resolve_nvtx_table(conn)

    def _has(event_type: int) -> bool | None:
        """Does a row of this eventType exist? None when the probe cannot run."""
        try:
            cur = adapter.execute(
                f"SELECT 1 FROM {nvtx_table} WHERE eventType = {int(event_type)} LIMIT 1"
            )
            return cur.fetchone() is not None
        except DB_ERRORS:
            return None

    # 59 first, deliberately. Measured on a 15.9M-row capture: the push/pop
    # probe returns on the first row in 0.05ms, while a miss costs a full scan
    # (0.48s), so probing 60 first would tax every healthy profile for nothing.
    # A GROUP BY census costs 2.3s and tells us no more than these two probes.
    has_pushpop = _has(59)
    if has_pushpop is None or has_pushpop:
        return None

    has_startend = _has(60)
    if has_startend:
        return abstain(
            f"This profile's NVTX annotation consists only of Start/End ranges "
            f"(eventType 60). {needs} maps kernels to ranges with a per-thread "
            f"nesting sweep, which is valid only for Push/Pop ranges "
            f"(eventType 59): those are kept on a per-thread stack and so are "
            f"strictly nested. Start/End ranges may begin and end on different "
            f"threads and may overlap arbitrarily, so they are not attributed. "
            f"Re-annotate with nvtxRangePush/nvtxRangePop "
            f"(torch.cuda.nvtx.range_push/range_pop, or the "
            f"torch.cuda.nvtx.range context manager) to use this skill.",
            nvtx_event_types="60",
        )

    # Neither kind the in-process NVTX range API produces is present. What the
    # rows actually *are* is now measured rather than asserted: an earlier
    # version of this branch told the user the table held "only marks,
    # categories or domain records, which are instants", having probed nothing
    # but 59 and 60. That is false for a profile imported from an .nvtxt file,
    # whose ranges land as eventType 70/71 and do have extent — a confident
    # wrong claim about the user's data, which is the one thing an abstention
    # must never be. So census the column and name what is there.
    #
    # The census is a third scan of a table the two misses above have already
    # scanned twice, and it runs only on a profile that is abstaining either
    # way, so no healthy profile pays for it: measured on the 15.9M-row capture
    # here, DISTINCT eventType costs 0.60s, while the guard on that same
    # profile — which hits on the first 59 row and never reaches this line —
    # takes 0.25ms.
    types = _distinct_event_types(adapter, nvtx_table)
    if not types:
        # None means the census could not run (fail open, as above). Empty means
        # an NVTX table with no rows, which is a different question and not this
        # guard's: callers already handle it in ways an abstention would make
        # worse — `code_attribution_candidates` returns a structured row
        # carrying the requested window and the skill's limitations, which is
        # more useful than a reason string.
        return None

    present = ", ".join(_describe_event_type(adapter, t) for t in types)
    return abstain(
        f"This profile's {nvtx_table} table holds no Push/Pop ranges "
        f"(eventType 59) — the only kind {needs.lower()} attributes a kernel "
        f"to — and no Start/End ranges (eventType 60) either. What it does "
        f"hold: {present}. Annotate the code you want attributed with "
        f"nvtxRangePush/nvtxRangePop (torch.cuda.nvtx.range_push/range_pop, or "
        f"the torch.cuda.nvtx.range context manager) so the ranges are "
        f"recorded as eventType 59.",
        nvtx_event_types=",".join(str(t) for t in types),
    )


def _distinct_event_types(adapter, nvtx_table: str) -> list[int] | None:
    """The eventTypes present in ``nvtx_table``, or None when it cannot be read.

    None and ``[]`` are different answers and both callers depend on it: None is
    "no eventType column, or the column is not readable as numbers" and must
    fail open, ``[]`` is "the table has no rows", which is a state this module
    deliberately says nothing about. NULL eventTypes are dropped rather than
    reported, so a table of nothing but NULLs reads as ``[]`` and is likewise
    left alone.

    ``ValueError``/``TypeError`` are caught alongside the driver errors because
    SQLite columns are dynamically typed: a fixture or an unusual export can put
    text in ``eventType``, and the whole point of this guard is that a schema
    surprise never becomes a claim about the user's annotation.
    """
    try:
        cur = adapter.execute(f"SELECT DISTINCT eventType FROM {nvtx_table}")
        return sorted({int(r[0]) for r in cur.fetchall() if r[0] is not None})
    except (*DB_ERRORS, ValueError, TypeError):
        return None


def _describe_event_type(adapter, event_type: int) -> str:
    """``"70 (NvtxtPushPopRange)"``, or just ``"70"`` when the name is unknown.

    The name comes from the profile's own ``ENUM_NSYS_EVENT_TYPE`` table, which
    is authoritative for the export in hand — 155 rows on the fixture here, of
    which this package hard-codes none. Missing table, missing row or a read
    error all degrade to the bare number: a name is a convenience, and inventing
    one from a built-in list that may not match this Nsight version would be the
    same class of mistake this branch exists to correct.
    """
    try:
        row = adapter.execute(
            "SELECT label FROM ENUM_NSYS_EVENT_TYPE WHERE id = ?", (int(event_type),)
        ).fetchone()
    except DB_ERRORS:
        return str(event_type)
    if not row or not row[0]:
        return str(event_type)
    return f"{event_type} ({row[0]})"


def is_abstention(rows: list[dict] | None) -> bool:
    """True when ``rows`` is a skill saying it could not run.

    Every consumer that would otherwise treat the row as data needs this:
    formatters index data columns that an abstention row does not have,
    ``to_findings_fn`` would mint a finding out of it, and the agent would
    cite it as evidence with no metrics behind it.
    """
    return bool(rows) and is_abstention_row(rows[0])


def _compute_interval_union(intervals: list[tuple[int, int]]) -> int:
    """Computes the total non-overlapping duration of a list of [start, end] intervals."""
    if not intervals:
        return 0
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    total_ns = 0
    current_start, current_end = sorted_intervals[0]

    for start, end in sorted_intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total_ns += current_end - current_start
            current_start, current_end = start, end

    total_ns += current_end - current_start
    return total_ns


def compute_profiler_overhead_ns(
    conn,
    *,
    trim_start_ns: int | None = None,
    trim_end_ns: int | None = None,
) -> int:
    """Compute the non-overlapping union of profiler-overhead intervals.

    Probes both ``profiler_overhead`` and ``PROFILER_OVERHEAD`` (the two
    table names Nsight uses across export versions). When ``trim_start_ns``
    / ``trim_end_ns`` are supplied, intervals are clipped to that window
    before the union so the result is bounded by the requested scope.
    Returns 0 if no overhead table exists or no intervals fall inside
    the window.

    Called both by :meth:`Skill.execute` (to inject ``overhead_ns`` for
    sub-skills) and by callers that need to realign the value to a
    later-determined analysis window.
    """
    from ..connection import wrap_connection

    adapter = wrap_connection(conn)
    for oh_table in ("profiler_overhead", "PROFILER_OVERHEAD"):
        try:
            conds = []
            params: list[object] = []
            if trim_start_ns is not None:
                conds.append("[end] >= ?")
                params.append(int(trim_start_ns))
            if trim_end_ns is not None:
                conds.append("start <= ?")
                params.append(int(trim_end_ns))
            where_clause = "WHERE " + " AND ".join(conds) if conds else ""
            cur = adapter.execute(
                f"SELECT start, [end] FROM {oh_table} {where_clause}",
                params,
            )
            rows = cur.fetchall()
            intervals = []
            for row in rows:
                s, e = int(row[0]), int(row[1])
                if trim_start_ns is not None:
                    s = max(s, int(trim_start_ns))
                if trim_end_ns is not None:
                    e = min(e, int(trim_end_ns))
                if s < e:
                    intervals.append((s, e))
            return _compute_interval_union(intervals)
        except DB_ERRORS:
            continue
    return 0


# Track connections that have already been indexed to avoid repeated work.
_indexed_connections: set[int] = set()

# Indexes to create on Nsight SQLite profiles for skill query performance.
# Uses ``_nsysai_`` prefix to avoid conflicts with upstream tables.
# Table names can vary between Nsight versions (e.g. *_KERNEL_V2/V3), so we
# resolve the actual table names from sqlite_master at runtime instead of
# hard-coding them here.


def ensure_indexes(conn: sqlite3.Connection) -> None:
    """Create performance indexes on the profile DB if they don't already exist.

    Delegates to the centralized :func:`~nsys_ai.indexing.ensure_performance_indexes`.
    """
    from ..indexing import ensure_performance_indexes

    ensure_performance_indexes(conn)


@dataclass
class SkillParam:
    """One parameter a skill accepts."""

    name: str
    description: str
    type: str = "str"  # str, int, float
    required: bool = False
    default: object = None


@dataclass
class Skill:
    """A self-contained GPU profile analysis skill.

    Attributes:
        name:        Short identifier (e.g. "top_kernels")
        title:       Human-readable title
        description: What this skill analyzes and why
        category:    One of: kernels, memory, nvtx, communication, system, utility
        sql:         SQL query template with {param} placeholders
        params:      Accepted parameters
        format_fn:   Optional function(rows) → formatted string
        tags:        Search tags for skill discovery
        execute_fn:  Optional Python callable(conn, **kwargs) → list[dict].
                     When set, used instead of sql for execution.
        required_tables: Activity-table resolver keys required before execution.
                         Used for execute_fn skills whose SQL is not visible to
                         the generic template guard.
    """

    name: str
    title: str
    description: str
    category: str
    sql: str = ""
    params: list[SkillParam] = field(default_factory=list)
    format_fn: Callable | None = None
    tags: list[str] = field(default_factory=list)
    execute_fn: Callable | None = None
    to_findings_fn: Callable | None = None
    required_tables: tuple[str, ...] = ()

    def execute(self, conn: sqlite3.Connection, **kwargs) -> list[dict]:
        """Run the skill against a connection.

        If ``execute_fn`` is set, delegates to it.  Otherwise runs the
        skill's SQL query against *conn* — unless one of the activity tables
        that query reads cannot be resolved on this profile, in which case it
        abstains instead of running a query that is certain to fail.

        Args:
            conn: SQLite connection to an Nsight profile database
            **kwargs: Parameter values (substituted into SQL template).
                      Special keys ``trim_start_ns`` and ``trim_end_ns``
                      trigger ``{trim_clause}`` substitution if present
                      in the SQL template.

        Returns:
            List of result rows as dicts, or the single-row abstention marker
            :func:`abstain` builds. Callers that treat rows as data must test
            for it with :func:`is_abstention`.
        """
        # Auto-create performance indexes (one-time per connection).
        ensure_indexes(conn)

        from ..connection import wrap_connection

        adapter = wrap_connection(conn)

        # SQL skills can infer required activity tables from their template,
        # but execute_fn skills build SQL in Python and need to declare the
        # same contract explicitly. Check before parameter resolution and
        # memoization so a missing table is visible as an abstention rather
        # than silence or an untyped database error.
        if self.required_tables:
            tables = adapter.resolve_activity_tables()
            missing = [
                canonical
                for _placeholder, (table_key, canonical) in ACTIVITY_TABLE_PLACEHOLDERS.items()
                if table_key in self.required_tables and not tables.get(table_key)
            ]
            if missing:
                names = ", ".join(sorted(set(missing)))
                return abstain(
                    f"This profile has no {names} table, so '{self.name}' cannot "
                    "run. Either the capture did not trace that activity kind, "
                    "or the export names it something this version does not "
                    "recognise as a variant.",
                    missing_tables=sorted(set(missing)),
                )

        # Apply parameter defaults and required checks for all skill types.
        # Start from the provided kwargs so we preserve any extra arguments.
        resolved: dict[str, object] = dict(kwargs)
        for p in self.params:
            if p.name in resolved:
                # Caller-supplied value wins over default.
                continue
            if p.default is not None:
                resolved[p.name] = p.default
            elif p.required:
                from ..exceptions import SkillParameterError

                raise SkillParameterError(
                    f"skill '{self.name}' requires parameter '{p.name}'",
                    skill_name=self.name,
                    parameter=p.name,
                )

        # Handle {trim_clause} injection before execute_fn so {overhead_ns} can be computed correctly
        trim_start = resolved.get("trim_start_ns")
        trim_end = resolved.get("trim_end_ns")
        if trim_start is not None and trim_end is not None and "{trim_clause}" in self.sql:
            resolved["trim_clause"] = (
                f"AND k.start >= {int(trim_start)} AND k.[end] <= {int(trim_end)}"
            )
        elif "{trim_clause}" in self.sql:
            # No trim requested — replace with empty string
            resolved["trim_clause"] = ""

        # Compute profiler overhead union duration dynamically. Delegates
        # to the shared ``compute_profiler_overhead_ns`` helper so callers
        # that need to realign overhead to a later-determined window
        # (e.g. ``profile_health_manifest`` after auto-trim) can reuse the
        # same logic without copy-pasting the table-probe and clipping.
        if "overhead_ns" not in resolved:
            overhead_ns = compute_profiler_overhead_ns(
                conn, trim_start_ns=trim_start, trim_end_ns=trim_end
            )
            if overhead_ns == 0:
                _log.debug("No profiler overhead data found (table absent or empty)")
            resolved["overhead_ns"] = overhead_ns

        # Memoize per connection. One EvidenceBuilder.build() reaches skills
        # both directly and through the health manifest / root-cause matcher,
        # which re-run some analyses already requested by the pipeline.
        #
        # The key includes the resolved parameters, not just the name. The
        # repeats are mostly NOT identical — gpu_idle_gaps is called at limit=1
        # and limit=5 in the same build, and those legitimately differ — so a
        # name-only key would hand one caller another's results and silently
        # change findings. Resolved rather than raw kwargs so an explicit
        # limit=5 and a defaulted limit=5 share an entry.
        cache_key = f"skill:{self.name}:{_params_key(resolved)}"
        cached = cached_skill_rows(conn, cache_key)
        if cached is not SKILL_CACHE_MISS:
            # Copy on the way out as well, so two readers never share the row
            # dicts themselves.
            return _copy_rows(cached)

        # Python-level skill: delegate to execute_fn with resolved params.
        if self.execute_fn is not None:
            rows = self.execute_fn(conn, **resolved)
            # Store a copy, not the list being returned. Copying only on read
            # leaves the first caller holding the cached object itself, so its
            # in-place edits reach everyone after it.
            set_cached_skill_rows(conn, cache_key, _copy_rows(rows))
            return rows

        # --- SQL Execution Path ---
        # Inject resolved activity table names for versioned-table support.
        # SQL templates use {kernel_table} etc. instead of hardcoding
        # CUPTI_ACTIVITY_KIND_KERNEL which may be _KERNEL_V2/_V3 in
        # newer Nsight Systems versions.
        tables = adapter.resolve_activity_tables()
        unresolved: list[str] = []
        for placeholder, (key, canonical) in ACTIVITY_TABLE_PLACEHOLDERS.items():
            if placeholder in resolved:
                # Caller named the table explicitly; that is their business.
                continue
            resolved[placeholder] = tables.get(key, canonical)
            if tables.get(key) is None and "{" + placeholder + "}" in self.sql:
                unresolved.append(canonical)
        if unresolved:
            # The template reads a table this profile does not have. Substituting
            # the canonical literal anyway — which is what the ``.get`` fallback
            # above does, and all this branch used to do — sends the skill into
            # ``sqlite3.OperationalError: no such table``. ``abstain`` is the
            # contract for "cannot run"; see its docstring.
            #
            # Where that is visible, precisely, because the obvious answer is
            # wrong: NOT in ``EvidenceBuilder``. Its ``_invoke_to_findings``
            # drops abstention rows before they reach ``to_findings_fn``, so an
            # abstaining skill contributes no finding — exactly as a raising one
            # did. Measured on a profile whose only memcpy table is the P2P
            # ``..._MEMCPY2``: 16 findings either way, none of them memcpy.
            # It shows up where a caller formats the rows rather than turning
            # them into findings: ``Skill.format_rows`` prints "not applicable
            # to this profile" with the reason, for both of its callers —
            # ``skill run`` and ``Agent.analyze``.
            # The reason to raise it here is the contract, not a change in what
            # the evidence pipeline emits.
            #
            # Only a placeholder the template actually uses counts: the fallbacks
            # are still written into ``resolved`` because ``str.format`` needs a
            # value for every key it is given, not only the ones in this template.
            names = ", ".join(sorted(set(unresolved)))
            return abstain(
                f"This profile has no {names} table, so '{self.name}' has "
                f"nothing to read. Either the capture did not trace that "
                f"activity kind, or the export names it something this version "
                f"does not recognise as a variant.",
                missing_tables=sorted(set(unresolved)),
            )

        # NVTX text resolution: handle both legacy (text column only)
        # and modern schemas (textId → StringIds lookup).
        if "{nvtx_text_expr}" in self.sql or "{nvtx_text_join}" in self.sql:
            has_textid = adapter.detect_nvtx_text_id()
            if has_textid:
                resolved.setdefault("nvtx_text_expr", "COALESCE(n.text, s2.value)")
                resolved.setdefault("nvtx_text_join", "LEFT JOIN StringIds s2 ON n.textId = s2.id")
            else:
                resolved.setdefault("nvtx_text_expr", "n.text")
                resolved.setdefault("nvtx_text_join", "")

        sql = self.sql.format(**resolved) if resolved else self.sql
        try:
            cursor = adapter.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            # Store here too. The lookup above sits over both branches, but the
            # store used to live only in the execute_fn branch, so the five
            # SQL-only skills built a key and then took a guaranteed miss on
            # every call — paying for the cache without ever using it.
            set_cached_skill_rows(conn, cache_key, _copy_rows(rows))
            return rows
        except Exception as exc:
            db_errors = (sqlite3.Error,)
            try:
                import duckdb

                db_errors += (duckdb.Error,)
            except ImportError:
                pass

            if not isinstance(exc, db_errors):
                raise

            from nsys_ai.exceptions import SkillExecutionError

            raise SkillExecutionError(f"SQL failed: {exc}", skill_name=self.name) from exc

    def format_rows(self, rows: list[dict]) -> str:
        """Format pre-computed rows as text (no re-execution).

        Abstention is rendered here rather than in each ``format_fn``. The
        contract is defined in this module, so honouring it belongs here too —
        and a per-skill formatter that forgot would crash on the missing data
        columns instead of printing the reason.
        """
        if is_abstention(rows):
            return f"{self.title}: not applicable to this profile.\n\n{rows[0]['reason']}"
        if self.format_fn:
            return self.format_fn(rows)
        return _default_format(self, rows)

    def run(self, conn: sqlite3.Connection, **kwargs) -> str:
        """Execute and format results as text."""
        return self.format_rows(self.execute(conn, **kwargs))

    def to_tool_description(self) -> str:
        """Return a one-paragraph description suitable for an LLM tool catalog."""
        params_desc = ""
        if self.params:
            params_desc = " Parameters: " + ", ".join(
                f"{p.name} ({p.type}, {'required' if p.required else 'optional'})"
                for p in self.params
            )
        return f"[{self.name}] {self.title}: {self.description}{params_desc}"


def _default_format(skill: Skill, rows: list[dict]) -> str:
    """Simple tabular format for skill results."""
    if not rows:
        return f"({skill.title}: no results)"

    cols = list(rows[0].keys())
    # Compute column widths
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("─" * widths[c] for c in cols)
    lines = [f"── {skill.title} ──", header, sep]
    for row in rows:
        lines.append("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))
    return "\n".join(lines)
