"""Top GPU kernels by total execution time."""

from nsys_ai.connection import DB_ERRORS

from ..base import Skill, SkillParam


def _format(rows):
    if not rows:
        return "(No kernels found)"
    lines = [
        "── Top GPU Kernels by Total Time ──",
        f"{'TC':<4s}  {'Kernel':<57s}  {'Count':>7s}  {'Total(ms)':>10s}  {'Avg(ms)':>9s}  {'Min(ms)':>9s}  {'Max(ms)':>9s}",
        "-" * 116,
    ]
    for r in rows:
        name = r["kernel_name"]
        if len(name) > 55:
            name = name[:52] + "..."

        tc_status = "[-]"
        if r.get("tc_eligible") is None:
            tc_status = "N/A "
        elif r.get("tc_eligible"):
            tc_status = "[✓]" if r.get("tc_active") else "[⚠️]"

        lines.append(
            f"{tc_status:<4s}  {name:<57s}  {r['invocations']:>7d}  {r['total_ms']:>10.2f}  "
            f"{r['avg_ms']:>9.2f}  {r['min_ms']:>9.2f}  {r['max_ms']:>9.2f}"
        )
    return "\n".join(lines)


def _execute(conn, **kwargs):
    limit = int(kwargs.get("limit", 15))
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")

    try:
        conn.execute("SELECT 1 FROM kernels LIMIT 1")
        has_kernels = True
    except DB_ERRORS:
        has_kernels = False

    params = []
    if has_kernels:
        trim_clause = ""
        if trim_start is not None and trim_end is not None:
            trim_clause = 'AND start >= ? AND "end" <= ?'
            params.extend([trim_start, trim_end])

        sql = f"""
            SELECT name AS kernel_name,
                   COUNT(*) AS invocations,
                   ROUND(SUM("end" - start) / 1e6, 2) AS total_ms,
                   ROUND(AVG("end" - start) / 1e6, 2) AS avg_ms,
                   ROUND(MIN("end" - start) / 1e6, 2) AS min_ms,
                   ROUND(MAX("end" - start) / 1e6, 2) AS max_ms,
                   SUM(is_tc_eligible) > 0 AS tc_eligible,
                   SUM(uses_tc) > 0 AS tc_active,
                   MIN(start) AS span_start_ns,
                   MAX("end") AS span_end_ns
            FROM kernels
            WHERE 1=1 {trim_clause}
            GROUP BY name
            ORDER BY total_ms DESC
            LIMIT {limit}
        """
        rows = conn.execute(sql, params).fetchall()
        cols = [
            "kernel_name",
            "invocations",
            "total_ms",
            "avg_ms",
            "min_ms",
            "max_ms",
            "tc_eligible",
            "tc_active",
            "span_start_ns",
            "span_end_ns",
        ]
        return [dict(zip(cols, r)) for r in rows]
    else:
        # Pure SQLite fallback (lacks TC eligibility analysis)
        from nsys_ai.connection import wrap_connection

        tables = wrap_connection(conn).resolve_activity_tables()
        kernel_table = tables.get("kernel", "CUPTI_ACTIVITY_KIND_KERNEL")

        trim_clause = ""
        if trim_start is not None and trim_end is not None:
            trim_clause = 'AND k.start >= ? AND k."end" <= ?'
            params.extend([trim_start, trim_end])

        sql = f"""
            SELECT COALESCE(d.value, s.value, 'kernel_' || CAST(k.shortName AS VARCHAR)) AS kernel_name,
                   COUNT(*) AS invocations,
                   ROUND(SUM(k."end" - k.start) / 1e6, 2) AS total_ms,
                   ROUND(AVG(k."end" - k.start) / 1e6, 2) AS avg_ms,
                   ROUND(MIN(k."end" - k.start) / 1e6, 2) AS min_ms,
                   ROUND(MAX(k."end" - k.start) / 1e6, 2) AS max_ms,
                   NULL AS tc_eligible,
                   NULL AS tc_active,
                   MIN(k.start) AS span_start_ns,
                   MAX(k."end") AS span_end_ns
            FROM {kernel_table} k
            LEFT JOIN StringIds s ON k.shortName = s.id
            LEFT JOIN StringIds d ON k.demangledName = d.id
            WHERE 1=1 {trim_clause}
            GROUP BY COALESCE(d.value, s.value, 'kernel_' || CAST(k.shortName AS VARCHAR))
            ORDER BY total_ms DESC
            LIMIT {limit}
        """
        rows = conn.execute(sql, params).fetchall()
        cols = [
            "kernel_name",
            "invocations",
            "total_ms",
            "avg_ms",
            "min_ms",
            "max_ms",
            "tc_eligible",
            "tc_active",
            "span_start_ns",
            "span_end_ns",
        ]
        return [dict(zip(cols, r)) if isinstance(r, tuple) else dict(r) for r in rows]


# ── Structured findings (v0.1) ──────────────────────────────────────

# ── Thresholds ──────────────────────────────────────────────────────
# Centralized so the firing policy is in one place. Confidence helpers
# below reuse these where the cutoff and the trigger are the same value.

# Top kernel must account for ≥ this fraction of top-K kernel time to
# fire ``top_kernel_dominates``.
_DOMINATES_PCT_THRESHOLD = 30.0

# Top-3 kernels must account for ≥ this fraction of top-K kernel time
# to fire ``top_kernels_concentrated``.
_CONCENTRATED_TOP3_PCT_THRESHOLD = 60.0

# Below this total per-kernel time, the cost of fixing TC routing is
# unlikely to be worth the optimization; suppress ``tc_eligible_inactive``
# to keep the noise floor low.
_TC_INACTIVE_MIN_TOTAL_MS = 10.0

# Variability findings require at least this many invocations to keep
# the max/avg ratio from being dominated by noise (first-call warmup,
# straggler tails, etc.).
_VARIABILITY_MIN_INVOCATIONS = 20

# Minimum max/avg duration ratio for ``top_kernel_high_variability`` to
# fire (5× ≈ heavy tail / bimodal distribution; below this, normal
# scheduling jitter accounts for the variance).
_VARIABILITY_MIN_RATIO = 5.0


_DOMINATES_EXPLANATION = (
    "One kernel accounts for a disproportionate fraction of total kernel "
    "time. Optimization leverage here is high — improving this kernel "
    "shifts step time directly."
)
_CONCENTRATED_EXPLANATION = (
    "The top-3 kernels concentrate most kernel-time mass. The workload "
    "has a focused hot path; optimization effort outside the top-K has "
    "low leverage."
)
_TC_INACTIVE_EXPLANATION = (
    "Kernel is eligible to use Tensor Cores (TC) but fell back to a "
    "non-TC code path. Common causes: dtype is FP32 instead of FP16/BF16, "
    "non-aligned matrix shapes, or a non-TC cuBLAS / cuBLASLt routing "
    "decision."
)
_HIGH_VARIABILITY_EXPLANATION = (
    "Kernel duration is highly variable (max/avg ratio above 5×). "
    "Common causes: input-shape variance (dynamic shapes), kernel "
    "selection switching across invocations, or stragglers when the "
    "kernel runs concurrently with other work."
)

_DOMINATES_ACTIONS = [
    "Profile the kernel internals (Nsight Compute) for memory-bound vs compute-bound",
    "Check whether a faster algorithm or library exists (FlashAttention, fused norms, cuBLASLt)",
    "Confirm dtype, tile shape, and Tensor Core eligibility for matmul kernels",
]
_CONCENTRATED_ACTIONS = [
    "Focus optimization on the top-3 — gains outside the head have low leverage",
    "Check whether the top-3 share a common kernel family (attention, GEMM, norm) — upgrading the family lifts all three",
]
_TC_INACTIVE_ACTIONS = [
    "Verify input dtype is FP16/BF16 (not FP32) for matmul kernels",
    "Confirm matrix dims are multiples of 8 (FP16) or 16 (BF16) on the inner axis",
    "Check whether the cuBLAS/cuBLASLt heuristic selected a Tensor Core kernel vs a SIMT fallback",
]
_HIGH_VARIABILITY_ACTIONS = [
    "Inspect input shapes across invocations — dynamic shapes can re-pick a kernel each call",
    "Check for concurrent kernels sharing the same SMs (compute-compute contention)",
    "Look for first-call warmup vs steady-state cost — exclude warmup from averages",
]

_DOMINATES_FP_NOTES = [
    "A single dominant kernel is not always a bug — workloads with one heavy GEMM are correctly characterized this way",
    "Short profiles can over-emphasize the largest kernel; multi-iteration windows are more reliable",
]
_CONCENTRATED_FP_NOTES = [
    "Concentration is expected in inference workloads with a small set of recurring blocks",
]
_TC_INACTIVE_FP_NOTES = [
    "tc_eligible can be unknown on the pure-SQLite fallback path (no cache); findings only fire when the analyzer is confident the kernel is TC-eligible",
    "Some matmul kernels are intentionally non-TC for numerical reasons (small-K, mixed-precision accumulation)",
]
_HIGH_VARIABILITY_FP_NOTES = [
    "Low invocation count (<20) makes max/avg ratios noisy; the finding fires only above the sample-size guard",
    "First-call vs steady-state — recompile, warmup, and lazy initialization can inflate max without indicating a problem",
]


def _safe_id(name: str) -> str:
    """Sanitize a kernel name for use inside a finding/selection id."""
    out = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return out[:64] or "unknown"


def _dominance_confidence(pct: float, invocations: int) -> float:
    """Higher pct + more samples → higher confidence."""
    if invocations < 10:
        return round(min(0.5, 0.2 + 0.03 * invocations), 3)
    # ~30% → 0.7; ~50% → 0.8; ~70% → 0.9
    base = 0.6 + 0.005 * max(pct - 30.0, 0.0)
    return round(min(0.95, base), 3)


def _concentrated_confidence(pct_top3: float, n_kernels: int) -> float:
    if n_kernels < 5:
        return 0.5
    return round(min(0.9, 0.5 + 0.005 * max(pct_top3 - 60.0, 0.0)), 3)


def _tc_inactive_confidence(total_ms: float, invocations: int) -> float:
    """Higher total_ms + more invocations → higher confidence."""
    if invocations < 5 or total_ms < 10.0:
        return 0.5
    return round(min(0.9, 0.6 + 0.003 * max(total_ms - 10.0, 0.0) + 0.004 * min(invocations, 50)), 3)


def _variability_confidence(ratio: float, invocations: int) -> float:
    if invocations < _VARIABILITY_MIN_INVOCATIONS:
        return 0.4  # below the guard — should not fire
    return round(min(0.9, 0.55 + 0.05 * max(ratio - _VARIABILITY_MIN_RATIO, 0.0)), 3)


def _span_from_row(r: dict) -> tuple[bool, int | None, int | None]:
    """Read ``span_start_ns`` / ``span_end_ns`` off a top_kernels row.

    Returns ``(has_span, start_ns_or_None, end_ns_or_None)``. The boolean
    is the authoritative source for whether to emit ``Finding.type="region"``
    or ``"highlight"`` and is reused by both the ``Finding`` and the
    ``TraceSelection`` constructors so the time-anchor decision is taken
    in one place. ``Finding.start_ns`` is typed ``int`` (required), so
    callers fall back to ``0`` on the no-span branch and rely on
    ``type="highlight"`` to signal "not time-anchored" downstream;
    ``TraceSelection`` accepts ``None`` directly.
    """
    s = r.get("span_start_ns")
    e = r.get("span_end_ns")
    if s is None or e is None:
        return False, None, None
    return True, int(s), int(e)


def _to_findings(rows: list[dict], *, context: dict | None = None) -> list:
    """Emit structured findings from top_kernels rows.

    Four finding types: a single kernel that dominates the top-K total
    time, a concentrated top-3, a Tensor-Core-eligible kernel that ran
    on a non-TC path, and a top kernel with high duration variability.
    Empty input or all-zero totals return ``[]`` (Trust Contract:
    silence over weak claims).
    """
    from nsys_ai.annotation import EvidenceRow, Finding, TraceSelection

    findings: list = []
    if not rows:
        return findings
    rows = [r for r in rows if "error" not in r]
    if not rows:
        return findings

    profile_id = (context or {}).get("profile_id", "unknown")
    total_top_ms = sum(float(r.get("total_ms", 0) or 0) for r in rows)
    if total_top_ms <= 0:
        return findings

    n_kernels = len(rows)
    top = rows[0]

    # Finding 1: single kernel dominates.
    top_total_ms = float(top.get("total_ms", 0) or 0)
    top_pct = 100.0 * top_total_ms / total_top_ms
    if top_pct >= _DOMINATES_PCT_THRESHOLD:
        kname = top.get("kernel_name", "<unknown>")
        invocations = int(top.get("invocations", 0) or 0)
        has_span, span_start_ns, span_end_ns = _span_from_row(top)
        finding_id = f"top_kernel_dominates_{_safe_id(kname)}"
        selection = TraceSelection(
            id=f"sel_{finding_id}",
            profile_id=profile_id,
            source="skill:top_kernels",
            start_ns=span_start_ns,
            end_ns=span_end_ns,
            label=f"Top kernel: {kname} ({top_pct:.1f}%)",
        )
        ev = EvidenceRow(
            id=f"ev_{finding_id}",
            source_skill="top_kernels",
            values={
                "kernel_name": kname,
                "total_ms": round(top_total_ms, 3),
                "invocations": invocations,
                "avg_ms": round(float(top.get("avg_ms", 0) or 0), 3),
                "pct_of_top": round(top_pct, 2),
            },
            units={"total_ms": "ms", "avg_ms": "ms", "pct_of_top": "percent"},
            selection_id=selection.id,
            provenance={"row_kind": "top_kernel_dominates"},
        )
        findings.append(
            Finding(
                type="region" if has_span else "highlight",
                label=f"Top kernel dominates ({top_pct:.1f}%): {kname[:48]}",
                start_ns=span_start_ns if has_span else 0,
                end_ns=span_end_ns,
                severity="warning",
                note=(
                    f"Kernel '{kname}' accounts for {top_pct:.1f}% of total "
                    f"top-{n_kernels} kernel time ({top_total_ms:.1f}ms over "
                    f"{invocations} invocations)."
                ),
                id=finding_id,
                category="compute",
                confidence=_dominance_confidence(top_pct, invocations),
                evidence=[ev],
                selection=selection,
                explanation=_DOMINATES_EXPLANATION,
                suggested_actions=list(_DOMINATES_ACTIONS),
                false_positive_notes=list(_DOMINATES_FP_NOTES),
                provenance={"skill": "top_kernels", "row_kind": "top_kernel_dominates"},
            )
        )

    # Finding 2: top-3 concentration.
    # Genuinely global — top-3 is an aggregate property of the top-K
    # distribution, not anchored to a time window. ``Finding.start_ns``
    # is required-int, so we use ``0`` as the "no time anchor" sentinel;
    # ``type="highlight"`` (vs ``"region"``) is the semantic signal to
    # downstream consumers that this is not a trace location.
    if n_kernels >= 3:
        top3_ms = sum(float(r.get("total_ms", 0) or 0) for r in rows[:3])
        top3_pct = 100.0 * top3_ms / total_top_ms
        if top3_pct >= _CONCENTRATED_TOP3_PCT_THRESHOLD:
            names = [r.get("kernel_name", "<unknown>") for r in rows[:3]]
            finding_id = "top_kernels_concentrated"
            selection = TraceSelection(
                id=f"sel_{finding_id}",
                profile_id=profile_id,
                source="skill:top_kernels",
                label=f"Top-3 concentrate {top3_pct:.0f}% of kernel time",
            )
            ev = EvidenceRow(
                id=f"ev_{finding_id}",
                source_skill="top_kernels",
                values={
                    "top3_kernels": names,
                    "top3_total_ms": round(top3_ms, 3),
                    "top3_pct": round(top3_pct, 2),
                    "n_kernels_considered": n_kernels,
                },
                units={"top3_total_ms": "ms", "top3_pct": "percent"},
                selection_id=selection.id,
                provenance={"row_kind": "top_kernels_concentrated"},
            )
            findings.append(
                Finding(
                    type="highlight",
                    label=f"Top-3 kernels = {top3_pct:.0f}% of kernel time",
                    start_ns=0,
                    severity="info",
                    note=(
                        f"Top-3 kernels concentrate {top3_pct:.1f}% of total "
                        f"top-{n_kernels} kernel time. Optimization leverage "
                        f"is in the head of the distribution."
                    ),
                    id=finding_id,
                    category="compute",
                    confidence=_concentrated_confidence(top3_pct, n_kernels),
                    evidence=[ev],
                    selection=selection,
                    explanation=_CONCENTRATED_EXPLANATION,
                    suggested_actions=list(_CONCENTRATED_ACTIONS),
                    false_positive_notes=list(_CONCENTRATED_FP_NOTES),
                    provenance={"skill": "top_kernels", "row_kind": "top_kernels_concentrated"},
                )
            )

    # Finding 3: Tensor Core eligible but inactive.
    for r in rows:
        tc_eligible = r.get("tc_eligible")
        tc_active = r.get("tc_active")
        # tc_eligible is None on the SQLite-fallback path — silence over false claim.
        if tc_eligible is None:
            continue
        if not tc_eligible or tc_active:
            continue
        total_ms = float(r.get("total_ms", 0) or 0)
        invocations = int(r.get("invocations", 0) or 0)
        if total_ms < _TC_INACTIVE_MIN_TOTAL_MS:
            continue
        kname = r.get("kernel_name", "<unknown>")
        has_span, span_start_ns, span_end_ns = _span_from_row(r)
        finding_id = f"tc_eligible_inactive_{_safe_id(kname)}"
        selection = TraceSelection(
            id=f"sel_{finding_id}",
            profile_id=profile_id,
            source="skill:top_kernels",
            start_ns=span_start_ns,
            end_ns=span_end_ns,
            label=f"TC eligible, inactive: {kname}",
        )
        ev = EvidenceRow(
            id=f"ev_{finding_id}",
            source_skill="top_kernels",
            values={
                "kernel_name": kname,
                "total_ms": round(total_ms, 3),
                "invocations": invocations,
                "tc_eligible": True,
                "tc_active": False,
            },
            units={"total_ms": "ms"},
            selection_id=selection.id,
            provenance={"row_kind": "tc_eligible_inactive"},
        )
        findings.append(
            Finding(
                type="region" if has_span else "highlight",
                label=f"TC eligible, inactive: {kname[:48]} ({total_ms:.0f}ms)",
                start_ns=span_start_ns if has_span else 0,
                end_ns=span_end_ns,
                severity="warning",
                note=(
                    f"Kernel '{kname}' is Tensor-Core eligible but ran in a "
                    f"non-TC code path across {invocations} invocations "
                    f"({total_ms:.1f}ms total)."
                ),
                id=finding_id,
                category="compute",
                confidence=_tc_inactive_confidence(total_ms, invocations),
                evidence=[ev],
                selection=selection,
                explanation=_TC_INACTIVE_EXPLANATION,
                suggested_actions=list(_TC_INACTIVE_ACTIONS),
                false_positive_notes=list(_TC_INACTIVE_FP_NOTES),
                provenance={"skill": "top_kernels", "row_kind": "tc_eligible_inactive"},
            )
        )

    # Finding 4: high variability on a top kernel.
    for r in rows:
        invocations = int(r.get("invocations", 0) or 0)
        if invocations < _VARIABILITY_MIN_INVOCATIONS:
            continue
        avg_ms = float(r.get("avg_ms", 0) or 0)
        max_ms = float(r.get("max_ms", 0) or 0)
        if avg_ms <= 0:
            continue
        ratio = max_ms / avg_ms
        if ratio < _VARIABILITY_MIN_RATIO:
            continue
        kname = r.get("kernel_name", "<unknown>")
        has_span, span_start_ns, span_end_ns = _span_from_row(r)
        finding_id = f"top_kernel_high_variability_{_safe_id(kname)}"
        selection = TraceSelection(
            id=f"sel_{finding_id}",
            profile_id=profile_id,
            source="skill:top_kernels",
            start_ns=span_start_ns,
            end_ns=span_end_ns,
            label=f"Variability ({ratio:.1f}x): {kname}",
        )
        ev = EvidenceRow(
            id=f"ev_{finding_id}",
            source_skill="top_kernels",
            values={
                "kernel_name": kname,
                "avg_ms": round(avg_ms, 3),
                "max_ms": round(max_ms, 3),
                "max_avg_ratio": round(ratio, 2),
                "invocations": invocations,
            },
            units={"avg_ms": "ms", "max_ms": "ms", "max_avg_ratio": "ratio"},
            selection_id=selection.id,
            provenance={"row_kind": "top_kernel_high_variability"},
        )
        findings.append(
            Finding(
                type="region" if has_span else "highlight",
                label=f"High variability ({ratio:.1f}x): {kname[:48]}",
                start_ns=span_start_ns if has_span else 0,
                end_ns=span_end_ns,
                severity="info",
                note=(
                    f"Kernel '{kname}' max={max_ms:.1f}ms vs avg={avg_ms:.1f}ms "
                    f"({ratio:.1f}× ratio over {invocations} invocations) — "
                    f"possible shape variance or stragglers."
                ),
                id=finding_id,
                category="compute",
                confidence=_variability_confidence(ratio, invocations),
                evidence=[ev],
                selection=selection,
                explanation=_HIGH_VARIABILITY_EXPLANATION,
                suggested_actions=list(_HIGH_VARIABILITY_ACTIONS),
                false_positive_notes=list(_HIGH_VARIABILITY_FP_NOTES),
                provenance={"skill": "top_kernels", "row_kind": "top_kernel_high_variability"},
            )
        )

    return findings


SKILL = Skill(
    name="top_kernels",
    title="Top GPU Kernels by Total Time",
    description=(
        "Lists the heaviest GPU kernels ranked by cumulative execution time. "
        "Use this to identify hotspots. TC column shows Tensor Core usage "
        "(✓=Active, ⚠️=Eligible but Fallback, -=Ineligible)."
    ),
    category="kernels",
    execute_fn=_execute,
    params=[SkillParam("limit", "Max number of kernels to return", "int", False, 15)],
    format_fn=_format,
    to_findings_fn=_to_findings,
    tags=["hotspot", "kernel", "duration", "performance", "top", "tensor_core"],
)
