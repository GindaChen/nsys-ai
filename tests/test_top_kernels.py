"""Tests for top_kernels structured findings.

The structured-findings path (``_to_findings``) is exercised here with
synthetic row dicts; it has no DB dependency and matches the shape the
``_execute`` SQL emits (after the v0.1 column upgrade that added
``span_start_ns`` / ``span_end_ns``).
"""

import json

from nsys_ai.annotation import EvidenceRow, Finding, TraceSelection
from nsys_ai.skills.builtins.top_kernels import SKILL


# ── Row factory ─────────────────────────────────────────────────────


def _row(
    *,
    kernel_name: str = "ampere_fp16_s16816gemm",
    invocations: int = 100,
    total_ms: float = 100.0,
    avg_ms: float = 1.0,
    min_ms: float = 0.9,
    max_ms: float = 1.2,
    tc_eligible: bool | None = True,
    tc_active: bool | None = True,
    span_start_ns: int = 1_000_000,
    span_end_ns: int = 9_000_000,
) -> dict:
    return {
        "kernel_name": kernel_name,
        "invocations": invocations,
        "total_ms": total_ms,
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "tc_eligible": tc_eligible,
        "tc_active": tc_active,
        "span_start_ns": span_start_ns,
        "span_end_ns": span_end_ns,
    }


def _find(findings, prefix):
    return [f for f in findings if f.id and f.id.startswith(prefix)]


# ── Finding 1 — top_kernel_dominates ────────────────────────────────


def test_dominates_fires_on_single_heavy_kernel():
    # Dominant kernel at 60% of top-K total; even, small tail.
    rows = [
        _row(kernel_name="big_gemm", total_ms=600.0, invocations=200),
        _row(kernel_name="small_a", total_ms=200.0, invocations=200),
        _row(kernel_name="small_b", total_ms=200.0, invocations=200),
    ]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "p"})

    dom = _find(findings, "top_kernel_dominates_")
    assert dom, f"expected dominates finding, got {[f.label for f in findings]}"
    f = dom[0]
    assert f.category == "compute"
    assert f.severity == "warning"
    assert "big_gemm" in f.label
    assert 0.0 <= f.confidence <= 1.0
    assert f.explanation and "leverage" in f.explanation
    assert f.suggested_actions and f.false_positive_notes
    assert f.provenance == {"skill": "top_kernels", "row_kind": "top_kernel_dominates"}

    assert isinstance(f.selection, TraceSelection)
    assert f.selection.profile_id == "p"
    assert f.selection.source == "skill:top_kernels"
    assert f.selection.start_ns == 1_000_000
    assert f.selection.end_ns == 9_000_000

    assert f.evidence and isinstance(f.evidence[0], EvidenceRow)
    ev = f.evidence[0]
    assert ev.source_skill == "top_kernels"
    assert ev.values["kernel_name"] == "big_gemm"
    assert ev.values["total_ms"] == 600.0
    assert ev.values["pct_of_top"] == 60.0
    assert ev.units["pct_of_top"] == "percent"
    assert ev.selection_id == f.selection.id


def test_dominates_does_not_fire_when_evenly_distributed():
    # Equal sizes across 5 kernels: top kernel is only 20%.
    rows = [_row(kernel_name=f"k{i}", total_ms=100.0) for i in range(5)]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "p"})
    assert not _find(findings, "top_kernel_dominates_"), (
        f"unexpected dominates finding on even distribution: {[f.label for f in findings]}"
    )


# ── Finding 2 — top_kernels_concentrated ────────────────────────────


def test_concentrated_fires_when_top3_is_majority():
    # Top-3 are 80% of the total.
    rows = [
        _row(kernel_name="a", total_ms=400.0),
        _row(kernel_name="b", total_ms=300.0),
        _row(kernel_name="c", total_ms=100.0),
        *[_row(kernel_name=f"tail_{i}", total_ms=10.0) for i in range(20)],
    ]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "p"})

    conc = _find(findings, "top_kernels_concentrated")
    assert conc, f"expected concentrated finding, got {[f.label for f in findings]}"
    f = conc[0]
    assert f.category == "compute"
    assert f.severity == "info"
    ev = f.evidence[0]
    assert ev.values["top3_kernels"] == ["a", "b", "c"]
    # 800 / (400+300+100 + 20*10) = 800 / 1000 = 80%
    assert ev.values["top3_pct"] == 80.0
    assert ev.values["n_kernels_considered"] == 23


def test_concentrated_does_not_fire_when_tail_is_heavy():
    # Top-3 at 30% of total — long heavy tail.
    rows = [
        *[_row(kernel_name=f"k{i}", total_ms=100.0) for i in range(3)],
        *[_row(kernel_name=f"tail_{i}", total_ms=100.0) for i in range(7)],
    ]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "p"})
    assert not _find(findings, "top_kernels_concentrated"), (
        f"unexpected concentrated finding: {[f.label for f in findings]}"
    )


# ── Finding 3 — tc_eligible_inactive ────────────────────────────────


def test_tc_inactive_fires_on_eligible_inactive_kernel():
    rows = [
        _row(kernel_name="eligible_inactive", total_ms=50.0,
             tc_eligible=True, tc_active=False, invocations=100),
        _row(kernel_name="active", total_ms=40.0,
             tc_eligible=True, tc_active=True),
    ]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "p"})

    tc = _find(findings, "tc_eligible_inactive_")
    assert tc, f"expected TC-inactive finding, got {[f.label for f in findings]}"
    f = tc[0]
    assert "eligible_inactive" in f.label
    assert f.category == "compute"
    assert f.severity == "warning"
    ev = f.evidence[0]
    assert ev.values["tc_eligible"] is True
    assert ev.values["tc_active"] is False
    assert ev.values["total_ms"] == 50.0


def test_tc_unknown_does_not_fire_on_sqlite_fallback():
    # SQLite fallback path emits tc_eligible=None — must NOT fire (Trust
    # Contract: silence over false claim).
    rows = [
        _row(kernel_name="sqlite_path", total_ms=100.0,
             tc_eligible=None, tc_active=None),
    ]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "p"})
    assert not _find(findings, "tc_eligible_inactive_"), (
        f"unexpected TC finding on unknown TC eligibility: {[f.label for f in findings]}"
    )


# ── Finding 4 — high variability ────────────────────────────────────


def test_high_variability_fires_on_bimodal_top_kernel():
    rows = [
        _row(kernel_name="bimodal", invocations=180, avg_ms=1.0, max_ms=8.0,
             total_ms=180.0, tc_eligible=False),
    ]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "p"})

    var = _find(findings, "top_kernel_high_variability_")
    assert var, f"expected variability finding, got {[f.label for f in findings]}"
    f = var[0]
    assert f.category == "compute"
    assert f.severity == "info"
    ev = f.evidence[0]
    assert ev.values["max_avg_ratio"] == 8.0
    assert ev.units["max_avg_ratio"] == "ratio"


def test_high_variability_does_not_fire_below_sample_size_guard():
    # max/avg ratio is 10× but invocations=5 — below the guard, no fire.
    rows = [
        _row(kernel_name="few_calls", invocations=5, avg_ms=1.0, max_ms=10.0,
             total_ms=5.0, tc_eligible=False),
    ]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "p"})
    assert not _find(findings, "top_kernel_high_variability_"), (
        f"unexpected variability finding below guard: {[f.label for f in findings]}"
    )


# ── Empty / no-fire / JSON round-trip / context ─────────────────────


def test_empty_input_returns_no_findings():
    assert SKILL.to_findings_fn([], context={"profile_id": "p"}) == []


def test_all_zero_total_ms_returns_no_findings():
    rows = [_row(kernel_name="zero", total_ms=0.0, invocations=0)]
    assert SKILL.to_findings_fn(rows, context={"profile_id": "p"}) == []


def test_findings_round_trip_through_json():
    rows = [
        _row(kernel_name="big_gemm", total_ms=600.0, invocations=200,
             avg_ms=3.0, max_ms=18.0),
        _row(kernel_name="medium", total_ms=200.0),
        _row(kernel_name="small", total_ms=200.0),
    ]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "/tmp/p.sqlite"})
    assert findings
    for f in findings:
        d = f.to_dict()
        # selection / evidence get nested-dataclass treatment
        if d.get("selection") is not None:
            assert isinstance(d["selection"], dict)
        if d.get("evidence") is not None:
            assert isinstance(d["evidence"], list)
        restored = Finding.from_dict(json.loads(json.dumps(d)))
        assert restored.id == f.id
        assert restored.category == "compute"
        if f.selection is not None:
            assert isinstance(restored.selection, TraceSelection)
            assert restored.selection.profile_id == "/tmp/p.sqlite"
        if f.evidence:
            assert isinstance(restored.evidence[0], EvidenceRow)


def test_findings_without_context_use_unknown_profile_id():
    rows = [
        _row(kernel_name="big_gemm", total_ms=600.0, invocations=200),
        _row(kernel_name="small_a", total_ms=200.0),
        _row(kernel_name="small_b", total_ms=200.0),
    ]
    findings = SKILL.to_findings_fn(rows)
    assert findings
    dom = _find(findings, "top_kernel_dominates_")
    assert dom and dom[0].selection.profile_id == "unknown"


def test_skill_registers_to_findings_fn():
    assert SKILL.to_findings_fn is not None
