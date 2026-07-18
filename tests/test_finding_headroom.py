"""Tests for Finding.headroom_ms and opportunity-based ranking (#190)."""

from nsys_ai.annotation import Finding, rank_findings
from nsys_ai.loop_state import _normalize_findings


def _f(label, *, severity="info", headroom_ms=None, confidence=None, type="region"):
    return Finding(
        type=type,
        label=label,
        start_ns=0,
        end_ns=1,
        severity=severity,
        headroom_ms=headroom_ms,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Field round-trip
# ---------------------------------------------------------------------------


def test_headroom_serializes_when_set():
    f = _f("x", headroom_ms=12.5)
    d = f.to_dict()
    assert d["headroom_ms"] == 12.5
    assert Finding.from_dict(d).headroom_ms == 12.5


def test_headroom_dropped_when_none():
    d = _f("x").to_dict()
    assert "headroom_ms" not in d  # legacy JSON stays compact
    assert Finding.from_dict(d).headroom_ms is None


# ---------------------------------------------------------------------------
# rank_findings (Finding objects — the evidence_builder path)
# ---------------------------------------------------------------------------


def test_large_headroom_outranks_high_severity():
    """The issue's example: a low-severity finding with large headroom beats a
    dramatic-looking (critical) finding with little room to improve."""
    dramatic = _f("critical but little upside", severity="critical", headroom_ms=2.0)
    opportunity = _f("info but big upside", severity="info", headroom_ms=200.0)
    ranked = rank_findings([dramatic, opportunity])
    assert [f.label for f in ranked] == [
        "info but big upside",
        "critical but little upside",
    ]


def test_findings_without_headroom_sort_last():
    a = _f("no headroom A")
    b = _f("has headroom", headroom_ms=50.0)
    c = _f("no headroom B")
    ranked = rank_findings([a, b, c])
    assert ranked[0].label == "has headroom"
    # The two headroom-less findings keep their original relative order.
    assert [f.label for f in ranked[1:]] == ["no headroom A", "no headroom B"]


def test_no_headroom_anywhere_is_unchanged():
    """When nothing carries a headroom, ranking is a no-op (stable order)."""
    items = [_f("first", severity="info"), _f("second", severity="critical")]
    ranked = rank_findings(items)
    assert [f.label for f in ranked] == ["first", "second"]


def test_rank_findings_empty():
    assert rank_findings([]) == []


# ---------------------------------------------------------------------------
# _normalize_findings (dict-based — the guided-loop path)
# ---------------------------------------------------------------------------


def test_normalize_findings_opportunity_first():
    findings = [
        {"label": "crit small", "severity": "critical", "type": "region", "headroom_ms": 3.0},
        {"label": "info big", "severity": "info", "type": "region", "headroom_ms": 300.0},
    ]
    ranked = _normalize_findings(findings)
    assert [f["label"] for f in ranked] == ["info big", "crit small"]


def test_normalize_findings_legacy_when_no_headroom():
    """No headroom present -> legacy severity heuristic order is preserved."""
    findings = [
        {"label": "info", "severity": "info", "type": "region"},
        {"label": "critical", "severity": "critical", "type": "region"},
    ]
    ranked = _normalize_findings(findings)
    assert [f["label"] for f in ranked] == ["critical", "info"]


# ---------------------------------------------------------------------------
# Producers populate headroom from their existing recoverable-ms evidence
# ---------------------------------------------------------------------------


def test_gpu_idle_gaps_headroom():
    from nsys_ai.skills.builtins.gpu_idle_gaps import _to_findings

    rows = [
        {  # summary row
            "_summary": True, "pct_of_profile": 20, "gpu_id": 0,
            "profile_start_ns": 0, "profile_end_ns": 100_000_000,
            "total_idle_ms": 40.0, "gap_count": 3,
        },
        {  # a single 5ms gap
            "gap_ns": 5_000_000, "start_ns": 10_000_000, "end_ns": 15_000_000,
            "deviceId": 0, "streamId": 7,
        },
    ]
    findings = _to_findings(rows, context={"profile_id": "p"})
    summary = next(f for f in findings if "Summary" in f.label)
    gap = next(f for f in findings if "Gap" in f.label)
    assert summary.headroom_ms == 40.0  # total recoverable idle
    assert gap.headroom_ms == 5.0  # the gap duration in ms


def test_overlap_breakdown_headroom_is_exposed_nccl():
    from nsys_ai.skills.builtins.overlap_breakdown import _to_findings

    rows = [{
        "nccl_only_ms": 30.0, "overlap_ms": 5.0, "compute_only_ms": 10.0,
        "overlap_pct": 14, "total_ms": 45.0,
        "span_start_ns": 0, "span_end_ns": 45_000_000, "device_id": 0,
    }]
    findings = _to_findings(rows, context={"profile_id": "p"})
    assert findings  # low-overlap and comm-dominated both fire
    for f in findings:
        assert f.headroom_ms == 30.0  # exposed (non-overlapped) NCCL


def test_iteration_timing_headroom_is_slack_over_median():
    from nsys_ai.skills.builtins.iteration_timing import _to_findings

    rows = [
        {"iteration": 1, "duration_ms": 10.0, "gpu_start_ns": 0,
         "gpu_end_ns": 10_000_000, "kernel_count": 5},
        {"iteration": 2, "duration_ms": 10.0, "gpu_start_ns": 10_000_000,
         "gpu_end_ns": 20_000_000, "kernel_count": 5},
        {"iteration": 3, "duration_ms": 40.0, "gpu_start_ns": 20_000_000,
         "gpu_end_ns": 60_000_000, "kernel_count": 5},
    ]
    findings = _to_findings(rows)
    assert len(findings) == 1  # only iter 3 exceeds 1.5x median
    assert findings[0].headroom_ms == 30.0  # 40ms - median(10ms)


def test_nccl_variability_headroom_is_straggler_slack():
    from nsys_ai.skills.builtins.nccl_breakdown import _to_findings

    rows = [{
        "type": "allreduce", "pct": 50.0, "total_ms": 100.0,
        "avg_ms": 5.0, "max_ms": 20.0, "count": 10, "stream_id": 7,
        "device_id": 0, "span_start_ns": 0, "span_end_ns": 100_000_000,
    }]
    findings = _to_findings(rows, context={"profile_id": "p"})
    var = next(f for f in findings if "Variability" in f.label)
    assert var.headroom_ms == 15.0  # max(20) - avg(5)


# ---------------------------------------------------------------------------
# region_mfu speed-of-light headroom (Phase 3) + formatter fix
# ---------------------------------------------------------------------------


def _mfu_result(mfu_union, union_s=0.1):
    return {
        "name": "attn", "matched_text": "attn", "source": "nvtx",
        "device_id": 0, "device_ids": [0],
        "wall_time_s": union_s, "gpu_kernel_sum_s": union_s, "gpu_kernel_union_s": union_s,
        "mfu_pct_kernel_union": mfu_union,
        "achieved_tflops_kernel_union": 100.0, "peak_tflops": 300.0,
    }


def test_sol_headroom_formula():
    from nsys_ai.skills.builtins.region_mfu import _sol_headroom_ms

    # 100ms union at 40% MFU -> 60ms recoverable at peak.
    assert _sol_headroom_ms(_mfu_result(40.0, union_s=0.1)) == 60.0
    # No MFU (no FLOPs supplied) -> no headroom, not a bogus zero.
    assert _sol_headroom_ms(_mfu_result(0.0)) is None
    assert _sol_headroom_ms({"gpu_kernel_union_s": 0.1}) is None


def test_region_mfu_emits_headroom_finding():
    from nsys_ai.skills.builtins.region_mfu import _to_findings

    findings = _to_findings([_mfu_result(30.0, union_s=0.1)], context={"profile_id": "p"})
    assert len(findings) == 1
    f = findings[0]
    assert f.headroom_ms == 70.0  # 100ms * (1 - 0.30)
    assert f.category == "compute"
    assert f.severity == "warning"  # low MFU is a real opportunity


def test_region_mfu_no_finding_without_flops():
    from nsys_ai.skills.builtins.region_mfu import _to_findings

    assert _to_findings([_mfu_result(0.0)]) == []
    assert _to_findings([{"error": {"code": "no_flops"}}]) == []


def test_region_mfu_format_renders_ms_not_zeros():
    """Regression: the old formatter read non-existent timing.*/mfu.* keys and
    printed zeros; it must now render the real flat, seconds-based values."""
    from nsys_ai.skills.builtins.region_mfu import _format

    text = _format([_mfu_result(40.0, union_s=0.05)])  # 50ms union
    assert "50.00ms" in text  # kernel union rendered from seconds
    assert "40.0%" in text  # MFU rendered
    assert "SOL headroom" in text
