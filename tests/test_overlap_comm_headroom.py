"""Exposed NCCL must be reported as recoverable time in every overlap regime.

`overlap_breakdown` owns the communication bucket — `nccl_breakdown` and
`critical_path` both defer their comm headroom to it — so a coverage gap here
means the time is claimed by nobody and drops out of the ROI ranking entirely.

Its two diagnoses are threshold-shaped: one fires below 30% overlap, the other
below a 0.5 compute/NCCL ratio. A profile between them (issue #252) tripped
neither and reported no comm headroom at all, despite having exposed NCCL.
"""

import pytest

from nsys_ai.skills.builtins.overlap_breakdown import _to_findings


def _row(*, overlap_pct, compute_only_ms, nccl_only_ms, overlap_ms, total_ms):
    return {
        "device_id": 0,
        "overlap_pct": overlap_pct,
        "compute_only_ms": compute_only_ms,
        "nccl_only_ms": nccl_only_ms,
        "overlap_ms": overlap_ms,
        "idle_ms": 0.0,
        "total_ms": total_ms,
        "span_start_ns": 0,
        "span_end_ns": int(total_ms * 1e6),
    }


def _comm_headroom(findings):
    return [f.headroom_ms for f in findings if f.headroom_ms is not None]


# ── The regression this closes ─────────────────────────────────────────────


def test_moderate_overlap_balanced_profile_reports_its_exposed_nccl():
    """~40% overlap, compute and NCCL balanced — neither threshold fires.

    Before #252 this returned no finding carrying headroom, so 30ms of real
    recoverable time was invisible to the ranking.
    """
    row = _row(
        overlap_pct=40,
        compute_only_ms=100.0,
        nccl_only_ms=30.0,
        overlap_ms=20.0,
        total_ms=150.0,
    )
    findings = _to_findings([row], context={"profile_id": "p1"})

    claimed = _comm_headroom(findings)
    assert claimed == [30.0], f"expected the exposed NCCL to be claimed once, got {claimed}"

    f = next(f for f in findings if f.headroom_ms is not None)
    assert f.category == "communication"
    # It reports recoverable time without diagnosing a cause, so it must not
    # masquerade as a diagnosis.
    assert f.severity == "info"
    assert f.provenance["row_kind"] == "exposed_communication"


# ── Single-count discipline: exactly one finding may claim the ms ──────────


@pytest.mark.parametrize(
    "label,row",
    [
        (
            "low overlap",
            _row(overlap_pct=10, compute_only_ms=100.0, nccl_only_ms=50.0,
                 overlap_ms=5.0, total_ms=160.0),
        ),
        (
            "comm dominated",
            _row(overlap_pct=40, compute_only_ms=10.0, nccl_only_ms=50.0,
                 overlap_ms=20.0, total_ms=90.0),
        ),
        (
            "both fire at once",
            _row(overlap_pct=10, compute_only_ms=10.0, nccl_only_ms=50.0,
                 overlap_ms=5.0, total_ms=70.0),
        ),
        (
            "moderate overlap (the new path)",
            _row(overlap_pct=40, compute_only_ms=100.0, nccl_only_ms=30.0,
                 overlap_ms=20.0, total_ms=150.0),
        ),
    ],
)
def test_exposed_nccl_is_claimed_exactly_once(label, row):
    """Whichever finding fires, the same millisecond is never counted twice.

    Two findings can describe one inefficiency, so carrying the headroom on
    both would corrupt the opportunity ranking.
    """
    findings = _to_findings([row], context={"profile_id": "p1"})
    claimed = _comm_headroom(findings)
    assert len(claimed) <= 1, f"{label}: {len(claimed)} findings claimed headroom"
    if claimed:
        assert claimed[0] == row["nccl_only_ms"], (
            f"{label}: claimed {claimed[0]} but exposed NCCL is {row['nccl_only_ms']}"
        )


def test_total_comm_headroom_never_exceeds_exposed_nccl():
    """The issue's stated done-criterion, across a sweep of regimes."""
    for overlap_pct in (0, 10, 29, 30, 40, 60, 90):
        for compute in (5.0, 50.0, 200.0):
            row = _row(
                overlap_pct=overlap_pct,
                compute_only_ms=compute,
                nccl_only_ms=30.0,
                overlap_ms=20.0,
                total_ms=compute + 50.0,
            )
            total = sum(_comm_headroom(_to_findings([row], context={"profile_id": "p1"})))
            assert total <= 30.0, (
                f"overlap={overlap_pct}% compute={compute}: claimed {total} > exposed 30.0"
            )


# ── Not manufacturing findings out of noise ────────────────────────────────


def test_a_sliver_of_exposed_nccl_is_not_reported():
    """Calling a ~1% cost a bottleneck is the failure that erases trust.

    A healthy profile with a trace of unhidden collective must stay quiet.
    """
    row = _row(
        overlap_pct=95,
        compute_only_ms=1000.0,
        nccl_only_ms=1.0,  # 0.1% of the span
        overlap_ms=100.0,
        total_ms=1101.0,
    )
    assert _comm_headroom(_to_findings([row], context={"profile_id": "p1"})) == []


def test_no_nccl_at_all_reports_nothing():
    row = _row(
        overlap_pct=0,
        compute_only_ms=100.0,
        nccl_only_ms=0.0,
        overlap_ms=0.0,
        total_ms=100.0,
    )
    assert _comm_headroom(_to_findings([row], context={"profile_id": "p1"})) == []
