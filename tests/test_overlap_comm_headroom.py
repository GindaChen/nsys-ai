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


def test_real_capture_window_that_previously_reported_nothing():
    """Numbers taken from a real capture, not invented.

    `mfu_h1002_nsys.sqlite` device 1, a 1.34s window: 30.8% overlap with
    compute vastly exceeding NCCL. It clears the low-overlap threshold by
    0.8 points and is nowhere near communication-dominated, so before this
    change 31.19ms of recoverable time was claimed by nobody.

    Scanning that capture and its postfix counterpart at two window sizes
    turned up 37 windows in this regime, so it is the normal shape of a
    well-overlapped training step rather than a contrived corner.
    """
    row = _row(
        overlap_pct=30.8,
        compute_only_ms=1251.17,
        nccl_only_ms=31.19,
        overlap_ms=13.9,
        total_ms=1341.3,
    )
    findings = _to_findings([row], context={"profile_id": "real"})
    assert _comm_headroom(findings) == [31.19]


def test_improving_overlap_does_not_silence_the_report():
    """Crossing the 30% threshold must not make recoverable time vanish.

    The same capture has windows at 24-27% overlap that fire `low_overlap`
    and report their exposed NCCL. If improving overlap past 30% dropped the
    report to nothing, the tool would answer an optimisation by going quiet —
    and a before/after diff would read the improvement as headroom disappearing
    rather than shrinking.
    """
    below = _row(overlap_pct=26.6, compute_only_ms=1251.17, nccl_only_ms=31.19,
                 overlap_ms=13.9, total_ms=1341.3)
    above = _row(overlap_pct=30.8, compute_only_ms=1251.17, nccl_only_ms=31.19,
                 overlap_ms=13.9, total_ms=1341.3)
    assert _comm_headroom(_to_findings([below], context={"profile_id": "p"})) == [31.19]
    assert _comm_headroom(_to_findings([above], context={"profile_id": "p"})) == [31.19]


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
