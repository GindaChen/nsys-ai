"""Turn deterministic kernel regressions into candidate-profile findings.

The diff engine answers a pair question. This module closes that comparison
by projecting only the regressions onto the candidate profile as ordinary
``Finding`` objects. It deliberately accepts an in-memory
``ProfileDiffSummary`` and performs no profile or session I/O, so callers can
keep the diff calculation and session publication as separate boundaries.
"""

from __future__ import annotations

import hashlib

from .annotation import DiffLineage, Finding, TraceSelection, rank_findings
from .diff import KernelDiff, ProfileDiffSummary


def _finding_id(summary: ProfileDiffSummary, kernel: KernelDiff, rank: int) -> str:
    payload = f"{summary.diff_id}\0{rank}\0{kernel.key}".encode()
    return f"finding_diff_regression_{hashlib.sha256(payload).hexdigest()[:16]}"


def _selection(summary: ProfileDiffSummary, kernel: KernelDiff, rank: int) -> TraceSelection:
    """Return the diff selection, with a deterministic defensive fallback."""
    if kernel.selection is not None:
        return kernel.selection
    payload = f"{summary.diff_id}\0{kernel.key}\0{rank}".encode()
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return TraceSelection(
        id=f"sel_diff_regression_{digest}",
        profile_id=summary.after.profile_id,
        source="diff",
        gpu_ids=[summary.after.gpu] if summary.after.gpu is not None else None,
        label=f"{kernel.name} {kernel.delta_ns / 1e6:+.2f}ms",
    )


def _suggested_action(kernel: KernelDiff) -> str:
    """Explain the deterministic decomposition that a user can verify."""
    if kernel.classification == "new":
        return (
            f"Identify what introduced {kernel.name!r}; it is absent from the "
            "baseline and contributes new GPU work."
        )
    if kernel.delta_count > 0 and kernel.delta_avg_ns <= 0:
        return (
            f"Investigate why call count rose from {kernel.before_count} to "
            f"{kernel.after_count} for {kernel.name!r}; per-call time did not "
            "increase, so inspect loop or batching structure."
        )
    if kernel.delta_count == 0 and kernel.delta_avg_ns > 0:
        return (
            f"Compare shapes, dtypes, and occupancy for {kernel.name!r}; call "
            f"count stayed at {kernel.after_count}, while per-call time rose by "
            f"{kernel.delta_avg_ns / 1e3:.2f}us."
        )
    if kernel.delta_count != 0 and kernel.delta_avg_ns != 0:
        return (
            f"Separate call-frequency cost from per-call cost for {kernel.name!r}: "
            f"count changed by {kernel.delta_count:+d} and average time changed by "
            f"{kernel.delta_avg_ns / 1e3:+.2f}us before optimizing."
        )
    return (
        f"Inspect the workload path for {kernel.name!r}; aggregate GPU time rose "
        f"by {kernel.delta_ns / 1e6:.2f}ms without a distinct count or average-time shift."
    )


def _finding_note(kernel: KernelDiff) -> str:
    return (
        f"Candidate GPU time is {kernel.after_total_ns / 1e6:.3f}ms versus "
        f"{kernel.before_total_ns / 1e6:.3f}ms in the baseline "
        f"({kernel.delta_ns / 1e6:+.3f}ms); calls "
        f"{kernel.before_count} -> {kernel.after_count}, average "
        f"{kernel.before_avg_ns / 1e3:.3f}us -> {kernel.after_avg_ns / 1e3:.3f}us."
    )


def findings_from_diff(summary: ProfileDiffSummary) -> list[Finding]:
    """Mint ranked, candidate-anchored findings from per-kernel regressions.

    ``step_time_delta_ms`` and the diff verdict are intentionally not turned
    into a whole-run finding. A proposal needs a concrete trace selection, so
    only the already-ranked ``top_regressions`` are projected. An empty list
    means that this pair surfaced no per-kernel regression.
    """
    findings: list[Finding] = []
    for rank, kernel in enumerate(summary.top_regressions):
        if kernel.delta_ns <= 0:
            continue
        selection = _selection(summary, kernel, rank)
        findings.append(
            Finding(
                type="region",
                label=f"Regression: {kernel.name}",
                start_ns=selection.start_ns if selection.start_ns is not None else 0,
                end_ns=selection.end_ns,
                gpu_id=selection.gpu_ids[0] if selection.gpu_ids else None,
                color="rgba(255, 170, 0, 0.30)",
                severity="warning",
                note=_finding_note(kernel),
                id=_finding_id(summary, kernel, rank),
                category="compute",
                confidence=summary.comparability_confidence,
                selection=selection,
                explanation=(
                    f"{kernel.name} regressed by {kernel.delta_ns / 1e6:.3f}ms "
                    "relative to the selected baseline."
                ),
                suggested_actions=[_suggested_action(kernel)],
                false_positive_notes=[
                    "Aggregate kernel time can overlap across concurrent streams.",
                    "The headroom is a baseline-relative delta, not a guaranteed gain.",
                ],
                provenance={
                    "skill": "diff",
                    "row_kind": "kernel_regression",
                    "diff_id": summary.diff_id,
                    "kernel_key": kernel.key,
                },
                diff_lineage=DiffLineage(
                    diff_id=summary.diff_id,
                    role="regression",
                    rank=rank,
                    baseline_profile_id=summary.before.profile_id,
                ),
                headroom_ms=kernel.delta_ns / 1e6,
                headroom_basis="baseline_delta",
            )
        )
    return rank_findings(findings)
