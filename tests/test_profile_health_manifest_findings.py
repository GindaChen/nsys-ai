"""Tests for the structured roll-up findings emitted by profile_health_manifest.

These exercise `_to_findings` directly against synthetic manifest dicts so
the seven roll-up types can be unit-tested without an .sqlite profile.
End-to-end validation against `rich7421/fastvideo-wan-l40s-nsys` lives in
`pod-runbook-profile-health-manifest-findings.md`.
"""

from nsys_ai.skills.builtins.profile_health_manifest import (
    _COMM_BOUND_OVERLAP_PCT,
    _IDLE_DOMINANT_PCT,
    _ITER_VARIANCE_SPIKE_RATIO,
    _KERNEL_HOTSPOT_PCT,
    _MIN_ITERATIONS_FOR_NVTX_COVERAGE,
    _OVERHEAD_CONTAMINATED_PCT,
    _SYNC_BOUND_DENSITY_PCT,
    _to_findings,
)
from nsys_ai.skills.registry import get_skill


def _healthy_manifest() -> dict:
    """A manifest that should fire zero findings (all sub-thresholds healthy)."""
    return {
        "gpu": "NVIDIA L40S",
        "profile_span_ms": 5000.0,
        "top_kernels": [
            {"name": "kernel_a", "total_ms": 100.0, "count": 10},
            {"name": "kernel_b", "total_ms": 80.0, "count": 8},
            {"name": "kernel_c", "total_ms": 70.0, "count": 7},
        ],
        "total_kernel_ms": 500.0,
        "overlap": {
            "compute_only_ms": 300.0,
            "nccl_only_ms": 100.0,
            "overlap_pct": 75,
            "idle_ms": 50.0,
        },
        "nccl": {
            "streams": 2,
            "collectives": 50,
            "dominant_type": "AllReduce",
            "dominant_pct": 60,
            "total_nccl_ms": 150.0,
        },
        "sync": {"sync_density_pct": 5.0, "total_sync_wall_ms": 50.0},
        "idle": {"gap_count": 10, "idle_pct": 5.0, "total_idle_ms": 250.0},
        "nvtx": {
            "has_nvtx": True,
            "iteration_count": 10,
            "median_iter_ms": 100.0,
            "slowest_iter_ms": 110.0,
        },
        "data_quality": {"overhead_pct_raw": 0.1, "overhead_pct": 0.1, "profiler_overhead_ms": 5.0},
    }


# ── Trust Contract: silence over weak claims ─────────────────────────


class TestTrustContract:
    def test_empty_rows_returns_empty(self):
        assert _to_findings([]) == []

    def test_healthy_manifest_fires_nothing(self):
        # A profile inside every threshold should produce zero findings;
        # any false-positive here means a threshold drifted.
        assert _to_findings([_healthy_manifest()]) == []

    def test_skill_registers_to_findings_fn(self):
        skill = get_skill("profile_health_manifest")
        assert skill is not None
        assert skill.to_findings_fn is _to_findings


# ── Individual roll-up findings ──────────────────────────────────────


class TestOverheadContaminated:
    def test_fires_above_threshold(self):
        m = _healthy_manifest()
        m["data_quality"]["overhead_pct_raw"] = _OVERHEAD_CONTAMINATED_PCT + 0.5
        findings = _to_findings([m])
        ids = [f.id for f in findings]
        assert "profile_overhead_contaminated" in ids
        f = next(f for f in findings if f.id == "profile_overhead_contaminated")
        assert f.severity == "critical"
        assert f.category == "profile_quality"
        assert f.type == "highlight"
        assert f.start_ns == 0  # no time anchor
        assert f.end_ns is None

    def test_silent_at_threshold(self):
        m = _healthy_manifest()
        m["data_quality"]["overhead_pct_raw"] = _OVERHEAD_CONTAMINATED_PCT
        assert all(f.id != "profile_overhead_contaminated" for f in _to_findings([m]))


class TestSyncBound:
    def test_fires_above_threshold(self):
        m = _healthy_manifest()
        m["sync"]["sync_density_pct"] = _SYNC_BOUND_DENSITY_PCT + 5.0
        findings = _to_findings([m])
        ids = [f.id for f in findings]
        assert "profile_sync_bound" in ids
        f = next(f for f in findings if f.id == "profile_sync_bound")
        assert f.category == "sync"
        assert f.evidence[0].values["sync_density_pct"] == _SYNC_BOUND_DENSITY_PCT + 5.0

    def test_silent_at_threshold(self):
        m = _healthy_manifest()
        m["sync"]["sync_density_pct"] = _SYNC_BOUND_DENSITY_PCT
        assert all(f.id != "profile_sync_bound" for f in _to_findings([m]))


class TestCommBound:
    def test_fires_on_low_overlap(self):
        m = _healthy_manifest()
        m["overlap"]["overlap_pct"] = _COMM_BOUND_OVERLAP_PCT - 5
        m["overlap"]["nccl_only_ms"] = 200.0
        findings = _to_findings([m])
        f = next((f for f in findings if f.id == "profile_comm_bound"), None)
        assert f is not None
        assert f.category == "communication"
        assert "low_overlap" in f.provenance["triggers"]

    def test_fires_on_nccl_exceeds_compute(self):
        m = _healthy_manifest()
        m["overlap"]["compute_only_ms"] = 100.0
        m["nccl"]["total_nccl_ms"] = 500.0
        findings = _to_findings([m])
        f = next((f for f in findings if f.id == "profile_comm_bound"), None)
        assert f is not None
        assert "nccl_exceeds_compute" in f.provenance["triggers"]

    def test_low_overlap_with_no_nccl_does_not_fire(self):
        # Single-rank profile: overlap_pct is meaningless when nccl_only_ms=0.
        m = _healthy_manifest()
        m["overlap"]["overlap_pct"] = 0
        m["overlap"]["nccl_only_ms"] = 0
        m["nccl"]["total_nccl_ms"] = 0
        m["overlap"]["compute_only_ms"] = 500.0
        assert all(f.id != "profile_comm_bound" for f in _to_findings([m]))

    def test_zero_overlap_with_nccl_fires(self):
        # Regression: a Python ``x or 100`` default would mistake 0.0 for
        # "missing" and silence comm_bound on full-serialization profiles.
        # Caught during L40S validation on the uncompiled perf.sqlite,
        # where overlap_pct=0.0 with nccl_only_ms=8654ms must trigger
        # low_overlap.
        m = _healthy_manifest()
        m["overlap"]["overlap_pct"] = 0.0
        m["overlap"]["nccl_only_ms"] = 500.0
        findings = _to_findings([m])
        f = next((f for f in findings if f.id == "profile_comm_bound"), None)
        assert f is not None, "comm_bound must fire when overlap_pct is exactly 0.0"
        assert "low_overlap" in f.provenance["triggers"]

    def test_label_reflects_only_active_trigger(self):
        # When only nccl_exceeds_compute fires (overlap healthy), the label
        # must not cite "overlap 100%" because that's a non-signal. Same the
        # other way for low_overlap with compute dominant. Regression for
        # Copilot review MED #5.
        m = _healthy_manifest()
        # Trigger only nccl_dominates: overlap healthy, nccl > compute
        m["overlap"]["overlap_pct"] = 80
        m["overlap"]["nccl_only_ms"] = 0
        m["overlap"]["compute_only_ms"] = 100.0
        m["nccl"]["total_nccl_ms"] = 500.0
        f = next(f for f in _to_findings([m]) if f.id == "profile_comm_bound")
        assert "NCCL dominates" in f.label
        assert "overlap" not in f.label.lower()

        # Trigger only low_overlap: overlap < 30 with nccl traffic, compute big
        m = _healthy_manifest()
        m["overlap"]["overlap_pct"] = 10
        m["overlap"]["nccl_only_ms"] = 200.0
        m["overlap"]["compute_only_ms"] = 1000.0
        m["nccl"]["total_nccl_ms"] = 200.0
        f = next(f for f in _to_findings([m]) if f.id == "profile_comm_bound")
        assert "low overlap" in f.label
        assert "dominates" not in f.label.lower()

        # Both triggers: combined label
        m = _healthy_manifest()
        m["overlap"]["overlap_pct"] = 10
        m["overlap"]["nccl_only_ms"] = 200.0
        m["overlap"]["compute_only_ms"] = 100.0
        m["nccl"]["total_nccl_ms"] = 500.0
        f = next(f for f in _to_findings([m]) if f.id == "profile_comm_bound")
        assert "overlap 10%" in f.label
        assert "NCCL 500ms vs compute 100ms" in f.label

    def test_missing_overlap_pct_defaults_healthy(self):
        # Defensive: if overlap_breakdown errored out and overlap dict has
        # no overlap_pct key at all, default to 100 (healthy) so a missing
        # measurement doesn't produce a false-positive low-overlap finding.
        m = _healthy_manifest()
        del m["overlap"]["overlap_pct"]
        m["overlap"]["nccl_only_ms"] = 500.0
        # nccl < compute so the dominance trigger also stays silent
        m["overlap"]["compute_only_ms"] = 1000.0
        m["nccl"]["total_nccl_ms"] = 500.0
        assert all(f.id != "profile_comm_bound" for f in _to_findings([m]))


class TestIdleDominant:
    def test_fires_above_threshold(self):
        m = _healthy_manifest()
        m["idle"]["idle_pct"] = _IDLE_DOMINANT_PCT + 5.0
        findings = _to_findings([m])
        f = next((f for f in findings if f.id == "profile_idle_dominant"), None)
        assert f is not None
        assert f.category == "idle"


class TestKernelHotspot:
    def test_fires_when_top_dominates(self):
        m = _healthy_manifest()
        # One kernel = 70% of total kernel time
        m["top_kernels"] = [{"name": "fa_fwd", "total_ms": 700.0, "count": 100}]
        m["total_kernel_ms"] = 1000.0
        findings = _to_findings([m])
        f = next((f for f in findings if f.id == "profile_kernel_hotspot"), None)
        assert f is not None
        assert f.category == "compute"
        assert f.evidence[0].values["kernel_name"] == "fa_fwd"
        assert f.evidence[0].values["pct_of_total_kernel_ms"] == 70.0

    def test_silent_on_balanced_workload(self):
        m = _healthy_manifest()
        # Top kernel = exactly threshold% → should NOT fire (strict >)
        m["top_kernels"] = [{"name": "k", "total_ms": _KERNEL_HOTSPOT_PCT, "count": 1}]
        m["total_kernel_ms"] = 100.0
        assert all(f.id != "profile_kernel_hotspot" for f in _to_findings([m]))

    def test_silent_when_total_zero(self):
        # Defensive: aggregate_kernels could return error rows.
        m = _healthy_manifest()
        m["total_kernel_ms"] = 0
        assert all(f.id != "profile_kernel_hotspot" for f in _to_findings([m]))


class TestIterationVarianceSpike:
    def test_fires_on_spike(self):
        m = _healthy_manifest()
        m["nvtx"]["median_iter_ms"] = 100.0
        m["nvtx"]["slowest_iter_ms"] = 100.0 * (_ITER_VARIANCE_SPIKE_RATIO + 0.5)
        findings = _to_findings([m])
        f = next((f for f in findings if f.id == "profile_iteration_variance_spike"), None)
        assert f is not None
        assert f.category == "nvtx"
        assert f.evidence[0].values["ratio"] >= _ITER_VARIANCE_SPIKE_RATIO

    def test_silent_with_zero_median(self):
        # Defensive: missing iteration_timing → median 0 → no division.
        m = _healthy_manifest()
        m["nvtx"]["median_iter_ms"] = 0
        m["nvtx"]["slowest_iter_ms"] = 100.0
        assert all(
            f.id != "profile_iteration_variance_spike" for f in _to_findings([m])
        )


class TestInsufficientNvtxCoverage:
    def test_fires_without_nvtx(self):
        m = _healthy_manifest()
        m["nvtx"] = {"has_nvtx": False}
        findings = _to_findings([m])
        f = next((f for f in findings if f.id == "profile_insufficient_nvtx_coverage"), None)
        assert f is not None
        assert f.severity == "info"
        assert f.category == "profile_quality"

    def test_fires_below_min_iterations(self):
        m = _healthy_manifest()
        m["nvtx"]["iteration_count"] = _MIN_ITERATIONS_FOR_NVTX_COVERAGE - 1
        findings = _to_findings([m])
        assert any(
            f.id == "profile_insufficient_nvtx_coverage" for f in findings
        )
        f = next(f for f in findings if f.id == "profile_insufficient_nvtx_coverage")
        # The reason string should mention the iteration count, not "no NVTX".
        assert "iteration" in f.label.lower()

    def test_silent_at_min_iterations(self):
        m = _healthy_manifest()
        m["nvtx"]["iteration_count"] = _MIN_ITERATIONS_FOR_NVTX_COVERAGE
        m["nvtx"]["has_nvtx"] = True
        assert all(
            f.id != "profile_insufficient_nvtx_coverage" for f in _to_findings([m])
        )


# ── Cross-finding invariants ─────────────────────────────────────────


class TestStructuralInvariants:
    def test_all_findings_have_required_envelope(self):
        # Trigger every finding type at once.
        m = _healthy_manifest()
        m["data_quality"]["overhead_pct_raw"] = 5.0
        m["sync"]["sync_density_pct"] = 30.0
        m["overlap"]["overlap_pct"] = 10
        m["overlap"]["nccl_only_ms"] = 200.0
        m["idle"]["idle_pct"] = 25.0
        m["top_kernels"] = [{"name": "k", "total_ms": 900.0, "count": 1}]
        m["total_kernel_ms"] = 1000.0
        m["nvtx"]["median_iter_ms"] = 100.0
        m["nvtx"]["slowest_iter_ms"] = 200.0
        m["nvtx"]["has_nvtx"] = False  # also trip coverage
        findings = _to_findings([m])
        # 7 distinct types possible; all should fire (or all but one if
        # has_nvtx=False zeros out the iteration spike inputs — but we
        # set median/slowest on the same dict so it still fires).
        ids = {f.id for f in findings}
        expected = {
            "profile_overhead_contaminated",
            "profile_sync_bound",
            "profile_comm_bound",
            "profile_idle_dominant",
            "profile_kernel_hotspot",
            "profile_iteration_variance_spike",
            "profile_insufficient_nvtx_coverage",
        }
        assert ids == expected, f"missing: {expected - ids}, extra: {ids - expected}"

        for f in findings:
            # Envelope invariants every roll-up must satisfy.
            assert f.type == "highlight"
            assert f.start_ns == 0
            assert f.end_ns is None
            assert f.id and f.id.startswith("profile_")
            assert f.category in {
                "compute",
                "communication",
                "idle",
                "sync",
                "nvtx",
                "profile_quality",
            }
            assert f.severity in {"critical", "warning", "info"}
            assert f.evidence and len(f.evidence) == 1
            assert f.selection is not None
            assert f.selection.id == f"sel_{f.id}"
            assert f.evidence[0].selection_id == f.selection.id
            assert f.evidence[0].source_skill == "profile_health_manifest"
            assert f.provenance["skill"] == "profile_health_manifest"
            assert f.explanation
            assert f.suggested_actions and len(f.suggested_actions) >= 1

    def test_findings_serialize_to_dict(self):
        # Ensures the Finding/EvidenceRow/TraceSelection round-trip cleanly
        # for downstream JSON consumption (the agent + diff CLI consume to_dict()).
        m = _healthy_manifest()
        m["data_quality"]["overhead_pct_raw"] = 5.0
        findings = _to_findings([m])
        for f in findings:
            d = f.to_dict()
            assert "id" in d
            assert "evidence" in d
            assert "selection" in d
            assert "suggested_actions" in d
