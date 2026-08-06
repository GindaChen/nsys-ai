"""Tests for gpu_idle_gaps' v0.1 Finding output.

Direct unit tests against the skill's ``_to_findings`` helper plus
integration tests through ``EvidenceBuilder`` that exercise the
context-passing wire-up.
"""

import json

from nsys_ai.annotation import DiffLineage, EvidenceRow, Finding, TraceSelection
from nsys_ai.skills.builtins.gpu_idle_gaps import _gap_confidence, _to_findings

# ──────────────────────────────────────────────────────────────────────
# Row builders
# ──────────────────────────────────────────────────────────────────────


def _per_gap_row(
    *,
    gap_ns: int = 12_500_000,  # 12.5 ms
    start_ns: int = 1_000_000,
    end_ns: int | None = None,
    device_id: int = 0,
    stream_id: int = 7,
    attribution: dict | None = None,
) -> dict:
    return {
        "gap_ns": gap_ns,
        "start_ns": start_ns,
        "end_ns": end_ns if end_ns is not None else start_ns + gap_ns,
        "deviceId": device_id,
        "streamId": stream_id,
        "attribution": attribution or {},
    }


def _summary_row(
    *,
    pct_of_profile: float = 18.0,
    total_idle_ms: float = 250.0,
    gap_count: int = 12,
    gpu_id: int = 0,
    profile_start_ns: int = 0,
    profile_end_ns: int = 10_000_000_000,
) -> dict:
    return {
        "_summary": True,
        "pct_of_profile": pct_of_profile,
        "total_idle_ms": total_idle_ms,
        "gap_count": gap_count,
        "gpu_id": gpu_id,
        "profile_start_ns": profile_start_ns,
        "profile_end_ns": profile_end_ns,
    }


# ──────────────────────────────────────────────────────────────────────
# Per-gap finding shape
# ──────────────────────────────────────────────────────────────────────


class TestPerGapFinding:
    def test_emits_one_finding_per_row(self):
        findings = _to_findings([_per_gap_row(), _per_gap_row(start_ns=99)])
        assert len(findings) == 2

    def test_existing_display_fields_preserved(self):
        """Legacy display fields keep their pre-v0.1 values for overlay code."""
        findings = _to_findings([_per_gap_row(gap_ns=5_000_000, stream_id=3)])
        f = findings[0]
        assert f.type == "region"
        assert f.label == "GPU Idle Gap (5.00ms)"
        assert f.gpu_id == 0
        assert f.stream == "3"
        assert f.severity == "warning"
        assert "Stream 3" in f.note

    def test_v01_fields_populated(self):
        findings = _to_findings(
            [_per_gap_row(gap_ns=12_500_000, start_ns=500, stream_id=7)],
            context={"profile_id": "/tmp/a.sqlite"},
        )
        f = findings[0]
        assert f.id == "idle_gap_gpu0_stream7_500"
        assert f.category == "idle"
        assert isinstance(f.confidence, float)
        assert 0.0 <= f.confidence <= 1.0
        assert f.explanation and "idle" in f.explanation.lower()
        assert f.suggested_actions and len(f.suggested_actions) >= 1
        assert f.false_positive_notes and len(f.false_positive_notes) >= 1
        assert f.provenance == {"skill": "gpu_idle_gaps", "row_kind": "per_gap"}

    def test_selection_pins_profile_and_region(self):
        findings = _to_findings(
            [_per_gap_row(start_ns=100, end_ns=200, stream_id=9, device_id=3)],
            context={"profile_id": "/tmp/p.sqlite"},
        )
        sel = findings[0].selection
        assert isinstance(sel, TraceSelection)
        assert sel.profile_id == "/tmp/p.sqlite"
        assert sel.source == "skill:gpu_idle_gaps"
        assert sel.start_ns == 100
        assert sel.end_ns == 200
        assert sel.gpu_ids == [3]
        assert sel.stream_ids == [9]
        assert sel.id == findings[0].evidence[0].selection_id

    def test_evidence_row_has_metric_and_units(self):
        findings = _to_findings([_per_gap_row(gap_ns=8_000_000)])
        ev = findings[0].evidence
        assert len(ev) == 1
        row = ev[0]
        assert isinstance(row, EvidenceRow)
        assert row.source_skill == "gpu_idle_gaps"
        assert row.values["gap_ms"] == 8.0
        assert row.values["gap_ns"] == 8_000_000
        assert row.units["gap_ms"] == "ms"
        assert row.units["gap_ns"] == "ns"

    def test_evidence_row_includes_cpu_attribution_when_available(self):
        attr = {"top_apis": [{"name": "cudaMalloc_v11000", "total_ms": 4.2}]}
        findings = _to_findings([_per_gap_row(attribution=attr)])
        row = findings[0].evidence[0]
        assert row.values["top_cpu_api"] == "cudaMalloc"
        assert row.values["top_cpu_api_ms"] == 4.2
        assert row.units["top_cpu_api_ms"] == "ms"

    def test_evidence_row_omits_cpu_attribution_when_absent(self):
        findings = _to_findings([_per_gap_row(attribution={})])
        row = findings[0].evidence[0]
        assert "top_cpu_api" not in row.values
        assert "top_cpu_api_ms" not in row.values

    def test_diff_lineage_not_set_by_default(self):
        """gpu_idle_gaps is upstream of diff; lineage stays None here."""
        findings = _to_findings([_per_gap_row()])
        assert findings[0].diff_lineage is None
        assert not isinstance(findings[0].diff_lineage, DiffLineage)


# ──────────────────────────────────────────────────────────────────────
# Summary finding shape
# ──────────────────────────────────────────────────────────────────────


class TestSummaryFinding:
    def test_emitted_only_above_5_pct(self):
        below = _to_findings([_summary_row(pct_of_profile=4.0)])
        assert below == []

        above = _to_findings([_summary_row(pct_of_profile=10.0)])
        assert len(above) == 1

    def test_v01_fields_on_summary(self):
        findings = _to_findings(
            [_summary_row(pct_of_profile=22.0, total_idle_ms=300, gap_count=15, gpu_id=2)],
            context={"profile_id": "p"},
        )
        f = findings[0]
        assert f.id == "idle_summary_gpu2"
        assert f.category == "idle"
        assert f.selection.gpu_ids == [2]
        assert f.selection.profile_id == "p"
        assert f.provenance["row_kind"] == "summary"

    def test_summary_evidence_carries_aggregates(self):
        findings = _to_findings(
            [_summary_row(pct_of_profile=20.0, total_idle_ms=250, gap_count=12)]
        )
        ev = findings[0].evidence[0]
        assert ev.values["pct_of_profile"] == 20.0
        assert ev.values["total_idle_ms"] == 250
        assert ev.values["gap_count"] == 12
        assert ev.units["pct_of_profile"] == "percent"
        assert ev.provenance["row_kind"] == "summary"

    def test_severity_threshold_at_15_pct(self):
        low = _to_findings([_summary_row(pct_of_profile=10.0)])
        high = _to_findings([_summary_row(pct_of_profile=20.0)])
        assert low[0].severity == "info"
        assert high[0].severity == "warning"


# ──────────────────────────────────────────────────────────────────────
# Context handling + confidence heuristic
# ──────────────────────────────────────────────────────────────────────


class TestContextHandling:
    def test_no_context_falls_back_to_unknown(self):
        findings = _to_findings([_per_gap_row()])
        assert findings[0].selection.profile_id == "unknown"

    def test_empty_context_falls_back_to_unknown(self):
        findings = _to_findings([_per_gap_row()], context={})
        assert findings[0].selection.profile_id == "unknown"

    def test_context_with_extra_keys_is_tolerated(self):
        findings = _to_findings(
            [_per_gap_row()],
            context={"profile_id": "/p", "unrelated": 42},
        )
        assert findings[0].selection.profile_id == "/p"

    def test_error_rows_are_skipped(self):
        findings = _to_findings([{"error": "boom"}, _per_gap_row()])
        assert len(findings) == 1
        assert findings[0].id.startswith("idle_gap_")


class TestGapConfidence:
    def test_monotonic_increasing(self):
        for a, b in [(0.5, 5), (5, 50), (50, 500)]:
            assert _gap_confidence(a) < _gap_confidence(b)

    def test_bounds(self):
        assert 0.0 <= _gap_confidence(0.1) <= 1.0
        assert 0.0 <= _gap_confidence(10_000) <= 1.0

    def test_small_gaps_low_confidence(self):
        """Gaps under the launch-overhead floor should rank ≤0.5."""
        assert _gap_confidence(0.5) <= 0.5


# ──────────────────────────────────────────────────────────────────────
# Serialization round-trip
# ──────────────────────────────────────────────────────────────────────


class TestSerialization:
    def test_full_finding_json_round_trip(self):
        findings = _to_findings(
            [
                _per_gap_row(
                    gap_ns=12_000_000,
                    start_ns=100,
                    stream_id=5,
                    attribution={"top_apis": [{"name": "cudaLaunchKernel", "total_ms": 2.1}]},
                ),
                _summary_row(pct_of_profile=18.0),
            ],
            context={"profile_id": "/tmp/test.sqlite"},
        )
        for f in findings:
            d = f.to_dict()
            # Nested types are serialized via their own to_dict.
            assert isinstance(d["selection"], dict)
            assert isinstance(d["evidence"], list)
            assert isinstance(d["evidence"][0], dict)
            # JSON-clean and re-loadable.
            restored = Finding.from_dict(json.loads(json.dumps(d)))
            assert restored.id == f.id
            assert restored.category == "idle"
            assert isinstance(restored.selection, TraceSelection)
            assert restored.selection.profile_id == "/tmp/test.sqlite"
            assert isinstance(restored.evidence[0], EvidenceRow)


# ──────────────────────────────────────────────────────────────────────
# EvidenceBuilder integration
# ──────────────────────────────────────────────────────────────────────


class TestEvidenceBuilderIntegration:
    def test_builder_passes_profile_id_to_idle_gaps(self, minimal_nsys_db_path):
        """Real DB integration: TraceSelections produced under EvidenceBuilder
        carry the content-derived ``profile_id`` (``nsys2:sha256:...``)
        from ``fingerprint.get_profile_id``, *not* the filesystem path."""
        from nsys_ai.evidence_builder import EvidenceBuilder
        from nsys_ai.profile import Profile

        with Profile(minimal_nsys_db_path) as prof:
            builder = EvidenceBuilder(prof, device=0)
            report = builder.build(only=["idle_gaps"])

        # The envelope must always be stamped with a content-derived id,
        # regardless of whether the fixture happened to produce findings.
        assert report.profile_id.startswith("nsys2:"), report.profile_id
        assert report.profile_id != str(minimal_nsys_db_path), (
            "profile_id must be a hash, not the filesystem path"
        )

        # The envelope and every nested selection share one profile_id;
        # the minimal fixture may or may not produce idle-gap findings on
        # any given run, so we only check the per-finding contract when
        # at least one finding exists.
        idle = [f for f in report.findings if f.category == "idle"]
        for f in idle:
            assert f.selection is not None
            assert f.selection.profile_id == report.profile_id
            assert f.selection.source == "skill:gpu_idle_gaps"

    def test_builder_still_works_with_legacy_single_arg_skills(self, minimal_nsys_db_path):
        """Skills whose to_findings_fn is the legacy (rows) single-arg form
        keep working alongside the upgraded gpu_idle_gaps."""
        from nsys_ai.evidence_builder import EvidenceBuilder
        from nsys_ai.profile import Profile

        with Profile(minimal_nsys_db_path) as prof:
            builder = EvidenceBuilder(prof, device=0)
            # Calls multiple skills; no error means context dispatch
            # is correctly back-compatible.
            report = builder.build()
        assert report is not None
        assert isinstance(report.findings, list)

    def test_builder_accepts_pathlib_profile_path(self, minimal_nsys_db_path):
        """End-to-end regression: passing ``pathlib.Path`` to ``Profile``
        must flow through ``EvidenceBuilder`` → ``EvidenceReport`` → JSON
        without crashing on ``Path.encode`` or
        ``Object of type PosixPath is not JSON serializable``.
        """
        import json
        from pathlib import Path

        from nsys_ai.evidence_builder import EvidenceBuilder
        from nsys_ai.profile import Profile

        with Profile(Path(minimal_nsys_db_path)) as prof:
            builder = EvidenceBuilder(prof, device=0)
            report = builder.build()

        # builder coerces and ``EvidenceReport.__post_init__`` coerces
        # again as a belt-and-braces guarantee.
        assert isinstance(report.profile_path, str)
        assert report.profile_id.startswith("nsys2:"), report.profile_id
        # JSON-dumpable end-to-end (this is the failure mode if either
        # coercion regresses).
        json.dumps(report.to_dict())
