"""Tests for kernel_instances skill."""

import json

import pytest

from nsys_ai.annotation import EvidenceRow, Finding, TraceSelection
from nsys_ai.skills.registry import get_skill


@pytest.fixture
def ki_skill():
    skill = get_skill("kernel_instances")
    assert skill is not None, "kernel_instances skill not registered"
    return skill


def _kernel_row(**overrides):
    row = {
        "kernel_name": "void flash_attention_kernel()",
        "short_name": "flash_attention_kernel",
        "start_ns": 1_000,
        "end_ns": 6_001_000,
        "duration_ms": 6.0,
        "stream_id": 7,
        "device_id": 2,
    }
    row.update(overrides)
    return row


class TestKernelInstances:
    def test_returns_instances_with_ns(self, minimal_nsys_conn, ki_skill):
        rows = ki_skill.execute(minimal_nsys_conn, device=0, limit=5)
        assert isinstance(rows, list)
        assert len(rows) > 0
        r = rows[0]
        assert "start_ns" in r
        assert "end_ns" in r
        assert "duration_ms" in r
        assert r["start_ns"] < r["end_ns"]

    def test_has_both_names(self, minimal_nsys_conn, ki_skill):
        rows = ki_skill.execute(minimal_nsys_conn, device=0, limit=1)
        assert len(rows) > 0
        r = rows[0]
        assert "kernel_name" in r  # demangled
        assert "short_name" in r

    def test_name_filter(self, minimal_nsys_conn, ki_skill):
        """Filtering by name should return only matching kernels."""
        all_rows = ki_skill.execute(minimal_nsys_conn, device=0, limit=100)
        if not all_rows:
            pytest.skip("No kernels in test data")

        # Use a substring from the first kernel's name
        target = all_rows[0]["short_name"][:5]
        filtered = ki_skill.execute(minimal_nsys_conn, device=0, name=target, limit=100)
        assert len(filtered) >= 1
        assert len(filtered) <= len(all_rows)
        target_lower = target.lower()
        for row in filtered:
            short_name = (row.get("short_name") or "").lower()
            kernel_name = (row.get("kernel_name") or "").lower()
            assert target_lower in short_name or target_lower in kernel_name

    def test_limit_respected(self, minimal_nsys_conn, ki_skill):
        rows = ki_skill.execute(minimal_nsys_conn, device=0, limit=2)
        assert len(rows) <= 2

    def test_empty_name_returns_all(self, minimal_nsys_conn, ki_skill):
        rows = ki_skill.execute(minimal_nsys_conn, device=0, name="", limit=100)
        assert isinstance(rows, list)

    def test_sql_injection_safe(self, minimal_nsys_conn, ki_skill):
        """Name containing SQL injection chars should not crash."""
        rows = ki_skill.execute(
            minimal_nsys_conn, device=0, name="'; DROP TABLE StringIds; --", limit=5
        )
        assert isinstance(rows, list)

    def test_format_output(self, minimal_nsys_conn, ki_skill):
        rows = ki_skill.execute(minimal_nsys_conn, device=0, limit=3)
        text = ki_skill.format_rows(rows)
        assert "Kernel Instances" in text

    def test_duckdb_path(self, duckdb_conn, ki_skill):
        rows = ki_skill.execute(duckdb_conn, device=0, limit=3)
        assert isinstance(rows, list)
        if rows:
            assert "start_ns" in rows[0]


class TestKernelInstanceFindings:
    def test_compute_hotspot_has_v01_evidence_shape(self, ki_skill):
        findings = ki_skill.to_findings_fn(
            [_kernel_row()],
            context={"profile_id": "nsys1:sha256:test"},
        )
        assert len(findings) == 1
        f = findings[0]

        assert f.type == "highlight"
        assert f.label == "Hotspot: flash_attention_kernel"
        assert f.start_ns == 1_000
        assert f.end_ns == 6_001_000
        assert f.gpu_id == 2
        assert f.stream == "7"
        assert f.severity == "info"
        assert f.category == "compute"
        assert isinstance(f.confidence, float)
        assert 0.0 <= f.confidence <= 1.0
        assert f.explanation and "compute kernel" in f.explanation
        assert f.suggested_actions
        assert f.false_positive_notes
        assert f.provenance == {"skill": "kernel_instances", "row_kind": "kernel_hotspot"}

        assert isinstance(f.selection, TraceSelection)
        assert f.selection.profile_id == "nsys1:sha256:test"
        assert f.selection.source == "skill:kernel_instances"
        assert f.selection.start_ns == 1_000
        assert f.selection.end_ns == 6_001_000
        assert f.selection.gpu_ids == [2]
        assert f.selection.stream_ids == [7]

        assert f.evidence and isinstance(f.evidence[0], EvidenceRow)
        ev = f.evidence[0]
        assert ev.source_skill == "kernel_instances"
        assert ev.selection_id == f.selection.id
        assert ev.values["kernel_name"] == "void flash_attention_kernel()"
        assert ev.values["short_name"] == "flash_attention_kernel"
        assert ev.values["duration_ms"] == 6.0
        assert ev.values["duration_ns"] == 6_000_000
        assert ev.values["device_id"] == 2
        assert ev.values["stream_id"] == 7
        assert ev.values["is_nccl"] is False
        assert ev.units["duration_ms"] == "ms"
        assert ev.units["duration_ns"] == "ns"

    def test_nccl_kernel_has_communication_category_and_critical_severity(self, ki_skill):
        findings = ki_skill.to_findings_fn(
            [
                _kernel_row(
                    kernel_name="ncclDevKernel_AllReduce",
                    short_name="ncclDevKernel_AllReduce",
                    duration_ms=12.5,
                    start_ns=10,
                    end_ns=12_500_010,
                )
            ],
            context={"profile_id": "p"},
        )
        assert len(findings) == 1
        f = findings[0]
        assert f.label == "Long NCCL (12.50ms)"
        assert f.category == "communication"
        assert f.severity == "warning"
        assert f.confidence == 0.7
        assert f.selection.profile_id == "p"
        assert f.provenance["row_kind"] == "long_nccl"
        ev = f.evidence[0]
        assert ev.values["is_nccl"] is True
        assert ev.values["duration_ms"] == 12.5
        assert ev.values["duration_ns"] == 12_500_000
        assert ev.provenance["row_kind"] == "long_nccl"

    def test_short_nccl_kernel_keeps_warning_severity(self, ki_skill):
        findings = ki_skill.to_findings_fn(
            [_kernel_row(short_name="ncclTinyKernel", kernel_name="ncclTinyKernel", duration_ms=2.0)]
        )
        assert findings[0].label == "NCCL kernel (2.00ms)"
        assert findings[0].severity == "warning"
        assert findings[0].category == "communication"

    def test_stream_id_is_normalized_in_structured_fields(self, ki_skill):
        findings = ki_skill.to_findings_fn([_kernel_row(stream_id="7")])

        f = findings[0]
        assert f.id == "kernel_instance_gpu2_stream7_1000"
        assert f.stream == "7"
        assert f.selection.stream_ids == [7]
        assert f.evidence[0].values["stream_id"] == 7
        assert f.evidence[0].provenance["stream"] == 7

    def test_error_rows_are_skipped(self, ki_skill):
        findings = ki_skill.to_findings_fn([{"error": "boom"}, _kernel_row()])
        assert len(findings) == 1
        assert findings[0].id.startswith("kernel_instance_")

    def test_no_context_falls_back_to_unknown_profile_id(self, ki_skill):
        findings = ki_skill.to_findings_fn([_kernel_row()])
        assert findings[0].selection.profile_id == "unknown"

    def test_finding_round_trips_through_json(self, ki_skill):
        findings = ki_skill.to_findings_fn(
            [_kernel_row()],
            context={"profile_id": "/tmp/profile.sqlite"},
        )
        d = findings[0].to_dict()
        assert isinstance(d["selection"], dict)
        assert isinstance(d["evidence"], list)

        restored = Finding.from_dict(json.loads(json.dumps(d)))
        assert restored.id == findings[0].id
        assert restored.category == "compute"
        assert isinstance(restored.selection, TraceSelection)
        assert restored.selection.profile_id == "/tmp/profile.sqlite"
        assert isinstance(restored.evidence[0], EvidenceRow)
