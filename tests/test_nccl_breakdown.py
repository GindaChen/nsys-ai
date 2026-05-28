"""Tests for per-stream NCCL breakdown (Issue #24).

Uses the minimal_nsys_conn and duckdb_conn fixtures from conftest.py.

Test data recap (from conftest.py):
  Stream 7: nccl_ReduceScatter [4.5-5.5ms] (same-stream anti-pattern)
  Stream 8: nccl_AllReduce [2.5-3.5ms]
"""

import json

import pytest

from nsys_ai.annotation import EvidenceRow, Finding, TraceSelection
from nsys_ai.skills.builtins.nccl_breakdown import SKILL
from nsys_ai.skills.registry import get_skill

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def nccl_skill():
    skill = get_skill("nccl_breakdown")
    assert skill is not None, "nccl_breakdown skill not registered"
    return skill


# ── Python engine tests (overlap.nccl_breakdown) ─────────────────


class TestPerStreamGrouping:
    def test_returns_stream_id(self, minimal_nsys_conn):
        from nsys_ai.overlap import nccl_breakdown
        from nsys_ai.profile import Profile

        prof = Profile._from_conn(minimal_nsys_conn)
        rows = nccl_breakdown(prof, device=0)
        assert len(rows) > 0
        for r in rows:
            assert "stream_id" in r, f"Missing stream_id in {r}"

    def test_separate_streams(self, minimal_nsys_conn):
        """AllReduce on Stream 8, ReduceScatter on Stream 7."""
        from nsys_ai.overlap import nccl_breakdown
        from nsys_ai.profile import Profile

        prof = Profile._from_conn(minimal_nsys_conn)
        rows = nccl_breakdown(prof, device=0)
        stream_types = {(r["stream_id"], r["type"]) for r in rows}
        assert (7, "reducescatter") in stream_types
        assert (8, "allreduce") in stream_types

    def test_sorted_by_stream_then_total(self, minimal_nsys_conn):
        from nsys_ai.overlap import nccl_breakdown
        from nsys_ai.profile import Profile

        prof = Profile._from_conn(minimal_nsys_conn)
        rows = nccl_breakdown(prof, device=0)
        stream_ids = [r["stream_id"] for r in rows]
        assert stream_ids == sorted(stream_ids), "Results not sorted by stream_id"

    def test_pct_is_global(self, minimal_nsys_conn):
        """pct should be relative to total NCCL time across all streams."""
        from nsys_ai.overlap import nccl_breakdown
        from nsys_ai.profile import Profile

        prof = Profile._from_conn(minimal_nsys_conn)
        rows = nccl_breakdown(prof, device=0)
        total_pct = sum(r["pct"] for r in rows)
        assert total_pct == pytest.approx(100.0, abs=0.5)

    def test_empty_device_returns_empty(self, minimal_nsys_conn):
        from nsys_ai.overlap import nccl_breakdown
        from nsys_ai.profile import Profile

        prof = Profile._from_conn(minimal_nsys_conn)
        rows = nccl_breakdown(prof, device=99)
        assert rows == []


# ── Format tests ─────────────────────────────────────────────────


class TestFormatNccl:
    def test_stream_headers(self, minimal_nsys_conn):
        from nsys_ai.overlap import format_nccl, nccl_breakdown
        from nsys_ai.profile import Profile

        prof = Profile._from_conn(minimal_nsys_conn)
        rows = nccl_breakdown(prof, device=0)
        text = format_nccl(rows)
        assert "[Stream 7]" in text
        assert "[Stream 8]" in text
        assert "NCCL Collective Breakdown" in text

    def test_empty_returns_no_collectives(self):
        from nsys_ai.overlap import format_nccl

        assert "No NCCL" in format_nccl([])


# ── Skill integration tests ──────────────────────────────────────


class TestSkillIntegration:
    def test_skill_execute(self, minimal_nsys_conn, nccl_skill):
        rows = nccl_skill.execute(minimal_nsys_conn, device=0)
        assert isinstance(rows, list)
        assert len(rows) > 0
        assert "stream_id" in rows[0]

    def test_skill_format(self, minimal_nsys_conn, nccl_skill):
        rows = nccl_skill.execute(minimal_nsys_conn, device=0)
        text = nccl_skill.format_rows(rows)
        assert "[Stream" in text
        assert "ms" in text

    def test_duckdb_path(self, duckdb_conn, nccl_skill):
        rows = nccl_skill.execute(duckdb_conn, device=0)
        assert isinstance(rows, list)
        assert len(rows) > 0
        assert "stream_id" in rows[0]


# ── Finding tests (v0.1) ─────────────────────────────────────────


def _nccl_row(**overrides):
    row = {
        "stream_id": 7,
        "type": "allreduce",
        "count": 100,
        "total_ms": 500.0,
        "avg_ms": 5.0,
        "min_ms": 4.5,
        "max_ms": 5.5,
        "pct": 80.0,
    }
    row.update(overrides)
    return row


def test_sendrecv_dominated_finding():
    rows = [_nccl_row(type="sendrecv", pct=85.0)]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    assert any(f.id == "nccl_sendrecv_dominated" for f in findings)
    f = next(f for f in findings if f.id == "nccl_sendrecv_dominated")
    assert f.type == "region"
    assert f.category == "communication"
    assert f.severity == "info"
    assert isinstance(f.confidence, float) and 0.0 <= f.confidence <= 1.0
    assert f.explanation and "SendRecv" in f.explanation
    assert f.suggested_actions
    assert f.false_positive_notes
    assert isinstance(f.selection, TraceSelection)
    assert f.selection.profile_id == "test"
    assert isinstance(f.evidence[0], EvidenceRow)
    assert f.evidence[0].values["sendrecv_pct"] == 85.0
    assert f.evidence[0].units["sendrecv_pct"] == "percent"


def test_allgather_dominated_finding():
    rows = [_nccl_row(type="allgather", pct=75.0)]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    assert any(f.id == "nccl_allgather_dominated" for f in findings)
    f = next(f for f in findings if f.id == "nccl_allgather_dominated")
    assert f.category == "communication"
    assert f.severity == "info"
    assert f.evidence[0].values["allgather_pct"] == 75.0
    assert f.evidence[0].units["allgather_pct"] == "percent"


def test_allreduce_dominated_finding():
    rows = [_nccl_row(type="allreduce", pct=80.0)]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    assert any(f.id == "nccl_allreduce_dominated" for f in findings)
    f = next(f for f in findings if f.id == "nccl_allreduce_dominated")
    assert f.category == "communication"
    assert f.severity == "info"
    assert f.evidence[0].values["allreduce_pct"] == 80.0
    assert f.evidence[0].units["allreduce_pct"] == "percent"


def test_high_variability_finding():
    rows = [_nccl_row(type="sendrecv", pct=50.0, avg_ms=10.0, max_ms=30.0)]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    assert any("high_variability" in (f.id or "") for f in findings)
    f = next(f for f in findings if "high_variability" in (f.id or ""))
    assert f.severity == "warning"
    assert f.category == "communication"
    assert f.evidence[0].values["max_avg_ratio"] == pytest.approx(3.0, abs=0.1)
    assert f.evidence[0].units["max_avg_ratio"] == "ratio"
    assert f.evidence[0].provenance["collective_type"] == "sendrecv"


def test_mixed_yields_no_dominated_finding():
    rows = [
        _nccl_row(type="sendrecv",  pct=40.0),
        _nccl_row(type="allgather", pct=35.0),
        _nccl_row(type="allreduce", pct=25.0),
    ]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    dominated = [f for f in findings if "dominated" in (f.id or "")]
    assert dominated == []


def test_empty_rows_returns_empty_findings():
    findings = SKILL.to_findings_fn([], context={"profile_id": "test"})
    assert findings == []


def test_json_round_trip():
    rows = [_nccl_row(type="sendrecv", pct=85.0)]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    for f in findings:
        d = f.to_dict()
        assert isinstance(d["selection"], dict)
        assert isinstance(d["evidence"], list)
        restored = Finding.from_dict(json.loads(json.dumps(d)))
        assert restored.id == f.id
        assert restored.category == "communication"
        assert isinstance(restored.selection, TraceSelection)
        assert restored.selection.profile_id == "test"
        assert isinstance(restored.evidence[0], EvidenceRow)


def test_findings_without_context_use_unknown_profile_id():
    findings = SKILL.to_findings_fn([_nccl_row(type="sendrecv", pct=85.0)])
    assert findings
    assert findings[0].selection.profile_id == "unknown"


def test_l40s_realistic_input():
    """Mirror the real L40S profile distribution from rich7421/fastvideo-wan-l40s-nsys."""
    rows = [
        _nccl_row(type="sendrecv",  count=10802, pct=97.7, avg_ms=24.551, max_ms=235.945,
                  total_ms=265196.86, span_start_ns=0, span_end_ns=700_000_000_000),
        _nccl_row(type="allgather", count=181,   pct=2.3,  avg_ms=35.169, max_ms=39.593,
                  total_ms=6365.54,  span_start_ns=0, span_end_ns=700_000_000_000),
        _nccl_row(type="allreduce", count=2,     pct=0.0,  avg_ms=0.699,  max_ms=1.010,
                  total_ms=1.4,      span_start_ns=0, span_end_ns=700_000_000_000),
    ]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "l40s_test"})
    assert any(f.id == "nccl_sendrecv_dominated" for f in findings)
    assert any("high_variability_sendrecv" in (f.id or "") for f in findings)
    assert any(f.id == "nccl_serialization" for f in findings)
    assert not any(f.id == "nccl_allgather_dominated" for f in findings)
    assert not any(f.id == "nccl_allreduce_dominated" for f in findings)


def test_nccl_serialization_finding():
    """NCCL > 20% of captured time triggers Root Cause #3 finding."""
    # captured = 1000ms, total NCCL = 300ms (30%) → should trigger
    rows = [
        _nccl_row(type="allreduce", total_ms=300.0, pct=100.0,
                  span_start_ns=0, span_end_ns=1_000_000_000),
    ]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    assert any(f.id == "nccl_serialization" for f in findings)
    f = next(f for f in findings if f.id == "nccl_serialization")
    assert f.severity == "warning"
    assert f.category == "communication"
    assert f.evidence[0].values["nccl_capture_pct"] == pytest.approx(30.0, abs=0.5)
    assert f.evidence[0].units["nccl_capture_pct"] == "percent"
    assert f.evidence[0].provenance["root_cause"] == "#3"
    assert "Root Cause #3" in f.explanation


def test_nccl_serialization_does_not_trigger_below_threshold():
    """NCCL < 20% of captured time should NOT trigger #3 finding."""
    # captured = 1000ms, total NCCL = 100ms (10%) → should NOT trigger
    rows = [
        _nccl_row(type="allreduce", total_ms=100.0, pct=100.0,
                  span_start_ns=0, span_end_ns=1_000_000_000),
    ]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    assert not any(f.id == "nccl_serialization" for f in findings)


def test_nccl_serialization_pct_clipped_at_100():
    """Per-stream sum can exceed captured time when NCCL overlaps across streams;
    pct must be clipped to 100% so the displayed value stays sane."""
    # captured = 100ms, 3 streams each running 100ms of NCCL → sum = 300ms (300%)
    # After clip, pct should be 100% (not 300%).
    rows = [
        _nccl_row(stream_id=7, type="sendrecv",  total_ms=100.0, pct=33.3,
                  span_start_ns=0, span_end_ns=100_000_000),
        _nccl_row(stream_id=8, type="allreduce", total_ms=100.0, pct=33.3,
                  span_start_ns=0, span_end_ns=100_000_000),
        _nccl_row(stream_id=9, type="allgather", total_ms=100.0, pct=33.4,
                  span_start_ns=0, span_end_ns=100_000_000),
    ]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    f = next(f for f in findings if f.id == "nccl_serialization")
    assert f.evidence[0].values["nccl_capture_pct"] == 100.0
