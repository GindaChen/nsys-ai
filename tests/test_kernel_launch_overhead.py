"""Tests for kernel_launch_overhead structured findings (src/nsys_ai/data/book.md Root Causes #5 and #10).

Uses the minimal_nsys_conn and duckdb_conn fixtures from conftest.py.
"""

import json

import pytest

from nsys_ai.annotation import EvidenceRow, Finding, TraceSelection
from nsys_ai.skills.builtins.kernel_launch_overhead import SKILL
from nsys_ai.skills.registry import get_skill

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def klo_skill():
    skill = get_skill("kernel_launch_overhead")
    assert skill is not None, "kernel_launch_overhead skill not registered"
    return skill


# ── _execute integration tests (SQL on real fixtures) ────────────


class TestExecuteIntegration:
    def test_returns_aggregated_rows(self, minimal_nsys_conn):
        rows = SKILL.execute_fn(minimal_nsys_conn, device=0, min_launches=1)
        assert len(rows) > 0, "Fixture should yield at least one aggregated kernel row"
        for r in rows:
            assert "kernel_name" in r
            assert "launch_count" in r
            assert "total_overhead_ms" in r
            assert "avg_overhead_us" in r
            assert "overhead_pct" in r

    def test_injects_device_and_span(self, minimal_nsys_conn):
        rows = SKILL.execute_fn(minimal_nsys_conn, device=0, min_launches=1)
        assert len(rows) > 0
        for r in rows:
            assert r["device_id"] == 0
            assert "span_start_ns" in r
            assert "span_end_ns" in r
            assert "_global_sync_count" in r

    def test_min_overhead_non_negative(self, minimal_nsys_conn):
        """Filtering runtime rows to launch APIs only is what actually prevents
        negative overheads — without that filter, non-launch APIs sharing a
        correlationId can produce r.start > k.start."""
        rows = SKILL.execute_fn(minimal_nsys_conn, device=0, min_launches=1)
        assert len(rows) > 0
        for r in rows:
            assert r["min_overhead_us"] >= 0, f"{r['kernel_name']}: negative overhead"

    def test_duckdb_path(self, duckdb_conn):
        rows = SKILL.execute_fn(duckdb_conn, device=0, min_launches=1)
        assert len(rows) > 0
        for r in rows:
            assert r["min_overhead_us"] >= 0


# ── Format tests ──────────────────────────────────────────────────


class TestFormat:
    def test_per_kernel_header(self, minimal_nsys_conn):
        rows = SKILL.execute_fn(minimal_nsys_conn, device=0, min_launches=1)
        text = SKILL.format_fn(rows)
        assert "per-kernel aggregated" in text

    def test_empty_returns_placeholder(self):
        assert "No kernel launch overhead data" in SKILL.format_fn([])


# ── Finding tests (v0.1) ─────────────────────────────────────────


def _klo_row(**overrides):
    """Build a synthetic per-kernel aggregated row matching _execute()'s schema."""
    row = {
        "kernel_name": "elementwise_kernel",
        "launch_count": 1000,
        "total_overhead_ms": 50.0,
        "avg_overhead_us": 50.0,
        "max_overhead_us": 100.0,
        "min_overhead_us": 5.0,
        "total_kernel_ms": 100.0,
        "overhead_pct": 33.3,
        "device_id": 0,
        "span_start_ns": 0,
        "span_end_ns": 1_000_000_000,
        "_global_sync_count": 0,
    }
    row.update(overrides)
    return row


# ── Finding 1: Small Kernel Overhead (Root Cause #5) ────────────────


def test_small_kernel_overhead_triggers():
    """Tiny kernel (<10μs avg) with overhead > kernel duration → #5 fires."""
    # total_kernel_ms=5, launch_count=1000 → avg_kernel_us = 5μs
    rows = [_klo_row(
        kernel_name="tiny_op",
        launch_count=1000,
        total_kernel_ms=5.0,
        total_overhead_ms=100.0,
        overhead_pct=95.0,
    )]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    assert any("small_kernel_overhead" in (f.id or "") for f in findings)
    f = next(f for f in findings if "small_kernel_overhead" in (f.id or ""))
    assert f.severity == "warning"
    assert f.category == "kernels"
    assert f.evidence[0].provenance["root_cause"] == "src/nsys_ai/data/book.md#5"
    assert f.evidence[0].units["avg_kernel_us"] == "microseconds"


def test_small_kernel_does_not_trigger_when_kernel_large():
    """avg_kernel_us >= 10 → should NOT fire."""
    # total_kernel_ms=100, launch_count=1000 → avg_kernel_us = 100μs (too big)
    rows = [_klo_row(
        kernel_name="medium_op",
        launch_count=1000,
        total_kernel_ms=100.0,
        total_overhead_ms=500.0,
        overhead_pct=80.0,
    )]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    assert not any("small_kernel_overhead" in (f.id or "") for f in findings)


def test_small_kernel_triggers_regardless_of_overhead_pct():
    """Per reviewer feedback: overhead_pct is queue-depth-confounded in async
    workloads (near-100% for nearly every small kernel), so it is NOT used as
    a gate. A tiny+frequent kernel must fire even with low overhead_pct."""
    rows = [_klo_row(
        kernel_name="tiny_op",
        launch_count=1000,
        total_kernel_ms=5.0,        # avg_kernel_us = 5μs (tiny)
        overhead_pct=30.0,           # low — but no longer a gate
    )]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    assert any("small_kernel_overhead" in (f.id or "") for f in findings)


def test_small_kernel_does_not_trigger_when_launch_count_low():
    """launch_count < 100 → low statistical confidence, should NOT fire."""
    rows = [_klo_row(
        kernel_name="rare_op",
        launch_count=50,
        total_kernel_ms=0.25,        # avg_kernel_us = 5μs
        overhead_pct=95.0,
    )]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    assert not any("small_kernel_overhead" in (f.id or "") for f in findings)


# ── Finding 2: Excessive Synchronization (Root Cause #12) ───────────


def test_excessive_sync_triggers():
    """sync_count > 100 → #12 fires."""
    rows = [_klo_row(_global_sync_count=500)]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    assert any(f.id == "klo_excessive_sync" for f in findings)
    f = next(f for f in findings if f.id == "klo_excessive_sync")
    assert f.severity == "warning"
    assert f.category == "kernels"
    assert f.evidence[0].values["sync_count"] == 500
    assert f.evidence[0].units["sync_count"] == "count"
    assert f.evidence[0].provenance["root_cause"] == "src/nsys_ai/data/book.md#10"
    assert "src/nsys_ai/data/book.md Root Cause #10" in f.explanation


def test_excessive_sync_does_not_trigger_below_threshold():
    """sync_count <= 100 → should NOT fire."""
    rows = [_klo_row(_global_sync_count=50)]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    assert not any(f.id == "klo_excessive_sync" for f in findings)


def test_excessive_sync_fires_when_no_kernel_data():
    """Per reviewer feedback: sync finding must fire even when no kernels
    meet min_launches (sync_count is independent of kernel data).
    Simulates _execute output with a metadata-only synthetic row."""
    rows = [{
        "kernel_name": None,
        "launch_count": 0,
        "total_overhead_ms": 0.0,
        "avg_overhead_us": 0.0,
        "max_overhead_us": 0.0,
        "min_overhead_us": 0.0,
        "total_kernel_ms": 0.0,
        "overhead_pct": 0.0,
        "device_id": 0,
        "span_start_ns": 0,
        "span_end_ns": 1_000_000_000,
        "_global_sync_count": 500,
    }]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    assert any(f.id == "klo_excessive_sync" for f in findings)
    # Small kernel finding must NOT fire (no real kernel data)
    assert not any("small_kernel_overhead" in (f.id or "") for f in findings)


# ── Edge cases ─────────────────────────────────────────────────────


def test_empty_rows_returns_empty_findings():
    findings = SKILL.to_findings_fn([], context={"profile_id": "test"})
    assert findings == []


def test_findings_without_context_use_unknown_profile_id():
    rows = [_klo_row(_global_sync_count=500)]
    findings = SKILL.to_findings_fn(rows)
    assert findings
    assert findings[0].selection.profile_id == "unknown"


def test_findings_propagate_device_id_from_rows():
    """Per reviewer feedback on nccl_breakdown: device must come from rows, not hardcoded."""
    rows = [_klo_row(_global_sync_count=500, device_id=3)]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    assert findings
    assert findings[0].gpu_id == 3
    assert findings[0].selection.gpu_ids == [3]


def test_findings_use_span_from_rows():
    """start_ns/end_ns should come from injected span, not be hardcoded 0."""
    rows = [_klo_row(
        _global_sync_count=500,
        span_start_ns=12345,
        span_end_ns=67890,
    )]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    assert findings
    assert findings[0].start_ns == 12345
    assert findings[0].end_ns == 67890


# ── JSON round-trip ───────────────────────────────────────────────


def test_json_round_trip():
    rows = [_klo_row(
        kernel_name="tiny_op",
        launch_count=1000,
        total_kernel_ms=5.0,
        overhead_pct=95.0,
        _global_sync_count=500,
    )]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "test"})
    assert findings
    for f in findings:
        d = f.to_dict()
        assert isinstance(d["selection"], dict)
        assert isinstance(d["evidence"], list)
        restored = Finding.from_dict(json.loads(json.dumps(d)))
        assert restored.id == f.id
        assert restored.category == "kernels"
        assert isinstance(restored.selection, TraceSelection)
        assert isinstance(restored.evidence[0], EvidenceRow)


# ── L40S realistic input ──────────────────────────────────────────


def test_l40s_realistic_input():
    """Mirrors the L40S perf.sqlite top kernels (post-SQL fix)."""
    rows = [
        # Small kernels — should trigger Finding 1
        _klo_row(
            kernel_name="elementwise_kernel_with_index",
            launch_count=1764,
            total_kernel_ms=2.07,         # avg_kernel_us = 1.17μs (tiny)
            avg_overhead_us=494952.2,
            overhead_pct=99.998,
            _global_sync_count=500,
        ),
        # Large kernel — should NOT trigger Finding 1
        _klo_row(
            kernel_name="flash_fwd_kernel",
            launch_count=43200,
            total_kernel_ms=670693.09,     # avg_kernel_us = 15523μs (big)
            avg_overhead_us=29207.7,
            overhead_pct=65.3,
            _global_sync_count=500,
        ),
    ]
    findings = SKILL.to_findings_fn(rows, context={"profile_id": "l40s_test"})
    assert any("small_kernel_overhead_elementwise_kernel_with_index" in (f.id or "") for f in findings)
    assert not any("small_kernel_overhead_flash_fwd_kernel" in (f.id or "") for f in findings)
    assert any(f.id == "klo_excessive_sync" for f in findings)


# ── Skill registration sanity ────────────────────────────────────


def test_skill_registered(klo_skill):
    assert klo_skill.name == "kernel_launch_overhead"
    assert klo_skill.to_findings_fn is not None
    assert klo_skill.execute_fn is not None
    assert klo_skill.format_fn is not None
