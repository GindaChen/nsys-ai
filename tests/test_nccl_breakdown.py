"""Tests for per-stream NCCL breakdown (Issue #24).

Uses the minimal_nsys_conn and duckdb_conn fixtures from conftest.py.

Test data recap (from conftest.py):
  Stream 7: nccl_ReduceScatter [4.5-5.5ms] (same-stream anti-pattern)
  Stream 8: nccl_AllReduce [2.5-3.5ms]
"""

import pytest

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
