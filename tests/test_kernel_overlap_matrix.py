"""Tests for the kernel_overlap_matrix skill (Issue #23).

Uses the minimal_nsys_conn and duckdb_conn fixtures from conftest.py.

Test data recap (from conftest.py):
  Stream 7: kernel_A [1-2ms], kernel_B [3-4ms], nccl_ReduceScatter [4.5-5.5ms], kernel_A [8-9ms]
  Stream 8: nccl_AllReduce [2.5-3.5ms]
  Memcpy H2D: [0.1-0.2ms], [0.3-0.4ms], [2.1-2.2ms], [5.1-5.2ms], [23.1-23.4ms]
"""

import pytest

from nsys_ai.skills.registry import get_skill


@pytest.fixture
def overlap_skill():
    return get_skill("kernel_overlap_matrix")


def _find_pair(rows, cat_a, cat_b):
    """Find the row matching the given category pair (order-insensitive)."""
    for r in rows:
        a, b = r["category_a"], r["category_b"]
        if (a == cat_a and b == cat_b) or (a == cat_b and b == cat_a):
            return r
    return None


class TestResultStructure:
    def test_returns_list_of_dicts(self, minimal_nsys_conn, overlap_skill):
        rows = overlap_skill.execute(minimal_nsys_conn, device=0)
        assert isinstance(rows, list)
        assert len(rows) > 0
        assert isinstance(rows[0], dict)

    def test_required_keys(self, minimal_nsys_conn, overlap_skill):
        rows = overlap_skill.execute(minimal_nsys_conn, device=0)
        required = {
            "category_a", "category_b", "overlap_ns", "overlap_ms",
            "is_diagonal", "pct_of_a", "pct_of_b",
        }
        for r in rows:
            assert required <= set(r.keys()), f"Missing keys in: {r}"

    def test_diagonal_entries_exist(self, minimal_nsys_conn, overlap_skill):
        rows = overlap_skill.execute(minimal_nsys_conn, device=0)
        diagonals = [r for r in rows if r["is_diagonal"]]
        assert len(diagonals) >= 2  # at least compute + one NCCL type
        for d in diagonals:
            assert d["category_a"] == d["category_b"]
            assert d["pct_of_a"] is None
            assert d["pct_of_b"] is None


class TestOverlapDetection:
    def test_compute_nccl_overlap_detected(self, minimal_nsys_conn, overlap_skill):
        """kernel_B [3-4ms] overlaps nccl_AllReduce [2.5-3.5ms] → 0.5ms overlap."""
        rows = overlap_skill.execute(minimal_nsys_conn, device=0)
        pair = _find_pair(rows, "compute", "nccl_allreduce")
        assert pair is not None, f"compute × nccl_allreduce not found in {[r['category_a'] + '×' + r['category_b'] for r in rows]}"
        assert not pair["is_diagonal"]
        # Overlap should be ~0.5ms (500,000 ns)
        assert pair["overlap_ns"] == pytest.approx(500_000, abs=50_000)
        assert pair["overlap_ms"] == pytest.approx(0.5, abs=0.1)

    def test_comm_comm_no_overlap(self, minimal_nsys_conn, overlap_skill):
        """AllReduce [2.5-3.5ms] and ReduceScatter [4.5-5.5ms] don't overlap."""
        rows = overlap_skill.execute(minimal_nsys_conn, device=0)
        pair = _find_pair(rows, "nccl_allreduce", "nccl_reducescatter")
        assert pair is not None
        assert pair["overlap_ns"] == 0
        assert pair["overlap_ms"] == 0

    def test_diagonal_is_self_time(self, minimal_nsys_conn, overlap_skill):
        """Compute diagonal = total compute kernel time (3 kernels: 1+1+1 = 3ms)."""
        rows = overlap_skill.execute(minimal_nsys_conn, device=0)
        pair = _find_pair(rows, "compute", "compute")
        assert pair is not None
        assert pair["is_diagonal"]
        # kernel_A [1-2ms] + kernel_B [3-4ms] + kernel_A [8-9ms] = 3ms
        assert pair["overlap_ns"] == pytest.approx(3_000_000, abs=50_000)

    def test_overlap_percentage(self, minimal_nsys_conn, overlap_skill):
        """pct_of_b for compute×allreduce should be ~50% (0.5ms out of 1ms)."""
        rows = overlap_skill.execute(minimal_nsys_conn, device=0)
        pair = _find_pair(rows, "compute", "nccl_allreduce")
        assert pair is not None
        # AllReduce total = 1ms, overlap = 0.5ms → 50%
        assert pair["pct_of_b"] == pytest.approx(50.0, abs=5.0)


class TestMemcpy:
    def test_memcpy_in_matrix(self, minimal_nsys_conn, overlap_skill):
        """H2D memcpy should appear as a category in the matrix."""
        rows = overlap_skill.execute(minimal_nsys_conn, device=0)
        categories = {r["category_a"] for r in rows} | {r["category_b"] for r in rows}
        assert "memcpy_h2d" in categories

    def test_memcpy_has_self_time(self, minimal_nsys_conn, overlap_skill):
        rows = overlap_skill.execute(minimal_nsys_conn, device=0)
        pair = _find_pair(rows, "memcpy_h2d", "memcpy_h2d")
        assert pair is not None
        assert pair["is_diagonal"]
        assert pair["overlap_ns"] > 0


class TestEdgeCases:
    def test_empty_device_returns_error(self, minimal_nsys_conn, overlap_skill):
        """Device 99 has no kernels — should return error dict."""
        rows = overlap_skill.execute(minimal_nsys_conn, device=99)
        assert len(rows) == 1
        assert "error" in rows[0]

    def test_trim_filters_kernels(self, minimal_nsys_conn, overlap_skill):
        """Trim to [1ms, 4ms] should exclude nccl_ReduceScatter and late kernel."""
        rows = overlap_skill.execute(
            minimal_nsys_conn,
            device=0,
            trim_start_ns=1_000_000,
            trim_end_ns=4_000_000,
        )
        categories = {r["category_a"] for r in rows} | {r["category_b"] for r in rows}
        assert "compute" in categories
        # ReduceScatter starts at 4.5ms, outside trim
        assert "nccl_reducescatter" not in categories


class TestFormatFn:
    def test_format_produces_ascii_matrix(self, minimal_nsys_conn, overlap_skill):
        rows = overlap_skill.execute(minimal_nsys_conn, device=0)
        text = overlap_skill.format_rows(rows)
        assert "Kernel Overlap Matrix" in text
        assert "compute" in text
        assert "ms" in text

    def test_format_handles_error(self, minimal_nsys_conn, overlap_skill):
        rows = overlap_skill.execute(minimal_nsys_conn, device=99)
        text = overlap_skill.format_rows(rows)
        assert "error" in text.lower() or "no kernels" in text.lower()


class TestDuckDBPath:
    def test_duckdb_basic(self, duckdb_conn, overlap_skill):
        """Same basic assertions pass on DuckDB fixture."""
        rows = overlap_skill.execute(duckdb_conn, device=0)
        assert isinstance(rows, list)
        assert len(rows) > 0
        # Check compute × allreduce overlap exists
        pair = _find_pair(rows, "compute", "nccl_allreduce")
        assert pair is not None
        assert pair["overlap_ns"] == pytest.approx(500_000, abs=50_000)
