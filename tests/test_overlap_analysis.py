"""
Tests for ``overlap.overlap_analysis``.

Exercises both the SQL sweep-line path (``_overlap_analysis_sql``) and
the Python interval-merge fallback (``_overlap_analysis_python``), and
pins them to the same numeric output on the shared fixtures.

Test data recap (from conftest.py, device 0):
  Stream 7: kernel_A [1-2ms], kernel_B [3-4ms],
            nccl_ReduceScatter [4.5-5.5ms], kernel_A [8-9ms]
  Stream 8: nccl_AllReduce [2.5-3.5ms]

Expected merged intervals on device 0:
  compute_merged: [(1ms,2ms), (3ms,4ms), (8ms,9ms)] → 3ms total
  nccl_merged   : [(2.5ms,3.5ms), (4.5ms,5.5ms)]     → 2ms total
  overlap       : compute(3-4ms) ∩ nccl(2.5-3.5ms)   → 0.5ms
  span          : 1ms .. 9ms                          → 8ms total
"""

from __future__ import annotations

import pytest

from nsys_ai.overlap import (
    _no_kernels_diag,
    _overlap_analysis_python,
    _overlap_analysis_sql,
    overlap_analysis,
)
from nsys_ai.profile import Profile


def _expect_device_zero(result: dict) -> None:
    assert result["compute_kernels"] == 3
    assert result["nccl_kernels"] == 2
    assert result["compute_only_ms"] == pytest.approx(2.5, abs=0.01)
    assert result["nccl_only_ms"] == pytest.approx(1.5, abs=0.01)
    assert result["overlap_ms"] == pytest.approx(0.5, abs=0.01)
    assert result["total_ms"] == pytest.approx(8.0, abs=0.01)
    assert result["idle_ms"] == pytest.approx(3.5, abs=0.01)
    assert result["overlap_pct"] == pytest.approx(25.0, abs=0.5)
    assert result["span_start_ns"] == 1_000_000
    assert result["span_end_ns"] == 9_000_000


class TestSqlPath:
    def test_duckdb_sql_path_matches_expected(self, duckdb_conn):
        prof = Profile._from_conn(duckdb_conn)
        try:
            result = overlap_analysis(prof, device=0)
        finally:
            prof.close()
        _expect_device_zero(result)

    def test_sql_path_explicit_invocation(self, duckdb_conn):
        """The SQL helper, called directly, must return the same numbers
        as the public entry point."""
        prof = Profile._from_conn(duckdb_conn)
        try:
            result = _overlap_analysis_sql(prof, 0, None)
        finally:
            prof.close()
        assert result is not None
        _expect_device_zero(result)

    def test_sql_and_python_paths_agree(self, duckdb_conn):
        """Numerical parity between the two implementations on the shared
        fixture — this is the regression anchor for the SQL pushdown."""
        prof = Profile._from_conn(duckdb_conn)
        try:
            sql_res = _overlap_analysis_sql(prof, 0, None)
            py_res = _overlap_analysis_python(prof, 0, None)
        finally:
            prof.close()
        assert sql_res is not None
        for key in (
            "compute_only_ms",
            "nccl_only_ms",
            "overlap_ms",
            "idle_ms",
            "total_ms",
            "overlap_pct",
            "compute_kernels",
            "nccl_kernels",
            "span_start_ns",
            "span_end_ns",
        ):
            assert sql_res[key] == py_res[key], f"mismatch on {key}"

    def test_trim_window_filters_out_kernels(self, duckdb_conn):
        """Trim that includes only the second compute kernel + AllReduce."""
        prof = Profile._from_conn(duckdb_conn)
        try:
            # [2.5ms, 4ms] window covers kernel_B [3-4] and nccl_AllReduce
            # [2.5-3.5]. The Reduce on stream 7 [4.5-5.5] sits outside.
            sql_res = _overlap_analysis_sql(prof, 0, (2_500_000, 4_000_000))
            py_res = _overlap_analysis_python(prof, 0, (2_500_000, 4_000_000))
        finally:
            prof.close()
        assert sql_res is not None
        assert sql_res["compute_kernels"] == 1
        assert sql_res["nccl_kernels"] == 1
        # SQL and Python must agree under trim too.
        for key in (
            "compute_only_ms",
            "nccl_only_ms",
            "overlap_ms",
            "idle_ms",
            "total_ms",
            "compute_kernels",
            "nccl_kernels",
        ):
            assert sql_res[key] == py_res[key], f"mismatch on {key}"

    def test_returns_none_when_device_missing(self, duckdb_conn):
        """Caller relies on ``None`` to trigger the diagnostic payload."""
        prof = Profile._from_conn(duckdb_conn)
        try:
            sql_res = _overlap_analysis_sql(prof, device=99, trim=None)
        finally:
            prof.close()
        assert sql_res is None

    def test_compute_only_profile_zero_overlap_pct(self, duckdb_conn):
        """Trim that excludes every NCCL kernel must yield overlap_pct=0
        without a division-by-zero. Exercises the ``if nccl_ns else 0``
        guard inside the SQL path."""
        prof = Profile._from_conn(duckdb_conn)
        try:
            # [1ms, 2ms] only catches kernel_A (compute), no NCCL kernel
            # has start >= 1ms and end <= 2ms.
            sql_res = _overlap_analysis_sql(prof, 0, (1_000_000, 2_000_000))
            py_res = _overlap_analysis_python(prof, 0, (1_000_000, 2_000_000))
        finally:
            prof.close()
        assert sql_res is not None
        assert sql_res["nccl_kernels"] == 0
        assert sql_res["compute_kernels"] == 1
        assert sql_res["overlap_ms"] == 0
        assert sql_res["overlap_pct"] == 0
        assert sql_res["overlap_pct"] == py_res["overlap_pct"]


class TestPythonFallback:
    def test_sqlite_path_uses_python_implementation(self, minimal_nsys_conn):
        """SQLite-only profiles (no DuckDB) go through the Python path."""
        prof = Profile._from_conn(minimal_nsys_conn)
        try:
            result = overlap_analysis(prof, device=0)
        finally:
            prof.close()
        _expect_device_zero(result)


class TestDiagnostics:
    def test_no_kernels_returns_diagnostic(self, duckdb_conn):
        prof = Profile._from_conn(duckdb_conn)
        try:
            result = overlap_analysis(prof, device=99)
        finally:
            prof.close()
        assert result.get("error") == "no kernels found"
        assert result["requested_device"] == 99
        assert "hint" in result
        assert "available_devices" in result
        assert 0 in result["available_devices"]

    def test_no_kernels_diag_helper(self, duckdb_conn):
        prof = Profile._from_conn(duckdb_conn)
        try:
            diag = _no_kernels_diag(prof, device=99, trim=None)
        finally:
            prof.close()
        assert diag["error"] == "no kernels found"
        assert diag["available_devices"][0] == 5


class TestMemoization:
    """overlap_analysis is memoized per connection (issue #242): a build asks
    for the same (connection, device, trim) up to seven times and must compute
    the sweep only once."""

    def _count_computes(self, monkeypatch):
        import nsys_ai.overlap as ov

        calls = []
        real = ov._overlap_analysis_uncached
        monkeypatch.setattr(
            ov,
            "_overlap_analysis_uncached",
            lambda p, d, t: calls.append((d, t)) or real(p, d, t),
        )
        return calls

    def test_repeated_calls_compute_once(self, duckdb_conn, monkeypatch):
        calls = self._count_computes(monkeypatch)
        prof = Profile._from_conn(duckdb_conn)
        try:
            first = overlap_analysis(prof, device=0)
            for _ in range(6):
                again = overlap_analysis(prof, device=0)
            assert len(calls) == 1, "sweep recomputed despite identical args"
            assert again == first
        finally:
            prof.close()

    def test_cache_hit_equals_fresh_compute(self, duckdb_conn):
        prof = Profile._from_conn(duckdb_conn)
        try:
            fresh = _overlap_analysis_python(prof, 0, None)
            overlap_analysis(prof, device=0)  # populate cache
            cached = overlap_analysis(prof, device=0)
            for k, v in fresh.items():
                assert cached[k] == pytest.approx(v) if isinstance(v, float) else cached[k] == v
        finally:
            prof.close()

    def test_caller_mutation_does_not_corrupt_cache(self, duckdb_conn):
        """overlap_breakdown adds keys like device_id to the result; a shared
        cached dict would leak that into the next caller. Every returned dict is
        mutated here — both the miss-path result and a hit-path result — so a
        missing copy on *either* path corrupts the final read."""
        prof = Profile._from_conn(duckdb_conn)
        try:
            a = overlap_analysis(prof, device=0)  # miss -> compute
            baseline_idle = a["idle_ms"]
            a["device_id"] = 999  # mutate the miss-path result
            a["idle_ms"] = -1
            b = overlap_analysis(prof, device=0)  # hit
            b["injected"] = ["x"]  # mutate a hit-path result
            b["idle_ms"] = -2
            c = overlap_analysis(prof, device=0)  # hit -> must be pristine
            assert "device_id" not in c and "injected" not in c
            assert c["idle_ms"] == baseline_idle
        finally:
            prof.close()

    def test_distinct_trim_recomputes(self, duckdb_conn, monkeypatch):
        calls = self._count_computes(monkeypatch)
        prof = Profile._from_conn(duckdb_conn)
        try:
            overlap_analysis(prof, device=0, trim=None)
            overlap_analysis(prof, device=0, trim=(1_000_000, 9_000_000))
            overlap_analysis(prof, device=0, trim=None)  # back to first key -> hit
            assert len(calls) == 2, "trim must be part of the cache key"
        finally:
            prof.close()

    def test_shared_across_profile_wrappers_on_one_conn(self, duckdb_conn, monkeypatch):
        """gpu_idle_gaps reaches overlap_analysis through a throwaway
        Profile._from_conn around the same connection; the cache keys on the
        connection so that call reuses the entry rather than recomputing."""
        calls = self._count_computes(monkeypatch)
        p1 = Profile._from_conn(duckdb_conn)
        p2 = Profile._from_conn(duckdb_conn)  # distinct Profile, same conn
        try:
            overlap_analysis(p1, device=0)
            overlap_analysis(p2, device=0)
            assert len(calls) == 1, "cache must key on the connection, not the Profile"
        finally:
            p1.close()
            p2.close()
