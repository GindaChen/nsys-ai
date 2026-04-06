"""Tests for profile_health_manifest skill and --max-rows truncation.

Uses the minimal_nsys_conn and duckdb_conn fixtures from conftest.py.
"""

import pytest

from nsys_ai.skills.registry import get_skill

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def manifest_skill():
    skill = get_skill("profile_health_manifest")
    assert skill is not None, "profile_health_manifest skill not registered"
    return skill


# ── Manifest skill tests ─────────────────────────────────────────


class TestManifestExecute:
    def test_returns_single_row(self, minimal_nsys_conn, manifest_skill):
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        assert isinstance(rows, list)
        assert len(rows) == 1

    def test_has_required_keys(self, minimal_nsys_conn, manifest_skill):
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        m = rows[0]
        required_keys = [
            "gpu",
            "profile_span_ms",
            "top_kernels",
            "total_kernel_ms",
            "overlap",
            "nccl",
            "idle",
            "root_cause_count",
            "root_causes",
            "data_quality",
        ]
        for key in required_keys:
            assert key in m, f"Missing key '{key}' in manifest"

    def test_top_kernels_is_list(self, minimal_nsys_conn, manifest_skill):
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        m = rows[0]
        assert isinstance(m["top_kernels"], list)

    def test_nccl_summary_has_streams(self, minimal_nsys_conn, manifest_skill):
        """Our test data has NCCL kernels, so streams > 0."""
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        m = rows[0]
        assert m["nccl"]["streams"] > 0

    def test_root_causes_capped(self, minimal_nsys_conn, manifest_skill):
        """root_causes list should never exceed 5 entries."""
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        m = rows[0]
        assert len(m["root_causes"]) <= 5

    def test_empty_device_returns_manifest(self, minimal_nsys_conn, manifest_skill):
        """Device with no data should still return a valid manifest."""
        rows = manifest_skill.execute(minimal_nsys_conn, device=99)
        assert isinstance(rows, list)
        assert len(rows) == 1
        m = rows[0]
        assert m["nccl"]["streams"] == 0

    def test_data_quality_metrics(self, minimal_nsys_conn, manifest_skill):
        """Ensure data_quality metrics are properly computed."""
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        dq = rows[0]["data_quality"]

        assert "profiler_overhead_ms" in dq
        assert "overhead_pct" in dq

        assert isinstance(dq["profiler_overhead_ms"], (int, float))
        assert isinstance(dq["overhead_pct"], (int, float))


class TestManifestFormat:
    def test_format_includes_header(self, minimal_nsys_conn, manifest_skill):
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        text = manifest_skill.format_rows(rows)
        assert "Profile Health Manifest" in text

    def test_format_includes_gpu(self, minimal_nsys_conn, manifest_skill):
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        text = manifest_skill.format_rows(rows)
        assert "GPU:" in text


class TestManifestDuckDB:
    def test_duckdb_path(self, duckdb_conn, manifest_skill):
        rows = manifest_skill.execute(duckdb_conn, device=0)
        assert isinstance(rows, list)
        assert len(rows) == 1
        assert "gpu" in rows[0]


# ── Token budget protection (--max-rows) ─────────────────────────


class TestMaxRowsTruncation:
    def test_truncation_applied(self):
        from nsys_ai.cli.handlers import _apply_max_rows_truncation

        rows = [{"id": i} for i in range(10)]
        max_rows = 3
        truncated = _apply_max_rows_truncation(rows, max_rows)
        assert len(truncated) == max_rows + 1
        assert truncated[-1]["_truncated"] is True
        assert truncated[-1]["_total_rows"] == 10
        assert truncated[-1]["_shown_rows"] == 3

    def test_no_truncation_when_under_limit(self):
        from nsys_ai.cli.handlers import _apply_max_rows_truncation

        rows = [{"id": i} for i in range(3)]
        truncated = _apply_max_rows_truncation(rows, 100)
        assert len(truncated) == 3
        assert not any(r.get("_truncated") for r in truncated)

    def test_negative_max_rows_raises(self):
        from nsys_ai.cli.handlers import _apply_max_rows_truncation

        rows = [{"id": i} for i in range(3)]
        with pytest.raises(ValueError, match="non-negative integer"):
            _apply_max_rows_truncation(rows, -1)

    def test_error_payload_not_truncated_when_max_rows_zero(self):
        """Error payloads (e.g., [{'error': ...}]) should not be dropped by max-rows."""
        from nsys_ai.cli.handlers import _apply_max_rows_truncation

        rows = [{"error": "Something went wrong"}]
        truncated = _apply_max_rows_truncation(rows, 0)

        # The single error row should be preserved and not replaced by a truncation marker.
        assert isinstance(truncated, list)
        assert len(truncated) == 1
        assert "error" in truncated[0]
        assert truncated[0]["error"] == "Something went wrong"
        # Error payloads should not be marked as truncated.
        assert not truncated[0].get("_truncated")
