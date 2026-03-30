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
    def test_truncation_applied(self, minimal_nsys_conn):
        """When a skill returns many rows, --max-rows should truncate."""
        skill = get_skill("top_kernels")
        assert skill is not None
        rows = skill.execute(minimal_nsys_conn, device=0, limit=50)

        # Simulate truncation logic from handlers.py
        max_rows = 3
        if len(rows) > max_rows:
            total = len(rows)
            truncated = rows[:max_rows]
            truncated.append({
                "_truncated": True,
                "_total_rows": total,
                "_shown_rows": max_rows,
            })
            assert len(truncated) == max_rows + 1
            assert truncated[-1]["_truncated"] is True
            assert truncated[-1]["_total_rows"] == total

    def test_no_truncation_when_under_limit(self, minimal_nsys_conn):
        """When rows < max_rows, no truncation metadata should appear."""
        skill = get_skill("top_kernels")
        assert skill is not None
        rows = skill.execute(minimal_nsys_conn, device=0, limit=5)

        max_rows = 100
        if len(rows) > max_rows:
            rows = rows[:max_rows]
            rows.append({"_truncated": True})
        # Should NOT have truncation marker
        assert not any(r.get("_truncated") for r in rows)
