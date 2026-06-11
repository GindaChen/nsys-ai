"""Unit tests for the `doctor` diagnostics (nsys_ai.doctor)."""

from __future__ import annotations

import sqlite3
import types

import pytest

from nsys_ai.doctor import (
    DoctorReport,
    _has_overhead_table,
    _overhead_check,
    _profiler_overhead_pct,
    format_doctor_text,
    run_doctor,
)


def _section(report: DoctorReport, name: str):
    return next((s for s in report.sections if s.name == name), None)


def _check(report: DoctorReport, name: str):
    return next((c for c in report.all_checks() if c.name == name), None)


# ---------------------------------------------------------------------------
# Environment-only mode (no profile)
# ---------------------------------------------------------------------------


def test_env_only_report_shape():
    report = run_doctor()
    assert report.schema_version == "0.1"
    assert report.producer == "nsys-ai"
    assert report.profile_path is None
    assert report.profile_id is None
    names = [s.name for s in report.sections]
    assert names == ["System", "Profile support", "Optional features"]


def test_env_checks_present():
    report = run_doctor()
    assert _check(report, "Python") is not None
    assert _check(report, "SQLite analysis").status == "ok"
    # AI provider + CUTracer are always reported (configured or not).
    assert _check(report, "AI provider (litellm)") is not None
    assert _check(report, "CUTracer") is not None


def test_to_dict_roundtrip_and_summary():
    report = run_doctor()
    d = report.to_dict()
    assert d["schema_version"] == "0.1"
    assert "summary" in d
    total = sum(d["summary"].values())
    assert total == len(report.all_checks())
    # Each check serializes name/status/detail/hint/sub.
    first = d["sections"][0]["checks"][0]
    assert {"name", "status", "detail", "hint", "sub"} <= set(first)


def test_check_names_are_clean_for_machine_consumers():
    # Nesting is carried by the `sub` flag, never by leading/trailing spaces in
    # the name — so the web GUI / plugin can match names directly.
    for c in run_doctor().all_checks():
        assert c.name == c.name.strip()
        assert isinstance(c.sub, bool)


def test_text_render_includes_capture_hint():
    text = format_doctor_text(run_doctor())
    assert "Summary:" in text
    # Points users at nsys for capture-side checks (we only cover analysis).
    assert "nsys status -e" in text


def test_include_env_false_skips_environment():
    report = run_doctor(include_env=False, include_health=False)
    assert report.sections == []


# ---------------------------------------------------------------------------
# Profile-health mode
# ---------------------------------------------------------------------------


def test_health_section_on_minimal_profile(minimal_nsys_db_path):
    report = run_doctor(minimal_nsys_db_path)
    health = _section(report, "Profile health")
    assert health is not None
    assert report.profile_path == minimal_nsys_db_path
    # profile_id is content-derived when Nsight metadata is reachable.
    assert report.profile_id is None or report.profile_id.startswith("nsys1:")
    by_name = {c.name: c for c in health.checks}
    assert "Duration" in by_name
    assert "GPUs" in by_name
    assert "RunSpec attached" in by_name


def test_missing_profile_is_reported_as_failure(tmp_path):
    missing = str(tmp_path / "does_not_exist.sqlite")
    report = run_doctor(missing)
    assert report.has_failures()
    health = _section(report, "Profile health")
    assert health is not None
    assert any(c.status == "fail" for c in health.checks)


def test_health_only_mode_has_no_env(minimal_nsys_db_path):
    report = run_doctor(minimal_nsys_db_path, include_env=False)
    assert [s.name for s in report.sections] == ["Profile health"]


# ---------------------------------------------------------------------------
# Profiler-overhead thresholds (pure function)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pct, expected",
    [
        (None, "skipped"),  # no overhead table
        (5.0, "ok"),  # below warn
        (10.0, "ok"),  # exactly at warn boundary (strictly-greater triggers)
        (15.0, "warn"),  # warn band
        (20.0, "warn"),  # exactly at fail boundary (strictly-greater triggers)
        (25.0, "fail"),  # fail band
        (150.0, "skipped"),  # physically impossible -> unreliable, not a failure
    ],
)
def test_overhead_check_thresholds(pct, expected):
    assert _overhead_check(pct).status == expected


def test_has_overhead_table_matches_either_case():
    # SQLite (PROFILER_OVERHEAD) and the DuckDB cache (profiler_overhead) both count.
    for name in ("PROFILER_OVERHEAD", "profiler_overhead"):
        prof = types.SimpleNamespace(schema=types.SimpleNamespace(tables=[name]))
        assert _has_overhead_table(prof) is True
    prof = types.SimpleNamespace(schema=types.SimpleNamespace(tables=["NVTX_EVENTS"]))
    assert _has_overhead_table(prof) is False


def test_overhead_pct_is_none_without_table():
    """An absent overhead table is unmeasurable, not a measured 0%."""
    prof = types.SimpleNamespace(schema=types.SimpleNamespace(tables=["NVTX_EVENTS"]))
    assert _profiler_overhead_pct(prof, span_ns=1_000_000) is None


def test_overhead_skipped_when_profile_has_no_overhead_table(minimal_nsys_db_path):
    """The minimal profile has no overhead table -> skipped, not a false 'ok 0.0%'."""
    # Guard the premise: the fixture genuinely lacks the table.
    with sqlite3.connect(minimal_nsys_db_path) as conn:
        names = {
            r[0].lower()
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "profiler_overhead" not in names

    report = run_doctor(minimal_nsys_db_path, include_env=False)
    overhead = _check(report, "Profiler overhead")
    assert overhead is not None
    assert overhead.status == "skipped"
