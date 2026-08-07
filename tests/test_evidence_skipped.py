"""An analysis that could not run has to survive into the report.

`abstain()` distinguishes "ran, found nothing" from "could not run", and that
distinction reached `skill run` and `Agent.analyze` — but not `EvidenceBuilder`,
which is the path behind `analyze --format json`, `evidence build`, the web
`/api/analyze` endpoint. There the abstention row
was dropped before `to_findings_fn` and nothing else was recorded, so a profile
missing a table it needs produced a report identical in shape to a clean one.
Findings and no explanation is the failure mode a verify-first tool can least
afford, so the abstentions now travel beside the findings as `skipped`.

They travel *beside* them, not among them: a finding asserts something about the
profile's performance, an abstention asserts something about the analysis's
coverage, and ranking the second among the first would put bookkeeping in the
middle of a priority list.
"""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

from nsys_ai.annotation import EvidenceReport, Finding, SkippedAnalysis
from nsys_ai.evidence_builder import EvidenceBuilder

# No NVTX_EVENTS table at all, so `iteration_timing` — a pipeline entry —
# genuinely cannot run. `tests/test_no_nvtx_profile.py` pins the premise.
NO_NVTX = Path(__file__).resolve().parent / "fixtures" / "healthy_1pct.sqlite"

# The profile #338 opens with: everything present except the memcpy tables.
ANNOTATED = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"


@pytest.fixture(scope="module")
def no_memcpy_profile(tmp_path_factory):
    """A copy of the annotated fixture with its memcpy tables removed.

    Copied and stripped rather than shipped as a second fixture so the premise
    stays visible: this is the same profile as `ANNOTATED` minus one table, so
    a difference in the report is attributable to that table and nothing else.
    """
    import shutil
    import sqlite3

    dst = tmp_path_factory.mktemp("no_memcpy") / "no_memcpy.sqlite"
    shutil.copy(ANNOTATED, dst)
    conn = sqlite3.connect(str(dst))
    try:
        conn.execute("DROP TABLE CUPTI_ACTIVITY_KIND_MEMCPY")
        conn.execute("DROP TABLE ENUM_CUDA_MEMCPY_OPER")
        conn.commit()
    finally:
        conn.close()
    return dst


def _build(path) -> EvidenceReport:
    from nsys_ai import profile as profile_module

    with profile_module.open(str(path)) as prof:
        return EvidenceBuilder(prof, device=0).build()


def _capture(handler, args):
    from nsys_ai import profile as profile_module

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        handler(args, profile_module)
    return out.getvalue(), err.getvalue()


# ── The report says what it could not analyse ──────────────────────────────


def test_an_abstaining_analysis_reaches_the_report():
    report = _build(NO_NVTX)

    skipped = {s.analyzer: s for s in report.skipped}
    assert "slow_iterations" in skipped, (
        f"iteration_timing abstains on a profile with no NVTX, but the report "
        f"lists only {sorted(skipped)}"
    )
    entry = skipped["slow_iterations"]
    assert entry.skill == "iteration_timing", entry
    assert "NVTX" in entry.reason, entry.reason
    # The abstention explains a gap in coverage; it must not be dressed up as
    # something found in the profile.
    assert all(
        "iteration_timing" not in (f.label or "") for f in report.findings
    ), "an abstention leaked into the findings list"


def test_a_profile_with_no_memcpy_table_names_the_memory_analyses(no_memcpy_profile):
    """The case #338 opens with, end to end.

    On this profile the memcpy-backed analyses cannot run at all. Before this
    change they abstained and the report had nowhere to say so — the rows were
    dropped before `to_findings_fn` and nothing else was recorded, so the
    output was findings and no explanation.
    """
    report = _build(no_memcpy_profile)

    assert report.skipped, "a profile with no memcpy table reported nothing skipped"
    by_analyzer = {s.analyzer: s for s in report.skipped}
    assert "memory_anomalies" in by_analyzer, sorted(by_analyzer)
    assert "h2d_spikes" in by_analyzer, sorted(by_analyzer)
    assert by_analyzer["memory_anomalies"].skill == "memory_bandwidth"
    assert by_analyzer["h2d_spikes"].skill == "h2d_distribution"
    for analyzer in ("memory_anomalies", "h2d_spikes"):
        assert "MEMCPY" in by_analyzer[analyzer].reason, by_analyzer[analyzer].reason
    # Still a usable report: the analyses that could run still ran.
    assert report.findings, "dropping one table should not empty the report"


def test_the_memcpy_skills_abstain_rather_than_returning_empty(no_memcpy_profile):
    """The premise this file rests on, pinned so it cannot drift.

    Nothing here is this change's doing — the abstentions come from
    `Skill.execute`'s unresolved-table guard and `memory_bandwidth`'s own
    check, both already in place. But `skipped` can only carry what the skills
    emit, so if these three stopped abstaining the report assertions above
    would go vacuous rather than fail.
    """
    from nsys_ai import profile as profile_module
    from nsys_ai.skills.base import is_abstention
    from nsys_ai.skills.registry import get_skill

    with profile_module.open(str(no_memcpy_profile)) as prof:
        conn = prof.query_conn()
        for name in ("memory_transfers", "memory_bandwidth", "h2d_distribution"):
            rows = get_skill(name).execute(conn, device=0)
            assert is_abstention(rows), f"{name} did not abstain: {rows[:1]}"


def test_no_abstention_is_dropped_silently():
    """The regression guard: every pipeline skill that abstains is listed.

    Written against the pipeline rather than one known skill, so a future entry
    that starts abstaining cannot quietly go missing from the report.
    """
    from nsys_ai import profile as profile_module
    from nsys_ai.skills.base import is_abstention
    from nsys_ai.skills.registry import get_skill

    abstaining = set()
    with profile_module.open(str(NO_NVTX)) as prof:
        builder = EvidenceBuilder(prof, device=0)
        conn = prof.query_conn()
        for analyzer_name, (skill_name, params) in builder._SKILL_PIPELINE.items():
            skill = get_skill(skill_name)
            if skill is None:
                continue
            kwargs = {**params, "device": 0}
            if builder.trim:
                kwargs["trim_start_ns"] = builder.trim[0]
                kwargs["trim_end_ns"] = builder.trim[1]
            try:
                rows = skill.execute(conn, **kwargs)
            except Exception:
                continue
            if is_abstention(rows):
                abstaining.add(analyzer_name)
        report = builder.build()

    assert abstaining, "fixture no longer makes any pipeline analyzer abstain"
    assert abstaining <= {s.analyzer for s in report.skipped}


def test_a_report_lists_each_analyzer_once():
    """One entry per analyzer: the report is a coverage list, not a log.

    Two pipeline entries share `kernel_instances`, so keying on the skill would
    collapse them and lose which analyzer went uncovered; keying on the
    analyzer cannot repeat, because the pipeline is a dict.
    """
    report = _build(NO_NVTX)
    names = [s.analyzer for s in report.skipped]
    assert len(names) == len(set(names)), names


def test_a_profile_where_nothing_abstains_is_unchanged(minimal_nsys_db_path):
    report = _build(minimal_nsys_db_path)
    assert report.skipped == []
    assert report.findings, "fixture stopped producing findings; test is vacuous"


# ── The JSON envelope and the text renderer ────────────────────────────────


def test_analyze_json_emits_skipped_and_names_it_on_stderr():
    from nsys_ai.cli.handlers import _cmd_analyze

    args = SimpleNamespace(
        profile=str(NO_NVTX), gpu=0, trim=None, output=None, format="json"
    )
    stdout, stderr = _capture(_cmd_analyze, args)

    payload = json.loads(stdout)
    listed = {entry["analyzer"]: entry for entry in payload["skipped"]}
    assert "slow_iterations" in listed, payload["skipped"]
    assert listed["slow_iterations"]["skill"] == "iteration_timing"
    assert listed["slow_iterations"]["reason"]
    # stdout stays a single JSON document, so the readable notice is on stderr.
    assert "slow_iterations (iteration_timing) — skipped:" in stderr, stderr


def test_analyze_json_on_a_clean_profile_prints_no_skipped_notice(minimal_nsys_db_path):
    from nsys_ai.cli.handlers import _cmd_analyze

    args = SimpleNamespace(
        profile=str(minimal_nsys_db_path), gpu=0, trim=None, output=None, format="json"
    )
    stdout, stderr = _capture(_cmd_analyze, args)

    assert json.loads(stdout)["skipped"] == []
    assert "Skipped" not in stderr, stderr


def test_evidence_build_text_prints_a_skipped_section():
    from nsys_ai.cli.handlers import _cmd_evidence

    args = SimpleNamespace(
        profile=str(NO_NVTX),
        evidence_action="build",
        format="text",
        analyzers=None,
        trim=None,
        gpu=0,
        output=None,
    )
    stdout, _ = _capture(_cmd_evidence, args)

    assert "── Evidence Findings" in stdout
    assert "── Skipped (" in stdout
    assert "slow_iterations (iteration_timing) — skipped:" in stdout
    # After the findings, not among them.
    assert stdout.index("── Evidence Findings") < stdout.index("── Skipped (")


# ── Additive: consumers that ignore the field keep working ─────────────────


def test_skipped_is_keyword_only():
    """The class convention: envelope fields are ``kw_only``.

    Not because a four-positional caller exists to protect — a fourth
    positional argument was a ``TypeError`` before this field too — but so the
    next field is not inserted ahead of ``findings``, where it would rebind
    the callers that do pass three.
    """
    with pytest.raises(TypeError):
        EvidenceReport("T", "/p", [], [SkippedAnalysis("a", "s", "r")])  # type: ignore[misc]

    report = EvidenceReport("T", "/some/profile.sqlite", [])
    assert report.profile_path == "/some/profile.sqlite"
    assert report.skipped == []


def test_findings_serialization_is_untouched():
    """Evidence surfaces read only ``report.findings``; skipped must not leak."""
    f = Finding(type="region", label="L", start_ns=10)
    report = EvidenceReport(
        title="T",
        findings=[f],
        skipped=[SkippedAnalysis("slow_iterations", "iteration_timing", "no NVTX")],
    )
    assert [d for d in (f.to_dict() for f in report.findings) if "skipped" in d] == []
    assert report.to_dict()["findings"] == [f.to_dict()]


def test_evidence_builder_ignores_skipped_and_still_ranks_findings():
    from nsys_ai import profile as profile_module
    from nsys_ai.evidence_builder import EvidenceBuilder
    from nsys_ai.loop_state import _normalize_findings

    with profile_module.open(str(NO_NVTX)) as prof:
        report = EvidenceBuilder(prof, device=0).build()
        ranked = _normalize_findings([f.to_dict() for f in report.findings])

    assert isinstance(ranked, list)
    assert len(ranked) == len(report.findings)


# ── Round-trip ─────────────────────────────────────────────────────────────


def test_skipped_round_trips_through_json():
    entry = SkippedAnalysis("slow_iterations", "iteration_timing", "no NVTX_EVENTS table")
    report = EvidenceReport(title="T", profile_path="/p", skipped=[entry])
    restored = EvidenceReport.from_dict(json.loads(json.dumps(report.to_dict())))
    assert restored.skipped == [entry]


def test_a_payload_written_before_the_field_existed_loads_empty():
    restored = EvidenceReport.from_dict(
        {"title": "Legacy", "profile_path": "/x.sqlite", "findings": []}
    )
    assert restored.skipped == []


def test_an_explicit_null_skipped_loads_empty():
    restored = EvidenceReport.from_dict({"title": "T", "skipped": None})
    assert restored.skipped == []
