"""Regression tests for the shared CLI analysis contract."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace


def test_report_iteration_analysis_uses_registry(monkeypatch):
    from nsys_ai import report

    calls: list[tuple[str, dict]] = []
    expected = [{"iteration": 0, "duration_ms": 12.5, "gpu_start_ns": 10}]

    def fake_run_skill(name, conn, *, raw, **kwargs):
        calls.append((name, {"raw": raw, **kwargs, "conn": conn}))
        return expected

    monkeypatch.setattr("nsys_ai.skills.registry.run_skill", fake_run_skill)
    profile = SimpleNamespace(query_conn=lambda: "shared-connection")

    rows = report._run_iteration_skill(profile, 2, (100, 200))

    assert rows == expected
    assert calls == [
        (
            "iteration_timing",
            {
                "raw": True,
                "device": 2,
                "trim_start_ns": 100,
                "trim_end_ns": 200,
                "conn": "shared-connection",
            },
        )
    ]


def test_diagnose_json_matches_published_findings(tmp_path: Path):
    from nsys_ai.diagnose_command import run_diagnose

    fixture = Path(__file__).parent / "fixtures" / "h100_2gpu_1s.sqlite"
    stdout = io.StringIO()
    stderr = io.StringIO()
    output = tmp_path / "nested" / "findings.json"

    assert (
        run_diagnose(
            profile_path=fixture,
            session_id="json-contract",
            gpu=0,
            session_root=tmp_path / "sessions",
            format="json",
            output=output,
            stdout=stdout,
            stderr=stderr,
        )
        == 0
    )

    rendered = json.loads(stdout.getvalue())
    saved = json.loads(output.read_text(encoding="utf-8"))
    published = json.loads(
        (tmp_path / "sessions" / "json-contract" / "findings.json").read_text(
            encoding="utf-8"
        )
    )
    assert rendered == saved == published
    assert "Session:" not in stdout.getvalue()
    assert f"Saved findings: {output}" in stderr.getvalue()


def test_report_and_diagnose_expose_the_same_json_format():
    from nsys_ai.cli.parsers import _build_parser

    parser = _build_parser()
    diagnose = parser.parse_args(
        ["diagnose", "profile.sqlite", "--format", "json", "--output", "findings.json"]
    )
    report = parser.parse_args(
        ["report", "profile.sqlite", "--gpu", "0", "--trim", "1", "2", "--format", "json"]
    )

    assert diagnose.format == report.format == "json"
    assert diagnose.output == "findings.json"
