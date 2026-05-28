"""Tests for ``nsys-ai analyze --format json`` and the corresponding
deprecation behavior on ``nsys-ai evidence build``.
"""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace


def _capture(handler, args, profile_module):
    """Invoke a CLI handler, capturing stdout + stderr.

    Returns (stdout_str, stderr_str).
    """
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        handler(args, profile_module)
    return out.getvalue(), err.getvalue()


def _make_args(profile_path: str, **overrides) -> SimpleNamespace:
    """Build a minimal argparse-like namespace for the analyze / evidence handlers."""
    base = dict(
        profile=str(profile_path),
        gpu=0,
        trim=None,
        output=None,
        format="text",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ──────────────────────────────────────────────────────────────────────
# analyze --format json
# ──────────────────────────────────────────────────────────────────────


class TestAnalyzeFormatJson:
    def test_emits_v01_envelope(self, minimal_nsys_db_path):
        from nsys_ai import profile as profile_module
        from nsys_ai.cli.handlers import _cmd_analyze

        args = _make_args(minimal_nsys_db_path, format="json")
        stdout, _ = _capture(_cmd_analyze, args, profile_module)

        payload = json.loads(stdout)
        assert payload["schema_version"] == "0.1"
        assert payload["producer"] == "nsys-ai"
        assert isinstance(payload["producer_version"], str) and payload["producer_version"]
        assert "findings" in payload
        assert isinstance(payload["findings"], list)
        # v0.1 content-derived profile id
        assert isinstance(payload["profile_id"], str)
        assert payload["profile_id"].startswith("nsys1:"), payload["profile_id"]

    def test_profile_path_present(self, minimal_nsys_db_path):
        """Envelope contains a profile_path string sourced from the opened Profile."""
        from nsys_ai import profile as profile_module
        from nsys_ai.cli.handlers import _cmd_analyze

        args = _make_args(minimal_nsys_db_path, format="json")
        stdout, _ = _capture(_cmd_analyze, args, profile_module)

        payload = json.loads(stdout)
        assert isinstance(payload["profile_path"], str)
        assert payload["profile_path"]  # non-empty
        # For a .sqlite input the resolved Profile.path equals the CLI arg.
        assert payload["profile_path"] == str(minimal_nsys_db_path)

    def test_envelope_and_selection_profile_id_agree(self, minimal_nsys_db_path):
        """Single source of truth: envelope profile_id matches every
        Finding.selection.profile_id inside the same payload.

        ``profile_id`` is the content-derived hash (``nsys1:sha256:...``)
        from ``fingerprint.get_profile_id`` — both the envelope and every
        ``TraceSelection`` produced during the same build must carry it.
        """
        from nsys_ai import profile as profile_module
        from nsys_ai.cli.handlers import _cmd_analyze

        args = _make_args(minimal_nsys_db_path, format="json")
        stdout, _ = _capture(_cmd_analyze, args, profile_module)
        payload = json.loads(stdout)

        envelope_id = payload["profile_id"]
        assert envelope_id.startswith("nsys1:"), envelope_id
        for f in payload["findings"]:
            sel = f.get("selection")
            if sel is None:
                continue
            assert sel["profile_id"] == envelope_id, (
                f"Finding {f.get('id')} selection.profile_id={sel['profile_id']!r} "
                f"disagrees with envelope profile_id={envelope_id!r}"
            )

    def test_no_deprecation_warning_for_analyze(self, minimal_nsys_db_path):
        """analyze --format json is the canonical path; no deprecation noise."""
        from nsys_ai import profile as profile_module
        from nsys_ai.cli.handlers import _cmd_analyze

        args = _make_args(minimal_nsys_db_path, format="json")
        _, stderr = _capture(_cmd_analyze, args, profile_module)
        assert "deprecated" not in stderr.lower()

    def test_findings_have_v01_shape_when_idle_gaps_present(self, minimal_nsys_db_path):
        """If the fixture produces idle-gap findings, they carry v0.1 fields.

        Defensive: the minimal fixture may not always trigger idle gaps,
        so we only inspect findings if at least one is produced.
        """
        from nsys_ai import profile as profile_module
        from nsys_ai.cli.handlers import _cmd_analyze

        args = _make_args(minimal_nsys_db_path, format="json")
        stdout, _ = _capture(_cmd_analyze, args, profile_module)
        payload = json.loads(stdout)
        # Filter by source rather than category: `idle` category is shared
        # across skills (gpu_idle_gaps and profile_health_manifest's
        # profile_idle_dominant both legitimately emit it), so category is
        # not a uniqueness key for "idle-gap findings" specifically.
        idle = [
            f for f in payload["findings"]
            if f.get("selection", {}).get("source") == "skill:gpu_idle_gaps"
        ]
        envelope_id = payload["profile_id"]
        for f in idle:
            assert f.get("id"), "Finding must have an id when v0.1-aware"
            assert "selection" in f, "Idle findings should have a selection"
            # Selection.profile_id is the same content-derived hash carried
            # by the envelope — both sourced from EvidenceBuilder context.
            assert f["selection"]["profile_id"] == envelope_id
            assert f["selection"]["source"] == "skill:gpu_idle_gaps"

    def test_writes_to_output_file_when_provided(self, minimal_nsys_db_path, tmp_path):
        from nsys_ai import profile as profile_module
        from nsys_ai.annotation import load_findings
        from nsys_ai.cli.handlers import _cmd_analyze

        out_path = tmp_path / "findings.json"
        args = _make_args(minimal_nsys_db_path, format="json", output=str(out_path))
        stdout, stderr = _capture(_cmd_analyze, args, profile_module)

        # stdout still has the JSON dump
        json.loads(stdout)
        # file was written via save_findings → load_findings round-trips
        loaded = load_findings(str(out_path))
        assert loaded.title  # legacy field still present
        # save count message goes to stderr
        assert "Saved" in stderr and "→" in stderr

    def test_stdout_and_file_profile_path_agree(self, minimal_nsys_db_path, tmp_path):
        """stdout payload and the persisted file must report the same profile_path."""
        from nsys_ai import profile as profile_module
        from nsys_ai.cli.handlers import _cmd_analyze

        out_path = tmp_path / "findings.json"
        args = _make_args(minimal_nsys_db_path, format="json", output=str(out_path))
        stdout, _ = _capture(_cmd_analyze, args, profile_module)

        stdout_payload = json.loads(stdout)
        with open(out_path) as f:
            file_payload = json.load(f)

        assert stdout_payload["profile_path"] == file_payload["profile_path"]
        assert stdout_payload["profile_path"] == str(minimal_nsys_db_path)


class TestAnalyzeFormatTextWithoutTrim:
    def test_text_mode_without_trim_fails_with_clear_error(self, minimal_nsys_db_path):
        """Without --trim, text mode prints a guidance error rather than crashing on trim[0]."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import pytest

        from nsys_ai import profile as profile_module
        from nsys_ai.cli.handlers import _cmd_analyze

        args = _make_args(minimal_nsys_db_path, format="text", trim=None)
        out = io.StringIO()
        err = io.StringIO()
        with pytest.raises(SystemExit) as excinfo:
            with redirect_stdout(out), redirect_stderr(err):
                _cmd_analyze(args, profile_module)

        assert excinfo.value.code == 1
        stderr_text = err.getvalue()
        # The error message must guide the user (don't just exit silently),
        # and must mention both the missing flag and the JSON workaround.
        assert "--trim" in stderr_text
        assert "--format json" in stderr_text
        # No tracebacks leak to stderr from this guarded path.
        assert "Traceback" not in stderr_text

    def test_json_mode_bypasses_text_pipeline(self, minimal_nsys_db_path):
        """``--format json`` must take the evidence path, not ``report.run_analyze``.

        ``run_analyze`` requires a populated ``--trim`` window; the JSON
        path goes through ``EvidenceBuilder``, which works without trim.
        Passing ``trim=None`` and observing a successful JSON output is
        a behavioral guarantee that the format switch took the early
        return before hitting the text pipeline.
        """
        from nsys_ai import profile as profile_module
        from nsys_ai.cli.handlers import _cmd_analyze

        args = _make_args(minimal_nsys_db_path, format="json", trim=None)
        stdout, _ = _capture(_cmd_analyze, args, profile_module)
        # If the early return failed and we fell into run_analyze, this
        # would have raised TypeError on ``trim[0]``.
        payload = json.loads(stdout)
        assert "findings" in payload


# ──────────────────────────────────────────────────────────────────────
# evidence build (deprecated alias)
# ──────────────────────────────────────────────────────────────────────


class TestEvidenceBuildDeprecation:
    def test_evidence_build_still_works(self, minimal_nsys_db_path):
        """The legacy command still produces valid JSON output."""
        from nsys_ai import profile as profile_module
        from nsys_ai.cli.handlers import _cmd_evidence

        args = SimpleNamespace(
            profile=str(minimal_nsys_db_path),
            evidence_action="build",
            format="json",
            analyzers=None,
            trim=None,
            gpu=0,
            output=None,
        )
        stdout, stderr = _capture(_cmd_evidence, args, profile_module)
        payload = json.loads(stdout)
        assert "findings" in payload

    def test_evidence_build_prints_deprecation_warning(self, minimal_nsys_db_path):
        """A deprecation notice is emitted on stderr to help users migrate."""
        from nsys_ai import profile as profile_module
        from nsys_ai.cli.handlers import _cmd_evidence

        args = SimpleNamespace(
            profile=str(minimal_nsys_db_path),
            evidence_action="build",
            format="json",
            analyzers=None,
            trim=None,
            gpu=0,
            output=None,
        )
        _, stderr = _capture(_cmd_evidence, args, profile_module)
        assert "deprecated" in stderr.lower()
        # Mentions the canonical replacement so users know where to go.
        assert "analyze --format json" in stderr

    def test_evidence_build_json_now_carries_envelope(self, minimal_nsys_db_path):
        """Previously the stdout JSON skipped the envelope; verify it now matches analyze."""
        from nsys_ai import profile as profile_module
        from nsys_ai.cli.handlers import _cmd_evidence

        args = SimpleNamespace(
            profile=str(minimal_nsys_db_path),
            evidence_action="build",
            format="json",
            analyzers=None,
            trim=None,
            gpu=0,
            output=None,
        )
        stdout, _ = _capture(_cmd_evidence, args, profile_module)
        payload = json.loads(stdout)
        assert payload["schema_version"] == "0.1"
        assert payload["producer"] == "nsys-ai"
        assert isinstance(payload["producer_version"], str)


# ──────────────────────────────────────────────────────────────────────
# Cross-command equivalence
# ──────────────────────────────────────────────────────────────────────


class TestCrossCommandEquivalence:
    def test_analyze_json_matches_evidence_build(self, minimal_nsys_db_path):
        """analyze --format json and evidence build emit equivalent JSON shape."""
        from nsys_ai import profile as profile_module
        from nsys_ai.cli.handlers import _cmd_analyze, _cmd_evidence

        analyze_args = _make_args(minimal_nsys_db_path, format="json")
        analyze_out, _ = _capture(_cmd_analyze, analyze_args, profile_module)

        evidence_args = SimpleNamespace(
            profile=str(minimal_nsys_db_path),
            evidence_action="build",
            format="json",
            analyzers=None,
            trim=None,
            gpu=0,
            output=None,
        )
        evidence_out, _ = _capture(_cmd_evidence, evidence_args, profile_module)

        a = json.loads(analyze_out)
        e = json.loads(evidence_out)

        # Envelope must agree.
        for key in ("schema_version", "producer", "producer_version"):
            assert a[key] == e[key]

        # Finding count and key structure should match — both go through
        # the same EvidenceBuilder pipeline.
        assert len(a["findings"]) == len(e["findings"])
        for af, ef in zip(a["findings"], e["findings"]):
            assert af.get("type") == ef.get("type")
            assert af.get("label") == ef.get("label")
            assert af.get("start_ns") == ef.get("start_ns")
            assert af.get("id") == ef.get("id")
            assert af.get("category") == ef.get("category")
