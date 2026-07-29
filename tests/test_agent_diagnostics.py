"""Tests for agent diagnostics.json output (issue #207).

Covers ``AgentResult`` / ``ask_result`` / ``analyze_result``: the structured
Diagnostic is built from the same skill evidence the run already collected
(no EvidenceBuilder re-run), round-trips through JSON, and stays in sync
with the human-readable sections.
"""

import pytest


@pytest.fixture
def no_llm(monkeypatch):
    """Force the deterministic no-LLM path."""
    monkeypatch.setattr("nsys_ai.chat_config._get_model_and_key", lambda: (None, None))


def _assert_sections_match_text(diagnostic, text):
    """Every human section is rendered from the corresponding Diagnostic field."""
    assert f"## Summary\n{diagnostic.summary}" in text
    assert diagnostic.root_cause_hypotheses, "diagnostic must carry a primary diagnosis"
    assert f"## Primary Diagnosis\n{diagnostic.root_cause_hypotheses[0]}" in text
    assert f"## Recommended Action\n{diagnostic.recommendation}" in text
    assert f"`{diagnostic.verification_command}`" in text
    # The Confidence label opens with the same numeric value the JSON carries.
    assert f"## Confidence\n{diagnostic.confidence:.2f}" in text
    evidence_rows = [
        row
        for finding in diagnostic.primary_findings
        for row in (finding.evidence or [])
    ]
    assert text.count("- evidence_id=") == len(evidence_rows)
    for row in evidence_rows:
        assert f"- evidence_id={row.id}; source_skill={row.source_skill};" in text


class TestAskResult:
    def test_returns_agent_result_with_diagnostic(self, minimal_nsys_db_path, no_llm):
        from nsys_ai.agent.loop import Agent, AgentResult
        from nsys_ai.annotation import Diagnostic

        agent = Agent(minimal_nsys_db_path)
        try:
            result = agent.ask_result("why is this slow?")
        finally:
            agent.close()

        assert isinstance(result, AgentResult)
        assert isinstance(result.diagnostic, Diagnostic)
        assert result.text.startswith("## Summary")
        assert result.diagnostic.id.startswith("diag_")
        assert 0.0 < result.diagnostic.confidence <= 1.0
        assert result.diagnostic.verification_command.startswith("nsys-ai skill run")

    def test_ask_text_matches_ask_result_text(self, minimal_nsys_db_path, no_llm, monkeypatch):
        """Backwards compatibility: ask() returns exactly ask_result().text."""
        from nsys_ai.agent.loop import Agent, AgentResult

        agent = Agent(minimal_nsys_db_path)
        try:
            real = agent.ask_result("why is this slow?")
            sentinel = AgentResult(text="sentinel-text", diagnostic=real.diagnostic)
            monkeypatch.setattr(Agent, "ask_result", lambda self, question: sentinel)
            assert agent.ask("why is this slow?") == "sentinel-text"
        finally:
            agent.close()

    def test_diagnostic_fields_match_human_sections(self, minimal_nsys_db_path, no_llm):
        from nsys_ai.agent.loop import Agent

        agent = Agent(minimal_nsys_db_path)
        try:
            result = agent.ask_result("why is this slow?")
        finally:
            agent.close()

        _assert_sections_match_text(result.diagnostic, result.text)

    def test_json_round_trip_rerenders_same_sections(
        self, minimal_nsys_db_path, no_llm, tmp_path
    ):
        """A loaded diagnostic re-renders the same human sections as the original."""
        from nsys_ai.agent.loop import Agent
        from nsys_ai.annotation import load_diagnostic, save_diagnostic

        agent = Agent(minimal_nsys_db_path)
        try:
            result = agent.ask_result("why is this slow?")
        finally:
            agent.close()

        out = tmp_path / "diagnostics.json"
        save_diagnostic(result.diagnostic, str(out))
        loaded = load_diagnostic(str(out))

        assert loaded == result.diagnostic
        _assert_sections_match_text(loaded, result.text)
        assert agent._render_answer_text(loaded) == result.text

    def test_llm_summary_flows_into_diagnostic(self, minimal_nsys_db_path, monkeypatch):
        """With an LLM, the synthesized Summary lands in diagnostic.summary."""
        from nsys_ai.agent.loop import Agent

        monkeypatch.setattr(
            "nsys_ai.chat_config._get_model_and_key",
            lambda: ("test-model", "test-key"),
        )
        monkeypatch.setattr(
            Agent,
            "_try_llm_triage",
            lambda self, question, evidence: ["top_kernels"],
        )
        monkeypatch.setattr(
            Agent,
            "_try_llm_synthesis",
            lambda self, question, evidence, *, summary_only=False: (
                "The model synthesized this grounded performance summary."
            ),
        )

        agent = Agent(minimal_nsys_db_path)
        try:
            result = agent.ask_result("why is this slow?")
        finally:
            agent.close()

        assert result.diagnostic.summary == (
            "The model synthesized this grounded performance summary."
        )
        assert result.text.startswith(f"## Summary\n{result.diagnostic.summary}")

    def test_deterministic_diagnostic_id(self, minimal_nsys_db_path, no_llm):
        """Same profile + mode + question → same id; any change → new id."""
        from nsys_ai.agent.loop import Agent

        agent = Agent(minimal_nsys_db_path)
        try:
            first = agent.ask_result("why is this slow?")
            second = agent.ask_result("why is this slow?")
            other_question = agent.ask_result("is NCCL overlapping with compute?")
            analyze = agent.analyze_result()
        finally:
            agent.close()

        assert first.diagnostic.id == second.diagnostic.id
        assert first.diagnostic.id != other_question.diagnostic.id
        assert first.diagnostic.id != analyze.diagnostic.id

    def test_diagnostic_id_includes_trim_scope(self, minimal_nsys_db_path, no_llm):
        """Full-profile and trimmed runs must not collide."""
        from nsys_ai.agent.loop import Agent

        full_agent = Agent(minimal_nsys_db_path)
        trimmed_agent = Agent(minimal_nsys_db_path, trim_ns=(0, 1_000_000))
        other_trim_agent = Agent(minimal_nsys_db_path, trim_ns=(0, 2_000_000))
        try:
            full = full_agent.ask_result("why is this slow?")
            trimmed = trimmed_agent.ask_result("why is this slow?")
            same_trim = trimmed_agent.ask_result("why is this slow?")
            other_trim = other_trim_agent.ask_result("why is this slow?")
        finally:
            full_agent.close()
            trimmed_agent.close()
            other_trim_agent.close()

        assert full.diagnostic.id != trimmed.diagnostic.id
        assert trimmed.diagnostic.id == same_trim.diagnostic.id
        assert trimmed.diagnostic.id != other_trim.diagnostic.id

    def test_primary_findings_reuse_skill_converters(self, minimal_nsys_db_path, no_llm):
        """Findings come from to_findings_fn over the rows this run already collected."""
        from nsys_ai.agent.loop import Agent

        agent = Agent(minimal_nsys_db_path)
        try:
            result = agent.ask_result("why is this slow?")
        finally:
            agent.close()

        findings = result.diagnostic.primary_findings
        assert findings, "expected gpu_idle_gaps converter findings on the fixture"
        assert len(findings) <= Agent._MAX_PRIMARY_FINDINGS
        for finding in findings:
            assert finding.id
            assert finding.evidence, "finding must carry structured EvidenceRow objects"
            assert finding.selection is not None
            assert finding.selection.profile_id.startswith("nsys1:")


class TestAnalyzeResult:
    def test_llm_summary_is_shared_by_analyze_text_and_diagnostic(
        self, minimal_nsys_db_path, monkeypatch
    ):
        """Analyze requests one summary and renders it only through Diagnostic."""
        from nsys_ai.agent.loop import Agent

        call = {}

        def fake_synthesis(self, question, evidence, *, summary_only=False):
            call["summary_only"] = summary_only
            return (
                "## Analysis Overview\n"
                "GPU idle time is the strongest measured signal.\n\n"
                "## Extra Detail\n"
                "This section must not leak into the persisted summary."
            )

        monkeypatch.setattr(Agent, "_try_llm_synthesis", fake_synthesis)

        agent = Agent(minimal_nsys_db_path)
        try:
            result = agent.analyze_result()
        finally:
            agent.close()

        assert call["summary_only"] is True
        assert result.diagnostic.summary == (
            "GPU idle time is the strongest measured signal."
        )
        assert f"## Summary\n{result.diagnostic.summary}" in result.text
        assert "── AI Analysis ──" not in result.text
        assert "This section must not leak" not in result.text

    def test_analyze_result_builds_diagnostic_without_rerunning_skills(
        self, minimal_nsys_db_path, no_llm, monkeypatch
    ):
        """analyze_result derives the diagnostic from its own collected rows.

        EvidenceBuilder.build is rigged to raise: if the diagnostic path
        re-ran the evidence pipeline instead of reusing executed rows, this
        test would fail.
        """
        from nsys_ai.agent.loop import Agent
        from nsys_ai.evidence_builder import EvidenceBuilder

        def _forbidden_build(self, *args, **kwargs):
            raise AssertionError("EvidenceBuilder.build must not run inside analyze_result")

        monkeypatch.setattr(EvidenceBuilder, "build", _forbidden_build)

        agent = Agent(minimal_nsys_db_path)
        try:
            result = agent.analyze_result()
            # analyze() is a thin wrapper returning analyze_result().text.
            from nsys_ai.agent.loop import AgentResult

            sentinel = AgentResult(text="sentinel-analyze", diagnostic=result.diagnostic)
            monkeypatch.setattr(Agent, "analyze_result", lambda self: sentinel)
            assert agent.analyze() == "sentinel-analyze"
        finally:
            agent.close()

        assert result.text.startswith("═══ nsys-ai Auto-Analysis Report ═══")
        assert result.text.endswith("═══ End of Report ═══")

        diagnostic = result.diagnostic
        assert diagnostic.id.startswith("diag_")
        assert diagnostic.confidence == pytest.approx(0.60)
        assert diagnostic.verification_command.startswith("nsys-ai skill run")
        assert diagnostic.summary
        assert diagnostic.recommendation
        assert diagnostic.root_cause_hypotheses
        _assert_sections_match_text(diagnostic, result.text)

    def test_analyze_diagnostic_json_round_trip(self, minimal_nsys_db_path, no_llm, tmp_path):
        from nsys_ai.agent.loop import Agent
        from nsys_ai.annotation import load_diagnostic, save_diagnostic

        agent = Agent(minimal_nsys_db_path)
        try:
            result = agent.analyze_result()
        finally:
            agent.close()

        out = tmp_path / "diagnostics.json"
        save_diagnostic(result.diagnostic, str(out))
        assert load_diagnostic(str(out)) == result.diagnostic


class TestEmptyEvidenceFallback:
    def test_fallback_diagnostic_has_runnable_verify_command(self, minimal_nsys_db_path):
        from nsys_ai.agent.loop import Agent

        agent = Agent(minimal_nsys_db_path)
        try:
            diagnostic = agent._build_diagnostic(
                mode="ask",
                question="what happened?",
                evidence={},
                selected_skills=[],
                llm_summary=None,
            )
            text = agent._render_answer_text(diagnostic)
        finally:
            agent.close()

        assert diagnostic.confidence == pytest.approx(0.20)
        assert diagnostic.primary_findings == []
        # The fallback is still a runnable nsys-ai command, not prose.
        assert diagnostic.verification_command == "nsys-ai skill list"
        assert "Could not build a runnable verification command" in text
        assert text.strip().splitlines()[-1] == "`nsys-ai skill list`"

    def test_fallback_diagnostic_json_round_trip(self, minimal_nsys_db_path, tmp_path):
        from nsys_ai.agent.loop import Agent
        from nsys_ai.annotation import load_diagnostic, save_diagnostic

        agent = Agent(minimal_nsys_db_path)
        try:
            diagnostic = agent._build_diagnostic(
                mode="ask",
                question="what happened?",
                evidence={},
                selected_skills=[],
                llm_summary=None,
            )
        finally:
            agent.close()

        out = tmp_path / "diagnostics.json"
        save_diagnostic(diagnostic, str(out))
        assert load_diagnostic(str(out)) == diagnostic
