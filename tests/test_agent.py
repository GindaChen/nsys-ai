"""Tests for the agent persona and loop."""


def test_agent_identity():
    """Agent identity should have expected fields."""
    from nsys_ai.agent.persona import AGENT_IDENTITY

    assert AGENT_IDENTITY["name"] == "nsys-ai"
    assert "CUDA" in AGENT_IDENTITY["role"]
    assert len(AGENT_IDENTITY["expertise"]) >= 5
    assert len(AGENT_IDENTITY["principles"]) >= 5


def test_system_prompt():
    """System prompt should contain key sections."""
    from nsys_ai.agent.persona import SYSTEM_PROMPT

    assert "Identity" in SYSTEM_PROMPT
    assert "Core Principles" in SYSTEM_PROMPT
    assert "Evidence over intuition" in SYSTEM_PROMPT
    assert "Analysis Workflow" in SYSTEM_PROMPT
    assert "Book of Root Causes" in SYSTEM_PROMPT


def test_build_system_prompt():
    """Built prompt should inject skill catalog."""
    from nsys_ai.agent.persona import build_system_prompt

    prompt = build_system_prompt()
    assert "top_kernels" in prompt
    assert "gpu_idle_gaps" in prompt
    assert "{skill_catalog}" not in prompt  # should be substituted


def test_agent_skill_selection(minimal_nsys_db_path):
    """Agent should select relevant skills for a question."""
    from nsys_ai.agent.loop import Agent

    # Use in-memory DB (won't run skills successfully but tests selection)
    agent = Agent(minimal_nsys_db_path)
    try:
        selected = agent._select_skills("why are there bubbles in the GPU pipeline?")
        assert "gpu_idle_gaps" in selected

        selected = agent._select_skills("is NCCL overlapping with compute?")
        assert "nccl_breakdown" in selected
        assert "nccl_communicator_analysis" in selected

        selected = agent._select_skills("which communicator is slow in tensor parallel allreduce?")
        assert "nccl_communicator_analysis" in selected

        selected = agent._select_skills("what is the top kernel?")
        assert "top_kernels" in selected

        selected = agent._select_skills("how is memory being used?")
        assert "memory_transfers" in selected
    finally:
        agent.close()


def test_agent_run_skill(minimal_nsys_db_path):
    """Agent should be able to run schema_inspect on a real db."""
    from nsys_ai.agent.loop import Agent

    agent = Agent(minimal_nsys_db_path)
    try:
        # schema_inspect should work on any SQLite db
        result = agent.run_skill("schema_inspect")
        # In-memory DB has no tables by default, but shouldn't error
        assert isinstance(result, str)
    finally:
        agent.close()


def test_agent_ask_uses_evidence_first_template(minimal_nsys_db_path, monkeypatch):
    """Targeted answers should be grounded and end with a runnable verify command."""
    from nsys_ai.agent.loop import Agent

    monkeypatch.setattr("nsys_ai.chat_config._get_model_and_key", lambda: (None, None))

    agent = Agent(minimal_nsys_db_path)
    try:
        answer = agent.ask("why is this slow?")
    finally:
        agent.close()

    assert answer.startswith("## Summary")
    for heading in (
        "## Primary Diagnosis",
        "## Evidence",
        "## Confidence",
        "## Recommended Action",
        "## Verify",
    ):
        assert heading in answer
    assert "source_skill=" in answer
    assert "window=" in answer
    assert "`nsys-ai skill run" in answer
    assert answer.strip().splitlines()[-1].startswith("`nsys-ai skill run")


def test_agent_ask_includes_llm_synthesis(minimal_nsys_db_path, monkeypatch):
    """A paid synthesis result should be used as the answer Summary."""
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
    synthesis_call = {}

    def fake_synthesis(self, question, evidence, *, summary_only=False):
        synthesis_call["question"] = question
        synthesis_call["evidence"] = evidence
        synthesis_call["summary_only"] = summary_only
        return "The model synthesized this grounded performance summary."

    monkeypatch.setattr(Agent, "_try_llm_synthesis", fake_synthesis)

    agent = Agent(minimal_nsys_db_path)
    try:
        answer = agent.ask("why is this slow?")
    finally:
        agent.close()

    assert answer.startswith(
        "## Summary\nThe model synthesized this grounded performance summary."
    )
    assert synthesis_call["question"] == "why is this slow?"
    assert synthesis_call["evidence"]
    assert synthesis_call["summary_only"] is True
    assert answer.strip().splitlines()[-1].startswith("`nsys-ai skill run")


def test_agent_confidence_reflects_root_cause_severity(minimal_nsys_db_path):
    """Critical, warning, and info findings should not report the same confidence."""
    from nsys_ai.agent.loop import Agent

    agent = Agent(minimal_nsys_db_path)
    try:
        confidence = {}
        for severity in ("critical", "warning", "info"):
            row = {"pattern": "Test finding", "severity": severity}
            confidence[severity] = agent._confidence_label(
                {"root_cause_matcher": [row]},
                row,
            )
    finally:
        agent.close()

    assert confidence["critical"].startswith("0.90 (high)")
    assert confidence["warning"].startswith("0.75 (medium-high)")
    assert confidence["info"].startswith("0.55 (medium)")
    assert len(set(confidence.values())) == 3


def test_agent_verify_fallback_when_no_skill_evidence(minimal_nsys_db_path):
    from nsys_ai.agent.loop import Agent

    agent = Agent(minimal_nsys_db_path)
    try:
        answer = agent._format_evidence_first_answer("what happened?", {}, [])
    finally:
        agent.close()

    assert "Could not build a runnable verification command" in answer
    assert "`nsys-ai skill list`" in answer
    assert answer.strip().splitlines()[-1] == "`nsys-ai skill list`"
