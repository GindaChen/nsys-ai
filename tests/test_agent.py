"""Tests for the agent persona and loop."""
import sqlite3


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


def test_agent_skill_selection():
    """Agent should select relevant skills for a question."""
    from nsys_ai.agent.loop import Agent
    # Use in-memory DB (won't run skills successfully but tests selection)
    agent = Agent(":memory:")
    try:
        selected = agent._select_skills("why are there bubbles in the GPU pipeline?")
        assert "gpu_idle_gaps" in selected

        selected = agent._select_skills("is NCCL overlapping with compute?")
        assert "nccl_breakdown" in selected

        selected = agent._select_skills("what is the top kernel?")
        assert "top_kernels" in selected

        selected = agent._select_skills("how is memory being used?")
        assert "memory_transfers" in selected
    finally:
        agent.close()


def test_agent_run_skill():
    """Agent should be able to run schema_inspect on a real db."""
    from nsys_ai.agent.loop import Agent
    # Create a minimal in-memory DB
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE test (id INTEGER)")
    conn.close()

    agent = Agent(":memory:")
    try:
        # schema_inspect should work on any SQLite db
        result = agent.run_skill("schema_inspect")
        # In-memory DB has no tables by default, but shouldn't error
        assert isinstance(result, str)
    finally:
        agent.close()
