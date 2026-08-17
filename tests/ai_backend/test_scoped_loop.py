"""Scoped backend-loop tests for AI tool orchestration."""

import sys
from unittest.mock import MagicMock, patch

from nsys_ai import chat as chat_mod


def _make_tool_response(tool_name: str, arguments: str, tool_id: str = "c1"):
    fn = MagicMock()
    fn.name = tool_name
    fn.arguments = arguments

    tc = MagicMock()
    tc.id = tool_id
    tc.function = fn

    msg = MagicMock()
    msg.content = ""
    msg.tool_calls = [tc]

    choice = MagicMock()
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_text_response(text: str):
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = []

    choice = MagicMock()
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_run_agent_loop_db_query_then_answer():
    """Exploratory SQL alone must not become a grounded final answer."""
    mock_lt = MagicMock()
    mock_lt.completion.side_effect = [
        _make_tool_response("query_profile_db", '{"sql_query":"SELECT 1"}', "db1"),
        _make_text_response("Done."),
    ]

    queries = []

    def runner(sql: str) -> str:
        queries.append(sql)
        return '[{"x": 1}]'

    api_messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        content, actions = chat_mod.run_agent_loop(
            model="gpt-4o",
            api_messages=api_messages,
            query_runner=runner,
            max_turns=3,
        )

    assert "cannot answer this profile question" in content
    assert "Done." not in content
    assert actions == []
    assert queries == ["SELECT 1"]

    # Ensure tool result was fed back into the loop.
    tool_msgs = [m for m in api_messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "db1"


def test_run_agent_loop_max_turns_in_scoped_db_loop():
    """Loop should stop cleanly when max_turns is hit during repeated tool calls."""
    mock_lt = MagicMock()
    mock_lt.completion.side_effect = [
        _make_tool_response("query_profile_db", '{"sql_query":"SELECT bad"}', "db1"),
        _make_tool_response("query_profile_db", '{"sql_query":"SELECT bad"}', "db2"),
    ]

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        content, actions = chat_mod.run_agent_loop(
            model="gpt-4o",
            api_messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            query_runner=lambda _sql: "Error: bad query",
            max_turns=2,
        )

    assert "cannot answer this profile question" in content
    assert "bad query" in content
    assert actions == []
