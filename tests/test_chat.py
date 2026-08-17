"""Unit tests for nsys_ai.chat (AI Brain + Navigator)."""

import json
import sys
from unittest.mock import MagicMock, patch

from nsys_ai import chat as chat_mod


def test_get_model_and_key_none(monkeypatch):
    """With no API keys set, returns (None, None)."""
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    model, key = chat_mod._get_model_and_key()
    assert model is None
    assert key is None


def test_get_model_and_key_anthropic(monkeypatch):
    """ANTHROPIC_API_KEY is preferred."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    model, key = chat_mod._get_model_and_key()
    assert "anthropic" in model
    assert key == "sk-ant-x"


def test_get_model_and_key_openai(monkeypatch):
    """OPENAI_API_KEY used when Anthropic not set."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    model, key = chat_mod._get_model_and_key()
    assert "gpt" in model
    assert key == "sk-openai"


def test_get_model_and_key_gemini(monkeypatch):
    """GEMINI_API_KEY used when others not set."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    model, key = chat_mod._get_model_and_key()
    assert "gemini" in model
    assert key == "gemini-key"


def test_get_model_and_key_priority(monkeypatch):
    """Order: Anthropic > OpenAI > Gemini."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "b")
    monkeypatch.setenv("GEMINI_API_KEY", "c")
    model, _ = chat_mod._get_model_and_key()
    assert "anthropic" in model


def test_get_model_and_key_preferred(monkeypatch):
    """preferred_model (or NSYS_AI_MODEL) overrides default."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
    monkeypatch.setenv("GEMINI_API_KEY", "sk-g")
    model, key = chat_mod._get_model_and_key("gemini/gemini-1.5-flash")
    assert "gemini" in model
    assert key == "sk-g"
    model2, _ = chat_mod._get_model_and_key("gpt-4o-mini")
    assert "gpt" in model2
    assert model2 == "gpt-4o-mini"


def test_model_to_key():
    """_model_to_key maps model id to env var name."""
    assert chat_mod._model_to_key("anthropic/claude-3") == "ANTHROPIC_API_KEY"
    assert chat_mod._model_to_key("gpt-4o") == "OPENAI_API_KEY"
    assert chat_mod._model_to_key("gemini/gemini-1.5-pro") == "GEMINI_API_KEY"
    assert chat_mod._model_to_key("") is None


def test_get_available_models(monkeypatch):
    """get_available_models returns only models with API key set."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert chat_mod.get_available_models() == []
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    opts = chat_mod.get_available_models()
    assert any(o["id"] == "gpt-4o" for o in opts)


def test_build_system_prompt():
    """System prompt contains ui_context as JSON code block and MFU reference formulas."""
    ctx = {"view_state": {"scope": "all"}, "global_top_kernels": []}
    out = chat_mod._build_system_prompt(ctx)
    assert "```json" in out
    assert "view_state" in out
    assert "global_top_kernels" in out
    assert "CURRENT UI CONTEXT" in out
    # MFU reference formulas
    assert "MFU REFERENCE" in out
    assert "flops_per_layer" in out
    assert "SANITY CHECK" in out
    assert "num_gpus" in out


def test_tools_openai():
    """Tools include navigate, zoom, NVTX fit, query_profile_db, get_gpu_peak_tflops, compute_mfu, compute_region_mfu, submit_finding, get_gpu_overlap_stats, get_nccl_breakdown."""
    tools = chat_mod._tools_openai()
    assert len(tools) == 13
    names = {t["function"]["name"] for t in tools}
    assert names == {
        "navigate_to_kernel",
        "zoom_to_time_range",
        "fit_nvtx_range",
        "request_clarification",
        "answer_from_ui_context",
        "query_profile_db",
        "get_gpu_peak_tflops",
        "compute_mfu",
        "compute_region_mfu",
        "compute_theoretical_flops",
        "submit_finding",
        "get_gpu_overlap_stats",
        "get_nccl_breakdown",
    }
    clarification = next(
        t for t in tools if t["function"]["name"] == "request_clarification"
    )
    assert set(clarification["function"]["parameters"]["properties"]) == {
        "missing_information"
    }
    ui_response = next(
        t for t in tools if t["function"]["name"] == "answer_from_ui_context"
    )
    assert set(ui_response["function"]["parameters"]["properties"]) == {
        "context_paths"
    }
    nav = next(t for t in tools if t["function"]["name"] == "navigate_to_kernel")
    assert "target_name" in nav["function"]["parameters"]["properties"]
    region = next(t for t in tools if t["function"]["name"] == "compute_region_mfu")
    props = region["function"]["parameters"]["properties"]
    assert "name" in props
    assert "source" in props
    assert "num_gpus" in props
    zoom = next(t for t in tools if t["function"]["name"] == "zoom_to_time_range")
    assert "start_s" in zoom["function"]["parameters"]["properties"]
    assert "end_s" in zoom["function"]["parameters"]["properties"]
    fit = next(t for t in tools if t["function"]["name"] == "fit_nvtx_range")
    assert "nvtx_name" in fit["function"]["parameters"]["properties"]
    assert "start_s" in fit["function"]["parameters"]["properties"]
    assert "end_s" in fit["function"]["parameters"]["properties"]


def test_parse_tool_call_navigate():
    """navigate_to_kernel with required and optional args."""
    action = chat_mod._parse_tool_call(
        "navigate_to_kernel",
        '{"target_name": "my_kernel", "occurrence_index": 2, "reason": "bottleneck"}',
    )
    assert action == {
        "type": "navigate_to_kernel",
        "target_name": "my_kernel",
        "occurrence_index": 2,
        "reason": "bottleneck",
    }


def test_parse_tool_call_navigate_minimal():
    """navigate_to_kernel defaults occurrence_index to 1."""
    action = chat_mod._parse_tool_call("navigate_to_kernel", '{"target_name": "k"}')
    assert action["target_name"] == "k"
    assert action["occurrence_index"] == 1


def test_parse_tool_call_navigate_missing_target():
    """navigate_to_kernel without target_name returns None."""
    assert chat_mod._parse_tool_call("navigate_to_kernel", "{}") is None


def test_parse_tool_call_zoom():
    """zoom_to_time_range parses start_s and end_s."""
    action = chat_mod._parse_tool_call("zoom_to_time_range", '{"start_s": 1.5, "end_s": 2.5}')
    assert action == {
        "type": "zoom_to_time_range",
        "start_s": 1.5,
        "end_s": 2.5,
    }


def test_parse_tool_call_zoom_missing():
    """zoom_to_time_range missing start_s or end_s returns None."""
    assert chat_mod._parse_tool_call("zoom_to_time_range", '{"start_s": 1}') is None
    assert chat_mod._parse_tool_call("zoom_to_time_range", '{"end_s": 1}') is None


def test_parse_tool_call_fit_nvtx_by_name():
    """fit_nvtx_range can target by NVTX name."""
    action = chat_mod._parse_tool_call(
        "fit_nvtx_range", '{"nvtx_name": "flash_fwd", "occurrence_index": 2}'
    )
    assert action == {
        "type": "fit_nvtx_range",
        "nvtx_name": "flash_fwd",
        "occurrence_index": 2,
    }


def test_parse_tool_call_fit_nvtx_by_time_range():
    """fit_nvtx_range can target by explicit start/end seconds."""
    action = chat_mod._parse_tool_call("fit_nvtx_range", '{"start_s": 35.0, "end_s": 35.4}')
    assert action == {
        "type": "fit_nvtx_range",
        "start_s": 35.0,
        "end_s": 35.4,
    }


def test_parse_tool_call_fit_nvtx_missing():
    """fit_nvtx_range without name or full time range returns None."""
    assert chat_mod._parse_tool_call("fit_nvtx_range", "{}") is None
    assert chat_mod._parse_tool_call("fit_nvtx_range", '{"start_s": 1}') is None


def test_parse_tool_call_invalid_json():
    """Invalid JSON arguments return None."""
    assert chat_mod._parse_tool_call("navigate_to_kernel", "not json") is None
    assert chat_mod._parse_tool_call("navigate_to_kernel", "") is None


def test_parse_tool_call_unknown():
    """Unknown tool name returns None."""
    assert chat_mod._parse_tool_call("other_tool", '{"x": 1}') is None


def test_chat_completion_invalid_body(monkeypatch):
    """Invalid JSON body returns error content and empty actions."""
    monkeypatch.setattr(chat_mod, "_get_model_and_key", lambda preferred=None: ("gpt-4o", "key"))
    out = chat_mod.chat_completion(b"not json")
    assert out is not None
    assert "content" in out
    assert "Invalid" in out["content"]
    assert out["actions"] == []


def test_chat_completion_no_model(monkeypatch):
    """When no LLM is configured, returns None (501)."""
    monkeypatch.setattr(chat_mod, "_get_model_and_key", lambda preferred=None: (None, None))
    out = chat_mod.chat_completion(b'{"messages": [{"role": "user", "content": "hi"}]}')
    assert out is None


def test_web_chat_import_failure_uses_not_configured_contract(monkeypatch):
    from nsys_ai import chat, web

    def unavailable(_body):
        raise ImportError("optional AI backend is unavailable")

    monkeypatch.setattr(chat, "chat_completion", unavailable)

    assert web._handle_chat_request(b"{}") is None


def test_chat_completion_success_mock(monkeypatch):
    """With mocked litellm, returns content and actions from completion."""
    fake_message = MagicMock(content="Hello.", tool_calls=[])
    fake_choice = MagicMock(message=fake_message)
    fake_response = MagicMock(choices=[fake_choice])

    mock_lt = MagicMock()
    mock_lt.completion.return_value = fake_response

    monkeypatch.setattr(chat_mod, "_get_model_and_key", lambda preferred=None: ("gpt-4o", "key"))
    with patch.dict(sys.modules, {"litellm": mock_lt}):
        # Clear cached litellm in chat module so next import uses our mock
        if "litellm" in chat_mod.__dict__:
            del chat_mod.__dict__["litellm"]
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode("utf-8")
        out = chat_mod.chat_completion(body)
    assert out is not None
    assert out["content"] == "Hello."
    assert out["actions"] == []


def test_chat_completion_tool_calls_mock(monkeypatch):
    """Mock response with tool_calls produces actions."""
    fn = MagicMock()
    fn.name = "navigate_to_kernel"
    fn.arguments = '{"target_name": "kernel_a", "reason": "test"}'
    tc = MagicMock(function=fn)
    fake_message = MagicMock(content="Going there.", tool_calls=[tc])
    fake_choice = MagicMock(message=fake_message)
    fake_response = MagicMock(choices=[fake_choice])

    mock_lt = MagicMock()
    mock_lt.completion.return_value = fake_response

    monkeypatch.setattr(chat_mod, "_get_model_and_key", lambda preferred=None: ("gpt-4o", "key"))
    with patch.dict(sys.modules, {"litellm": mock_lt}):
        if "litellm" in chat_mod.__dict__:
            del chat_mod.__dict__["litellm"]
        body = json.dumps({"messages": [{"role": "user", "content": "go to kernel_a"}]}).encode(
            "utf-8"
        )
        out = chat_mod.chat_completion(body)
    assert out is not None
    assert out["content"] == "Going there."
    assert len(out["actions"]) == 1
    assert out["actions"][0]["type"] == "navigate_to_kernel"
    assert out["actions"][0]["target_name"] == "kernel_a"


def test_chat_completion_profile_finding_event_is_returned(monkeypatch):
    """Non-streaming chat propagates finding overlays and preserves the index."""
    conn = MagicMock()
    monkeypatch.setattr(
        chat_mod,
        "_prepare_session",
        lambda *_args: (conn, "/profile.sqlite", "system", lambda _sql: "[]"),
    )
    monkeypatch.setattr(chat_mod, "_get_model_and_key", lambda preferred=None: ("gpt-4o", "key"))

    fn = MagicMock(name="function")
    fn.name = "submit_finding"
    fn.arguments = json.dumps({"label": "slow kernel", "severity": "warning"})
    tc = MagicMock(id="finding-call", function=fn)
    tool_response = MagicMock(
        choices=[MagicMock(message=MagicMock(content="", tool_calls=[tc]))]
    )
    final_response = MagicMock(
        choices=[MagicMock(message=MagicMock(content="done", tool_calls=[]))]
    )
    mock_lt = MagicMock()
    mock_lt.completion.side_effect = [tool_response, final_response]

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        if "litellm" in chat_mod.__dict__:
            del chat_mod.__dict__["litellm"]
        body = json.dumps(
            {
                "profile_path": "/profile.sqlite",
                "findings_count": 4,
                "messages": [{"role": "user", "content": "mark it"}],
            }
        ).encode("utf-8")
        out = chat_mod.chat_completion(body)

    assert out["findings"][0]["index"] == 5
    assert out["findings"][0]["label"] == "slow kernel"
    assert out["actions"] == []
    conn.close.assert_called_once()


def test_chat_completion_preserves_findings_when_followup_llm_fails(monkeypatch):
    """A later provider failure must not erase an already emitted finding."""
    conn = MagicMock()
    monkeypatch.setattr(
        chat_mod,
        "_prepare_session",
        lambda *_args: (conn, "/profile.sqlite", "system", lambda _sql: "[]"),
    )
    monkeypatch.setattr(chat_mod, "_get_model_and_key", lambda preferred=None: ("gpt-4o", "key"))

    fn = MagicMock(name="function")
    fn.name = "submit_finding"
    fn.arguments = json.dumps({"label": "slow kernel"})
    tc = MagicMock(id="finding-call", function=fn)
    tool_response = MagicMock(
        choices=[MagicMock(message=MagicMock(content="", tool_calls=[tc]))]
    )
    mock_lt = MagicMock()
    mock_lt.completion.side_effect = [tool_response, RuntimeError("provider unavailable")]

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        if "litellm" in chat_mod.__dict__:
            del chat_mod.__dict__["litellm"]
        body = json.dumps(
            {"profile_path": "/profile.sqlite", "messages": [{"role": "user", "content": "mark"}]}
        ).encode("utf-8")
        out = chat_mod.chat_completion(body)

    assert out["findings"][0]["label"] == "slow kernel"
    assert out["actions"] == []
    conn.close.assert_called_once()


def test_sse_event():
    """_sse_event produces valid SSE line format."""
    raw = chat_mod._sse_event("text", {"chunk": "hi"})
    assert raw.startswith(b"event: text\n")
    assert b"data: " in raw
    assert "hi" in raw.decode("utf-8")


# --- 11.8.4 Stage 1: stream_agent_loop headless integration tests ---


def test_stream_agent_loop_yields_text_and_done(monkeypatch):
    """stream_agent_loop with mocked stream yields at least one text event and a done event."""
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(content="Hi", tool_calls=[]))]
    chunk1.usage = None
    chunk2 = MagicMock()
    chunk2.choices = []
    chunk2.usage = MagicMock(prompt_tokens=5, completion_tokens=2)

    mock_lt = MagicMock()
    mock_lt.completion.return_value = iter([chunk1, chunk2])

    monkeypatch.setattr(chat_mod, "_get_model_and_key", lambda preferred=None: ("gpt-4o", "key"))
    with patch.dict(sys.modules, {"litellm": mock_lt}):
        if "litellm" in chat_mod.__dict__:
            del chat_mod.__dict__["litellm"]
        events = list(
            chat_mod.stream_agent_loop(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                ui_context={},
                profile_path=None,
                max_turns=2,
            )
        )
    types = [e.get("type") for e in events]
    assert "text" in types
    assert "done" in types
    text_events = [e for e in events if e.get("type") == "text"]
    assert any("Hi" in (e.get("content") or "") for e in text_events)
    done_events = [e for e in events if e.get("type") == "done"]
    assert len(done_events) >= 1


def test_stream_agent_loop_terminates_with_done(monkeypatch):
    """stream_agent_loop always ends with a done event (§11.8.4 Stage 1)."""
    mock_lt = MagicMock()
    mock_lt.completion.return_value = iter([])  # empty stream -> no tool_calls, exit with done

    monkeypatch.setattr(chat_mod, "_get_model_and_key", lambda preferred=None: ("gpt-4o", "key"))
    with patch.dict(sys.modules, {"litellm": mock_lt}):
        if "litellm" in chat_mod.__dict__:
            del chat_mod.__dict__["litellm"]
        events = list(
            chat_mod.stream_agent_loop(
                model="gpt-4o",
                messages=[{"role": "user", "content": "x"}],
                ui_context={},
                profile_path=None,
                max_turns=1,
            )
        )
    assert events
    assert events[-1].get("type") == "done"


# --- 11.8.4 Stage 2: tool error feedback in run_agent_loop (§11.7.1) ---


def test_run_agent_loop_query_error_rolls_back_failed_batch(monkeypatch):
    """A query error is reported without leaving an unmatched tool batch."""
    # Turn 1: model returns a query_profile_db tool call
    fn1 = MagicMock()
    fn1.name = "query_profile_db"
    fn1.arguments = '{"sql_query": "SELECT bad FROM t"}'
    tc1 = MagicMock()
    tc1.id = "call_1"
    tc1.function = fn1
    msg1 = MagicMock(content="", tool_calls=[tc1])
    resp1 = MagicMock(choices=[MagicMock(message=msg1)])

    # Turn 2: model returns plain text (no tool calls)
    msg2 = MagicMock(content="I see, that column does not exist.", tool_calls=[])
    resp2 = MagicMock(choices=[MagicMock(message=msg2)])

    mock_lt = MagicMock()
    mock_lt.completion.side_effect = [resp1, resp2]

    def query_runner(sql):
        return "Error: no such column: bad"

    api_messages = [
        {"role": "system", "content": "You are a test."},
        {"role": "user", "content": "What is in table t?"},
    ]
    with patch.dict(sys.modules, {"litellm": mock_lt}):
        if "litellm" in chat_mod.__dict__:
            del chat_mod.__dict__["litellm"]
        content, actions = chat_mod.run_agent_loop(
            model="gpt-4o",
            api_messages=api_messages,
            tools=chat_mod._tools_openai(),
            query_runner=query_runner,
            max_turns=5,
        )
    assert "cannot answer this profile question" in content
    assert "no such column" in content
    assert "I see" not in content
    assert mock_lt.completion.call_count == 1
    assert all(not message.get("tool_calls") for message in api_messages)
    assert all(message.get("role") != "tool" for message in api_messages)


def test_tool_result_failed_detects_nested_abstention():
    assert chat_mod._tool_result_failed(
        json.dumps(
            {
                "device_id": 0,
                "collectives": [{"_abstained": True, "reason": "no NCCL tables"}],
            }
        )
    )
    assert not chat_mod._tool_result_failed(json.dumps({"collectives": []}))


def test_run_agent_loop_sql_alone_cannot_ground_a_profile_answer():
    query_fn = MagicMock()
    query_fn.name = "query_profile_db"
    query_fn.arguments = '{"sql_query": "SELECT 1"}'
    query_call = MagicMock(id="db1", function=query_fn)
    query_response = MagicMock(
        choices=[MagicMock(message=MagicMock(content="", tool_calls=[query_call]))]
    )
    answer_response = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="SQL proves NCCL is the bottleneck.", tool_calls=[]
                )
            )
        ]
    )
    mock_lt = MagicMock()
    mock_lt.completion.side_effect = [query_response, answer_response]

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        content, _ = chat_mod.run_agent_loop(
            model="gpt-4o",
            api_messages=[{"role": "system", "content": "s"}],
            query_runner=lambda _sql: "[{\"value\": 1}]",
        )

    assert "cannot answer this profile question" in content
    assert "SQL proves" not in content
    assert mock_lt.completion.call_count == 2


def test_run_agent_loop_uses_registry_for_profile_tools(minimal_nsys_conn):
    from nsys_ai.tool_dispatch import ToolDispatcher

    def tool_call(call_id, name, arguments):
        fn = MagicMock()
        fn.name = name
        fn.arguments = json.dumps(arguments)
        return MagicMock(id=call_id, function=fn)

    # Deliberately put the pure calculation before the SQL call: grounding
    # must not depend on provider tool-call ordering.
    first = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="",
                    tool_calls=[
                        tool_call(
                            "mfu1",
                            "compute_mfu",
                            {
                                "step_time_s": 1.0,
                                "model_flops_per_step": 1e12,
                                "peak_tflops": 100.0,
                            },
                        ),
                        tool_call("db1", "query_profile_db", {"sql_query": "SELECT 1"}),
                    ],
                )
            )
        ]
    )
    final = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Grounded web result.", tool_calls=[]))]
    )
    mock_lt = MagicMock()
    mock_lt.completion.side_effect = [first, final]

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        content, _ = chat_mod.run_agent_loop(
            model="gpt-4o",
            api_messages=[{"role": "system", "content": "s"}],
            query_runner=lambda _sql: "[{\"value\": 1}]",
            dispatcher=ToolDispatcher(conn=minimal_nsys_conn),
        )

    assert content == "Grounded web result."
    assert mock_lt.completion.call_count == 2


def test_run_agent_loop_exits_after_navigate(monkeypatch):
    """run_agent_loop exits immediately after navigate_to_kernel (no extra LLM turn)."""
    fn1 = MagicMock()
    fn1.name = "navigate_to_kernel"
    fn1.arguments = '{"target_name": "fast_kernel", "reason": "bottleneck"}'
    tc1 = MagicMock()
    tc1.id = "call_nav"
    tc1.function = fn1
    msg1 = MagicMock(
        content="NCCL is definitely the bottleneck. Navigating.",
        tool_calls=[tc1],
    )
    resp1 = MagicMock(choices=[MagicMock(message=msg1)])

    mock_lt = MagicMock()
    mock_lt.completion.return_value = resp1

    api_messages = [
        {"role": "system", "content": "You are a test."},
        {"role": "user", "content": "Go to fast_kernel"},
    ]
    with patch.dict(sys.modules, {"litellm": mock_lt}):
        if "litellm" in chat_mod.__dict__:
            del chat_mod.__dict__["litellm"]
        content, actions = chat_mod.run_agent_loop(
            model="gpt-4o",
            api_messages=api_messages,
            tools=chat_mod._tools_openai(),
            query_runner=lambda _sql: "[]",
            max_turns=5,
        )
    # Must exit after 1 LLM call, not loop again
    assert mock_lt.completion.call_count == 1
    assert len(actions) == 1
    assert actions[0]["type"] == "navigate_to_kernel"
    assert actions[0]["target_name"] == "fast_kernel"
    assert content == ""
    # No orphaned tool messages for navigation tools
    tool_msgs = [m for m in api_messages if m.get("role") == "tool"]
    assert not tool_msgs


def test_stream_agent_loop_yields_action_and_done(monkeypatch):
    """stream_agent_loop with navigate_to_kernel yields action event then done (§11.8.4)."""
    # Chunk 1: text delta
    chunk_text = MagicMock()
    chunk_text.choices = [
        MagicMock(
            delta=MagicMock(
                content="Going there.",
                tool_calls=[],
            )
        )
    ]
    chunk_text.usage = None
    # Chunk 2: tool_call delta
    fn_delta = MagicMock()
    fn_delta.name = "navigate_to_kernel"
    fn_delta.arguments = '{"target_name": "k1"}'
    tc_delta = MagicMock()
    tc_delta.index = 0
    tc_delta.id = "call_1"
    tc_delta.function = fn_delta
    chunk_tc = MagicMock()
    chunk_tc.choices = [MagicMock(delta=MagicMock(content=None, tool_calls=[tc_delta]))]
    chunk_tc.usage = None

    mock_lt = MagicMock()
    mock_lt.completion.return_value = iter([chunk_text, chunk_tc])

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        if "litellm" in chat_mod.__dict__:
            del chat_mod.__dict__["litellm"]
        events = list(
            chat_mod.stream_agent_loop(
                model="gpt-4o",
                messages=[{"role": "user", "content": "go"}],
                ui_context={},
                profile_path=None,
                max_turns=3,
            )
        )

    types = [e.get("type") for e in events]
    assert "text" not in types
    assert "action" in types
    assert types[-1] == "done"
    # Only one LLM call (exits after external tool)
    assert mock_lt.completion.call_count == 1
    action_ev = next(e for e in events if e.get("type") == "action")
    assert action_ev["action"]["type"] == "navigate_to_kernel"
    assert action_ev["action"]["target_name"] == "k1"


def test_compact_old_tool_results_compacts_previous_turns():
    """_compact_old_tool_results replaces large old tool content (§11.9 Phase 2.2)."""
    api_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "query_profile_db", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "query_profile_db", "content": "x" * 300},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "query_profile_db", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "name": "query_profile_db", "content": "y" * 300},
    ]
    chat_mod._compact_old_tool_results(api_messages)
    # Tool message from turn 1 (before last assistant) should be compacted.
    assert api_messages[3]["content"] == "[Summary: DB query returned results.]"
    # Tool message from turn 2 (most recent) should be unchanged.
    assert api_messages[5]["content"] == "y" * 300


def test_compact_old_tool_results_noop_first_turn():
    """_compact_old_tool_results is a no-op when only one tool turn exists."""
    api_messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "z" * 300},
    ]
    import copy

    original = copy.deepcopy(api_messages)
    chat_mod._compact_old_tool_results(api_messages)
    assert api_messages == original


def test_compact_old_tool_results_preserves_failure_state():
    api_messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": json.dumps({"error": "query failed", "detail": "x" * 250}),
        },
        {"role": "assistant", "tool_calls": [{"id": "c2"}]},
        {"role": "tool", "tool_call_id": "c2", "content": "[]"},
    ]

    chat_mod._compact_old_tool_results(api_messages)

    assert api_messages[2]["content"] == (
        "[Summary: Tool failed; no profile evidence was produced.]"
    )


def test_run_agent_loop_stops_after_first_query_error(monkeypatch):
    """A failed query never gives the provider a second chance to guess."""
    # Turn 1: query_profile_db → error
    fn1 = MagicMock(name_attr="query_profile_db", arguments='{"sql_query": "bad"}')
    fn1.name = "query_profile_db"
    fn1.arguments = '{"sql_query": "bad1"}'
    tc1 = MagicMock()
    tc1.id = "c1"
    tc1.function = fn1
    msg1 = MagicMock(content="", tool_calls=[tc1])
    resp1 = MagicMock(choices=[MagicMock(message=msg1)])

    # Turn 2: query_profile_db → error again (2nd consecutive)
    fn2 = MagicMock()
    fn2.name = "query_profile_db"
    fn2.arguments = '{"sql_query": "bad2"}'
    tc2 = MagicMock()
    tc2.id = "c2"
    tc2.function = fn2
    msg2 = MagicMock(content="", tool_calls=[tc2])
    resp2 = MagicMock(choices=[MagicMock(message=msg2)])

    # Turn 3: model gives up and answers in text
    msg3 = MagicMock(content="I cannot retrieve this data.", tool_calls=[])
    resp3 = MagicMock(choices=[MagicMock(message=msg3)])

    mock_lt = MagicMock()
    mock_lt.completion.side_effect = [resp1, resp2, resp3]

    api_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]
    with patch.dict(sys.modules, {"litellm": mock_lt}):
        if "litellm" in chat_mod.__dict__:
            del chat_mod.__dict__["litellm"]
        content, _ = chat_mod.run_agent_loop(
            model="gpt-4o",
            api_messages=api_messages,
            tools=chat_mod._tools_openai(),
            query_runner=lambda sql: "Error: no such table",
            max_turns=5,
        )
    assert "cannot answer this profile question" in content
    assert "no such table" in content
    assert mock_lt.completion.call_count == 1
    assert all(not message.get("tool_calls") for message in api_messages)
    assert all(message.get("role") != "tool" for message in api_messages)


def test_run_agent_loop_rejects_profile_answer_without_query(monkeypatch):
    """A loaded schema is context, not evidence for a profile diagnosis."""
    message = MagicMock(
        content="NCCL serialization is definitely the bottleneck.",
        tool_calls=[],
    )
    mock_lt = MagicMock()
    mock_lt.completion.return_value = MagicMock(choices=[MagicMock(message=message)])

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        content, _ = chat_mod.run_agent_loop(
            model="gpt-4o",
            api_messages=[{"role": "system", "content": "s"}],
            query_runner=lambda _sql: "[]",
        )

    assert "cannot answer this profile question" in content
    assert "did not query" in content
    assert "NCCL serialization" not in content


def test_run_agent_loop_later_query_failure_revokes_grounding():
    def tool_response(call_id, sql):
        fn = MagicMock(name="query_profile_db", arguments=json.dumps({"sql_query": sql}))
        fn.name = "query_profile_db"
        fn.arguments = json.dumps({"sql_query": sql})
        tc = MagicMock(id=call_id, function=fn)
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content="", tool_calls=[tc]))]
        )

    guess = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="The earlier result proves NCCL is the bottleneck.",
                    tool_calls=[],
                )
            )
        ]
    )
    mock_lt = MagicMock()
    mock_lt.completion.side_effect = [
        tool_response("db1", "SELECT 1"),
        tool_response("db2", "SELECT bad"),
        guess,
    ]

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        content, _ = chat_mod.run_agent_loop(
            model="gpt-4o",
            api_messages=[{"role": "system", "content": "s"}],
            query_runner=lambda sql: "Error: later query failed" if "bad" in sql else "[]",
        )

    assert "cannot answer this profile question" in content
    assert "later query failed" in content
    assert "earlier result proves" not in content


def test_run_agent_loop_any_invalid_or_unexecuted_tool_revokes_grounding():
    def tool_response(call_id, name, arguments):
        fn = MagicMock()
        fn.name = name
        fn.arguments = json.dumps(arguments)
        return MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="",
                        tool_calls=[MagicMock(id=call_id, function=fn)],
                    )
                )
            ]
        )

    cases = [
        ("compute_mfu", {"step_time_s": 1.0}),
        (
            "request_clarification",
            {
                "missing_information": "region_name",
                "question": "NCCL is the bottleneck, right?",
            },
        ),
    ]
    outputs = []
    for name, arguments in cases:
        mock_lt = MagicMock()
        mock_lt.completion.side_effect = [
            tool_response("db1", "query_profile_db", {"sql_query": "SELECT 1"}),
            tool_response("bad1", name, arguments),
            MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="The earlier query proves a regression.",
                            tool_calls=[],
                        )
                    )
                ]
            ),
        ]
        with patch.dict(sys.modules, {"litellm": mock_lt}):
            outputs.append(
                chat_mod.run_agent_loop(
                    model="gpt-4o",
                    api_messages=[{"role": "system", "content": "s"}],
                    query_runner=lambda _sql: "[]",
                )[0]
            )

    assert all("cannot answer this profile question" in output for output in outputs)
    assert all("earlier query proves" not in output for output in outputs)


def test_run_agent_loop_allows_structured_clarifications():
    def clarification(call_id, missing):
        fn = MagicMock()
        fn.name = "request_clarification"
        fn.arguments = json.dumps({"missing_information": missing})
        return MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="Unsupported diagnosis.",
                        tool_calls=[MagicMock(id=call_id, function=fn)],
                    )
                )
            ]
        )

    missing_inputs = ["model_flops_per_step", "region_name"]
    mock_lt = MagicMock()
    mock_lt.completion.side_effect = [
        clarification(f"clarify-{index}", missing)
        for index, missing in enumerate(missing_inputs)
    ]

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        outputs = [
            chat_mod.run_agent_loop(
                model="gpt-4o",
                api_messages=[{"role": "system", "content": "s"}],
                query_runner=lambda _sql: "[]",
            )[0]
            for _ in missing_inputs
        ]

    assert outputs == [
        "What is model_flops_per_step?",
        "Which NVTX region or CUDA kernel should I analyze?",
    ]
    assert all("Unsupported diagnosis" not in output for output in outputs)


def test_run_agent_loop_sibling_failure_suppresses_valid_clarification():
    clarification_fn = MagicMock()
    clarification_fn.name = "request_clarification"
    clarification_fn.arguments = json.dumps({"missing_information": "region_name"})
    malformed_fn = MagicMock()
    malformed_fn.name = "invented_tool"
    malformed_fn.arguments = "{}"
    valid_call = MagicMock(id="clarify1", function=clarification_fn)
    malformed_call = MagicMock(id=None, function=malformed_fn)

    outputs = []
    call_counts = []
    histories = []
    for tool_calls in ([valid_call, malformed_call], [malformed_call, valid_call]):
        response = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content="", tool_calls=tool_calls)
                )
            ]
        )
        mock_lt = MagicMock()
        mock_lt.completion.return_value = response
        api_messages = [{"role": "system", "content": "s"}]
        with patch.dict(sys.modules, {"litellm": mock_lt}):
            outputs.append(
                chat_mod.run_agent_loop(
                    model="gpt-4o",
                    api_messages=api_messages,
                    query_runner=lambda _sql: "[]",
                    max_turns=3,
                )[0]
            )
        call_counts.append(mock_lt.completion.call_count)
        histories.append(api_messages)

    assert all("cannot answer this profile question" in output for output in outputs)
    assert all("Which NVTX region" not in output for output in outputs)
    assert call_counts == [1, 1]
    assert all(
        all(not message.get("tool_calls") for message in history)
        and all(message.get("role") != "tool" for message in history)
        for history in histories
    )


def test_run_agent_loop_allows_answer_from_valid_ui_context():
    fn = MagicMock()
    fn.name = "answer_from_ui_context"
    fn.arguments = json.dumps(
        {
            "context_paths": [
                "selected_kernel.name",
                "selected_kernel.duration_ms",
            ],
        }
    )
    response = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="",
                    tool_calls=[MagicMock(id="ui1", function=fn)],
                )
            )
        ]
    )
    mock_lt = MagicMock()
    mock_lt.completion.return_value = response

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        content, _ = chat_mod.run_agent_loop(
            model="gpt-4o",
            api_messages=[{"role": "system", "content": "s"}],
            query_runner=lambda _sql: "[]",
            ui_context={
                "selected_kernel": {"name": "flash_fwd", "duration_ms": 1.25}
            },
        )

    assert content == (
        'UI context: selected_kernel.name="flash_fwd"; '
        "selected_kernel.duration_ms=1.25"
    )


def test_run_agent_loop_rejects_ui_answer_with_missing_context_path():
    fn = MagicMock()
    fn.name = "answer_from_ui_context"
    fn.arguments = json.dumps(
        {
            "context_paths": ["selected_kernel.nccl_pct"],
        }
    )
    tool_response = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="",
                    tool_calls=[MagicMock(id="ui1", function=fn)],
                )
            )
        ]
    )
    guess_response = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="NCCL is definitely the bottleneck.",
                    tool_calls=[],
                )
            )
        ]
    )
    mock_lt = MagicMock()
    mock_lt.completion.side_effect = [tool_response, guess_response]

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        content, _ = chat_mod.run_agent_loop(
            model="gpt-4o",
            api_messages=[{"role": "system", "content": "s"}],
            query_runner=lambda _sql: "[]",
            ui_context={"selected_kernel": {"name": "flash_fwd"}},
        )

    assert "cannot answer this profile question" in content
    assert "NCCL is definitely" not in content


def test_control_responses_reject_model_prose_and_sensitive_context():
    clarification, clarification_error = chat_mod._resolve_control_response(
        "request_clarification",
        json.dumps(
            {
                "missing_information": "region_name",
                "question": "NCCL is the bottleneck, which region?",
            }
        ),
        {},
    )
    ui_answer, ui_error = chat_mod._resolve_control_response(
        "answer_from_ui_context",
        json.dumps(
            {
                "context_paths": ["selected_kernel.name"],
                "answer": "NCCL is definitely the bottleneck.",
            }
        ),
        {"selected_kernel": {"name": "flash_fwd"}},
    )
    secret_answer, secret_error = chat_mod._resolve_control_response(
        "answer_from_ui_context",
        json.dumps({"context_paths": ["session.api_key"]}),
        {"session": {"api_key": "do-not-render"}},
    )

    assert clarification is None and "only missing_information" in clarification_error
    assert ui_answer is None and "only context_paths" in ui_error
    assert secret_answer is None and "sensitive" in secret_error
    assert "do-not-render" not in secret_error


def test_stream_agent_loop_suppresses_guess_after_query_failure(monkeypatch):
    fn = MagicMock(name="query_profile_db", arguments='{"sql_query":"SELECT bad"}')
    fn.name = "query_profile_db"
    fn.arguments = '{"sql_query":"SELECT bad"}'
    tc = MagicMock(index=0, id="db1", function=fn)
    tool_chunk = MagicMock(
        choices=[MagicMock(delta=MagicMock(content=None, tool_calls=[tc]))],
        usage=None,
    )
    guess_chunk = MagicMock(
        choices=[
            MagicMock(
                delta=MagicMock(
                    content="The profile proves an NCCL bottleneck.",
                    tool_calls=[],
                )
            )
        ],
        usage=None,
    )
    mock_lt = MagicMock()
    mock_lt.completion.side_effect = [iter([tool_chunk]), iter([guess_chunk])]
    conn = MagicMock()
    monkeypatch.setattr(
        chat_mod,
        "_prepare_session",
        lambda *args, **kwargs: (conn, "/tmp/profile.sqlite", "system", lambda _sql: "Error: bad query"),
    )

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        events = list(
            chat_mod.stream_agent_loop(
                model="gpt-4o",
                messages=[{"role": "user", "content": "why slow?"}],
                ui_context={},
                profile_path="/tmp/profile.sqlite",
                max_turns=3,
            )
        )

    text = "".join(event.get("content", "") for event in events if event["type"] == "text")
    assert "cannot answer this profile question" in text
    assert "bad query" in text
    assert "profile proves" not in text
    conn.close.assert_called_once()


def test_stream_agent_loop_buffers_text_before_failing_tool(monkeypatch):
    fn = MagicMock(name="query_profile_db", arguments='{"sql_query":"SELECT bad"}')
    fn.name = "query_profile_db"
    fn.arguments = '{"sql_query":"SELECT bad"}'
    tool_chunk = MagicMock(
        choices=[
            MagicMock(
                delta=MagicMock(
                    content="The profile definitely has an NCCL bottleneck.",
                    tool_calls=[MagicMock(index=0, id="db1", function=fn)],
                )
            )
        ],
        usage=None,
    )
    mock_lt = MagicMock()
    mock_lt.completion.return_value = iter([tool_chunk])
    conn = MagicMock()
    monkeypatch.setattr(
        chat_mod,
        "_prepare_session",
        lambda *args, **kwargs: (conn, "/tmp/profile.sqlite", "system", lambda _sql: "Error: bad query"),
    )

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        events = list(
            chat_mod.stream_agent_loop(
                model="gpt-4o",
                messages=[{"role": "user", "content": "why slow?"}],
                ui_context={},
                profile_path="/tmp/profile.sqlite",
                max_turns=1,
            )
        )

    text = "".join(event.get("content", "") for event in events if event["type"] == "text")
    assert "cannot answer this profile question" in text
    assert "definitely has" not in text


def test_stream_agent_loop_allows_structured_region_clarification(monkeypatch):
    fn = MagicMock()
    fn.name = "request_clarification"
    fn.arguments = json.dumps(
        {"missing_information": "region_name"}
    )
    chunk = MagicMock(
        choices=[
            MagicMock(
                delta=MagicMock(
                    content="Unsupported diagnosis.",
                    tool_calls=[MagicMock(index=0, id="clarify1", function=fn)],
                )
            )
        ],
        usage=None,
    )
    mock_lt = MagicMock()
    mock_lt.completion.return_value = iter([chunk])
    conn = MagicMock()
    monkeypatch.setattr(
        chat_mod,
        "_prepare_session",
        lambda *args, **kwargs: (
            conn,
            "/tmp/profile.sqlite",
            "system",
            lambda _sql: "[]",
        ),
    )

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        events = list(
            chat_mod.stream_agent_loop(
                model="gpt-4o",
                messages=[{"role": "user", "content": "analyze a region"}],
                ui_context={},
                profile_path="/tmp/profile.sqlite",
            )
        )

    text = "".join(event.get("content", "") for event in events if event["type"] == "text")
    assert text == "Which NVTX region or CUDA kernel should I analyze?"
    assert "Unsupported diagnosis" not in text


def test_stream_agent_loop_later_failure_skips_forced_summary(monkeypatch):
    def query_chunk(call_id, sql):
        fn = MagicMock(name="query_profile_db", arguments=json.dumps({"sql_query": sql}))
        fn.name = "query_profile_db"
        fn.arguments = json.dumps({"sql_query": sql})
        return MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(
                        content=None,
                        tool_calls=[MagicMock(index=0, id=call_id, function=fn)],
                    )
                )
            ],
            usage=None,
        )

    mock_lt = MagicMock()
    mock_lt.completion.side_effect = [
        iter([query_chunk("db1", "SELECT 1")]),
        iter([query_chunk("db2", "SELECT bad")]),
    ]
    conn = MagicMock()
    monkeypatch.setattr(
        chat_mod,
        "_prepare_session",
        lambda *args, **kwargs: (
            conn,
            "/tmp/profile.sqlite",
            "system",
            lambda sql: "Error: later query failed" if "bad" in sql else "[]",
        ),
    )

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        events = list(
            chat_mod.stream_agent_loop(
                model="gpt-4o",
                messages=[{"role": "user", "content": "why slow?"}],
                ui_context={},
                profile_path="/tmp/profile.sqlite",
                max_turns=2,
            )
        )

    text = "".join(event.get("content", "") for event in events if event["type"] == "text")
    assert "cannot answer this profile question" in text
    assert "later query failed" in text
    assert mock_lt.completion.call_count == 2


def test_stream_agent_loop_failed_compute_revokes_prior_evidence(monkeypatch):
    def tool_chunk(call_id, name, arguments):
        fn = MagicMock()
        fn.name = name
        fn.arguments = json.dumps(arguments)
        return MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(
                        content=None,
                        tool_calls=[MagicMock(index=0, id=call_id, function=fn)],
                    )
                )
            ],
            usage=None,
        )

    mock_lt = MagicMock()
    mock_lt.completion.side_effect = [
        iter([tool_chunk("db1", "query_profile_db", {"sql_query": "SELECT 1"})]),
        iter([tool_chunk("mfu1", "compute_mfu", {"step_time_s": 1.0})]),
    ]
    conn = MagicMock()
    monkeypatch.setattr(
        chat_mod,
        "_prepare_session",
        lambda *args, **kwargs: (
            conn,
            "/tmp/profile.sqlite",
            "system",
            lambda _sql: "[]",
        ),
    )

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        events = list(
            chat_mod.stream_agent_loop(
                model="gpt-4o",
                messages=[{"role": "user", "content": "calculate MFU"}],
                ui_context={},
                profile_path="/tmp/profile.sqlite",
                max_turns=2,
            )
        )

    text = "".join(event.get("content", "") for event in events if event["type"] == "text")
    assert "cannot answer this profile question" in text
    assert mock_lt.completion.call_count == 2


def test_stream_agent_loop_successful_compute_preserves_prior_evidence(monkeypatch):
    def tool_chunk(call_id, name, arguments):
        fn = MagicMock()
        fn.name = name
        fn.arguments = json.dumps(arguments)
        return MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(
                        content=None,
                        tool_calls=[MagicMock(index=0, id=call_id, function=fn)],
                    )
                )
            ],
            usage=None,
        )

    final_chunk = MagicMock(
        choices=[MagicMock(delta=MagicMock(content="Grounded result.", tool_calls=[]))],
        usage=None,
    )
    mock_lt = MagicMock()
    mock_lt.completion.side_effect = [
        iter([tool_chunk("db1", "query_profile_db", {"sql_query": "SELECT 1"})]),
        iter(
            [
                tool_chunk(
                    "mfu1",
                    "compute_mfu",
                    {
                        "step_time_s": 1.0,
                        "model_flops_per_step": 1e12,
                        "peak_tflops": 100.0,
                    },
                )
            ]
        ),
        iter([final_chunk]),
    ]
    conn = MagicMock()
    monkeypatch.setattr(
        chat_mod,
        "_prepare_session",
        lambda *args, **kwargs: (
            conn,
            "/tmp/profile.sqlite",
            "system",
            lambda _sql: "[]",
        ),
    )

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        events = list(
            chat_mod.stream_agent_loop(
                model="gpt-4o",
                messages=[{"role": "user", "content": "calculate MFU"}],
                ui_context={},
                profile_path="/tmp/profile.sqlite",
                max_turns=3,
            )
        )

    text = "".join(event.get("content", "") for event in events if event["type"] == "text")
    assert text == "Grounded result."


def test_stream_agent_loop_mixed_success_and_unknown_tool_failure_is_not_grounded(
    monkeypatch,
):
    query_fn = MagicMock()
    query_fn.name = "query_profile_db"
    query_fn.arguments = json.dumps({"sql_query": "SELECT 1"})
    unknown_fn = MagicMock()
    unknown_fn.name = "invented_tool"
    unknown_fn.arguments = "{}"
    chunk = MagicMock(
        choices=[
            MagicMock(
                delta=MagicMock(
                    content="The profile definitely regressed.",
                    tool_calls=[
                        MagicMock(index=0, id="db1", function=query_fn),
                        MagicMock(index=1, id="unknown1", function=unknown_fn),
                    ],
                )
            )
        ],
        usage=None,
    )
    mock_lt = MagicMock()
    mock_lt.completion.return_value = iter([chunk])
    conn = MagicMock()
    monkeypatch.setattr(
        chat_mod,
        "_prepare_session",
        lambda *args, **kwargs: (
            conn,
            "/tmp/profile.sqlite",
            "system",
            lambda _sql: "[]",
        ),
    )

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        events = list(
            chat_mod.stream_agent_loop(
                model="gpt-4o",
                messages=[{"role": "user", "content": "compare"}],
                ui_context={},
                profile_path="/tmp/profile.sqlite",
                max_turns=1,
            )
        )

    text = "".join(event.get("content", "") for event in events if event["type"] == "text")
    assert "cannot answer this profile question" in text
    assert "definitely regressed" not in text


def test_stream_agent_loop_missing_tool_id_revokes_without_execution(monkeypatch):
    def tool_chunk(call_id, name, arguments):
        fn = MagicMock()
        fn.name = name
        fn.arguments = json.dumps(arguments)
        return MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(
                        content=None,
                        tool_calls=[MagicMock(index=0, id=call_id, function=fn)],
                    )
                )
            ],
            usage=None,
        )

    malformed_calls = [
        ("query_profile_db", {"sql_query": "SELECT must_not_run"}),
        ("request_clarification", {"missing_information": "region_name"}),
    ]
    outputs = []
    query_log = []
    conn = MagicMock()
    monkeypatch.setattr(
        chat_mod,
        "_prepare_session",
        lambda *args, **kwargs: (
            conn,
            "/tmp/profile.sqlite",
            "system",
            lambda sql: query_log.append(sql) or "[]",
        ),
    )

    for malformed_name, malformed_args in malformed_calls:
        mock_lt = MagicMock()
        mock_lt.completion.side_effect = [
            iter(
                [
                    tool_chunk(
                        "db1",
                        "query_profile_db",
                        {"sql_query": "SELECT 1"},
                    )
                ]
            ),
            iter([tool_chunk(None, malformed_name, malformed_args)]),
            iter(
                [
                    MagicMock(
                        choices=[
                            MagicMock(
                                delta=MagicMock(
                                    content="The stale result proves a regression.",
                                    tool_calls=[],
                                )
                            )
                        ],
                        usage=None,
                    )
                ]
            ),
        ]
        with patch.dict(sys.modules, {"litellm": mock_lt}):
            events = list(
                chat_mod.stream_agent_loop(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "analyze"}],
                    ui_context={},
                    profile_path="/tmp/profile.sqlite",
                    max_turns=3,
                )
            )
        outputs.append(
            "".join(
                event.get("content", "")
                for event in events
                if event["type"] == "text"
            )
        )

    assert query_log == ["SELECT 1", "SELECT 1"]
    assert all("cannot answer this profile question" in output for output in outputs)
    assert all("stale result proves" not in output for output in outputs)
    assert all("Which NVTX region" not in output for output in outputs)


def test_stream_agent_loop_sibling_failure_suppresses_valid_ui_response(monkeypatch):
    ui_fn = MagicMock()
    ui_fn.name = "answer_from_ui_context"
    ui_fn.arguments = json.dumps({"context_paths": ["selected_kernel.name"]})
    malformed_fn = MagicMock()
    malformed_fn.name = None
    malformed_fn.arguments = "{}"
    conn = MagicMock()
    monkeypatch.setattr(
        chat_mod,
        "_prepare_session",
        lambda *args, **kwargs: (
            conn,
            "/tmp/profile.sqlite",
            "system",
            lambda _sql: "[]",
        ),
    )

    outputs = []
    call_counts = []
    histories = []
    for valid_index, malformed_index in ((0, 1), (1, 0)):
        valid_call = MagicMock(index=valid_index, id="ui1", function=ui_fn)
        malformed_call = MagicMock(
            index=malformed_index,
            id="malformed1",
            function=malformed_fn,
        )
        chunk = MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(
                        content="Unsupported prose.",
                        tool_calls=[valid_call, malformed_call],
                    )
                )
            ],
            usage=None,
        )
        mock_lt = MagicMock()
        mock_lt.completion.return_value = iter([chunk])
        with patch.dict(sys.modules, {"litellm": mock_lt}):
            events = list(
                chat_mod.stream_agent_loop(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "selected kernel?"}],
                    ui_context={"selected_kernel": {"name": "flash_fwd"}},
                    profile_path="/tmp/profile.sqlite",
                    max_turns=3,
                )
            )
        call_counts.append(mock_lt.completion.call_count)
        histories.append(mock_lt.completion.call_args.kwargs["messages"])
        outputs.append(
            "".join(
                event.get("content", "")
                for event in events
                if event["type"] == "text"
            )
        )

    assert all("cannot answer this profile question" in output for output in outputs)
    assert all("UI context:" not in output for output in outputs)
    assert all("Unsupported prose" not in output for output in outputs)
    assert call_counts == [1, 1]
    assert all(
        all(not message.get("tool_calls") for message in history)
        and all(message.get("role") != "tool" for message in history)
        for history in histories
    )


def test_stream_agent_loop_assembles_id_from_later_chunk(monkeypatch):
    """A successfully assembled SQL call still cannot ground a diagnosis alone."""
    first_delta = {
        "content": None,
        "tool_calls": [
            {
                "index": 0,
                "id": None,
                "function": {
                    "name": "query_profile_db",
                    "arguments": '{"sql_query":',
                },
            }
        ],
    }
    second_delta = {
        "content": None,
        "tool_calls": [
            {
                "index": 0,
                "id": "db-late",
                "function": {"name": None, "arguments": '"SELECT 1"}'},
            }
        ],
    }
    tool_stream = [
        MagicMock(choices=[MagicMock(delta=first_delta)], usage=None),
        MagicMock(choices=[MagicMock(delta=second_delta)], usage=None),
    ]
    answer_stream = [
        MagicMock(
            choices=[
                MagicMock(delta=MagicMock(content="Grounded answer.", tool_calls=[]))
            ],
            usage=None,
        )
    ]
    mock_lt = MagicMock()
    mock_lt.completion.side_effect = [iter(tool_stream), iter(answer_stream)]
    queries = []
    conn = MagicMock()
    monkeypatch.setattr(
        chat_mod,
        "_prepare_session",
        lambda *args, **kwargs: (
            conn,
            "/tmp/profile.sqlite",
            "system",
            lambda sql: queries.append(sql) or "[]",
        ),
    )

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        events = list(
            chat_mod.stream_agent_loop(
                model="gpt-4o",
                messages=[{"role": "user", "content": "analyze"}],
                ui_context={},
                profile_path="/tmp/profile.sqlite",
                max_turns=2,
            )
        )

    text = "".join(event.get("content", "") for event in events if event["type"] == "text")
    assert queries == ["SELECT 1"]
    assert "cannot answer this profile question" in text
    assert "Grounded answer." not in text


def test_stream_agent_loop_rejects_diff_answer_without_tool_evidence():
    guess_chunk = MagicMock(
        choices=[
            MagicMock(
                delta=MagicMock(
                    content="The after profile is definitely slower because of NCCL.",
                    tool_calls=[],
                )
            )
        ],
        usage=None,
    )
    mock_lt = MagicMock()
    mock_lt.completion.return_value = iter([guess_chunk])
    diff_context = MagicMock(marker=None)

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        events = list(
            chat_mod.stream_agent_loop(
                model="gpt-4o",
                messages=[{"role": "user", "content": "what regressed?"}],
                ui_context={},
                profile_path=None,
                max_turns=2,
                diff_context=diff_context,
                diff_paths=("/tmp/before.sqlite", "/tmp/after.sqlite"),
            )
        )

    text = "".join(event.get("content", "") for event in events if event["type"] == "text")
    assert "cannot answer this profile question" in text
    assert "did not query" in text
    assert "definitely slower" not in text


def test_stream_agent_loop_diff_compute_only_does_not_authorize_claims():
    assert "compute_mfu" not in chat_mod._DIFF_GROUNDING_TOOLS
    assert "get_source_code_context" not in chat_mod._DIFF_GROUNDING_TOOLS

    fn = MagicMock()
    fn.name = "compute_mfu"
    fn.arguments = json.dumps(
        {
            "step_time_s": 1.0,
            "model_flops_per_step": 1e12,
            "peak_tflops": 100.0,
        }
    )
    chunk = MagicMock(
        choices=[
            MagicMock(
                delta=MagicMock(
                    content="The after profile is faster.",
                    tool_calls=[MagicMock(index=0, id="mfu1", function=fn)],
                )
            )
        ],
        usage=None,
    )
    mock_lt = MagicMock()
    mock_lt.completion.return_value = iter([chunk])

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        events = list(
            chat_mod.stream_agent_loop(
                model="gpt-4o",
                messages=[{"role": "user", "content": "compare the profiles"}],
                ui_context={},
                profile_path=None,
                max_turns=1,
                diff_context=MagicMock(marker=None),
                diff_paths=("/tmp/before.sqlite", "/tmp/after.sqlite"),
            )
        )

    text = "".join(event.get("content", "") for event in events if event["type"] == "text")
    assert "cannot answer this profile question" in text
    assert "after profile is faster" not in text


def test_stream_agent_loop_token_warning(monkeypatch):
    """stream_agent_loop yields a system warning when prompt_tokens exceeds threshold (§11.9 Phase 4.1)."""
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content="Answer.", tool_calls=[]))]
    chunk.usage = MagicMock(prompt_tokens=35_000, completion_tokens=100)

    mock_lt = MagicMock()
    mock_lt.completion.return_value = iter([chunk])

    monkeypatch.setattr(chat_mod, "_get_model_and_key", lambda preferred=None: ("gpt-4o", "key"))
    with patch.dict(sys.modules, {"litellm": mock_lt}):
        if "litellm" in chat_mod.__dict__:
            del chat_mod.__dict__["litellm"]
        events = list(
            chat_mod.stream_agent_loop(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                ui_context={},
                profile_path=None,
                max_turns=1,
            )
        )
    system_events = [e for e in events if e.get("type") == "system"]
    # Should contain a token budget warning
    assert any("tokens" in (e.get("content") or "").lower() for e in system_events)


def test_build_system_prompt_with_schema_includes_sqlite_note():
    """System prompt with schema includes SQLite3 dialect note (§11.9 Pitfall 1)."""
    out = chat_mod._build_system_prompt({}, profile_schema="CREATE TABLE k(id INT)")
    assert "SQLite3" in out or "sqlite" in out.lower()
    assert "strftime" in out


def test_chat_completion_stream_no_db_agent(monkeypatch):
    """chat_completion_stream without DB agent uses stream_agent_loop (no profile)."""
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content="Hello", tool_calls=[]))]
    chunk.usage = None

    mock_lt = MagicMock()
    mock_lt.completion.return_value = iter([chunk])

    monkeypatch.setattr(chat_mod, "_get_model_and_key", lambda preferred=None: ("gpt-4o", "key"))
    monkeypatch.delenv("NSYS_AI_DB_AGENT", raising=False)
    with patch.dict(sys.modules, {"litellm": mock_lt}):
        if "litellm" in chat_mod.__dict__:
            del chat_mod.__dict__["litellm"]
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        raw = b"".join(chat_mod.chat_completion_stream(body))
    assert b"Hello" in raw
    assert b"event: done" in raw


def test_chat_completion_stream_no_model_reports_configuration_error(monkeypatch):
    """Streaming chat should emit visible text when no model/API key is configured."""
    monkeypatch.setattr(chat_mod, "_get_model_and_key", lambda preferred=None: (None, None))
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True}).encode()

    raw = b"".join(chat_mod.chat_completion_stream(body)).decode("utf-8")

    assert "event: text" in raw
    assert "LLM not configured" in raw
    assert "event: done" in raw


def test_stream_agent_loop_skill_names_injected(monkeypatch, tmp_path):
    """skill_names causes SESSION SKILL CONTEXT to appear in the system prompt sent to LLM."""
    import nsys_ai.prompt_loader as pl

    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "test_skill.md").write_text("UNIQUE_SKILL_MARKER_ABC123")
    monkeypatch.setattr(pl, "SKILLS_DIR", tmp_path)

    captured_messages: list = []

    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content="Ok", tool_calls=[]))]
    chunk.usage = None

    def fake_completion(**kwargs):
        captured_messages.extend(kwargs.get("messages", []))
        return iter([chunk])

    mock_lt = MagicMock()
    mock_lt.completion.side_effect = fake_completion

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        if "litellm" in chat_mod.__dict__:
            del chat_mod.__dict__["litellm"]
        list(
            chat_mod.stream_agent_loop(
                model="gpt-4o",
                messages=[{"role": "user", "content": "help"}],
                ui_context={},
                profile_path=None,
                max_turns=1,
                skill_names=["skills/test_skill.md"],
            )
        )

    system_msg = next((m for m in captured_messages if m.get("role") == "system"), None)
    assert system_msg is not None
    assert "UNIQUE_SKILL_MARKER_ABC123" in system_msg["content"]
    assert "SESSION SKILL CONTEXT" in system_msg["content"]


def test_stream_agent_loop_no_skill_names_backward_compat(monkeypatch):
    """Without skill_names, SESSION SKILL CONTEXT is absent (backward compatible)."""
    captured_messages: list = []

    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content="Hi", tool_calls=[]))]
    chunk.usage = None

    def fake_completion(**kwargs):
        captured_messages.extend(kwargs.get("messages", []))
        return iter([chunk])

    mock_lt = MagicMock()
    mock_lt.completion.side_effect = fake_completion

    with patch.dict(sys.modules, {"litellm": mock_lt}):
        if "litellm" in chat_mod.__dict__:
            del chat_mod.__dict__["litellm"]
        list(
            chat_mod.stream_agent_loop(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hello"}],
                ui_context={},
                profile_path=None,
                max_turns=1,
            )
        )

    system_msg = next((m for m in captured_messages if m.get("role") == "system"), None)
    assert system_msg is not None
    assert "SESSION SKILL CONTEXT" not in system_msg["content"]


# --- History distillation tests (§11.7) ---


def test_distill_history_compresses_tool_turns():
    """distill_history replaces intermediate tool call/result sequences with summaries."""
    messages = [
        {"role": "system", "content": "You are a test."},
        {"role": "user", "content": "What is the slowest kernel?"},
        # Intermediate: assistant with tool_calls + tool result
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "query_profile_db", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "query_profile_db",
            "content": '[{"name": "axpy", "total_ms": 42}]',
        },
        # Final assistant answer (no tool_calls)
        {"role": "assistant", "content": "The slowest kernel is axpy at 42ms."},
    ]
    result = chat_mod.distill_history(messages)
    # System and user messages preserved
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"
    # Intermediate tool turn compressed into a single summary
    assert result[2]["role"] == "system"
    assert "query_profile_db" in result[2]["content"]
    assert "1 result" in result[2]["content"]
    # Final assistant answer preserved
    assert result[3]["role"] == "assistant"
    assert "axpy" in result[3]["content"]
    # Total messages: 4 (system, user, summary, assistant) instead of 5
    assert len(result) == 4


def test_distill_history_preserves_simple_conversation():
    """distill_history does not modify conversations without tool calls."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Explain this kernel"},
        {"role": "assistant", "content": "This kernel is..."},
    ]
    result = chat_mod.distill_history(messages)
    assert result == messages
    assert len(result) == 4


def test_distill_history_empty():
    """distill_history returns empty list for empty input."""
    assert chat_mod.distill_history([]) == []


# ---------------------------------------------------------------------------
# On-demand skill routing tests
# ---------------------------------------------------------------------------


def test_route_skill_names_mfu():
    """MFU keywords → mfu.md injected."""
    msgs = [{"role": "user", "content": "what's my MFU?"}]
    result = chat_mod._route_skill_names(msgs)
    assert "skills/mfu.md" in result


def test_route_skill_names_navigation_empty():
    """Pure navigation query → no skills injected."""
    msgs = [{"role": "user", "content": "go to the volta_gemm_kernel please zoom in"}]
    result = chat_mod._route_skill_names(msgs)
    assert result == []


def test_route_skill_names_nccl():
    """NCCL keywords → distributed.md injected."""
    msgs = [{"role": "user", "content": "why is nccl so slow on multi-gpu?"}]
    result = chat_mod._route_skill_names(msgs)
    assert "skills/distributed.md" in result


def test_build_system_prompt_no_principles_by_default():
    """CORE PRINCIPLES block must NOT appear in the base prompt (removed from auto-injection)."""
    out = chat_mod._build_system_prompt({})
    assert "CORE PRINCIPLES" not in out, (
        "PRINCIPLES.md is still being auto-injected; it should only be loaded on-demand"
    )


def test_build_system_prompt_mfu_reference_still_present():
    """Hardcoded MFU REFERENCE block must still be in the base prompt."""
    out = chat_mod._build_system_prompt({})
    assert "MFU REFERENCE" in out
    assert "flops_per_layer" in out
    assert "SANITY CHECK" in out
    assert "num_gpus" in out
