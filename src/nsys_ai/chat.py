# ruff: noqa: I001
"""
chat.py — Agent loop and web-API handlers for the AI chat layer.

Architecture (three layers):
  chat_config.py   — Model registry and API-key resolution
  chat_tools.py    — Tool definitions, system prompt, action parsing
  chat.py  (this)  — LLM API calls, multi-turn agent loop, web/SSE handlers

Public names are re-exported from the sub-modules so that existing callers
(tests, tui_textual.py, web.py) can continue to do ``from .chat import …``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Callable

# ---------------------------------------------------------------------------
# Sub-module re-exports — keep the public API stable for existing callers.
# ---------------------------------------------------------------------------
from .chat_config import (  # noqa: F401
    MODEL_OPTIONS,
    _get_model_and_key,
    _model_to_key,
    get_available_models,
    get_default_model,
)
from .ai.backend.chat_tools import (  # noqa: F401
    TOOL_ANSWER_FROM_UI_CONTEXT,
    TOOL_REQUEST_CLARIFICATION,
    _build_system_prompt,
    _parse_tool_call,
    _tools_openai,
)
from .ai.backend.profile_db_tool import (
    get_profile_schema_cached,
    open_profile_readonly,
    query_profile_db,
)
from .diff_tools import TOOLS_DIFF_OPENAI, build_diff_system_prompt
from .skills.base import is_abstention_row

_log = logging.getLogger(__name__)
_telemetry_log = logging.getLogger("nsys_ai.telemetry")

# _finding_counter was previously module-level global state.
# Now handled per-request inside stream_agent_loop via nonlocal.
# (Removed global to avoid unbounded growth across requests.)


# ---------------------------------------------------------------------------
# Skill routing — keyword-based on-demand injection
# ---------------------------------------------------------------------------


def _route_skill_names(messages: list) -> list[str]:
    """Detect user intent from the last user message; return skill paths to inject.

    Called just before ``stream_agent_loop`` / ``_build_system_prompt`` so the
    right skill context is loaded for each query without injecting everything
    every time.

    Keyword → skill mappings:
      mfu / efficiency / utilization / tflops / flops / flash → skills/mfu.md
      bottleneck / triage / analyze / slow / investigate       → skills/triage.md
      nccl / distributed / multi-gpu / scaling / imbalance     → skills/distributed.md
      variance / spiky / spike / inconsistent / jitter         → skills/variance.md

    Navigation-only queries ("go to", "show", "zoom", "fit") return [].
    """
    last = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last = (m.get("content") or "").lower()
            break
    if not last:
        return []

    skills: list[str] = []
    if any(k in last for k in ("mfu", "efficiency", "utilization", "tflops", "flops", "flash")):
        skills.append("skills/mfu.md")
    if any(
        k in last for k in ("bottleneck", "triage", "analyze", "slow", "investigate", "what's in")
    ):
        skills.append("skills/triage.md")
    if any(k in last for k in ("nccl", "distributed", "multi-gpu", "scaling", "imbalance")):
        skills.append("skills/distributed.md")
    if any(k in last for k in ("variance", "spiky", "spike", "inconsistent", "jitter")):
        skills.append("skills/variance.md")
    return skills


# ---------------------------------------------------------------------------
# Agent-loop constants
# ---------------------------------------------------------------------------

# Cap total messages sent to the LLM per request; keeps token budget bounded.
MAX_AGENT_MESSAGES = 100
# Warn when prompt tokens exceed this threshold.
PROMPT_TOKEN_WARNING_THRESHOLD = 30_000
# Consecutive DB errors before injecting a break-cycle hint.
MAX_CONSECUTIVE_DB_ERRORS = 2
# Cap assistant content stored in history to prevent thinking-token leakage
# (Gemini 2.5 Pro streams thinking tokens as delta.content; if not capped they
# accumulate in api_messages and cause ContextWindowExceededError on turn N+1).
MAX_ASSISTANT_CONTENT_CHARS = 8_000
# thinking budget_tokens for Gemini 2.5 thinking models (limits per-turn thinking).
GEMINI_THINKING_BUDGET = 8_000

# Tools whose successful result is profile evidence rather than a UI action or
# a calculation from caller-supplied numbers.
_PROFILE_GROUNDING_TOOLS = frozenset(
    {
        "query_profile_db",
        "get_gpu_peak_tflops",
        "compute_region_mfu",
        "get_gpu_overlap_stats",
        "get_nccl_breakdown",
    }
)
_DIFF_GROUNDING_TOOLS = frozenset(
    {
        "search_nvtx_regions",
        "get_iteration_boundaries",
        "explore_nvtx_hierarchy",
        "get_top_nvtx_diffs",
        "get_iteration_diff",
        "get_region_diff",
        "summarize_nvtx_subtree",
        "get_launch_config_diff",
        "get_gpu_imbalance_stats",
        "get_global_diff",
        "get_memory_profile_diff",
        "get_gpu_peak_tflops",
    }
)
_CONTROL_RESPONSE_TOOLS = frozenset(
    {"request_clarification", "answer_from_ui_context"}
)
_CLARIFICATION_TEXT = {
    "model_flops_per_step": "What is model_flops_per_step?",
    "region_name": "Which NVTX region or CUDA kernel should I analyze?",
    "peak_tflops": "What peak TFLOPS value should I use for this GPU and precision?",
    "iteration_index": "Which iteration index should I analyze?",
}
_SENSITIVE_UI_PATH_TERMS = frozenset(
    {"api_key", "token", "secret", "password", "credential", "authorization", "cookie"}
)
_MAX_UI_CONTEXT_PATHS = 5
_MAX_UI_CONTEXT_VALUE_CHARS = 256


def _tool_result_failed(content: str) -> bool:
    """Return whether a tool result contains no usable grounding evidence."""
    text = (content or "").strip()
    if not text or text.startswith(
        ("Error:", "Not executed", "No diff context", "No profile loaded")
    ):
        return True
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False

    def _contains_unavailable(value) -> bool:
        if is_abstention_row(value):
            return True
        if isinstance(value, dict):
            return bool(value.get("error")) or any(
                _contains_unavailable(child) for child in value.values()
            )
        if isinstance(value, list):
            return any(_contains_unavailable(child) for child in value)
        return False

    return _contains_unavailable(payload)


def _cannot_answer_from_profile(reason: str | None = None) -> str:
    detail = (reason or "no profile data tool returned usable evidence").splitlines()[0]
    return (
        "I cannot answer this profile question because no supporting profile "
        f"evidence was retrieved. Reason: {detail}"
    )


def _with_control_response_tools(tools: list[dict]) -> list[dict]:
    """Expose explicit safe exits without duplicating caller-provided tools."""
    names = {tool.get("function", {}).get("name") for tool in tools}
    additions = [
        tool
        for tool in (TOOL_REQUEST_CLARIFICATION, TOOL_ANSWER_FROM_UI_CONTEXT)
        if tool["function"]["name"] not in names
    ]
    return [*tools, *additions]


def _ui_context_value(ui_context: dict, dotted_path: str):
    value = ui_context
    for part in dotted_path.split("."):
        if not part or not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _ui_path_is_sensitive(dotted_path: str) -> bool:
    segments = [segment.lower() for segment in dotted_path.split(".")]
    return any(
        sensitive in segment
        for segment in segments
        for sensitive in _SENSITIVE_UI_PATH_TERMS
    )


def _contains_sensitive_mapping_key(value) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or _ui_path_is_sensitive(key):
                return True
            if _contains_sensitive_mapping_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_sensitive_mapping_key(item) for item in value)
    return False


def _render_ui_context_value(value) -> str | None:
    if value in (None, "", [], {}):
        return None
    if _contains_sensitive_mapping_key(value):
        return None
    try:
        rendered = json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return None
    if len(rendered) > _MAX_UI_CONTEXT_VALUE_CHARS:
        return None
    return rendered


def _resolve_control_response(
    name: str,
    arguments: str,
    ui_context: dict | None,
) -> tuple[str | None, str | None]:
    """Validate a structured clarification or UI-grounded response."""
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return None, "invalid control-response arguments"
    if not isinstance(args, dict):
        return None, "control-response arguments must be an object"

    if name == "request_clarification":
        if set(args) != {"missing_information"}:
            return None, "clarification accepts only missing_information"
        missing = args.get("missing_information")
        question = _CLARIFICATION_TEXT.get(missing) if isinstance(missing, str) else None
        if question is None:
            return None, "unsupported missing_information identifier"
        return question, None

    if name == "answer_from_ui_context":
        if set(args) != {"context_paths"}:
            return None, "UI-context response accepts only context_paths"
        paths = args.get("context_paths")
        if (
            not isinstance(paths, list)
            or not paths
            or len(paths) > _MAX_UI_CONTEXT_PATHS
            or any(not isinstance(path, str) for path in paths)
            or len(set(paths)) != len(paths)
        ):
            return None, "UI-context response requires 1-5 unique context_paths"
        if not ui_context:
            return None, "no UI context was supplied"
        rendered_values = []
        for path in paths:
            if (
                len(path) > 120
                or _ui_path_is_sensitive(path)
            ):
                return None, "UI context path is invalid or sensitive"
            rendered = _render_ui_context_value(_ui_context_value(ui_context, path))
            if rendered is None:
                return None, "UI context path is missing, empty, or too large"
            rendered_values.append(f"{path}={rendered}")
        return "UI context: " + "; ".join(rendered_values), None

    return None, f"unknown control-response tool: {name}"


# ---------------------------------------------------------------------------
# History utilities
# ---------------------------------------------------------------------------


def _compact_old_tool_results(api_messages: list) -> None:
    """Replace large tool-result content from previous agent turns with summaries.

    Reduces prompt size when the model makes multiple DB queries per response.
    Tool results from all-but-the-last tool turn are replaced with a short
    placeholder if they exceed 200 chars.  The most recent turn's results are
    left intact so the model can use them for its final answer.
    """
    tool_turn_indices = [
        i
        for i, m in enumerate(api_messages)
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    if len(tool_turn_indices) < 2:
        return
    cutoff = tool_turn_indices[-1]
    for m in api_messages[:cutoff]:
        if m.get("role") == "tool" and len(m.get("content", "")) > 200:
            if _tool_result_failed(m["content"]):
                m["content"] = "[Summary: Tool failed; no profile evidence was produced.]"
            else:
                m["content"] = "[Summary: DB query returned results.]"


def distill_history(messages: list) -> list:
    """Compress intermediate tool call/result pairs from previous conversation turns.

    Strategy:
    - Keep system and user messages as-is.
    - Keep final assistant messages (no ``tool_calls``, i.e. the actual answers).
    - Replace each assistant-with-tool-calls + following tool-result sequence
      with a single system-role summary.

    Returns a new list; does **not** mutate the input.
    """
    if not messages:
        return messages

    result = []
    i = 0
    while i < len(messages):
        m = messages[i]
        role = m.get("role", "")

        if role in ("system", "user"):
            result.append(m)
            i += 1
            continue

        if role == "assistant" and m.get("tool_calls"):
            tool_names = [
                (tc.get("function") or {}).get("name", "unknown") for tc in m["tool_calls"]
            ]
            i += 1
            tool_count = 0
            while i < len(messages) and messages[i].get("role") == "tool":
                tool_count += 1
                i += 1
            summary = f"[Agent called {', '.join(tool_names)} ({tool_count} result(s) consumed)]"
            result.append({"role": "system", "content": summary})
            continue

        result.append(m)
        i += 1

    return result


# ---------------------------------------------------------------------------
# Multi-turn agent loop (non-streaming) — used by web.py and tests
# ---------------------------------------------------------------------------


def run_agent_loop(
    model: str,
    api_messages: list,
    tools: list | None = None,
    query_runner: Callable[[str], str] | None = None,
    max_turns: int = 5,
    ui_context: dict | None = None,
    dispatcher=None,
    event_sink: list[dict] | None = None,
) -> tuple[str, list]:
    """Run a multi-turn agent loop until the model stops calling tools.

    Args:
        model:         LiteLLM model identifier.
        api_messages:  Mutable message list (modified in-place as turns progress).
        tools:         OpenAI-style tool spec list; defaults to :func:`_tools_openai`.
        query_runner:  Callable ``(sql: str) -> str`` for ``query_profile_db``.
                       Pass ``None`` to treat DB calls as no-ops.
        dispatcher: Optional canonical profile-tool dispatcher. Web callers
                    provide this so non-streaming chat uses the same registry
                    path as the streaming loop.
        event_sink: Optional list receiving dispatcher side-effect events,
                    including finding overlays.
        max_turns:     Maximum number of LLM round-trips.
        ui_context:    Structured UI state available to ``answer_from_ui_context``.

    Returns:
        ``(final_content, actions)`` — *actions* are parsed navigation/zoom dicts.
    """
    try:
        import litellm
    except ImportError:
        return ("LLM not available (install litellm).", [])

    tools = _with_control_response_tools(tools if tools is not None else _tools_openai())
    actions: list = []
    consecutive_db_errors = 0
    profile_grounding_required = query_runner is not None
    grounding_attempted = False
    evidence_ready = False
    grounding_failure: str | None = None
    exploratory_query_succeeded = False

    for _ in range(max_turns):
        _compact_old_tool_results(api_messages)
        if len(api_messages) > MAX_AGENT_MESSAGES:
            api_messages[:] = [api_messages[0]] + api_messages[-(MAX_AGENT_MESSAGES - 1) :]
        response = litellm.completion(
            model=model,
            messages=api_messages,
            tools=tools,
            tool_choice="auto",
        )
        choice = response.choices[0] if response.choices else None
        if not choice:
            if profile_grounding_required and not evidence_ready:
                return (_cannot_answer_from_profile("the provider returned no response"), actions)
            return ("", actions)
        message = choice.message
        if isinstance(message, dict):
            content = (message.get("content") or "").strip()
            tool_calls = message.get("tool_calls") or []
        else:
            content = (getattr(message, "content", None) or "").strip()
            tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            if (profile_grounding_required or grounding_attempted) and not evidence_ready:
                return (
                    _cannot_answer_from_profile(
                        grounding_failure or "the model did not query the loaded profile"
                    ),
                    actions,
                )
            return (content, actions)

        tc_list = []
        for tc in tool_calls:
            fn = (
                getattr(tc, "function", None)
                if not isinstance(tc, dict)
                else tc.get("function") or {}
            )
            tc_id = getattr(tc, "id", None) if not isinstance(tc, dict) else tc.get("id")
            name = getattr(fn, "name", None) if not isinstance(fn, dict) else fn.get("name")
            args_str = (
                getattr(fn, "arguments", None) if not isinstance(fn, dict) else fn.get("arguments")
            ) or "{}"
            tc_list.append((tc_id, name, args_str))

        batch_history_start = len(api_messages)
        valid_tc_list = [(tid, name, args) for tid, name, args in tc_list if tid and name]
        if valid_tc_list:
            api_messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": tid,
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }
                        for tid, name, args in valid_tc_list
                    ],
                }
            )

        has_external = False
        control_response: str | None = None
        turn_grounding_succeeded = False
        turn_tool_failed = False
        compute_mfu_succeeded = False
        for tc_id, name, args_str in tc_list:
            if not name or not tc_id:
                grounding_failure = "Invalid tool call: missing name or id."
                turn_tool_failed = True
                continue
            action = _parse_tool_call(name, args_str)
            if action:
                has_external = True
                actions.append(action)
            elif name in _CONTROL_RESPONSE_TOOLS:
                response_text, control_error = _resolve_control_response(
                    name, args_str, ui_context
                )
                if response_text is not None:
                    control_response = response_text
                else:
                    grounding_failure = f"Error: {control_error}"
                    turn_tool_failed = True
                    api_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": name,
                            "content": f"Error: {control_error}",
                        }
                    )
            elif name == "query_profile_db":
                grounding_attempted = True
                if query_runner is not None:
                    try:
                        sql = json.loads(args_str).get("sql_query", "")
                        result = query_runner(sql)
                    except Exception as e:
                        _log.debug("Tool query_profile_db failed: %s", e, exc_info=True)
                        result = f"Error: {e}"
                    if result.startswith("Error:"):
                        grounding_failure = result
                        turn_tool_failed = True
                        consecutive_db_errors += 1
                        if consecutive_db_errors >= MAX_CONSECUTIVE_DB_ERRORS:
                            result += (
                                "\n[System: Repeated SQL errors. "
                                "Do not infer profile facts. Unless another successful tool result "
                                "supports the answer, state that the question cannot be answered.]"
                            )
                    else:
                        # Exploratory SQL is intentionally not sufficient to
                        # ground a profile diagnosis. A registered analysis
                        # skill must provide the evidence path.
                        exploratory_query_succeeded = True
                        consecutive_db_errors = 0
                else:
                    result = "Not executed (no profile loaded)."
                    grounding_failure = result
                    turn_tool_failed = True
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": name,
                        "content": result,
                    }
                )
            else:
                if dispatcher is not None and dispatcher.knows(name):
                    dispatched = dispatcher.dispatch(name, args_str)
                    if event_sink is not None:
                        event_sink.extend(dispatched.events)
                    tool_failed = _tool_result_failed(dispatched.content)
                    if tool_failed:
                        grounding_failure = dispatched.content
                        turn_tool_failed = True
                    if name in _PROFILE_GROUNDING_TOOLS:
                        grounding_attempted = True
                        if not tool_failed and name != "query_profile_db":
                            turn_grounding_succeeded = True
                    if name == "compute_mfu" and not tool_failed:
                        compute_mfu_succeeded = True
                    tool_result = dispatched.content
                else:
                    # Tools unavailable to this compatibility wrapper get an
                    # explicit response rather than silently being skipped.
                    if name in {
                        "get_gpu_peak_tflops",
                        "compute_mfu",
                        "compute_region_mfu",
                        "compute_theoretical_flops",
                    }:
                        tool_result = (
                            f"Tool '{name}' is only supported in the streaming API path "
                            "and cannot be executed in this non-streaming request."
                        )
                    else:
                        tool_result = "Not executed."
                    if name in _PROFILE_GROUNDING_TOOLS:
                        grounding_attempted = True
                    grounding_failure = tool_result
                    turn_tool_failed = True
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": name,
                        "content": tool_result,
                    }
                )

        if (
            not turn_tool_failed
            and exploratory_query_succeeded
            and compute_mfu_succeeded
        ):
            # The pure calculation is grounded because its profile-derived
            # input was retrieved in this conversation, independent of tool
            # call ordering within the batch.
            turn_grounding_succeeded = True

        if turn_tool_failed:
            evidence_ready = False
            del api_messages[batch_history_start:]
            return (
                _cannot_answer_from_profile(
                    grounding_failure or "a tool call was invalid or failed"
                ),
                actions,
            )
        elif turn_grounding_succeeded:
            evidence_ready = True

        if control_response is not None and not turn_tool_failed:
            return (control_response, actions)
        if has_external:
            # The action itself is structured and validated. Companion model
            # prose is not, so never let a navigation call smuggle a diagnosis.
            return ("", actions)

    if (profile_grounding_required or grounding_attempted) and not evidence_ready:
        return (_cannot_answer_from_profile(grounding_failure or "maximum turns reached"), actions)
    return ("Max turns reached.", actions)


# ---------------------------------------------------------------------------
# Web-API handler (non-streaming)
# ---------------------------------------------------------------------------


def _prepare_session(
    profile_path: str | None,
    messages: list,
    ui_context: dict,
    explicit_skills: list[str] | None = None,
) -> tuple:
    """Common setup: resolve profile → open readonly → schema → skill routing → system prompt.

    Returns (conn, sqlite_path, system_prompt, query_runner).
    Raises RuntimeError on profile path resolution errors.

    Thread affinity — why this does not use ``Profile.query_conn()``.
    ``open_profile_readonly`` returns a fresh connection per call, never a
    cursor on a shared one: ``duckdb.connect()`` on the Parquet-cache path — a
    private in-memory database, so concurrent sessions do not even contend on
    DuckDB's per-database query lock — and, when that cache cannot be opened,
    a read-only ``sqlite3.connect(..., uri=True)`` fallback. ``query_conn()``
    solves a different problem: the analysis path must share one database
    because of ``CREATE TEMP TABLE`` scratch tables and the memoized skill bag,
    and hands out per-thread cursors so that sharing stays correct.

    The affinity is real, not incidental: this runs *inside* the
    ``stream_agent_loop`` generator body (and inside the synchronous
    ``chat_completion``), so the connection belongs to whichever thread
    advances the generator, and it is closed on that same thread. Every caller
    builds and drains the generator on one thread — ``tui_textual.py``'s
    ``@work(thread=True)`` worker, ``tree/chat.py``'s stream worker,
    ``cli/handlers.py``, and ``web.py``'s per-request handler thread. This is
    the same rule ``open_profile_readonly_for_worker`` states for callers that
    open a connection themselves; the difference is only that here the opening
    is implicit in when the generator is first advanced.

    Two changes would break it, for two *different* reasons — do not conflate
    them. **Memoizing per path** collapses the private databases into one
    shared handle, which serves concurrent queries wrong rows with nothing
    raised. **Hoisting the setup out of the generator** shares nothing (each
    call still opens its own connection) but creates it on the thread that
    builds the generator and closes it on the thread that drains it. On DuckDB
    that mismatch is merely unenforced; on the SQLite fallback it raises
    ``sqlite3.ProgrammingError`` — ``open_profile_readonly`` leaves
    ``check_same_thread`` at its default, so the handle is usable and closable
    only from its creating thread. Both are pinned by
    ``tests/test_chat_connection_threading.py``.
    """
    from .profile import resolve_profile_path

    conn = None
    sqlite_path = None
    schema_str = None
    query_runner = None

    if profile_path:
        sqlite_path = resolve_profile_path(profile_path)
        conn = open_profile_readonly(sqlite_path)
        try:
            schema_str = get_profile_schema_cached(conn, sqlite_path)
        except Exception:
            _log.debug("Schema cache failed, closing connection", exc_info=True)
            conn.close()
            raise

        def _runner(sql, c=conn):
            return query_profile_db(c, sql)

        query_runner = _runner

    _effective_skills = explicit_skills
    if not _effective_skills and messages:
        try:
            _effective_skills = _route_skill_names(messages)
        except Exception:
            _log.debug("Skill name routing failed", exc_info=True)

    _skill_docs = None
    if _effective_skills:
        try:
            from .prompt_loader import load_skill_context

            _skill_docs = load_skill_context(_effective_skills) or None
        except Exception:
            _log.debug("Skill context loading failed", exc_info=True)

    system_prompt = _build_system_prompt(
        ui_context, profile_schema=schema_str, skill_docs=_skill_docs
    )

    return conn, sqlite_path, system_prompt, query_runner


def chat_completion(body_bytes: bytes) -> dict | None:
    """Handle a POST ``/api/chat`` request body.

    Returns ``{"content": str, "actions": list}`` or ``None`` for 501
    (LLM not configured / not installed).
    """
    try:
        payload = json.loads(body_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"content": "Invalid request body.", "actions": [], "findings": []}

    try:
        import litellm
    except ImportError:
        return None

    model, _ = _get_model_and_key(payload.get("model"))
    if not model:
        return None

    messages = payload.get("messages") or []
    ui_context = payload.get("ui_context") or {}
    profile_path = payload.get("profile_path")

    from nsys_ai.exceptions import NsysAiError

    try:
        conn, sqlite_path, system_prompt, query_runner = _prepare_session(
            profile_path, messages, ui_context
        )
    except (RuntimeError, NsysAiError) as e:
        return {"content": f"Profile error: {e}", "actions": [], "findings": []}

    api_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        if m.get("role") and m.get("content") is not None:
            api_messages.append({"role": m["role"], "content": m["content"]})

    if profile_path and conn:
        try:
            from .tool_dispatch import ToolDispatcher

            local_finding_counter = payload.get("findings_count", 0)
            if not isinstance(local_finding_counter, int) or local_finding_counter < 0:
                local_finding_counter = 0

            def _next_finding_index() -> int:
                nonlocal local_finding_counter
                local_finding_counter += 1
                return local_finding_counter

            dispatcher_events: list[dict] = []
            content, actions = run_agent_loop(
                model=model,
                api_messages=api_messages,
                tools=_tools_openai(),
                query_runner=query_runner,
                max_turns=5,
                ui_context=ui_context,
                event_sink=dispatcher_events,
                dispatcher=ToolDispatcher(
                    conn=conn,
                    sqlite_path=sqlite_path,
                    query_runner=query_runner,
                    finding_counter=_next_finding_index,
                ),
            )
            findings = [
                event["finding"]
                for event in dispatcher_events
                if event.get("type") == "finding" and isinstance(event.get("finding"), dict)
            ]
            return {"content": content, "actions": actions, "findings": findings}
        except Exception as e:
            return {
                "content": f"LLM error: {_friendly_error(model, e)}",
                "actions": [],
                "findings": [],
            }
        finally:
            conn.close()

    try:
        response = litellm.completion(
            model=model,
            messages=api_messages,
            tools=_tools_openai(),
            tool_choice="auto",
        )
    except Exception as e:
        return {
            "content": f"LLM error: {_friendly_error(model, e)}",
            "actions": [],
            "findings": [],
        }

    choice = response.choices[0] if response.choices else None
    if not choice:
        return {"content": "", "actions": [], "findings": []}
    message = choice.message
    if isinstance(message, dict):
        content = (message.get("content") or "").strip()
        tool_calls = message.get("tool_calls") or []
    else:
        content = (getattr(message, "content", None) or "").strip()
        tool_calls = getattr(message, "tool_calls", None) or []

    actions = []
    for tc in tool_calls:
        fn = getattr(tc, "function", None) if not isinstance(tc, dict) else tc.get("function") or {}
        name = getattr(fn, "name", None) if not isinstance(fn, dict) else fn.get("name")
        args_str = (
            getattr(fn, "arguments", None) if not isinstance(fn, dict) else fn.get("arguments")
        ) or "{}"
        if name:
            action = _parse_tool_call(name, args_str)
            if action:
                actions.append(action)
    return {"content": content, "actions": actions, "findings": []}


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------


def _sse_event(evt: str, data: dict) -> bytes:
    return f"event: {evt}\ndata: {json.dumps(data)}\n\n".encode()


# ---------------------------------------------------------------------------
# Streaming agent loop — UI-agnostic generator (used by tui_textual + web)
# ---------------------------------------------------------------------------


def _stream_litellm_content(stream, usage: dict):
    """Consume a litellm stream; yield text events and update usage in place."""
    for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if not choice:
            continue
        delta = getattr(choice, "delta", None) or (
            choice.get("delta") if isinstance(choice, dict) else None
        )
        if not delta:
            continue
        c = getattr(delta, "content", None) if not isinstance(delta, dict) else delta.get("content")
        if c:
            yield {"type": "text", "content": c}
        u = getattr(chunk, "usage", None) or (
            chunk.get("usage") if isinstance(chunk, dict) else None
        )
        if u:
            usage.clear()
            usage.update(
                u
                if isinstance(u, dict)
                else {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0),
                    "completion_tokens": getattr(u, "completion_tokens", 0),
                }
            )


def stream_agent_loop(
    model: str,
    messages: list,
    ui_context: dict,
    tools: list | None = None,
    profile_path: str | None = None,
    diff_context=None,
    diff_paths: tuple[str, str] | None = None,
    max_turns: int = 5,
    skill_names: list[str] | None = None,
    findings_count: int = 0,
):
    """UI-agnostic streaming agent loop — yields event dicts.

    Yielded event types:

    * ``{"type": "text",   "content": str}``   — streamed text fragment
    * ``{"type": "system", "content": str}``   — status / warning message
    * ``{"type": "action", "action": dict}``   — navigation/zoom action
    * ``{"type": "done",   "usage": dict}``    — final event with token usage

    When *diff_context* is set, uses Phase C diff tools and no single-profile DB.
    *diff_paths* must be (before_path, after_path) for the system prompt.
    The profile connection (when *profile_path* is given) is opened in this
    generator and closed in the ``finally`` block.  Call this from a background
    thread (e.g. Textual ``@work(thread=True)``) so the main thread's UI
    remains responsive during DB queries and LLM streaming.

    Build and drain the generator on the same thread. No connection or cursor
    is shared between invocations — the only mutable module-level state this
    path touches is ``profile_db_tool._schema_cache``, which is lock-guarded
    and holds strings — so two turns may overlap freely, and they do:
    ``@work(thread=True, exclusive=True)`` cancels the asyncio task, not the OS
    thread, so a cancelled chat turn is not stopped by anything Textual does
    (its own docs: thread workers cannot be interrupted, only asked to check
    ``is_cancelled``). Both Textual consumers do the asking — ``tui_textual``
    on ``worker.is_cancelled``, ``tree/chat.py`` on its own cancel event — and
    break out of the loop, so the overlap now lasts until the abandoned turn's
    next event rather than to the end of its turn. Bounded, not eliminated: the
    break cannot land mid-event, and the replacement turn starts before the
    abandoned one notices. That overlap is harmless only because each
    invocation holds its own connection; see ``_prepare_session``. The
    *diff_context* path holds no connection of its own — its ``ToolDispatcher``
    is built with ``conn=None``, as is the one in ``diff_tools.run_diff_tool``
    — but its ``DiffContext`` (and the two ``Profile`` objects inside it) is
    caller-owned, so it inherits its owner's thread; today's only caller is the
    single-threaded ``nsys-ai diff --chat`` REPL.

    *skill_names* — optional list of skill file paths relative to the
    ``src/nsys_ai/agent_skills/`` directory (e.g. ``["skills/mfu.md"]``). When
    provided, their contents are concatenated and appended to the system
    prompt as a SESSION SKILL CONTEXT block.  Uses ``prompt_loader``
    internally; missing files are silently ignored.
    """
    try:
        import litellm
    except ImportError:
        yield {"type": "text", "content": "LLM not available (install litellm)."}
        yield {"type": "done", "usage": {}}
        return

    # Per-request finding counter (replaces old module-level global).
    _local_finding_counter = findings_count

    def _next_finding_index() -> int:
        nonlocal _local_finding_counter
        _local_finding_counter += 1
        return _local_finding_counter

    use_diff = diff_context is not None and diff_paths is not None
    tools = tools if tools is not None else (TOOLS_DIFF_OPENAI if use_diff else _tools_openai())
    tools = _with_control_response_tools(tools)
    if use_diff:
        system_prompt = build_diff_system_prompt(
            diff_context, diff_paths[0], diff_paths[1], snapshot=None
        )
        conn = None
        sqlite_path = None
        query_runner = None
    else:
        from nsys_ai.exceptions import NsysAiError

        try:
            conn, sqlite_path, system_prompt, query_runner = _prepare_session(
                profile_path, messages, ui_context, skill_names
            )
        except (RuntimeError, NsysAiError) as e:
            yield {"type": "text", "content": f"Profile error: {e}"}
            yield {"type": "done", "usage": {}}
            return
        except Exception as e:
            _log.warning("Profile session setup failed: %s", e, exc_info=True)
            yield {"type": "text", "content": f"Error loading profile data: {e}"}
            yield {"type": "done", "usage": {}}
            return

    # Fix 2: Filter out DB-dependent tools when no profile is connected.
    # This prevents LLM from calling tools that always fail, avoiding retry spirals.
    _DB_TOOLS = {
        "query_profile_db",
        "get_gpu_peak_tflops",
        "compute_region_mfu",
        "get_gpu_overlap_stats",
        "get_nccl_breakdown",
    }
    if not use_diff and conn is None and tools:
        tools = [t for t in tools if t.get("function", {}).get("name") not in _DB_TOOLS]

    api_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        if m.get("role") and m.get("content") is not None:
            api_messages.append({"role": m["role"], "content": m["content"]})

    usage: dict = {}
    turn_count = 0

    # Centralized tool dispatcher (replaces if/elif chain)
    from .tool_dispatch import ToolDispatcher

    dispatcher = ToolDispatcher(
        conn=conn,
        sqlite_path=sqlite_path,
        query_runner=query_runner,
        finding_counter=_next_finding_index,
        mode="diff" if use_diff else "profile",
        diff_context=diff_context,
    )
    grounding_required = use_diff or query_runner is not None
    grounding_tools = _DIFF_GROUNDING_TOOLS if use_diff else _PROFILE_GROUNDING_TOOLS
    grounding_attempted = False
    evidence_ready = False
    grounding_failure: str | None = None
    exploratory_query_succeeded = False

    try:
        for _ in range(max_turns):
            turn_count += 1
            _compact_old_tool_results(api_messages)
            if len(api_messages) > MAX_AGENT_MESSAGES:
                api_messages[:] = [api_messages[0]] + api_messages[-(MAX_AGENT_MESSAGES - 1) :]

            extra_kwargs: dict = {}
            if "gemini-2.5" in model:
                # Fix 4: Use smaller thinking budget for tool-result turns
                # to speed up tool-call processing.
                budget = GEMINI_THINKING_BUDGET if turn_count == 1 else 2000
                extra_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget,
                }

            try:
                stream = litellm.completion(
                    model=model,
                    messages=api_messages,
                    tools=tools,
                    tool_choice="auto",
                    stream=True,
                    **extra_kwargs,
                )
            except Exception as e:
                yield {"type": "text", "content": f"LLM error: {_friendly_error(model, e)}"}
                yield {"type": "done", "usage": usage}
                return

            content_parts: list[str] = []
            tool_calls_by_index: dict[int, dict] = {}

            try:
                for chunk in stream:
                    choice = chunk.choices[0] if chunk.choices else None
                    if not choice:
                        continue
                    delta = getattr(choice, "delta", None) or (
                        choice.get("delta") if isinstance(choice, dict) else None
                    )
                    if not delta:
                        continue
                    c = (
                        getattr(delta, "content", None)
                        if not isinstance(delta, dict)
                        else delta.get("content")
                    )
                    if c:
                        content_parts.append(c)

                    tcs = (
                        getattr(delta, "tool_calls", None)
                        if not isinstance(delta, dict)
                        else delta.get("tool_calls")
                    ) or []
                    for tc in tcs:
                        idx = (
                            getattr(tc, "index", 0)
                            if not isinstance(tc, dict)
                            else tc.get("index", 0)
                        )
                        tc_id = (
                            getattr(tc, "id", None) if not isinstance(tc, dict) else tc.get("id")
                        )
                        fn = (
                            getattr(tc, "function", None)
                            if not isinstance(tc, dict)
                            else tc.get("function") or {}
                        )
                        if isinstance(fn, dict):
                            name, args = fn.get("name"), fn.get("arguments") or ""
                        else:
                            name, args = (
                                getattr(fn, "name", None),
                                getattr(fn, "arguments", None) or "",
                            )
                        entry = tool_calls_by_index.setdefault(
                            idx, {"id": None, "name": None, "arguments": ""}
                        )
                        if tc_id:
                            entry["id"] = tc_id
                        if name:
                            entry["name"] = name
                        entry["arguments"] += args

                    u = getattr(chunk, "usage", None) or (
                        chunk.get("usage") if isinstance(chunk, dict) else None
                    )
                    if u:
                        usage = (
                            u
                            if isinstance(u, dict)
                            else {
                                "prompt_tokens": getattr(u, "prompt_tokens", 0),
                                "completion_tokens": getattr(u, "completion_tokens", 0),
                            }
                        )
            except litellm.exceptions.ContextWindowExceededError:
                yield {
                    "type": "text",
                    "content": (
                        "\n\n⚠ Context window exceeded — the conversation history grew too large "
                        "(likely due to accumulated thinking tokens). "
                        "Please start a new chat session to continue."
                    ),
                }
                yield {"type": "done", "usage": usage}
                return

            full_content = "".join(content_parts).strip() if content_parts else ""
            # Cap stored content to prevent thinking-token leakage into future turns.
            if len(full_content) > MAX_ASSISTANT_CONTENT_CHARS:
                full_content = full_content[:MAX_ASSISTANT_CONTENT_CHARS]
            tc_list = [
                (t.get("id"), t.get("name"), t.get("arguments") or "{}")
                for _, t in sorted(tool_calls_by_index.items())
            ]

            if usage:
                pt = (
                    usage.get("prompt_tokens", 0)
                    if isinstance(usage, dict)
                    else getattr(usage, "prompt_tokens", 0)
                )
                if isinstance(pt, int) and pt > PROMPT_TOKEN_WARNING_THRESHOLD:
                    _log.warning(
                        "stream_agent_loop: high prompt token usage (%d). model=%s", pt, model
                    )
                    yield {
                        "type": "system",
                        "content": f"⚠ Large context ({pt:,} tokens). Consider starting a new chat to reduce cost.",
                    }

            if not tc_list:
                if grounding_required and not evidence_ready:
                    yield {
                        "type": "text",
                        "content": _cannot_answer_from_profile(
                            grounding_failure or "the model did not query the loaded profile"
                        ),
                    }
                elif full_content:
                    # Buffer the complete assistant turn until we know it has
                    # no later tool call whose failure would invalidate it.
                    yield {"type": "text", "content": full_content}
                yield {"type": "done", "usage": usage}
                return

            batch_history_start = len(api_messages)
            valid_tc_list = [(tid, name, args) for tid, name, args in tc_list if tid and name]
            if valid_tc_list:
                api_messages.append(
                    {
                        "role": "assistant",
                        "content": full_content or None,
                        "tool_calls": [
                            {
                                "id": tid,
                                "type": "function",
                                "function": {"name": name, "arguments": args},
                            }
                            for tid, name, args in valid_tc_list
                        ],
                    }
                )

            has_external = False
            control_response: str | None = None
            turn_grounding_succeeded = False
            turn_tool_failed = False
            compute_mfu_succeeded = False
            for tid, name, args_str in tc_list:
                if not name or not tid:
                    grounding_failure = "Invalid tool call: missing name or id."
                    turn_tool_failed = True
                    continue

                if name in _CONTROL_RESPONSE_TOOLS:
                    response_text, control_error = _resolve_control_response(
                        name, args_str, ui_context
                    )
                    if response_text is not None:
                        control_response = response_text
                    else:
                        grounding_failure = f"Error: {control_error}"
                        turn_tool_failed = True
                        api_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tid,
                                "name": name,
                                "content": f"Error: {control_error}",
                            }
                        )
                    continue

                # 1) Profile and Diff tools — use the centralized dispatcher
                if dispatcher.knows(name):
                    tr = dispatcher.dispatch(name, args_str)
                    tool_failed = _tool_result_failed(tr.content)
                    if tool_failed:
                        grounding_failure = tr.content
                        turn_tool_failed = True
                    if name in grounding_tools:
                        grounding_attempted = True
                        # Exploratory SQL can support a registered analysis,
                        # but it cannot ground a diagnosis by itself.
                        if not tool_failed and name != "query_profile_db":
                            turn_grounding_succeeded = True
                        if name == "query_profile_db" and not tool_failed:
                            exploratory_query_succeeded = True
                    elif name == "query_profile_db":
                        grounding_attempted = True
                        if not tool_failed:
                            exploratory_query_succeeded = True
                    elif name == "compute_mfu":
                        # A pure MFU calculation is grounded only when its
                        # profile-derived input was retrieved first.
                        if not tool_failed:
                            compute_mfu_succeeded = True
                    yield from tr.events
                    if not tr.skip_tool_message:
                        api_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tid,
                                "name": name,
                                "content": tr.content,
                            }
                        )
                    continue

                # 3) Navigation / zoom / fit_nvtx — external actions
                action = _parse_tool_call(name, args_str)
                if action:
                    has_external = True
                    yield {"type": "action", "action": action}
                else:
                    # Unknown tool — send a stub response to avoid LLM confusion
                    grounding_failure = "An unknown tool was not executed."
                    turn_tool_failed = True
                    api_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tid,
                            "name": name,
                            "content": "Not executed.",
                        }
                    )

            if (
                not turn_tool_failed
                and exploratory_query_succeeded
                and compute_mfu_succeeded
            ):
                # Resolve the dependency after the whole tool batch so the
                # model's tool-call ordering cannot change grounding.
                turn_grounding_succeeded = True

            if turn_tool_failed:
                evidence_ready = False
                del api_messages[batch_history_start:]
                yield {
                    "type": "text",
                    "content": _cannot_answer_from_profile(
                        grounding_failure or "a tool call was invalid or failed"
                    ),
                }
                yield {"type": "done", "usage": usage}
                return
            elif turn_grounding_succeeded:
                evidence_ready = True

            if control_response is not None and not turn_tool_failed:
                yield {"type": "text", "content": control_response}
                yield {"type": "done", "usage": usage}
                return
            if has_external:
                yield {"type": "done", "usage": usage}
                return

        if grounding_required and not evidence_ready:
            yield {
                "type": "text",
                "content": _cannot_answer_from_profile(
                    grounding_failure
                    or (
                        "profile tools returned no usable evidence"
                        if grounding_attempted
                        else "the model did not query the loaded profile"
                    )
                ),
            }
            yield {"type": "done", "usage": usage}
            return

        # Exhausted max_turns; last message was a tool result. One more LLM call with
        # tool_choice="none" so the model can synthesize a final summary.
        if api_messages and api_messages[-1].get("role") == "tool":
            extra = {}
            if "gemini-2.5" in model:
                extra["thinking"] = {"type": "enabled", "budget_tokens": GEMINI_THINKING_BUDGET}
            try:
                stream = litellm.completion(
                    model=model,
                    messages=api_messages,
                    tools=tools,
                    tool_choice="none",
                    stream=True,
                    **extra,
                )
                yield from _stream_litellm_content(stream, usage)
            except Exception as e:
                _log.debug("Summary LLM call failed: %s", e, exc_info=True)
                yield {
                    "type": "text",
                    "content": f"\n\n(Summary skipped: {_friendly_error(model, e)})",
                }
        yield {"type": "done", "usage": usage}

    finally:
        if usage:
            _telemetry_log.info(
                "agent_usage model=%s prompt_tokens=%s completion_tokens=%s turns=%d",
                model,
                usage.get("prompt_tokens", "?")
                if isinstance(usage, dict)
                else getattr(usage, "prompt_tokens", "?"),
                usage.get("completion_tokens", "?")
                if isinstance(usage, dict)
                else getattr(usage, "completion_tokens", "?"),
                turn_count,
            )
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Web-API streaming handler (SSE)
# ---------------------------------------------------------------------------


def chat_completion_stream(body_bytes: bytes):
    """Generator yielding SSE bytes for the streaming web endpoint.

    Always delegates to :func:`stream_agent_loop`.  The *profile_path* field,
    when provided in the request payload, is passed through directly.
    """
    try:
        payload = json.loads(body_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        yield _sse_event("text", {"chunk": "Invalid request body."})
        yield _sse_event("done", {})
        return

    model, _ = _get_model_and_key(payload.get("model"))
    if not model:
        yield _sse_event(
            "text",
            {
                "chunk": (
                    "LLM not configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
                    "or GEMINI_API_KEY before starting nsys-ai."
                )
            },
        )
        yield _sse_event("done", {"error": "LLM not configured"})
        return

    messages = payload.get("messages") or []
    ui_context = payload.get("ui_context") or {}
    profile_path = payload.get("profile_path")
    # skill_context: optional list of skill paths (e.g. ["skills/mfu.md"]).
    # When provided, those files are loaded from src/nsys_ai/agent_skills/ and appended
    # to the system prompt as SESSION SKILL CONTEXT. Unknown paths are silently ignored.
    skill_context: list[str] | None = payload.get("skill_context") or None
    effective_profile = profile_path if profile_path else None

    findings_count = 0
    raw_fc = payload.get("findings_count")
    if isinstance(raw_fc, int) and raw_fc >= 0:
        findings_count = raw_fc

    try:
        for ev in stream_agent_loop(
            model=model,
            messages=messages,
            ui_context=ui_context,
            tools=_tools_openai(),
            profile_path=effective_profile,
            max_turns=5,
            skill_names=skill_context,
            findings_count=findings_count,
        ):
            t = ev.get("type")
            if t == "text":
                yield _sse_event("text", {"chunk": ev.get("content", "")})
            elif t == "system":
                yield _sse_event("system", {"content": ev.get("content", "")})
            elif t == "action":
                yield _sse_event("action", ev.get("action", {}))
            elif t == "finding":
                yield _sse_event("finding", ev.get("finding", {}))
            elif t == "done":
                yield _sse_event("done", ev.get("usage") or {})
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    except Exception as e:
        err_msg = str(e)
        print(f"[nsys-ai] stream_agent_loop error (model={model!r}): {err_msg}", file=sys.stderr)
        try:
            yield _sse_event("text", {"chunk": f"Stream error: {err_msg}"})
            yield _sse_event("done", {"error": err_msg})
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _db_agent_flag_enabled() -> bool:
    """Return True when ``NSYS_AI_DB_AGENT`` env var is set to a truthy value."""
    val = os.environ.get("NSYS_AI_DB_AGENT", "").strip().lower()
    return bool(val) and val not in ("0", "false", "no", "off")


def _friendly_error(model: str, exc: Exception) -> str:
    """Convert a raw LiteLLM exception into a user-friendly message."""
    err = str(exc)
    print(f"[nsys-ai] LiteLLM error (model={model!r}): {err}", file=sys.stderr)
    if "429" in err or "RateLimitError" in type(exc).__name__ or "quota" in err.lower():
        return "Quota exceeded (429). Try a different model or check API billing."
    return err
