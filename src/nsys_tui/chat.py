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
from .chat_tools import (  # noqa: F401
    _build_system_prompt,
    _parse_tool_call,
    _tools_openai,
)
from .tools_profile import (
    get_profile_schema_cached,
    open_profile_readonly,
    query_profile_db,
)

_log = logging.getLogger(__name__)
_telemetry_log = logging.getLogger("nsys_tui.telemetry")

# ---------------------------------------------------------------------------
# Agent-loop constants
# ---------------------------------------------------------------------------

# Cap total messages sent to the LLM per request; keeps token budget bounded.
MAX_AGENT_MESSAGES = 100
# Warn when prompt tokens exceed this threshold.
PROMPT_TOKEN_WARNING_THRESHOLD = 30_000
# Consecutive DB errors before injecting a break-cycle hint.
MAX_CONSECUTIVE_DB_ERRORS = 2


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
        i for i, m in enumerate(api_messages)
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    if len(tool_turn_indices) < 2:
        return
    cutoff = tool_turn_indices[-1]
    for m in api_messages[:cutoff]:
        if m.get("role") == "tool" and len(m.get("content", "")) > 200:
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
                (tc.get("function") or {}).get("name", "unknown")
                for tc in m["tool_calls"]
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
) -> tuple[str, list]:
    """Run a multi-turn agent loop until the model stops calling tools.

    Args:
        model:         LiteLLM model identifier.
        api_messages:  Mutable message list (modified in-place as turns progress).
        tools:         OpenAI-style tool spec list; defaults to :func:`_tools_openai`.
        query_runner:  Callable ``(sql: str) -> str`` for ``query_profile_db``.
                       Pass ``None`` to treat DB calls as no-ops.
        max_turns:     Maximum number of LLM round-trips.

    Returns:
        ``(final_content, actions)`` — *actions* are parsed navigation/zoom dicts.
    """
    try:
        import litellm
    except ImportError:
        return ("LLM not available (install litellm).", [])

    tools = tools if tools is not None else _tools_openai()
    actions: list = []
    consecutive_db_errors = 0

    for _ in range(max_turns):
        _compact_old_tool_results(api_messages)
        if len(api_messages) > MAX_AGENT_MESSAGES:
            api_messages[:] = [api_messages[0]] + api_messages[-(MAX_AGENT_MESSAGES - 1):]
        response = litellm.completion(
            model=model,
            messages=api_messages,
            tools=tools,
            tool_choice="auto",
        )
        choice = response.choices[0] if response.choices else None
        if not choice:
            return ("", actions)
        message = choice.message
        if isinstance(message, dict):
            content = (message.get("content") or "").strip()
            tool_calls = message.get("tool_calls") or []
        else:
            content = (getattr(message, "content", None) or "").strip()
            tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            return (content, actions)

        tc_list = []
        for tc in tool_calls:
            fn = getattr(tc, "function", None) if not isinstance(tc, dict) else tc.get("function") or {}
            tc_id = getattr(tc, "id", None) if not isinstance(tc, dict) else tc.get("id")
            name = getattr(fn, "name", None) if not isinstance(fn, dict) else fn.get("name")
            args_str = (getattr(fn, "arguments", None) if not isinstance(fn, dict) else fn.get("arguments")) or "{}"
            tc_list.append((tc_id, name, args_str))

        api_messages.append({
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {"id": tid, "type": "function", "function": {"name": n, "arguments": a}}
                for tid, n, a in tc_list
            ],
        })

        has_external = False
        for tc_id, name, args_str in tc_list:
            if not name or not tc_id:
                continue
            action = _parse_tool_call(name, args_str)
            if action:
                has_external = True
                actions.append(action)
            elif name == "query_profile_db":
                if query_runner is not None:
                    try:
                        sql = json.loads(args_str).get("sql_query", "")
                        result = query_runner(sql)
                    except Exception as e:
                        result = f"Error: {e}"
                    if result.startswith("Error:"):
                        consecutive_db_errors += 1
                        if consecutive_db_errors >= MAX_CONSECUTIVE_DB_ERRORS:
                            result += (
                                "\n[System: Repeated SQL errors. "
                                "Please answer from available context without further queries.]"
                            )
                    else:
                        consecutive_db_errors = 0
                else:
                    result = "Not executed (no profile loaded)."
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": name,
                    "content": result,
                })
            else:
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": name,
                    "content": "Not executed.",
                })

        if has_external:
            return (content, actions)

    return ("Max turns reached.", actions)


# ---------------------------------------------------------------------------
# Web-API handler (non-streaming)
# ---------------------------------------------------------------------------

def chat_completion(body_bytes: bytes) -> dict | None:
    """Handle a POST ``/api/chat`` request body.

    Returns ``{"content": str, "actions": list}`` or ``None`` for 501
    (LLM not configured / not installed).
    """
    try:
        payload = json.loads(body_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"content": "Invalid request body.", "actions": []}

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

    # Feature flag: NSYS_AI_DB_AGENT enables the DB-backed agent for the web path.
    db_agent_enabled = _db_agent_flag_enabled()

    if profile_path and db_agent_enabled:
        try:
            from .profile import resolve_profile_path
            sqlite_path = resolve_profile_path(profile_path)
        except RuntimeError as e:
            return {"content": f"Profile error: {e}", "actions": []}
        conn = open_profile_readonly(sqlite_path)
        try:
            schema_str = get_profile_schema_cached(conn, sqlite_path)
            system_prompt = _build_system_prompt(ui_context, profile_schema=schema_str)
            api_messages = [{"role": "system", "content": system_prompt}]
            for m in messages:
                if m.get("role") and m.get("content") is not None:
                    api_messages.append({"role": m["role"], "content": m["content"]})
            query_runner = lambda sql: query_profile_db(conn, sql)  # noqa: E731
            try:
                content, actions = run_agent_loop(
                    model=model,
                    api_messages=api_messages,
                    tools=_tools_openai(),
                    query_runner=query_runner,
                    max_turns=5,
                )
                return {"content": content, "actions": actions}
            except Exception as e:
                return {"content": f"LLM error: {_friendly_error(model, e)}", "actions": []}
        finally:
            conn.close()

    system_prompt = _build_system_prompt(ui_context)
    api_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        if m.get("role") and m.get("content") is not None:
            api_messages.append({"role": m["role"], "content": m["content"]})

    try:
        response = litellm.completion(
            model=model,
            messages=api_messages,
            tools=_tools_openai(),
            tool_choice="auto",
        )
    except Exception as e:
        return {"content": f"LLM error: {_friendly_error(model, e)}", "actions": []}

    choice = response.choices[0] if response.choices else None
    if not choice:
        return {"content": "", "actions": []}
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
        args_str = (getattr(fn, "arguments", None) if not isinstance(fn, dict) else fn.get("arguments")) or "{}"
        if name:
            action = _parse_tool_call(name, args_str)
            if action:
                actions.append(action)
    return {"content": content, "actions": actions}


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse_event(evt: str, data: dict) -> bytes:
    return f"event: {evt}\ndata: {json.dumps(data)}\n\n".encode()


# ---------------------------------------------------------------------------
# Streaming agent loop — UI-agnostic generator (used by tui_textual + web)
# ---------------------------------------------------------------------------

def stream_agent_loop(
    model: str,
    messages: list,
    ui_context: dict,
    tools: list | None = None,
    profile_path: str | None = None,
    max_turns: int = 5,
):
    """UI-agnostic streaming agent loop — yields event dicts.

    Yielded event types:

    * ``{"type": "text",   "content": str}``   — streamed text fragment
    * ``{"type": "system", "content": str}``   — status / warning message
    * ``{"type": "action", "action": dict}``   — navigation/zoom action
    * ``{"type": "done",   "usage": dict}``    — final event with token usage

    The profile connection (when *profile_path* is given) is opened in this
    generator and closed in the ``finally`` block.  Call this from a background
    thread (e.g. Textual ``@work(thread=True)``) so the main thread's UI
    remains responsive during DB queries and LLM streaming.
    """
    try:
        import litellm
    except ImportError:
        yield {"type": "text", "content": "LLM not available (install litellm)."}
        yield {"type": "done", "usage": {}}
        return

    tools = tools if tools is not None else _tools_openai()
    conn = None
    query_runner = None

    if profile_path:
        try:
            from .profile import resolve_profile_path
            sqlite_path = resolve_profile_path(profile_path)
        except RuntimeError as e:
            yield {"type": "text", "content": f"Profile error: {e}"}
            yield {"type": "done", "usage": {}}
            return
        conn = open_profile_readonly(sqlite_path)
        try:
            schema_str = get_profile_schema_cached(conn, sqlite_path)
            query_runner = lambda sql: query_profile_db(conn, sql)  # noqa: E731
        except Exception:
            if conn:
                conn.close()
            raise
    else:
        schema_str = None

    system_prompt = _build_system_prompt(ui_context, profile_schema=schema_str)
    api_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        if m.get("role") and m.get("content") is not None:
            api_messages.append({"role": m["role"], "content": m["content"]})

    usage: dict = {}
    consecutive_db_errors = 0
    turn_count = 0

    try:
        for _ in range(max_turns):
            turn_count += 1
            _compact_old_tool_results(api_messages)
            if len(api_messages) > MAX_AGENT_MESSAGES:
                api_messages[:] = [api_messages[0]] + api_messages[-(MAX_AGENT_MESSAGES - 1):]

            try:
                stream = litellm.completion(
                    model=model,
                    messages=api_messages,
                    tools=tools,
                    tool_choice="auto",
                    stream=True,
                )
            except Exception as e:
                yield {"type": "text", "content": f"LLM error: {_friendly_error(model, e)}"}
                yield {"type": "done", "usage": usage}
                return

            content_parts: list[str] = []
            tool_calls_by_index: dict[int, dict] = {}

            for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue
                delta = (
                    getattr(choice, "delta", None)
                    or (choice.get("delta") if isinstance(choice, dict) else None)
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
                    yield {"type": "text", "content": c}

                tcs = (
                    getattr(delta, "tool_calls", None)
                    if not isinstance(delta, dict)
                    else delta.get("tool_calls")
                ) or []
                for tc in tcs:
                    idx = getattr(tc, "index", 0) if not isinstance(tc, dict) else tc.get("index", 0)
                    tc_id = getattr(tc, "id", None) if not isinstance(tc, dict) else tc.get("id")
                    fn = (
                        getattr(tc, "function", None)
                        if not isinstance(tc, dict)
                        else tc.get("function") or {}
                    )
                    if isinstance(fn, dict):
                        name, args = fn.get("name"), fn.get("arguments") or ""
                    else:
                        name, args = getattr(fn, "name", None), getattr(fn, "arguments", None) or ""
                    entry = tool_calls_by_index.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                    if tc_id:
                        entry["id"] = tc_id
                    if name:
                        entry["name"] = name
                    entry["arguments"] += args

                u = (
                    getattr(chunk, "usage", None)
                    or (chunk.get("usage") if isinstance(chunk, dict) else None)
                )
                if u:
                    usage = u if isinstance(u, dict) else {
                        "prompt_tokens": getattr(u, "prompt_tokens", 0),
                        "completion_tokens": getattr(u, "completion_tokens", 0),
                    }

            full_content = "".join(content_parts).strip() if content_parts else ""
            tc_list = [
                (t.get("id") or f"call_{idx}", t.get("name"), t.get("arguments") or "{}")
                for idx, t in sorted(tool_calls_by_index.items())
            ]

            if usage:
                pt = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else getattr(usage, "prompt_tokens", 0)
                if isinstance(pt, int) and pt > PROMPT_TOKEN_WARNING_THRESHOLD:
                    _log.warning("stream_agent_loop: high prompt token usage (%d). model=%s", pt, model)
                    yield {
                        "type": "system",
                        "content": f"⚠ Large context ({pt:,} tokens). Consider starting a new chat to reduce cost.",
                    }

            if not tc_list:
                yield {"type": "done", "usage": usage}
                return

            api_messages.append({
                "role": "assistant",
                "content": full_content or None,
                "tool_calls": [
                    {"id": tid, "type": "function", "function": {"name": n, "arguments": a}}
                    for tid, n, a in tc_list
                ],
            })

            has_external = False
            for tid, name, args_str in tc_list:
                if not name or not tid:
                    continue
                if name == "query_profile_db" and query_runner is not None:
                    yield {"type": "system", "content": "Running DB query..."}
                    try:
                        sql = json.loads(args_str).get("sql_query", "")
                        result = query_runner(sql)
                    except Exception as e:
                        result = f"Error: {e}"
                    if result.startswith("Error:"):
                        consecutive_db_errors += 1
                        if consecutive_db_errors >= MAX_CONSECUTIVE_DB_ERRORS:
                            result += (
                                "\n[System: Repeated SQL errors. "
                                "Please answer from available context without further queries.]"
                            )
                    else:
                        consecutive_db_errors = 0
                    api_messages.append({
                        "role": "tool",
                        "tool_call_id": tid,
                        "name": name,
                        "content": result,
                    })
                else:
                    if name == "query_profile_db":
                        api_messages.append({
                            "role": "tool",
                            "tool_call_id": tid,
                            "name": name,
                            "content": "Not executed (no profile loaded).",
                        })
                    action = _parse_tool_call(name, args_str)
                    if action:
                        has_external = True
                        yield {"type": "action", "action": action}

            if has_external:
                yield {"type": "done", "usage": usage}
                return

        yield {"type": "done", "usage": usage}

    finally:
        if usage:
            _telemetry_log.info(
                "agent_usage model=%s prompt_tokens=%s completion_tokens=%s turns=%d",
                model,
                usage.get("prompt_tokens", "?") if isinstance(usage, dict) else getattr(usage, "prompt_tokens", "?"),
                usage.get("completion_tokens", "?") if isinstance(usage, dict) else getattr(usage, "completion_tokens", "?"),
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

    Always delegates to :func:`stream_agent_loop`.  Uses *profile_path* when
    ``NSYS_AI_DB_AGENT`` environment variable is set.
    """
    try:
        payload = json.loads(body_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        yield _sse_event("text", {"chunk": "Invalid request body."})
        yield _sse_event("done", {})
        return

    model, _ = _get_model_and_key(payload.get("model"))
    if not model:
        yield _sse_event("done", {"error": "LLM not configured"})
        return

    messages = payload.get("messages") or []
    ui_context = payload.get("ui_context") or {}
    profile_path = payload.get("profile_path")
    effective_profile = profile_path if (profile_path and _db_agent_flag_enabled()) else None

    try:
        for ev in stream_agent_loop(
            model=model,
            messages=messages,
            ui_context=ui_context,
            tools=_tools_openai(),
            profile_path=effective_profile,
            max_turns=5,
        ):
            t = ev.get("type")
            if t == "text":
                yield _sse_event("text", {"chunk": ev.get("content", "")})
            elif t == "system":
                yield _sse_event("system", {"content": ev.get("content", "")})
            elif t == "action":
                yield _sse_event("action", ev.get("action", {}))
            elif t == "done":
                yield _sse_event("done", ev.get("usage") or {})
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    except Exception as e:
        err_msg = str(e)
        print(f"[nsys-tui] stream_agent_loop error (model={model!r}): {err_msg}", file=sys.stderr)
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
    print(f"[nsys-tui] LiteLLM error (model={model!r}): {err}", file=sys.stderr)
    if "429" in err or "RateLimitError" in type(exc).__name__ or "quota" in err.lower():
        return "Quota exceeded (429). Try a different model or check API billing."
    return err
