"""
Backend integration tests for AI chat using Gemini 2.5 Flash.
Run with: GEMINI_API_KEY=<your-key> pytest tests/test_chat_gemini.py -v
Skips all tests if GEMINI_API_KEY is not set.
"""
import json
import os

import pytest

from nsys_ai import chat as chat_mod

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_25_FLASH = "gemini/gemini-2.5-flash"


@pytest.mark.skipif(not GEMINI_KEY, reason="GEMINI_API_KEY not set")
def test_chat_completion_gemini_25_flash():
    """Stage 1: Non-stream chat_completion with Gemini 2.5 Flash returns content."""
    payload = {
        "model": MODEL_25_FLASH,
        "messages": [{"role": "user", "content": "Reply with exactly the word OK and nothing else."}],
        "ui_context": {"view_state": {}, "stats": {}, "global_top_kernels": [], "visible_kernels_summary": []},
    }
    body = json.dumps(payload).encode("utf-8")
    out = chat_mod.chat_completion(body)
    assert out is not None, "chat_completion should not return None when GEMINI_API_KEY is set"
    assert "content" in out
    assert isinstance(out["content"], str)
    assert len(out["content"].strip()) > 0
    assert "actions" in out
    assert isinstance(out["actions"], list)


@pytest.mark.skipif(not GEMINI_KEY, reason="GEMINI_API_KEY not set")
def test_chat_completion_stream_gemini_25_flash():
    """Stage 2: Stream chat_completion_stream with Gemini 2.5 Flash yields text then done."""
    payload = {
        "model": MODEL_25_FLASH,
        "messages": [{"role": "user", "content": "Say hello in one short word."}],
        "ui_context": {"view_state": {}, "stats": {}, "global_top_kernels": [], "visible_kernels_summary": []},
        "stream": True,
    }
    body = json.dumps(payload).encode("utf-8")
    gen = chat_mod.chat_completion_stream(body)
    events = []
    for chunk in gen:
        assert isinstance(chunk, bytes)
        s = chunk.decode("utf-8")
        if s.startswith("event:"):
            ev = s.split("\n")[0].replace("event:", "").strip()
            events.append(ev)
    assert "text" in events, "Expected at least one event: text"
    assert "done" in events, "Expected event: done"


@pytest.mark.skipif(not GEMINI_KEY, reason="GEMINI_API_KEY not set")
def test_get_model_and_key_preferred_25_flash():
    """Stage 3: _get_model_and_key with preferred_model=2.5-flash returns that model and key."""
    model, key = chat_mod._get_model_and_key(MODEL_25_FLASH)
    assert model == MODEL_25_FLASH
    assert key == GEMINI_KEY


@pytest.mark.skipif(not GEMINI_KEY, reason="GEMINI_API_KEY not set")
def test_get_available_models_includes_gemini():
    """Stage 4: get_available_models() includes Gemini 2.5 Flash when key is set."""
    opts = chat_mod.get_available_models()
    ids = [o["id"] for o in opts]
    assert MODEL_25_FLASH in ids
