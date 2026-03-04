"""
Backend-only test: call chat_completion_stream twice in sequence (like first + second user message).
Use this to see if the second reply fails on the server side or only in the frontend.

Run from repo root with GEMINI_API_KEY set and Gemini Flash available:
  python -m pytest tests/test_chat_two_requests_backend.py -v -s
  or:  GEMINI_API_KEY=xxx python tests/test_chat_two_requests_backend.py
"""
import json
import os
import sys

# Minimal ui_context similar to what the frontend sends
MINIMAL_UI_CONTEXT = {
    "view_state": {"time_range": {"min_ns": 0, "max_ns": 1e9}, "scope": "All regions"},
    "selected_kernel": None,
    "stats": {"total_gpu_ms": 100.0, "kernel_count": 10, "nvtx_count": 5, "total_kernel_count": 10},
    "global_top_kernels": [],
    "visible_kernels_summary": [],
}


def _build_payload(messages: list, model: str = "gemini/gemini-2.5-flash") -> bytes:
    payload = {
        "messages": messages,
        "ui_context": MINIMAL_UI_CONTEXT,
        "stream": True,
        "model": model,
    }
    return json.dumps(payload).encode("utf-8")


def _consume_sse_stream(gen):
    """Consume generator of SSE bytes; return (full_text, got_done, error_from_done)."""
    buf = b""
    full_text = ""
    got_done = False
    error_from_done = None
    for chunk in gen:
        buf += chunk
        while b"\n\n" in buf:
            block, buf = buf.split(b"\n\n", 1)
            block = block.decode("utf-8", errors="replace")
            ev = ""
            dat = ""
            for line in block.split("\n"):
                if line.startswith("event:"):
                    ev = line[6:].strip()
                elif line.startswith("data:"):
                    dat = line[5:].strip()
            if ev == "text" and dat:
                try:
                    d = json.loads(dat)
                    full_text += d.get("chunk") or ""
                except json.JSONDecodeError:
                    pass
            elif ev == "done" and dat:
                got_done = True
                try:
                    d = json.loads(dat)
                    error_from_done = d.get("error")
                except json.JSONDecodeError:
                    pass
    return full_text.strip(), got_done, error_from_done


def run_two_requests():
    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY to run this test.", file=sys.stderr)
        return 2

    from nsys_ai import chat

    model = "gemini/gemini-2.5-flash"

    # Request 1: single user message
    messages_1 = [{"role": "user", "content": "Explain this kernel: nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN"}]
    body_1 = _build_payload(messages_1, model=model)
    print("--- Request 1 (first message) ---")
    gen_1 = chat.chat_completion_stream(body_1)
    text_1, done_1, err_1 = _consume_sse_stream(gen_1)
    if err_1:
        print(f"Request 1 error: {err_1}")
        return 1
    if not done_1:
        print("Request 1: stream ended without event:done")
        return 1
    print(f"Request 1: got_done=OK, response length={len(text_1)} chars")
    print(f"Request 1 reply (first 200 chars): {text_1[:200]}...")

    # Request 2: user, assistant (first reply), user (second question) — simulates "second turn"
    messages_2 = [
        {"role": "user", "content": "Explain this kernel: nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN"},
        {"role": "assistant", "content": text_1[:8000] if len(text_1) > 8000 else text_1},
        {"role": "user", "content": "Explain this kernel: ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL"},
    ]
    body_2 = _build_payload(messages_2, model=model)
    print("\n--- Request 2 (second message, with history) ---")
    gen_2 = chat.chat_completion_stream(body_2)
    text_2, done_2, err_2 = _consume_sse_stream(gen_2)
    if err_2:
        print(f"Request 2 error: {err_2}")
        return 1
    if not done_2:
        print("Request 2: stream ended without event:done")
        return 1
    print(f"Request 2: got_done=OK, response length={len(text_2)} chars")
    print(f"Request 2 reply (first 200 chars): {text_2[:200]}...")

    print("\n--- Both requests completed successfully from backend. ---")
    return 0


def test_two_requests_backend():
    """Pytest: run two stream requests in sequence. Skip if no GEMINI_API_KEY."""
    if not os.environ.get("GEMINI_API_KEY"):
        import pytest
        pytest.skip("GEMINI_API_KEY not set")
    assert run_two_requests() == 0


if __name__ == "__main__":
    sys.exit(run_two_requests())
