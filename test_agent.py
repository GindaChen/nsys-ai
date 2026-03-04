#!/usr/bin/env python3
"""
test_agent.py — CLI to run the Text-to-SQL agent loop against a profile DB (§11.5).

Usage:
  python test_agent.py [--model MODEL] <profile.sqlite|profile.nsys-rep> [question]
  python test_agent.py -m gemini/gemini-2.5-flash profile.sqlite "What is the first kernel?"
  echo "What is the first kernel?" | python test_agent.py <profile.sqlite>

Model selection:
  - CLI: --model / -m  (e.g. -m gemini/gemini-2.5-flash, -m anthropic/claude-sonnet-4-5-20250929)
  - Global: set NSYS_AI_MODEL in the environment (same model id as above)

Accepts .sqlite or .nsys-rep (resolved to .sqlite via nsys export when needed).
Requires: model (via -m/--model or NSYS_AI_MODEL) and one of GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY.
"""
import argparse
import sys
import threading


def _resolve_profile(path: str) -> str:
    """Return .sqlite path; convert .nsys-rep via resolve_profile_path when needed."""
    from nsys_ai.profile import resolve_profile_path
    return resolve_profile_path(path)


def _spinner(stop_event: threading.Event, message: str = "Thinking") -> None:
    """Run a terminal spinner on stderr until stop_event is set."""
    chars = "|/-\\"
    idx = 0
    while not stop_event.is_set():
        print(f"\r{message} {chars[idx % len(chars)]}  ", end="", file=sys.stderr, flush=True)
        idx += 1
        stop_event.wait(0.12)
    print("\r" + " " * (len(message) + 4) + "\r", end="", file=sys.stderr, flush=True)


def _main():
    parser = argparse.ArgumentParser(
        description="Run Text-to-SQL agent loop against a profile DB (§11.5).",
        epilog="Model: use --model/NSYS_AI_MODEL (e.g. gemini/gemini-2.5-flash). API key: GEMINI_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY.",
    )
    parser.add_argument(
        "-m", "--model",
        metavar="MODEL",
        default=None,
        help="Model id (e.g. gemini/gemini-2.5-flash). Overrides NSYS_AI_MODEL if set.",
    )
    parser.add_argument("profile", help="Path to profile (.sqlite or .nsys-rep)")
    parser.add_argument("question", nargs="?", default=None, help="Question to ask (or read from stdin)")
    args = parser.parse_args()

    profile_path = args.profile
    question = args.question
    if not question:
        question = sys.stdin.read().strip()
    if not question:
        print("No question provided (argument or stdin).", file=sys.stderr)
        sys.exit(1)

    try:
        from nsys_ai.chat import (
            _build_system_prompt,
            _get_model_and_key,
            _tools_openai,
            run_agent_loop,
        )
        from nsys_ai.tools_profile import (
            get_profile_schema,
            open_profile_readonly,
            query_profile_db,
        )
    except ImportError as e:
        print(f"Import error: {e}. Install: pip install -e '.[ai]'", file=sys.stderr)
        sys.exit(1)

    model, _ = _get_model_and_key(args.model)
    if not model:
        print("No LLM configured. Use --model MODEL or set NSYS_AI_MODEL and the corresponding API key.", file=sys.stderr)
        sys.exit(1)

    try:
        sqlite_path = _resolve_profile(profile_path)
    except RuntimeError as e:
        print(f"Profile resolve error: {e}", file=sys.stderr)
        sys.exit(1)
    conn = open_profile_readonly(sqlite_path)
    try:
        schema_str = get_profile_schema(conn)
        system_prompt = _build_system_prompt(ui_context={}, profile_schema=schema_str)
        api_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        def query_runner(sql_query: str):
            return query_profile_db(conn, sql_query)

        stop = threading.Event()
        spinner = threading.Thread(target=_spinner, args=(stop, "Thinking"), daemon=True)
        spinner.start()
        try:
            content, _actions = run_agent_loop(
                model=model,
                api_messages=api_messages,
                tools=_tools_openai(),
                query_runner=query_runner,
                max_turns=5,
            )
        finally:
            stop.set()
            spinner.join(timeout=0.5)
        print(content)
    finally:
        conn.close()


if __name__ == "__main__":
    _main()
