"""
test_trajectories.py — Trajectory-based regression tests for the nsys-ai agent.

Each trajectory JSON in tests/trajectories/**/*.json describes one benchmark
question with:
  - expected_tool_calls: SQL fragment requirements (which tables, ORDER BY, etc.)
  - expected_response.must_contain: key strings the final answer must include

Skip conditions (per-test):
  - Profile file missing (profile.path not found relative to repo root)
  - No API key available for any supported provider

Model selection (in priority order):
  1. NSYS_TRAJ_MODEL env var  — use this exact model
  2. NSYS_AI_MODEL env var    — use this exact model
  3. Auto-detect              — first provider with a key (Anthropic → OpenAI → Gemini)
  4. Trajectory JSON model    — fallback to the model used during generation

Usage:
    pytest tests/test_trajectories.py -v
    pytest tests/test_trajectories.py -v -k "q001"
    NSYS_TRAJ_MODEL=anthropic/claude-sonnet-4-6 pytest tests/test_trajectories.py -v
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return _REPO_ROOT


def _load_trajectories() -> list[dict]:
    """Return all trajectory dicts found under tests/trajectories/**/*.json.

    Only files whose JSON root object contains an ``"id"`` key are treated as
    trajectory files; other JSON files (e.g. schemas, READMEs) are skipped.
    """
    traj_dir = _repo_root() / "tests" / "trajectories"
    trajs = []
    for path in sorted(traj_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            pytest.fail(f"Failed to parse trajectory JSON {path}: {exc}")
        if not isinstance(data, dict) or "id" not in data:
            continue  # not a trajectory file
        data["_path"] = str(path)
        trajs.append(data)
    return trajs


def _model_to_key(model_id: str) -> str | None:
    if not model_id:
        return None
    if model_id.startswith("anthropic/"):
        return "ANTHROPIC_API_KEY"
    if model_id.startswith("openai/") or model_id.startswith("gpt-"):
        return "OPENAI_API_KEY"
    if model_id.startswith("gemini/"):
        return "GEMINI_API_KEY"
    return None


def _resolve_model(traj_model_id: str) -> str | None:
    """Resolve which model to actually use for this test run.

    Priority:
    1. NSYS_TRAJ_MODEL env var (trajectory-test-specific override)
    2. NSYS_AI_MODEL env var (global app override)
    3. Auto-detect: first provider with an API key (Anthropic → OpenAI → Gemini)
    4. Trajectory JSON model (fallback)

    Returns None if no provider has a key configured.
    """
    # 1 & 2: explicit overrides
    for env_var in ("NSYS_TRAJ_MODEL", "NSYS_AI_MODEL"):
        override = os.environ.get(env_var)
        if override:
            key = _model_to_key(override)
            if key and os.environ.get(key):
                return override

    # 3: auto-detect by available key
    _PROVIDER_DEFAULTS = [
        ("ANTHROPIC_API_KEY", "anthropic/claude-sonnet-4-6"),
        ("OPENAI_API_KEY",    "gpt-4o"),
        ("GEMINI_API_KEY",    "gemini/gemini-2.5-flash"),
    ]
    for key_name, default_model in _PROVIDER_DEFAULTS:
        if os.environ.get(key_name):
            return default_model

    # 4: fallback to trajectory JSON model (will skip if key missing)
    key = _model_to_key(traj_model_id)
    if key and os.environ.get(key):
        return traj_model_id

    return None


# ---------------------------------------------------------------------------
# Parametrize: one test per trajectory file
# ---------------------------------------------------------------------------

_ALL_TRAJECTORIES = _load_trajectories()

def _traj_id(traj: dict) -> str:
    return traj.get("id", Path(traj["_path"]).stem)


# When no trajectory files exist yet, skip rather than error.
_TRAJ_PARAMS = _ALL_TRAJECTORIES or [pytest.param(None, marks=pytest.mark.skip(reason="No trajectory files found"))]
_TRAJ_IDS    = [_traj_id(t) for t in _ALL_TRAJECTORIES] if _ALL_TRAJECTORIES else ["no-trajectories"]


@pytest.mark.parametrize("traj", _TRAJ_PARAMS, ids=_TRAJ_IDS)
def test_trajectory(traj: dict) -> None:
    """Run one trajectory: spy on SQL, verify fragments and must_contain."""
    # --- 1. Skip if profile is missing ---------------------------------
    profile_rel = traj.get("profile", {}).get("path", "")
    profile_path = _repo_root() / profile_rel
    if not profile_path.exists():
        pytest.skip(f"Profile not found: {profile_rel}")

    # --- 2. Resolve model (any available provider) ---------------------
    traj_model_id = traj.get("model", {}).get("id", "")
    model_id = _resolve_model(traj_model_id)
    if not model_id:
        pytest.skip("No API key configured (set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY)")

    # --- 3. Import agent components ------------------------------------
    try:
        from nsys_ai.chat import run_agent_loop
        from nsys_ai.chat_tools import _build_system_prompt
        from nsys_ai.tools_profile import (
            get_profile_schema_cached,
            open_profile_readonly,
            query_profile_db,
        )
    except ImportError as exc:
        pytest.skip(f"Agent dependencies not installed: {exc}")

    # --- 4. Set up spy on query_profile_db -----------------------------
    captured_sql: list[str] = []
    conn = open_profile_readonly(str(profile_path))

    def spy_query_runner(sql: str) -> str:
        captured_sql.append(sql)
        return query_profile_db(conn, sql)

    # --- 5. Build messages and run agent loop --------------------------
    schema = get_profile_schema_cached(conn, str(profile_path))
    ui_context: dict = {
        "profile_path": str(profile_path),
        "note": "DB agent enabled — use query_profile_db for whole-profile questions.",
    }
    system_prompt = _build_system_prompt(ui_context, profile_schema=schema)

    question = traj["entrypoint"]["args"]["question"]
    api_messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]

    try:
        answer, _actions = run_agent_loop(
            model=model_id,  # resolved model (may differ from trajectory JSON)
            api_messages=api_messages,
            query_runner=spy_query_runner,
            max_turns=8,
        )
    finally:
        conn.close()

    # --- 6. Assert SQL fragments ---------------------------------------
    all_sql_text = "\n".join(captured_sql).upper()
    for expected_call in traj.get("expected_tool_calls", []):
        if expected_call.get("name") != "query_profile_db":
            continue

        min_calls = expected_call.get("min_calls", 1)
        max_calls = expected_call.get("max_calls", 20)
        assert min_calls <= len(captured_sql) <= max_calls, (
            f"Expected {min_calls}–{max_calls} SQL calls, got {len(captured_sql)}"
        )

        # Support both "sql_fragments" (our convention) and
        # "sql_contains" (plan's original field name) for forward compatibility.
        # Only check fragments when SQL was actually called (min_calls=0 means SQL optional).
        fragments = expected_call.get("sql_fragments") or expected_call.get("sql_contains", [])
        if len(captured_sql) > 0:
            for fragment in fragments:
                assert fragment.upper() in all_sql_text, (
                    f"SQL fragment not found: {fragment!r}\n"
                    f"Actual SQL:\n{chr(10).join(captured_sql)}"
                )

    # --- 7. Assert must_contain / must_not_contain ---------------------
    expected_resp = traj.get("expected_response", {})
    for required in expected_resp.get("must_contain", []):
        assert required in answer, (
            f"Required string missing from answer: {required!r}\n"
            f"Answer:\n{answer}"
        )
    for forbidden in expected_resp.get("must_not_contain", []):
        assert forbidden not in answer, (
            f"Forbidden string found in answer: {forbidden!r}\n"
            f"Answer:\n{answer}"
        )
