"""Tests for invocation-owned artifact placement (#430)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from nsys_ai.artifact_root import (
    ARTIFACT_ROOT_ENV_VAR,
    artifact_root,
    default_decision_path,
    profile_root,
    session_root,
)
from nsys_ai.profile_command import default_output_leaf
from nsys_ai.session_cli import resolve_session_location
from nsys_ai.session_store import SessionStore


def test_default_layout_remains_under_dot_nsys_ai(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(ARTIFACT_ROOT_ENV_VAR, raising=False)

    assert artifact_root() == tmp_path / ".nsys-ai"
    assert session_root() == tmp_path / ".nsys-ai" / "sessions"
    assert profile_root() == tmp_path / ".nsys-ai" / "profiles"
    assert default_decision_path(cwd=tmp_path) == tmp_path / "diff.json"
    assert default_output_leaf(tmp_path).parent == tmp_path / ".nsys-ai" / "profiles"


def test_env_root_relocates_session_profile_and_decision_outputs(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    configured = tmp_path / "ci-artifacts"
    monkeypatch.setenv(ARTIFACT_ROOT_ENV_VAR, str(configured))

    assert artifact_root() == configured
    assert session_root() == configured / "sessions"
    assert profile_root() == configured / "profiles"
    assert default_decision_path() == configured / "decisions" / "diff.json"
    assert default_output_leaf(tmp_path).parent == configured / "profiles"

    location = resolve_session_location("run-001")
    assert location is not None
    assert location.root == configured / "sessions"

    # The public default SessionStore must use the same resolver as the CLI
    # facade; this is the guard against in-process surfaces silently diverging.
    store = SessionStore()
    store.create("run-001")
    assert (configured / "sessions" / "run-001" / "session.json").is_file()
    assert not (tmp_path / ".nsys-ai").exists()


def test_explicit_session_directory_wins_over_artifact_root(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv(ARTIFACT_ROOT_ENV_VAR, str(tmp_path / "configured"))
    explicit = tmp_path / "handoff" / "run-002"

    location = resolve_session_location(explicit)

    assert location is not None
    assert location.explicit is True
    assert location.directory == explicit
    assert location.root == explicit.parent


def test_diagnose_cli_keeps_working_directory_clean_with_artifact_root(tmp_path, monkeypatch):
    artifact_root = tmp_path / "ci-artifacts"
    profile = Path(__file__).parent / "fixtures" / "mock.sqlite"
    env = os.environ.copy()
    env[ARTIFACT_ROOT_ENV_VAR] = str(artifact_root)
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nsys_ai",
            "diagnose",
            str(profile),
            "--session",
            "run-003",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert (artifact_root / "sessions" / "run-003" / "findings.json").is_file()
    assert not (tmp_path / ".nsys-ai").exists()
