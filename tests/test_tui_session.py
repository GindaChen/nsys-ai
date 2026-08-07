"""TUI session mode: CLI-published sessions open in tree/timeline; decisions hand off."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nsys_ai.runspec import RunSpec
from nsys_ai.session_cli import session_dir
from nsys_ai.timeline.app import NsysTimelineApp
from nsys_ai.tree.app import NsysTreeApp

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / "tests" / "fixtures" / "mfu_2gpu_before.sqlite"
AFTER = ROOT / "tests" / "fixtures" / "mfu_2gpu_after.sqlite"

# Minimal in-memory trees so headless tests skip SQLite NVTX load while still
# opening a real SessionStore via db_path (tree's _json_roots branch).
TREE_JSON = [
    {
        "name": "root",
        "type": "nvtx",
        "duration_ms": 1.0,
        "heat": 0.0,
        "stream": "0",
        "relative_pct": 100,
        "path": "root",
        "demangled": "",
        "start_ns": 0,
        "end_ns": 1_000_000,
        "children": [],
    }
]

TIMELINE_JSON = [
    {
        "name": "root",
        "type": "nvtx",
        "duration_ms": 1.0,
        "heat": 0.0,
        "stream": "0",
        "relative_pct": 100,
        "path": "root",
        "demangled": "",
        "start_ns": 0,
        "end_ns": 1_000_000,
        "children": [
            {
                "name": "k",
                "type": "kernel",
                "duration_ms": 0.5,
                "heat": 0.5,
                "stream": "0",
                "relative_pct": 50,
                "path": "root",
                "demangled": "",
                "start_ns": 0,
                "end_ns": 500_000,
                "children": [],
            }
        ],
    }
]


def _subprocess_environment(**extra: str) -> dict[str, str]:
    environment = dict(os.environ)
    source = str(ROOT / "src")
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not current else f"{source}{os.pathsep}{current}"
    environment.update(extra)
    return environment


def _run_cli(cwd: Path, *args: str, timeout: float = 180.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "nsys_ai", *args],
        cwd=cwd,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _finding_id_from_evidence_stdout(stdout: str) -> str:
    payload = json.loads(stdout)
    return next(
        finding["id"]
        for finding in payload["findings"]
        if finding.get("id") and finding.get("suggested_actions")
    )


def _publish_through_propose(tmp_path: Path, session_id: str) -> None:
    if not BEFORE.is_file():
        raise FileNotFoundError(f"missing fixture profile: {BEFORE}")
    before = BEFORE.resolve()
    runspec_path = tmp_path / "runspec.json"
    runspec_path.write_bytes(RunSpec(argv=("true",)).canonical_json_bytes())

    evidence = _run_cli(
        tmp_path,
        "evidence",
        "build",
        str(before),
        "--format",
        "json",
        "--gpu",
        "0",
        "--session",
        session_id,
        "--analyzers",
        "overlap_ratio",
    )
    assert evidence.returncode == 0, evidence.stderr
    finding_id = _finding_id_from_evidence_stdout(evidence.stdout)

    propose = _run_cli(
        tmp_path,
        "propose",
        "--session",
        session_id,
        "--finding-id",
        finding_id,
        "--runspec",
        str(runspec_path),
    )
    assert propose.returncode == 0, propose.stderr
    assert "Abstained:" not in propose.stdout


def _publish_full_session(tmp_path: Path, session_id: str) -> None:
    _publish_through_propose(tmp_path, session_id)
    before = BEFORE.resolve()
    after = AFTER.resolve()
    diff = _run_cli(
        tmp_path,
        "diff",
        str(before),
        str(after),
        "--gpu",
        "0",
        "--format",
        "json",
        "--no-ai",
        "--session",
        session_id,
    )
    assert diff.returncode == 0, diff.stderr


def _reload_decision_in_new_process(tmp_path: Path, session_id: str) -> dict:
    script = """
import json, sys
from nsys_ai.session_store import SessionStore
snapshot = SessionStore(sys.argv[1]).load(sys.argv[2])
decision = snapshot.diff.get("decision") if snapshot.diff else None
print(json.dumps({
    "phase": snapshot.state.phase,
    "status": None if decision is None else decision.get("status"),
    "reason": None if decision is None else decision.get("reason"),
}))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path / ".nsys-ai" / "sessions"),
            session_id,
        ],
        cwd=tmp_path,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.asyncio
async def test_cli_session_opens_in_tree_tui(tmp_path: Path, monkeypatch):
    """Acceptance: a CLI-created session opens in the tree TUI (headless Textual)."""
    if not BEFORE.is_file():
        raise FileNotFoundError(f"missing fixture profile: {BEFORE}")
    session_id = "tui-tree-open"
    _publish_through_propose(tmp_path, session_id)
    monkeypatch.chdir(tmp_path)

    directory = session_dir(session_id, root=tmp_path / ".nsys-ai" / "sessions")
    assert (directory / "findings.json").is_file()
    assert (directory / "proposal.json").is_file()

    app = NsysTreeApp(
        db_path=str(BEFORE.resolve()),
        device=0,
        trim=None,
        json_roots=TREE_JSON,
        session=session_id,
    )
    assert app._session_mode() is True
    assert app._session_id == session_id
    assert app.analysis_phase == "propose"
    assert app._session_projection is not None
    assert app._session_projection["session_mode"] is True
    assert app._session_projection["diagnose_ran"] is True
    assert isinstance(app._session_projection["proposal"], dict)

    async with app.run_test(size=(120, 40)) as pilot:
        notes: list[tuple] = []
        app.notify = lambda *a, **k: notes.append((a, k))  # type: ignore[method-assign]
        app.action_loop_diagnose()
        await pilot.pause()
        assert notes and any("evidence" in str(a).lower() for a, _ in notes)
        assert app.analysis_phase == "propose"


@pytest.mark.asyncio
async def test_tree_tui_decision_visible_to_new_cli_process(tmp_path: Path, monkeypatch):
    """Acceptance: tree TUI decision survives a real second process."""
    if not BEFORE.is_file() or not AFTER.is_file():
        raise FileNotFoundError(f"missing fixture profiles: {BEFORE} / {AFTER}")
    session_id = "tui-tree-decide"
    _publish_full_session(tmp_path, session_id)
    monkeypatch.chdir(tmp_path)

    app = NsysTreeApp(
        db_path=str(BEFORE.resolve()),
        device=0,
        trim=None,
        json_roots=TREE_JSON,
        session=session_id,
    )
    assert app.analysis_phase == "diff"
    assert app._session_projection is not None
    assert app._session_projection["diff_summary"]
    assert app._session_projection["decision"] is None

    async with app.run_test(size=(120, 40)) as pilot:
        app.set_loop_decision("accept", "tree TUI verified improvement")
        await pilot.pause()
        assert app.analysis_phase == "accept"
        assert app._session_projection is not None
        assert app._session_projection["decision"] == "accept"
        assert app._session_projection["decision_path"].endswith("diff.json")

    reloaded = _reload_decision_in_new_process(tmp_path, session_id)
    assert reloaded["phase"] == "accept"
    assert reloaded["status"] == "accepted"
    assert reloaded["reason"] == "tree TUI verified improvement"


@pytest.mark.asyncio
async def test_cli_session_opens_in_timeline_tui(tmp_path: Path, monkeypatch):
    """Acceptance: a CLI-created session opens in the timeline TUI (headless Textual)."""
    if not BEFORE.is_file():
        raise FileNotFoundError(f"missing fixture profile: {BEFORE}")
    session_id = "tui-timeline-open"
    _publish_through_propose(tmp_path, session_id)
    monkeypatch.chdir(tmp_path)

    app = NsysTimelineApp(
        db_path=str(BEFORE.resolve()),
        device=0,
        trim=None,
        json_roots=TIMELINE_JSON,
        session=session_id,
    )
    assert app._session_mode() is True
    assert app._session_id == session_id
    assert app.analysis_phase == "propose"
    assert app._session_projection is not None
    assert app._session_projection["diagnose_ran"] is True

    async with app.run_test(size=(120, 40)) as pilot:
        notes: list[tuple] = []
        app.notify = lambda *a, **k: notes.append((a, k))  # type: ignore[method-assign]
        app.action_loop_propose()
        await pilot.pause()
        assert notes and any("propose" in str(a).lower() for a, _ in notes)
        assert app.analysis_phase == "propose"


@pytest.mark.asyncio
async def test_timeline_tui_decision_visible_to_new_cli_process(tmp_path: Path, monkeypatch):
    """Acceptance: timeline TUI decision survives a real second process."""
    if not BEFORE.is_file() or not AFTER.is_file():
        raise FileNotFoundError(f"missing fixture profiles: {BEFORE} / {AFTER}")
    session_id = "tui-timeline-decide"
    _publish_full_session(tmp_path, session_id)
    monkeypatch.chdir(tmp_path)

    app = NsysTimelineApp(
        db_path=str(BEFORE.resolve()),
        device=0,
        trim=None,
        json_roots=TIMELINE_JSON,
        session=session_id,
    )
    assert app.analysis_phase == "diff"

    async with app.run_test(size=(120, 40)) as pilot:
        app.set_loop_decision("reject", "timeline TUI rejected regression")
        await pilot.pause()
        assert app.analysis_phase == "accept"
        assert app._session_projection is not None
        assert app._session_projection["decision"] == "reject"

    reloaded = _reload_decision_in_new_process(tmp_path, session_id)
    assert reloaded["phase"] == "accept"
    assert reloaded["status"] == "rejected"
    assert reloaded["reason"] == "timeline TUI rejected regression"
