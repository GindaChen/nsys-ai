"""End-to-end coverage for the diagnose and review front doors."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from nsys_ai import web
from nsys_ai.diagnose_command import run_diagnose
from nsys_ai.session_cli import session_dir
from nsys_ai.session_store import SessionStore

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / "tests" / "fixtures" / "mfu_2gpu_before.sqlite"
AFTER = ROOT / "tests" / "fixtures" / "mfu_2gpu_after.sqlite"
SINGLE = ROOT / "tests" / "fixtures" / "h100_2gpu_1s.sqlite"


def _subprocess_environment(**extra: str) -> dict[str, str]:
    environment = dict(os.environ)
    source = str(ROOT / "src")
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not current else f"{source}{os.pathsep}{current}"
    environment.update(extra)
    return environment


def _run_cli(
    cwd: Path, *args: str, timeout: float = 180.0, **env: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "nsys_ai", *args],
        cwd=cwd,
        env=_subprocess_environment(**env),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_diagnose_then_review_pair_both_succeed(tmp_path: Path):
    """diagnose then review <before> <after> must compose — the previous shape failed here."""
    if not BEFORE.is_file() or not AFTER.is_file():
        raise FileNotFoundError(f"missing fixture profiles: {BEFORE} / {AFTER}")
    before = BEFORE.resolve()
    after = AFTER.resolve()
    session_id = "compose-e2e"

    diagnose = _run_cli(
        tmp_path,
        "diagnose",
        str(before),
        "--gpu",
        "0",
        "--session",
        session_id,
    )
    assert diagnose.returncode == 0, diagnose.stderr
    assert f"Session: {session_id}" in diagnose.stdout
    assert "Evidence Findings" in diagnose.stdout
    assert (tmp_path / ".nsys-ai" / "sessions" / session_id / "findings.json").is_file()

    review = _run_cli(tmp_path, "review", str(before), str(after))
    assert review.returncode == 0, review.stderr
    assert "Profile Diff" in review.stdout
    assert "Next:" in review.stderr
    # Pair form must not invent or mutate a session under .nsys-ai/
    sessions_root = tmp_path / ".nsys-ai" / "sessions"
    assert {path.name for path in sessions_root.iterdir()} == {session_id}
    session_payload = json.loads(
        (sessions_root / session_id / "session.json").read_text(encoding="utf-8")
    )
    assert session_payload["phase"] == "diagnose"
    assert "mode" not in session_payload
    assert not (sessions_root / session_id / "diff.json").exists()


def test_review_pair_without_session_does_not_create_nsys_ai(tmp_path: Path):
    if not BEFORE.is_file() or not AFTER.is_file():
        raise FileNotFoundError(f"missing fixture profiles: {BEFORE} / {AFTER}")
    before = BEFORE.resolve()
    after = AFTER.resolve()

    result = _run_cli(tmp_path, "review", str(before), str(after))
    assert result.returncode == 0, result.stderr
    assert "Next:" in result.stderr
    assert not (tmp_path / ".nsys-ai").exists()


def test_review_pair_stdout_is_byte_identical_to_diff_no_ai(tmp_path: Path):
    """review <before> <after> IS the canonical diff, not a smaller rendering of it.

    review's default (--gpu unset) must take the all-GPU shape diff takes:
    executive summary, per-GPU overview and per-GPU regressions, with the GPU
    label rendered as "all" rather than leaking the Python None.
    """
    if not BEFORE.is_file() or not AFTER.is_file():
        raise FileNotFoundError(f"missing fixture profiles: {BEFORE} / {AFTER}")
    before = BEFORE.resolve()
    after = AFTER.resolve()

    canonical = _run_cli(tmp_path, "diff", str(before), str(after), "--no-ai")
    review = _run_cli(tmp_path, "review", str(before), str(after))
    assert canonical.returncode == 0, canonical.stderr
    assert review.returncode == 0, review.stderr

    # Named sections first, so a regression reports which shape was lost.
    for section in (
        "Profile Diff (All GPUs)",
        "Executive Summary",
        "Per-GPU Overview",
        "GPU:    all",
    ):
        assert section in review.stdout, f"review dropped section: {section}"
    assert "GPU:    None" not in review.stdout
    assert review.stdout == canonical.stdout

    # Hints are an addition on stderr, never a substitution inside the report.
    assert "Next: nsys-ai diagnose" in review.stderr
    assert "Next:" not in review.stdout


def test_review_gpu_scoped_stdout_is_byte_identical_to_diff_no_ai(tmp_path: Path):
    """The --gpu path must stay on the same renderer as diff --gpu --no-ai."""
    if not BEFORE.is_file() or not AFTER.is_file():
        raise FileNotFoundError(f"missing fixture profiles: {BEFORE} / {AFTER}")
    before = BEFORE.resolve()
    after = AFTER.resolve()

    canonical = _run_cli(tmp_path, "diff", str(before), str(after), "--no-ai", "--gpu", "1")
    review = _run_cli(tmp_path, "review", str(before), str(after), "--gpu", "1")
    assert canonical.returncode == 0, canonical.stderr
    assert review.returncode == 0, review.stderr
    assert "Executive Summary" in review.stdout
    assert review.stdout == canonical.stdout


def test_review_session_resumes_diagnose_session_decision_path(tmp_path: Path):
    profile = SINGLE if SINGLE.is_file() else BEFORE
    if not profile.is_file():
        raise FileNotFoundError(f"missing fixture profile: {profile}")
    profile = profile.resolve()
    session_id = "resume-e2e"

    diagnose = _run_cli(
        tmp_path,
        "diagnose",
        str(profile),
        "--gpu",
        "0",
        "--session",
        session_id,
    )
    assert diagnose.returncode == 0, diagnose.stderr

    resume = _run_cli(tmp_path, "review", "--session", session_id)
    assert resume.returncode == 0, resume.stderr
    assert f"Session: {session_id}" in resume.stdout
    assert "Decision path:" in resume.stdout or "Phase:" in resume.stdout

    directory = session_dir(session_id, root=tmp_path / ".nsys-ai" / "sessions")
    store = SessionStore(tmp_path / ".nsys-ai" / "sessions")
    snapshot = store.load(session_id)
    assert snapshot.state.phase == "diagnose"
    assert snapshot.findings is not None
    assert snapshot.diff is None
    assert (directory / "findings.json").is_file()


def test_diagnose_publishes_findings_and_survives_a_second_process(tmp_path: Path):
    profile = SINGLE if SINGLE.is_file() else BEFORE
    if not profile.is_file():
        raise FileNotFoundError(f"missing fixture profile: {profile}")
    profile = profile.resolve()
    session_id = "diagnose-e2e"

    result = _run_cli(
        tmp_path,
        "diagnose",
        str(profile),
        "--gpu",
        "0",
        "--session",
        session_id,
    )
    assert result.returncode == 0, result.stderr
    assert f"Session: {session_id}" in result.stdout
    assert "Limitations / skipped analyses" in result.stdout
    assert "Next:" in result.stdout

    directory = session_dir(session_id, root=tmp_path / ".nsys-ai" / "sessions")
    assert (directory / "findings.json").is_file()
    session_payload = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    assert "mode" not in session_payload

    script = """
import json, sys
from nsys_ai.session_store import SessionStore
snapshot = SessionStore(sys.argv[1]).load(sys.argv[2])
print(json.dumps({
    "phase": snapshot.state.phase,
    "findings_count": len(snapshot.findings.findings),
    "skipped_count": len(snapshot.findings.skipped),
    "before_id": snapshot.state.before_profile.profile_id,
}))
"""
    reloaded = subprocess.run(
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
    assert reloaded.returncode == 0, reloaded.stderr
    payload = json.loads(reloaded.stdout)
    assert payload["phase"] == "diagnose"
    assert payload["findings_count"] >= 1
    assert payload["before_id"]


def test_diagnose_web_serves_the_caller_session_root(tmp_path: Path, monkeypatch):
    """--web must open the session root diagnose published into, not cwd's default.

    serve_timeline is driven for real; only the blocking serve step is replaced,
    so the handler state under test is the state a browser would have seen.
    """
    profile_source = SINGLE if SINGLE.is_file() else BEFORE
    if not profile_source.is_file():
        raise FileNotFoundError(f"missing fixture profile: {profile_source}")
    # Copy out of tests/fixtures/: serve_timeline writes a timeline cache next
    # to the profile, which must not land on a committed fixture.
    profile = tmp_path / "profile.sqlite"
    shutil.copyfile(profile_source, profile)

    root = tmp_path / "elsewhere" / "sessions"
    session_id = "web-root-e2e"
    published = run_diagnose(
        profile_path=str(profile),
        session_id=session_id,
        gpu=0,
        session_root=root,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert published == 0
    assert (root / session_id / "findings.json").is_file()

    captured: dict[str, object] = {}

    def _capture_instead_of_serving(server, open_url, prof):
        captured["session_root"] = web._ViewerHandler._session_root
        captured["session_id"] = web._ViewerHandler._session_id
        captured["findings"] = list(web._ViewerHandler._findings)
        server.server_close()

    monkeypatch.setattr(web, "_run_server", _capture_instead_of_serving)

    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    opened = run_diagnose(
        profile_path=None,
        session_id=session_id,
        web=True,
        port=0,
        open_browser=False,
        session_root=root,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert opened == 0
    assert captured["session_root"] == os.fspath(root)
    assert captured["session_id"] == session_id
    assert captured["findings"], "findings from the custom session root were not loaded"
    # The default root must never be created as a side effect of honouring a custom one.
    assert not (workdir / ".nsys-ai").exists()
    assert not (Path.cwd() / ".nsys-ai").exists()


def test_review_gpu_default_matches_diff_all_gpus():
    """Parser default for review --gpu must be None (all GPUs), like diff."""
    from nsys_ai.cli.parsers import _build_parser

    parser = _build_parser()
    review = parser.parse_args(["review", "before.sqlite", "after.sqlite"])
    diff = parser.parse_args(["diff", "before.sqlite", "after.sqlite"])
    assert review.gpu is None
    assert diff.gpu is None


def test_undecided_review_names_the_cli_decision_path_and_it_runs(tmp_path: Path):
    """The printed next step must be a command that works, not only the browser.

    `diff --session --accept/--reject` records a decision, so offering the browser
    alone sends a CLI user to a surface they may not want. This drives the printed
    command verbatim rather than asserting on its text alone.
    """
    from nsys_ai.runspec import RunSpec

    before, after = BEFORE.resolve(), AFTER.resolve()
    session_id = "review-decision-path"
    runspec_path = tmp_path / "runspec.json"
    runspec_path.write_bytes(RunSpec(argv=("true",)).canonical_json_bytes())

    evidence = _run_cli(
        tmp_path, "evidence", "build", str(before), "--format", "json",
        "--gpu", "0", "--session", session_id, "--analyzers", "overlap_ratio",
    )
    assert evidence.returncode == 0, evidence.stderr
    finding_id = next(
        f["id"]
        for f in json.loads(evidence.stdout)["findings"]
        if f.get("id") and f.get("suggested_actions")
    )
    assert _run_cli(
        tmp_path, "propose", "--session", session_id,
        "--finding-id", finding_id, "--runspec", str(runspec_path),
    ).returncode == 0
    assert _run_cli(
        tmp_path, "diff", str(before), str(after), "--gpu", "0",
        "--format", "json", "--no-ai", "--session", session_id,
    ).returncode == 0

    review = _run_cli(tmp_path, "review", "--session", session_id)
    assert review.returncode == 0, review.stderr
    line = next(
        (ln for ln in review.stdout.splitlines() if ln.startswith("Decision path: undecided")),
        None,
    )
    assert line is not None, review.stdout
    assert "nsys-ai diff" in line, f"CLI decision path not offered: {line}"
    assert "--accept|--reject" in line

    # Drive it, substituting a concrete choice for the alternation.
    decided = _run_cli(
        tmp_path, "diff", str(before), str(after), "--gpu", "0",
        "--format", "json", "--no-ai", "--session", session_id,
        "--reject", "--reason", "following the printed instruction",
    )
    assert decided.returncode == 0, decided.stderr

    directory = session_dir(session_id, root=tmp_path / ".nsys-ai" / "sessions")
    payload = json.loads(
        (directory / "decisions.json").read_text(encoding="utf-8")
    )
    assert payload["decisions"][-1]["status"] == "rejected"
    assert json.loads((directory / "session.json").read_text())["phase"] == "propose"
    assert not (directory / "diff.json").exists()
