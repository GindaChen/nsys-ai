"""Ordinary mistakes must produce an error, not a traceback and not a hang.

Every case here was reproduced against the shipped CLI: a typo, a busy port, a
terminal that is not a terminal. Each one is a *usage* mistake — the command was
asked for something it cannot accept — so each is answered with the coded error
line the rest of the CLI already uses and with exit status 2.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nsys_ai import profile as _profile
from nsys_ai import web

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "tests" / "fixtures" / "h100_2gpu_1s.sqlite"


def _environment() -> dict[str, str]:
    """Environment whose PYTHONPATH reaches this checkout's sources first."""
    environment = dict(os.environ)
    source = str(ROOT / "src")
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not current else f"{source}{os.pathsep}{current}"
    return environment


def _run_cli(*args: str, timeout: float = 180.0, cwd: Path | None = None):
    return subprocess.run(
        [sys.executable, "-m", "nsys_ai", *args],
        cwd=str(cwd) if cwd is not None else str(ROOT),
        env=_environment(),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=timeout,
        check=False,
    )


def _profile_window_seconds() -> tuple[float, float]:
    with _profile.open(str(PROFILE)) as prof:
        lo_ns, hi_ns = prof.meta.time_range
    return lo_ns / 1e9, hi_ns / 1e9


# ── A missing required skill parameter ─────────────────────────────────


def test_skill_run_without_a_required_parameter_is_a_coded_usage_error():
    """`skill run <name> <profile>` is the documented form; 7 skills need more.

    It used to end in a ValueError traceback, which a user cannot tell apart
    from a bug in the tool.
    """
    result = _run_cli("skill", "run", "region_mfu", str(PROFILE))

    assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
    combined = result.stdout + result.stderr
    assert "Traceback" not in combined, combined
    assert "Error [SKILL_PARAMETER_REQUIRED]:" in result.stderr, result.stderr
    # The message has to name the parameter, the way to pass it, and where to
    # look the rest up — that is what turns the error into a next step.
    assert "'name'" in result.stderr, result.stderr
    assert "-p name=VALUE" in result.stderr, result.stderr
    assert "nsys-ai skill info region_mfu" in result.stderr, result.stderr


def test_skill_run_without_a_required_parameter_stays_parseable_json():
    """In --format json a machine consumer must still get JSON on stdout."""
    result = _run_cli("skill", "run", "region_mfu", str(PROFILE), "--format", "json")

    assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
    assert "Traceback" not in result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "SKILL_PARAMETER_REQUIRED", payload
    assert payload["error"]["skill"] == "region_mfu", payload
    assert payload["error"]["parameter"] == "name", payload
    # A usage mistake is not an abstention: the profile could have answered.
    assert "_abstained" not in payload, payload


def test_skill_list_marks_the_skills_that_need_a_parameter():
    """Discoverable beforehand, so the error above is rarely reached at all."""
    result = _run_cli("skill", "list")

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    marked = [ln for ln in lines if ln.startswith("region_mfu ")]
    assert marked and marked[0].split()[1] == "*", result.stdout
    plain = [ln for ln in lines if ln.startswith("top_kernels ")]
    assert plain and plain[0].split()[1] != "*", result.stdout
    assert "-p KEY=VALUE" in result.stdout, result.stdout
    assert "nsys-ai skill info <name>" in result.stdout, result.stdout


def test_skill_list_marks_only_the_parameters_that_stop_execution():
    """`required` with a default never stops a run, so it must not be marked.

    The text marker reads the same condition ``Skill.execute`` reads, not the
    ``required`` flag on its own — otherwise it would warn about parameters
    that resolve themselves.
    """
    from nsys_ai.cli.handlers import _required_param_names
    from nsys_ai.skills.registry import get_skill

    region_mfu = get_skill("region_mfu")
    declared = {p.name for p in region_mfu.params if p.required}
    blocking = set(_required_param_names(region_mfu))
    assert "name" in blocking
    assert blocking <= declared
    assert {p.name for p in region_mfu.params if p.required and p.default is not None}.isdisjoint(
        blocking
    )
    assert _required_param_names(get_skill("top_kernels")) == []


# ── A --trim window that selects nothing ───────────────────────────────


def test_trim_outside_the_profile_window_names_both_windows():
    """`--trim 0 1` reads on the capture clock, which does not start at zero.

    It used to exit 0 after printing a single newline, which reads as "nothing
    to see here" rather than "you asked for a window this profile never had".
    """
    result = _run_cli("tui", str(PROFILE), "--gpu", "0", "--trim", "0", "1")

    assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
    assert "Traceback" not in result.stdout + result.stderr
    assert "Error [TRIM_OUT_OF_RANGE]:" in result.stderr, result.stderr
    lo_s, hi_s = _profile_window_seconds()
    assert f"{lo_s:.3f}" in result.stderr, result.stderr
    assert f"{hi_s:.3f}" in result.stderr, result.stderr
    assert "--trim 0.000 1.000" in result.stderr, result.stderr


@pytest.mark.parametrize("command", ["tree", "open", "overlap", "nccl", "search", "summary"])
def test_trim_outside_the_window_is_refused_by_every_single_profile_command(command):
    """One command guarded is not the contract; the reader picks the command.

    `tree` is `tui`'s non-interactive twin and reproduced the reported symptom
    verbatim -- exit 0, one byte of output -- while `tui` next to it exited 2.
    `open` was worse: it re-derived the nanosecond window inline and never
    reached the check at all.
    """
    extra = ["--query", "x"] if command == "search" else []
    if command == "open":
        extra = ["--viewer", "web", "--no-browser"]
    result = _run_cli(command, str(PROFILE), "--gpu", "0", "--trim", "0", "1", *extra)

    assert result.returncode == 2, (command, result.returncode, result.stdout, result.stderr)
    assert "Traceback" not in result.stdout + result.stderr, result.stderr
    assert "Error [TRIM_OUT_OF_RANGE]:" in result.stderr, result.stderr


def test_a_zero_width_trim_window_is_out_of_range():
    """`--trim 156 156` sits inside the profile and still selects nothing.

    Same empty result as a window that misses entirely, reached by different
    arithmetic, so it gets the same answer.
    """
    lo_s, _ = _profile_window_seconds()
    midpoint = f"{lo_s + 0.5:.6f}"
    result = _run_cli("export-csv", str(PROFILE), "--gpu", "0", "--trim", midpoint, midpoint)

    assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
    assert "Error [TRIM_OUT_OF_RANGE]:" in result.stderr, result.stderr


def test_trim_inside_the_profile_window_is_untouched(tmp_path: Path):
    """The check must reject only a window that misses the profile entirely."""
    lo_s, hi_s = _profile_window_seconds()
    out = tmp_path / "kernels.csv"
    result = _run_cli(
        "export-csv",
        str(PROFILE),
        "--gpu",
        "0",
        "--trim",
        f"{lo_s:.6f}",
        f"{hi_s:.6f}",
        "-o",
        str(out),
    )

    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert out.is_file()


def test_trim_overlapping_only_partly_is_accepted():
    """A window that starts before the profile still selects real events."""
    from argparse import Namespace

    from nsys_ai.cli.handlers import _check_trim_window, _parse_trim

    with _profile.open(str(PROFILE)) as prof:
        lo_ns, _ = prof.meta.time_range
        trim = _parse_trim(Namespace(trim=[0.0, lo_ns / 1e9 + 0.5]))
        _check_trim_window(trim, prof)  # must not raise


@pytest.mark.parametrize(
    ("command", "extra"),
    [
        ("diagnose", ["--no-browser"]),
        ("diff", ["--gpu", "0", "--no-ai"]),
        ("review", ["--gpu", "0"]),
    ],
)
def test_session_verbs_reject_trim_outside_the_profile_window(command, extra):
    """The session-era front doors must share the established trim guard."""
    args = [str(PROFILE), "--trim", "0", "1"]
    if command in {"diff", "review"}:
        args = [str(PROFILE), str(PROFILE), *args[1:]]
    result = _run_cli(command, *args, *extra)

    assert result.returncode == 2, (command, result.returncode, result.stderr)
    assert "Error [TRIM_OUT_OF_RANGE]:" in result.stderr, result.stderr
    if command == "diff":
        assert "before profile:" in result.stderr, result.stderr


@pytest.mark.parametrize(
    ("command", "args", "needs_side_label"),
    [
        ("skill", ["run", "top_kernels"], False),
        ("agent", ["analyze"], False),
        ("cutracer", ["plan"], False),
        ("cutracer", ["analyze", str(ROOT / "tests" / "fixtures" / "cutracer")], False),
        ("cutracer", ["run", "--launch-cmd", "true", "--dry-run"], False),
        ("diff-web", [str(PROFILE)], True),
    ],
)
def test_every_trim_consumer_rejects_an_out_of_range_window(command, args, needs_side_label):
    """Every profile-backed --trim entry point must share the range guard."""
    if command == "skill":
        argv = [command, *args, str(PROFILE), "--trim", "0", "1", "--format", "json"]
    elif command == "agent":
        argv = [command, *args, str(PROFILE), "--trim", "0", "1"]
    elif command == "cutracer":
        argv = [command, *args, str(PROFILE), "--trim", "0", "1"]
        if args[0] == "analyze":
            argv = [command, *args[:1], str(PROFILE), args[1], "--trim", "0", "1"]
    else:
        argv = [command, str(PROFILE), str(PROFILE), "--trim", "0", "1", "--no-browser"]

    result = _run_cli(*argv, timeout=30.0)

    assert result.returncode == 2, (argv, result.returncode, result.stdout, result.stderr)
    assert "Traceback" not in result.stdout + result.stderr
    assert "Error [TRIM_OUT_OF_RANGE]:" in result.stderr, result.stderr
    if needs_side_label:
        assert "before profile:" in result.stderr, result.stderr


def test_optimize_checks_trim_before_starting_the_loop(monkeypatch):
    """An invalid optimize window must stop before the runner can capture."""
    from argparse import Namespace

    from nsys_ai.cli import handlers
    from nsys_ai.exceptions import TrimOutOfRangeError

    checked = []

    def record_check(trim, path, profile, **kwargs):
        checked.append((trim, path, profile, kwargs))
        raise TrimOutOfRangeError("invalid trim")

    monkeypatch.setattr(handlers, "_check_trim_window_for_path", record_check)
    monkeypatch.setattr(
        "nsys_ai.optimize_command.run_optimize",
        lambda **_kwargs: pytest.fail("optimize loop started before trim validation"),
    )
    args = Namespace(
        profile=str(PROFILE),
        repo=".",
        workload=["true"],
        trim=[0.0, 1.0],
        nsys="nsys",
        gpu=0,
        session=None,
    )

    with pytest.raises(TrimOutOfRangeError, match="invalid trim"):
        handlers._cmd_optimize(args, _profile)

    assert checked and checked[0][0] == (0, 1_000_000_000)


# ── chat without a terminal ────────────────────────────────────────────


def test_chat_without_a_terminal_refuses_before_the_app_starts():
    """It used to draw a screen into the pipe and then wait for input forever.

    The check has to sit in the handler: once Textual is running its stream
    capture answers isatty() True unconditionally, so a later check sees a
    terminal that is not there.
    """
    started = time.monotonic()
    try:
        result = _run_cli("chat", str(PROFILE), timeout=30.0)
    except subprocess.TimeoutExpired:  # pragma: no cover - the bug being fixed
        pytest.fail("chat hung with no terminal instead of refusing")
    elapsed = time.monotonic() - started

    assert elapsed < 15.0, f"chat took {elapsed:.1f}s to refuse"
    assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
    assert "Traceback" not in result.stdout + result.stderr
    assert "Error [NOT_A_TERMINAL]:" in result.stderr, result.stderr
    # Refusing is only half an answer; the message names the way through.
    assert "nsys-ai ask" in result.stderr, result.stderr


def test_chat_accepts_a_terminal_whose_stdout_is_redirected(monkeypatch, tmp_path: Path):
    """`nsys-ai chat p.sqlite > log` is a working invocation, not a mistake.

    Textual draws through stderr, so with a real terminal the UI is usable
    while stdout goes to a file. Gating on stdout as well as stdin would refuse
    it -- a case that worked before the hang was fixed.
    """
    from argparse import Namespace

    from nsys_ai import tui_textual
    from nsys_ai.cli import handlers

    started: list[str] = []
    monkeypatch.setattr(tui_textual, "run_chat_tui", lambda path: started.append(path))

    class _Terminal:
        @staticmethod
        def isatty():
            return True

    monkeypatch.setattr(sys, "stdin", _Terminal())
    # ...while stdout is a plain file, as it is under a shell redirect.
    monkeypatch.setattr(sys, "stdout", (tmp_path / "log").open("w"))
    try:
        handlers._cmd_chat(Namespace(profile=str(PROFILE)), None)
    finally:
        sys.stdout.close()

    assert started == [str(PROFILE)]


def test_chat_session_opens_handoff_and_passes_it_to_tui(monkeypatch, tmp_path: Path):
    """The standalone chat entry point can create a portable session handoff."""
    from argparse import Namespace

    from nsys_ai import tui_textual
    from nsys_ai.cli import handlers
    from nsys_ai.session_store import SessionStore

    started: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        tui_textual,
        "run_chat_tui",
        lambda path, **kwargs: started.append((path, kwargs)),
    )

    class _Terminal:
        @staticmethod
        def isatty():
            return True

    monkeypatch.setattr(sys, "stdin", _Terminal())
    session_dir = tmp_path / "chat-session"
    handlers._cmd_chat(
        Namespace(profile=str(PROFILE), session=str(session_dir)),
        None,
    )

    assert started == [
        (
            str(PROFILE),
            {"session_id": "chat-session", "session_root": str(tmp_path)},
        )
    ]
    snapshot = SessionStore(tmp_path).load("chat-session")
    assert snapshot.state.before_profile is not None
    assert snapshot.state.before_profile.path == str(PROFILE)


# ── a busy port ────────────────────────────────────────────────────────


def test_bind_local_server_falls_back_when_the_port_is_taken():
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    busy = holder.getsockname()[1]
    try:
        server = web._bind_local_server(busy, web._ViewerHandler)
    finally:
        holder.close()
    try:
        assert server.server_address[1] != busy
    finally:
        server.server_close()


def test_timeline_web_survives_a_busy_port(tmp_path: Path, monkeypatch):
    """`web` and `diff-web` already stepped aside; timeline-web raised instead."""
    # serve_timeline writes a timeline cache next to the profile, which must
    # not land on a committed fixture.
    profile_copy = tmp_path / "profile.sqlite"
    shutil.copyfile(PROFILE, profile_copy)

    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    busy = holder.getsockname()[1]

    bound: dict[str, int] = {}

    def _capture_instead_of_serving(server, open_url, prof):
        bound["port"] = server.server_address[1]
        server.server_close()

    monkeypatch.setattr(web, "_run_server", _capture_instead_of_serving)
    monkeypatch.chdir(tmp_path)

    try:
        with _profile.open(str(profile_copy)) as prof:
            web.serve_timeline(prof, [0], None, port=busy, open_browser=False)
    finally:
        holder.close()

    assert bound.get("port") not in (None, busy), bound


def test_timeline_web_trim_uses_progressive_shell_and_bounded_prebuild(
    tmp_path: Path, monkeypatch
):
    profile_copy = tmp_path / "profile.sqlite"
    shutil.copyfile(PROFILE, profile_copy)
    calls: list[tuple[int, int]] = []
    captured: dict[str, bytes] = {}

    def _fake_timeline_data(_prof, devices, window, **_kwargs):
        calls.append(window)
        return [{"id": dev, "kernels": [], "nvtx_spans": []} for dev in devices]

    def _capture_instead_of_serving(server, _open_url, _prof):
        captured["html"] = web._ViewerHandler.html_bytes
        server.server_close()

    monkeypatch.setattr(web, "build_timeline_gpu_data", _fake_timeline_data)
    monkeypatch.setattr(web, "_run_server", _capture_instead_of_serving)
    monkeypatch.chdir(tmp_path)

    with _profile.open(str(profile_copy)) as prof:
        start_ns, end_ns = prof.meta.time_range
        trim = (start_ns, min(end_ns, start_ns + 2_000_000_000))
        web.serve_timeline(prof, [0], trim, port=0, open_browser=False)

    html = captured["html"].decode("utf-8")
    assert "INITIAL_DATA: null" in html
    assert "PROGRESSIVE: '1' === '1'" in html
    assert f"LOOP_TRIM_NS: [{trim[0]}, {trim[1]}]" in html
    assert calls and all(window == trim for window in calls)


def test_timeline_web_bind_failure_waits_for_the_background_worker(tmp_path: Path, monkeypatch):
    """A bind that still fails must not leave the NVTX worker running.

    serve_timeline starts a background thread on the profile's DuckDB
    connection before it binds. If the bind escapes, the caller's ``with``
    closes that connection under the running thread, which segfaults the
    process instead of raising — the error is then invisible.
    """
    import threading

    profile_copy = tmp_path / "profile.sqlite"
    shutil.copyfile(PROFILE, profile_copy)

    def _always_fails(port, handler):
        raise OSError("bind refused")

    monkeypatch.setattr(web, "_bind_local_server", _always_fails)
    monkeypatch.chdir(tmp_path)

    with _profile.open(str(profile_copy)) as prof:
        with pytest.raises(OSError):
            web.serve_timeline(prof, [0], None, port=0, open_browser=False)
        alive = [
            t
            for t in threading.enumerate()
            if t.name == "timeline-nvtx-warmup" and t.is_alive()
        ]
        assert not alive, "the background worker still holds the connection being closed"


# ── propose into a session that has moved on ───────────────────────────


def test_propose_reports_a_session_conflict_as_a_coded_error(tmp_path: Path):
    """Re-running the command `diagnose` prints used to raise a bare ValueError."""
    from nsys_ai.diagnose_command import run_diagnose
    from nsys_ai.propose_command import ProposeCommandError, run_propose
    from nsys_ai.runspec import RunSpec

    profile_copy = tmp_path / "profile.sqlite"
    shutil.copyfile(PROFILE, profile_copy)
    root = tmp_path / "sessions"
    session_id = "propose-conflict"

    assert (
        run_diagnose(
            profile_path=str(profile_copy),
            session_id=session_id,
            gpu=0,
            session_root=root,
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        == 0
    )
    report = json.loads((root / session_id / "findings.json").read_text())
    finding_id = next(f["id"] for f in report["findings"] if f.get("id"))

    # First proposal: no RunSpec, so the session records "no verification path".
    assert (
        run_propose(
            finding_id=finding_id,
            session_id=session_id,
            session_root=root,
            stdout=io.StringIO(),
        )
        == 0
    )

    runspec_path = tmp_path / "runspec.json"
    runspec_path.write_text(json.dumps(RunSpec(argv=["python", "train.py"]).to_dict()))

    with pytest.raises(ProposeCommandError) as caught:
        run_propose(
            finding_id=finding_id,
            session_id=session_id,
            runspec_path=str(runspec_path),
            session_root=root,
            stdout=io.StringIO(),
        )
    assert caught.value.error_code == "PROPOSE_COMMAND_INVALID"
    assert session_id in str(caught.value)


# ── the shape of the contract itself ───────────────────────────────────


def test_usage_errors_exit_two_and_runtime_errors_exit_one():
    """Exit 1 means the command broke; exit 2 means it was asked wrongly."""
    from nsys_ai.exceptions import NsysAiError, ProfileNotFoundError, UsageError

    assert NsysAiError.exit_code == 1
    assert ProfileNotFoundError.exit_code == 1
    assert UsageError.exit_code == 2


def test_a_missing_profile_still_exits_one():
    """The established runtime-error status must not shift under this change."""
    result = _run_cli("summary", "nope.sqlite")

    assert result.returncode == 1, (result.returncode, result.stdout, result.stderr)
    assert "Error [PROFILE_NOT_FOUND]:" in result.stderr, result.stderr
