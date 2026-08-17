"""Basic smoke tests for nsys-ai package."""

import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_help():
    """CLI --help should exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "nsys-ai" in result.stdout


def test_import():
    """Package should be importable and expose __version__."""
    import nsys_ai

    assert hasattr(nsys_ai, "__version__")
    assert isinstance(nsys_ai.__version__, str)
    assert nsys_ai.__version__  # non-empty


def test_subcommands():
    """Public CLI surface should stay small and web/AI focused."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "--help"], capture_output=True, text=True
    )
    for cmd in [
        "open",
        "web",
        "timeline-web",
        "chat",
        "ask",
        "report",
        "diff",
        "diff-web",
        "loop",
        "export",
        "agent",
        "agent-guide",
        "propose",
        "diagnose",
        "review",
        "optimize",
        "info",
        "warm",
        "skill",
        "evidence",
        "cutracer",
    ]:
        assert cmd in result.stdout, f"Missing subcommand: {cmd}"

    # Legacy command names should be hidden from top-level help.
    usage_text = result.stdout.split("positional arguments:", 1)[0]
    assert "loop" in usage_text
    for hidden in ["summary", "overlap", "analyze"]:
        assert hidden not in usage_text

    # 'agent' is public. The zero-arg banner tells the reader that
    # `nsys-ai agent analyze` exists, and that report has no other entry point
    # (`report`/`analyze` run a different pipeline), so --help must confirm it.
    assert ",agent," in usage_text

    # The usage metavar is hand-maintained; keep it honest about what is
    # actually registered so machine consumers can trust it.
    import argparse

    from nsys_ai.cli.parsers import _build_parser

    registered = set()
    for action in _build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            registered = set(action.choices)
    assert registered, "could not find the subparsers action on the public parser"
    advertised = set(usage_text.split("{", 1)[1].split("}", 1)[0].split(","))
    assert registered == advertised, (
        f"usage metavar out of sync: missing {sorted(registered - advertised)}, "
        f"stale {sorted(advertised - registered)}"
    )


def test_agent_subcommand_is_reachable():
    """`nsys-ai agent` must parse the same whether or not it is in --help."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "agent", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "{analyze,ask}" in result.stdout

    # Bare `agent` still prints its usage line and exits non-zero (unchanged).
    bare = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "agent"], capture_output=True, text=True
    )
    assert bare.returncode == 1
    assert "Usage: nsys-ai agent {analyze,ask}" in bare.stdout


def test_custom_help_mentions_default_profile_shortcut():
    """The getting-started help should advertise the bare profile shortcut."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "nsys-ai <profile>" in result.stdout
    assert "Open web timeline UI (default)" in result.stdout


def test_default_profile_command_routes_to_timeline_web():
    """Bare profile paths should keep working as the default web timeline command."""
    from nsys_ai.cli.app import _normalize_default_profile_command

    assert _normalize_default_profile_command(["nsys-ai", "profile.nsys-rep"]) == [
        "nsys-ai",
        "timeline-web",
        "profile.nsys-rep",
    ]
    assert _normalize_default_profile_command(
        ["nsys-ai", "profile.nsys-rep", "--no-browser"]
    ) == [
        "nsys-ai",
        "timeline-web",
        "profile.nsys-rep",
        "--no-browser",
    ]


def test_default_profile_command_accepts_supported_profile_paths_only():
    """The documented shorthand applies only to profile paths the opener supports."""
    from nsys_ai.cli.app import _normalize_default_profile_command

    assert _normalize_default_profile_command(["nsys-ai", "profile.sqlite"])[1] == "timeline-web"
    assert _normalize_default_profile_command(["nsys-ai", "PROFILE.SQLITE"]) == [
        "nsys-ai",
        "timeline-web",
        "PROFILE.SQLITE",
    ]
    assert _normalize_default_profile_command(["nsys-ai", "profile.nsys-rep.zst"]) == [
        "nsys-ai",
        "profile.nsys-rep.zst",
    ]


def test_default_profile_command_leaves_subcommands_unchanged():
    """Named commands still parse through the normal public/legacy command tables."""
    from nsys_ai.cli.app import _normalize_default_profile_command

    assert _normalize_default_profile_command(["nsys-ai", "open", "profile.nsys-rep"]) == [
        "nsys-ai",
        "open",
        "profile.nsys-rep",
    ]


def test_chat_subcommand_help():
    """chat subcommand should have --help and accept a profile argument."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "chat", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "profile" in result.stdout


def test_diff_web_subcommand_help():
    """diff-web subcommand should have --help and accept before/after paths."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff-web", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "before" in result.stdout
    assert "after" in result.stdout


def test_diff_subcommand_help():
    """diff subcommand should have --help and accept before/after paths."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "before" in result.stdout
    assert "after" in result.stdout


def test_loop_subcommand_help():
    """loop should expose the baseline and surface inputs, and no candidate path.

    The help used to say "Omit --after to enter the candidate path later in the
    web UI", which /api/loop/reprofile refuses.
    """
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "loop", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "before" in result.stdout
    assert "--after" not in result.stdout
    assert "--surface" in result.stdout
    assert "--h100-preset" in result.stdout


def test_propose_subcommand_help():
    """propose should expose strict artifact inputs without session mutation."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "propose", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "findings" in result.stdout
    assert "--finding-id" in result.stdout
    assert "--runspec" in result.stdout
    assert "--output" in result.stdout


def test_diagnose_subcommand_help():
    """diagnose should expose session publish and --web reopen flags."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diagnose", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--session" in result.stdout
    assert "--web" in result.stdout


def test_review_subcommand_help():
    """review should expose before/after, --session resume, and --web."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "review", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--session" in result.stdout
    assert "--web" in result.stdout
    assert "before" in result.stdout
    # gpu must default like diff (all GPUs), not silently to device 0
    assert "default: all GPUs" in result.stdout


def test_optimize_subcommand_help():
    """optimize should document the whole loop, its inputs and its exit codes."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "optimize", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--repo" in result.stdout
    assert "--session" in result.stdout
    assert "profile" in result.stdout
    # argparse re-wraps the description, so compare against unwrapped text.
    unwrapped = " ".join(result.stdout.split())
    # A wrapping script needs to know that only a decision exits 0.
    assert "0 only when a decision was recorded" in unwrapped
    # The abstention promise is part of the contract, not an implementation note.
    assert "stops before re-profiling" in unwrapped


def test_optimize_documented_argument_order_parses():
    """``optimize <profile> --repo R -- <cmd>`` is the spelling the help shows."""
    from nsys_ai.cli.app import _normalize_optimize_command
    from nsys_ai.cli.parsers import _build_parser

    documented = [
        "nsys-ai",
        "optimize",
        "before.sqlite",
        "--repo",
        "/repo",
        "--session",
        "sid",
        "--",
        "./axpy",
        "--launches",
        "20",
    ]
    args = _build_parser().parse_args(_normalize_optimize_command(documented)[1:])
    assert args.command == "optimize"
    assert args.profile == "before.sqlite"
    assert args.repo == "/repo"
    assert args.session == "sid"
    # Options belonging to the workload must survive, not be parsed by nsys-ai.
    assert args.workload == ["./axpy", "--launches", "20"]

    # The options-first spelling argparse handles natively must agree.
    options_first = [
        "nsys-ai",
        "optimize",
        "--repo",
        "/repo",
        "--session",
        "sid",
        "before.sqlite",
        "--",
        "./axpy",
        "--launches",
        "20",
    ]
    assert _normalize_optimize_command(options_first) == options_first
    same = _build_parser().parse_args(options_first[1:])
    assert (same.profile, same.repo, same.session, same.workload) == (
        args.profile,
        args.repo,
        args.session,
        args.workload,
    )

    # A workload may contain its own '--'; nsys-ai must hand it over untouched.
    nested = ["nsys-ai", "optimize", "b.sqlite", "--repo", "/r", "--", "./a", "--", "-v"]
    assert _build_parser().parse_args(
        _normalize_optimize_command(nested)[1:]
    ).workload == ["./a", "--", "-v"]


def test_optimize_does_not_rewrite_an_option_into_the_profile_slot():
    """Rewriting must never feed the profile to an option waiting for a value."""
    import pytest

    from nsys_ai.cli.app import _normalize_optimize_command
    from nsys_ai.cli.parsers import _build_parser

    malformed = ["nsys-ai", "optimize", "b.sqlite", "--repo", "/r", "--session", "--", "./a"]
    assert _normalize_optimize_command(malformed) == malformed
    with pytest.raises(SystemExit) as excinfo:
        _build_parser().parse_args(_normalize_optimize_command(malformed)[1:])
    assert excinfo.value.code == 2


def test_optimize_without_a_workload_exits_2(tmp_path):
    """No verification workload means no loop; say so instead of profiling."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nsys_ai",
            "optimize",
            str(tmp_path / "before.sqlite"),
            "--repo",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "a verification workload is required" in result.stderr
    assert "Traceback" not in result.stderr


def test_loop_rejects_after(tmp_path):
    """`loop --after` must fail as an unknown flag, not be accepted and dropped.

    The session store is the loop's single source of truth: the CLI writes a
    session and every loop surface only renders one, so a candidate passed here
    lost to the session's own empty after profile without a word. A flag that is
    parsed, validated and then discarded is the same promise in a costlier form,
    so the flag is gone. The before profile is a real fixture: the rejection has
    to come from the flag, not from a path that does not resolve.
    """
    fixtures = Path(__file__).resolve().parent / "fixtures"
    result = subprocess.run(
        [
            sys.executable, "-m", "nsys_ai", "loop",
            str(fixtures / "mfu_2gpu_before.sqlite"),
            "--after", str(fixtures / "mfu_2gpu_after.sqlite"), "--no-browser",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 2, result.stderr
    assert "unrecognized arguments: --after" in result.stderr, result.stderr
    assert "Traceback" not in result.stderr, result.stderr


@pytest.mark.parametrize("surface", ["tree", "timeline"])
def test_loop_runs_every_surface_it_advertises(tmp_path, surface):
    """Two of `loop`'s three surfaces died on the way in, with no test either.

    The handler passes `session=` to the package-level `run_tui`/`run_timeline`,
    and neither wrapper accepted it -- so `--surface tree` and
    `--surface timeline` raised `TypeError` before doing any work, for every
    profile and every invocation. `--surface` is advertised in `loop --help`,
    which is the only reason anyone would type it.

    stdout is captured here, so both surfaces take their non-TTY fallback.
    That is the path CI and any piped use hits, and it is where the second
    failure lived: the static tree read the trim window as a pair and `loop`
    passes none, so it printed "Error loading profile" and still exited 0.
    """
    fixtures = Path(__file__).resolve().parent / "fixtures"
    result = subprocess.run(
        [
            sys.executable, "-m", "nsys_ai", "loop",
            str(fixtures / "mfu_2gpu_before.sqlite"),
            "--surface", surface,
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    # Not `stderr == ""`: a cold Parquet cache writes "[nsys-ai] Cache ready"
    # there, so a fresh clone would fail this on progress output rather than on
    # a defect. Match the sibling tests and assert the absence of a crash.
    assert "Traceback" not in result.stderr, result.stderr
    assert "Error loading profile" not in result.stderr, result.stderr

    # Assert a non-zero count, not merely non-empty output. An empty timeline
    # still prints "Timeline summary: GPU 0  0 kernels", so `stdout.strip()`
    # is satisfied by exactly the failure this test exists to catch. (An empty
    # tree does render as "", so that half would have bitten -- the asymmetry
    # is the point.)
    if surface == "timeline":
        counts = re.findall(r"(\d+) kernels", result.stdout)
        assert counts, result.stdout
        assert int(counts[0]) > 0, f"timeline reported {counts[0]} kernels"
    else:
        assert "📦" in result.stdout, result.stdout[:400]


@pytest.mark.parametrize(
    ("surface", "entry"),
    [("tree", "run_tui"), ("timeline", "run_timeline")],
)
def test_loop_hands_its_surfaces_a_resolved_trim_window(monkeypatch, surface, entry):
    """The surfaces cannot represent "no trim", so the handler must resolve it.

    `build_nvtx_tree` subscripts the window, and both TUI apps coerce None to
    `(0, 0)` -- which selects nothing on a capture clock that starts at 60 s.
    So passing None through rendered an empty tree and an empty timeline, and
    the capture in front of the user simply looked like it had no work in it.

    This asserts at the handler rather than through the CLI on purpose: a
    subprocess captures stdout, which sends both surfaces down their non-TTY
    fallback, so it cannot see what the interactive path is handed.
    """
    import nsys_ai.timeline as timeline_pkg
    import nsys_ai.tree as tree_pkg
    from nsys_ai import profile as profile_mod
    from nsys_ai.cli import handlers

    fixture = Path(__file__).resolve().parent / "fixtures" / "mfu_2gpu_before.sqlite"
    with profile_mod.open(str(fixture)) as prof:
        expected = (int(prof.meta.time_range[0]), int(prof.meta.time_range[1]))

    seen = {}

    def _capture(_path, _gpu, trim, **kwargs):
        seen["trim"] = trim

    monkeypatch.setattr(tree_pkg, "run_tui", _capture)
    monkeypatch.setattr(timeline_pkg, "run_timeline", _capture)

    args = SimpleNamespace(
        before=str(fixture), surface=surface, gpu=None, trim=None,
        session=None, h100_preset=False, port=None, no_browser=True,
    )
    handlers._cmd_loop(args, profile_mod)

    assert seen["trim"] == expected, (
        f"{entry} was handed {seen['trim']!r}; the whole capture is {expected!r}"
    )


@pytest.mark.parametrize("surface", ["tree", "timeline"])
def test_loop_defaults_to_a_device_the_capture_recorded_on(monkeypatch, tmp_path, surface):
    """Defaulting to GPU 0 renders nothing for a capture that has no GPU 0.

    A rank pinned across GPUs 1-7 is ordinary in a distributed run, and every
    committed fixture happens to record on device 0 -- so the whole test corpus
    could not express this, and `--surface timeline` reported
    "GPU 0  0 kernels" against a real 7-GPU capture full of work.

    The fixture is copied and remapped rather than read, because the committed
    ones are not to be written to.
    """
    import sqlite3

    import nsys_ai.timeline as timeline_pkg
    import nsys_ai.tree as tree_pkg
    from nsys_ai import profile as profile_mod
    from nsys_ai.cli import handlers

    source = Path(__file__).resolve().parent / "fixtures" / "mfu_2gpu_before.sqlite"
    remapped = tmp_path / "no_gpu_zero.sqlite"
    shutil.copyfile(source, remapped)
    with sqlite3.connect(remapped) as conn:
        # 0 -> 3 and 1 -> 4, applied high-to-low so the two do not collide.
        conn.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET deviceId = 4 WHERE deviceId = 1")
        conn.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET deviceId = 3 WHERE deviceId = 0")
    with profile_mod.open(str(remapped)) as prof:
        assert prof.meta.devices == [3, 4], prof.meta.devices

    seen = {}

    def _capture(_path, gpu, _trim, **kwargs):
        seen["gpu"] = gpu

    monkeypatch.setattr(tree_pkg, "run_tui", _capture)
    monkeypatch.setattr(timeline_pkg, "run_timeline", _capture)

    args = SimpleNamespace(
        before=str(remapped), surface=surface, gpu=None, trim=None,
        session=None, h100_preset=False, port=None, no_browser=True,
    )
    handlers._cmd_loop(args, profile_mod)

    assert seen["gpu"] == 3, f"defaulted to GPU {seen['gpu']}, which recorded nothing"


@pytest.mark.parametrize("surface", ["tree", "timeline"])
def test_loop_says_so_when_a_piped_surface_cannot_run_the_session(tmp_path, surface):
    """Exit 0 with a rendered view would otherwise read as "the loop ran".

    Piped, both surfaces fall back to a static render that opens no session and
    records no decision -- verified: no `.nsys-ai/` is created. The session is
    what `loop` is for, so dropping it silently reports success for work that
    did not happen.
    """
    fixtures = Path(__file__).resolve().parent / "fixtures"
    result = subprocess.run(
        [
            sys.executable, "-m", "nsys_ai", "loop",
            str(fixtures / "mfu_2gpu_before.sqlite"),
            "--surface", surface, "--session", "piped-demo",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "piped-demo" in result.stderr, result.stderr
    assert "was ignored" in result.stderr, result.stderr
    assert not (tmp_path / ".nsys-ai").exists(), "a session was opened after all"


def test_trim_resolution_always_yields_a_pair(tmp_path):
    """Consumers subscript the window without checking, so None is not an option.

    `viewer.generate_html` does `trim[0] / 1e9` and `_build_single_thread_tree`
    does `trim[0] - pad`. Returning None for a capture whose kernel span is
    degenerate turned `open --viewer web` into the same
    "'NoneType' object is not subscriptable" the resolution exists to prevent.
    """
    import sqlite3

    from nsys_ai import profile as profile_mod
    from nsys_ai.cli.handlers import _resolve_trim_window

    source = Path(__file__).resolve().parent / "fixtures" / "mfu_2gpu_before.sqlite"
    degenerate = tmp_path / "degenerate.sqlite"
    shutil.copyfile(source, degenerate)
    with sqlite3.connect(degenerate) as conn:
        conn.execute(
            "DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE rowid NOT IN "
            "(SELECT rowid FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 1)"
        )
        conn.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET [end] = start")

    with profile_mod.open(str(degenerate)) as prof:
        lo, hi = prof.meta.time_range
        assert hi <= lo, f"fixture is not degenerate: {(lo, hi)}"
        window = _resolve_trim_window(None, prof)

    # A pair, and the same pair `open` produced before the resolution moved
    # into a helper. Narrowing a degenerate span to (0, 0) would be a behaviour
    # change, not a safety net: (0, 0) is truthy, so it filters rather than
    # meaning "no window".
    assert window == (int(lo), int(hi)), window


def test_loop_reports_a_bad_trim_as_a_trim_error(tmp_path):
    """A window outside the capture is not "could not open before profile".

    `_resolve_trim_window` raises TrimOutOfRangeError, which cli/app.py renders
    with its error code and exit status. Catching it beside the profile open
    relabelled it and dropped both, so `loop` disagreed with `tui` about the
    same mistake.
    """
    fixtures = Path(__file__).resolve().parent / "fixtures"
    result = subprocess.run(
        [
            sys.executable, "-m", "nsys_ai", "loop",
            str(fixtures / "mfu_2gpu_before.sqlite"),
            "--surface", "tree", "--trim", "0", "1",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 2, result.stderr
    assert "TRIM_OUT_OF_RANGE" in result.stderr, result.stderr
    assert "could not open before profile" not in result.stderr, result.stderr


def test_timeline_web_rejects_loop_after(tmp_path):
    """The same dead flag one subcommand over, and the last way to reach it.

    `--loop-after` was parsed, forwarded through `serve_timeline`, `run_tui` and
    `run_timeline`, and read by none of them -- the session store replaced it.
    Since `loop`'s default surface is `timeline-web`, it was also the only
    remaining way to hand a loop surface a candidate it cannot register.
    """
    fixtures = Path(__file__).resolve().parent / "fixtures"
    result = subprocess.run(
        [
            sys.executable, "-m", "nsys_ai", "timeline-web",
            str(fixtures / "mfu_2gpu_before.sqlite"),
            "--loop-after", str(fixtures / "mfu_2gpu_after.sqlite"), "--no-browser",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 2, result.stderr
    assert "unrecognized arguments: --loop-after" in result.stderr, result.stderr
    assert "Traceback" not in result.stderr, result.stderr


def test_no_surface_still_takes_a_candidate_it_cannot_register(tmp_path):
    """The parameter is gone from the whole chain, not just from the parser.

    A flag removed while the parameter it fed survives is half a fix: the next
    caller re-adds the flag because the plumbing still looks ready for it.
    """
    src = Path(__file__).resolve().parents[1] / "src" / "nsys_ai"
    offenders = [
        f"{path.relative_to(src)}:{i}"
        for path in src.rglob("*.py")
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if "loop_after" in line
    ]
    assert not offenders, f"loop_after survives at: {offenders}"

    # The name is only half of it. `_open_session_web` kept an `after_path`
    # parameter after its one use was deleted, and its caller kept computing a
    # value to hand it -- plumbing that still looks ready for the flag, which
    # is exactly how the flag comes back. Ruff's enabled rules do not flag an
    # unused argument, so nothing else catches this.
    import inspect

    from nsys_ai.review_command import _open_session_web

    accepted = set(inspect.signature(_open_session_web).parameters)
    assert not accepted & {"after_path", "after", "loop_after"}, (
        f"_open_session_web still takes a candidate it cannot register: {sorted(accepted)}"
    )


def test_h100_preset_hint_names_a_command_that_parses():
    """The only recovery instruction on the preset path must still be runnable.

    `loop --h100-preset` with the dataset absent exits 1 after printing this
    hint, so a command in it that the parser rejects leaves the caller with no
    way forward. It named `--after` until that flag was removed.
    """
    import shlex

    from nsys_ai.cli.parsers import _build_parser
    from nsys_ai.loop_state import h100_preset_download_hint

    commands = [
        line.strip()
        for line in h100_preset_download_hint().splitlines()
        if line.strip().startswith("nsys-ai ")
    ]
    assert commands, "the hint stopped offering a command at all"
    for command in commands:
        argv = shlex.split(command)[1:]
        # Reaching a Namespace at all is the assertion: argparse exits 2 on an
        # unknown flag, which is exactly the failure this guards.
        _build_parser().parse_args(argv)


def test_loop_missing_profile_has_friendly_error(tmp_path):
    """loop should not dump a sqlite traceback for placeholder/missing paths."""
    missing = tmp_path / "missing.sqlite"
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "loop", str(missing), "--no-browser"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "before profile not found" in result.stderr
    assert "Traceback" not in result.stderr


def test_cutracer_subcommand_help():
    """cutracer subcommand should expose expected actions."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "cutracer", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    for action in ["check", "analyze", "plan", "install", "run"]:
        assert action in result.stdout


def test_legacy_analyze_still_available():
    """Hidden legacy command should still parse and show help."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "analyze", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--gpu" in result.stdout


def test_agent_guide():
    """agent-guide subcommand should print the system prompt payload."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "agent-guide"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "nsys-ai Agent Guide" in result.stdout
    assert "Orient" in result.stdout
    assert "Available Skills" in result.stdout


def test_doctor_no_profile():
    """doctor without a profile reports environment checks and exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "doctor"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "System" in result.stdout
    assert "Optional features" in result.stdout
    assert "Summary:" in result.stdout


def test_doctor_json():
    """doctor --format json emits a versioned, parseable envelope."""
    import json

    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "doctor", "--format", "json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "0.1"
    assert payload["producer"] == "nsys-ai"
    assert [s["name"] for s in payload["sections"]] == [
        "System",
        "Profile support",
        "Optional features",
    ]
    assert "summary" in payload


def test_doctor_with_profile(minimal_nsys_db_path):
    """doctor on a profile adds a health section."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "doctor", minimal_nsys_db_path],
        capture_output=True,
        text=True,
    )
    # May exit 1 if the synthetic profile trips a FAIL check; output is what matters.
    assert "Profile health" in result.stdout
    assert "Duration" in result.stdout


def test_doctor_missing_profile_exits_nonzero(tmp_path):
    """A missing profile is a FAIL, so doctor exits non-zero (can gate CI)."""
    missing = str(tmp_path / "nope.sqlite")
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "doctor", missing],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "FAIL" in result.stdout


def test_skill_info():
    """skill info subcommand should return a JSON schema."""
    import json

    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "skill", "info", "top_kernels"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    schema = json.loads(result.stdout)
    assert schema["name"] == "top_kernels"
    assert "description" in schema
    assert "parameters" in schema
    assert "limit" in schema["parameters"]
    assert schema["parameters"]["limit"]["type"] == "int"
    assert schema["parameters"]["limit"]["default"] == 15


def test_hidden_skill_management_commands():
    """Hidden skill management subcommands like add/remove/save should still parse correctly."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "skill", "add", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "skill_file" in result.stdout


def test_evidence_requires_subcommand():
    """'nsys-ai evidence' without a sub-action should fail fast (exit != 0)."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "evidence"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "build" in result.stderr  # argparse should mention valid choices


def test_skill_run_duckdb_cache(tmp_path):
    """skill run should work end-to-end, preferring DuckDB/Parquet cache when available."""
    import json
    import sqlite3

    # Create a minimal profile with tables the cache builder needs
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, "end" INTEGER, deviceId INTEGER,
            streamId INTEGER, correlationId INTEGER,
            shortName INTEGER, mangledName TEXT, demangledName INTEGER
        );
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
            (1000, 2000, 0, 7, 1, 1, 'kernel_a', 1);
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO StringIds VALUES (1, 'kernel_a');
        CREATE TABLE NVTX_EVENTS (
            start INTEGER, "end" INTEGER, globalTid INTEGER,
            text TEXT, textId INTEGER, eventType INTEGER, rangeId INTEGER
        );
    """)
    conn.close()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nsys_ai",
            "skill",
            "run",
            "schema_inspect",
            str(db_path),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    rows = json.loads(result.stdout)
    assert isinstance(rows, list)
    assert len(rows) >= 1
    table_names = {r.get("table_name") for r in rows}
    assert "kernels" in table_names

    # Verify the DuckDB/Parquet cache was actually built (not the SQLite fallback)
    cache_dir = db_path.with_suffix(".nsys-cache")
    assert cache_dir.exists(), f"Cache directory {cache_dir} was not created"
    parquet_files = list(cache_dir.glob("*.parquet"))
    assert len(parquet_files) >= 1, "No .parquet files found in cache directory"


def _write_min_profile(path, *, dur_ns):
    """Minimal Nsight-like SQLite export sufficient for the diff pipeline."""
    import sqlite3

    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE StringIds(id INT PRIMARY KEY, value TEXT)")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
        "start INT, [end] INT, deviceId INT, streamId INT, correlationId INT, "
        "shortName INT, demangledName INT)"
    )
    conn.execute("CREATE TABLE NVTX_EVENTS(text TEXT, globalTid INT, start INT, [end] INT)")
    conn.executemany(
        "INSERT INTO StringIds(id, value) VALUES(?,?)",
        [(1, "kA"), (2, "kA_dem")],
    )
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start, [end], deviceId, streamId, "
        "correlationId, shortName, demangledName) VALUES(?,?,?,?,?,?,?)",
        (0, dur_ns, 0, 7, 1, 1, 2),
    )
    conn.commit()
    conn.close()


def test_baseline_subcommand_help():
    """baseline subcommand should expose tag/list/show."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "baseline", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for word in ("tag", "list", "show"):
        assert word in result.stdout


def test_baseline_tag_list_show_roundtrip(tmp_path):
    """tag records a snapshot + meta.json; list/show read it back."""
    import json

    prof = tmp_path / "before.sqlite"
    _write_min_profile(prof, dur_ns=10_000_000)

    tag = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "baseline", "tag", "v1", str(prof),
         "--reason", "known good"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert tag.returncode == 0, f"stderr: {tag.stderr}\nstdout: {tag.stdout}"

    entry = tmp_path / ".nsys-ai-baselines" / "v1"
    assert (entry / "snapshot.sqlite").is_file()
    meta = json.loads((entry / "meta.json").read_text(encoding="utf-8"))
    assert meta["name"] == "v1"
    assert meta["reason"] == "known good"
    assert meta["runspec"] is None
    assert meta["profile_id"].startswith("nsys2:")
    assert meta["tagger"]
    assert meta["tagged_at"].endswith("Z")

    listed = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "baseline", "list"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert listed.returncode == 0
    assert "v1" in listed.stdout

    shown = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "baseline", "show", "v1"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert shown.returncode == 0
    assert json.loads(shown.stdout)["name"] == "v1"


def test_baseline_tag_blank_reason_rejected(tmp_path):
    prof = tmp_path / "before.sqlite"
    _write_min_profile(prof, dur_ns=10_000_000)
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "baseline", "tag", "v1", str(prof),
         "--reason", "   "],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 2
    assert "reason" in result.stderr.lower()


def test_baseline_show_unknown_rejected(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "baseline", "show", "nope"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 2
    assert "unknown baseline" in result.stderr.lower()


def test_diff_against_baseline_ref(tmp_path):
    """diff --against baseline:<name> resolves the tag and produces diff output."""
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _write_min_profile(before, dur_ns=10_000_000)
    _write_min_profile(after, dur_ns=30_000_000)

    tag = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "baseline", "tag", "v1", str(before),
         "--reason", "known good"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert tag.returncode == 0, f"stderr: {tag.stderr}"

    # --against form
    diff1 = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff", "--against", "baseline:v1",
         str(after), "--format", "markdown", "--no-ai"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert diff1.returncode == 0, f"stderr: {diff1.stderr}\nstdout: {diff1.stdout}"
    assert "not found" not in diff1.stderr.lower()
    assert diff1.stdout.strip()

    # positional token form
    diff2 = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff", "baseline:v1", str(after),
         "--format", "markdown", "--no-ai"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert diff2.returncode == 0, f"stderr: {diff2.stderr}\nstdout: {diff2.stdout}"
    assert diff2.stdout.strip()


def test_diff_against_unknown_baseline_errors(tmp_path):
    after = tmp_path / "after.sqlite"
    _write_min_profile(after, dur_ns=30_000_000)
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff", "--against", "baseline:missing",
         str(after)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 2
    assert "unknown baseline" in result.stderr.lower()


def test_warm_subcommand_help():
    """warm should be a public verb taking a profile path."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "warm", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "profile" in result.stdout


def test_warm_missing_profile_exits_nonzero(tmp_path):
    """A missing profile is an error, not a silent no-op."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "warm", str(tmp_path / "nope.sqlite")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr


def test_warm_builds_the_nvtx_kernel_map_and_then_reports_already_warm(tmp_path):
    """The behavioural gate: warm must leave the map on disk, not defer it.

    Building the cache alone does not build the map — that is deliberate
    (``_should_defer_nvtx_kernel_map``), and it is exactly the cost warm exists
    to move off the first NVTX-attribution query. If warm ever stops calling
    ``materialize_cached_nvtx_kernel_map``, the two Parquets below go missing.
    """
    import shutil
    from pathlib import Path

    fixture = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"
    profile = tmp_path / "p.sqlite"
    shutil.copy(fixture, profile)

    first = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "warm", str(profile)],
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, f"stderr: {first.stderr}\nstdout: {first.stdout}"
    assert "warmed in" in first.stdout

    cache_dir = tmp_path / "p.nsys-cache"
    assert (cache_dir / "nvtx_kernel_map.parquet").is_file()
    assert (cache_dir / "nvtx_path_dict.parquet").is_file()
    assert "nvtx kernel map: 0 rows" not in first.stdout

    # Second run finds both halves already on disk and builds nothing.
    second = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "warm", str(profile)],
        capture_output=True,
        text=True,
    )
    assert second.returncode == 0, f"stderr: {second.stderr}\nstdout: {second.stdout}"
    assert "already warm" in second.stdout
    assert "warmed in" not in second.stdout


def test_warm_on_a_read_only_cache_directory_says_why_and_exits_nonzero(tmp_path):
    """A warm that silently did not warm defeats the point of running it.

    A prebuilt cache on a read-only mount still serves queries — the lazy map
    build degrades to an in-memory one — so nothing else notices. ``warm`` has
    to, because the artifact it exists to persist cannot be persisted.
    """
    import os
    import shutil
    from pathlib import Path

    from nsys_ai import parquet_cache

    fixture = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"
    profile = tmp_path / "ro.sqlite"
    shutil.copy(fixture, profile)
    cache_dir = Path(parquet_cache.build_cache(str(profile)))
    assert not (cache_dir / "nvtx_kernel_map.parquet").exists(), (
        "the cache build produced the map, so this test is no longer testing warm"
    )

    mode = cache_dir.stat().st_mode
    parent_mode = cache_dir.parent.stat().st_mode
    # The lock file lives beside the cache dir, the staging dir inside it.
    os.chmod(cache_dir, 0o500)
    os.chmod(cache_dir.parent, 0o500)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "nsys_ai", "warm", str(profile)],
            capture_output=True,
            text=True,
        )
    finally:
        os.chmod(cache_dir.parent, parent_mode)
        os.chmod(cache_dir, mode)

    assert result.returncode == 1, f"stdout: {result.stdout}"
    assert "cannot warm" in result.stderr
    assert "already warm" not in result.stdout


def test_warm_when_only_the_lock_directory_is_read_only(tmp_path):
    """The map's lock file lives *beside* the cache, not inside it.

    ``_build_lock`` creates ``<cache_dir>.build.lock`` in the parent directory,
    so a writable cache dir under a read-only parent stops the build just as
    dead as a read-only cache dir does — and probing the cache dir's own mode
    cannot see it. warm has to learn this from the build, not infer it.
    """
    import os
    import shutil
    from pathlib import Path

    from nsys_ai import parquet_cache

    fixture = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"
    profile = tmp_path / "lock.sqlite"
    shutil.copy(fixture, profile)
    cache_dir = Path(parquet_cache.build_cache(str(profile)))
    lock = cache_dir.parent / f"{cache_dir.name}.build.lock"
    if lock.exists():
        lock.unlink()

    parent_mode = cache_dir.parent.stat().st_mode
    os.chmod(cache_dir.parent, 0o500)  # cache_dir itself stays writable
    try:
        result = subprocess.run(
            [sys.executable, "-m", "nsys_ai", "warm", str(profile)],
            capture_output=True,
            text=True,
        )
    finally:
        os.chmod(cache_dir.parent, parent_mode)

    assert result.returncode == 1, f"stdout: {result.stdout}"
    assert "cannot warm" in result.stderr
    assert "already warm" not in result.stdout
    assert not (cache_dir / "nvtx_kernel_map.parquet").exists()


def test_a_command_run_after_warm_builds_nothing(tmp_path):
    """The payoff the file-existence assertions do not pin.

    warm is only worth running if the next process finds both halves on disk.
    A cold ``skill run`` prints the cache build's own progress lines to stderr,
    so their absence after a warm is the observable form of "nothing was built".
    """
    import shutil
    from pathlib import Path

    fixture = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"
    profile = tmp_path / "p.sqlite"
    shutil.copy(fixture, profile)

    warm = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "warm", str(profile)],
        capture_output=True,
        text=True,
    )
    assert warm.returncode == 0, f"stderr: {warm.stderr}\nstdout: {warm.stdout}"

    after = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "skill", "run", "nvtx_layer_breakdown", str(profile)],
        capture_output=True,
        text=True,
    )
    assert after.returncode == 0, f"stderr: {after.stderr}"
    noise = after.stderr + after.stdout
    for marker in ("Building cache", "Cache ready", "skipping map"):
        assert marker not in noise, f"{marker!r} means warm left work behind:\n{noise}"


def test_warm_does_not_report_already_warm_after_an_empty_sweep(tmp_path, capsys, monkeypatch):
    """An empty sweep publishes nothing, so the next call repeats it.

    ``already warm`` is the signal a caller uses to decide repeat warms are
    free. For this outcome they are not: the sweep runs to completion, finds no
    kernel inside any range, writes no Parquet, and the whole cost returns on
    the next invocation. The outcome is forced here for the same reason as in
    ``test_an_empty_sweep_is_told_apart_from_a_missing_source``.
    """
    import argparse
    import shutil
    from pathlib import Path

    from nsys_ai import parquet_cache
    from nsys_ai import profile as profile_module
    from nsys_ai.cli.handlers import _cmd_warm

    fixture = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"
    profile = tmp_path / "empty.sqlite"
    shutil.copy(fixture, profile)
    # Base cache valid, map absent: exactly the state that used to print
    # "already warm" while the sweep it just paid for cached nothing.
    parquet_cache.build_cache(str(profile))
    monkeypatch.setattr(
        parquet_cache,
        "materialize_cached_nvtx_kernel_map_outcome",
        lambda conn: (parquet_cache.MAP_NO_ATTRIBUTION, ""),
    )

    _cmd_warm(argparse.Namespace(profile=str(profile)), profile_module)

    out = capsys.readouterr().out
    assert "already warm" not in out, out
    assert "partly warm" in out, out


def test_warm_succeeds_when_there_is_nothing_for_the_sweep_to_read(tmp_path):
    """A profile the sweep can find nothing in is warm, not broken.

    This minimal export has no RUNTIME table, so ``runtime.parquet`` never
    reaches the cache and the map build stops before the sweep. That is a
    property of the capture, not a failure to warm: exit 0, and the missing
    source named rather than described as "no attribution".
    """
    profile = tmp_path / "min.sqlite"
    _write_min_profile(profile, dur_ns=10_000_000)

    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "warm", str(profile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "nothing for the sweep to read" in result.stdout
    assert "runtime.parquet" in result.stdout
    assert (tmp_path / "min.nsys-cache").is_dir()


def _help_screen() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _declared_subcommands() -> set[str]:
    """Every command `main()` can dispatch, across both parsers.

    `main()` picks `_build_legacy_parser` by name for the legacy commands
    (summary, tui, viewer, export-csv, ...) and `_build_parser` otherwise, so
    neither one alone is the full surface.
    """
    from nsys_ai.cli.parsers import _build_legacy_parser, _build_parser

    names: set[str] = set()
    for parser in (_build_parser(), _build_legacy_parser()):
        for action in parser._subparsers._group_actions:  # noqa: SLF001 - argparse has no public API
            names.update(action.choices)
    return names


def test_getting_started_screen_names_the_loop_verbs():
    """`nsys-ai help` is the front door; the loop is the headline feature.

    It previously named none of `loop`, `diff`, `doctor`, `baseline`, so the whole
    diagnose -> propose -> re-profile -> diff -> decide path was undiscoverable
    from the tool's own introduction.
    """
    screen = _help_screen()
    for verb in ("optimize", "diagnose", "propose", "diff", "review", "loop", "doctor", "baseline"):
        assert f"nsys-ai {verb}" in screen, f"help screen does not name `{verb}`"


def test_every_dispatchable_command_is_visible_from_dash_dash_help():
    """`nsys-ai --help` must not omit a command that works.

    `main()` picks between two parsers by command name, so `--help` rendered
    only one of them: five working commands (`summary`, `tui`, `viewer`,
    `timeline`, `export-csv`, ...) appeared on the getting-started screen and
    nowhere in `--help`, and seven appeared in `--help` and not on the screen.
    A reader had no way to tell which list was authoritative.

    The getting-started screen stays a curated selection -- that is what it is
    for, and it says so. `--help` is the one that must be complete.
    """
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0

    # Read the two lists `--help` actually presents, rather than searching the
    # whole page for each name. Searching false-passes on prose: `viewer`
    # occurs in "Serve interactive web viewer" and `timeline` in "Open web
    # timeline UI", so both stay "present" after the epilog naming them is
    # deleted -- which is the regression this test exists to catch.
    # Horizontal whitespace only: `\s` spans newlines, so a greedy run would
    # swallow the following help line and drop the command it names.
    listed = set(
        re.findall(r"^[ \t]{2,}([a-z][a-z0-9-]*)[ \t]{2,}\S", result.stdout, re.MULTILINE)
    )
    epilog = result.stdout.split("also available:", 1)
    assert len(epilog) == 2, f"--help no longer carries the epilog:\n{result.stdout}"
    also_available = {
        name.strip() for name in epilog[1].split("\n\n", 1)[0].split(",") if name.strip()
    }

    missing = sorted(_declared_subcommands() - listed - also_available)
    assert not missing, (
        "these commands dispatch but are invisible from `--help`:\n  "
        + "\n  ".join(missing)
    )
    stale = sorted(also_available - _declared_subcommands())
    assert not stale, (
        "`--help` advertises commands that do not dispatch:\n  " + "\n  ".join(stale)
    )


def test_the_legacy_routing_set_matches_the_parser_that_serves_it():
    """The routed set is exactly the commands only the legacy parser has.

    Asserting equality rather than containment is the point. A subset check
    passes for the two ways this actually breaks: a command routed to a parser
    that does not register it (`invalid choice`), and a command only the legacy
    parser registers that nothing routes to it (unreachable, and invisible on
    every help surface). It also passes for the set derived the naive way --
    every name the legacy parser registers -- which would silently reroute
    `doctor`, `info`, `evidence`, `help` and `skill` to a different
    implementation than the one users get today.
    """
    from nsys_ai.cli.parsers import (
        LEGACY_ROUTED_COMMANDS,
        _build_legacy_parser,
        _build_parser,
    )

    def _subcommands(parser) -> set[str]:
        names: set[str] = set()
        for action in parser._subparsers._group_actions:  # noqa: SLF001
            names.update(action.choices)
        return names

    legacy = _subcommands(_build_legacy_parser())
    primary = _subcommands(_build_parser())

    assert LEGACY_ROUTED_COMMANDS == legacy - primary, (
        "the routing set must be exactly the commands only the legacy parser "
        f"registers.\n  routed but unregistered: {sorted(LEGACY_ROUTED_COMMANDS - legacy)}"
        f"\n  legacy-only but unreachable: {sorted(legacy - primary - LEGACY_ROUTED_COMMANDS)}"
        f"\n  routed though the primary parser owns it: "
        f"{sorted(LEGACY_ROUTED_COMMANDS & primary)}"
    )


def test_getting_started_screen_only_advertises_commands_that_exist():
    """Every `nsys-ai <verb>` the screen shows must be a real subcommand.

    A getting-started screen that names a command the parser rejects is worse than
    one that omits it: the reader runs it and gets `invalid choice`.
    """
    import re

    declared = _declared_subcommands()
    advertised = {
        m.group(1)
        for m in re.finditer(r"nsys-ai ([a-z][a-z0-9-]*)", _help_screen())
        if m.group(1) not in {"help"}
    }
    missing = sorted(advertised - declared)
    assert not missing, f"help screen advertises non-existent subcommands: {missing}"


def test_optimize_without_the_delimiter_does_not_blame_the_workload():
    """Omitting '--' must not report the workload as an unreadable profile.

    The argv normaliser used to slide the profile token past a workload that had
    lost its '--', so `optimize before.sqlite --repo /tmp ./axpy 40` failed with
    "could not resolve before profile: .../axpy" -- naming a file the caller never
    offered as a profile.
    """
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "optimize", "before.sqlite",
         "--repo", "/tmp", "./axpy", "40"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "could not resolve before profile" not in combined, combined
    assert "axpy" not in combined, combined


def test_optimize_reports_a_flag_led_workload_as_a_missing_delimiter():
    """No executable is named by a flag, so this can only be a dropped '--'."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "optimize", "--repo", "/tmp",
         "before.sqlite", "--verbose", "40"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "the workload must follow '--'" in result.stdout + result.stderr


def test_optimize_argv_normaliser_leaves_a_delimiterless_workload_alone():
    from nsys_ai.cli.app import _normalize_optimize_command

    # Two bare tokens in a row cannot both be option values -> a lost '--'.
    argv = ["x", "optimize", "b.sqlite", "--repo", "/r", "./axpy", "40"]
    assert _normalize_optimize_command(argv) == argv

    # The documented spellings still normalise to the options-first form.
    assert _normalize_optimize_command(
        ["x", "optimize", "b.sqlite", "--repo", "/r", "--", "./axpy", "40"]
    ) == ["x", "optimize", "--repo", "/r", "b.sqlite", "--", "./axpy", "40"]
    assert _normalize_optimize_command(["x", "optimize", "b.sqlite", "--repo", "/r"]) == [
        "x", "optimize", "--repo", "/r", "b.sqlite", "--",
    ]


def test_no_ai_help_does_not_call_itself_a_no_op():
    """--no-ai selects the deterministic narrative; it is not decorative."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "No-op" not in result.stdout


def _advertised_invocations() -> list[tuple[str, list[str]]]:
    """Turn each `nsys-ai <verb> ...` line on the help screen into a real argv.

    The screen is column-aligned, so the description cannot be found by splitting
    on runs of spaces. Instead the command ends at the first token that reads as
    prose: capitalised and not the value of a preceding flag.
    """
    import shlex

    values = {
        "<profile>": "p.sqlite", "<profile.sqlite>": "p.sqlite",
        "<before>": "b.sqlite", "<after>": "a.sqlite", "<command>": "true",
        "<name>": "top_kernels", "<file.md>": "f.md", "<id>": "x",
        "DIR": "out", "TEXT": "why", "ID": "x", "PATH": "p", "N": "0",
        "S": "0", "E": "1", "runspec.json": "r.json", "findings.json": "f.json",
    }
    out: list[tuple[str, list[str]]] = []
    for line in _help_screen().splitlines():
        stripped = line.strip()
        if not stripped.startswith("nsys-ai "):
            continue
        tokens = shlex.split(stripped)[1:]  # drop "nsys-ai"
        if not tokens or tokens[0].startswith(("<", "(")):
            continue  # the bare-profile shortcut, or a prose line
        argv: list[str] = []
        shown = ["nsys-ai"]
        for token in tokens:
            if token.startswith("["):
                break  # optional group; everything after is optional or prose
            # Every flag value on this screen is a placeholder (N, S, E, DIR,
            # TEXT, ID, PATH), so a capitalised token that is not one is where
            # the right-hand description begins.
            if token[:1].isupper() and token not in values and not token.startswith("-"):
                break
            shown.append(token)
            argv.append(values.get(token, token))
        if argv:
            out.append((" ".join(shown), argv))
    return out


def test_every_advertised_invocation_parses():
    """The screen must show forms that run, not forms that error on required flags.

    `nsys-ai viewer <profile> --gpu N` was advertised for five commands that also
    require --trim, so a newcomer typing what the front door showed got
    "error: the following arguments are required: --trim" five times in a row.
    """
    import contextlib
    import io

    from nsys_ai.cli.parsers import _build_legacy_parser, _build_parser

    failures = []
    for shown, argv in _advertised_invocations():
        if not argv:
            continue
        if argv[0] == "optimize":
            from nsys_ai.cli.app import _normalize_optimize_command

            argv = _normalize_optimize_command(["nsys-ai", *argv])[1:]
        legacy = argv[0] in {
            "summary", "tui", "timeline", "viewer", "export-csv", "analyze",
            "overlap", "nccl", "iters", "tree", "markdown", "search",
            "export-json", "timeline-html", "perfetto",
        }
        parser = _build_legacy_parser() if legacy else _build_parser()
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr):
                parser.parse_args(argv)
        except SystemExit:
            message = stderr.getvalue().strip().splitlines()
            failures.append(f"{shown!r} -> {message[-1] if message else 'SystemExit'}")
    assert not failures, "help screen shows invocations that do not parse:\n  " + "\n  ".join(failures)


def test_the_perfetto_ui_integration_is_gone_but_the_export_remains():
    """We do not ship an integration with someone else's hosted service.

    `nsys-ai perfetto` served the trace locally with `Access-Control-Allow-Origin: *`
    and opened ui.perfetto.dev, a page we do not run, over a link the user may not
    have. It could not participate in the loop either -- no findings overlay, no
    session, nothing came back. The Chrome Trace Event export stays: it is a format,
    not a service, and anyone who wants Perfetto can open the file there themselves.
    """
    import nsys_ai.web as web_module

    assert not hasattr(web_module, "serve_perfetto")
    assert not hasattr(web_module, "_PerfettoHandler")
    assert "ui.perfetto.dev" not in _help_screen()

    declared = _declared_subcommands()
    assert "perfetto" not in declared
    assert "export" in declared, "the trace export must survive"

    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "perfetto", "p.sqlite"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
