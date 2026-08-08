"""Basic smoke tests for nsys-ai package."""

import subprocess
import sys


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
    """loop subcommand should expose before/after workflow inputs."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "loop", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "before" in result.stdout
    assert "--after" in result.stdout
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
