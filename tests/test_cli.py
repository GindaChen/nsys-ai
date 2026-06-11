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
        "agent-guide",
        "info",
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

    # 'agent-guide' is public, but 'agent' should be hidden
    assert ",agent," not in usage_text
    assert ",agent}" not in usage_text


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
