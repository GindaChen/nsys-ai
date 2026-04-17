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
        "export",
        "agent-guide",
        "info",
        "skill",
        "evidence",
        "cutracer",
    ]:
        assert cmd in result.stdout, f"Missing subcommand: {cmd}"

    # Legacy command names should be hidden from top-level help.
    usage_line = result.stdout.splitlines()[0]
    for hidden in ["summary", "overlap", "analyze"]:
        assert hidden not in usage_line

    # 'agent-guide' is public, but 'agent' should be hidden
    assert ",agent," not in usage_line
    assert ",agent}" not in usage_line


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
