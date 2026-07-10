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
    assert meta["profile_id"].startswith("nsys1:")
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
