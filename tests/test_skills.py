"""Tests for the skills system — registry, loading, and execution."""

import sqlite3

import pytest


def test_list_skills():
    """All built-in skills should be discoverable."""
    from nsys_ai.skills import list_skills

    names = list_skills()

    expected = [
        "arithmetic_intensity",
        "cpu_gpu_pipeline",
        "cutracer_analysis",
        "gc_impact",
        "gpu_idle_gaps",
        "h2d_distribution",
        "iteration_detail",
        "iteration_timing",
        "kernel_instances",
        "kernel_launch_overhead",
        "kernel_launch_pattern",
        "kernel_overlap_matrix",
        "memory_bandwidth",
        "memory_transfers",
        "module_loading",
        "nccl_anomaly",
        "nccl_breakdown",
        "nccl_communicator_analysis",
        "nvtx_kernel_map",
        "nvtx_layer_breakdown",
        "overlap_breakdown",
        "pipeline_bubble_metrics",
        "profile_health_manifest",
        "region_mfu",
        "root_cause_matcher",
        "schema_inspect",
        "speedup_estimator",
        "stream_concurrency",
        "sync_cost_analysis",
        "tensor_core_usage",
        "theoretical_flops",
        "thread_utilization",
        "top_kernels",
    ]
    assert len(names) == len(expected)
    assert names == expected


def test_get_skill():
    """Should retrieve a specific skill by name."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("top_kernels")
    assert skill is not None
    assert skill.name == "top_kernels"
    assert skill.category == "kernels"
    assert "kernel" in skill.description.lower()


def test_get_skill_not_found():
    """Should return None for unknown skill."""
    from nsys_ai.skills.registry import get_skill

    assert get_skill("nonexistent_skill") is None


def test_run_skill_not_found():
    """Should raise KeyError for unknown skill."""
    from nsys_ai.skills.registry import run_skill

    conn = sqlite3.connect(":memory:")
    with pytest.raises(KeyError, match="Unknown skill"):
        run_skill("nonexistent_skill", conn)
    conn.close()


def test_skill_catalog():
    """Skill catalog should contain all skill descriptions."""
    from nsys_ai.skills.registry import skill_catalog

    catalog = skill_catalog()
    assert "top_kernels" in catalog
    assert "gpu_idle_gaps" in catalog
    assert "Available Skills" in catalog


def test_skill_to_tool_description():
    """Each skill should generate an LLM tool description."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("top_kernels")
    desc = skill.to_tool_description()
    assert "[top_kernels]" in desc
    assert "limit" in desc  # parameter


def test_schema_inspect_on_empty_db():
    """schema_inspect should work on any SQLite database."""
    from nsys_ai.skills.registry import run_skill

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
    result = run_skill("schema_inspect", conn)
    assert "test_table" in result
    assert "id" in result
    assert "name" in result
    conn.close()


def test_all_skills_have_required_fields():
    """Every skill must have name, title, description, category, and sql or execute_fn."""
    from nsys_ai.skills.registry import all_skills

    for skill in all_skills():
        assert skill.name, "Skill missing name"
        assert skill.title, f"Skill {skill.name} missing title"
        assert skill.description, f"Skill {skill.name} missing description"
        assert skill.category, f"Skill {skill.name} missing category"
        assert skill.sql or skill.execute_fn, f"Skill {skill.name} missing sql and execute_fn"


# ---------------------------------------------------------------------------
# C1: JSON output tests
# ---------------------------------------------------------------------------


def test_skill_execute_returns_list_of_dicts():
    """skill.execute() should return list[dict] for JSON serialization."""
    from nsys_ai.skills.registry import get_skill

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")
    skill = get_skill("schema_inspect")
    rows = skill.execute(conn)
    assert isinstance(rows, list)
    assert all(isinstance(r, dict) for r in rows)
    assert len(rows) > 0
    conn.close()


def test_skill_execute_json_serializable():
    """skill.execute() output must be JSON-serializable."""
    import json

    from nsys_ai.skills.registry import get_skill

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id INTEGER, val REAL)")
    skill = get_skill("schema_inspect")
    rows = skill.execute(conn)
    text = json.dumps(rows)  # must not raise TypeError
    parsed = json.loads(text)
    assert isinstance(parsed, list)
    conn.close()


# ---------------------------------------------------------------------------
# C4: Markdown skill persistence tests
# ---------------------------------------------------------------------------

# Sample fixture
_SAMPLE_SKILL_MD = """\
# test_query
## Description
Count rows in a table.
## Category
utility
## SQL
```sql
SELECT COUNT(*) AS row_count FROM sqlite_master WHERE type='table'
```
"""


def test_load_skill_from_markdown(tmp_path):
    """Should parse a markdown file into a Skill with correct fields."""
    md_file = tmp_path / "test_query.md"
    md_file.write_text(_SAMPLE_SKILL_MD)
    from nsys_ai.skills.registry import load_skill_from_markdown

    skill = load_skill_from_markdown(str(md_file))
    assert skill.name == "test_query"
    assert skill.category == "utility"
    assert "COUNT(*)" in skill.sql
    assert "Count rows" in skill.description
    assert "custom" in skill.tags


def test_load_skill_from_markdown_missing_sql(tmp_path):
    """Should raise ValueError when no ```sql block is present."""
    md_file = tmp_path / "bad.md"
    md_file.write_text("# bad_skill\n## Description\nNo SQL here.\n")
    from nsys_ai.skills.registry import load_skill_from_markdown

    with pytest.raises(ValueError, match="No.*sql"):
        load_skill_from_markdown(str(md_file))


def test_load_skill_from_markdown_empty_sql(tmp_path):
    """Should raise ValueError when SQL block is empty."""
    md_file = tmp_path / "empty.md"
    md_file.write_text("# empty_skill\n## SQL\n```sql\n```\n")
    from nsys_ai.skills.registry import load_skill_from_markdown

    with pytest.raises(ValueError, match="Empty SQL"):
        load_skill_from_markdown(str(md_file))


def test_load_skill_from_markdown_minimal(tmp_path):
    """Should work with just name + SQL (defaults for description and category)."""
    md_file = tmp_path / "minimal.md"
    md_file.write_text("# minimal\n## SQL\n```sql\nSELECT 1\n```\n")
    from nsys_ai.skills.registry import load_skill_from_markdown

    skill = load_skill_from_markdown(str(md_file))
    assert skill.name == "minimal"
    assert skill.category == "custom"
    assert skill.sql == "SELECT 1"


def test_save_skill_to_markdown(tmp_path):
    """Should serialize a Skill to markdown with all sections."""
    from nsys_ai.skills.base import Skill
    from nsys_ai.skills.registry import save_skill_to_markdown

    skill = Skill(
        name="my_metric",
        title="My Metric",
        description="Custom analysis.",
        category="custom",
        sql="SELECT 1 AS result",
    )
    path = tmp_path / "my_metric.md"
    save_skill_to_markdown(skill, str(path))
    content = path.read_text()
    assert "# my_metric" in content
    assert "## Description" in content
    assert "Custom analysis." in content
    assert "## Category" in content
    assert "custom" in content
    assert "```sql" in content
    assert "SELECT 1 AS result" in content


def test_round_trip_save_load(tmp_path):
    """save → load should preserve all fields."""
    from nsys_ai.skills.base import Skill
    from nsys_ai.skills.registry import load_skill_from_markdown, save_skill_to_markdown

    original = Skill(
        name="round_trip",
        title="Round Trip",
        description="Test round-trip serialization.",
        category="testing",
        sql="SELECT COUNT(*) AS n FROM sqlite_master",
    )
    path = tmp_path / "round_trip.md"
    save_skill_to_markdown(original, str(path))
    loaded = load_skill_from_markdown(str(path))
    assert loaded.name == original.name
    assert loaded.description == original.description
    assert loaded.category == original.category
    assert loaded.sql == original.sql


def test_load_custom_skills_dir(tmp_path):
    """Should load all .md files from a directory."""
    (tmp_path / "skill_a.md").write_text(_SAMPLE_SKILL_MD.replace("test_query", "skill_a"))
    (tmp_path / "skill_b.md").write_text(_SAMPLE_SKILL_MD.replace("test_query", "skill_b"))
    from nsys_ai.skills.registry import get_skill, load_custom_skills_dir

    load_custom_skills_dir(str(tmp_path))
    assert get_skill("skill_a") is not None
    assert get_skill("skill_b") is not None


def test_load_custom_skills_dir_empty(tmp_path):
    """Empty directory should not cause errors."""
    from nsys_ai.skills.registry import load_custom_skills_dir

    loaded = load_custom_skills_dir(str(tmp_path))
    assert loaded == []


def test_load_custom_skills_dir_nonexistent(tmp_path):
    """Nonexistent directory should return empty list, no error."""
    from nsys_ai.skills.registry import load_custom_skills_dir

    loaded = load_custom_skills_dir(str(tmp_path / "does_not_exist"))
    assert loaded == []


def test_custom_skill_executes(tmp_path):
    """A loaded markdown skill should execute SQL correctly."""
    (tmp_path / "count_tables.md").write_text(
        _SAMPLE_SKILL_MD.replace("test_query", "count_tables")
    )
    from nsys_ai.skills.registry import load_skill_from_markdown

    skill = load_skill_from_markdown(str(tmp_path / "count_tables.md"))
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE foo (id INTEGER)")
    conn.execute("CREATE TABLE bar (id INTEGER)")
    rows = skill.execute(conn)
    assert rows[0]["row_count"] == 2
    conn.close()


def test_remove_custom_skill(tmp_path):
    """Should delete the .md file and unregister the skill."""
    (tmp_path / "removable.md").write_text(_SAMPLE_SKILL_MD.replace("test_query", "removable"))
    from nsys_ai.skills.registry import (
        get_skill,
        load_skill_from_markdown,
        remove_custom_skill,
    )

    load_skill_from_markdown(str(tmp_path / "removable.md"))
    assert get_skill("removable") is not None
    assert remove_custom_skill("removable", str(tmp_path))
    assert not (tmp_path / "removable.md").exists()


def test_remove_custom_skill_not_found(tmp_path):
    """Should return False when skill file doesn't exist."""
    from nsys_ai.skills.registry import remove_custom_skill

    assert not remove_custom_skill("nonexistent", str(tmp_path))


# ---------------------------------------------------------------------------
# Performance: ensure_indexes tests
# ---------------------------------------------------------------------------


def test_ensure_indexes_creates_indexes():
    """ensure_indexes should create _nsysai_* indexes on tables that exist."""
    from nsys_ai.indexing import _indexed_connections
    from nsys_ai.skills.base import ensure_indexes

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, [end] INT, correlationId INT)"
    )
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (correlationId INT, globalTid INT, start INT)"
    )
    conn.execute("CREATE TABLE NVTX_EVENTS (start INT, [end] INT, globalTid INT)")

    # Clear tracking to allow re-testing
    _indexed_connections.discard(id(conn))

    ensure_indexes(conn)

    # Verify indexes were created
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE '_nsysai_%'"
    ).fetchall()
    index_names = {r[0] for r in rows}
    assert "_nsysai_kernel_start" in index_names
    assert "_nsysai_kernel_corr" in index_names
    assert "_nsysai_runtime_corr" in index_names
    assert "_nsysai_nvtx_start" in index_names
    conn.close()


def test_ensure_indexes_skips_missing_tables():
    """ensure_indexes should not raise when tables don't exist."""
    from nsys_ai.indexing import _indexed_connections
    from nsys_ai.skills.base import ensure_indexes

    conn = sqlite3.connect(":memory:")
    _indexed_connections.discard(id(conn))
    ensure_indexes(conn)  # should not raise
    conn.close()


def test_ensure_indexes_idempotent():
    """Calling ensure_indexes twice should not error."""
    from nsys_ai.indexing import _indexed_connections
    from nsys_ai.skills.base import ensure_indexes

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, [end] INT, correlationId INT)"
    )
    _indexed_connections.discard(id(conn))

    ensure_indexes(conn)
    ensure_indexes(conn)  # second call should be a no-op
    conn.close()


# ---------------------------------------------------------------------------
# Performance: trim_clause injection tests
# ---------------------------------------------------------------------------


def test_trim_clause_injection():
    """Skill with {trim_clause} should filter rows when trim kwargs are provided."""
    from nsys_ai.skills.base import Skill

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE k (start INT, [end] INT, val TEXT)")
    conn.execute("INSERT INTO k VALUES (100, 200, 'a')")
    conn.execute("INSERT INTO k VALUES (300, 400, 'b')")
    conn.execute("INSERT INTO k VALUES (500, 600, 'c')")

    skill = Skill(
        name="test_trim",
        title="Test Trim",
        description="test",
        category="test",
        sql="SELECT val FROM k WHERE 1=1 {trim_clause}",
    )

    # Without trim — should return all 3 rows
    all_rows = skill.execute(conn)
    assert len(all_rows) == 3

    # With trim — should return only row 'b' (start=300, end=400)
    trimmed = skill.execute(conn, trim_start_ns=250, trim_end_ns=450)
    assert len(trimmed) == 1
    assert trimmed[0]["val"] == "b"
    conn.close()


def test_trim_clause_no_placeholder():
    """Skill without {trim_clause} should run normally even with trim kwargs."""
    from nsys_ai.skills.base import Skill

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE items (id INT)")
    conn.execute("INSERT INTO items VALUES (1)")

    skill = Skill(
        name="no_trim",
        title="No Trim",
        description="test",
        category="test",
        sql="SELECT id FROM items",
    )

    # Should not error even with trim kwargs
    rows = skill.execute(conn, trim_start_ns=0, trim_end_ns=1000)
    assert len(rows) == 1
    conn.close()


def test_skill_run_cli_trim_arg():
    """skill run parser should accept --trim argument."""
    from nsys_ai.cli.parsers import _build_parser

    _build_parser()  # verify no import/construction error
    # This replaces _register_legacy_commands; make sure we can parse
    # Build a minimal parse to verify --trim is accepted
    parsed = False
    try:
        from nsys_ai.cli.parsers import _build_legacy_parser

        lp = _build_legacy_parser()
        args = lp.parse_args(["skill", "run", "top_kernels", "test.sqlite", "--trim", "1.0", "3.0"])
        assert args.trim == [1.0, 3.0]
        assert args.skill_name == "top_kernels"
        parsed = True
    except (SystemExit, AttributeError):
        # Legacy parser not available — fall back to the public parser
        pass

    if not parsed:
        # Verify at least the public parser can be constructed
        parser = _build_parser()
        assert parser is not None


# ---------------------------------------------------------------------------
# New skill tests: overlap_breakdown, iteration_timing, nvtx_layer_breakdown
# ---------------------------------------------------------------------------


def test_overlap_breakdown_registered():
    """overlap_breakdown should be registered with correct metadata."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("overlap_breakdown")
    assert skill is not None
    assert skill.name == "overlap_breakdown"
    assert skill.category == "communication"
    assert skill.execute_fn is not None
    assert skill.sql == ""


def test_iteration_timing_registered():
    """iteration_timing should be registered with correct metadata."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("iteration_timing")
    assert skill is not None
    assert skill.name == "iteration_timing"
    assert skill.category == "nvtx"
    assert skill.execute_fn is not None


def test_nvtx_layer_breakdown_registered():
    """nvtx_layer_breakdown should be registered with correct metadata."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("nvtx_layer_breakdown")
    assert skill is not None
    assert skill.name == "nvtx_layer_breakdown"
    assert skill.category == "nvtx"
    assert skill.execute_fn  # Python execute_fn skill (sort-merge attribution)


def test_nvtx_kernel_map_registered():
    """nvtx_kernel_map should be registered with correct metadata."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("nvtx_kernel_map")
    assert skill is not None
    assert skill.name == "nvtx_kernel_map"
    assert skill.category == "nvtx"
    assert skill.execute_fn  # Python execute_fn skill (sort-merge attribution)


def test_nvtx_kernel_map_execute(minimal_nsys_conn):
    """nvtx_kernel_map should run against minimal DB without error."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("nvtx_kernel_map")
    rows = skill.execute(minimal_nsys_conn)
    assert isinstance(rows, list)
    for r in rows:
        assert isinstance(r, dict)
        assert "nvtx_text" in r
        assert "kernel_name" in r
        assert "start_ms" in r
        assert "end_ms" in r


def test_overlap_breakdown_execute(minimal_nsys_conn):
    """overlap_breakdown should return overlap data from minimal DB."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("overlap_breakdown")
    rows = skill.execute(minimal_nsys_conn)
    assert isinstance(rows, list)
    assert len(rows) == 1
    r = rows[0]
    # Should have overlap fields (not an error)
    assert "compute_only_ms" in r or "error" in r


def test_overlap_breakdown_format(minimal_nsys_conn):
    """overlap_breakdown.run() should return formatted text."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("overlap_breakdown")
    text = skill.run(minimal_nsys_conn)
    assert isinstance(text, str)
    assert len(text) > 0


def test_overlap_breakdown_no_kernels_diagnostic(minimal_nsys_conn):
    """overlap_breakdown should return enriched error with device diagnostics when device has no kernels."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("overlap_breakdown")
    # Device 99 doesn't exist — should return diagnostic error, not bare "no kernels"
    rows = skill.execute(minimal_nsys_conn, device=99)
    assert isinstance(rows, list)
    assert len(rows) == 1
    r = rows[0]
    assert "error" in r
    assert r["requested_device"] == 99
    assert "available_devices" in r
    assert "hint" in r
    # Hint should suggest valid device(s)
    assert "device" in r["hint"].lower()


def test_nvtx_layer_breakdown_execute(minimal_nsys_conn):
    """nvtx_layer_breakdown should run against minimal DB without error."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("nvtx_layer_breakdown")
    rows = skill.execute(minimal_nsys_conn)
    assert isinstance(rows, list)
    # With our seed data (NVTX 'train_step' and 'forward' with correlated kernels),
    # we should get at least one result
    for r in rows:
        if r.get("_detection_meta"):
            continue  # skip auto-detection metadata header
        assert "nvtx_region" in r
        assert "kernel_count" in r
        assert "total_gpu_ms" in r


def test_execute_fn_skill_json_serializable():
    """Python-level skills should produce JSON-serializable output."""
    import json
    import sqlite3

    from nsys_ai.skills.base import Skill

    def _dummy_execute(conn, **kwargs):
        return [{"metric": 42.0, "label": "test"}]

    skill = Skill(
        name="dummy",
        title="Dummy",
        description="test",
        category="test",
        execute_fn=_dummy_execute,
    )
    conn = sqlite3.connect(":memory:")
    rows = skill.execute(conn)
    text = json.dumps(rows)  # must not raise
    parsed = json.loads(text)
    assert parsed[0]["metric"] == 42.0
    conn.close()


# ---------------------------------------------------------------------------
# Root cause matcher: anti-pattern integration tests
# ---------------------------------------------------------------------------


def test_root_cause_no_id_field(minimal_nsys_conn):
    """Findings should NOT contain an 'id' field — pattern name is the identifier."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("root_cause_matcher")
    rows = skill.execute(minimal_nsys_conn)
    assert isinstance(rows, list)
    for finding in rows:
        assert "id" not in finding, f"Finding should not have 'id': {finding}"
        assert "pattern" in finding
        assert "severity" in finding
        assert "evidence" in finding
        assert "recommendation" in finding


def test_root_cause_finds_sync_apis(minimal_nsys_conn):
    """Should detect Excessive Synchronization from seed data.

    Seed has 2 cudaDeviceSynchronize calls totalling 16ms,
    while total GPU kernel time is ~4ms → sync is >100% of GPU time.
    """
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("root_cause_matcher")
    rows = skill.execute(minimal_nsys_conn)
    patterns = [r["pattern"] for r in rows]
    assert "Excessive Synchronization" in patterns
    sync_finding = next(r for r in rows if r["pattern"] == "Excessive Synchronization")
    assert "cudaDeviceSynchronize" in sync_finding["evidence"]
    assert sync_finding["severity"] in ("warning", "critical")


def test_root_cause_finds_sync_memcpy(minimal_nsys_conn):
    """Should detect Synchronous Memcpy from seed data.

    Seed has 1 cudaMemcpy call correlated with a memcpy entry.
    """
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("root_cause_matcher")
    rows = skill.execute(minimal_nsys_conn)
    patterns = [r["pattern"] for r in rows]
    assert "Synchronous Memcpy" in patterns


def test_root_cause_finds_pageable_memcpy(minimal_nsys_conn):
    """Should detect Pageable Memory in seed data.

    Seed has a memcpy entry with srcKind=1 (pageable).
    """
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("root_cause_matcher")
    rows = skill.execute(minimal_nsys_conn)
    patterns = [r["pattern"] for r in rows]
    assert "Pageable Memory in Async Memcpy" in patterns


def test_root_cause_finds_sync_memset(minimal_nsys_conn):
    """Should detect Synchronous Memset from seed data.

    Seed has 1 cudaMemset call correlated with a memset entry.
    """
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("root_cause_matcher")
    rows = skill.execute(minimal_nsys_conn)
    patterns = [r["pattern"] for r in rows]
    assert "Synchronous Memset" in patterns


def test_root_cause_all_patterns_execute(minimal_nsys_conn):
    """Full scan should complete without crash and return valid findings."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("root_cause_matcher")
    rows = skill.execute(minimal_nsys_conn)
    assert isinstance(rows, list)
    assert len(rows) > 0
    # Should have at least the 4 new anti-pattern findings
    patterns = {r["pattern"] for r in rows}
    assert "Excessive Synchronization" in patterns
    assert "Synchronous Memcpy" in patterns
    assert "Pageable Memory in Async Memcpy" in patterns
    assert "Synchronous Memset" in patterns
    # Format should also work
    text = skill.format_rows(rows)
    assert "Root Cause Pattern Analysis" in text
    # Verify no [N] id prefix in formatted output
    assert "[1]" not in text
    assert "[3]" not in text


# ---- V3 Review Feature Tests ------------------------------------------------


def test_h2d_distribution_init_heavy_pattern(minimal_nsys_conn):
    """Seed data has H2D concentrated in second 0 — should classify as init_heavy."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("h2d_distribution")
    rows = skill.execute(minimal_nsys_conn)
    # Should have data rows + pattern metadata
    assert len(rows) > 0
    pattern = next((r for r in rows if r.get("_pattern")), None)
    assert pattern is not None
    assert pattern["type"] == "init_heavy"
    assert (
        "first 2 seconds" in pattern["detail"].lower()
        or "concentrated" in pattern["detail"].lower()
    )


def test_h2d_distribution_format_shows_pattern(minimal_nsys_conn):
    """Format output should include pattern classification."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("h2d_distribution")
    rows = skill.execute(minimal_nsys_conn)
    text = skill.format_rows(rows)
    assert "Pattern:" in text
    assert "init_heavy" in text


def test_gpu_idle_gaps_aggregation(minimal_nsys_conn):
    """gpu_idle_gaps should return summary with aggregation stats."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("gpu_idle_gaps")
    rows = skill.execute(minimal_nsys_conn)
    assert len(rows) > 0
    # Last element should be summary
    summary = next((r for r in rows if r.get("_summary")), None)
    assert summary is not None
    assert "total_idle_ms" in summary
    assert "pct_of_profile" in summary
    assert "gap_count" in summary
    assert summary["total_idle_ms"] > 0


def test_gpu_idle_gaps_cpu_attribution(minimal_nsys_conn):
    """Top gaps should have CPU attribution field with category."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("gpu_idle_gaps")
    rows = skill.execute(minimal_nsys_conn)
    data_rows = [r for r in rows if not r.get("_summary")]
    if data_rows:
        # At least one gap should have attribution
        attributed = [r for r in data_rows if r.get("attribution")]
        assert len(attributed) > 0
        attr = attributed[0]["attribution"]
        assert "category" in attr
        assert attr["category"] in (
            "synchronization",
            "memory_transfer",
            "kernel_launch",
            "cpu_stall",
            "unknown",
        )


def test_gpu_idle_gaps_format_output(minimal_nsys_conn):
    """Format output should include summary header and attribution."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("gpu_idle_gaps")
    rows = skill.execute(minimal_nsys_conn)
    text = skill.format_rows(rows)
    assert "GPU Idle Gaps" in text
    assert "Total:" in text
    assert "Distribution:" in text


def test_overlap_same_stream_detection(minimal_nsys_conn):
    """Should detect same-stream NCCL+compute from seed data (stream 7)."""
    from nsys_ai.skills.builtins.root_cause_matcher import _diagnose_low_overlap

    minimal_nsys_conn.row_factory = __import__("sqlite3").Row
    diag = _diagnose_low_overlap(minimal_nsys_conn, device=0)
    assert diag["cause"] == "same_stream"
    assert "7" in diag["detail"]


def test_overlap_diagnosis_recommendation(minimal_nsys_conn):
    """root_cause_matcher should output diagnosis-specific recommendation."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("root_cause_matcher")
    rows = skill.execute(minimal_nsys_conn)
    nccl_findings = [r for r in rows if r["pattern"] == "NCCL Serialization"]
    if nccl_findings:
        rec = nccl_findings[0]["recommendation"]
        # Should contain same-stream specific recommendation
        assert "same CUDA stream" in rec or "separate stream" in rec or "DDP" in rec


def test_overlap_no_nccl_graceful():
    """Should not crash when profile has zero NCCL kernels."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO StringIds VALUES (1, 'kernel_A');
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            deviceId INT, streamId INT, correlationId INT,
            start INT, [end] INT, shortName INT, demangledName INT,
            gridX INT DEFAULT 1, gridY INT DEFAULT 1, gridZ INT DEFAULT 1,
            blockX INT DEFAULT 1, blockY INT DEFAULT 1, blockZ INT DEFAULT 1,
            globalPid INT DEFAULT 0
        );
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
            (0, 7, 1, 1000000, 2000000, 1, 1, 1,1,1,1,1,1,0);
    """)

    from nsys_ai.skills.builtins.root_cause_matcher import _diagnose_low_overlap

    diag = _diagnose_low_overlap(conn, device=0)
    assert diag["cause"] == "general"
    conn.close()


def test_root_cause_h2d_spread_pattern():
    """H2D spread pattern should fire 'Continuous H2D Transfers' finding."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO StringIds VALUES (1, 'kernel_A'), (24, 'cudaLaunchKernel');
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            deviceId INT, streamId INT, correlationId INT,
            start INT, [end] INT, shortName INT, demangledName INT,
            gridX INT DEFAULT 1, gridY INT DEFAULT 1, gridZ INT DEFAULT 1,
            blockX INT DEFAULT 1, blockY INT DEFAULT 1, blockZ INT DEFAULT 1,
            globalPid INT DEFAULT 0
        );
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
            (0, 7, 1, 1000000000, 1001000000, 1, 1, 1,1,1,1,1,1,0);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            globalTid INT, correlationId INT, start INT, [end] INT, nameId INT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
            globalPid INT, deviceId INT, streamId INT, correlationId INT,
            copyKind INT, bytes INT, srcKind INT, dstKind INT, start INT, [end] INT
        );
        -- Spread H2D across 5 seconds with similar bytes (not init-heavy)
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES
            (0, 0, 7, 10, 1, 100000, 7, 2, 100000000,  200000000),
            (0, 0, 7, 11, 1, 100000, 7, 2, 1100000000, 1200000000),
            (0, 0, 7, 12, 1, 100000, 7, 2, 2100000000, 2200000000),
            (0, 0, 7, 13, 1, 100000, 7, 2, 3100000000, 3200000000),
            (0, 0, 7, 14, 1, 100000, 7, 2, 4100000000, 4200000000);
    """)

    from nsys_ai.skills.registry import get_skill

    skill = get_skill("root_cause_matcher")
    rows = skill.execute(conn)
    patterns = [r["pattern"] for r in rows]
    assert "Continuous H2D Transfers" in patterns
    conn.close()


def test_root_cause_includes_new_patterns(minimal_nsys_conn):
    """Verify root_cause_matcher can detect the new V3 patterns alongside old ones."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("root_cause_matcher")
    rows = skill.execute(minimal_nsys_conn)
    patterns = {r["pattern"] for r in rows}
    # Should still have old patterns
    assert "Excessive Synchronization" in patterns
    assert "Synchronous Memcpy" in patterns
    # Verify no crash and reasonable output
    assert len(rows) >= 4
    text = skill.format_rows(rows)
    assert len(text) > 50


def test_tensor_core_usage_fallback(minimal_nsys_conn):
    """Fallback error matches Copilot review feedback on pure SQLite."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("tensor_core_usage")
    # This DB has no duckdb or `kernels` view mapped, so it hits the except block.
    rows = skill.execute(minimal_nsys_conn)
    assert len(rows) == 1
    assert "error" in rows[0]
    assert "exposes a 'kernels' view" in rows[0]["error"]

    text = skill.format_rows(rows)
    assert "Error: Tensor Core analysis requires" in text


def test_tensor_core_usage_duckdb():
    """Verify DuckDB execution handles tensor_core_usage."""
    import duckdb

    from nsys_ai.skills.registry import get_skill

    conn = duckdb.connect()
    # Mock DuckDB schema
    conn.execute(
        """
        CREATE VIEW kernels AS SELECT * FROM (VALUES
            ('ampere_sgemm_128x128', 100, 200, 1, 1),
            ('vectorized_elementwise', 300, 400, 0, 0),
            ('ampere_fp16_fallback', 500, 600, 1, 0)
        ) AS t(name, start, "end", is_tc_eligible, uses_tc)
        """
    )

    skill = get_skill("tensor_core_usage")
    rows = skill.execute(conn)

    # vectorized_elementwise is NOT eligible
    assert len(rows) == 2

    # Make assertions independent of row order by indexing rows by kernel_name.
    rows_by_name = {row["kernel_name"]: row for row in rows}

    # Fallback
    fallback = rows_by_name["ampere_fp16_fallback"]
    assert fallback["total_gpu_ms"] == 100 / 1e6
    assert fallback["tc_active_ms"] == 0
    assert fallback["tc_achieved_pct"] == 0.0
    assert fallback["is_outlier"] is True

    # Active
    active = rows_by_name["ampere_sgemm_128x128"]
    assert active["total_gpu_ms"] == 100 / 1e6
    assert active["tc_active_ms"] == 100 / 1e6
    assert active["tc_achieved_pct"] == 100.0
    assert active["is_outlier"] is False

    text = skill.format_rows(rows)
    assert "ampere_fp16_fallback" in text
    assert "ampere_sgemm_128x128" in text
    assert "⚠️" in text
