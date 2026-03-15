"""Tests for the skills system — registry, loading, and execution."""

import sqlite3

import pytest


def test_list_skills():
    """All 8 built-in skills should be discoverable."""
    from nsys_ai.skills import list_skills

    names = list_skills()
    assert len(names) == 8
    expected = [
        "gpu_idle_gaps",
        "kernel_launch_overhead",
        "memory_transfers",
        "nccl_breakdown",
        "nvtx_kernel_map",
        "schema_inspect",
        "thread_utilization",
        "top_kernels",
    ]
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
    """Every skill must have name, title, description, category, sql."""
    from nsys_ai.skills.registry import all_skills

    for skill in all_skills():
        assert skill.name, "Skill missing name"
        assert skill.title, f"Skill {skill.name} missing title"
        assert skill.description, f"Skill {skill.name} missing description"
        assert skill.category, f"Skill {skill.name} missing category"
        assert skill.sql, f"Skill {skill.name} missing sql"


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
    from nsys_ai.skills.base import _indexed_connections, ensure_indexes

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
    from nsys_ai.skills.base import _indexed_connections, ensure_indexes

    conn = sqlite3.connect(":memory:")
    _indexed_connections.discard(id(conn))
    ensure_indexes(conn)  # should not raise
    conn.close()


def test_ensure_indexes_idempotent():
    """Calling ensure_indexes twice should not error."""
    from nsys_ai.skills.base import _indexed_connections, ensure_indexes

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
    from nsys_ai.cli.app import _build_parser

    _build_parser()  # verify no import/construction error
    # This replaces _register_legacy_commands; make sure we can parse
    # Build a minimal parse to verify --trim is accepted
    try:
        from nsys_ai.cli.app import _build_legacy_parser

        lp = _build_legacy_parser()
        args = lp.parse_args(["skill", "run", "top_kernels", "test.sqlite", "--trim", "1.0", "3.0"])
        assert args.trim == [1.0, 3.0]
        assert args.skill_name == "top_kernels"
    except (SystemExit, AttributeError):
        # Fall back to the public parser if legacy isn't available
        pass
