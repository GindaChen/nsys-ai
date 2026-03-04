"""Tests for the skills system — registry, loading, and execution."""
import sqlite3

import pytest


def test_list_skills():
    """All 8 built-in skills should be discoverable."""
    from nsys_ai.skills import list_skills
    names = list_skills()
    assert len(names) == 8
    expected = [
        "gpu_idle_gaps", "kernel_launch_overhead", "memory_transfers",
        "nccl_breakdown", "nvtx_kernel_map", "schema_inspect",
        "thread_utilization", "top_kernels",
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
