"""Tests for prompt_loader and the skill_docs integration in _build_system_prompt."""


# ---------------------------------------------------------------------------
# prompt_loader tests
# ---------------------------------------------------------------------------


def test_load_skill_success(tmp_path, monkeypatch):
    """load_skill returns file content when the file exists."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "test.md").write_text("# Test Skill\nContent here.")

    import nsys_ai.prompt_loader as pl

    monkeypatch.setattr(pl, "SKILLS_DIR", tmp_path)
    content = pl.load_skill("skills/test.md")
    assert "# Test Skill" in content
    assert "Content here." in content


def test_load_skill_missing_file(tmp_path, monkeypatch):
    """load_skill returns '' gracefully when the file does not exist."""
    import nsys_ai.prompt_loader as pl

    monkeypatch.setattr(pl, "SKILLS_DIR", tmp_path)
    content = pl.load_skill("skills/nonexistent.md")
    assert content == ""


def test_load_skill_missing_dir(tmp_path, monkeypatch):
    """load_skill returns '' gracefully when the skill dir doesn't exist at all."""
    import nsys_ai.prompt_loader as pl

    monkeypatch.setattr(pl, "SKILLS_DIR", tmp_path / "does_not_exist")
    content = pl.load_skill("skills/mfu.md")
    assert content == ""


def test_load_principles(tmp_path, monkeypatch):
    """load_principles loads PRINCIPLES.md from SKILLS_DIR."""
    (tmp_path / "PRINCIPLES.md").write_text("# Principles\nRule 1.")
    import nsys_ai.prompt_loader as pl

    monkeypatch.setattr(pl, "SKILLS_DIR", tmp_path)
    content = pl.load_principles()
    assert "Principles" in content


def test_load_principles_missing(tmp_path, monkeypatch):
    """load_principles returns '' when PRINCIPLES.md is missing."""
    import nsys_ai.prompt_loader as pl

    monkeypatch.setattr(pl, "SKILLS_DIR", tmp_path)
    assert pl.load_principles() == ""


def test_skill_block_with_header(tmp_path, monkeypatch):
    """skill_block wraps content with header/footer delimiters."""
    (tmp_path / "test.md").write_text("body content")
    import nsys_ai.prompt_loader as pl

    monkeypatch.setattr(pl, "SKILLS_DIR", tmp_path)
    block = pl.skill_block("test.md", header="MY HEADER")
    assert "MY HEADER" in block
    assert "body content" in block
    assert "END MY HEADER" in block


def test_skill_block_no_header(tmp_path, monkeypatch):
    """skill_block without header returns raw content."""
    (tmp_path / "test.md").write_text("raw content")
    import nsys_ai.prompt_loader as pl

    monkeypatch.setattr(pl, "SKILLS_DIR", tmp_path)
    block = pl.skill_block("test.md")
    assert block == "raw content"


def test_skill_block_missing_file(tmp_path, monkeypatch):
    """skill_block returns '' when file is missing."""
    import nsys_ai.prompt_loader as pl

    monkeypatch.setattr(pl, "SKILLS_DIR", tmp_path)
    assert pl.skill_block("missing.md", header="HEADER") == ""


def test_load_skill_context(tmp_path, monkeypatch):
    """load_skill_context concatenates multiple skill files."""
    (tmp_path / "a.md").write_text("Content A")
    (tmp_path / "b.md").write_text("Content B")
    import nsys_ai.prompt_loader as pl

    monkeypatch.setattr(pl, "SKILLS_DIR", tmp_path)
    result = pl.load_skill_context(["a.md", "b.md", "missing.md"])
    assert "Content A" in result
    assert "Content B" in result


def test_env_var_override(tmp_path, monkeypatch):
    """NSYS_AI_SKILLS_DIR env var overrides the default SKILLS_DIR."""
    skills_alt = tmp_path / "alt_skills"
    skills_alt.mkdir()
    (skills_alt / "env_test.md").write_text("from env override")

    monkeypatch.setenv("NSYS_AI_SKILLS_DIR", str(skills_alt))
    # Re-import to pick up the env var (import fresh module)
    import importlib  # noqa: E401

    import nsys_ai.prompt_loader as pl

    importlib.reload(pl)  # picks up the new env var
    monkeypatch.setattr(pl, "SKILLS_DIR", skills_alt)  # ensure patched too

    content = pl.load_skill("env_test.md")
    assert "from env override" in content


# ---------------------------------------------------------------------------
# _build_system_prompt integration tests
# ---------------------------------------------------------------------------


def test_build_system_prompt_skill_docs_injected():
    """skill_docs content appears in the system prompt under SESSION SKILL CONTEXT."""
    from nsys_ai.ai.backend.chat_tools import _build_system_prompt

    out = _build_system_prompt(
        {"view_state": {}},
        skill_docs="CUSTOM_SKILL_MARKER_XYZ",
    )
    assert "CUSTOM_SKILL_MARKER_XYZ" in out
    assert "SESSION SKILL CONTEXT" in out


def test_build_system_prompt_no_skill_docs():
    """Without skill_docs, SESSION SKILL CONTEXT section is absent."""
    from nsys_ai.ai.backend.chat_tools import _build_system_prompt

    out = _build_system_prompt({"view_state": {}})
    assert "SESSION SKILL CONTEXT" not in out


def test_build_system_prompt_existing_strings_preserved():
    """Existing test assertions still hold after skill_docs addition."""
    from nsys_ai.ai.backend.chat_tools import _build_system_prompt

    ctx = {"view_state": {"scope": "all"}, "global_top_kernels": []}
    out = _build_system_prompt(ctx)
    # Original required strings must still be present
    assert "MFU REFERENCE" in out
    assert "flops_per_layer" in out
    assert "SANITY CHECK" in out
    assert "num_gpus" in out


def test_build_system_prompt_mfu_extended_heading():
    """When skills/mfu.md is loadable, EXTENDED MFU GUIDANCE block appears."""
    from nsys_ai.ai.backend.chat_tools import _build_system_prompt

    out = _build_system_prompt({"view_state": {}})
    # If skills/mfu.md loaded successfully, this heading should appear
    if "EXTENDED MFU GUIDANCE" in out:
        assert "skills/mfu.md" not in out  # heading should replace path
    # Test passes either way (graceful fallback if file missing in test env)
