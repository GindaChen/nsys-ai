"""
registry.py — Skill auto-discovery and lookup.

Walks the builtins/ directory for Python files that export a SKILL constant.
Also supports loading skills from markdown files in custom directories.
"""

import importlib
import logging
import pkgutil
import re
import sqlite3
from pathlib import Path

from .base import Skill

_log = logging.getLogger(__name__)

# Global registry
_SKILLS: dict[str, Skill] = {}
_LOADED = False


def _load_builtins():
    """Import all modules in nsys_ai.skills.builtins and register their SKILL."""
    global _LOADED
    if _LOADED:
        return

    from . import builtins

    for _importer, modname, _ispkg in pkgutil.iter_modules(builtins.__path__):
        mod = importlib.import_module(f".builtins.{modname}", package="nsys_ai.skills")
        skill = getattr(mod, "SKILL", None)
        if isinstance(skill, Skill):
            _SKILLS[skill.name] = skill

    _LOADED = True


def register(skill: Skill):
    """Register a skill manually (for user-contributed skills)."""
    _SKILLS[skill.name] = skill


def list_skills() -> list[str]:
    """Return sorted list of all registered skill names."""
    _load_builtins()
    return sorted(_SKILLS.keys())


def get_skill(name: str) -> Skill | None:
    """Look up a skill by name. Returns None if not found."""
    _load_builtins()
    return _SKILLS.get(name)


def all_skills() -> list[Skill]:
    """Return all registered Skill objects."""
    _load_builtins()
    return sorted(_SKILLS.values(), key=lambda s: s.name)


def run_skill(name: str, conn: sqlite3.Connection, **kwargs) -> str:
    """Look up and run a skill, returning formatted text."""
    skill = get_skill(name)
    if not skill:
        available = ", ".join(list_skills())
        raise KeyError(f"Unknown skill '{name}'. Available: {available}")
    return skill.run(conn, **kwargs)


def skill_catalog() -> str:
    """Return the full skill catalog as text (for agent system prompts)."""
    _load_builtins()
    lines = ["## Available Skills", ""]
    for skill in all_skills():
        lines.append(skill.to_tool_description())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown skill persistence
# ---------------------------------------------------------------------------


def load_skill_from_markdown(path: str) -> Skill:
    """Load a skill defined in markdown format and register it.

    Expected format::

        # skill_name
        ## Description
        What this skill analyzes.
        ## Category
        kernels
        ## SQL
        ```sql
        SELECT ... FROM ...
        ```

    Returns the created Skill object (also registered in the global registry).

    Raises:
        ValueError: if the markdown does not contain a ```sql code block.
    """
    text = Path(path).read_text(encoding="utf-8")

    # Parse name from first H1
    name_match = re.search(r"^# (\w+)", text, re.MULTILINE)
    name = name_match.group(1) if name_match else Path(path).stem

    # Parse description (text between ## Description and next ##)
    desc_match = re.search(r"## Description\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL)
    description = desc_match.group(1).strip() if desc_match else ""

    # Parse category
    cat_match = re.search(r"## Category\s*\n(\w+)", text)
    category = cat_match.group(1) if cat_match else "custom"

    # Parse SQL from fenced code block
    sql_match = re.search(r"```sql\s*\n(.*?)```", text, re.DOTALL)
    if not sql_match:
        raise ValueError(f"No ```sql code block found in {path}")
    sql = sql_match.group(1).strip()
    if not sql:
        raise ValueError(f"Empty SQL block in {path}")

    skill = Skill(
        name=name,
        title=name.replace("_", " ").title(),
        description=description,
        category=category,
        sql=sql,
        tags=["custom", "runtime"],
    )
    register(skill)
    _log.debug("Loaded custom skill '%s' from %s", name, path)
    return skill


def save_skill_to_markdown(skill: Skill, path: str) -> None:
    """Serialize a Skill to a markdown file.

    Creates parent directories if needed. The output follows the same
    format that ``load_skill_from_markdown`` expects.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {skill.name}",
        "",
        "## Description",
        "",
        skill.description,
        "",
        "## Category",
        "",
        skill.category,
        "",
        "## SQL",
        "",
        "```sql",
        skill.sql,
        "```",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    _log.debug("Saved skill '%s' to %s", skill.name, path)


def load_custom_skills_dir(dir_path: str) -> list[Skill]:
    """Scan a directory for ``*.md`` files and load each as a skill.

    Files that fail to parse are logged and skipped (no exception raised).

    Returns:
        List of successfully loaded Skill objects.
    """
    _load_builtins()
    loaded: list[Skill] = []
    p = Path(dir_path)
    if not p.is_dir():
        _log.debug("Custom skills dir does not exist: %s", dir_path)
        return loaded
    for md_file in sorted(p.glob("*.md")):
        try:
            skill = load_skill_from_markdown(str(md_file))
            loaded.append(skill)
        except (ValueError, OSError) as exc:
            _log.debug("Skipping %s: %s", md_file, exc)
    return loaded


def remove_custom_skill(name: str, dir_path: str) -> bool:
    """Delete a custom skill's markdown file from the given directory.

    Returns True if the file was found and deleted, False otherwise.
    """
    target = Path(dir_path) / f"{name}.md"
    if target.exists():
        target.unlink()
        if name in _SKILLS:
            del _SKILLS[name]
        _log.debug("Removed custom skill '%s' from %s", name, dir_path)
        return True
    return False
