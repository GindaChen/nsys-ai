"""
registry.py — Skill auto-discovery and lookup.

Walks the builtins/ directory for Python files that export a SKILL constant.
Also supports loading skills from external directories.
"""
import importlib
import pkgutil
import sqlite3

from .base import Skill

# Global registry
_SKILLS: dict[str, Skill] = {}
_LOADED = False


def _load_builtins():
    """Import all modules in nsys_tui.skills.builtins and register their SKILL."""
    global _LOADED
    if _LOADED:
        return

    from . import builtins
    for _importer, modname, _ispkg in pkgutil.iter_modules(builtins.__path__):
        mod = importlib.import_module(f".builtins.{modname}", package="nsys_tui.skills")
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
