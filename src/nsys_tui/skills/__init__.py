"""
nsys_tui.skills — Standardized SQL query skills for GPU profile analysis.

Skills are self-contained analysis units: each bundles a SQL query template,
parameter definitions, result formatting, and documentation. They work
standalone (no LLM required) and serve as the agent's toolbox.

Public API:
    list_skills()          → list of all registered skill names
    get_skill(name)        → Skill object by name
    run_skill(name, conn)  → execute a skill against a SQLite connection
"""
from .registry import get_skill, list_skills, run_skill

__all__ = ["list_skills", "get_skill", "run_skill"]
