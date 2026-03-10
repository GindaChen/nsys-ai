"""
prompt_loader.py — Load agent skill markdown files into the system prompt.

Design goals:
- Graceful degradation: returns '' when files are not found (packaged / offline).
- No caching: each call reads from disk so edits take effect immediately in dev.
- Pure stdlib: pathlib only, no external dependencies.
- Override via env var NSYS_AI_SKILLS_DIR (useful in tests and packaged installs).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_log = logging.getLogger(__name__)

# Resolve the default skills directory relative to this file:
#   src/nsys_ai/prompt_loader.py  →  ../../../docs/agent_skills
_DEFAULT_SKILLS_DIR = Path(__file__).parent.parent.parent / "docs" / "agent_skills"

# Allow override via environment variable for testing and packaged installs.
SKILLS_DIR: Path = Path(os.environ.get("NSYS_AI_SKILLS_DIR", str(_DEFAULT_SKILLS_DIR)))


def load_skill(relative_path: str) -> str:
    """Load a skill file relative to SKILLS_DIR.

    Args:
        relative_path: e.g. "skills/mfu.md" or "PRINCIPLES.md"

    Returns:
        File content as a string, or '' if the file cannot be read.
    """
    try:
        path = SKILLS_DIR / relative_path
        content = path.read_text(encoding="utf-8")
        _log.debug("prompt_loader: loaded %s (%d chars)", path, len(content))
        return content
    except (OSError, IOError) as exc:
        _log.debug("prompt_loader: could not load '%s': %s", relative_path, exc)
        return ""


def load_principles() -> str:
    """Load PRINCIPLES.md.

    Returns:
        File content, or '' if not found.
    """
    return load_skill("PRINCIPLES.md")


def skill_block(relative_path: str, header: str = "") -> str:
    """Load a skill file and wrap it in a named delimiter block.

    Args:
        relative_path: path relative to SKILLS_DIR.
        header: Optional label for the delimiter (e.g. "MFU EXTENDED").
                If empty, content is returned unwrapped.

    Returns:
        Wrapped content string, or '' if the file is not found.
    """
    content = load_skill(relative_path)
    if not content:
        return ""
    if not header:
        return content
    sep = "=" * 40
    return f"{sep}\n{header}\n{sep}\n{content}\n{sep}\nEND {header}\n{sep}"


def load_skill_context(skill_names: list[str]) -> str:
    """Load and concatenate multiple skill files, separated by blank lines.

    Args:
        skill_names: list of relative paths, e.g. ["skills/mfu.md", "skills/triage.md"]

    Returns:
        Concatenated content of all found files, or '' if none found.
    """
    parts = []
    for name in skill_names:
        content = load_skill(name)
        if content:
            parts.append(content.strip())
    return "\n\n".join(parts)
