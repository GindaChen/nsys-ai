"""Every non-Python file the package ships must be declared as package data.

`persona.py` reads `persona.md` at module import time, and `agent/*.md` was not
in `[tool.setuptools.package-data]`. Source installs worked because the file was
simply there; the wheel could not import `nsys_ai.agent.persona` at all:

    FileNotFoundError: .../site-packages/nsys_ai/agent/persona.md

Nothing caught it, because every test runs against the source tree. This walks
the package instead and checks each data file against the declared globs.
"""

from __future__ import annotations

import ast
import fnmatch
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "nsys_ai"

#: Extensions that are build or editor droppings rather than shipped data.
_IGNORED_SUFFIXES = {".pyc", ".pyo", ".so", ".pyd"}


def _declared_globs() -> list[str]:
    """Read the package-data globs without a TOML parser.

    ``tomllib`` is 3.11+, and this repo supports 3.10. The value is a literal
    list of strings in a known section, so a targeted match is enough — and it
    fails loudly rather than silently returning nothing if the section moves.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(
        r"^\[tool\.setuptools\.package-data\]\s*$.*?^nsys_ai\s*=\s*(\[.*?\])",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match, "could not find [tool.setuptools.package-data] nsys_ai in pyproject.toml"
    return ast.literal_eval(match.group(1))


def test_every_shipped_data_file_is_declared_as_package_data():
    globs = _declared_globs()
    undeclared = []
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file() or path.suffix == ".py":
            continue
        if path.suffix in _IGNORED_SUFFIXES or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(PACKAGE).as_posix()
        if not any(fnmatch.fnmatch(relative, pattern) for pattern in globs):
            undeclared.append(relative)
    assert not undeclared, (
        "these files ship in the source tree but are not in "
        "[tool.setuptools.package-data]; a wheel will not contain them:\n  "
        + "\n  ".join(undeclared)
        + f"\ndeclared globs: {globs}"
    )


def test_the_agent_persona_is_readable_the_way_persona_py_reads_it():
    """Guards the specific import-time read, not just the declaration."""
    from nsys_ai.agent import persona

    assert persona.SYSTEM_PROMPT.strip(), "persona.md loaded but empty"
    assert "{skill_catalog}" in persona.SYSTEM_PROMPT
    assert persona.build_system_prompt().strip()
