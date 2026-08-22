"""Small guards for the user-facing documentation front doors."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_points_new_users_to_the_documentation_index_and_entry_pages():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required = (
        "docs/README.md",
        "docs/user/migrating-to-0.3.0.md",
        "docs/user/troubleshooting.md",
        "docs/user/profile-inputs.md",
        "docs/user-guide.md",
    )
    missing = [target for target in required if target not in readme]
    assert not missing, "README front door is missing documentation links: " + ", ".join(missing)


def test_every_runtime_nsis_ai_environment_variable_is_documented():
    names: set[str] = set()
    for path in (ROOT / "src").rglob("*.py"):
        names.update(re.findall(r"\bNSYS_AI_[A-Z0-9_]+\b", path.read_text(encoding="utf-8")))

    # This is an exception error-code prefix, not an environment variable.
    names.discard("NSYS_AI_ERROR")
    docs = (ROOT / "docs/user/environment-variables.md").read_text(encoding="utf-8")
    missing = sorted(name for name in names if f"`{name}`" not in docs)
    assert not missing, "runtime environment variables missing from the reference: " + ", ".join(
        missing
    )


def test_agent_command_family_is_documented_with_its_dependency_boundary():
    docs = (ROOT / "docs/user/skills.md").read_text(encoding="utf-8")
    required = (
        "nsys-ai agent analyze",
        "nsys-ai agent ask",
        "nsys-ai agent-guide",
        "nsys-ai[agent]",
        "NSYS_AI_MODEL",
        "provider credential",
    )
    missing = [target for target in required if target not in docs]
    assert not missing, "agent command documentation is incomplete: " + ", ".join(missing)
