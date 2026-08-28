"""Small guards for the user-facing documentation front doors."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


DOCUMENTATION_IMAGES = (
    "web-tree.png",
    "timeline-web.png",
    "diff-web.png",
    "guided-loop.png",
)
MAX_DOCUMENTATION_IMAGE_BYTES = 300 * 1024


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


def test_documentation_screenshots_are_present_small_pngs_and_referenced():
    images = ROOT / "docs/images"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    viewers = (ROOT / "docs/user/viewers.md").read_text(encoding="utf-8")
    guided_loop = (ROOT / "docs/guided-loop-setup.md").read_text(encoding="utf-8")

    missing = [name for name in DOCUMENTATION_IMAGES if not (images / name).is_file()]
    assert not missing, "documentation screenshot files are missing: " + ", ".join(missing)

    for name in DOCUMENTATION_IMAGES:
        path = images / name
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
        assert path.stat().st_size < MAX_DOCUMENTATION_IMAGE_BYTES, (
            f"{path} exceeds the {MAX_DOCUMENTATION_IMAGE_BYTES // 1024} KiB screenshot budget"
        )

    assert "docs/images/timeline-web.png" in readme
    assert all(f"../images/{name}" in viewers for name in DOCUMENTATION_IMAGES[:3])
    assert "images/guided-loop.png" in guided_loop


def test_site_references_the_committed_viewer_screenshots():
    site = (ROOT / "site/index.html").read_text(encoding="utf-8")
    missing = [name for name in DOCUMENTATION_IMAGES[:3] if f"docs/images/{name}" not in site]
    assert not missing, "landing page is missing viewer screenshots: " + ", ".join(missing)


def test_landing_page_has_a_keyboard_safe_install_action_and_workflow():
    site = (ROOT / "site/index.html").read_text(encoding="utf-8")
    assert '<button type="button" class="install-box"' in site
    assert "onclick=\"copyInstallCommand(this)\"" in site
    assert '<div class="install-box"' not in site
    assert ":focus-visible" in site
    assert "prefers-reduced-motion" in site
    for step in ("Diagnose", "Propose", "Re-profile", "Diff", "Decide"):
        assert f">{step}<" in site
    assert not re.search(r"[🚀📚🧠🌐🖥️🌲🔍📊🤖🔁]", site)


def test_claude_release_process_points_to_the_canonical_release_guide():
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    release = (ROOT / "docs/dev/release.md").read_text(encoding="utf-8")

    assert "docs/dev/release.md" in claude
    assert "git push origin main --tags" not in claude
    assert "git push upstream v0.4.0" in release
    assert re.search(r"`origin`\s+is\s+the\s+fork", release)
    assert re.search(r"`upstream`\s+is\s+the\s+canonical", release)


#: Documentation placeholders that stand in for a real value. The parser only has to
#: accept the shape of the line, so these are substituted with something well-typed.
DOCUMENTATION_PLACEHOLDERS = {"START_S": "0", "END_S": "5", "N": "0"}


def _documented_command_lines() -> list[tuple[Path, int, str]]:
    """Every fenced `nsys-ai ...` invocation across the documentation tree.

    Only fenced blocks count. A sentence that happens to open with the program name is
    prose, and a line carrying shell syntax (a continuation, a redirect, a pipe) is not
    a single argv this can hand to argparse.
    """
    lines: list[tuple[Path, int, str]] = []
    roots = (ROOT.glob("*.md"), ROOT.glob("docs/**/*.md"), ROOT.glob("examples/**/*.md"))
    for path in sorted({p for root in roots for p in root}):
        inside_fence = False
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if raw.lstrip().startswith("```"):
                inside_fence = not inside_fence
                continue
            if not inside_fence:
                continue
            line = raw.strip().removeprefix("$ ").strip()
            if not line.startswith("nsys-ai ") or "`" in line or "](" in line:
                continue
            if line.rstrip().endswith((".", ",", ":", ";")):
                continue
            if any(token in line for token in ("\\", ">", "|")):
                continue
            lines.append((path.relative_to(ROOT), number, line))
    return lines


def test_there_are_documented_commands_to_check():
    """Guard the guard: an empty sweep would make the assertion below vacuous."""
    assert _documented_command_lines(), "no fenced nsys-ai commands found in the docs tree"


def test_every_documented_command_line_is_accepted_by_the_cli():
    """A copy-pasted documentation command must not die on argparse.

    This is the failure the docs keep shipping: the command exists, the reader copies
    the line, and it stops at `error: the following arguments are required: --trim`, or
    at a subcommand that was removed a release ago. Nothing connected the documented
    examples to the parser that has to accept them, so the drift was invisible until a
    reader hit it. Parsing is deliberately all this asserts -- running the commands
    would need real profiles -- but argparse is where every one of those failures was.
    """
    import contextlib
    import io
    import shlex
    import sys

    saved_argv = sys.argv
    sys.argv = ["nsys-ai"]
    try:
        from nsys_ai.cli.app import (
            _normalize_default_profile_command,
            _normalize_optimize_command,
        )
        from nsys_ai.cli.parsers import (
            LEGACY_ROUTED_COMMANDS,
            _build_legacy_parser,
            _build_parser,
        )
    finally:
        sys.argv = saved_argv

    rejected = []
    for path, number, line in _documented_command_lines():
        try:
            argv = shlex.split(line.split(" #")[0])
        except ValueError:
            continue
        argv = [DOCUMENTATION_PLACEHOLDERS.get(token, token) for token in argv]
        # main() rewrites the bare `nsys-ai PROFILE` form before it reaches a parser.
        argv = _normalize_optimize_command(_normalize_default_profile_command(argv))
        command = argv[1] if len(argv) > 1 else ""
        parser = _build_legacy_parser() if command in LEGACY_ROUTED_COMMANDS else _build_parser()
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                parser.parse_args(argv[1:])
        except SystemExit as exit_status:
            if exit_status.code:
                reason = (stderr.getvalue().strip().splitlines() or ["rejected"])[-1]
                rejected.append(f"{path}:{number}: {line}\n    {reason}")

    assert not rejected, "documented commands the CLI rejects:\n" + "\n".join(rejected)
