# ruff: noqa: I001
"""Simplified CLI application entrypoint.

Public surface is focused on web UI and AI workflows:
- profile
- open
- web
- timeline-web
- chat
- ask
- report
- export

Legacy commands remain available as hidden aliases for compatibility.

Zero-arg behavior: running ``nsys-ai`` with no arguments shows help (not an
interactive launcher). ``nsys-ai <profile.sqlite>`` still opens the timeline
web UI. This is an intentional product choice after the curses→Textual cleanup.
"""

from __future__ import annotations

import sys


# ---------------------------------------------------------------------------
# Help (moved from main_page; no curses)
# ---------------------------------------------------------------------------


_HELP_BANNER = r"""
  ┌─────────────────────────────────────────────┐
  │              🔬  nsys-ai                     │
  │   AI-powered GPU profile analysis            │
  │                                              │
  │   Navigate timelines · Diagnose bottlenecks  │
  │   Explore NVTX trees · Run analysis skills   │
  └─────────────────────────────────────────────┘
"""


def show_help():
    """Print getting-started guide and command reference."""
    print(_HELP_BANNER)
    print("  Commands:")
    print("  ─────────────────────────────────────────────────────────")
    print("    nsys-ai                       Show this help")
    print("    nsys-ai <profile>             Open web timeline UI (default)")
    print("    nsys-ai help                  This help text")
    print()
    print("  Analysis:")
    print("    nsys-ai doctor  [profile]                Check the environment & profile health")
    print("    nsys-ai info    <profile>                Profile metadata & GPUs")
    print("    nsys-ai warm    <profile>                Pre-build cache & NVTX map")
    print("    nsys-ai summary <profile> [--gpu N]      Kernel stats & commentary")
    print("    nsys-ai timeline <profile> --gpu N --trim S E   Timeline TUI")
    print("    nsys-ai tui     <profile> --gpu N --trim S E   Tree TUI")
    print()
    print("  Capture:")
    print("    nsys-ai profile -- <command> [args...]       Capture a local profile")
    print()
    print("  Optimization loop (diagnose -> propose -> re-profile -> diff -> decide):")
    print("    nsys-ai optimize <profile> --repo PATH -- <command>   Whole loop, one command")
    print("    nsys-ai diagnose <profile>               Findings, published to a session")
    print("    nsys-ai propose --session --finding-id ID --runspec runspec.json")
    print("    nsys-ai diff <before> <after>            Compare two profiles")
    print("    nsys-ai diff <before> <after> --session --accept --reason TEXT   Record it")
    print("    nsys-ai review --session ID              Where a session stands")
    print("    nsys-ai loop <before> --after <after>    Same loop, guided in the browser")
    print("    nsys-ai baseline list                    Named baselines to diff against")
    print("    Full walkthrough: https://github.com/GindaChen/nsys-ai/blob/main/docs/user-guide.md")
    print()
    print("  Skills & Agent:")
    print("    nsys-ai skill list                       List analysis skills")
    print("    nsys-ai skill run <name> <profile>       Run a specific skill")
    print("    nsys-ai report  <profile> --gpu N --trim S E   Performance report")
    print("    nsys-ai agent analyze <profile>           Full auto-analysis")
    print('    nsys-ai agent ask <profile> "question"   Ask about a profile')
    print("    nsys-ai agent-guide                      Print agent System Prompt")
    print()
    print("  Root Causes:")
    print("    nsys-ai root-cause list                  List known root cause patterns")
    print("    nsys-ai root-cause show <name>           Show root cause details")
    print("    nsys-ai root-cause submit <file.md>      Submit a new pattern")
    print()
    print("  Export:")
    print("    nsys-ai export     <profile> --trim S E -o DIR   Perfetto JSON traces")
    print("    nsys-ai export-csv <profile> --gpu N --trim S E   CSV export")
    print("    nsys-ai viewer     <profile> --gpu N --trim S E   HTML report")
    print("    nsys-ai web        <profile> --gpu N --trim S E   Browser UI")
    print("    (--trim takes seconds; `nsys-ai info <profile>` prints the window)")
    print()
    print("  Getting Started:")
    print("    1. Profile:  nsys-ai profile -- python train.py")
    print("    2. Explore:  nsys-ai open <profile.sqlite>")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _looks_like_profile_path(value: str) -> bool:
    lower_value = value.lower()
    return (
        not value.startswith("-")
        and (
            lower_value.endswith(".sqlite")
            or lower_value.endswith(".nsys-rep")
        )
    )


def _normalize_default_profile_command(argv: list[str]) -> list[str]:
    """Route ``nsys-ai <profile>`` through the public timeline-web command."""
    if len(argv) > 1 and _looks_like_profile_path(argv[1]):
        return [argv[0], "timeline-web", *argv[1:]]
    return argv


def _normalize_optimize_command(argv: list[str]) -> list[str]:
    """Accept ``optimize <profile> [options] -- <command>`` as documented.

    ``optimize`` takes a leading positional and then a workload that must reach
    the child process verbatim, so the workload is an ``argparse.REMAINDER``.
    REMAINDER starts at the token after the profile, which means the documented
    order leaves ``--repo`` and the rest inside the workload. Moving the profile
    token to sit directly in front of the ``--`` delimiter (or, with no
    delimiter, to the end followed by an explicit ``--``) produces the
    options-first spelling argparse parses natively. Nothing after ``--`` is
    touched, so a workload keeps its own flags and its own ``--``, and both
    spellings reach the same Namespace.

    Anything else is returned unchanged so argparse still owns the error.
    """
    if len(argv) < 3 or argv[1] != "optimize":
        return argv
    profile = argv[2]
    if profile.startswith("-"):
        return argv  # already options-first, or -h/--help
    rest = argv[3:]
    try:
        delimiter = rest.index("--")
    except ValueError:
        # No workload at all. Everything after the profile is an nsys-ai option,
        # so the profile goes last and an empty workload is made explicit.
        if rest and rest[-1].startswith("-"):
            return argv
        # ...unless the tail is plainly a workload that lost its "--". Two bare
        # tokens in a row cannot both be option values, so rearranging here would
        # slide a workload token onto the profile positional and the failure would
        # surface as "could not resolve before profile", naming a file the caller
        # never offered as one. Leave it alone; the handler reports the real cause.
        for earlier, later in zip(rest, rest[1:]):
            if not earlier.startswith("-") and not later.startswith("-"):
                return argv
        return [argv[0], argv[1], *rest, profile, "--"]
    if delimiter and rest[delimiter - 1].startswith("-"):
        # The slot in front of "--" belongs to an option that is still waiting
        # for its value, so putting the profile there would hand it over.
        return argv
    return [argv[0], argv[1], *rest[:delimiter], profile, *rest[delimiter:]]


def main():
    from .parsers import _build_legacy_parser, _build_parser

    sys.argv = _normalize_default_profile_command(sys.argv)
    sys.argv = _normalize_optimize_command(sys.argv)

    legacy_commands = {
        "analyze",
        "summary",
        "overlap",
        "nccl",
        "iters",
        "tree",
        "markdown",
        "search",
        "export-csv",
        "export-json",
        "viewer",
        "timeline-html",
        "tui",
        "timeline",
    }
    use_legacy_skill_mgmt = (
        len(sys.argv) > 2 and sys.argv[1] == "skill" and sys.argv[2] in {"add", "remove", "save"}
    )
    if len(sys.argv) > 1 and (sys.argv[1] in legacy_commands or use_legacy_skill_mgmt):
        parser = _build_legacy_parser()
    else:
        parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        show_help()
        return

    if args.command == "help":
        show_help()
        return

    from nsys_ai import profile as _profile
    from nsys_ai.exceptions import NsysAiError

    try:
        args.handler(args, _profile)
    except NsysAiError as e:
        import json as _json
        import os

        if os.environ.get("NSYS_AI_AGENT") == "1":
            # Machine-readable output for external AI agents
            print(_json.dumps(e.to_dict()))
        else:
            # Human-readable output
            print(f"Error [{e.error_code}]: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        # Backward compatibility: catch plain RuntimeError from legacy code
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
