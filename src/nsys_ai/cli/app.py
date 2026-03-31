# ruff: noqa: I001
"""Simplified CLI application entrypoint.

Public surface is focused on web UI and AI workflows:
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
    print("    nsys-ai help                  This help text")
    print()
    print("  Analysis:")
    print("    nsys-ai info    <profile>                Profile metadata & GPUs")
    print("    nsys-ai summary <profile> [--gpu N]      Kernel stats & commentary")
    print("    nsys-ai timeline <profile> --gpu N --trim S E   Timeline TUI")
    print("    nsys-ai tui     <profile> --gpu N --trim S E   Tree TUI")
    print()
    print("  Skills & Agent:")
    print("    nsys-ai skill list                       List analysis skills")
    print("    nsys-ai skill run <name> <profile>       Run a specific skill")
    print("    nsys-ai agent analyze <profile>           Full auto-analysis")
    print('    nsys-ai agent ask <profile> "question"   Ask about a profile')
    print("    nsys-ai agent-guide                      Print agent System Prompt")
    print()
    print("  Export:")
    print("    nsys-ai export     <profile> -o DIR       Perfetto JSON traces")
    print("    nsys-ai export-csv <profile> --gpu N       CSV export")
    print("    nsys-ai viewer     <profile> --gpu N       HTML report")
    print("    nsys-ai web        <profile> --gpu N       Browser UI")
    print()
    print("  Getting Started:")
    print("    1. Profile:  nsys profile -o report python train.py")
    print("    2. Export:   nsys export --type sqlite report.nsys-rep")
    print("    3. Explore:  nsys-ai open <profile.sqlite>")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    from .parsers import _build_legacy_parser, _build_parser

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
        "perfetto",
        "tui",
        "timeline",
        "agent",
    }
    use_legacy_skill_mgmt = (
        len(sys.argv) > 2
        and sys.argv[1] == "skill"
        and sys.argv[2] in {"add", "remove", "save"}
    )
    if len(sys.argv) > 1 and (sys.argv[1] in legacy_commands or use_legacy_skill_mgmt):
        parser = _build_legacy_parser()
    else:
        parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        # Zero-arg / unknown: if first arg looks like a profile path, open timeline-web;
        # otherwise show help (intentional: no interactive launcher; see PR/docs for rationale).
        remaining = sys.argv[1:]
        if remaining and not remaining[0].startswith("-"):
            candidate = remaining[0]
            if (
                candidate.endswith(".sqlite")
                or candidate.endswith(".nsys-rep")
                or candidate.endswith(".nsys-rep.zst")
            ):
                from nsys_ai import profile as _profile
                from nsys_ai.web import serve_timeline

                with _profile.open(candidate) as prof:
                    devices = prof.meta.devices if prof.meta.devices else [0]
                    serve_timeline(prof, devices, None, port=8144, open_browser=True)
                return

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
