"""
parsers.py — Argument parser construction for nsys-ai CLI.

Extracted from app.py to reduce file size and improve maintainability.
"""

from __future__ import annotations

import argparse

from .handlers import (
    _add_gpu_trim,
    _cmd_agent,
    _cmd_agent_guide,
    _cmd_analyze,
    _cmd_ask,
    _cmd_chat,
    _cmd_diff,
    _cmd_diff_web,
    _cmd_evidence,
    _cmd_export,
    _cmd_export_csv,
    _cmd_export_json,
    _cmd_info,
    _cmd_iters,
    _cmd_markdown,
    _cmd_nccl,
    _cmd_open,
    _cmd_overlap,
    _cmd_perfetto,
    _cmd_report,
    _cmd_search,
    _cmd_skill,
    _cmd_summary,
    _cmd_timeline,
    _cmd_timeline_html,
    _cmd_timeline_web,
    _cmd_tree,
    _cmd_tui,
    _cmd_viewer,
    _cmd_web,
)

# ---------------------------------------------------------------------------
# Shared parser registration helpers — used by both main and legacy parsers
# ---------------------------------------------------------------------------


def _register_info_parser(sub):
    """Register the ``info`` subcommand on *sub*."""
    p = sub.add_parser("info", help="Show profile metadata and GPU info")
    p.add_argument("profile", help="Path to profile (.sqlite or .nsys-rep)")
    p.set_defaults(handler=_cmd_info)
    return p


def _register_skill_parser(sub, *, include_management=False):
    """Register the ``skill`` subcommand tree on *sub*.

    When *include_management* is True the admin sub-actions (add, remove, save)
    are also registered; otherwise only the agent-facing actions (list, info,
    run) are included.
    """
    p = sub.add_parser("skill", help="List or run analysis skills")
    p.add_argument(
        "--skills-dir",
        default=None,
        help="Directory for custom skills (.md files)",
    )
    skill_sub = p.add_subparsers(dest="skill_action")
    sp_list = skill_sub.add_parser("list", help="List all available skills")
    sp_list.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    sp_info = skill_sub.add_parser(
        "info",
        help="Get JSON schema for a skill (name, description, parameters)",
    )
    sp_info.add_argument("skill_name", help="Name of the skill")
    sp_run = skill_sub.add_parser("run", help="Run a skill against a profile")
    sp_run.add_argument("skill_name", help="Name of the skill to run")
    sp_run.add_argument("profile", help="Path to .sqlite file")
    sp_run.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    sp_run.add_argument(
        "--param",
        "-p",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Skill parameter, e.g. -p operation=full_model -p hidden_dim=2560",
    )
    sp_run.add_argument(
        "--trim",
        nargs=2,
        type=float,
        metavar=("START_S", "END_S"),
        default=None,
        help="Time window in seconds — filters data before analysis (recommended for large profiles)",
    )
    sp_run.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Truncate JSON output to at most N data rows (for token budget control). "
            "When rows are clipped, a final _truncated metadata entry is appended (so "
            "the array may contain N+1 items)."
        ),
    )
    sp_run.add_argument(
        "--iteration",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Auto-trim to the N-th training iteration (0-based). "
            "Cannot be used with --trim. Uses NVTX markers when available and "
            "falls back to a kernel-gap heuristic when markers are missing."
        ),
    )
    sp_run.add_argument(
        "--marker",
        type=str,
        default="sample_0",
        help=(
            "NVTX marker for iteration boundary detection when using --iteration "
            "(default: sample_0)."
        ),
    )

    if include_management:
        sp_add = skill_sub.add_parser("add", help="Add a custom skill from .md file")
        sp_add.add_argument("skill_file", help="Path to skill .md file")
        sp_rm = skill_sub.add_parser("remove", help="Remove a custom skill")
        sp_rm.add_argument("skill_name", help="Name of the skill to remove")
        sp_save = skill_sub.add_parser("save", help="Export a skill to .md file")
        sp_save.add_argument("skill_name", help="Name of the skill to export")
        sp_save.add_argument("-o", "--output", required=True, help="Output .md file")

    p.set_defaults(handler=_cmd_skill)
    return p


def _register_evidence_parser(sub):
    """Register the ``evidence`` subcommand tree on *sub*."""
    p = sub.add_parser("evidence", help="Build evidence findings for timeline overlay")
    evidence_sub = p.add_subparsers(dest="evidence_action", required=True)
    sp_build = evidence_sub.add_parser(
        "build", help="Run heuristic analyzers → findings JSON"
    )
    sp_build.add_argument("profile", help="Path to profile (.sqlite or .nsys-rep)")
    sp_build.add_argument(
        "--format",
        choices=["text", "json"],
        default="json",
        help="Output format (default: json)",
    )
    sp_build.add_argument(
        "--analyzers",
        default=None,
        help="Comma-separated analyzer names: slow_iterations,idle_gaps,"
        "nccl_stalls,kernel_hotspots,overlap_ratio,memory_anomalies,h2d_spikes",
    )
    sp_build.add_argument(
        "--trim",
        nargs=2,
        type=float,
        metavar=("START_S", "END_S"),
        default=None,
        help="Time window in seconds",
    )
    sp_build.add_argument("--gpu", type=int, default=0, help="GPU device ID (default: 0)")
    sp_build.add_argument(
        "-o", "--output", default=None, help="Write findings JSON to file"
    )
    p.set_defaults(handler=_cmd_evidence)
    return p


# ---------------------------------------------------------------------------
# Main parser — public CLI surface visible to ``nsys-ai --help``
# ---------------------------------------------------------------------------


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="nsys-ai",
        description="Web-first Nsight Systems analysis CLI (with AI backend tools)",
    )
    sub = parser.add_subparsers(
        dest="command",
        metavar=(
            "{open,web,timeline-web,chat,ask,agent-guide,"
            "info,skill,evidence,report,diff,diff-web,export,help}"
        ),
    )

    # Public commands (simplified)
    p = sub.add_parser("open", help="Open profile quickly in Perfetto/web/TUI")
    p.add_argument("profile", help="Path to profile (.sqlite or .nsys-rep)")
    p.add_argument(
        "--gpu", type=int, default=None, help="GPU device ID (default: first GPU in profile)"
    )
    p.add_argument(
        "--trim",
        nargs=2,
        type=float,
        metavar=("START_S", "END_S"),
        default=None,
        help="Time window in seconds (default: full profile)",
    )
    p.add_argument(
        "--viewer",
        choices=["perfetto", "web", "tui"],
        default="perfetto",
        help="Viewer to use (default: perfetto)",
    )
    p.add_argument(
        "--port", type=int, default=None, help="HTTP port for perfetto/web (default: 8143/8142)"
    )
    p.add_argument(
        "--no-browser", action="store_true", help="Don't auto-open browser (perfetto/web)"
    )
    p.set_defaults(handler=_cmd_open)

    p = sub.add_parser("web", help="Serve interactive web viewer")
    _add_gpu_trim(p)
    p.add_argument("--port", type=int, default=8142, help="HTTP port (default: 8142)")
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    p.set_defaults(handler=_cmd_web)

    p = sub.add_parser("timeline-web", help="Serve timeline-focused web UI")
    _add_gpu_trim(p, gpu_required=False, trim_required=False)
    p.add_argument("--port", type=int, default=8144, help="HTTP port (default: 8144)")
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    p.add_argument("--findings", default=None, help="Path to findings.json for evidence overlay")
    p.add_argument(
        "--auto-analyze", action="store_true", help="Run AI analysis on startup and show findings"
    )
    p.set_defaults(handler=_cmd_timeline_web)

    p = sub.add_parser("chat", help="AI chat TUI")
    p.add_argument("profile", help="Path to profile (.sqlite or .nsys-rep)")
    p.set_defaults(handler=_cmd_chat)

    p = sub.add_parser("ask", help="Ask AI a backend analysis question")
    p.add_argument("profile", help="Path to .sqlite file")
    p.add_argument("question", help="Natural language question")
    p.set_defaults(handler=_cmd_ask)

    p = sub.add_parser("agent-guide", help="Print machine-readable guide for AI agents")
    p.set_defaults(handler=_cmd_agent_guide)

    p = sub.add_parser("report", help="Generate performance report")
    _add_gpu_trim(p)
    p.add_argument("-o", "--output", default=None, help="Write markdown report to file")
    p.set_defaults(handler=_cmd_report)

    p = sub.add_parser("diff", help="Compare two profiles (before/after)")
    p.add_argument("before", help="Path to baseline profile (.sqlite or .nsys-rep)")
    p.add_argument("after", help="Path to candidate profile (.sqlite or .nsys-rep)")
    p.add_argument("--gpu", type=int, default=None, help="GPU device ID (default: all GPUs)")
    p.add_argument(
        "--trim",
        nargs=2,
        type=float,
        required=False,
        metavar=("START_S", "END_S"),
        help="Time window in seconds (apply to both profiles)",
    )
    p.add_argument(
        "--iteration",
        type=int,
        default=None,
        metavar="N",
        help="Compare only the N-th iteration (0-based; uses NVTX marker)",
    )
    p.add_argument(
        "--marker",
        type=str,
        default="sample_0",
        help="NVTX marker for iteration boundaries (default: sample_0)",
    )
    p.add_argument(
        "--format",
        choices=["terminal", "markdown", "json"],
        default="terminal",
        help="Output format (default: terminal)",
    )
    p.add_argument("-o", "--output", default=None, help="Write rendered output to file")
    p.add_argument(
        "--limit", type=int, default=15, help="Top regressions/improvements (default: 15)"
    )
    p.add_argument(
        "--sort",
        choices=["delta", "percent", "total"],
        default="delta",
        help="Sort mode for top changes (default: delta)",
    )
    p.add_argument("--no-ai", action="store_true", help="No-op v1 flag (reserved for AI narrative)")
    p.add_argument(
        "--chat",
        action="store_true",
        help="Start interactive AI chat for diff analysis (Phase C tools)",
    )
    p.set_defaults(handler=_cmd_diff)

    p = sub.add_parser("diff-web", help="Serve web diff viewer for two profiles")
    p.add_argument("before", help="Path to baseline profile (.sqlite or .nsys-rep)")
    p.add_argument("after", help="Path to candidate profile (.sqlite or .nsys-rep)")
    p.add_argument("--gpu", type=int, default=None, help="GPU device ID (default: all GPUs)")
    p.add_argument(
        "--trim",
        nargs=2,
        type=float,
        required=False,
        metavar=("START_S", "END_S"),
        help="Time window in seconds (apply to both profiles)",
    )
    p.add_argument("--port", type=int, default=8145, help="HTTP port (default: 8145)")
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    p.set_defaults(handler=_cmd_diff_web)

    p = sub.add_parser("export", help="Export Perfetto JSON traces")
    _add_gpu_trim(p, gpu_required=False)
    p.add_argument("-o", "--output", default=".", help="Output directory")
    p.set_defaults(handler=_cmd_export)

    # Agent-facing commands (promoted from legacy so --help exposes them)
    _register_info_parser(sub)
    _register_skill_parser(sub, include_management=False)
    _register_evidence_parser(sub)

    sub.add_parser("help", help="Show getting-started guide and available commands")

    return parser


def _register_legacy_commands(sub):
    """Register legacy commands on the provided subparser collection."""
    _register_info_parser(sub)

    p = sub.add_parser(
        "analyze", help="Full auto-report: bottlenecks, overlap, iters, NVTX hierarchy"
    )
    _add_gpu_trim(p)
    p.add_argument("-o", "--output", default=None, help="Write markdown report to file")
    p.set_defaults(handler=_cmd_analyze)

    p = sub.add_parser("summary", help="GPU kernel summary with top kernels")
    _add_gpu_trim(p, gpu_required=False, trim_required=False)
    p.set_defaults(handler=_cmd_summary)

    p = sub.add_parser("overlap", help="Compute/NCCL overlap analysis")
    _add_gpu_trim(p)
    p.set_defaults(handler=_cmd_overlap)

    p = sub.add_parser("nccl", help="NCCL collective breakdown")
    _add_gpu_trim(p)
    p.set_defaults(handler=_cmd_nccl)

    p = sub.add_parser("iters", help="Detect training iterations")
    _add_gpu_trim(p)
    p.set_defaults(handler=_cmd_iters)

    p = sub.add_parser("tree", help="NVTX hierarchy as text")
    _add_gpu_trim(p)
    p.set_defaults(handler=_cmd_tree)

    p = sub.add_parser("markdown", help="NVTX hierarchy as markdown")
    _add_gpu_trim(p)
    p.set_defaults(handler=_cmd_markdown)

    p = sub.add_parser("search", help="Search kernels/NVTX by name")
    p.add_argument("profile", help="Path to profile (.sqlite or .nsys-rep)")
    p.add_argument("--query", "-q", required=True, help="Search query (substring)")
    p.add_argument("--gpu", type=int, default=None, help="GPU device ID")
    p.add_argument(
        "--trim", nargs=2, type=float, metavar=("START_S", "END_S"), help="Time window in seconds"
    )
    p.add_argument("--parent", default=None, help="NVTX parent pattern for hierarchical search")
    p.add_argument(
        "--type",
        choices=["kernel", "nvtx", "hierarchy"],
        default="kernel",
        help="Search type (default: kernel)",
    )
    p.add_argument("--limit", type=int, default=200, help="Max results")
    p.set_defaults(handler=_cmd_search)

    p = sub.add_parser("export-csv", help="Export kernel data as flat CSV")
    _add_gpu_trim(p)
    p.add_argument("-o", "--output", default=None, help="Output file (default: stdout)")
    p.set_defaults(handler=_cmd_export_csv)

    p = sub.add_parser("export-json", help="Export kernel data as flat JSON")
    _add_gpu_trim(p)
    p.add_argument("-o", "--output", default=None, help="Output file (default: stdout)")
    p.add_argument("--summary", action="store_true", help="Export summary instead of flat list")
    p.set_defaults(handler=_cmd_export_json)

    p = sub.add_parser("viewer", help="Generate interactive HTML viewer")
    _add_gpu_trim(p)
    p.add_argument("-o", "--output", default="nvtx_tree.html", help="Output HTML file")
    p.set_defaults(handler=_cmd_viewer)

    p = sub.add_parser("timeline-html", help="Generate horizontal timeline HTML")
    _add_gpu_trim(p)
    p.add_argument("-o", "--output", default="timeline.html", help="Output HTML file")
    p.set_defaults(handler=_cmd_timeline_html)

    p = sub.add_parser("perfetto", help="Open trace in Perfetto UI")
    _add_gpu_trim(p)
    p.add_argument("--port", type=int, default=8143, help="HTTP port for trace (default: 8143)")
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    p.set_defaults(handler=_cmd_perfetto)

    p = sub.add_parser("tui", help="Terminal tree view; press A for AI chat")
    _add_gpu_trim(p)
    p.add_argument("--depth", type=int, default=-1, help="Max tree depth (-1=all)")
    p.add_argument("--min-ms", type=float, default=0, help="Min duration to show (ms)")
    p.set_defaults(handler=_cmd_tui)

    p = sub.add_parser("timeline", help="Horizontal timeline; press A for AI chat")
    _add_gpu_trim(p, gpu_required=False)
    p.add_argument("--min-ms", type=float, default=0, help="Min duration to show (ms)")
    p.set_defaults(handler=_cmd_timeline)

    _register_skill_parser(sub, include_management=True)

    p = sub.add_parser("agent", help="AI agent for profile analysis")
    agent_sub = p.add_subparsers(dest="agent_action")
    sp_analyze = agent_sub.add_parser("analyze", help="Full auto-analysis report")
    sp_analyze.add_argument("profile", help="Path to .sqlite file")
    sp_analyze.add_argument(
        "--trim",
        nargs=2,
        type=float,
        metavar=("START_S", "END_S"),
        default=None,
        help="Time window in seconds (recommended for large profiles)",
    )
    sp_analyze.add_argument(
        "--evidence", action="store_true", help="Also output evidence findings JSON"
    )
    sp_analyze.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path for findings JSON (default: findings.json)",
    )
    sp_ask = agent_sub.add_parser("ask", help="Ask a question about a profile")
    sp_ask.add_argument("profile", help="Path to .sqlite file")
    sp_ask.add_argument("question", help="Natural language question")
    p.set_defaults(handler=_cmd_agent)

    # ── evidence ──────────────────────────────────────────────────
    _register_evidence_parser(sub)

    sub.add_parser("help", help="Show getting-started guide and available commands")


def _build_legacy_parser():
    """Build full legacy parser used for explicit legacy command invocations."""
    parser = argparse.ArgumentParser(
        prog="nsys-ai",
        description="Legacy Nsight Systems CLI (full command surface)",
    )
    sub = parser.add_subparsers(dest="command")
    _register_legacy_commands(sub)
    return parser
