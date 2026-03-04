"""
CLI entry point: python -m nsys_tui <command> [options]

Profile: path to .sqlite or .nsys-rep (open/analyze/etc accept both;
.nsys-rep is converted when nsys is on PATH).

Commands:
    info          <profile>                           Show GPU hardware and profile metadata
    analyze       <profile> --gpu N --trim S E        Full auto-report (bottlenecks, overlap, iters, NVTX)
    summary       <profile> [--gpu N]                 GPU kernel summary with top kernels
    overlap       <profile> --gpu N --trim S E        Compute/NCCL overlap analysis
    nccl          <profile> --gpu N --trim S E        NCCL collective breakdown
    iters         <profile> --gpu N --trim S E        Detect training iterations
    tree          <profile> --gpu N --trim S E        NVTX hierarchy as text
    markdown      <profile> --gpu N --trim S E        NVTX hierarchy as markdown
    search        <profile> --query Q                 Search kernels/NVTX by name
    export-csv    <profile> --gpu N --trim S E        Export flat CSV
    export-json   <profile> --gpu N --trim S E        Export flat JSON
    export        <profile> [--gpu N] -o DIR          Export Perfetto JSON traces
    viewer        <profile> --gpu N --trim S E -o F   Generate interactive HTML viewer
    timeline-html <profile> --gpu N --trim S E -o F   Generate horizontal timeline HTML
    web           <profile> --gpu N --trim S E         Serve viewer in browser (local HTTP)
    open          <profile> [--gpu N] [--trim S E]    One-click: Perfetto / web / TUI
    perfetto      <profile> --gpu N --trim S E         Open in Perfetto UI (via local trace server)
    timeline-web  <profile> --gpu N --trim S E         Horizontal timeline in browser
    tui           <profile> --gpu N --trim S E         Terminal tree view
    timeline      <profile> --gpu N --trim S E         Horizontal timeline (Perfetto-style)
    chat          <profile>                            AI chat TUI (requires litellm + textual)
    skill         list | run <name> <profile>          List or run analysis skills
    agent         analyze | ask <profile> [question]   AI agent for profile analysis
"""
import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Shared argument helpers
# ---------------------------------------------------------------------------

def _add_gpu_trim(p, gpu_required=True, trim_required=True):
    """Attach standard --gpu and --trim arguments to a subparser."""
    p.add_argument("profile", help="Path to profile (.sqlite or .nsys-rep)")
    p.add_argument("--gpu", type=int, required=gpu_required,
                   default=None, help="GPU device ID")
    p.add_argument("--trim", nargs=2, type=float,
                   required=trim_required,
                   metavar=("START_S", "END_S"),
                   help="Time window in seconds")


def _parse_trim(args):
    """Convert --trim seconds to a nanoseconds tuple, or None."""
    if args.trim:
        return (int(args.trim[0] * 1e9), int(args.trim[1] * 1e9))
    return None


# ---------------------------------------------------------------------------
# Command handlers — one function per subcommand.
# Each handler receives (args, _profile) where _profile is the lazily-imported
# nsys_tui.profile module.  Handlers that do not open a profile ignore it.
# ---------------------------------------------------------------------------

def _cmd_info(args, _profile):
    with _profile.open(args.profile) as prof:
        m = prof.meta
        print(f"Profile: {args.profile}")
        if getattr(prof, "schema", None) and getattr(prof.schema, "version", None):
            print(f"  Nsight version (heuristic): {prof.schema.version}")
        print(f"  GPUs: {m.devices}")
        print(f"  Kernels: {m.kernel_count}  |  NVTX: {m.nvtx_count}")
        print(f"  Time: {m.time_range[0]/1e9:.3f}s – {m.time_range[1]/1e9:.3f}s")
        print()
        for dev, info in m.gpu_info.items():
            print(f"  GPU {dev}: {info.name} | PCI={info.pci_bus} | "
                  f"SMs={info.sm_count} | Mem={info.memory_bytes/1e9:.0f}GB | "
                  f"Kernels={info.kernel_count} | Streams={info.streams}")


def _cmd_analyze(args, _profile):
    from .report import format_report_markdown, format_report_terminal, run_analyze
    with _profile.open(args.profile) as prof:
        trim = _parse_trim(args)
        data = run_analyze(prof, args.gpu, trim)
        print(format_report_terminal(data))
        if args.output:
            md = format_report_markdown(data, args.profile, trim)
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output, "w", encoding="utf-8", newline="\n") as f:
                f.write(md)
            print(f"Markdown report written to {args.output}")


def _cmd_summary(args, _profile):
    from .summary import auto_commentary, format_text, gpu_summary
    with _profile.open(args.profile) as prof:
        trim = _parse_trim(args)
        gpus = [args.gpu] if args.gpu is not None else prof.meta.devices
        for gpu in gpus:
            s = gpu_summary(prof, gpu, trim)
            print(format_text(s))
            print()
            print(auto_commentary(s))
            print()


def _cmd_overlap(args, _profile):
    from .overlap import format_overlap, overlap_analysis
    with _profile.open(args.profile) as prof:
        print(format_overlap(overlap_analysis(prof, args.gpu, _parse_trim(args))))


def _cmd_nccl(args, _profile):
    from .overlap import format_nccl, nccl_breakdown
    with _profile.open(args.profile) as prof:
        print(format_nccl(nccl_breakdown(prof, args.gpu, _parse_trim(args))))


def _cmd_iters(args, _profile):
    from .overlap import detect_iterations, format_iterations
    with _profile.open(args.profile) as prof:
        print(format_iterations(detect_iterations(prof, args.gpu, _parse_trim(args))))


def _cmd_tree(args, _profile):
    from .tree import build_nvtx_tree, format_text
    with _profile.open(args.profile) as prof:
        roots = build_nvtx_tree(prof, args.gpu, _parse_trim(args))
        print(format_text(roots))


def _cmd_markdown(args, _profile):
    from .tree import build_nvtx_tree, format_markdown
    with _profile.open(args.profile) as prof:
        roots = build_nvtx_tree(prof, args.gpu, _parse_trim(args))
        print(format_markdown(roots))


def _cmd_search(args, _profile):
    from .search import format_results, search_hierarchy, search_kernels, search_nvtx
    with _profile.open(args.profile) as prof:
        trim = _parse_trim(args)
        if args.parent or args.type == "hierarchy":
            if args.gpu is None or not trim:
                print("Error: hierarchical search requires --gpu and --trim")
                return
            results = search_hierarchy(prof, args.parent or "", args.query,
                                       args.gpu, trim)
            print(format_results(results, "hierarchy"))
        elif args.type == "nvtx":
            results = search_nvtx(prof, args.query, args.gpu, trim, args.limit)
            print(format_results(results, "nvtx"))
        else:
            results = search_kernels(prof, args.query, args.gpu, trim, args.limit)
            print(format_results(results, "kernel"))


def _cmd_export_csv(args, _profile):
    from .export_flat import to_csv
    with _profile.open(args.profile) as prof:
        trim = _parse_trim(args)
        content = to_csv(prof, args.gpu, trim, args.output)
        if not args.output:
            print(content)
        else:
            print(f"CSV written to {args.output}")


def _cmd_export_json(args, _profile):
    import json as _json

    from .export_flat import to_json_flat, to_summary_json
    with _profile.open(args.profile) as prof:
        trim = _parse_trim(args)
        if args.summary:
            data = to_summary_json(prof, args.gpu, trim, args.output)
        else:
            data = to_json_flat(prof, args.gpu, trim, args.output)
        if not args.output:
            print(_json.dumps(data, indent=2))
        else:
            print(f"JSON written to {args.output}")


def _cmd_export(args, _profile):
    from . import export
    with _profile.open(args.profile) as prof:
        trim = _parse_trim(args)
        os.makedirs(args.output, exist_ok=True)
        gpus = [args.gpu] if args.gpu is not None else prof.meta.devices
        for gpu in gpus:
            events = export.gpu_trace(prof, gpu, trim)
            if not events:
                print(f"GPU {gpu}: no kernels, skipped")
                continue
            out = os.path.join(args.output, f"trace_gpu{gpu}.json")
            export.write_json(events, out)
            nk = sum(1 for e in events if e.get("cat") == "gpu_kernel")
            nn = sum(1 for e in events if e.get("cat") == "nvtx_projected")
            print(f"GPU {gpu}: {nk} kernels, {nn} NVTX → {out}")


def _cmd_viewer(args, _profile):
    from .viewer import write_html
    with _profile.open(args.profile) as prof:
        write_html(prof, args.gpu, _parse_trim(args), args.output)
        print(f"Written to {args.output} ({os.path.getsize(args.output)//1024} KB)")


def _cmd_timeline_html(args, _profile):
    from .viewer import write_timeline_html
    with _profile.open(args.profile) as prof:
        write_timeline_html(prof, args.gpu, _parse_trim(args), args.output)
        print(f"Written to {args.output} ({os.path.getsize(args.output)//1024} KB)")


def _cmd_web(args, _profile):
    from .web import serve
    with _profile.open(args.profile) as prof:
        serve(prof, args.gpu, _parse_trim(args),
              port=args.port, open_browser=not args.no_browser)


def _cmd_open(args, _profile):
    from .tui import run_tui
    from .web import serve, serve_perfetto
    with _profile.open(args.profile) as prof:
        gpu = args.gpu if args.gpu is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
        if args.trim:
            trim_ns = (int(args.trim[0] * 1e9), int(args.trim[1] * 1e9))
        else:
            trim_ns = (int(prof.meta.time_range[0]), int(prof.meta.time_range[1]))
        port = args.port if args.port is not None else (8143 if args.viewer == "perfetto" else 8142)
        if args.viewer == "perfetto":
            serve_perfetto(prof, gpu, trim_ns, port=port, open_browser=not args.no_browser)
        elif args.viewer == "web":
            serve(prof, gpu, trim_ns, port=port, open_browser=not args.no_browser)
        else:
            # TUI opens its own SQLite connection; save path before context exits.
            profile_path = prof.path
    if args.viewer == "tui":
        run_tui(profile_path, gpu, trim_ns, max_depth=-1, min_ms=0)


def _cmd_perfetto(args, _profile):
    from .web import serve_perfetto
    with _profile.open(args.profile) as prof:
        serve_perfetto(prof, args.gpu, _parse_trim(args),
                       port=args.port, open_browser=not args.no_browser)


def _cmd_timeline_web(args, _profile):
    from .web import serve_timeline
    with _profile.open(args.profile) as prof:
        serve_timeline(prof, args.gpu, _parse_trim(args),
                       port=args.port, open_browser=not args.no_browser)


def _cmd_tui(args, _profile):
    # run_tui opens its own SQLite connection internally.
    from .tree import run_tui  # Textual tree TUI (replaces curses tui.py)
    run_tui(args.profile, args.gpu, _parse_trim(args),
            max_depth=args.depth, min_ms=args.min_ms)


def _cmd_timeline(args, _profile):
    # run_timeline opens its own SQLite connection internally.
    from .timeline import run_timeline  # Textual timeline TUI (replaces curses tui_timeline.py)
    gpu = args.gpu if args.gpu is not None else 0
    run_timeline(args.profile, gpu, _parse_trim(args), min_ms=args.min_ms)


def _cmd_chat(args, _profile):
    try:
        from .tui_textual import run_chat_tui
    except ImportError:
        print("Error: 'textual' package is required. "
              "Install with: pip install 'textual>=0.80.0'")
        return
    run_chat_tui(args.profile)


def _cmd_skill(args, _profile):
    from .skills.registry import all_skills
    from .skills.registry import run_skill as _run_skill
    if args.skill_action == "list":
        skills = all_skills()
        print(f"{'Name':<25s}  {'Category':<15s}  Description")
        print("─" * 80)
        for s in skills:
            print(f"{s.name:<25s}  {s.category:<15s}  {s.description[:60]}")
    elif args.skill_action == "run":
        import sqlite3
        conn = sqlite3.connect(args.profile)
        try:
            print(_run_skill(args.skill_name, conn))
        finally:
            conn.close()
    else:
        # Fall through to print help when no sub-action given.
        print("Usage: nsys-ai skill {list,run} ...")
        sys.exit(1)


def _cmd_agent(args, _profile):
    from .agent.loop import Agent
    if args.agent_action == "analyze":
        agent = Agent(args.profile)
        try:
            print(agent.analyze())
        finally:
            agent.close()
    elif args.agent_action == "ask":
        agent = Agent(args.profile)
        try:
            print(agent.ask(args.question))
        finally:
            agent.close()
    else:
        print("Usage: nsys-ai agent {analyze,ask} ...")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        prog="nsys-ai",
        description="Terminal UI for NVIDIA Nsight Systems profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # ── info ──
    p = sub.add_parser("info", help="Show profile metadata and GPU info")
    p.add_argument("profile", help="Path to profile (.sqlite or .nsys-rep)")
    p.set_defaults(handler=_cmd_info)

    # ── analyze ──
    p = sub.add_parser("analyze", help="Full auto-report: bottlenecks, overlap, iters, NVTX hierarchy")
    _add_gpu_trim(p)
    p.add_argument("-o", "--output", default=None, help="Write markdown report to file (default: terminal only)")
    p.set_defaults(handler=_cmd_analyze)

    # ── summary ──
    p = sub.add_parser("summary", help="GPU kernel summary with top kernels")
    _add_gpu_trim(p, gpu_required=False, trim_required=False)
    p.set_defaults(handler=_cmd_summary)

    # ── overlap ──
    p = sub.add_parser("overlap", help="Compute/NCCL overlap analysis")
    _add_gpu_trim(p)
    p.set_defaults(handler=_cmd_overlap)

    # ── nccl ──
    p = sub.add_parser("nccl", help="NCCL collective breakdown")
    _add_gpu_trim(p)
    p.set_defaults(handler=_cmd_nccl)

    # ── iters ──
    p = sub.add_parser("iters", help="Detect training iterations")
    _add_gpu_trim(p)
    p.set_defaults(handler=_cmd_iters)

    # ── tree ──
    p = sub.add_parser("tree", help="NVTX hierarchy as text")
    _add_gpu_trim(p)
    p.set_defaults(handler=_cmd_tree)

    # ── markdown ──
    p = sub.add_parser("markdown", help="NVTX hierarchy as markdown")
    _add_gpu_trim(p)
    p.set_defaults(handler=_cmd_markdown)

    # ── search ──
    p = sub.add_parser("search", help="Search kernels/NVTX by name")
    p.add_argument("profile", help="Path to profile (.sqlite or .nsys-rep)")
    p.add_argument("--query", "-q", required=True, help="Search query (substring)")
    p.add_argument("--gpu", type=int, default=None, help="GPU device ID")
    p.add_argument("--trim", nargs=2, type=float, metavar=("START_S", "END_S"),
                   help="Time window in seconds")
    p.add_argument("--parent", default=None, help="NVTX parent pattern for hierarchical search")
    p.add_argument("--type", choices=["kernel", "nvtx", "hierarchy"],
                   default="kernel", help="Search type (default: kernel)")
    p.add_argument("--limit", type=int, default=200, help="Max results")
    p.set_defaults(handler=_cmd_search)

    # ── export-csv ──
    p = sub.add_parser("export-csv", help="Export kernel data as flat CSV")
    _add_gpu_trim(p)
    p.add_argument("-o", "--output", default=None, help="Output file (default: stdout)")
    p.set_defaults(handler=_cmd_export_csv)

    # ── export-json ──
    p = sub.add_parser("export-json", help="Export kernel data as flat JSON")
    _add_gpu_trim(p)
    p.add_argument("-o", "--output", default=None, help="Output file (default: stdout)")
    p.add_argument("--summary", action="store_true", help="Export summary instead of flat list")
    p.set_defaults(handler=_cmd_export_json)

    # ── export ──
    p = sub.add_parser("export", help="Export Perfetto JSON traces")
    _add_gpu_trim(p, gpu_required=False)
    p.add_argument("-o", "--output", default=".", help="Output directory")
    p.set_defaults(handler=_cmd_export)

    # ── viewer ──
    p = sub.add_parser("viewer", help="Generate interactive HTML viewer")
    _add_gpu_trim(p)
    p.add_argument("-o", "--output", default="nvtx_tree.html", help="Output HTML file")
    p.set_defaults(handler=_cmd_viewer)

    # ── timeline-html ──
    p = sub.add_parser("timeline-html", help="Generate horizontal timeline HTML")
    _add_gpu_trim(p)
    p.add_argument("-o", "--output", default="timeline.html", help="Output HTML file")
    p.set_defaults(handler=_cmd_timeline_html)

    # ── web ──
    p = sub.add_parser("web", help="Serve interactive viewer in browser")
    _add_gpu_trim(p)
    p.add_argument("--port", type=int, default=8142, help="HTTP port (default: 8142)")
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    p.set_defaults(handler=_cmd_web)

    # ── open ──
    p = sub.add_parser("open", help="One-click open: pick Perfetto, web viewer, or TUI")
    p.add_argument("profile", help="Path to profile (.sqlite or .nsys-rep)")
    p.add_argument("--gpu", type=int, default=None,
                   help="GPU device ID (default: first GPU in profile)")
    p.add_argument("--trim", nargs=2, type=float, metavar=("START_S", "END_S"), default=None,
                   help="Time window in seconds (default: full profile)")
    p.add_argument("--viewer", choices=["perfetto", "web", "tui"], default="perfetto",
                   help="Viewer to use (default: perfetto)")
    p.add_argument("--port", type=int, default=None,
                   help="HTTP port for perfetto/web (default: 8143/8142)")
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open browser (perfetto/web)")
    p.set_defaults(handler=_cmd_open)

    # ── perfetto ──
    p = sub.add_parser("perfetto", help="Open trace in Perfetto UI")
    _add_gpu_trim(p)
    p.add_argument("--port", type=int, default=8143, help="HTTP port for trace (default: 8143)")
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    p.set_defaults(handler=_cmd_perfetto)

    # ── timeline-web ──
    p = sub.add_parser("timeline-web", help="Horizontal timeline in browser")
    _add_gpu_trim(p)
    p.add_argument("--port", type=int, default=8144, help="HTTP port (default: 8144)")
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    p.set_defaults(handler=_cmd_timeline_web)

    # ── tui ──
    p = sub.add_parser("tui", help="Terminal tree view; press A for AI chat")
    _add_gpu_trim(p)
    p.add_argument("--depth", type=int, default=-1, help="Max tree depth (-1=all)")
    p.add_argument("--min-ms", type=float, default=0, help="Min duration to show (ms)")
    p.set_defaults(handler=_cmd_tui)

    # ── timeline ──
    p = sub.add_parser("timeline", help="Horizontal timeline; press A for AI chat")
    _add_gpu_trim(p, gpu_required=False)
    p.add_argument("--min-ms", type=float, default=0, help="Min duration to show (ms)")
    p.set_defaults(handler=_cmd_timeline)

    # ── chat ──
    p = sub.add_parser("chat", help="AI chat TUI with top-kernel navigator (requires litellm + textual)")
    p.add_argument("profile", help="Path to profile (.sqlite or .nsys-rep)")
    p.set_defaults(handler=_cmd_chat)

    # ── skill ──
    p = sub.add_parser("skill", help="List or run analysis skills")
    skill_sub = p.add_subparsers(dest="skill_action")
    skill_sub.add_parser("list", help="List all available skills")
    sp_run = skill_sub.add_parser("run", help="Run a skill against a profile")
    sp_run.add_argument("skill_name", help="Name of the skill to run")
    sp_run.add_argument("profile", help="Path to .sqlite file")
    p.set_defaults(handler=_cmd_skill)

    # ── agent ──
    p = sub.add_parser("agent", help="AI agent for profile analysis")
    agent_sub = p.add_subparsers(dest="agent_action")
    sp_analyze = agent_sub.add_parser("analyze", help="Full auto-analysis report")
    sp_analyze.add_argument("profile", help="Path to .sqlite file")
    sp_ask = agent_sub.add_parser("ask", help="Ask a question about a profile")
    sp_ask.add_argument("profile", help="Path to .sqlite file")
    sp_ask.add_argument("question", help="Natural language question")
    p.set_defaults(handler=_cmd_agent)

    # ── help ──
    sub.add_parser("help", help="Show getting-started guide and available commands")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        from .main_page import run_main_page
        run_main_page()
        return

    if args.command == "help":
        from .main_page import show_help
        show_help()
        return

    # Lazy import: avoids slowing down --help and unused-command paths.
    from . import profile as _profile

    try:
        args.handler(args, _profile)
    except RuntimeError as e:
        # Surface schema/validation issues without a full Python traceback.
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
