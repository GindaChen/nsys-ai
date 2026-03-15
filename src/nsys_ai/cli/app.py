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

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# Shared argument helpers
# ---------------------------------------------------------------------------


def _add_gpu_trim(p, gpu_required=True, trim_required=True):
    """Attach standard --gpu and --trim arguments to a subparser."""
    p.add_argument("profile", help="Path to profile (.sqlite or .nsys-rep)")
    p.add_argument("--gpu", type=int, required=gpu_required, default=None, help="GPU device ID")
    p.add_argument(
        "--trim",
        nargs=2,
        type=float,
        required=trim_required,
        metavar=("START_S", "END_S"),
        help="Time window in seconds",
    )


def _parse_trim(args):
    """Convert --trim seconds to a nanoseconds tuple, or None."""
    if getattr(args, "trim", None):
        return (int(args.trim[0] * 1e9), int(args.trim[1] * 1e9))
    return None


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
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_info(args, _profile):
    with _profile.open(args.profile) as prof:
        m = prof.meta
        print(f"Profile: {args.profile}")
        if getattr(prof, "schema", None) and getattr(prof.schema, "version", None):
            print(f"  Nsight version (heuristic): {prof.schema.version}")
        print(f"  GPUs: {m.devices}")
        print(f"  Kernels: {m.kernel_count}  |  NVTX: {m.nvtx_count}")
        print(f"  Time: {m.time_range[0] / 1e9:.3f}s - {m.time_range[1] / 1e9:.3f}s")
        print()
        for dev, info in m.gpu_info.items():
            print(
                f"  GPU {dev}: {info.name} | PCI={info.pci_bus} | "
                f"SMs={info.sm_count} | Mem={info.memory_bytes / 1e9:.0f}GB | "
                f"Kernels={info.kernel_count} | Streams={info.streams}"
            )


def _cmd_analyze(args, _profile):
    from nsys_ai.report import format_report_markdown, format_report_terminal, run_analyze

    with _profile.open(args.profile) as prof:
        trim = _parse_trim(args)
        data = run_analyze(prof, args.gpu, trim)
        print(format_report_terminal(data))
        if getattr(args, "output", None):
            md = format_report_markdown(data, args.profile, trim)
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output, "w", encoding="utf-8", newline="\n") as f:
                f.write(md)
            print(f"Markdown report written to {args.output}")


def _cmd_report(args, _profile):
    """Simplified alias for analyze."""
    _cmd_analyze(args, _profile)


def _cmd_diff(args, _profile):
    from nsys_ai.diff import diff_profiles
    from nsys_ai.diff_render import (
        format_diff_markdown,
        format_diff_markdown_multi,
        format_diff_terminal,
        format_diff_terminal_multi,
        to_diff_json,
    )
    from nsys_ai.diff_tools import DiffContext, get_iteration_boundaries

    no_ai = getattr(args, "no_ai", False)

    def _narrative_for(summary):
        if args.format not in ("terminal", "markdown"):
            return None
        from nsys_ai.ai.diff_narrative import (
            DiffNarrative,
            build_executive_summary,
            generate_diff_narrative,
        )

        if no_ai:
            return DiffNarrative(
                executive_summary=build_executive_summary(summary),
                ai_narrative=None,
                model=None,
                warning=None,
            )
        return generate_diff_narrative(summary)

    if getattr(args, "chat", False):
        _run_diff_chat(args, _profile)
        return

    trim = _parse_trim(args)
    trim_before = None
    trim_after = None
    if getattr(args, "iteration", None) is not None:
        with _profile.open(args.before) as before, _profile.open(args.after) as after:
            ctx = DiffContext(
                before=before, after=after, trim=trim, marker=getattr(args, "marker", "sample_0")
            )
            bounds = get_iteration_boundaries(
                ctx, marker=getattr(args, "marker", "sample_0"), target_gpu=args.gpu
            )
            bnds = bounds["boundaries"]
            idx = args.iteration
            if idx >= len(bnds):
                print(f"Error: iteration {idx} out of range (0..{len(bnds) - 1})", file=sys.stderr)
                return
            bnd = bnds[idx]
            if bnd["before"]["start_ns"] is not None and bnd["before"]["end_ns"] is not None:
                trim_before = (bnd["before"]["start_ns"], bnd["before"]["end_ns"])
            if bnd["after"]["start_ns"] is not None and bnd["after"]["end_ns"] is not None:
                trim_after = (bnd["after"]["start_ns"], bnd["after"]["end_ns"])
            if not trim_before or not trim_after:
                print(
                    "Error: no time window for this iteration in one or both profiles",
                    file=sys.stderr,
                )
                return

    with _profile.open(args.before) as before, _profile.open(args.after) as after:
        if trim_before is not None and trim_after is not None:
            summary = diff_profiles(
                before,
                after,
                gpu=args.gpu,
                trim_before=trim_before,
                trim_after=trim_after,
                limit=args.limit,
                sort=args.sort,
            )
            narrative = _narrative_for(summary)
            if args.format == "terminal":
                out = format_diff_terminal(summary, narrative=narrative)
            elif args.format == "markdown":
                out = format_diff_markdown(summary, narrative=narrative)
            elif args.format == "json":
                out = to_diff_json(summary)
            else:
                raise RuntimeError(f"Unknown format: {args.format}")
        elif args.gpu is not None:
            summary = diff_profiles(
                before,
                after,
                gpu=args.gpu,
                trim=trim,
                limit=args.limit,
                sort=args.sort,
            )
            narrative = _narrative_for(summary)
            if args.format == "terminal":
                out = format_diff_terminal(summary, narrative=narrative)
            elif args.format == "markdown":
                out = format_diff_markdown(summary, narrative=narrative)
            elif args.format == "json":
                out = to_diff_json(summary)
            else:
                raise RuntimeError(f"Unknown format: {args.format}")
        else:
            # Global (all GPUs) + per-GPU breakdown.
            global_summary = diff_profiles(
                before,
                after,
                gpu=None,
                trim=trim,
                limit=args.limit,
                sort=args.sort,
            )
            # For per-GPU we keep top-k small to avoid overwhelming output.
            per_gpu_limit = min(args.limit, 3)
            devices = sorted(set(before.meta.devices) | set(after.meta.devices))
            per_gpu = {}
            for dev in devices:
                per_gpu[dev] = diff_profiles(
                    before,
                    after,
                    gpu=dev,
                    trim=trim,
                    limit=per_gpu_limit,
                    sort=args.sort,
                )

            narrative = _narrative_for(global_summary)
            if args.format == "terminal":
                out = format_diff_terminal_multi(global_summary, per_gpu, narrative=narrative)
            elif args.format == "markdown":
                out = format_diff_markdown_multi(global_summary, per_gpu, narrative=narrative)
            elif args.format == "json":
                # For JSON, keep the contract simple: return only the global summary.
                out = to_diff_json(global_summary)
            else:
                raise RuntimeError(f"Unknown format: {args.format}")

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(out)
        print(f"Diff written to {args.output}")
    else:
        print(out, end="")


def _run_diff_chat(args, _profile):
    """Interactive diff chat: Phase C tools + cached ProfileDiffSummary."""
    from nsys_ai.chat import _get_model_and_key, distill_history, stream_agent_loop
    from nsys_ai.diff_tools import DiffContext, get_iteration_boundaries

    model, _ = _get_model_and_key()
    if not model:
        print(
            "Error: No LLM model configured. Set API key (e.g. OPENAI_API_KEY) and retry.",
            file=sys.stderr,
        )
        return

    trim = _parse_trim(args)
    marker = getattr(args, "marker", "sample_0") or "sample_0"
    gpu = getattr(args, "gpu", None)
    target_gpu = 0 if gpu is None else gpu

    with _profile.open(args.before) as before, _profile.open(args.after) as after:
        ctx = DiffContext(before=before, after=after, trim=trim, marker=marker)
        ctx.ensure_summary(target_gpu)

        bounds = get_iteration_boundaries(ctx, marker=marker, target_gpu=target_gpu)
        n_iters = len(bounds.get("boundaries") or [])
        print(f"Diff chat: {args.before} vs {args.after}")
        print(f"Iteration marker: {marker}  |  Boundaries: {n_iters} iteration(s)")
        print("Ask about regressions, regions, or iteration diffs. Empty line to exit.")
        print()

        chat_history: list = []
        while True:
            try:
                line = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            chat_history.append({"role": "user", "content": line})
            text_parts: list[str] = []
            for ev in stream_agent_loop(
                model=model,
                messages=list(chat_history),
                ui_context={},
                profile_path=None,
                diff_context=ctx,
                diff_paths=(args.before, args.after),
                max_turns=8,
            ):
                if ev.get("type") == "text" and ev.get("content"):
                    text_parts.append(ev["content"])
                    print(ev["content"], end="", flush=True)
                elif ev.get("type") == "system" and ev.get("content"):
                    print(f"\n[{ev['content']}]", flush=True)
            chat_history.append({"role": "assistant", "content": "".join(text_parts)})
            chat_history[:] = distill_history(chat_history)
            if text_parts:
                print()
            print()


def _cmd_diff_web(args, _profile):
    from nsys_ai.diff_web import serve_diff_web

    trim = _parse_trim(args)
    with _profile.open(args.before) as before, _profile.open(args.after) as after:
        serve_diff_web(
            before,
            after,
            gpu=args.gpu,
            trim=trim,
            port=args.port,
            open_browser=not args.no_browser,
        )


def _cmd_summary(args, _profile):
    from nsys_ai.summary import auto_commentary, format_text, gpu_summary

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
    from nsys_ai.overlap import format_overlap, overlap_analysis

    with _profile.open(args.profile) as prof:
        print(format_overlap(overlap_analysis(prof, args.gpu, _parse_trim(args))))


def _cmd_nccl(args, _profile):
    from nsys_ai.overlap import format_nccl, nccl_breakdown

    with _profile.open(args.profile) as prof:
        print(format_nccl(nccl_breakdown(prof, args.gpu, _parse_trim(args))))


def _cmd_iters(args, _profile):
    from nsys_ai.overlap import detect_iterations, format_iterations

    with _profile.open(args.profile) as prof:
        print(format_iterations(detect_iterations(prof, args.gpu, _parse_trim(args))))


def _cmd_tree(args, _profile):
    from nsys_ai.tree import build_nvtx_tree, format_text

    with _profile.open(args.profile) as prof:
        roots = build_nvtx_tree(prof, args.gpu, _parse_trim(args))
        print(format_text(roots))


def _cmd_markdown(args, _profile):
    from nsys_ai.tree import build_nvtx_tree, format_markdown

    with _profile.open(args.profile) as prof:
        roots = build_nvtx_tree(prof, args.gpu, _parse_trim(args))
        print(format_markdown(roots))


def _cmd_search(args, _profile):
    from nsys_ai.search import format_results, search_hierarchy, search_kernels, search_nvtx

    with _profile.open(args.profile) as prof:
        trim = _parse_trim(args)
        if args.parent or args.type == "hierarchy":
            if args.gpu is None or not trim:
                print("Error: hierarchical search requires --gpu and --trim")
                return
            results = search_hierarchy(prof, args.parent or "", args.query, args.gpu, trim)
            print(format_results(results, "hierarchy"))
        elif args.type == "nvtx":
            results = search_nvtx(prof, args.query, args.gpu, trim, args.limit)
            print(format_results(results, "nvtx"))
        else:
            results = search_kernels(prof, args.query, args.gpu, trim, args.limit)
            print(format_results(results, "kernel"))


def _cmd_export_csv(args, _profile):
    from nsys_ai.export_flat import to_csv

    with _profile.open(args.profile) as prof:
        trim = _parse_trim(args)
        content = to_csv(prof, args.gpu, trim, args.output)
        if not args.output:
            print(content)
        else:
            print(f"CSV written to {args.output}")


def _cmd_export_json(args, _profile):
    import json as _json

    from nsys_ai.export_flat import to_json_flat, to_summary_json

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
    from nsys_ai import export

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
            print(f"GPU {gpu}: {nk} kernels, {nn} NVTX -> {out}")


def _cmd_viewer(args, _profile):
    from nsys_ai.viewer import write_html

    with _profile.open(args.profile) as prof:
        write_html(prof, args.gpu, _parse_trim(args), args.output)
        print(f"Written to {args.output} ({os.path.getsize(args.output) // 1024} KB)")


def _cmd_timeline_html(args, _profile):
    from nsys_ai.viewer import write_timeline_html

    with _profile.open(args.profile) as prof:
        write_timeline_html(prof, args.gpu, _parse_trim(args), args.output)
        print(f"Written to {args.output} ({os.path.getsize(args.output) // 1024} KB)")


def _cmd_web(args, _profile):
    from nsys_ai.web import serve

    with _profile.open(args.profile) as prof:
        serve(prof, args.gpu, _parse_trim(args), port=args.port, open_browser=not args.no_browser)


def _cmd_open(args, _profile):
    from nsys_ai.tree import run_tui
    from nsys_ai.web import serve, serve_perfetto

    with _profile.open(args.profile) as prof:
        gpu = (
            args.gpu if args.gpu is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
        )
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
            profile_path = prof.path
    if args.viewer == "tui":
        run_tui(profile_path, gpu, trim_ns, max_depth=-1, min_ms=0)


def _cmd_perfetto(args, _profile):
    from nsys_ai.web import serve_perfetto

    with _profile.open(args.profile) as prof:
        serve_perfetto(
            prof, args.gpu, _parse_trim(args), port=args.port, open_browser=not args.no_browser
        )


def _cmd_timeline_web(args, _profile):
    from nsys_ai.web import serve_timeline

    with _profile.open(args.profile) as prof:
        if args.gpu is not None:
            devices = args.gpu
        else:
            devices = prof.meta.devices if prof.meta.devices else [0]

        # Auto-analyze: build findings in-process before serving
        auto_findings = None
        if getattr(args, "auto_analyze", False) and not getattr(args, "findings", None):
            from nsys_ai.evidence_builder import EvidenceBuilder

            device = devices[0] if isinstance(devices, list) else devices
            builder = EvidenceBuilder(prof, device=device)
            report = builder.build()
            auto_findings = [f.to_dict() for f in report.findings]
            print(f"Auto-analysis: {len(auto_findings)} finding(s)", flush=True)

        serve_timeline(
            prof,
            devices,
            _parse_trim(args),
            port=args.port,
            open_browser=not args.no_browser,
            findings_path=getattr(args, "findings", None),
            auto_findings=auto_findings,
        )


def _cmd_tui(args, _profile):
    from nsys_ai.tree import run_tui

    run_tui(args.profile, args.gpu, _parse_trim(args), max_depth=args.depth, min_ms=args.min_ms)


def _cmd_timeline(args, _profile):
    from nsys_ai.timeline import run_timeline

    gpu = args.gpu if args.gpu is not None else 0
    run_timeline(args.profile, gpu, _parse_trim(args), min_ms=args.min_ms)


def _cmd_chat(args, _profile):
    try:
        from nsys_ai.tui_textual import run_chat_tui
    except ImportError:
        print("Error: 'textual' package is required. Install with: pip install 'textual>=0.80.0'")
        return
    run_chat_tui(args.profile)


def _cmd_skill(args, _profile):
    import json as _json

    from nsys_ai.skills.registry import all_skills, get_skill, load_custom_skills_dir
    from nsys_ai.skills.registry import run_skill as _run_skill

    # Load custom skills from --skills-dir or env var
    skills_dir = getattr(args, "skills_dir", None) or os.environ.get("NSYS_AI_CUSTOM_SKILLS_DIR")
    if skills_dir and os.path.isdir(skills_dir):
        load_custom_skills_dir(skills_dir)

    if args.skill_action == "list":
        skills = all_skills()
        fmt = getattr(args, "format", "text")
        if fmt == "json":
            print(
                _json.dumps(
                    [
                        {
                            "name": s.name,
                            "title": s.title,
                            "description": s.description,
                            "category": s.category,
                            "params": [
                                {
                                    "name": p.name,
                                    "type": p.type,
                                    "required": p.required,
                                    "default": p.default,
                                }
                                for p in s.params
                            ],
                        }
                        for s in skills
                    ],
                    indent=2,
                )
            )
        else:
            print(f"{'Name':<25s}  {'Category':<15s}  Description")
            print("-" * 80)
            for s in skills:
                print(f"{s.name:<25s}  {s.category:<15s}  {s.description[:60]}")
    elif args.skill_action == "run":
        import sqlite3

        fmt = getattr(args, "format", "text")
        conn = sqlite3.connect(args.profile)

        # Build trim kwargs if --trim was provided
        trim_kwargs = {}
        trim = getattr(args, "trim", None)
        if trim:
            trim_kwargs["trim_start_ns"] = int(trim[0] * 1e9)
            trim_kwargs["trim_end_ns"] = int(trim[1] * 1e9)

        try:
            if fmt == "json":
                skill = get_skill(args.skill_name)
                if not skill:
                    available = ", ".join(s.name for s in all_skills())
                    print(
                        _json.dumps(
                            {
                                "error": f"Unknown skill '{args.skill_name}'",
                                "available": available,
                            }
                        )
                    )
                    sys.exit(1)
                rows = skill.execute(conn, **trim_kwargs)
                print(_json.dumps(rows, indent=2))
            else:
                print(_run_skill(args.skill_name, conn, **trim_kwargs))
        finally:
            conn.close()
    elif args.skill_action == "add":
        from pathlib import Path
        import shutil

        from nsys_ai.skills.registry import load_skill_from_markdown

        if not skills_dir:
            print("Error: --skills-dir is required for 'skill add'", file=sys.stderr)
            sys.exit(1)
        src = Path(args.skill_file)
        if not src.exists():
            print(f"Error: file not found: {src}", file=sys.stderr)
            sys.exit(1)
        dst_dir = Path(skills_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        # Copy to a temporary path based on the source filename.
        tmp_dst = dst_dir / src.name
        shutil.copy2(src, tmp_dst)
        # Load the skill to determine its canonical name.
        skill = load_skill_from_markdown(str(tmp_dst))
        normalized_dst = dst_dir / f"{skill.name}.md"
        # If the canonical filename differs, rename the copied file,
        # but avoid overwriting an existing skill file.
        if normalized_dst != tmp_dst:
            if normalized_dst.exists():
                print(
                    f"Error: a skill file for '{skill.name}' already exists at {normalized_dst}",
                    file=sys.stderr,
                )
                try:
                    tmp_dst.unlink()
                except OSError:
                    pass
                sys.exit(1)
            tmp_dst.rename(normalized_dst)
            dst = normalized_dst
        else:
            dst = tmp_dst
        print(f"Added skill '{skill.name}' → {dst}")
    elif args.skill_action == "remove":
        from pathlib import Path

        if not skills_dir:
            print("Error: --skills-dir is required for 'skill remove'", file=sys.stderr)
            sys.exit(1)
        target = Path(skills_dir) / f"{args.skill_name}.md"
        if target.exists():
            target.unlink()
            print(f"Removed skill '{args.skill_name}'")
        else:
            print(f"No custom skill file found: {target}")
    elif args.skill_action == "save":
        from nsys_ai.skills.registry import save_skill_to_markdown

        skill = get_skill(args.skill_name)
        if not skill:
            print(f"Unknown skill: {args.skill_name}", file=sys.stderr)
            sys.exit(1)
        save_skill_to_markdown(skill, args.output)
        print(f"Saved '{skill.name}' → {args.output}")
    else:
        print("Usage: nsys-ai skill {list,run,add,remove,save} ...")
        sys.exit(1)


def _cmd_agent(args, _profile):
    from nsys_ai.agent.loop import Agent

    if args.agent_action == "analyze":
        agent = Agent(args.profile)
        try:
            print(agent.analyze())
            # Optionally produce evidence findings JSON
            if getattr(args, "evidence", False):
                from nsys_ai.annotation import save_findings
                from nsys_ai.evidence_builder import EvidenceBuilder
                from nsys_ai.profile import Profile

                with Profile(args.profile) as prof:
                    builder = EvidenceBuilder(prof, device=0)
                    report = builder.build()
                    out = getattr(args, "output", None) or "findings.json"
                    save_findings(report, out)
                    print(f"Evidence: {len(report.findings)} finding(s) → {out}")
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


def _cmd_ask(args, _profile):
    """Simplified alias for `agent ask`."""
    from nsys_ai.agent.loop import Agent

    agent = Agent(args.profile)
    try:
        print(agent.ask(args.question))
    finally:
        agent.close()


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="nsys-ai",
        description="Web-first Nsight Systems analysis CLI (with AI backend tools)",
    )
    sub = parser.add_subparsers(
        dest="command",
        metavar="{open,web,timeline-web,chat,ask,report,diff,diff-web,export,help}",
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

    sub.add_parser("help", help="Show getting-started guide and available commands")

    return parser


def _register_legacy_commands(sub):
    """Register legacy commands on the provided subparser collection."""
    p = sub.add_parser("info", help="Show profile metadata and GPU info")
    p.add_argument("profile", help="Path to profile (.sqlite or .nsys-rep)")
    p.set_defaults(handler=_cmd_info)

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
        "--trim",
        nargs=2,
        type=float,
        metavar=("START_S", "END_S"),
        default=None,
        help="Time window in seconds — filters data before analysis (recommended for large profiles)",
    )
    sp_add = skill_sub.add_parser("add", help="Add a custom skill from .md file")
    sp_add.add_argument("skill_file", help="Path to skill .md file")
    sp_rm = skill_sub.add_parser("remove", help="Remove a custom skill")
    sp_rm.add_argument("skill_name", help="Name of the skill to remove")
    sp_save = skill_sub.add_parser("save", help="Export a skill to .md file")
    sp_save.add_argument("skill_name", help="Name of the skill to export")
    sp_save.add_argument("-o", "--output", required=True, help="Output .md file")
    p.set_defaults(handler=_cmd_skill)

    p = sub.add_parser("agent", help="AI agent for profile analysis")
    agent_sub = p.add_subparsers(dest="agent_action")
    sp_analyze = agent_sub.add_parser("analyze", help="Full auto-analysis report")
    sp_analyze.add_argument("profile", help="Path to .sqlite file")
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    legacy_commands = {
        "info",
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
        "skill",
        "agent",
    }
    if len(sys.argv) > 1 and sys.argv[1] in legacy_commands:
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

    try:
        args.handler(args, _profile)
    except RuntimeError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
