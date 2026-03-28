"""
handlers.py — CLI command handlers for nsys-ai.

Extracted from app.py to reduce file size and improve maintainability.
Each handler follows the signature ``handler(args, _profile)``.
"""

from __future__ import annotations

import os
import sys


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


def _coerce_param_value(raw_value, param_type):
    """Coerce a raw string CLI parameter to the type expected by the skill.

    Falls back to returning the raw string if no type information is
    available.  Exits the process with an error message if coercion fails.
    """
    # If the skill did not declare a type, keep the raw string.
    if param_type is None:
        return raw_value

    type_name = str(param_type).lower()

    try:
        if param_type is int or type_name in {"int", "integer"}:
            return int(raw_value)
        if param_type is float or type_name in {"float", "double"}:
            return float(raw_value)
        if param_type is bool or type_name in {"bool", "boolean"}:
            val = raw_value.strip().lower()
            if val in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if val in {"0", "false", "f", "no", "n", "off"}:
                return False
            raise ValueError(f"cannot interpret '{raw_value}' as boolean")
    except ValueError as exc:
        print(
            f"Error: cannot convert '{raw_value}' to {param_type}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Default: treat as string.
    return raw_value


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

    from nsys_ai.exceptions import SkillExecutionError, SkillNotFoundError
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
    elif args.skill_action == "info":
        skill = get_skill(args.skill_name)
        if skill is None:
            print(f"Error: Skill '{args.skill_name}' not found.", file=sys.stderr)
            sys.exit(1)
        schema = {
            "name": skill.name,
            "description": skill.description,
            "parameters": {
                p.name: {
                    "type": p.type,
                    "description": getattr(p, "description", ""),
                    "default": p.default,
                    "required": p.required
                } for p in skill.params
            }
        }
        print(_json.dumps(schema, indent=2))
    elif args.skill_action == "run":
        import sqlite3

        import duckdb

        from nsys_ai.parquet_cache import open_cached_db

        fmt = getattr(args, "format", "text")
        try:
            conn = open_cached_db(args.profile)
        except (duckdb.Error, RuntimeError, OSError) as exc:
            # Fallback to raw SQLite if DuckDB/Parquet cache fails
            import logging
            logging.getLogger("nsys_ai").warning(
                "DuckDB cache unavailable (%s), falling back to raw SQLite", exc
            )
            conn = sqlite3.connect(args.profile)

        # Build trim kwargs if --trim was provided
        trim_kwargs = {}
        trim = getattr(args, "trim", None)
        if trim:
            trim_kwargs["trim_start_ns"] = int(trim[0] * 1e9)
            trim_kwargs["trim_end_ns"] = int(trim[1] * 1e9)

        # Parse --param KEY=VALUE pairs into validated, typed kwargs
        param_kwargs = {}

        raw_params = getattr(args, "param", []) or []
        skill_for_params = None
        param_specs = None

        if raw_params:
            # Try to resolve the skill so we can validate and type-cast params.
            try:
                skill_for_params = get_skill(args.skill_name)
            except (SkillNotFoundError, KeyError):
                skill_for_params = None

            if skill_for_params is not None and hasattr(skill_for_params, "params"):
                param_specs = {
                    p.name: p for p in skill_for_params.params if getattr(p, "name", None)
                }
            else:
                param_specs = None

        for pv in raw_params:
            key, sep, val = pv.partition("=")
            if not sep:
                print(f"Error: --param must be KEY=VALUE, got: {pv}", file=sys.stderr)
                sys.exit(1)

            # If we have parameter metadata, validate the key and coerce the type.
            if param_specs is not None:
                if key not in param_specs:
                    valid = ", ".join(sorted(param_specs.keys()))
                    print(
                        f"Error: unknown parameter '{key}' for skill "
                        f"'{args.skill_name}'. "
                        f"Valid parameters: {valid}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                spec = param_specs[key]
                param_type = getattr(spec, "type", None)
                val = _coerce_param_value(val, param_type)

            param_kwargs[key] = val

        # Merge trim-related kwargs with validated/typed skill params.
        full_kwargs = {}
        full_kwargs.update(trim_kwargs)
        full_kwargs.update(param_kwargs)
        # Provide the sqlite path so execute_fn skills can find
        # the sibling .nsys-rep for nsys recipe acceleration.
        full_kwargs["_sqlite_path"] = args.profile

        try:
            if fmt == "json":
                skill = get_skill(args.skill_name)
                if not skill:
                    raise SkillNotFoundError(
                        f"Unknown skill '{args.skill_name}'",
                        available=[s.name for s in all_skills()],
                    )
                rows = skill.execute(conn, **full_kwargs)
                print(_json.dumps(rows, indent=2))
            else:
                print(_run_skill(args.skill_name, conn, **full_kwargs))
        except SkillNotFoundError as e:
            if fmt == "json":
                print(_json.dumps(e.to_dict()))
            else:
                print(f"Error [{e.error_code}]: {e}", file=sys.stderr)
            sys.exit(1)
        except (sqlite3.Error, SkillExecutionError) as e:
            if fmt == "json":
                if isinstance(e, SkillExecutionError):
                    payload = e.to_dict()
                else:
                    payload = {"error": {"code": "SKILL_EXECUTION_ERROR", "message": str(e)}}
                print(_json.dumps(payload))
            else:
                print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except duckdb.Error as e:
            if fmt == "json":
                payload = {"error": {"code": "SKILL_EXECUTION_ERROR", "message": str(e)}}
                print(_json.dumps(payload))
            else:
                print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            conn.close()
    elif args.skill_action == "add":
        import shutil
        from pathlib import Path

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
        try:
            skill = load_skill_from_markdown(str(tmp_dst))
        except ValueError as exc:
            # Parsing failed: clean up the temporary copy and report a clear error.
            print(
                f"Error: failed to parse skill markdown '{src}': {exc}",
                file=sys.stderr,
            )
            try:
                tmp_dst.unlink()
            except OSError:
                pass
            sys.exit(1)
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
            raise SkillNotFoundError(
                f"Unknown skill: {args.skill_name}",
                available=[s.name for s in all_skills()],
            )
        save_skill_to_markdown(skill, args.output)
        print(f"Saved '{skill.name}' → {args.output}")
    else:
        print("Usage: nsys-ai skill {list,info,run,add,remove,save} ...")
        sys.exit(1)


def _cmd_agent(args, _profile):
    from nsys_ai.agent.loop import Agent

    if args.agent_action == "analyze":
        trim_ns = None
        trim = getattr(args, "trim", None)
        if trim:
            trim_ns = (int(trim[0] * 1e9), int(trim[1] * 1e9))
        agent = Agent(args.profile, trim_ns=trim_ns)
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


def _cmd_agent_guide(args, _profile):
    """Print a machine-readable guide for external AI agents."""
    from nsys_ai.skills.registry import skill_catalog

    guide = """# nsys-ai Agent Guide
You are an AI performance tuning agent using `nsys-ai` to analyze NVIDIA Nsight Systems GPU profiles.
Your goal is to identify bottlenecks, correlate them with specific Python source code lines, and recommend actionable fixes.

## Performance Note
The first `skill run` on a large profile may take 60-90s for DuckDB cache initialization.
Subsequent runs on the same profile are faster (~10-30s). Plan your tool calls accordingly.

## Core Principles
1. Never guess NVTX names or kernel strings. Run `schema_inspect` or query NVTX tables first.
2. Always output metrics with units (ms, s, %, TFLOPS, GB/s).
3. **MANDATORY**: Correlate findings with local Python source code (via grep/find) to provide line-level recommendations.

## The 6-Stage Top-Down Triage Workflow
1. **Orient**: Run `nsys-ai info <profile>` for quick metadata (GPU name, kernel count, time range).
   Then run `nsys-ai skill run schema_inspect <profile>` to see available tables.
2. **Temporal Breakdown**: Check utilization and bubbles (`gpu_idle_gaps`, `top_kernels`).
   If `gpu_idle_gaps` returns a `_summary` row with `gap_count: 0`, the GPU is well-utilized — this is a GOOD result, not an error.
3. **Kernel Deep-Dive**: Identify the heaviest operations (`top_kernels`, `kernel_launch_overhead`).
4. **NVTX Mapping**: Attribute GPU time to code regions (`nvtx_layer_breakdown`).
   If auto-detection returns low confidence, retry with explicit `-p depth=1` or `-p depth=2`.
5. **Cross-GPU**: IF applicable, analyze multi-GPU communication (`nccl_breakdown`, `overlap_breakdown`).
6. **Root Cause**: Run `root_cause_matcher` for automated pattern detection. Synthesize all evidence
   and deliver specific, code-level actionable fixes.

## CLI Execution
You execute analysis dynamically via the CLI:
```bash
nsys-ai info <profile.sqlite>                                      # quick metadata
nsys-ai skill run <skill_name> <profile.sqlite> --format json [-p PARAM=VALUE]
```
Example: `nsys-ai skill run top_kernels baseline.sqlite --format json -p limit=5`
"""
    print(guide)
    print(skill_catalog())
