"""
CLI entry point: python -m nsys_tui <command> [options]

Commands:
    info       <profile.sqlite>                     Show GPU hardware and profile metadata
    summary    <profile.sqlite> [--gpu N]           GPU kernel summary with top kernels
    overlap    <profile.sqlite> --gpu N --trim S E  Compute/NCCL overlap analysis
    nccl       <profile.sqlite> --gpu N --trim S E  NCCL collective breakdown
    iters      <profile.sqlite> --gpu N --trim S E  Detect training iterations
    tree       <profile.sqlite> --gpu N --trim S E  NVTX hierarchy as text
    markdown   <profile.sqlite> --gpu N --trim S E  NVTX hierarchy as markdown
    search     <profile.sqlite> --query Q           Search kernels/NVTX by name
    export-csv <profile.sqlite> --gpu N --trim S E  Export flat CSV
    export-json <profile.sqlite> --gpu N --trim S E Export flat JSON
    export     <profile.sqlite> [--gpu N] -o DIR    Export Perfetto JSON traces
    viewer       <profile.sqlite> --gpu N --trim S E -o FILE     Generate interactive HTML viewer
    timeline-html <profile.sqlite> --gpu N --trim S E -o FILE    Generate horizontal timeline HTML
    web        <profile.sqlite> --gpu N --trim S E  Serve viewer in browser (local HTTP)
    perfetto   <profile.sqlite> --gpu N --trim S E  Open in Perfetto UI (via local trace server)
    timeline-web <profile.sqlite> --gpu N --trim S E Horizontal timeline in browser
    tui        <profile.sqlite> --gpu N --trim S E  Terminal tree view
    timeline   <profile.sqlite> --gpu N --trim S E  Horizontal timeline (Perfetto-style)
"""
import sys
import os
import argparse


def _add_gpu_trim(p, gpu_required=True, trim_required=True):
    """Add standard --gpu and --trim arguments to a subparser."""
    p.add_argument("profile", help="Path to .sqlite file")
    p.add_argument("--gpu", type=int, required=gpu_required,
                   default=None, help="GPU device ID")
    p.add_argument("--trim", nargs=2, type=float,
                   required=trim_required,
                   metavar=("START_S", "END_S"),
                   help="Time window in seconds")


def _parse_trim(args):
    """Convert --trim seconds to nanoseconds tuple, or None."""
    if args.trim:
        return (int(args.trim[0] * 1e9), int(args.trim[1] * 1e9))
    return None


def main():
    parser = argparse.ArgumentParser(
        prog="nsys-ai",
        description="Terminal UI for NVIDIA Nsight Systems profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # ── info ──
    p = sub.add_parser("info", help="Show profile metadata and GPU info")
    p.add_argument("profile", help="Path to .sqlite file")

    # ── summary ──
    p = sub.add_parser("summary", help="GPU kernel summary with top kernels")
    _add_gpu_trim(p, gpu_required=False, trim_required=False)

    # ── overlap ──
    p = sub.add_parser("overlap", help="Compute/NCCL overlap analysis")
    _add_gpu_trim(p)

    # ── nccl ──
    p = sub.add_parser("nccl", help="NCCL collective breakdown")
    _add_gpu_trim(p)

    # ── iters ──
    p = sub.add_parser("iters", help="Detect training iterations")
    _add_gpu_trim(p)

    # ── tree ──
    p = sub.add_parser("tree", help="NVTX hierarchy as text")
    _add_gpu_trim(p)

    # ── markdown ──
    p = sub.add_parser("markdown", help="NVTX hierarchy as markdown")
    _add_gpu_trim(p)

    # ── search ──
    p = sub.add_parser("search", help="Search kernels/NVTX by name")
    p.add_argument("profile", help="Path to .sqlite file")
    p.add_argument("--query", "-q", required=True, help="Search query (substring)")
    p.add_argument("--gpu", type=int, default=None, help="GPU device ID")
    p.add_argument("--trim", nargs=2, type=float, metavar=("START_S", "END_S"),
                   help="Time window in seconds")
    p.add_argument("--parent", default=None, help="NVTX parent pattern for hierarchical search")
    p.add_argument("--type", choices=["kernel", "nvtx", "hierarchy"],
                   default="kernel", help="Search type (default: kernel)")
    p.add_argument("--limit", type=int, default=200, help="Max results")

    # ── export-csv ──
    p = sub.add_parser("export-csv", help="Export kernel data as flat CSV")
    _add_gpu_trim(p)
    p.add_argument("-o", "--output", default=None, help="Output file (default: stdout)")

    # ── export-json ──
    p = sub.add_parser("export-json", help="Export kernel data as flat JSON")
    _add_gpu_trim(p)
    p.add_argument("-o", "--output", default=None, help="Output file (default: stdout)")
    p.add_argument("--summary", action="store_true", help="Export summary instead of flat list")

    # ── export ──
    p = sub.add_parser("export", help="Export Perfetto JSON traces")
    _add_gpu_trim(p, gpu_required=False)
    p.add_argument("-o", "--output", default=".", help="Output directory")

    # ── viewer ──
    p = sub.add_parser("viewer", help="Generate interactive HTML viewer")
    _add_gpu_trim(p)
    p.add_argument("-o", "--output", default="nvtx_tree.html", help="Output HTML file")

    # ── timeline-html ──
    p = sub.add_parser("timeline-html", help="Generate horizontal timeline HTML")
    _add_gpu_trim(p)
    p.add_argument("-o", "--output", default="timeline.html", help="Output HTML file")

    # ── web ──
    p = sub.add_parser("web", help="Serve interactive viewer in browser")
    _add_gpu_trim(p)
    p.add_argument("--port", type=int, default=8142, help="HTTP port (default: 8142)")
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")

    # ── perfetto ──
    p = sub.add_parser("perfetto", help="Open trace in Perfetto UI")
    _add_gpu_trim(p)
    p.add_argument("--port", type=int, default=8143, help="HTTP port for trace (default: 8143)")
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")

    # ── timeline-web ──
    p = sub.add_parser("timeline-web", help="Horizontal timeline in browser")
    _add_gpu_trim(p)
    p.add_argument("--port", type=int, default=8144, help="HTTP port (default: 8144)")
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")

    # ── tui ──
    p = sub.add_parser("tui", help="Terminal tree view (rich)")
    _add_gpu_trim(p)
    p.add_argument("--depth", type=int, default=-1, help="Max tree depth (-1=all)")
    p.add_argument("--min-ms", type=float, default=0, help="Min duration to show (ms)")

    # ── timeline ──
    p = sub.add_parser("timeline", help="Horizontal timeline view (Perfetto-style)")
    _add_gpu_trim(p)
    p.add_argument("--min-ms", type=float, default=0, help="Min duration to show (ms)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    # Import here to avoid slow startup for --help
    from . import profile as _profile

    try:
        if args.command == "info":
            prof = _profile.open(args.profile)
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
            prof.close()

        elif args.command == "summary":
            from .summary import gpu_summary, format_text, auto_commentary
            prof = _profile.open(args.profile)
            trim = _parse_trim(args)
            gpus = [args.gpu] if args.gpu is not None else prof.meta.devices
            for gpu in gpus:
                s = gpu_summary(prof, gpu, trim)
                print(format_text(s))
                print()
                print(auto_commentary(s))
                print()
            prof.close()

        elif args.command == "overlap":
            from .overlap import overlap_analysis, format_overlap
            prof = _profile.open(args.profile)
            print(format_overlap(overlap_analysis(prof, args.gpu, _parse_trim(args))))
            prof.close()

        elif args.command == "nccl":
            from .overlap import nccl_breakdown, format_nccl
            prof = _profile.open(args.profile)
            print(format_nccl(nccl_breakdown(prof, args.gpu, _parse_trim(args))))
            prof.close()

        elif args.command == "iters":
            from .overlap import detect_iterations, format_iterations
            prof = _profile.open(args.profile)
            print(format_iterations(detect_iterations(prof, args.gpu, _parse_trim(args))))
            prof.close()

        elif args.command == "tree":
            from .tree import build_nvtx_tree, format_text
            prof = _profile.open(args.profile)
            roots = build_nvtx_tree(prof, args.gpu, _parse_trim(args))
            print(format_text(roots))
            prof.close()

        elif args.command == "markdown":
            from .tree import build_nvtx_tree, format_markdown
            prof = _profile.open(args.profile)
            roots = build_nvtx_tree(prof, args.gpu, _parse_trim(args))
            print(format_markdown(roots))
            prof.close()

        elif args.command == "export":
            from . import export
            prof = _profile.open(args.profile)
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
            prof.close()

        elif args.command == "search":
            from .search import (search_kernels, search_nvtx,
                                 search_hierarchy, format_results)
            prof = _profile.open(args.profile)
            trim = _parse_trim(args)

            if args.parent or args.type == "hierarchy":
                if not args.gpu or not trim:
                    print("Error: hierarchical search requires --gpu and --trim")
                    prof.close()
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
            prof.close()

        elif args.command == "export-csv":
            from .export_flat import to_csv
            prof = _profile.open(args.profile)
            trim = _parse_trim(args)
            content = to_csv(prof, args.gpu, trim, args.output)
            if not args.output:
                print(content)
            else:
                print(f"CSV written to {args.output}")
            prof.close()

        elif args.command == "export-json":
            import json as _json
            from .export_flat import to_json_flat, to_summary_json
            prof = _profile.open(args.profile)
            trim = _parse_trim(args)
            if args.summary:
                data = to_summary_json(prof, args.gpu, trim, args.output)
            else:
                data = to_json_flat(prof, args.gpu, trim, args.output)
            if not args.output:
                print(_json.dumps(data, indent=2))
            else:
                print(f"JSON written to {args.output}")
            prof.close()

        elif args.command == "viewer":
            from .viewer import write_html
            prof = _profile.open(args.profile)
            write_html(prof, args.gpu, _parse_trim(args), args.output)
            print(f"Written to {args.output} ({os.path.getsize(args.output)//1024} KB)")
            prof.close()

        elif args.command == "timeline-html":
            from .viewer import write_timeline_html
            prof = _profile.open(args.profile)
            write_timeline_html(prof, args.gpu, _parse_trim(args), args.output)
            print(f"Written to {args.output} ({os.path.getsize(args.output)//1024} KB)")
            prof.close()

        elif args.command == "web":
            from .web import serve
            prof = _profile.open(args.profile)
            serve(prof, args.gpu, _parse_trim(args),
                  port=args.port, open_browser=not args.no_browser)

        elif args.command == "perfetto":
            from .web import serve_perfetto
            prof = _profile.open(args.profile)
            serve_perfetto(prof, args.gpu, _parse_trim(args),
                           port=args.port, open_browser=not args.no_browser)

        elif args.command == "timeline-web":
            from .web import serve_timeline
            prof = _profile.open(args.profile)
            serve_timeline(prof, args.gpu, _parse_trim(args),
                           port=args.port, open_browser=not args.no_browser)

        elif args.command == "tui":
            from .tui import run_tui
            run_tui(args.profile, args.gpu, _parse_trim(args),
                    max_depth=args.depth, min_ms=args.min_ms)

        elif args.command == "timeline":
            from .tui_timeline import run_timeline
            run_timeline(args.profile, args.gpu, _parse_trim(args),
                         min_ms=args.min_ms)
    except RuntimeError as e:
        # Surface schema/validation issues without a full Python traceback.
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
