#!/usr/bin/env python3
"""Benchmark the Parquet cache build + a representative skill workload.

Designed for the multi-phase cache-optimization plan. Measures:
  - cache build time (deletes cache + rebuilds from .sqlite)
  - a small basket of skill queries against the rebuilt cache
  - reports JSON to stdout and a one-line summary to stderr

Usage:
  python scripts/bench_cache.py <profile.sqlite> [--no-rebuild] [--trim S E] [--out FILE]
                                                 [--basket default|auto-policy]

--no-rebuild reuses the existing cache (only measures query time).
--trim narrows the analysis window for the skills (default: 400 420).
--basket picks the skill set. ``auto-policy`` is the eight-skill battery that every
figure in ``parquet_cache.open_auto_db`` and the README's cache table was measured
over; the default basket is a different twelve and will not reproduce them.

What the build column covers has changed, and comparing across that change is a
mistake. ``nvtx_kernel_map`` is no longer produced by ``build_cache`` — it is
built by the first skill that attributes kernels to NVTX ranges, and written back
into the cache. So the build number here now excludes it (it was 58% of the build
on an 881 MB capture and 64% on a 3.5 GB one) and the skill number includes it,
once, for whichever skill in the basket reaches it first. Set
``NSYS_AI_ALWAYS_BUILD_NVTX_KERNEL_MAP=1`` to measure the old split.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Skills with rough categories so the summary is readable.
# Each entry: (skill_name, extra_args, category)
DEFAULT_SKILLS = [
    ("top_kernels", [], "compute"),
    ("overlap_breakdown", ["-p", "device=0"], "compute"),
    ("nccl_breakdown", ["-p", "device=0"], "comms"),
    ("gpu_idle_gaps", ["-p", "device=0"], "idle"),
    ("sync_cost_analysis", ["-p", "device=0"], "idle"),
    ("kernel_overlap_matrix", ["-p", "device=0"], "comms"),
    ("pipeline_bubble_metrics", [], "comms"),
    ("kernel_launch_overhead", [], "launch"),
    ("stream_concurrency", [], "comms"),
    ("memory_transfers", [], "mem"),
    ("tensor_core_usage", [], "compute"),
    ("kernel_launch_pattern", [], "launch"),
]

# The exact basket behind every number in ``parquet_cache.open_auto_db``'s decision
# table and the README's "Analysis cache" table. It is NOT DEFAULT_SKILLS: this one
# is eight skills, that one is twelve, and the two do not produce comparable rows.
#
# Named here rather than left implicit because the auto policy's whole argument is a
# measurement, and a measurement whose workload is unstated cannot be reproduced or
# challenged. Select it with ``--basket auto-policy``.
#
# The two ``-p device=0`` arguments are part of the definition: both skills abstain
# without a device, so passing these names through bare ``--skills`` measures
# something else entirely.
AUTO_POLICY_SKILLS = [
    ("top_kernels", [], "compute"),
    ("gpu_idle_gaps", ["-p", "device=0"], "idle"),
    ("overlap_breakdown", ["-p", "device=0"], "compute"),
    ("memory_transfers", [], "mem"),
    ("kernel_launch_overhead", [], "launch"),
    ("stream_concurrency", [], "comms"),
    ("tensor_core_usage", [], "compute"),
    # Last on purpose: it is the only one that reaches nvtx_kernel_map, so on a cold
    # cache it absorbs the deferred map build (3.7 / 3.4 / 11.2 / 49.7 s at 93 MB /
    # 235 MB / 924 MB / 3.7 GB) and the ordering keeps that cost in one place.
    ("nvtx_layer_breakdown", [], "nvtx"),
]

BASKETS = {"default": DEFAULT_SKILLS, "auto-policy": AUTO_POLICY_SKILLS}


def _run(cmd: list[str], capture: bool = True, timeout: int = 900) -> tuple[int, str, str, float]:
    """Run a command and return (returncode, stdout, stderr, elapsed_seconds)."""
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or "", time.monotonic() - t0


def measure_cache_build(sqlite_path: Path, no_rebuild: bool) -> dict:
    """Measure cache build time. If no_rebuild is True, skip and report cached state."""
    cache_dir = sqlite_path.with_suffix(".nsys-cache")
    if no_rebuild:
        if cache_dir.exists():
            files = sorted(p.name for p in cache_dir.iterdir() if p.suffix == ".parquet")
            return {
                "rebuilt": False,
                "cache_dir": str(cache_dir),
                "parquet_files": files,
                "elapsed_s": None,
            }
        return {"rebuilt": False, "cache_dir": str(cache_dir), "parquet_files": [], "elapsed_s": None}

    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    # Trigger build via a cheap skill — schema_inspect is the fastest.
    rc, _, err, elapsed = _run(
        ["nsys-ai", "skill", "run", "schema_inspect", str(sqlite_path), "--format", "json"],
        timeout=1800,
    )
    files = sorted(p.name for p in cache_dir.iterdir() if p.suffix == ".parquet") if cache_dir.exists() else []
    sizes_mb = {
        p.name: round(p.stat().st_size / 1e6, 1)
        for p in cache_dir.iterdir()
        if p.is_file() and p.suffix == ".parquet"
    } if cache_dir.exists() else {}
    return {
        "rebuilt": True,
        "cache_dir": str(cache_dir),
        "parquet_files": files,
        "parquet_sizes_mb": sizes_mb,
        "elapsed_s": round(elapsed, 2),
        "exit_code": rc,
        "stderr_tail": err[-500:] if err else "",
    }


def measure_skill(sqlite_path: Path, skill: str, extra: list[str], trim: tuple[float, float] | None) -> dict:
    """Run a single skill, capture wall time and result size."""
    cmd = ["nsys-ai", "skill", "run", skill, str(sqlite_path), "--format", "json"]
    if trim is not None:
        cmd += ["--trim", str(trim[0]), str(trim[1])]
    cmd += extra
    rc, out, err, elapsed = _run(cmd, timeout=600)
    n_rows = None
    try:
        parsed = json.loads(out) if out.strip() else None
        if isinstance(parsed, list):
            n_rows = len(parsed)
    except json.JSONDecodeError:
        pass
    return {
        "skill": skill,
        "exit": rc,
        "elapsed_s": round(elapsed, 3),
        "rows": n_rows,
        "stdout_bytes": len(out),
        "stderr_tail": err[-200:] if err else "",
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("sqlite", type=Path)
    p.add_argument("--no-rebuild", action="store_true", help="reuse existing cache")
    p.add_argument("--trim", nargs=2, type=float, default=(400.0, 420.0),
                   help="trim window (start end) in seconds; default 400 420")
    p.add_argument("--skills", nargs="*", default=None,
                   help="override skill set; default = the --basket selection")
    p.add_argument("--basket", choices=sorted(BASKETS), default="default",
                   help="named skill basket; 'auto-policy' is the eight skills behind "
                        "the table in parquet_cache.open_auto_db")
    p.add_argument("--out", type=Path, default=None, help="JSON output path")
    p.add_argument("--label", default=None, help="label for this run (e.g. baseline / phase1)")
    args = p.parse_args()

    if not args.sqlite.exists():
        print(f"sqlite not found: {args.sqlite}", file=sys.stderr)
        return 2

    skills = []
    if args.skills:
        skills = [(s, [], "user") for s in args.skills]
    else:
        skills = BASKETS[args.basket]

    report = {
        "label": args.label,
        "sqlite": str(args.sqlite),
        "sqlite_size_mb": round(args.sqlite.stat().st_size / 1e6, 1),
        "trim": list(args.trim),
        "env": {k: v for k, v in os.environ.items() if k.startswith("NSYS_AI_")},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    print(f"[bench] sqlite={args.sqlite} ({report['sqlite_size_mb']} MB)", file=sys.stderr)
    print(f"[bench] step 1/2 — cache build (no_rebuild={args.no_rebuild})", file=sys.stderr)
    report["cache"] = measure_cache_build(args.sqlite, args.no_rebuild)
    if report["cache"].get("elapsed_s") is not None:
        print(f"[bench]   cache build took {report['cache']['elapsed_s']}s", file=sys.stderr)

    print(f"[bench] step 2/2 — running {len(skills)} skills with --trim {args.trim[0]} {args.trim[1]}", file=sys.stderr)
    skill_results = []
    for name, extra, _cat in skills:
        r = measure_skill(args.sqlite, name, extra, tuple(args.trim))
        marker = "OK " if r["exit"] == 0 else "FAIL"
        print(f"[bench]   {marker} {name:<30s} {r['elapsed_s']:>7.2f}s rows={r['rows']}", file=sys.stderr)
        skill_results.append(r)
    report["skills"] = skill_results
    report["skills_total_s"] = round(sum(r["elapsed_s"] for r in skill_results), 2)
    report["skills_ok"] = sum(1 for r in skill_results if r["exit"] == 0)
    report["skills_fail"] = sum(1 for r in skill_results if r["exit"] != 0)

    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text)
        print(f"[bench] wrote {args.out}", file=sys.stderr)
    print(text)

    print(f"[bench] summary: cache={report['cache'].get('elapsed_s')}s  skills_total={report['skills_total_s']}s  ok={report['skills_ok']}/{len(skills)}", file=sys.stderr)
    return 0 if report["skills_fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
