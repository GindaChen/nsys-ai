#!/usr/bin/env python3
"""
Benchmark timeline-web performance on the distca example profile.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


DEFAULT_PROFILE = Path(__file__).resolve().parent / "output" / "megatron_distca.sqlite"
DEFAULT_BUDGET = Path(__file__).resolve().parent / "timeline_web_perf_budget.json"


def main() -> int:
    from nsys_ai.timeline_benchmark import (
        check_timeline_web_budget,
        run_timeline_web_benchmark,
    )

    p = argparse.ArgumentParser(description="Benchmark timeline-web phases on distca profile")
    p.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    p.add_argument("--runs", type=int, default=1, help="Repeat each phase N times, report median")
    p.add_argument("--tile-start", type=float, default=37.0)
    p.add_argument("--tile-end", type=float, default=42.0)
    p.add_argument("--gpu", type=int, default=3, help="GPU id for NVTX tile benchmark")
    p.add_argument("--budget", type=Path, default=DEFAULT_BUDGET)
    p.add_argument("--check", action="store_true", help="Fail if budget exceeded")
    args = p.parse_args()

    if not args.profile.exists():
        print(f"Profile not found: {args.profile}", file=sys.stderr)
        return 2

    report = run_timeline_web_benchmark(
        args.profile,
        tile_start_s=args.tile_start,
        tile_end_s=args.tile_end,
        nvtx_gpu=args.gpu,
        runs=args.runs,
    )
    print(json.dumps(report, indent=2))

    if args.check:
        if not args.budget.exists():
            print(f"Budget not found: {args.budget}", file=sys.stderr)
            return 2
        budget = json.loads(args.budget.read_text())
        failures = check_timeline_web_budget(report, budget)
        if failures:
            print("\nBudget check FAILED:", file=sys.stderr)
            for f in failures:
                print(f"  - {f}", file=sys.stderr)
            return 1
        print("\nBudget check PASSED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
