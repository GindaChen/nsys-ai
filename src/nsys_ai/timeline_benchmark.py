"""
timeline_benchmark.py - Benchmark helpers for timeline-web performance.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from .profile import Profile
from .viewer import build_timeline_gpu_data, generate_timeline_html
from .web import _filter_timeline_gpu_entry


def _measure_seconds(fn, runs: int) -> tuple[dict, object]:
    samples: list[float] = []
    last = None
    n = max(1, int(runs))
    for _ in range(n):
        t0 = time.perf_counter()
        last = fn()
        samples.append(time.perf_counter() - t0)
    return {
        "runs": n,
        "min_s": min(samples),
        "max_s": max(samples),
        "median_s": float(statistics.median(samples)),
    }, last


def run_timeline_web_benchmark(
    profile_path: str | Path,
    *,
    tile_start_s: float = 37.0,
    tile_end_s: float = 42.0,
    nvtx_gpu: int = 3,
    runs: int = 1,
) -> dict:
    """
    Benchmark timeline-web critical phases on a profile.

    Returns a report dict with key phase timings and useful counts.
    """
    p = Path(profile_path)
    start_ns = int(tile_start_s * 1e9)
    end_ns = int(tile_end_s * 1e9)

    with Profile(str(p)) as prof:
        devices = sorted(prof.meta.gpu_info.keys())
        full_range = prof.meta.time_range

        html_stats, html = _measure_seconds(
            lambda: generate_timeline_html(prof, devices, None),
            runs,
        )

        kernel_stats, prebuilt = _measure_seconds(
            lambda: build_timeline_gpu_data(
                prof,
                devices,
                full_range,
                include_kernels=True,
                include_nvtx=False,
            ),
            runs,
        )

        filter_stats, filtered = _measure_seconds(
            lambda: [
                _filter_timeline_gpu_entry(
                    g,
                    start_ns,
                    end_ns,
                    filter_kernels=True,
                    filter_nvtx=False,
                )
                for g in prebuilt
            ],
            runs,
        )

        nvtx_stats, nvtx_tile = _measure_seconds(
            lambda: build_timeline_gpu_data(
                prof,
                [nvtx_gpu],
                (start_ns, end_ns),
                include_kernels=False,
                include_nvtx=True,
            ),
            runs,
        )

    return {
        "profile": str(p),
        "devices": devices,
        "gpu_count": len(devices),
        "tile_start_s": tile_start_s,
        "tile_end_s": tile_end_s,
        "tile_gpu": nvtx_gpu,
        "runs": max(1, int(runs)),
        "html_shell_s": html_stats["median_s"],
        "kernel_cache_build_full_s": kernel_stats["median_s"],
        "tile_filter_kernels_s": filter_stats["median_s"],
        "tile_nvtx_gpu_s": nvtx_stats["median_s"],
        "html_size_bytes": len(html.encode("utf-8")),
        "kernels_total": sum(len(g.get("kernels", [])) for g in prebuilt),
        "tile_kernels_total": sum(len(g.get("kernels", [])) for g in filtered),
        "tile_nvtx_spans": len(nvtx_tile[0].get("nvtx_spans", [])) if nvtx_tile else 0,
        "samples": {
            "html_shell": html_stats,
            "kernel_cache_build_full": kernel_stats,
            "tile_filter_kernels": filter_stats,
            "tile_nvtx_gpu": nvtx_stats,
        },
    }


def check_timeline_web_budget(report: dict, budget: dict) -> list[str]:
    """
    Compare benchmark report against budget config.

    Budget format:
      {"max_seconds": {"metric_name": limit, ...}}
    """
    failures: list[str] = []
    limits = budget.get("max_seconds", {})
    for metric, limit in limits.items():
        if metric not in report:
            failures.append(f"missing metric '{metric}' in benchmark report")
            continue
        value = float(report[metric])
        if value > float(limit):
            failures.append(f"{metric}: {value:.3f}s > budget {float(limit):.3f}s")
    return failures
