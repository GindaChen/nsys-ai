import json
from pathlib import Path

import pytest

from nsys_ai.timeline_benchmark import (
    check_timeline_web_budget,
    run_timeline_web_benchmark,
)

DISTCA_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "example-20-megatron-distca"
)
DISTCA_SQLITE = DISTCA_DIR / "output" / "megatron_distca.sqlite"
BUDGET_JSON = DISTCA_DIR / "timeline_web_perf_budget.json"


@pytest.mark.skipif(not DISTCA_SQLITE.exists(), reason="distca example sqlite not found")
def test_distca_timeline_web_perf_budget():
    budget = json.loads(BUDGET_JSON.read_text())
    report = run_timeline_web_benchmark(
        DISTCA_SQLITE,
        tile_start_s=37.0,
        tile_end_s=42.0,
        nvtx_gpu=3,
        runs=1,
    )

    failures = check_timeline_web_budget(report, budget)
    assert not failures, " ; ".join(failures)

    # Sanity + regression invariants:
    # - kernel-only path stays very fast
    # - NVTX tile annotation remains the expensive step
    assert report["kernels_total"] > 0
    assert report["tile_kernels_total"] > 0
    assert report["tile_nvtx_spans"] > 0
    assert report["tile_nvtx_gpu_s"] > report["kernel_cache_build_full_s"]
