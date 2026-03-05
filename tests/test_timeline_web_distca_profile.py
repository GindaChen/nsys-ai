import sqlite3
from pathlib import Path

import pytest

from nsys_ai.profile import Profile
from nsys_ai.viewer import build_timeline_gpu_data

DISTCA_SQLITE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "example-20-megatron-distca"
    / "output"
    / "megatron_distca.sqlite"
)


@pytest.mark.skipif(not DISTCA_SQLITE.exists(), reason="distca example sqlite not found")
def test_distca_timeline_web_contains_flash_backward_on_gpu3():
    target_ns = int(50_232.846 * 1e6)
    trim = (int(50_232.700 * 1e6), int(50_233.300 * 1e6))
    gpu = 3

    with Profile(str(DISTCA_SQLITE)) as prof:
        gpu_payload = build_timeline_gpu_data(prof, gpu, trim)[0]
        kernels = gpu_payload["kernels"]

    hit = [
        k for k in kernels
        if k["start_ns"] <= target_ns <= k["end_ns"] and "flash_bwd" in k["name"]
    ]
    assert hit, "Expected flash backward kernel at ~50,232.846ms on GPU3"

    conn = sqlite3.connect(str(DISTCA_SQLITE))
    conn.row_factory = sqlite3.Row
    try:
        db_overlap_count = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE deviceId = ? AND [end] >= ? AND start <= ?
            """,
            (gpu, trim[0], trim[1]),
        ).fetchone()["n"]
    finally:
        conn.close()

    assert len(kernels) == db_overlap_count
