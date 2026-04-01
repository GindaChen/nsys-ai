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

    with Profile(str(DISTCA_SQLITE), cache_mode="parquet") as prof:
        gpu_payload = build_timeline_gpu_data(prof, gpu, trim)[0]
        events = gpu_payload["kernels"]
        kernels = [e for e in events if e.get("type") == "kernel"]

    hit = [
        k for k in kernels if k["start_ns"] <= target_ns <= k["end_ns"] and "flash_bwd" in k["name"]
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


@pytest.mark.skipif(not DISTCA_SQLITE.exists(), reason="distca example sqlite not found")
def test_distca_timeline_web_includes_memcpy_memset_23s_to_24s():
    trim = (int(23.0 * 1e9), int(24.0 * 1e9))

    with Profile(str(DISTCA_SQLITE), cache_mode="parquet") as prof:
        devices = list(prof.meta.devices)
        payload_by_gpu = {
            gpu: build_timeline_gpu_data(prof, gpu, trim)[0]["kernels"] for gpu in devices
        }

    conn = sqlite3.connect(str(DISTCA_SQLITE))
    conn.row_factory = sqlite3.Row
    try:
        total_mem_events = 0
        for gpu in devices:
            events = payload_by_gpu[gpu]
            memcpy_events = [e for e in events if e.get("type") == "memcpy"]
            memset_events = [e for e in events if e.get("type") == "memset"]
            total_mem_events += len(memcpy_events) + len(memset_events)

            memcpy_count = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM CUPTI_ACTIVITY_KIND_MEMCPY
                WHERE deviceId = ? AND [end] >= ? AND start <= ?
                """,
                (gpu, trim[0], trim[1]),
            ).fetchone()["n"]
            memset_count = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM CUPTI_ACTIVITY_KIND_MEMSET
                WHERE deviceId = ? AND [end] >= ? AND start <= ?
                """,
                (gpu, trim[0], trim[1]),
            ).fetchone()["n"]

            assert len(memcpy_events) == memcpy_count
            assert len(memset_events) == memset_count
    finally:
        conn.close()

    assert total_mem_events > 0


@pytest.mark.skipif(not DISTCA_SQLITE.exists(), reason="distca example sqlite not found")
def test_distca_nvtx_path_depth_stable_across_adjacent_tiles():
    gpu = 3
    left_trim = (int(40.0 * 1e9), int(45.0 * 1e9))
    right_trim = (int(45.0 * 1e9), int(50.0 * 1e9))

    with Profile(str(DISTCA_SQLITE), cache_mode="parquet") as prof:
        left_spans = build_timeline_gpu_data(
            prof, gpu, left_trim, include_kernels=False, include_nvtx=True
        )[0]["nvtx_spans"]
        right_spans = build_timeline_gpu_data(
            prof, gpu, right_trim, include_kernels=False, include_nvtx=True
        )[0]["nvtx_spans"]

    # Boundary NVTX span we observed drifting from depth 1->0 when parent context
    # was missing in one tile.
    def _pick(spans):
        for s in spans:
            if (
                s["name"] == "TransformerLayer._forward_attention.self_attention"
                and 44.97e9 <= s["start"] <= 44.99e9
                and 45.02e9 <= s["end"] <= 45.03e9
            ):
                return s
        return None

    left = _pick(left_spans)
    right = _pick(right_spans)
    assert left is not None and right is not None
    assert left["depth"] == right["depth"]
    assert left["path"] == right["path"]

    right_bad_roots = [
        s
        for s in right_spans
        if (
            s["name"] == "TransformerLayer._forward_attention.self_attention"
            and s.get("depth", 0) == 0
            and 44.97e9 <= s["start"] <= 44.99e9
        )
    ]
    assert not right_bad_roots
