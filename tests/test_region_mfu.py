import sqlite3

from nsys_ai.region_mfu import (
    compute_mfu_metrics_for_region,
    compute_region_mfu_from_conn,
    compute_theoretical_flops,
    find_nvtx_ranges,
    get_region_kernels,
    select_nvtx_occurrence,
    summarize_region_kernel_times,
)


def _make_min_region_db(path: str):
    """Create a minimal DB with NVTX + RUNTIME + KERNEL for region_mfu tests."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE StringIds(id INT PRIMARY KEY, value TEXT)")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
        "start INT, [end] INT, deviceId INT, streamId INT, correlationId INT, "
        "shortName INT, demangledName INT)"
    )
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(globalTid INT, correlationId INT, start INT, [end] INT)"
    )
    conn.execute("CREATE TABLE NVTX_EVENTS(text TEXT, globalTid INT, start INT, [end] INT)")
    # GPU tables for peak TFLOPS lookup
    conn.execute(
        "CREATE TABLE TARGET_INFO_GPU(id INTEGER PRIMARY KEY, name TEXT, busLocation TEXT, "
        "totalMemory INTEGER, smCount INTEGER, chipName TEXT, memoryBandwidth INTEGER)"
    )
    conn.execute(
        "CREATE TABLE TARGET_INFO_CUDA_DEVICE(gpuId INTEGER, cudaId INTEGER, pid INTEGER, uuid TEXT, numMultiprocessors INTEGER)"
    )
    conn.execute("INSERT INTO TARGET_INFO_GPU(id, name) VALUES (0, 'NVIDIA H100 80GB HBM3')")
    conn.execute("INSERT INTO TARGET_INFO_CUDA_DEVICE(gpuId, cudaId) VALUES (0, 0)")

    conn.execute("INSERT INTO StringIds(id, value) VALUES (1,'k_flash'), (2,'k_flash_dem')")
    # One kernel 1000–2000 ns, correlationId 1 on device 0 / stream 7
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start, [end], deviceId, streamId, correlationId, shortName, demangledName) "
        "VALUES (1000, 2000, 0, 7, 1, 1, 2)"
    )
    # NVTX range 500–2500 containing the runtime launch
    conn.execute(
        "INSERT INTO NVTX_EVENTS(text, globalTid, start, [end]) VALUES ('FlashAttention', 1, 500, 2500)"
    )
    # Runtime 900–1000 so kernel 1000–2000 is inside the NVTX CPU span
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME(globalTid, correlationId, start, [end]) "
        "VALUES (1, 1, 900, 1000)"
    )
    conn.commit()
    conn.close()


def test_find_nvtx_ranges_and_select_occurrence(tmp_path):
    db = tmp_path / "nvtx.sqlite"
    _make_min_region_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        rows = find_nvtx_ranges(conn, "FlashAttention", match_mode="contains")
        assert rows
        chosen = select_nvtx_occurrence(rows, 1)
        assert "error" not in chosen
        assert chosen["text"] == "FlashAttention"
        assert chosen["occurrence_index"] == 1
    finally:
        conn.close()


def test_get_region_kernels_and_summarize(tmp_path):
    db = tmp_path / "kernels.sqlite"
    _make_min_region_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        nvtx_rows = find_nvtx_ranges(conn, "FlashAttention", match_mode="exact")
        chosen = select_nvtx_occurrence(nvtx_rows, 1)
        assert "error" not in chosen
        kernels = get_region_kernels(
            conn,
            nvtx_start_ns=chosen["start_ns"],
            nvtx_end_ns=chosen["end_ns"],
            global_tid=chosen.get("global_tid"),
            device_id=0,
        )
        assert len(kernels) == 1
        summary = summarize_region_kernel_times(kernels)
        assert summary["kernel_count"] == 1
        assert summary["kernel_sum_ns"] == 1000
        assert summary["kernel_union_ns"] == 1000
        assert summary["device_ids"] == [0]
        assert summary["stream_ids"] == [7]
    finally:
        conn.close()


def test_compute_mfu_metrics_for_region():
    out = compute_mfu_metrics_for_region(
        theoretical_flops=1e18,
        peak_tflops=989.0,
        wall_time_s=10.0,
        kernel_sum_s=10.0,
        kernel_union_s=10.0,
    )
    assert "error" not in out
    assert out["mfu_pct_wall"] == round(100.0 * ((1e18 / 10.0) / 1e12) / 989.0, 2)


def test_compute_region_mfu_from_conn_happy_path(tmp_path):
    db = tmp_path / "region.sqlite"
    _make_min_region_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        result = compute_region_mfu_from_conn(
            conn,
            str(db),
            "FlashAttention",
            theoretical_flops=1e18,
            peak_tflops=None,
            occurrence_index=1,
            device_id=0,
            match_mode="contains",
        )
        assert "error" not in result
        assert result["name"] == "FlashAttention"
        assert result["source"] == "nvtx"
        assert result["matched_text"] == "FlashAttention"
        assert result["kernel_count"] == 1
        assert result["wall_time_s"] > 0
        assert "mfu_pct_wall" in result
        assert "mfu_pct_kernel_union" in result
    finally:
        conn.close()


def _make_textid_region_db(path: str):
    """Create a DB using the newer textId→StringIds schema (n.text IS NULL)."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE StringIds(id INT PRIMARY KEY, value TEXT)")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
        "start INT, [end] INT, deviceId INT, streamId INT, correlationId INT, "
        "shortName INT, demangledName INT)"
    )
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(globalTid INT, correlationId INT, start INT, [end] INT)"
    )
    # textId column present → triggers has_text_id detection
    conn.execute(
        "CREATE TABLE NVTX_EVENTS(text TEXT, textId INT, globalTid INT, start INT, [end] INT)"
    )
    conn.execute(
        "CREATE TABLE TARGET_INFO_GPU(id INTEGER PRIMARY KEY, name TEXT, busLocation TEXT, "
        "totalMemory INTEGER, smCount INTEGER, chipName TEXT, memoryBandwidth INTEGER)"
    )
    conn.execute(
        "CREATE TABLE TARGET_INFO_CUDA_DEVICE(gpuId INTEGER, cudaId INTEGER, pid INTEGER, uuid TEXT, numMultiprocessors INTEGER)"
    )
    conn.execute("INSERT INTO TARGET_INFO_GPU(id, name) VALUES (0, 'NVIDIA H100 80GB HBM3')")
    conn.execute("INSERT INTO TARGET_INFO_CUDA_DEVICE(gpuId, cudaId) VALUES (0, 0)")

    # String IDs: 10='FlashAttnFwd', 1='k_flash', 2='k_flash_dem'
    conn.execute(
        "INSERT INTO StringIds(id, value) VALUES (1,'k_flash'), (2,'k_flash_dem'), (10,'FlashAttnFwd')"
    )
    # Kernel
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start, [end], deviceId, streamId, correlationId, shortName, demangledName) "
        "VALUES (1000, 2000, 0, 7, 1, 1, 2)"
    )
    # NVTX row: text IS NULL, textId=10 → resolved via StringIds to 'FlashAttnFwd'
    conn.execute(
        "INSERT INTO NVTX_EVENTS(text, textId, globalTid, start, [end]) VALUES (NULL, 10, 1, 500, 2500)"
    )
    # Runtime launch inside the NVTX span
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME(globalTid, correlationId, start, [end]) "
        "VALUES (1, 1, 900, 1000)"
    )
    conn.commit()
    conn.close()


def test_find_nvtx_ranges_with_textid_schema(tmp_path):
    """Ensure find_nvtx_ranges resolves names via textId→StringIds when n.text IS NULL."""
    db = tmp_path / "textid.sqlite"
    _make_textid_region_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        # exact match
        rows = find_nvtx_ranges(conn, "FlashAttnFwd", match_mode="exact")
        assert len(rows) == 1
        assert rows[0]["text"] == "FlashAttnFwd"

        # contains match
        rows = find_nvtx_ranges(conn, "FlashAttn", match_mode="contains")
        assert len(rows) == 1
        assert rows[0]["text"] == "FlashAttnFwd"
    finally:
        conn.close()


def test_compute_region_mfu_from_conn_textid_schema(tmp_path):
    """End-to-end test with textId schema variant."""
    db = tmp_path / "textid_mfu.sqlite"
    _make_textid_region_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        result = compute_region_mfu_from_conn(
            conn,
            str(db),
            "FlashAttnFwd",
            theoretical_flops=1e18,
            peak_tflops=None,
            occurrence_index=1,
            device_id=0,
            match_mode="exact",
        )
        assert "error" not in result, f"Unexpected error: {result}"
        assert result["matched_text"] == "FlashAttnFwd"
        assert result["kernel_count"] == 1
        assert "mfu_pct_wall" in result
    finally:
        conn.close()


def test_compute_region_mfu_from_conn_multi_gpu(tmp_path):
    """num_gpus=2 doubles effective peak and halves MFU vs num_gpus=1."""
    db = tmp_path / "multi_gpu.sqlite"
    _make_min_region_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        r1 = compute_region_mfu_from_conn(
            conn, str(db), "FlashAttention", 1e18,
            peak_tflops=None, num_gpus=1, device_id=0,
        )
        r2 = compute_region_mfu_from_conn(
            conn, str(db), "FlashAttention", 1e18,
            peak_tflops=None, num_gpus=2, device_id=0,
        )
        assert "error" not in r1 and "error" not in r2
        # num_gpus field
        assert r1["num_gpus"] == 1
        assert r2["num_gpus"] == 2
        # effective peak scales
        assert r2["effective_peak_tflops"] == r1["peak_tflops_per_gpu"] * 2
        # MFU halved with 2x peak
        assert abs(r2["mfu_pct_wall"] - r1["mfu_pct_wall"] / 2) < 0.01
    finally:
        conn.close()


def test_compute_region_mfu_kernel_mode(tmp_path):
    """source='kernel' queries kernels directly by shortName, no NVTX needed."""
    db = tmp_path / "kernel_mode.sqlite"
    _make_min_region_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        result = compute_region_mfu_from_conn(
            conn,
            str(db),
            "k_flash",  # matches kernel shortName via StringIds
            theoretical_flops=1e18,
            source="kernel",
            peak_tflops=None,
            device_id=0,
        )
        assert "error" not in result, f"Unexpected error: {result}"
        assert result["source"] == "kernel"
        assert result["kernel_count"] == 1
        assert "mfu_pct_wall" in result
        assert "mfu_pct_kernel_union" in result
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# compute_theoretical_flops tests
# ---------------------------------------------------------------------------


def test_compute_theoretical_flops_attention():
    """Flash attention FLOPs: 4 * S^2 * H * L."""
    result = compute_theoretical_flops(
        "attention", hidden_dim=4096, seq_len=131072, num_layers=32
    )
    assert "error" not in result
    # 4 * 131072^2 * 4096 = 2.81474976e14 per layer, * 32 = 9.00719924e15
    expected = 4 * 131072 * 131072 * 4096 * 32
    assert result["theoretical_flops"] == expected
    assert result["operation"] == "attention"
    assert "formula" in result


def test_compute_theoretical_flops_full_layer():
    """Full layer = attention + qkv_proj + output_proj + mlp."""
    result = compute_theoretical_flops(
        "full_layer", hidden_dim=4096, seq_len=1024, num_layers=1
    )
    assert "error" not in result
    H, S = 4096, 1024
    ffn = 4 * H  # default
    expected = (4 * S * S * H) + (6 * S * H * H) + (2 * S * H * H) + (4 * S * H * ffn)
    assert result["theoretical_flops"] == expected


def test_compute_theoretical_flops_linear():
    """Generic linear: 2 * M * N * K."""
    result = compute_theoretical_flops("linear", M=1024, N=2048, K=4096)
    assert "error" not in result
    assert result["theoretical_flops"] == 2 * 1024 * 2048 * 4096


def test_compute_theoretical_flops_invalid_operation():
    result = compute_theoretical_flops("invalid_op")
    assert "error" in result
    assert result["error"]["code"] == "INVALID_ARGUMENT"


def test_compute_theoretical_flops_missing_dims():
    result = compute_theoretical_flops("attention", hidden_dim=0, seq_len=0)
    assert "error" in result
    assert result["error"]["code"] == "INVALID_ARGUMENT"
