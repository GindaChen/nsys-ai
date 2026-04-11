"""Tests for per-layer NVTX breakdown (PR #74).

Tests cover:
  - Layer 1: nvtx_attribution with depth and path
  - Layer 2: nvtx_layer_breakdown skill with kernel composition
  - Layer 3: root_cause_matcher layer-aware patterns
  - Edge cases: no NVTX, flat NVTX
"""

import sqlite3

# ---------------------------------------------------------------------------
# Layer 1: nvtx_attribution depth + path
# ---------------------------------------------------------------------------


def test_nvtx_attribution_returns_depth_and_path(nested_nvtx_conn):
    """Attribution results should include nvtx_depth and nvtx_path fields."""
    from nsys_ai.nvtx_attribution import attribute_kernels_to_nvtx

    rows = attribute_kernels_to_nvtx(nested_nvtx_conn)
    assert len(rows) > 0
    for r in rows:
        assert "nvtx_depth" in r, f"Missing nvtx_depth in {r}"
        assert "nvtx_path" in r, f"Missing nvtx_path in {r}"
        assert isinstance(r["nvtx_depth"], int)
        assert isinstance(r["nvtx_path"], str)
        assert " > " in r["nvtx_path"] or r["nvtx_depth"] == 0


def test_nvtx_attribution_innermost_correct(nested_nvtx_conn):
    """Attribution should pick innermost NVTX (e.g., 'attention' not 'forward')."""
    from nsys_ai.nvtx_attribution import attribute_kernels_to_nvtx

    rows = attribute_kernels_to_nvtx(nested_nvtx_conn)
    # Compute kernels should be attributed to attention/mlp (depth 3),
    # NOT to forward/backward/train_step
    compute_texts = {r["nvtx_text"] for r in rows if "gemm" in r["kernel_name"].lower()}
    assert "attention" in compute_texts or "mlp" in compute_texts
    assert "forward" not in compute_texts
    assert "train_step" not in compute_texts


def test_nvtx_attribution_nccl_attributed_to_allreduce(nested_nvtx_conn):
    """NCCL kernels should be attributed to 'allreduce' NVTX."""
    from nsys_ai.nvtx_attribution import attribute_kernels_to_nvtx

    rows = attribute_kernels_to_nvtx(nested_nvtx_conn)
    nccl_texts = {r["nvtx_text"] for r in rows if "nccl" in r["kernel_name"].lower()}
    assert "allreduce" in nccl_texts


def test_nvtx_attribution_path_hierarchy(nested_nvtx_conn):
    """nvtx_path should reflect the full NVTX nesting (e.g. 'train_step > forward > layer_0 > attention')."""
    from nsys_ai.nvtx_attribution import attribute_kernels_to_nvtx

    rows = attribute_kernels_to_nvtx(nested_nvtx_conn)
    paths = {r["nvtx_path"] for r in rows}
    # At least one path should contain the full hierarchy
    has_deep_path = any(">" in p and len(p.split(" > ")) >= 3 for p in paths)
    assert has_deep_path, f"No deep paths found in: {paths}"


# ---------------------------------------------------------------------------
# Layer 2: nvtx_layer_breakdown skill with kernel composition
# ---------------------------------------------------------------------------


def test_nvtx_layer_breakdown_kernel_composition(nested_nvtx_conn):
    """Layer breakdown should split kernel time into compute and NCCL."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("nvtx_layer_breakdown")
    rows = skill.execute(nested_nvtx_conn, auto_depth=False)
    rows = [r for r in rows if not r.get("_detection_meta")]
    assert len(rows) > 0

    # Check that all rows have the new fields
    for r in rows:
        assert "compute_ms" in r
        assert "nccl_ms" in r
        assert "nccl_pct" in r
        assert "nvtx_depth" in r
        assert "nvtx_path" in r

    # 'attention' and 'mlp' should have compute_ms > 0 and nccl_ms == 0
    attention_rows = [r for r in rows if r["nvtx_region"] == "attention"]
    mlp_rows = [r for r in rows if r["nvtx_region"] == "mlp"]
    for region_rows in [attention_rows, mlp_rows]:
        if region_rows:
            assert region_rows[0]["compute_ms"] > 0
            assert region_rows[0]["nccl_ms"] == 0

    # 'allreduce' should have nccl_ms > 0
    allreduce_rows = [r for r in rows if r["nvtx_region"] == "allreduce"]
    if allreduce_rows:
        assert allreduce_rows[0]["nccl_ms"] > 0
        assert allreduce_rows[0]["nccl_pct"] > 0


def test_nvtx_layer_breakdown_nccl_percentage(nested_nvtx_conn):
    """nccl_pct should be correct per-layer."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("nvtx_layer_breakdown")
    rows = skill.execute(nested_nvtx_conn)
    rows = [r for r in rows if not r.get("_detection_meta")]
    for r in rows:
        if r["total_gpu_ms"] > 0:
            expected_pct = round(100 * r["nccl_ms"] / r["total_gpu_ms"], 1)
            assert abs(r["nccl_pct"] - expected_pct) < 0.2, (
                f"nccl_pct mismatch for {r['nvtx_region']}: "
                f"got {r['nccl_pct']}, expected {expected_pct}"
            )


def test_nvtx_layer_breakdown_depth_filtering(nested_nvtx_conn):
    """depth=3 should return only innermost NVTX (attention, mlp, allreduce)."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("nvtx_layer_breakdown")
    rows = skill.execute(nested_nvtx_conn, depth=3)
    assert len(rows) > 0
    regions = {r["nvtx_region"] for r in rows}
    # Should contain innermost regions
    assert regions <= {"attention", "mlp", "allreduce"}
    # Should NOT contain outer regions
    assert "forward" not in regions
    assert "train_step" not in regions
    assert "layer_0" not in regions


def test_nvtx_layer_breakdown_format(nested_nvtx_conn):
    """Format output should include new columns."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("nvtx_layer_breakdown")
    rows = skill.execute(nested_nvtx_conn)
    text = skill.format_rows(rows)
    assert "NVTX Region" in text
    assert "Compute" in text
    assert "NCCL" in text
    assert "NCCL%" in text


def test_nvtx_layer_breakdown_compact_matches_full_metrics(nested_nvtx_conn):
    """compact report must keep the same per-row metrics as full for the returned top-N."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("nvtx_layer_breakdown")
    limit = 10
    full_rows = skill.execute(nested_nvtx_conn, limit=limit, report="full")
    compact_rows = skill.execute(nested_nvtx_conn, limit=limit, report="compact")
    full_data = [r for r in full_rows if not r.get("_detection_meta")]
    compact_data = [r for r in compact_rows if not r.get("_detection_meta")]
    assert len(full_data) == len(compact_data) > 0
    keys = (
        "nvtx_region",
        "nvtx_depth",
        "nvtx_path",
        "kernel_count",
        "total_gpu_ms",
        "compute_ms",
        "nccl_ms",
        "nccl_pct",
        "tc_achieved_pct",
        "avg_kernel_ms",
        "max_kernel_ms",
        "is_outlier",
    )
    for fr, cr in zip(full_data, compact_data, strict=True):
        for k in keys:
            assert fr[k] == cr[k], (k, fr.get(k), cr.get(k))
        assert "top_kernels" in fr and fr["top_kernels"]
        assert "top_kernels" not in cr
        assert cr.get("kernels_included") is False

    full_meta = next((r for r in full_rows if r.get("_detection_meta")), None)
    compact_meta = next((r for r in compact_rows if r.get("_detection_meta")), None)
    if full_meta and compact_meta and full_meta.get("layer_names"):
        assert compact_meta.get("layer_names_omitted") is True
        assert compact_meta.get("layer_names_total") == len(full_meta["layer_names"])
        assert "layer_names" not in compact_meta


def test_nvtx_layer_breakdown_invalid_report(nested_nvtx_conn):
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("nvtx_layer_breakdown")
    rows = skill.execute(nested_nvtx_conn, report="bogus")
    assert len(rows) == 1
    assert "error" in rows[0]


# ---------------------------------------------------------------------------
# Layer 3: root_cause_matcher layer-aware patterns
# ---------------------------------------------------------------------------


def test_root_cause_layer_nccl_hotspot():
    """_check_layer_nccl_hotspot should fire when one layer dominates NCCL."""
    from nsys_ai.skills.builtins.root_cause_matcher import _check_layer_nccl_hotspot

    # Simulate: layer_1_bwd has 80ms NCCL, layer_0_bwd has 5ms NCCL
    layer_data = [
        {
            "nvtx_region": "layer_1_bwd",
            "nccl_ms": 80,
            "compute_ms": 10,
            "nvtx_path": "backward > layer_1_bwd",
        },
        {
            "nvtx_region": "layer_0_bwd",
            "nccl_ms": 5,
            "compute_ms": 15,
            "nvtx_path": "backward > layer_0_bwd",
        },
        {
            "nvtx_region": "attention",
            "nccl_ms": 0,
            "compute_ms": 30,
            "nvtx_path": "forward > attention",
        },
    ]
    findings = _check_layer_nccl_hotspot(layer_data)
    assert len(findings) >= 1
    assert findings[0]["pattern"] == "Layer NCCL Hotspot"
    assert "layer_1_bwd" in findings[0]["evidence"]
    assert findings[0]["severity"] == "warning"


def test_root_cause_layer_nccl_hotspot_no_fire():
    """_check_layer_nccl_hotspot should NOT fire when NCCL is balanced."""
    from nsys_ai.skills.builtins.root_cause_matcher import _check_layer_nccl_hotspot

    layer_data = [
        {"nvtx_region": "layer_0", "nccl_ms": 10, "compute_ms": 30},
        {"nvtx_region": "layer_1", "nccl_ms": 12, "compute_ms": 28},
        {"nvtx_region": "layer_2", "nccl_ms": 11, "compute_ms": 29},
    ]
    findings = _check_layer_nccl_hotspot(layer_data)
    assert len(findings) == 0


def test_root_cause_pipeline_imbalance():
    """_check_pipeline_imbalance should fire when compute varies > 3×."""
    from nsys_ai.skills.builtins.root_cause_matcher import _check_pipeline_imbalance

    layer_data = [
        {"nvtx_region": "layer_heavy", "compute_ms": 30.0, "nccl_ms": 0},
        {"nvtx_region": "layer_light", "compute_ms": 3.0, "nccl_ms": 0},
    ]
    findings = _check_pipeline_imbalance(layer_data)
    assert len(findings) == 1
    assert findings[0]["pattern"] == "Pipeline Imbalance"
    assert "layer_heavy" in findings[0]["evidence"]
    assert "layer_light" in findings[0]["evidence"]


def test_root_cause_pipeline_imbalance_no_fire():
    """_check_pipeline_imbalance should NOT fire when compute is balanced."""
    from nsys_ai.skills.builtins.root_cause_matcher import _check_pipeline_imbalance

    layer_data = [
        {"nvtx_region": "layer_0", "compute_ms": 10.0, "nccl_ms": 5},
        {"nvtx_region": "layer_1", "compute_ms": 12.0, "nccl_ms": 4},
    ]
    findings = _check_pipeline_imbalance(layer_data)
    assert len(findings) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_no_nvtx_graceful_fallback():
    """Profile with no NVTX should return empty list, no crash."""
    from nsys_ai.skills.registry import get_skill

    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO StringIds VALUES (1, 'kernel_A');
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            globalPid INT, deviceId INT, streamId INT, correlationId INT,
            start INT, [end] INT, shortName INT, demangledName INT,
            gridX INT, gridY INT, gridZ INT, blockX INT, blockY INT, blockZ INT
        );
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
            (0, 0, 7, 1, 1000000, 2000000, 1, 1, 1,1,1,1,1,1);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            globalTid INT, correlationId INT, start INT, [end] INT, nameId INT
        );
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (100, 1, 500000, 1000000, 0);
        CREATE TABLE NVTX_EVENTS (
            globalTid INT, start INT, [end] INT, text TEXT, eventType INT, rangeId INT
        );
    """)

    skill = get_skill("nvtx_layer_breakdown")
    rows = skill.execute(conn)
    assert isinstance(rows, list)
    assert len(rows) == 0
    conn.close()


def test_flat_nvtx_no_nesting():
    """Flat NVTX (all depth 0) should work without errors."""
    from nsys_ai.skills.registry import get_skill

    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO StringIds VALUES (1, 'kernel_A'), (24, 'cudaLaunchKernel');
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            globalPid INT, deviceId INT, streamId INT, correlationId INT,
            start INT, [end] INT, shortName INT, demangledName INT,
            gridX INT, gridY INT, gridZ INT, blockX INT, blockY INT, blockZ INT
        );
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
            (0, 0, 7, 1, 1000000, 2000000, 1, 1, 1,1,1,1,1,1);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            globalTid INT, correlationId INT, start INT, [end] INT, nameId INT
        );
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (100, 1, 500000, 1000000, 24);
        CREATE TABLE NVTX_EVENTS (
            globalTid INT, start INT, [end] INT, text TEXT, eventType INT, rangeId INT
        );
        INSERT INTO NVTX_EVENTS VALUES (100, 0, 3000000, 'flat_region', 59, 0);
    """)

    skill = get_skill("nvtx_layer_breakdown")
    rows = skill.execute(conn)
    # Filter out detection metadata if present
    data_rows = [r for r in rows if not r.get("_detection_meta")]
    assert len(data_rows) >= 1
    assert data_rows[0]["nvtx_region"] == "flat_region"
    assert data_rows[0]["nvtx_depth"] == 0
    assert data_rows[0]["compute_ms"] > 0
    conn.close()


# ---------------------------------------------------------------------------
# Layer auto-detection tests (nvtx_layer_detect module)
# ---------------------------------------------------------------------------


def test_detect_layer_depth_numbered_pattern(nested_nvtx_conn):
    """detect_layer_depth should find numbered layers (layer_0, layer_1) at depth 2."""
    from nsys_ai.nvtx_attribution import attribute_kernels_to_nvtx
    from nsys_ai.nvtx_layer_detect import detect_layer_depth

    rows = attribute_kernels_to_nvtx(nested_nvtx_conn)
    result = detect_layer_depth(rows)
    assert result["layer_depth"] == 2
    assert result["detection_method"] == "numbered_pattern"
    # Should find layer_0, layer_1, layer_0_bwd, layer_1_bwd
    assert any("layer_0" in n for n in result["layer_names"])
    assert any("layer_1" in n for n in result["layer_names"])
    assert result["confidence"] > 0


def test_detect_layer_depth_repeated_siblings(emit_nvtx_conn):
    """detect_layer_depth should find repeated ops (aten::linear × 3) in emit_nvtx."""
    from nsys_ai.nvtx_attribution import attribute_kernels_to_nvtx
    from nsys_ai.nvtx_layer_detect import detect_layer_depth

    rows = attribute_kernels_to_nvtx(emit_nvtx_conn)
    result = detect_layer_depth(rows)
    assert result["layer_depth"] is not None
    assert result["detection_method"] == "repeated_siblings"
    assert result["confidence"] > 0


def test_detect_layer_depth_empty():
    """Empty attribution rows should return fallback."""
    from nsys_ai.nvtx_layer_detect import detect_layer_depth

    result = detect_layer_depth([])
    assert result["layer_depth"] is None
    assert result["detection_method"] == "fallback_top_level"


# ---------------------------------------------------------------------------
# Auto-depth integration tests
# ---------------------------------------------------------------------------


def test_auto_depth_integration(nested_nvtx_conn):
    """Auto-depth should produce _detection_meta + depth-2 results."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("nvtx_layer_breakdown")
    rows = skill.execute(nested_nvtx_conn)
    assert len(rows) > 0

    # First row should be detection metadata
    assert rows[0].get("_detection_meta") is True
    assert rows[0]["layer_depth"] == 2
    assert rows[0]["detection_method"] == "numbered_pattern"

    # Data rows should be at depth 2
    data_rows = [r for r in rows if not r.get("_detection_meta")]
    assert len(data_rows) > 0
    for r in data_rows:
        assert r["nvtx_depth"] == 2


def test_auto_depth_disabled(nested_nvtx_conn):
    """auto_depth=False should return all regions without detection metadata."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("nvtx_layer_breakdown")
    rows_auto = skill.execute(nested_nvtx_conn)
    rows_no_auto = skill.execute(nested_nvtx_conn, auto_depth=False)
    assert len(rows_no_auto) > 0

    # Should NOT have detection metadata
    assert not rows_no_auto[0].get("_detection_meta")

    # Without auto-depth filtering, we should get MORE regions than with it
    # (auto groups at depth 2, no-auto returns all innermost at depth 3)
    auto_data = [r for r in rows_auto if not r.get("_detection_meta")]
    assert len(rows_no_auto) >= len(auto_data)


# ---------------------------------------------------------------------------
# Top kernels + outlier tests
# ---------------------------------------------------------------------------


def test_top_kernels_embedded(nested_nvtx_conn):
    """Each result row should have top_kernels list with kernel_name + total_ms."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("nvtx_layer_breakdown")
    rows = skill.execute(nested_nvtx_conn)
    data_rows = [r for r in rows if not r.get("_detection_meta")]
    assert len(data_rows) > 0

    for r in data_rows:
        assert "top_kernels" in r
        assert isinstance(r["top_kernels"], list)
        for tk in r["top_kernels"]:
            assert "kernel_name" in tk
            assert "total_ms" in tk
            assert isinstance(tk["total_ms"], float)


def test_is_outlier_basic():
    """is_outlier should flag a clear outlier."""
    from nsys_ai.nvtx_layer_detect import is_outlier

    assert is_outlier(100, [10, 11, 12, 13]) is True


def test_is_outlier_no_fire():
    """is_outlier should NOT flag normal values."""
    from nsys_ai.nvtx_layer_detect import is_outlier

    assert is_outlier(15, [10, 11, 12, 13, 14, 15]) is False


def test_is_outlier_small_sample():
    """Small sample (n<4) should fallback to 2× median."""
    from nsys_ai.nvtx_layer_detect import is_outlier

    # n=3: median=10, threshold=20
    assert is_outlier(25, [10, 10, 10]) is True
    assert is_outlier(15, [10, 10, 10]) is False


def test_root_cause_layer_outlier():
    """_check_layer_outlier should fire on skewed layer data."""
    from nsys_ai.skills.builtins.root_cause_matcher import _check_layer_outlier

    # is_outlier needs value > median * 1.5 AND > Q3 + 1.5 * IQR
    # With n=4 and outlier IN the list, Q3=outlier itself → fence too high.
    # Use 6+ normal layers so Q3 stays at ~12, IQR~2, fence=15,
    # and 100 > 15 AND 100 > 12*1.5=18 → both True.
    layer_data = [
        {
            "nvtx_region": "layer_0",
            "total_gpu_ms": 10.0,
            "compute_ms": 10.0,
            "top_kernels": [{"kernel_name": "gemm_k", "total_ms": 8.0}],
        },
        {"nvtx_region": "layer_1", "total_gpu_ms": 10.5, "compute_ms": 10.5, "top_kernels": []},
        {"nvtx_region": "layer_2", "total_gpu_ms": 11.0, "compute_ms": 11.0, "top_kernels": []},
        {"nvtx_region": "layer_3", "total_gpu_ms": 11.5, "compute_ms": 11.5, "top_kernels": []},
        {"nvtx_region": "layer_4", "total_gpu_ms": 12.0, "compute_ms": 12.0, "top_kernels": []},
        {"nvtx_region": "layer_5", "total_gpu_ms": 12.5, "compute_ms": 12.5, "top_kernels": []},
        {
            "nvtx_region": "layer_heavy",
            "total_gpu_ms": 100.0,
            "compute_ms": 100.0,
            "top_kernels": [{"kernel_name": "slow_kernel", "total_ms": 90.0}],
        },
    ]
    findings = _check_layer_outlier(layer_data)
    assert len(findings) >= 1
    assert findings[0]["pattern"] == "Layer Outlier"
    assert "layer_heavy" in findings[0]["evidence"]
    assert "slow_kernel" in findings[0]["evidence"]


def test_root_cause_layer_outlier_no_fire():
    """_check_layer_outlier should NOT fire when layers are balanced."""
    from nsys_ai.skills.builtins.root_cause_matcher import _check_layer_outlier

    layer_data = [
        {"nvtx_region": "layer_0", "total_gpu_ms": 10.0, "compute_ms": 10.0},
        {"nvtx_region": "layer_1", "total_gpu_ms": 11.0, "compute_ms": 11.0},
        {"nvtx_region": "layer_2", "total_gpu_ms": 12.0, "compute_ms": 12.0},
        {"nvtx_region": "layer_3", "total_gpu_ms": 10.5, "compute_ms": 10.5},
    ]
    findings = _check_layer_outlier(layer_data)
    assert len(findings) == 0
