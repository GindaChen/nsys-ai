def test_module_loading_execute(minimal_nsys_conn):
    """Test module_loading skill executes and correctly aggregates JIT events."""
    from nsys_ai.skills.registry import get_skill

    # Insert a synthetic module load runtime event
    minimal_nsys_conn.execute("INSERT INTO StringIds VALUES (999, 'cuModuleLoadData')")
    minimal_nsys_conn.execute(
        'INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (correlationId, start, "end", nameId) VALUES (10, 1000000, 2000000, 999)'
    )

    skill = get_skill("module_loading")
    rows = skill.execute(minimal_nsys_conn)
    assert len(rows) > 0
    names = [r["api_name"] for r in rows]
    assert "cuModuleLoadData" in names

    match = next(r for r in rows if r["api_name"] == "cuModuleLoadData")
    assert match["occurrences"] == 1
    assert match["total_ms"] == 1.0


def test_module_loading_duckdb(duckdb_conn):
    """Test module_loading works on DuckDB."""
    from nsys_ai.skills.registry import get_skill

    duckdb_conn.execute("INSERT INTO StringIds VALUES (999, 'cuModuleLoadData')")
    duckdb_conn.execute(
        'INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (correlationId, start, "end", nameId) VALUES (10, 1000000, 2000000, 999)'
    )

    skill = get_skill("module_loading")
    rows = skill.execute(duckdb_conn)
    names = [r["api_name"] for r in rows]
    assert "cuModuleLoadData" in names


def test_gc_impact_execute(minimal_nsys_conn):
    """Test gc_impact successfully processes both runtime and NVTX branches."""
    from nsys_ai.skills.registry import get_skill

    # Runtime branch: insert cudaFree
    minimal_nsys_conn.execute("INSERT INTO StringIds VALUES (998, 'cudaFree')")
    minimal_nsys_conn.execute(
        'INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (correlationId, start, "end", nameId) VALUES (11, 1000000, 3000000, 998)'
    )
    # NVTX textId branch
    minimal_nsys_conn.execute("INSERT INTO StringIds VALUES (997, 'GC collection generation 2')")
    minimal_nsys_conn.execute(
        'INSERT INTO NVTX_EVENTS (start, "end", text, textId, eventType) VALUES (5000000, 6000000, NULL, 997, 59)'
    )
    # NVTX text branch
    minimal_nsys_conn.execute(
        "INSERT INTO NVTX_EVENTS (start, \"end\", text, textId, eventType) VALUES (7000000, 8000000, 'GC phase 1', NULL, 59)"
    )

    skill = get_skill("gc_impact")
    rows = skill.execute(minimal_nsys_conn)

    names = [r["event_name"] for r in rows]
    assert "cudaFree" in names
    assert "GC collection generation 2" in names
    assert "GC phase 1" in names


def test_pipeline_bubble_metrics_sqlite(minimal_nsys_conn):
    """Test pipeline bubble metrics executes on SQLite with kernel, memcpy, and memset."""
    from nsys_ai.skills.registry import get_skill

    # Add a synthetic memset interval that partially overlaps
    minimal_nsys_conn.execute(
        'INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET (deviceId, start, "end") VALUES (0, 1500000, 2500000)'
    )

    skill = get_skill("pipeline_bubble_metrics")
    rows = skill.execute(minimal_nsys_conn)

    assert len(rows) > 0
    r = rows[0]
    assert "deviceId" in r
    assert "total_span_ms" in r
    assert "active_ms" in r
    assert "bubble_ms" in r
    assert "bubble_pct" in r
    assert r["bubble_pct"] >= 0.0


def test_pipeline_bubble_metrics_duckdb(duckdb_conn):
    """Test pipeline bubble math executes the same logic on DuckDB."""
    from nsys_ai.skills.registry import get_skill

    duckdb_conn.execute(
        'INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET (deviceId, start, "end") VALUES (0, 1500000, 2500000)'
    )

    skill = get_skill("pipeline_bubble_metrics")
    rows = skill.execute(duckdb_conn)

    assert len(rows) > 0
    assert "bubble_pct" in rows[0]


def test_pipeline_bubble_metrics_no_tables_graceful(minimal_nsys_conn):
    """Test pipeline bubble metrics abstains if all activity tables are omitted."""
    from nsys_ai.skills.base import is_abstention
    from nsys_ai.skills.registry import get_skill

    minimal_nsys_conn.execute("DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL")
    minimal_nsys_conn.execute("DROP TABLE CUPTI_ACTIVITY_KIND_MEMCPY")
    minimal_nsys_conn.execute("DROP TABLE CUPTI_ACTIVITY_KIND_MEMSET")

    skill = get_skill("pipeline_bubble_metrics")
    rows = skill.execute(minimal_nsys_conn)
    assert is_abstention(rows)


def _use_a100(conn):
    """Point the fixture's GPU row at an A100 so the chipName lookup resolves.

    TARGET_INFO_GPU already has id=0 from the fixture with chipName='TestChip',
    which is in no spec table.
    """
    conn.execute(
        "UPDATE TARGET_INFO_GPU SET name='NVIDIA A100-SXM4-80GB', "
        "chipName='GA100', memoryBandwidth=2039000000000 WHERE id=0"
    )


# The fixture's five kernels union to 4.5 ms, so achieved TFLOPS is
# theoretical_flops / 0.0045 / 1e12 and every FLOP count below is chosen from
# that. They used to be 1e15, which is 222,222 TFLOPS — 712x an A100's peak, and
# the skill classified it as good throughput (#385). A test that asserts on an
# impossible reading cannot notice when the impossible reading is the bug.
_FIXTURE_KERNEL_S = 0.0045


def test_arithmetic_intensity_execute(minimal_nsys_conn):
    """Test arithmetic_intensity produces a roofline classification."""
    from nsys_ai.skills.registry import get_skill

    _use_a100(minimal_nsys_conn)

    skill = get_skill("arithmetic_intensity")
    # 200 TFLOPS achieved against A100's 312 TFLOPS peak -> MFU 64%.
    rows = skill.execute(minimal_nsys_conn, theoretical_flops=200e12 * _FIXTURE_KERNEL_S)

    assert len(rows) == 1
    r = rows[0]
    assert "error" not in r
    assert "classification" in r
    assert "achieved_tflops" in r
    assert "mfu_pct" in r
    assert r["peak_fp16_tflops"] == 312.0  # GA100 lookup


def test_arithmetic_intensity_no_gpu_table(minimal_nsys_conn):
    """Test arithmetic_intensity falls back when TARGET_INFO_GPU is missing."""
    from nsys_ai.skills.registry import get_skill

    minimal_nsys_conn.execute("DROP TABLE TARGET_INFO_GPU")
    skill = get_skill("arithmetic_intensity")
    rows = skill.execute(
        minimal_nsys_conn,
        theoretical_flops=500e12 * _FIXTURE_KERNEL_S,
        peak_tflops=989.4,
        hbm_bw_gbps=3352,
    )

    assert len(rows) == 1
    r = rows[0]
    assert "error" not in r
    assert r["peak_fp16_tflops"] == 989.4  # User override


def test_arithmetic_intensity_abstains_when_achieved_exceeds_peak(minimal_nsys_conn):
    """An MFU over 100% is impossible, so the skill must refuse, not classify.

    Regression for #385: 989e12 FLOPs against a 989 TFLOPS peak came out as
    3651 TFLOPS achieved / 369% MFU and was reported as "High kernel throughput
    (likely compute-bound)" with a recommendation to go tune the kernel in NCU.
    """
    from nsys_ai.skills.base import is_abstention
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("arithmetic_intensity")
    rows = skill.execute(
        minimal_nsys_conn,
        theoretical_flops=3651.3e12 * _FIXTURE_KERNEL_S,
        peak_tflops=989.0,
        hbm_bw_gbps=3350,
    )

    assert is_abstention(rows), rows
    r = rows[0]
    # It must not classify and must not recommend an action.
    assert "classification" not in r
    assert "recommendation" not in r
    assert "severity" not in r

    reason = r["reason"]
    # The reason names the inconsistency and both inputs behind it.
    assert "3,651.3 TFLOPS" in reason
    assert "989.0 TFLOPS" in reason
    assert "3.7x" in reason
    assert "theoretical_flops" in reason
    assert "peak_tflops" in reason

    rendered = skill.format_rows(rows)
    assert "not applicable to this profile" in rendered
    assert "NCU" not in rendered
    assert "compute-bound" not in rendered


def test_arithmetic_intensity_abstains_on_a_one_percent_overshoot(minimal_nsys_conn):
    """No tolerance band: 1% over peak is as impossible as 300% over."""
    from nsys_ai.skills.base import is_abstention
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("arithmetic_intensity")
    rows = skill.execute(
        minimal_nsys_conn,
        theoretical_flops=1.01 * 312e12 * _FIXTURE_KERNEL_S,
        peak_tflops=312.0,
        hbm_bw_gbps=2039,
    )

    assert is_abstention(rows), rows
    assert "101.0%" in rows[0]["reason"]


def test_arithmetic_intensity_at_exactly_peak_still_classifies(minimal_nsys_conn):
    """100.0% MFU is the boundary and is reachable, so it must still be reported."""
    from nsys_ai.skills.base import is_abstention
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("arithmetic_intensity")
    rows = skill.execute(
        minimal_nsys_conn,
        theoretical_flops=312e12 * _FIXTURE_KERNEL_S,
        peak_tflops=312.0,
        hbm_bw_gbps=2039,
    )

    assert not is_abstention(rows)
    assert rows[0]["mfu_pct"] == 100.0
    assert rows[0]["classification"] == "High kernel throughput (likely compute-bound)"


def test_arithmetic_intensity_roofline_branch_also_abstains(minimal_nsys_conn):
    """bytes_moved reaches the other classification arm, off the same bad FLOPs.

    op_intensity divides the very ``theoretical_flops`` that produced the
    impossible throughput, so the roofline verdict is no more trustworthy than
    the MFU one and the guard has to sit ahead of both.
    """
    from nsys_ai.skills.base import is_abstention
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("arithmetic_intensity")
    rows = skill.execute(
        minimal_nsys_conn,
        theoretical_flops=3651.3e12 * _FIXTURE_KERNEL_S,
        peak_tflops=989.0,
        hbm_bw_gbps=3350,
        bytes_moved=1e9,
    )

    assert is_abstention(rows), rows


def test_arithmetic_intensity_abstains_when_peak_is_not_positive(minimal_nsys_conn):
    """A zero peak is a broken denominator, not a severely slow GPU."""
    from nsys_ai.skills.base import is_abstention
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("arithmetic_intensity")
    rows = skill.execute(
        minimal_nsys_conn,
        theoretical_flops=1e12,
        peak_tflops=0,
        hbm_bw_gbps=3350,
    )

    assert is_abstention(rows), rows
    assert "peak_tflops is 0" in rows[0]["reason"]
    assert "classification" not in rows[0]


def test_arithmetic_intensity_abstains_when_peak_unknown(minimal_nsys_conn):
    """An unknown GPU with no caller-supplied peak abstains, not an error row.

    This is the state ``doctor`` refuses on ("GPU model missing from CUPTI
    TARGET_INFO; MFU / efficiency cannot be computed"). The skill agrees, and
    says so through the abstention contract rather than a data row carrying an
    ``error`` key, which consumers treat as a result.
    """
    from nsys_ai.skills.base import is_abstention
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("arithmetic_intensity")
    # The fixture's chipName is 'TestChip', which is in no spec table.
    rows = skill.execute(minimal_nsys_conn, theoretical_flops=1e12)

    assert is_abstention(rows), rows
    assert "error" not in rows[0]
    reason = rows[0]["reason"]
    assert "TestChip" in reason
    assert "peak_tflops" in reason


def test_arithmetic_intensity_unknown_gpu_still_uses_a_supplied_peak(minimal_nsys_conn):
    """The converse: an unknown model must not veto a peak the caller supplied.

    Refusing on the label alone would make the documented override useless on
    exactly the profiles that need it — an unlisted GPU, or an export with no
    chipName.
    """
    from nsys_ai.skills.base import is_abstention
    from nsys_ai.skills.registry import get_skill

    # The state the issue reported: the export carries no model and no chipName,
    # so the header reads "GPU: Unknown GPU" — the same state doctor refuses on.
    minimal_nsys_conn.execute(
        "UPDATE TARGET_INFO_GPU SET name=NULL, chipName=NULL, memoryBandwidth=0 WHERE id=0"
    )

    skill = get_skill("arithmetic_intensity")
    rows = skill.execute(
        minimal_nsys_conn,
        theoretical_flops=150e12 * _FIXTURE_KERNEL_S,
        peak_tflops=989.0,
        hbm_bw_gbps=3350,
    )

    assert not is_abstention(rows)
    assert rows[0]["gpu_name"] == "Unknown GPU"
    assert rows[0]["peak_fp16_tflops"] == 989.0
    assert rows[0]["mfu_pct"] == 15.2


def test_arithmetic_intensity_abstains_when_device_has_no_kernels(minimal_nsys_conn):
    """An absent device has the shared diagnostic shape."""
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("arithmetic_intensity")
    rows = skill.execute(
        minimal_nsys_conn,
        theoretical_flops=1e12,
        peak_tflops=989.0,
        device=7,
    )

    assert rows[0]["error"] == "no kernels found"
    assert rows[0]["requested_device"] == 7
    assert rows[0]["available_devices"] == {0: 5}
    assert "Try:" in rows[0]["hint"]


def test_arithmetic_intensity_legitimate_verdicts_are_unchanged(minimal_nsys_conn):
    """A possible MFU classifies exactly as before, on both arms.

    Pinned verbatim: the guard above must refuse impossible readings without
    moving a single legitimate one.
    """
    from nsys_ai.skills.registry import get_skill

    _use_a100(minimal_nsys_conn)
    skill = get_skill("arithmetic_intensity")

    # MFU heuristic arm (no bytes_moved): 200 / 312 = 64.1%.
    r = skill.execute(minimal_nsys_conn, theoretical_flops=200e12 * _FIXTURE_KERNEL_S)[0]
    assert r["achieved_tflops"] == 200.0
    assert r["mfu_pct"] == 64.1
    assert r["classification"] == "High kernel throughput (likely compute-bound)"
    assert r["severity"] == "info"
    assert r["recommendation"] == (
        "Workload has good kernel throughput. "
        "For further gains, consider kernel-level optimization with NCU "
        "(occupancy, warp efficiency, instruction mix)."
    )

    # Roofline arm: AI = 9e11 / 1e10 = 90 FLOP/byte, under A100's 153 ridge.
    r = skill.execute(
        minimal_nsys_conn,
        theoretical_flops=200e12 * _FIXTURE_KERNEL_S,
        bytes_moved=1e10,
    )[0]
    assert r["classification"] == "Memory-bound (AI=90.0 < Ridge=153.0)"
    assert r["severity"] == "warning"
    assert r["recommendation"] == (
        "Workload is mathematically memory-bound (Arithmetic Intensity < Ridge Point). "
        "Increase batch size, use operator fusion, or verify memory access patterns."
    )


def test_arithmetic_intensity_abstention_does_not_publish_the_refused_numbers(
    minimal_nsys_conn,
):
    """The refused MFU must not come back under the healthy row's own key name.

    ``skill run --format json`` dumps the rows verbatim, so a consumer reading
    ``rows[0]["mfu_pct"]`` would otherwise receive the very figure the abstention
    exists to refuse. The diagnostic values are kept, prefixed, because they are
    what tells the caller which input was wrong.
    """
    from nsys_ai.skills.registry import get_skill

    _use_a100(minimal_nsys_conn)
    skill = get_skill("arithmetic_intensity")
    rows = skill.execute(
        minimal_nsys_conn,
        theoretical_flops=1e18,  # absurd total -> achieved far above peak
    )
    assert rows and rows[0].get("_abstained") is True, rows

    healthy_column_names = {"mfu_pct", "achieved_tflops", "classification", "recommendation"}
    leaked = healthy_column_names & set(rows[0])
    assert not leaked, f"abstention row republishes success columns: {sorted(leaked)}"
    assert rows[0]["implied_mfu_pct"] > 100.0


def test_arithmetic_intensity_abstains_on_a_non_positive_flop_count(minimal_nsys_conn):
    """Zero or negative FLOPs cannot describe work that ran.

    Left unguarded, 0 reported "Severely low kernel throughput" at critical
    severity and a negative input reported a negative achieved throughput --
    the same confident verdict from a mistyped number as an MFU above peak,
    one sign away.
    """
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("arithmetic_intensity")
    _use_a100(minimal_nsys_conn)
    for flops in (0, -1e14):
        rows = skill.execute(
            minimal_nsys_conn,
            theoretical_flops=flops,
        )
        assert rows and rows[0].get("_abstained") is True, (flops, rows)
        assert "theoretical_flops" in rows[0]["reason"], rows[0]["reason"]
        assert "classification" not in rows[0], rows[0]
