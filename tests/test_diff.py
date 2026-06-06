import json
import sqlite3
import subprocess
import sys


def _make_db_with_target_info(path: str, gpu_name: str = "NVIDIA A100-SXM4-80GB"):
    """Create a minimal SQLite DB with only TARGET_INFO_GPU + TARGET_INFO_CUDA_DEVICE (for get_first_gpu_name)."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE TARGET_INFO_GPU(id INTEGER PRIMARY KEY, name TEXT, busLocation TEXT, "
        "totalMemory INTEGER, smCount INTEGER, chipName TEXT, memoryBandwidth INTEGER)"
    )
    conn.execute(
        "CREATE TABLE TARGET_INFO_CUDA_DEVICE(gpuId INTEGER, cudaId INTEGER, pid INTEGER, uuid TEXT, numMultiprocessors INTEGER)"
    )
    conn.execute("INSERT INTO TARGET_INFO_GPU(id, name) VALUES (0, ?)", (gpu_name,))
    conn.execute("INSERT INTO TARGET_INFO_CUDA_DEVICE(gpuId, cudaId) VALUES (0, 0)")
    conn.commit()
    conn.close()


def _make_profile(path: str, *, kernels: list[tuple], nvtx: list[tuple] | None = None):
    """
    Create a minimal Nsight-like SQLite export sufficient for Profile().

    kernels entries: (start_ns, end_ns, deviceId, streamId, correlationId, shortNameId, demangledId)
    nvtx entries: (text, globalTid, start_ns, end_ns)
    """
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE StringIds(id INT PRIMARY KEY, value TEXT)")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
        "start INT, [end] INT, deviceId INT, streamId INT, correlationId INT, "
        "shortName INT, demangledName INT)"
    )
    conn.execute("CREATE TABLE NVTX_EVENTS(text TEXT, globalTid INT, start INT, [end] INT)")

    # StringIds
    strings = {
        1: "kA",
        2: "kA_dem",
        3: "kB",
        4: "kB_dem",
        5: "kC",
        6: "kC_dem",
    }
    conn.executemany("INSERT INTO StringIds(id, value) VALUES(?,?)", list(strings.items()))

    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start, [end], deviceId, streamId, correlationId, shortName, demangledName) "
        "VALUES(?,?,?,?,?,?,?)",
        kernels,
    )

    if nvtx:
        conn.executemany(
            "INSERT INTO NVTX_EVENTS(text, globalTid, start, [end]) VALUES(?,?,?,?)",
            nvtx,
        )

    conn.commit()
    conn.close()


def _make_profile_with_launch_config(
    path: str,
    *,
    kernels: list[tuple],
    shared_cols: tuple[str, str] = ("staticSharedMemory", "dynamicSharedMemory"),
):
    """
    Minimal profile with launch-config columns.

    kernels entries:
      (start_ns, end_ns, deviceId, streamId, correlationId, shortNameId, demangledId,
       gridX, gridY, gridZ, blockX, blockY, blockZ,
       registersPerThread, <static shared>, <dynamic shared>)

    shared_cols overrides the shared-memory column names so tests can exercise
    the staticSharedMemoryBytes / dynamicSharedMemoryBytes schema variants.
    """
    static_col, dynamic_col = shared_cols
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE StringIds(id INT PRIMARY KEY, value TEXT)")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
        "start INT, [end] INT, deviceId INT, streamId INT, correlationId INT, "
        "shortName INT, demangledName INT, "
        "gridX INT, gridY INT, gridZ INT, "
        "blockX INT, blockY INT, blockZ INT, "
        f"registersPerThread INT, {static_col} INT, {dynamic_col} INT)"
    )
    conn.execute("CREATE TABLE NVTX_EVENTS(text TEXT, globalTid INT, start INT, [end] INT)")
    # Empty RUNTIME table so iteration detection can run (and cleanly find no
    # iterations) instead of raising on a missing table.
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(globalTid INT, correlationId INT, start INT, [end] INT)"
    )
    conn.executemany(
        "INSERT INTO StringIds(id, value) VALUES(?,?)",
        [
            (1, "kA"),
            (2, "kA_dem"),
            (3, "kB"),
            (4, "kB_dem"),
        ],
    )
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL("
        "start, [end], deviceId, streamId, correlationId, shortName, demangledName, "
        "gridX, gridY, gridZ, blockX, blockY, blockZ, "
        f"registersPerThread, {static_col}, {dynamic_col}) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        kernels,
    )
    conn.commit()
    conn.close()


def _make_profile_with_memory_usage(
    path: str,
    *,
    events: list[tuple],
    kernels: list[tuple] | None = None,
    nvtx: list[tuple] | None = None,
    runtime: list[tuple] | None = None,
    include_memory_table: bool = True,
    include_mem_kind: bool = True,
):
    """
    Minimal profile with CUDA_GPU_MEMORY_USAGE_EVENTS.

    events entries (4 or 6 elements; memKind/contextId default to device kind 2 /
    context 1 when omitted):
      (start_ns, deviceId, bytes, memoryOperationType[, memKind, contextId])

    memoryOperationType follows Nsight Systems: 0 = alloc, 1 = free (None -> NULL).
    memKind follows ENUM_CUDA_MEM_KIND: 0/1 host, 2/3 device, 4/6 managed, 5 static.
    Pass include_mem_kind=False to emit the older schema without memKind/contextId.
    """
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
    conn.executemany(
        "INSERT INTO StringIds(id, value) VALUES(?,?)",
        [
            (1, "kA"),
            (2, "kA_dem"),
            (3, "kB"),
            (4, "kB_dem"),
        ],
    )

    if kernels is None:
        devices = sorted({int(e[1]) for e in events}) or [0]
        kernels = [
            (0, 100_000_000, dev, 7, idx + 1, 1, 2)
            for idx, dev in enumerate(devices)
        ]
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL("
        "start, [end], deviceId, streamId, correlationId, shortName, demangledName) "
        "VALUES(?,?,?,?,?,?,?)",
        kernels,
    )
    if runtime:
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME(globalTid, correlationId, start, [end]) "
            "VALUES(?,?,?,?)",
            runtime,
        )
    if nvtx:
        conn.executemany(
            "INSERT INTO NVTX_EVENTS(text, globalTid, start, [end]) VALUES(?,?,?,?)",
            nvtx,
        )
    if include_memory_table and include_mem_kind:
        conn.execute(
            "CREATE TABLE CUDA_GPU_MEMORY_USAGE_EVENTS("
            "start INT, deviceId INT, bytes INT, memoryOperationType INT, "
            "memKind INT, contextId INT)"
        )
        rows = [e if len(e) >= 6 else (e[0], e[1], e[2], e[3], 2, 1) for e in events]
        conn.executemany(
            "INSERT INTO CUDA_GPU_MEMORY_USAGE_EVENTS("
            "start, deviceId, bytes, memoryOperationType, memKind, contextId) "
            "VALUES(?,?,?,?,?,?)",
            rows,
        )
    elif include_memory_table:
        conn.execute(
            "CREATE TABLE CUDA_GPU_MEMORY_USAGE_EVENTS("
            "start INT, deviceId INT, bytes INT, memoryOperationType INT)"
        )
        conn.executemany(
            "INSERT INTO CUDA_GPU_MEMORY_USAGE_EVENTS(start, deviceId, bytes, memoryOperationType) "
            "VALUES(?,?,?,?)",
            [tuple(e[:4]) for e in events],
        )
    conn.commit()
    conn.close()


def _make_profile_with_runtime(
    path: str,
    *,
    marker: str = "step",
    tid: int = 1,
):
    """Minimal profile with RUNTIME + NVTX so detect_iterations finds one iteration."""
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
    conn.execute("INSERT INTO StringIds(id, value) VALUES (1,'k'), (2,'k_dem')")
    # One kernel 1000–2000 ns, correlationId 1
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start, [end], deviceId, streamId, correlationId, shortName, demangledName) "
        "VALUES (1000, 2000, 0, 7, 1, 1, 2)"
    )
    # NVTX range that contains the kernel launch; RUNTIME 500–1000 so kernel 1000–2000 is inside
    conn.execute(
        "INSERT INTO NVTX_EVENTS(text, globalTid, start, [end]) VALUES (?, ?, 500, 2500)",
        (marker, tid),
    )
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME(globalTid, correlationId, start, [end]) VALUES (?, 1, 900, 1000)",
        (tid,),
    )
    conn.commit()
    conn.close()


def test_diff_engine_math(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"

    # before:
    # - kA: 2 calls, 10ns each => 20ns
    # - kB: 1 call, 30ns => 30ns
    _make_profile(
        str(before),
        kernels=[
            (0, 10, 0, 7, 1, 1, 2),
            (20, 30, 0, 7, 2, 1, 2),
            (40, 70, 0, 7, 3, 3, 4),
        ],
        nvtx=[
            ("step", 1, 0, 100),
            ("warmup", 1, 0, 10),
        ],
    )

    # after:
    # - kA: 2 calls, 20ns each => 40ns (regression +20ns)
    # - kC: 1 call, 5ns => 5ns (new)
    _make_profile(
        str(after),
        kernels=[
            (0, 20, 0, 7, 1, 1, 2),
            (30, 50, 0, 7, 2, 1, 2),
            (60, 65, 0, 7, 3, 5, 6),
        ],
        nvtx=[
            ("step", 1, 0, 120),
        ],
    )

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        d = diff_profiles(b, a, gpu=0, trim=None, limit=10, sort="delta")

    # total GPU time = sum of aggregated kernel durations
    assert d.before.total_gpu_ns == 50
    assert d.after.total_gpu_ns == 45

    # kA regression should be present
    kA = [k for k in d.kernel_diffs if k.name == "kA"][0]
    assert kA.before_total_ns == 20
    assert kA.after_total_ns == 40
    assert kA.delta_ns == 20
    assert kA.classification == "regression"

    # kB removed
    kB = [k for k in d.kernel_diffs if k.name == "kB"][0]
    assert kB.before_total_ns == 30
    assert kB.after_total_ns == 0
    assert kB.classification == "removed"

    # kC new
    kC = [k for k in d.kernel_diffs if k.name == "kC"][0]
    assert kC.before_total_ns == 0
    assert kC.after_total_ns == 5
    assert kC.classification == "new"


def test_diff_top_regression_has_trace_selection(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 30_000_000, 0, 7, 1, 1, 2)])

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        diff = diff_profiles(b, a, gpu=0, limit=10)

    selection = diff.top_regressions[0].selection
    assert selection is not None
    assert selection.source == "diff"
    assert selection.profile_id == diff.after.profile_id
    assert selection.gpu_ids == [0]
    assert "kA" in selection.label
    assert "+20.00ms" in selection.label


def test_diff_top_regression_selection_serializes_to_json(tmp_path):
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 30_000_000, 0, 7, 1, 1, 2)])

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nsys_ai",
            "diff",
            str(before),
            str(after),
            "--gpu",
            "0",
            "--format",
            "json",
            "--no-ai",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    selection = payload["top_regressions"][0]["selection"]
    assert selection["id"].startswith("sel_diff_")
    assert selection["source"] == "diff"
    assert selection["profile_id"] == payload["after"]["profile_id"]
    assert selection["gpu_ids"] == [0]
    assert "kA" in selection["label"]


def test_diff_selection_round_trips_through_trace_selection_dict(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.annotation import TraceSelection
    from nsys_ai.diff import diff_profiles

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 30_000_000, 0, 7, 1, 1, 2)])

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        diff = diff_profiles(b, a, gpu=0, limit=10)

    selection = diff.top_regressions[0].selection
    assert selection is not None
    assert TraceSelection.from_dict(selection.to_dict()) == selection


def test_diff_selection_id_includes_diff_context(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles

    before_a = tmp_path / "before_a.sqlite"
    before_b = tmp_path / "before_b.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before_a), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(before_b), kernels=[(0, 5_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 30_000_000, 0, 7, 1, 1, 2)])

    with profile_mod.open(str(before_a)) as b1, profile_mod.open(str(after)) as a:
        first = diff_profiles(b1, a, gpu=0, limit=10)
    with profile_mod.open(str(before_b)) as b2, profile_mod.open(str(after)) as a:
        second = diff_profiles(b2, a, gpu=0, limit=10)

    first_selection = first.top_regressions[0].selection
    second_selection = second.top_regressions[0].selection
    assert first_selection is not None
    assert second_selection is not None
    assert first_selection.id != second_selection.id


def test_diff_node_wide_selection_omits_gpu_ids(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 30_000_000, 0, 7, 1, 1, 2)])

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        diff = diff_profiles(b, a, gpu=None, limit=10)

    selection = diff.top_regressions[0].selection
    assert selection is not None
    assert selection.gpu_ids is None
    assert "gpu_ids" not in selection.to_dict()


def test_diff_without_top_regressions_has_empty_selection_lists(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles
    from nsys_ai.diff_render import to_diff_json

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        diff = diff_profiles(b, a, gpu=0, limit=10)

    assert diff.top_regressions == []
    assert diff.top_improvements == []
    payload = json.loads(to_diff_json(diff))
    assert payload["top_regressions"] == []
    assert payload["top_improvements"] == []


def test_diff_cli_json_output(tmp_path):
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(
        str(before),
        kernels=[
            (0, 10, 0, 1, 1, 1, 2),
        ],
    )
    _make_profile(
        str(after),
        kernels=[
            (0, 20, 0, 1, 1, 1, 2),
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nsys_ai",
            "diff",
            str(before),
            str(after),
            "--gpu",
            "0",
            "--format",
            "json",
            "--no-ai",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["before"]["total_gpu_ns"] == 10
    assert payload["after"]["total_gpu_ns"] == 20
    assert payload["top_regressions"][0]["delta_ns"] == 10


def test_diff_with_trim_before_trim_after(tmp_path):
    """Phase C: diff_profiles supports trim_before/trim_after for iteration diff."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(
        str(before),
        kernels=[
            (100, 110, 0, 7, 1, 1, 2),
            (200, 230, 0, 7, 2, 3, 4),
        ],
        nvtx=[("step", 1, 0, 300)],
    )
    _make_profile(
        str(after),
        kernels=[
            (100, 130, 0, 7, 1, 1, 2),
            (250, 260, 0, 7, 2, 3, 4),
        ],
        nvtx=[("step", 1, 0, 300)],
    )
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        # Same window in both: 0–300 ns
        d = diff_profiles(
            b,
            a,
            gpu=0,
            trim_before=(0, 300),
            trim_after=(0, 300),
            limit=10,
        )
    assert d.before.total_gpu_ns == 40  # 10 + 30
    assert d.after.total_gpu_ns == 40  # 30 + 10
    kA = [k for k in d.kernel_diffs if k.name == "kA"][0]
    assert kA.delta_ns == 20  # 30 - 10
    kB = [k for k in d.kernel_diffs if k.name == "kB"][0]
    assert kB.delta_ns == -20  # 10 - 30


def test_diff_tools_search_nvtx_regions(tmp_path):
    """Phase C: search_nvtx_regions returns merged before/after NVTX names."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, search_nvtx_regions

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(
        str(before),
        kernels=[(0, 10, 0, 7, 1, 1, 2)],
        nvtx=[("Attention", 1, 0, 50), ("forward", 1, 0, 100)],
    )
    _make_profile(
        str(after),
        kernels=[(0, 10, 0, 7, 1, 1, 2)],
        nvtx=[("Attention", 1, 0, 60), ("backward", 1, 0, 80)],
    )
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        out = search_nvtx_regions(ctx, "Att", limit=10)
    assert "regions" in out
    assert out["query"] == "Att"
    names = [r["text"] for r in out["regions"]]
    assert "Attention" in names
    for r in out["regions"]:
        assert "in_before" in r and "in_after" in r
        assert "total_ns_before" in r and "total_ns_after" in r


def test_diff_tools_get_iteration_boundaries_shape(tmp_path):
    """Phase C: get_iteration_boundaries returns is_aligned and boundaries list."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_iteration_boundaries

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    # detect_iterations needs RUNTIME + NVTX with marker; use _make_profile_with_runtime
    _make_profile_with_runtime(str(before), marker="step", tid=1)
    _make_profile_with_runtime(str(after), marker="step", tid=1)
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="step")
        out = get_iteration_boundaries(ctx, marker="step", target_gpu=0)
    assert "is_aligned" in out
    assert "boundaries" in out
    assert "iteration_count_before" in out and "iteration_count_after" in out
    for bnd in out["boundaries"]:
        assert "before" in bnd and "after" in bnd
        assert "start_ns" in bnd["before"] or bnd["before"]["start_ns"] is None


def test_diff_cli_iteration_and_marker_help():
    """Phase C: diff --help shows --iteration and --marker."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--iteration" in result.stdout
    assert "iteration" in result.stdout.lower()
    assert "--marker" in result.stdout
    assert "sample_0" in result.stdout or "marker" in result.stdout.lower()


def test_diff_cli_chat_help():
    """Stage 6: diff --help shows --chat."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--chat" in result.stdout
    assert "chat" in result.stdout.lower()


def test_diff_cli_exit_on_regression_help():
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--exit-on-regression" in result.stdout
    assert "ci gate" in result.stdout.lower()


def test_diff_cli_exit_on_regression_fails_gate(tmp_path):
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 12_000_000, 0, 7, 1, 1, 2)])

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nsys_ai",
            "diff",
            str(before),
            str(after),
            "--gpu",
            "0",
            "--format",
            "json",
            "--exit-on-regression",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["verdict"] == "regression_likely"
    assert "Diff gate failed" in result.stderr
    assert "step_time_delta_ms=+2.000" in result.stderr
    assert "step_time_delta_pct=+20.00%" in result.stderr
    assert "comparability_confidence=1.000" in result.stderr


def test_diff_cli_exit_on_regression_allows_improvement(tmp_path):
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 12_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nsys_ai",
            "diff",
            str(before),
            str(after),
            "--gpu",
            "0",
            "--format",
            "json",
            "--exit-on-regression",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["verdict"] == "improvement_likely"


def test_diff_cli_exit_on_regression_allows_inconclusive(tmp_path):
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(
        str(after),
        kernels=[
            (0, 12_000_000, 0, 7, 1, 1, 2),
            (12_000_000, 24_000_000, 0, 7, 2, 1, 2),
            (24_000_000, 36_000_000, 0, 7, 3, 1, 2),
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nsys_ai",
            "diff",
            str(before),
            str(after),
            "--gpu",
            "0",
            "--format",
            "json",
            "--exit-on-regression",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["verdict"] == "inconclusive"
    assert payload["comparability_confidence"] < 0.5
    assert "Diff gate failed" not in result.stderr


def test_diff_cli_iteration_out_of_range_exits_nonzero(tmp_path):
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile_with_runtime(str(before), marker="step")
    _make_profile_with_runtime(str(after), marker="step")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nsys_ai",
            "diff",
            str(before),
            str(after),
            "--gpu",
            "0",
            "--iteration",
            "1",
            "--marker",
            "step",
            "--exit-on-regression",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "iteration 1 out of range" in result.stderr


def test_diff_cli_iteration_missing_window_exits_nonzero(tmp_path):
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile_with_runtime(str(before), marker="step")
    _make_profile(str(after), kernels=[])

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nsys_ai",
            "diff",
            str(before),
            str(after),
            "--gpu",
            "0",
            "--iteration",
            "0",
            "--marker",
            "step",
            "--exit-on-regression",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "no time window for this iteration" in result.stderr


def test_diff_tools_run_diff_tool_and_openai_tools(tmp_path):
    """Stage 6: run_diff_tool dispatches; TOOLS_DIFF_OPENAI and build_diff_system_prompt exist."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import (
        TOOLS_DIFF_OPENAI,
        DiffContext,
        build_diff_system_prompt,
        run_diff_tool,
    )

    assert len(TOOLS_DIFF_OPENAI) >= 10
    names = [t["function"]["name"] for t in TOOLS_DIFF_OPENAI]
    assert "search_nvtx_regions" in names
    assert "get_iteration_boundaries" in names
    assert "get_iteration_diff" in names
    assert "get_gpu_peak_tflops" in names
    assert "compute_mfu" in names

    before = tmp_path / "b.sqlite"
    after = tmp_path / "a.sqlite"
    _make_profile_with_runtime(str(before), marker="step")
    _make_profile_with_runtime(str(after), marker="step")
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="step")
        out = run_diff_tool(ctx, "get_iteration_boundaries", {})
    assert "boundaries" in out
    assert isinstance(out["boundaries"], list)
    peak_out = run_diff_tool(ctx, "get_gpu_peak_tflops", {})
    assert "gpu_name" in peak_out
    assert "peak_tflops" in peak_out or "error" in peak_out

    mfu_out = run_diff_tool(
        ctx,
        "compute_mfu",
        {"step_time_s": 10.0, "model_flops_per_step": 1e18, "peak_tflops": 989},
    )
    assert "MFU_pct" in mfu_out
    assert isinstance(mfu_out["MFU_pct"], (int, float))

    prompt = build_diff_system_prompt(ctx, "/before.sqlite", "/after.sqlite", snapshot=None)
    assert "Before profile:" in prompt and "After profile:" in prompt
    assert "/before.sqlite" in prompt and "/after.sqlite" in prompt


def test_diff_tools_global_diff_payload_includes_selection(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_global_diff

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 30_000_000, 0, 7, 1, 1, 2)])

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="step")
        payload = get_global_diff(ctx, target_gpu=0)

    selection = payload["top_regressions"][0]["selection"]
    assert selection["id"].startswith("sel_diff_")
    assert selection["source"] == "diff"
    assert selection["profile_id"].startswith("nsys1:")
    assert selection["gpu_ids"] == [0]
    assert "kA" in selection["label"]


def test_diff_tools_top_k_payload_includes_selection(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles
    from nsys_ai.diff_tools import _top_k_payload

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 30_000_000, 0, 7, 1, 1, 2)])

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        summary = diff_profiles(b, a, gpu=0, limit=10)

    regressions, _improvements, _others_ms = _top_k_payload(summary, top_n=5)
    selection = regressions[0]["selection"]
    assert selection["source"] == "diff"
    assert selection["profile_id"] == summary.after.profile_id
    assert selection["gpu_ids"] == [0]


def test_diff_tools_phase_c_prompt_export():
    """Phase C: system prompt and tool descriptions are exported for agent use."""
    from nsys_ai.diff_tools import DIFF_SYSTEM_PROMPT, TOOL_DESCRIPTIONS

    assert "Never guess names" in DIFF_SYSTEM_PROMPT
    assert "search_nvtx_regions" in DIFF_SYSTEM_PROMPT
    assert "get_launch_config_diff" in DIFF_SYSTEM_PROMPT or "Explain" in DIFF_SYSTEM_PROMPT
    assert "search_nvtx_regions" in TOOL_DESCRIPTIONS
    assert "get_iteration_diff" in TOOL_DESCRIPTIONS
    assert "get_global_diff" in TOOL_DESCRIPTIONS
    assert "get_region_diff" in TOOL_DESCRIPTIONS
    assert "get_gpu_imbalance_stats" in TOOL_DESCRIPTIONS
    assert "get_memory_profile_diff" in TOOL_DESCRIPTIONS
    assert "MFU" in DIFF_SYSTEM_PROMPT


def test_hardware_get_peak_tflops():
    """hardware.get_peak_tflops: known GPU returns peak_tflops, unknown/empty returns error."""
    from nsys_ai.hardware import GPU_SPECS, get_peak_tflops

    # Known GPUs (substring match)
    r = get_peak_tflops("NVIDIA A100-SXM4-80GB")
    assert r.get("gpu_name") == "NVIDIA A100-SXM4-80GB"
    assert "peak_tflops" in r and r["peak_tflops"] == 312.0
    assert "error" not in r

    r = get_peak_tflops("NVIDIA H100 80GB HBM3")
    assert "peak_tflops" in r and r["peak_tflops"] == 989.0

    r = get_peak_tflops("NVIDIA H100 SXM")
    assert r["peak_tflops"] == 989.0

    # Unknown GPU
    r = get_peak_tflops("NVIDIA Unknown GPU XYZ")
    assert "gpu_name" in r and "error" in r
    assert "peak_tflops" not in r

    # Empty / whitespace
    r = get_peak_tflops("")
    assert "error" in r
    r = get_peak_tflops("   ")
    assert "error" in r

    # Sanity: all keys in GPU_SPECS resolve
    for key in GPU_SPECS:
        r = get_peak_tflops(f"NVIDIA {key}")
        assert "peak_tflops" in r, f"Key {key!r} should resolve"
        assert r["peak_tflops"] == GPU_SPECS[key][0]


def test_profile_get_first_gpu_name(tmp_path):
    """profile.get_first_gpu_name returns name from TARGET_INFO_GPU when tables exist; empty when missing."""
    from nsys_ai.profile import get_first_gpu_name

    db_with_gpu = tmp_path / "with_gpu.sqlite"
    _make_db_with_target_info(str(db_with_gpu), "NVIDIA H100 80GB HBM3")
    with sqlite3.connect(str(db_with_gpu)) as conn:
        name = get_first_gpu_name(conn)
    assert name == "NVIDIA H100 80GB HBM3"

    # DB without TARGET_INFO tables
    no_gpu = tmp_path / "no_gpu.sqlite"
    conn_no = sqlite3.connect(str(no_gpu))
    conn_no.execute("CREATE TABLE other(id INT)")
    conn_no.commit()
    conn_no.close()
    with sqlite3.connect(str(no_gpu)) as conn:
        name = get_first_gpu_name(conn)
    assert name == ""


def test_mfu_single_and_compare():
    """MFU lives in nsys_ai.mfu; single and compare are pure math."""
    from nsys_ai.mfu import compute_mfu_compare, compute_mfu_single

    out = compute_mfu_single(10.0, 1e18, 989.0)
    assert out["MFU_pct"] == round(100.0 * (1e18 / 10.0 / 1e12) / 989.0, 2)
    assert "achieved_model_TFLOPS" in out

    err = compute_mfu_single(10.0, 0, 989.0)
    assert "error" in err
    assert "formula" in err

    cmp_out = compute_mfu_compare(10.0, 12.0, 1e18, 989.0)
    assert "MFU_pct" in cmp_out
    assert "before" in cmp_out["MFU_pct"] and "after" in cmp_out["MFU_pct"]
    assert "delta_MFU_pct" in cmp_out


def test_diff_tools_stage5_warning_flags(tmp_path):
    """Stage 5: get_iteration_diff sets JIT_Compilation_Warning for iteration 0; payload has Hardware_Warning."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_iteration_diff

    before = tmp_path / "b.sqlite"
    after = tmp_path / "a.sqlite"
    _make_profile_with_runtime(str(before), marker="step")
    _make_profile_with_runtime(str(after), marker="step")
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="step")
        out = get_iteration_diff(ctx, 0, marker="step", target_gpu=0)
    assert "error" not in out or "iteration_index" in out
    assert "JIT_Compilation_Warning" in out
    assert out["JIT_Compilation_Warning"] is True  # iteration_index == 0
    assert "Hardware_Warning" in out
    assert isinstance(out["Hardware_Warning"], bool)


def test_diff_tools_region_diff_and_stubs(tmp_path):
    """Phase C: get_region_diff, get_launch_config_diff, get_memory_profile_diff return expected shape or error."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import (
        DiffContext,
        get_launch_config_diff,
        get_memory_profile_diff,
        get_region_diff,
    )

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10, 0, 7, 1, 1, 2)], nvtx=[("Attention", 1, 0, 50)])
    _make_profile(str(after), kernels=[(0, 20, 0, 7, 1, 1, 2)], nvtx=[("Attention", 1, 0, 60)])
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        out = get_region_diff(ctx, "Attention", target_gpu=0)
    assert "nvtx_exact_match" in out or "error" in out
    assert "wall_clock_ms" in out or "error" in out

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        launch = get_launch_config_diff(ctx, "kA", target_gpu=0)
    assert "error" in launch or "kernel_name" in launch
    assert "uses_tensor_core_likely" in launch or "error" in launch

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        mem = get_memory_profile_diff(ctx, target_gpu=0)
    assert "error" in mem


def test_diff_tools_default_target_gpu_aggregates_all_gpus(tmp_path):
    """Omitting target_gpu aggregates every device; an explicit id scopes to one.

    The dispatcher and the diff_tools function signatures default target_gpu
    to None, which means "all GPUs". A query that does not name a device must
    therefore report the combined compute time of a multi-GPU profile, not
    just GPU 0.
    """
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_region_diff, run_diff_tool

    # GPU 0 runs a 10ms kernel, GPU 1 a 20ms kernel, both inside "step" and both
    # on stream 7. Reusing the same streamId across devices checks that the
    # per-GPU detail fields aggregate by (deviceId, streamId), not streamId alone
    # (streamId is device-scoped, so a naive count would collapse to one).
    kernels = [
        (0, 10_000_000, 0, 7, 1, 1, 2),  # deviceId 0, kA, 10ms, stream 7
        (0, 20_000_000, 1, 7, 2, 3, 4),  # deviceId 1, kB, 20ms, stream 7
    ]
    nvtx = [("step", 1, 0, 20_000_000)]
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=kernels, nvtx=nvtx)
    _make_profile(str(after), kernels=kernels, nvtx=nvtx)

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="step")

        # Default (target_gpu omitted): both devices counted → 10 + 20 = 30ms,
        # and the detail fields count two distinct (deviceId, streamId) pairs
        # even though both devices reuse streamId 7.
        all_gpus = get_region_diff(ctx, "step")
        assert all_gpus["top3_global_categories"]["Compute"]["before"] == 30.0
        assert all_gpus["unique_streams_count_before"] == 2

        # Explicit device still scopes to that GPU only → 10ms and one stream.
        gpu0 = get_region_diff(ctx, "step", target_gpu=0)
        assert gpu0["top3_global_categories"]["Compute"]["before"] == 10.0
        assert gpu0["unique_streams_count_before"] == 1

        # Dispatching without target_gpu in args matches the all-GPU default.
        dispatched = run_diff_tool(ctx, "get_region_diff", {"nvtx_exact_match": "step"})
        assert dispatched["top3_global_categories"]["Compute"]["before"] == 30.0
        assert dispatched["unique_streams_count_before"] == 2


def test_get_launch_config_diff_returns_config_delta_and_explanation(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_launch_config_diff

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile_with_launch_config(
        str(before),
        kernels=[
            # In-window representative config.
            (0, 10_000_000, 0, 7, 1, 1, 2, 256, 1, 1, 128, 1, 1, 64, 0, 0),
            # Outside ctx.trim and longer; must not win.
            (200_000_000, 260_000_000, 0, 7, 2, 1, 2, 999, 1, 1, 64, 1, 1, 32, 0, 0),
        ],
    )
    _make_profile_with_launch_config(
        str(after),
        kernels=[
            (0, 20_000_000, 0, 7, 1, 1, 2, 128, 1, 1, 128, 1, 1, 96, 0, 49_152),
            (200_000_000, 260_000_000, 0, 7, 2, 1, 2, 999, 1, 1, 64, 1, 1, 32, 0, 0),
        ],
    )

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=(0, 100_000_000), marker="sample_0")
        out = get_launch_config_diff(ctx, "kA", target_gpu=0)

    assert "error" not in out
    assert out["before"]["grid"] == [256, 1, 1]
    assert out["after"]["grid"] == [128, 1, 1]
    assert out["delta"]["gridX"] == {"before": 256, "after": 128, "delta": -128}
    assert out["delta"]["grid"]["delta"] == [-128, 0, 0]
    assert out["delta"]["registersPerThread"] == {"before": 64, "after": 96, "delta": 32}
    assert out["delta"]["sharedMemoryBytes"]["after"] == 49_152
    assert out["before"]["sample_count"] == 1
    assert "999" not in out["explanation"]
    assert "registers/thread 64 -> 96" in out["explanation"]
    assert "occupancy" in out["explanation"]


def test_get_launch_config_diff_partial_when_kernel_missing_one_side(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_launch_config_diff

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile_with_launch_config(
        str(before),
        kernels=[(0, 10, 0, 7, 1, 1, 2, 1, 1, 1, 128, 1, 1, 32, 0, 0)],
    )
    _make_profile_with_launch_config(
        str(after),
        kernels=[(0, 10, 0, 7, 1, 3, 4, 1, 1, 1, 128, 1, 1, 32, 0, 0)],
    )

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        out = get_launch_config_diff(ctx, "kA", target_gpu=0)

    assert out["error"] == "not comparable"
    assert out["before"]["matched_name"] == "kA_dem"
    assert out["after"] is None
    assert "only appears before" in out["explanation"]


def test_get_launch_config_diff_reports_distinct_configs_and_dominant_share(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_launch_config_diff

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    # before: kA launched with two distinct configs in-window —
    #   grid 256 x2 (20ms total) -> dominant by GPU time
    #   grid 128 x1 (5ms total)
    _make_profile_with_launch_config(
        str(before),
        kernels=[
            (0, 10_000_000, 0, 7, 1, 1, 2, 256, 1, 1, 128, 1, 1, 64, 0, 0),
            (10_000_000, 20_000_000, 0, 7, 2, 1, 2, 256, 1, 1, 128, 1, 1, 64, 0, 0),
            (20_000_000, 25_000_000, 0, 7, 3, 1, 2, 128, 1, 1, 128, 1, 1, 64, 0, 0),
        ],
    )
    # after: kA launched only one way.
    _make_profile_with_launch_config(
        str(after),
        kernels=[(0, 10_000_000, 0, 7, 1, 1, 2, 256, 1, 1, 128, 1, 1, 64, 0, 0)],
    )

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=(0, 100_000_000), marker="sample_0")
        out = get_launch_config_diff(ctx, "kA", target_gpu=0)

    assert "error" not in out
    # Dominant config is the one with the most GPU time (grid 256), not the
    # smallest or the most recent.
    assert out["before"]["grid"] == [256, 1, 1]
    assert out["before"]["distinct_configs"] == 2
    assert out["before"]["total_invocations"] == 3
    assert out["before"]["sample_count"] == 2
    assert out["before"]["dominant_share"] == round(2 / 3, 4)
    # after launched only one way -> dominant config is fully representative.
    assert out["after"]["distinct_configs"] == 1
    assert out["after"]["dominant_share"] == 1.0


def test_get_launch_config_diff_iteration_index_out_of_range(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_launch_config_diff

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile_with_launch_config(
        str(before),
        kernels=[(0, 10_000_000, 0, 7, 1, 1, 2, 256, 1, 1, 128, 1, 1, 64, 0, 0)],
    )
    _make_profile_with_launch_config(
        str(after),
        kernels=[(0, 10_000_000, 0, 7, 1, 1, 2, 256, 1, 1, 128, 1, 1, 64, 0, 0)],
    )

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        out = get_launch_config_diff(ctx, "kA", iteration_index=99, target_gpu=0)

    assert "out of range" in out["error"]
    assert out["iteration_index"] == 99


def test_get_launch_config_diff_unchanged_config_points_elsewhere(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_launch_config_diff

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    # Identical launch config; only the duration grew. The tool should rule
    # launch config OUT as the cause and point elsewhere.
    _make_profile_with_launch_config(
        str(before),
        kernels=[(0, 10_000_000, 0, 7, 1, 1, 2, 256, 1, 1, 128, 1, 1, 64, 0, 0)],
    )
    _make_profile_with_launch_config(
        str(after),
        kernels=[(0, 20_000_000, 0, 7, 1, 1, 2, 256, 1, 1, 128, 1, 1, 64, 0, 0)],
    )

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        out = get_launch_config_diff(ctx, "kA", target_gpu=0)

    assert "error" not in out
    assert out["delta"]["gridX"]["delta"] == 0
    assert out["delta"]["registersPerThread"]["delta"] == 0
    assert "unchanged" in out["explanation"]


def test_get_launch_config_diff_reads_shared_memory_bytes_column_variant(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_launch_config_diff

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    # Newer Nsight exports name the columns staticSharedMemoryBytes /
    # dynamicSharedMemoryBytes; the tool must detect those variants too.
    variant = ("staticSharedMemoryBytes", "dynamicSharedMemoryBytes")
    _make_profile_with_launch_config(
        str(before),
        kernels=[(0, 10_000_000, 0, 7, 1, 1, 2, 256, 1, 1, 128, 1, 1, 64, 0, 16_384)],
        shared_cols=variant,
    )
    _make_profile_with_launch_config(
        str(after),
        kernels=[(0, 10_000_000, 0, 7, 1, 1, 2, 256, 1, 1, 128, 1, 1, 64, 0, 49_152)],
        shared_cols=variant,
    )

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        out = get_launch_config_diff(ctx, "kA", target_gpu=0)

    assert "error" not in out
    # The *Bytes variant is mapped back to the canonical output key.
    assert out["columns_used"]["before"]["dynamicSharedMemory"] == "dynamicSharedMemoryBytes"
    assert out["delta"]["sharedMemoryBytes"]["before"] == 16_384
    assert out["delta"]["sharedMemoryBytes"]["after"] == 49_152
    assert out["delta"]["sharedMemoryBytes"]["delta"] == 32_768


def test_get_launch_config_diff_not_available_when_columns_absent(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_launch_config_diff

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    # Plain profiles: kernel table has no grid/block launch-config columns.
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        out = get_launch_config_diff(ctx, "kA", target_gpu=0)

    assert out["error"] == "not available"
    assert "gridX" not in out["available_columns"]["before"]


def test_get_launch_config_diff_not_available_when_columns_asymmetric(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_launch_config_diff

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    # before HAS launch-config columns, after does NOT -> common set is empty,
    # so there is nothing comparable and the tool reports "not available".
    _make_profile_with_launch_config(
        str(before),
        kernels=[(0, 10_000_000, 0, 7, 1, 1, 2, 256, 1, 1, 128, 1, 1, 64, 0, 0)],
    )
    _make_profile(str(after), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        out = get_launch_config_diff(ctx, "kA", target_gpu=0)

    assert out["error"] == "not available"
    # The asymmetry is surfaced for debugging.
    assert "gridX" in out["available_columns"]["before"]
    assert "gridX" not in out["available_columns"]["after"]


def test_get_launch_config_diff_negative_iteration_index_out_of_range(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_launch_config_diff

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile_with_launch_config(
        str(before),
        kernels=[(0, 10_000_000, 0, 7, 1, 1, 2, 256, 1, 1, 128, 1, 1, 64, 0, 0)],
    )
    _make_profile_with_launch_config(
        str(after),
        kernels=[(0, 10_000_000, 0, 7, 1, 1, 2, 256, 1, 1, 128, 1, 1, 64, 0, 0)],
    )

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        out = get_launch_config_diff(ctx, "kA", iteration_index=-1, target_gpu=0)

    # -1 must NOT silently select the last iteration via Python indexing.
    assert "out of range" in out["error"]
    assert out["iteration_index"] == -1


def test_get_memory_profile_diff_returns_peak_counts_net_delta_and_explanation(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_memory_profile_diff

    mib = 1024 * 1024
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile_with_memory_usage(
        str(before),
        events=[
            (0, 0, 1024 * mib, 0),  # pre-window baseline
            (10, 0, 512 * mib, 0),
            (20, 0, 128 * mib, 1),
            (200_000_000, 0, 10_000 * mib, 0),  # outside ctx.trim; must not win peak
        ],
    )
    _make_profile_with_memory_usage(
        str(after),
        events=[
            (0, 0, 1024 * mib, 0),
            (10, 0, 1024 * mib, 0),
            (20, 0, 128 * mib, 1),
            (200_000_000, 0, 10_000 * mib, 0),
        ],
    )

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=(5, 100), marker="sample_0")
        out = get_memory_profile_diff(ctx, target_gpu=0)

    assert "error" not in out
    assert out["before"]["baseline_vram_bytes"] == 1024 * mib
    assert out["before"]["peak_vram_bytes"] == 1536 * mib
    assert out["after"]["peak_vram_bytes"] == 2048 * mib
    assert out["delta"]["peak_vram_bytes"] == {
        "before": 1536 * mib,
        "after": 2048 * mib,
        "delta": 512 * mib,
    }
    assert out["before"]["alloc_count"] == 1
    assert out["before"]["free_count"] == 1
    assert out["before"]["allocated_bytes"] == 512 * mib
    assert out["before"]["freed_bytes"] == 128 * mib
    assert out["before"]["net_delta_bytes"] == 384 * mib
    assert out["after"]["net_delta_bytes"] == 896 * mib
    assert out["before"]["event_window_ns"] == [10, 20]
    assert "10000" not in out["explanation"]
    assert "peak VRAM" in out["explanation"]
    assert "higher peak" in out["explanation"]


def test_get_memory_profile_diff_default_target_gpu_aggregates_all_gpus(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_memory_profile_diff

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile_with_memory_usage(
        str(before),
        events=[
            (0, 0, 100, 0),
            (0, 1, 200, 0),
        ],
    )
    _make_profile_with_memory_usage(
        str(after),
        events=[
            (0, 0, 100, 0),
            (0, 1, 300, 0),
        ],
    )

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        all_gpus = get_memory_profile_diff(ctx)
        gpu1 = get_memory_profile_diff(ctx, target_gpu=1)

    assert all_gpus["before"]["peak_vram_bytes"] == 300
    assert all_gpus["after"]["peak_vram_bytes"] == 400
    assert all_gpus["delta"]["peak_vram_bytes"]["delta"] == 100
    assert gpu1["before"]["peak_vram_bytes"] == 200
    assert gpu1["after"]["peak_vram_bytes"] == 300
    assert gpu1["delta"]["peak_vram_bytes"]["delta"] == 100


def test_get_memory_profile_diff_uses_iteration_window(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_memory_profile_diff

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    kernels = [(1_000_000_000, 2_000_000_000, 0, 7, 1, 1, 2)]
    nvtx = [("step", 1, 500_000_000, 2_500_000_000)]
    runtime = [(1, 1, 900_000_000, 1_000_000_000)]
    _make_profile_with_memory_usage(
        str(before),
        events=[
            (100_000_000, 0, 1000, 0),
            (1_000_000_000, 0, 500, 0),
            (3_000_000_000, 0, 9999, 0),
        ],
        kernels=kernels,
        nvtx=nvtx,
        runtime=runtime,
    )
    _make_profile_with_memory_usage(
        str(after),
        events=[
            (100_000_000, 0, 1000, 0),
            (1_000_000_000, 0, 700, 0),
            (3_000_000_000, 0, 9999, 0),
        ],
        kernels=kernels,
        nvtx=nvtx,
        runtime=runtime,
    )

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="step")
        out = get_memory_profile_diff(ctx, iteration_index=0, target_gpu=0)

    assert "error" not in out
    assert out["trim_before_ns"] == [1_000_000_000, 2_000_000_000]
    assert out["trim_after_ns"] == [1_000_000_000, 2_000_000_000]
    assert out["before"]["baseline_vram_bytes"] == 1000
    assert out["before"]["peak_vram_bytes"] == 1500
    assert out["after"]["peak_vram_bytes"] == 1700
    assert out["delta"]["peak_vram_bytes"]["delta"] == 200


def test_get_memory_profile_diff_not_available_when_table_missing(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_memory_profile_diff

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile_with_memory_usage(str(before), events=[(0, 0, 100, 0)])
    _make_profile_with_memory_usage(
        str(after),
        events=[],
        include_memory_table=False,
    )

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        out = get_memory_profile_diff(ctx)

    assert out["error"] == "not available"
    assert out["tables_present"] == {"before": True, "after": False}
    assert "bytes" in out["available_columns"]["before"]
    assert out["available_columns"]["after"] == []


def test_get_memory_profile_diff_excludes_host_mem_kinds_from_vram(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_memory_profile_diff

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    # (start, deviceId, bytes, op, memKind, contextId): a device alloc (kind 2)
    # plus a pinned-host alloc (kind 1). Host must NOT count toward VRAM.
    events = [(10, 0, 1000, 0, 2, 1), (20, 0, 5000, 0, 1, 1)]
    _make_profile_with_memory_usage(str(before), events=events)
    _make_profile_with_memory_usage(str(after), events=events)

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        out = get_memory_profile_diff(ctx, target_gpu=0)

    assert "error" not in out
    assert out["before"]["mem_kind_available"] is True
    assert out["before"]["peak_vram_bytes"] == 1000  # host 5000 excluded
    assert out["before"]["alloc_count"] == 1  # device alloc only
    assert out["before"]["host_event_count"] == 1
    breakdown = {bk["mem_kind"]: bk for bk in out["before"]["mem_kind_breakdown"]}
    assert set(breakdown) == {1, 2}
    assert breakdown[1]["is_host"] is True
    assert breakdown[2]["is_host"] is False


def test_get_memory_profile_diff_same_timestamp_alloc_free_keeps_peak(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_memory_profile_diff

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    # alloc 1000 and free 1000 at the SAME timestamp: alloc must apply first so the
    # high-water mark is 1000, not 0.
    events = [(10, 0, 1000, 0, 2, 1), (10, 0, 1000, 1, 2, 1)]
    _make_profile_with_memory_usage(str(before), events=events)
    _make_profile_with_memory_usage(str(after), events=events)

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        out = get_memory_profile_diff(ctx, target_gpu=0)

    assert out["before"]["peak_vram_bytes"] == 1000
    assert out["before"]["net_delta_bytes"] == 0


def test_get_memory_profile_diff_counts_null_op_as_unknown(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_memory_profile_diff

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    # op=None -> NULL memoryOperationType; must be surfaced, not silently dropped.
    _make_profile_with_memory_usage(
        str(before), events=[(10, 0, 1000, 0, 2, 1), (20, 0, 500, None, 2, 1)]
    )
    _make_profile_with_memory_usage(str(after), events=[(10, 0, 1000, 0, 2, 1)])

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        out = get_memory_profile_diff(ctx, target_gpu=0)

    assert out["before"]["unknown_event_count"] == 1


def test_get_memory_profile_diff_reports_distinct_contexts(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_memory_profile_diff

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    # Two CUDA contexts on the same device; total device VRAM is the sum.
    events = [(10, 0, 1000, 0, 2, 1), (20, 0, 2000, 0, 2, 2)]
    _make_profile_with_memory_usage(str(before), events=events)
    _make_profile_with_memory_usage(str(after), events=events)

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        out = get_memory_profile_diff(ctx, target_gpu=0)

    assert out["before"]["distinct_contexts"] == 2
    assert out["before"]["peak_vram_bytes"] == 3000


def test_get_memory_profile_diff_best_effort_when_mem_kind_absent(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_memory_profile_diff

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    # Older schema without memKind/contextId columns.
    _make_profile_with_memory_usage(
        str(before), events=[(10, 0, 1000, 0)], include_mem_kind=False
    )
    _make_profile_with_memory_usage(
        str(after), events=[(10, 0, 1000, 0)], include_mem_kind=False
    )

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        out = get_memory_profile_diff(ctx, target_gpu=0)

    assert "error" not in out
    assert out["before"]["mem_kind_available"] is False
    assert out["before"]["peak_vram_bytes"] == 1000  # best effort: counts everything
    assert "memKind column is absent" in out["explanation"]


# ---------------------------------------------------------------------------
# AI narrative and executive summary (diff report augmentation)
# ---------------------------------------------------------------------------


def test_diff_build_executive_summary_with_tmp_path(tmp_path):
    """build_executive_summary with tmp_path fixture (stable content)."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.ai.diff_narrative import build_executive_summary
    from nsys_ai.diff import diff_profiles

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 20, 0, 7, 1, 1, 2)])
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        summary = diff_profiles(b, a, gpu=0, limit=10)
    text = build_executive_summary(summary)
    assert "slower" in text or "faster" in text
    assert "+10" in text or "10" in text


def test_diff_generate_narrative_no_model_returns_warning(tmp_path, monkeypatch):
    """generate_diff_narrative with no LLM configured returns warning, no exception."""
    import nsys_ai.chat_config as chat_config_mod
    from nsys_ai import profile as profile_mod
    from nsys_ai.ai.diff_narrative import DiffNarrative, generate_diff_narrative
    from nsys_ai.diff import diff_profiles

    monkeypatch.setattr(
        chat_config_mod, "_get_model_and_key", lambda _=None: (None, None), raising=False
    )

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 20, 0, 7, 1, 1, 2)])
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        summary = diff_profiles(b, a, gpu=0, limit=10)

    narrative = generate_diff_narrative(summary)
    assert isinstance(narrative, DiffNarrative)
    assert narrative.executive_summary
    assert narrative.ai_narrative is None
    assert narrative.warning is not None
    assert (
        "No LLM" in narrative.warning
        or "no-ai" in narrative.warning.lower()
        or "API" in narrative.warning
    )


def test_diff_format_terminal_with_narrative(tmp_path):
    """format_diff_terminal with narrative includes Executive Summary and optional AI block."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.ai.diff_narrative import DiffNarrative
    from nsys_ai.diff import diff_profiles
    from nsys_ai.diff_render import format_diff_terminal

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 20, 0, 7, 1, 1, 2)])
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        summary = diff_profiles(b, a, gpu=0, limit=10)
    narrative = DiffNarrative(
        executive_summary="Total GPU time increased by +10.00us.",
        ai_narrative="The main regression is in kernel kA.",
        model="test",
        warning=None,
    )
    out = format_diff_terminal(summary, narrative=narrative)
    assert "Executive Summary" in out
    assert "Total GPU time increased" in out
    assert "AI Narrative" in out
    assert "main regression" in out
    out_no_ai = format_diff_terminal(summary, narrative=None)
    assert "Executive Summary" not in out_no_ai
    assert "AI Narrative" not in out_no_ai


def test_diff_cli_terminal_no_ai_shows_executive_summary(tmp_path):
    """diff --format terminal --no-ai shows Executive Summary and no AI section."""
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 20, 0, 7, 1, 1, 2)])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nsys_ai",
            "diff",
            str(before),
            str(after),
            "--gpu",
            "0",
            "--format",
            "terminal",
            "--no-ai",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Executive Summary" in result.stdout
    assert "Profile Diff" in result.stdout
    assert "Top regressions" in result.stdout
    # With --no-ai we do not call the LLM; Note section may appear only if we tried AI and failed
    # So we only require that the numeric report is present
    assert "10" in result.stdout and "20" in result.stdout


def test_diff_cli_json_structure_unchanged(tmp_path):
    """diff --format json output does not include narrative fields (contract unchanged)."""
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 20, 0, 7, 1, 1, 2)])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nsys_ai",
            "diff",
            str(before),
            str(after),
            "--gpu",
            "0",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "before" in payload and "after" in payload and "top_regressions" in payload
    assert "executive_summary" not in payload
    assert "ai_narrative" not in payload


# ---------------------------------------------------------------------------
# v0.1 diff schema: envelope, verdict, category attribution, confidence
# ---------------------------------------------------------------------------


def _make_overlap_dict(compute_only_ms, nccl_only_ms, overlap_ms, idle_ms):
    """Helper to build a fake overlap dict matching overlap_analysis output."""
    total = compute_only_ms + nccl_only_ms + overlap_ms + idle_ms
    return {
        "compute_only_ms": compute_only_ms,
        "nccl_only_ms": nccl_only_ms,
        "overlap_ms": overlap_ms,
        "idle_ms": idle_ms,
        "total_ms": total,
        "overlap_pct": 0.0,
        "compute_kernels": 1,
        "nccl_kernels": 0,
    }


def test_v01_category_attribution_hta_convention():
    """compute_category_attribution: overlap_ms counts as compute (HTA convention)."""
    from nsys_ai.diff import ProfileSummary, compute_category_attribution

    # before: compute_only=100, nccl_only=20, overlap=10, idle=5 → compute=110, comm=20, idle=5
    before = ProfileSummary(
        path="b",
        gpu=0,
        schema_version=None,
        total_gpu_ns=0,
        kernel_rows=0,
        kernels=[],
        nvtx=[],
        overlap=_make_overlap_dict(100, 20, 10, 5),
    )
    # after: compute_only=120, nccl_only=25, overlap=15, idle=10 → compute=135, comm=25, idle=10
    after = ProfileSummary(
        path="a",
        gpu=0,
        schema_version=None,
        total_gpu_ns=0,
        kernel_rows=0,
        kernels=[],
        nvtx=[],
        overlap=_make_overlap_dict(120, 25, 15, 10),
    )
    cats = {c.category: c for c in compute_category_attribution(before, after)}
    assert cats["compute"].before_ms == 110.0  # 100 + 10 (overlap)
    assert cats["compute"].after_ms == 135.0  # 120 + 15
    assert cats["compute"].delta_ms == 25.0
    assert cats["communication"].before_ms == 20.0  # exposed_comm = nccl_only
    assert cats["communication"].after_ms == 25.0
    assert cats["idle"].before_ms == 5.0
    assert cats["idle"].after_ms == 10.0
    # launch_overhead is intentionally absent in v0.1 (PR-B will add it).
    assert "launch_overhead" not in cats


def test_v01_compute_verdict_thresholds():
    """compute_verdict applies ±5% threshold + confidence ≥ 0.5 gate."""
    from nsys_ai.diff import compute_verdict

    assert compute_verdict(None, 1.0) == "inconclusive"
    assert compute_verdict(10.0, 0.3) == "inconclusive"  # low confidence
    assert compute_verdict(4.9, 1.0) == "neutral"  # below +5%
    assert compute_verdict(-4.9, 1.0) == "neutral"  # above -5%
    assert compute_verdict(5.0, 1.0) == "regression_likely"
    assert compute_verdict(20.0, 0.7) == "regression_likely"
    assert compute_verdict(-5.0, 1.0) == "improvement_likely"
    assert compute_verdict(-20.0, 0.7) == "improvement_likely"


def test_v01_collect_sanity_warnings_returns_confidence():
    """collect_sanity_warnings now returns (warnings, confidence)."""
    from nsys_ai.diff import ProfileSummary, collect_sanity_warnings

    matched = ProfileSummary(
        path="x",
        gpu=0,
        schema_version="2024.1.1",
        total_gpu_ns=100,
        kernel_rows=100,
        kernels=[],
        nvtx=[],
        overlap={},
    )
    warnings, conf = collect_sanity_warnings(matched, matched)
    assert isinstance(warnings, list)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0
    assert conf == 1.0  # identical → perfect confidence
    assert warnings == []

    # Schema mismatch → C_schema = 0 → confidence = 0
    other = ProfileSummary(
        path="y",
        gpu=0,
        schema_version="2025.2.2",
        total_gpu_ns=100,
        kernel_rows=100,
        kernels=[],
        nvtx=[],
        overlap={},
    )
    warnings, conf = collect_sanity_warnings(matched, other)
    assert conf == 0.0
    assert any("schema" in w.lower() for w in warnings)


def test_v01_diff_json_envelope_and_verdict(tmp_path):
    """diff JSON v0.1: envelope + verdict + category_attribution + profile_id."""
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 20, 0, 7, 1, 1, 2)])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nsys_ai",
            "diff",
            str(before),
            str(after),
            "--gpu",
            "0",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    # Envelope
    assert payload["schema_version"] == "0.1"
    assert payload["producer"] == "nsys-ai"
    assert "producer_version" in payload
    assert payload["diff_id"].startswith("diff1:sha256:")
    # diff_id has a 64-char hex digest after the prefix
    assert len(payload["diff_id"]) == len("diff1:sha256:") + 64

    # profile_id in each side, content-derived
    assert payload["before"]["profile_id"].startswith("nsys1:")
    assert payload["after"]["profile_id"].startswith("nsys1:")

    # Verdict + confidence
    assert payload["verdict"] in {
        "neutral",
        "regression_likely",
        "improvement_likely",
        "inconclusive",
    }
    assert 0.0 <= payload["comparability_confidence"] <= 1.0

    # step_time block
    assert "step_time" in payload
    assert "delta_ms" in payload["step_time"]

    # category_attribution is a list of category bucket entries
    cats = payload["category_attribution"]
    assert isinstance(cats, list)
    seen = {c["category"] for c in cats}
    assert seen == {"compute", "communication", "idle"}


def test_v01_confidence_separates_schema_and_gpu_mismatch():
    """c_schema and c_gpu are independent factors; mismatching gpu alone zeros confidence."""
    from nsys_ai.diff import ProfileSummary, collect_sanity_warnings

    # Same schema, different gpu id → c_gpu = 0 → confidence = 0,
    # but the warning text mentions GPU (not schema).
    a = ProfileSummary(
        path="a",
        gpu=0,
        schema_version="2024.1.1",
        total_gpu_ns=100,
        kernel_rows=100,
        kernels=[],
        nvtx=[],
        overlap={},
    )
    b = ProfileSummary(
        path="b",
        gpu=1,  # different GPU id, same schema
        schema_version="2024.1.1",
        total_gpu_ns=100,
        kernel_rows=100,
        kernels=[],
        nvtx=[],
        overlap={},
    )
    warnings, conf = collect_sanity_warnings(a, b)
    assert conf == 0.0
    assert any("GPU" in w for w in warnings)
    assert not any("schema" in w.lower() for w in warnings)


def test_v01_no_signal_propagates_through_pipeline():
    """Overlap error → confidence drops, attribution empty, step_time fields None,
    JSON step_time is null (key present, value null). No fake-zero leakage."""
    from nsys_ai.diff import ProfileDiffSummary, ProfileSummary, collect_sanity_warnings
    from nsys_ai.diff_render import to_diff_json

    good = ProfileSummary(
        path="b",
        gpu=0,
        schema_version="2024.1.1",
        total_gpu_ns=100,
        kernel_rows=100,
        kernels=[],
        nvtx=[],
        overlap=_make_overlap_dict(100, 20, 10, 5),
    )
    bad = ProfileSummary(
        path="a",
        gpu=0,
        schema_version="2024.1.1",
        total_gpu_ns=0,
        kernel_rows=0,
        kernels=[],
        nvtx=[],
        overlap={"error": "no kernels found"},
    )

    # confidence must reflect the unavailability (c_overlap = 0 -> product 0)
    warnings, conf = collect_sanity_warnings(good, bad)
    assert conf == 0.0
    assert any("Overlap analysis unavailable" in w for w in warnings)

    # Build a summary that mirrors what diff_profiles would emit on this path
    # (empty attribution, both step_time fields None) and verify the JSON
    # never leaks fake zeros.
    summary = ProfileDiffSummary(
        before=good,
        after=bad,
        warnings=warnings,
        kernel_diffs=[],
        nvtx_diffs=[],
        overlap_before=good.overlap,
        overlap_after=bad.overlap,
        overlap_delta={},
        top_regressions=[],
        top_improvements=[],
        verdict="inconclusive",
        comparability_confidence=conf,
    )
    payload = json.loads(to_diff_json(summary))
    assert payload["step_time"] is None
    assert payload["category_attribution"] == []
    assert payload["verdict"] == "inconclusive"
    assert payload["comparability_confidence"] == 0.0


def test_v01_category_attribution_empty_on_overlap_error():
    """When either side has overlap error, attribution is [] (no fake zeros)."""
    from nsys_ai.diff import ProfileSummary, compute_category_attribution

    good = ProfileSummary(
        path="b",
        gpu=0,
        schema_version=None,
        total_gpu_ns=0,
        kernel_rows=0,
        kernels=[],
        nvtx=[],
        overlap=_make_overlap_dict(100, 20, 10, 5),
    )
    bad = ProfileSummary(
        path="a",
        gpu=0,
        schema_version=None,
        total_gpu_ns=0,
        kernel_rows=0,
        kernels=[],
        nvtx=[],
        overlap={"error": "no kernels found"},
    )
    assert compute_category_attribution(good, bad) == []
    assert compute_category_attribution(bad, good) == []
    assert compute_category_attribution(bad, bad) == []


def test_v01_confidence_serialization_truncates_not_rounds():
    """JSON-serialized confidence must never cross the 0.5 verdict gate
    via rounding (e.g. 0.4996 must NOT show as 0.500)."""
    from nsys_ai.diff import ProfileDiffSummary, ProfileSummary
    from nsys_ai.diff_render import to_diff_json

    bare = ProfileSummary(
        path="", gpu=0, schema_version=None, total_gpu_ns=0,
        kernel_rows=0, kernels=[], nvtx=[], overlap={},
    )
    summary = ProfileDiffSummary(
        before=bare, after=bare, warnings=[], kernel_diffs=[], nvtx_diffs=[],
        overlap_before={}, overlap_after={}, overlap_delta={},
        top_regressions=[], top_improvements=[],
        comparability_confidence=0.4996,
    )
    payload = json.loads(to_diff_json(summary))
    assert payload["comparability_confidence"] == 0.499


def test_v01_diff_id_is_stable_across_runs(tmp_path):
    """Same inputs → same diff_id (content-derived; not random)."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 20, 0, 7, 1, 1, 2)])

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        d1 = diff_profiles(b, a, gpu=0, limit=10)
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        d2 = diff_profiles(b, a, gpu=0, limit=10)

    assert d1.diff_id == d2.diff_id
    assert d1.diff_id.startswith("diff1:sha256:")
