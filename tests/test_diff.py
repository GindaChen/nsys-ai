import json
import os
import pathlib
import site
import sqlite3
import subprocess
import sys
from dataclasses import replace

import pytest


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


def _make_named_profile(path: str, *, kernels: list[tuple], strings: dict[int, str]):
    """
    Create a minimal profile with caller-supplied StringIds.

    kernels entries: (start_ns, end_ns, deviceId, streamId, correlationId, shortNameId, demangledId)
    """
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE StringIds(id INT PRIMARY KEY, value TEXT)")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
        "start INT, [end] INT, deviceId INT, streamId INT, correlationId INT, "
        "shortName INT, demangledName INT)"
    )
    conn.execute("CREATE TABLE NVTX_EVENTS(text TEXT, globalTid INT, start INT, [end] INT)")
    conn.executemany("INSERT INTO StringIds(id, value) VALUES(?,?)", list(strings.items()))
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start, [end], deviceId, streamId, correlationId, shortName, demangledName) "
        "VALUES(?,?,?,?,?,?,?)",
        kernels,
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


def test_diff_selection_anchors_slowest_after_instance(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    # kA twice in after: 10ms and 30ms. Bounds must be the slowest instance,
    # not the MIN..MAX envelope (0..50ms).
    _make_profile(
        str(after),
        kernels=[
            (0, 10_000_000, 0, 7, 1, 1, 2),
            (20_000_000, 50_000_000, 0, 7, 2, 1, 2),
        ],
    )

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        diff = diff_profiles(b, a, gpu=0, limit=10)

    selection = diff.top_regressions[0].selection
    assert selection is not None
    assert selection.start_ns == 20_000_000
    assert selection.end_ns == 50_000_000
    sel_dict = selection.to_dict()
    assert sel_dict["start_ns"] == 20_000_000
    assert sel_dict["end_ns"] == 50_000_000


def test_diff_selection_bounds_respect_trim_window(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    # The slowest kA instance (200..260ms) sits outside the trim window and
    # must not be chosen as the anchor.
    _make_profile(
        str(after),
        kernels=[
            (0, 30_000_000, 0, 7, 1, 1, 2),
            (200_000_000, 260_000_000, 0, 7, 2, 1, 2),
        ],
    )

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        diff = diff_profiles(b, a, gpu=0, trim=(0, 100_000_000), limit=10)

    selection = diff.top_regressions[0].selection
    assert selection is not None
    assert selection.start_ns == 0
    assert selection.end_ns == 30_000_000


def test_diff_selection_removed_kernel_has_no_time_bounds(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    # kB exists only in before -> "removed" improvement; it has no instances in
    # the after profile, so its selection stays a name+GPU anchor.
    _make_profile(
        str(before),
        kernels=[
            (0, 10_000_000, 0, 7, 1, 1, 2),
            (10_000_000, 15_000_000, 0, 7, 2, 3, 4),
        ],
    )
    _make_profile(str(after), kernels=[(0, 30_000_000, 0, 7, 1, 1, 2)])

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        diff = diff_profiles(b, a, gpu=0, limit=10)

    removed = [k for k in diff.top_improvements if k.classification == "removed"]
    assert removed, "expected kB to be a removed improvement"
    selection = removed[0].selection
    assert selection is not None
    assert selection.start_ns is None
    assert "start_ns" not in selection.to_dict()
    # The regressed kernel still gets bounds from its after instance.
    reg_sel = diff.top_regressions[0].selection
    assert reg_sel.start_ns == 0
    assert reg_sel.end_ns == 30_000_000


def test_diff_selection_time_bounds_serialize_to_json(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles
    from nsys_ai.diff_render import to_diff_json

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 30_000_000, 0, 7, 1, 1, 2)])

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        diff = diff_profiles(b, a, gpu=0, limit=10)

    payload = json.loads(to_diff_json(diff))
    selection = payload["top_regressions"][0]["selection"]
    assert selection["start_ns"] == 0
    assert selection["end_ns"] == 30_000_000


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


def test_diff_json_lineage_covers_regressions_and_improvements(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles
    from nsys_ai.diff_render import to_diff_dict

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(
        str(before),
        kernels=[
            (0, 10_000_000, 0, 7, 1, 1, 2),
            (20_000_000, 50_000_000, 0, 7, 2, 3, 4),
        ],
    )
    _make_profile(
        str(after),
        kernels=[
            (0, 30_000_000, 0, 7, 1, 1, 2),
            (40_000_000, 50_000_000, 0, 7, 2, 3, 4),
        ],
    )

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        payload = to_diff_dict(diff_profiles(b, a, gpu=0, limit=10))

    for field_name, role in (
        ("top_regressions", "regression"),
        ("top_improvements", "improvement"),
    ):
        assert payload[field_name]
        for rank, entry in enumerate(payload[field_name]):
            assert entry["diff_lineage"] == {
                "diff_id": payload["diff_id"],
                "role": role,
                "rank": rank,
                "baseline_profile_id": payload["before"]["profile_id"],
            }


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


def test_diff_id_keys_effective_asymmetric_trim_windows(tmp_path):
    """The id must distinguish the actual before/after windows used."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(100, 110, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(200, 220, 0, 7, 1, 1, 2)])

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        first = diff_profiles(
            b,
            a,
            gpu=0,
            trim_before=(0, 150),
            trim_after=(150, 250),
        )
        second = diff_profiles(
            b,
            a,
            gpu=0,
            trim_before=(0, 150),
            trim_after=(250, 350),
        )

    assert first.diff_id != second.diff_id


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
    # Two decimals, truncated, matching the report's own rendering: the gate
    # message used to round to three, so a score of 0.4996 printed as 0.500
    # beside "(minimum 0.50)" on the same line that refused it.
    assert "comparability_confidence=1.00 " in result.stderr


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


def test_diff_cli_exit_on_regression_blocks_inconclusive(tmp_path):
    """A gate blocks regressions; a comparison that could not be made has not shown
    their absence, so it must not exit 0 either."""
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
    assert result.returncode == 1, result.stderr
    payload = json.loads(result.stdout)
    assert payload["verdict"] == "inconclusive"
    assert payload["comparability_confidence"] < 0.5
    # The reason is stated, not implied by a number.
    assert "Diff gate could not be evaluated" in result.stderr
    assert "verdict=inconclusive" in result.stderr


@pytest.mark.parametrize("empty_side", ["after", "before", "both"])
def test_diff_cli_empty_capture_is_not_an_improvement(tmp_path, empty_side):
    """A capture that recorded nothing must not read as the best result ever seen.

    This is the failure mode that silently green-lights a broken pipeline: the
    workload crashed, the trace was truncated, nsys was misconfigured, or the
    wrong artifact was uploaded. Every delta then points the "improvement" way.
    """
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    kernels = [(0, 10_000_000, 0, 7, 1, 1, 2), (20_000_000, 30_000_000, 0, 7, 2, 3, 4)]
    _make_profile(str(before), kernels=[] if empty_side in ("before", "both") else kernels)
    _make_profile(str(after), kernels=[] if empty_side in ("after", "both") else kernels)

    result = _run_diff_cli(before, after, "--no-ai", "--gate", "5")
    # 2. The gate does not pass a comparison that could not be made.
    assert result.returncode == 1, f"stdout={result.stdout}\nstderr={result.stderr}"
    # 1. No improvement claim in the executive summary.
    assert "Total GPU time went" not in result.stdout
    assert "Largest improvement" not in result.stdout
    # 3. The reason is stated, in the report and on stderr, not implied by a number.
    assert "no GPU kernel activity" in result.stdout
    assert "Diff gate could not be evaluated" in result.stderr
    assert "no GPU kernel activity" in result.stderr

    payload = json.loads(_run_diff_cli(before, after, "--no-ai", "--format", "json").stdout)
    assert payload["verdict"] == "inconclusive"
    assert payload["comparability_confidence"] == 0.0
    assert any("no GPU kernel activity" in w for w in payload["warnings"])


def test_empty_side_is_refused_before_the_model_is_asked(tmp_path, monkeypatch):
    """The LLM path must not narrate a comparison the deterministic path refused.

    ``generate_diff_narrative`` builds a prompt listing "Top improvements" from the
    same deltas and instructs the model to mention them, so without this an LLM
    would narrate the vanished kernels as a win directly beneath the refusal.
    """
    from nsys_ai.ai import diff_narrative as narrative_module

    def _fail(*args, **kwargs):
        raise AssertionError("the model was consulted about an empty capture")

    monkeypatch.setattr(narrative_module, "_get_model_and_key", _fail, raising=False)
    monkeypatch.setattr(
        "nsys_ai.chat_config._get_model_and_key", _fail, raising=False
    )

    from nsys_ai.diff import diff_profiles
    from nsys_ai.profile import Profile

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[])
    with Profile(str(before)) as bp, Profile(str(after)) as ap:
        summary = diff_profiles(bp, ap, gpu=0)

    result = narrative_module.generate_diff_narrative(summary)

    assert result.ai_narrative is None
    assert "no GPU kernel activity" in result.executive_summary


def test_diff_cli_all_gpu_empty_capture_is_not_an_improvement(tmp_path):
    """The issue's own invocation: no --gpu, so every device is compared.

    The parametrized test above passes --gpu 0, and on that path an empty side
    already tripped the pre-existing "Overlap analysis unavailable" warning, so it
    was inconclusive before this fix too. The reported defect was the all-GPU
    default: it returned improvement_likely at confidence 1.0 and exit 0, and
    nothing covered it end to end.
    """
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    kernels = [(0, 10_000_000, 0, 7, 1, 1, 2), (20_000_000, 30_000_000, 0, 7, 2, 3, 4)]
    _make_profile(str(before), kernels=kernels)
    _make_profile(str(after), kernels=[])

    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff", str(before), str(after),
         "--no-ai", "--gate", "5"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "Total GPU time went" not in result.stdout
    assert "Largest improvement" not in result.stdout
    assert "no GPU kernel activity" in result.stdout

    payload = json.loads(
        subprocess.run(
            [sys.executable, "-m", "nsys_ai", "diff", str(before), str(after),
             "--no-ai", "--format", "json"],
            capture_output=True, text=True,
        ).stdout
    )
    assert payload["verdict"] == "inconclusive"
    assert payload["comparability_confidence"] == 0.0


def _run_diff_cli(before, after, *extra, cwd=None, env=None):
    return subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff", str(before), str(after), "--gpu", "0", *extra],
        capture_output=True,
        cwd=cwd,
        env=env,
        text=True,
    )


def _decision_cli_env(tmp_path):
    env = os.environ.copy()
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )
    empty_gitconfig = tmp_path / "empty.gitconfig"
    empty_gitconfig.write_text("", encoding="utf-8")
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = str(empty_gitconfig)
    env["USER"] = "fallback-user"
    return env


def test_diff_cli_gate_help_and_validation(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--gate" in result.stdout

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    # Non-finite values would make the gate silently never fire (fail-open):
    # NaN compares false against everything, inf exceeds any delta. The =form
    # keeps argparse from reading "-inf" as an option name.
    for invalid in ("-3", "0", "nan", "inf", "-inf"):
        bad = _run_diff_cli(before, after, f"--gate={invalid}")
        assert bad.returncode == 2, f"--gate {invalid} should be rejected"
        assert "positive percentage" in bad.stderr


def test_diff_cli_derived_session_does_not_pollute_session_root(tmp_path):
    """Bare ``--session`` derives its index directory after opening the profile."""
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 9_000_000, 0, 7, 1, 1, 2)])

    result = _run_diff_cli(
        before,
        after,
        "--format",
        "json",
        "--no-ai",
        "--session",
        cwd=tmp_path,
        env=_decision_cli_env(tmp_path),
    )

    # No session exists yet, so the command must fail before profile analysis or
    # memo publication. In particular, it must not leave an orphan handoff that
    # blocks a later evidence build using the same derived session id.
    assert result.returncode != 0
    assert not (tmp_path / ".nsys-ai").exists()


def test_diff_cli_gate_tightens_threshold_and_implies_exit(tmp_path):
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    # +4% step time: passes the default 5% verdict but fails a 3% gate.
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 10_400_000, 0, 7, 1, 1, 2)])

    default_gate = _run_diff_cli(before, after, "--format", "json", "--exit-on-regression")
    assert default_gate.returncode == 0, default_gate.stderr
    assert json.loads(default_gate.stdout)["verdict"] == "neutral"

    # --gate alone implies the CI gate; the verdict reflects the custom threshold.
    tight = _run_diff_cli(before, after, "--format", "json", "--gate", "3.0")
    assert tight.returncode == 1
    payload = json.loads(tight.stdout)
    assert payload["verdict"] == "regression_likely"
    assert "Diff gate failed" in tight.stderr
    assert "step_time_delta_pct=+4.00%" in tight.stderr
    assert "gate_pct=3.00%" in tight.stderr


def test_diff_cli_gate_loosens_threshold(tmp_path):
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    # +20% fails the default gate but passes a 30% gate, and the verdict agrees.
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 12_000_000, 0, 7, 1, 1, 2)])

    loose = _run_diff_cli(before, after, "--format", "json", "--gate", "30")
    assert loose.returncode == 0, loose.stderr
    assert json.loads(loose.stdout)["verdict"] == "neutral"
    assert "Diff gate failed" not in loose.stderr


def test_diff_cli_decision_help():
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--accept" in result.stdout
    assert "--reject" in result.stdout
    assert "--reason" in result.stdout


def test_diff_cli_accept_writes_stable_decision_json(tmp_path):
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 9_000_000, 0, 7, 1, 1, 2)])

    result = _run_diff_cli(
        before,
        after,
        "--format",
        "json",
        "--accept",
        "--reason",
        "candidate is faster",
        cwd=tmp_path,
        env=_decision_cli_env(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    diff_path = tmp_path / "diff.json"
    assert diff_path.exists()
    record = json.loads(diff_path.read_text(encoding="utf-8"))
    stdout_payload = json.loads(result.stdout)

    assert record["diff_id"] == stdout_payload["diff_id"]
    assert record["verdict"] == stdout_payload["verdict"]
    assert record["comparability_confidence"] == stdout_payload["comparability_confidence"]
    assert record["category_attribution"] == stdout_payload["category_attribution"]
    assert record["before"]["profile_id"].startswith("nsys2:")
    assert record["after"]["profile_id"].startswith("nsys2:")
    assert record["decision"]["status"] == "accepted"
    assert record["decision"]["reason"] == "candidate is faster"
    assert record["decision"]["decider"] == "fallback-user"
    assert record["decision"]["decided_at"].endswith("Z")
    assert diff_path.read_text(encoding="utf-8") == json.dumps(
        record, indent=2, sort_keys=True
    ) + "\n"
    assert "Diff decision written to diff.json" in result.stderr


def test_diff_cli_reject_writes_decision_status(tmp_path):
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 12_000_000, 0, 7, 1, 1, 2)])

    result = _run_diff_cli(
        before,
        after,
        "--reject",
        "--reason",
        "regression is too large",
        cwd=tmp_path,
        env=_decision_cli_env(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / "diff.json").read_text(encoding="utf-8"))
    assert record["decision"]["status"] == "rejected"
    assert record["decision"]["reason"] == "regression is too large"


def test_diff_cli_decision_json_carries_lineage_for_top_regressions(tmp_path):
    from nsys_ai.annotation import DiffLineage

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(
        str(before),
        kernels=[
            (0, 10_000_000, 0, 7, 1, 1, 2),
            (10_000_000, 20_000_000, 0, 7, 2, 3, 4),
        ],
    )
    _make_profile(
        str(after),
        kernels=[
            (0, 20_000_000, 0, 7, 1, 1, 2),
            (20_000_000, 35_000_000, 0, 7, 2, 3, 4),
        ],
    )

    result = _run_diff_cli(
        before,
        after,
        "--format",
        "json",
        "--limit",
        "2",
        "--accept",
        "--reason",
        "regressions are expected",
        cwd=tmp_path,
        env=_decision_cli_env(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / "diff.json").read_text(encoding="utf-8"))
    regressions = record["top_regressions"]
    assert len(regressions) == 2
    for rank, row in enumerate(regressions):
        lineage = row["diff_lineage"]
        restored = DiffLineage.from_dict(lineage)
        assert restored.to_dict() == lineage
        assert lineage["diff_id"] == record["diff_id"]
        assert lineage["role"] == "regression"
        assert lineage["rank"] == rank
        assert lineage["baseline_profile_id"] == record["before"]["profile_id"]


def test_diff_cli_decision_missing_reason_is_refused(tmp_path):
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 9_000_000, 0, 7, 1, 1, 2)])

    result = _run_diff_cli(
        before,
        after,
        "--accept",
        cwd=tmp_path,
        env=_decision_cli_env(tmp_path),
    )

    assert result.returncode == 2
    assert "--reason is required" in result.stderr
    assert not (tmp_path / "diff.json").exists()


def test_diff_cli_decision_can_be_written_outside_the_working_directory(tmp_path):
    """A tool advertised for CI must be able to run in a checkout without dirtying it.

    The decision record was hardcoded to ``diff.json`` beside wherever the
    command ran, with no way to redirect it. In CI that directory is the
    repository under test.
    """
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 9_000_000, 0, 7, 1, 1, 2)])
    destination = tmp_path / "artifacts" / "decision.json"

    result = _run_diff_cli(
        before,
        after,
        "--accept",
        "--reason",
        "verified",
        "--decision-out",
        str(destination),
        cwd=tmp_path,
        env=_decision_cli_env(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(destination.read_text(encoding="utf-8"))["decision"]["status"] == "accepted"
    assert not (tmp_path / "diff.json").exists(), "the record was still left in the CWD"


def test_diff_cli_decision_defaults_to_the_working_directory(tmp_path):
    """Without the flag nothing moves; this is a redirect, not a relocation."""
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 9_000_000, 0, 7, 1, 1, 2)])

    result = _run_diff_cli(
        before, after, "--accept", "--reason", "verified",
        cwd=tmp_path, env=_decision_cli_env(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "diff.json").exists()


def test_diff_cli_unwritable_decision_path_is_a_configuration_error(tmp_path):
    """Exit 2, not 1: a CI job reads 1 as "the performance gate failed".

    ``--decision-out`` is caller-supplied, so an unwritable directory is a
    misconfiguration. Letting the OSError escape gave a traceback and exit 1 --
    indistinguishable from a real regression, on the exact path CI uses.
    """
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 9_000_000, 0, 7, 1, 1, 2)])
    unwritable = tmp_path / "readonly"
    unwritable.mkdir()
    unwritable.chmod(0o500)

    try:
        result = _run_diff_cli(
            before, after, "--accept", "--reason", "verified",
            "--decision-out", str(unwritable / "decision.json"),
            cwd=tmp_path, env=_decision_cli_env(tmp_path),
        )
    finally:
        unwritable.chmod(0o700)

    assert result.returncode == 2, result.stderr
    assert "cannot write the decision record" in result.stderr
    assert "Traceback" not in result.stderr


def test_diff_cli_decision_out_expands_a_home_relative_path(tmp_path):
    """CI invokes this without a shell, so nothing else expands the tilde.

    Left literal, ``atomic_write_bytes`` created a directory named ``~`` in the
    working directory -- the pollution the option exists to prevent.
    """
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 9_000_000, 0, 7, 1, 1, 2)])
    home = tmp_path / "home"
    home.mkdir()
    env = _decision_cli_env(tmp_path)
    env["HOME"] = str(home)
    # Relocating HOME also relocates the per-user site-packages the runtime
    # dependencies may live in, so keep the original one importable.
    env["PYTHONPATH"] = os.pathsep.join([env["PYTHONPATH"], site.getusersitepackages()])

    result = _run_diff_cli(
        before, after, "--accept", "--reason", "verified",
        "--decision-out", "~/artifacts/decision.json",
        cwd=tmp_path, env=env,
    )

    assert result.returncode == 0, result.stderr
    assert (home / "artifacts" / "decision.json").exists()
    assert not (tmp_path / "~").exists(), "a directory named ~ was created in the CWD"


def test_diff_cli_rejects_decision_out_with_session(tmp_path):
    """The session owns its record; accepting the flag and writing nothing lies.

    A CI job that passed both exited 0 and then failed reading a file that was
    never created, with nothing in stderr explaining why.
    """
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 9_000_000, 0, 7, 1, 1, 2)])

    result = _run_diff_cli(
        before, after, "--session", "--accept", "--reason", "verified",
        "--decision-out", str(tmp_path / "elsewhere.json"),
        cwd=tmp_path, env=_decision_cli_env(tmp_path),
    )

    assert result.returncode == 2, result.stderr
    assert "--decision-out cannot be combined with --session" in result.stderr
    assert not (tmp_path / "elsewhere.json").exists()


def test_diff_cli_refuses_to_write_report_and_decision_to_one_path(tmp_path):
    """Two artifacts, one name: say which is which and how to separate them.

    ``-o`` writes the rendered report in ``--format``; the decision record is a
    separate JSON artifact. The old message named the collision without naming
    a way out, because there was none.
    """
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 9_000_000, 0, 7, 1, 1, 2)])

    result = _run_diff_cli(
        before, after, "--format", "json", "--accept", "--reason", "v",
        "-o", "same.json", "--decision-out", "same.json",
        cwd=tmp_path, env=_decision_cli_env(tmp_path),
    )

    assert result.returncode == 2
    assert "--decision-out" in result.stderr, result.stderr
    assert "Traceback" not in result.stderr


def test_diff_cli_decision_low_comparability_stamps_inconclusive(tmp_path):
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

    result = _run_diff_cli(
        before,
        after,
        "--format",
        "json",
        "--accept",
        "--reason",
        "ship despite mismatch",
        cwd=tmp_path,
        env=_decision_cli_env(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / "diff.json").read_text(encoding="utf-8"))
    assert record["decision"]["status"] == "accepted"
    assert record["verdict"] == "inconclusive"
    assert record["comparability_confidence"] < 0.5
    assert any("stamping verdict as inconclusive" in w for w in record["warnings"])
    assert "Warning: comparability_confidence" in result.stderr


def test_diff_decision_requires_profile_ids(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles
    from nsys_ai.diff_decision import build_diff_decision_record

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 9_000_000, 0, 7, 1, 1, 2)])

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        summary = diff_profiles(b, a, gpu=0)

    missing_before_id = replace(summary, before=replace(summary.before, profile_id=""))
    try:
        build_diff_decision_record(
            missing_before_id,
            decision="accepted",
            reason="candidate is faster",
            decider="tester@example.com",
            decided_at="2026-06-18T00:00:00Z",
        )
    except ValueError as exc:
        assert "profile_id" in str(exc)
    else:
        raise AssertionError("expected missing profile_id to be refused")


def test_diff_decision_dict_path_matches_summary_path(tmp_path):
    """The web loop (dict) writer must be byte-identical to the CLI (summary) writer."""
    import json as _json

    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles
    from nsys_ai.diff_decision import (
        write_diff_decision_json,
        write_diff_decision_json_from_diff_dict,
    )
    from nsys_ai.diff_render import to_diff_json

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 9_000_000, 0, 7, 1, 1, 2)])

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        summary = diff_profiles(b, a, gpu=0)

    # The web loop stores the diff payload as JSON (to_diff_json) and reloads it.
    stored_dict = _json.loads(to_diff_json(summary))

    kwargs = dict(
        decision="accepted",
        reason="candidate is faster",
        decider="tester@example.com",
        decided_at="2026-06-18T00:00:00Z",
    )
    cli_path, _, _ = write_diff_decision_json(
        summary, path=tmp_path / "cli.json", **kwargs
    )
    gui_path, _, _ = write_diff_decision_json_from_diff_dict(
        stored_dict, path=tmp_path / "gui.json", **kwargs
    )

    assert cli_path.read_bytes() == gui_path.read_bytes()
    # The stored dict must not be mutated by the writer. The key is now always
    # present and `null` until decided, so "undecided" is `is None` rather than
    # absent — the writer leaving it None is the same guarantee as before.
    assert stored_dict["decision"] is None


def test_compute_verdict_custom_regression_pct():
    from nsys_ai.diff import compute_verdict

    assert compute_verdict(4.0, 1.0) == "neutral"
    assert compute_verdict(4.0, 1.0, regression_pct=3.0) == "regression_likely"
    assert compute_verdict(-4.0, 1.0, regression_pct=3.0) == "improvement_likely"
    assert compute_verdict(20.0, 1.0, regression_pct=30.0) == "neutral"
    # Confidence gating still wins over any threshold.
    assert compute_verdict(50.0, 0.4, regression_pct=3.0) == "inconclusive"


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
    assert "cannot answer" in prompt


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
    assert selection["profile_id"].startswith("nsys2:")
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


def _make_overlap_dict(
    compute_only_ms, nccl_only_ms, overlap_ms, idle_ms, launch_overhead_ms=0.0
):
    """Helper to build a fake overlap dict matching overlap_analysis output."""
    total = compute_only_ms + nccl_only_ms + overlap_ms + idle_ms
    return {
        "compute_only_ms": compute_only_ms,
        "nccl_only_ms": nccl_only_ms,
        "overlap_ms": overlap_ms,
        "idle_ms": idle_ms,
        "launch_overhead_ms": launch_overhead_ms,
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
    # No launch_overhead in these fixtures, so idle is unchanged and the
    # launch bucket is zero.
    assert cats["idle"].before_ms == 5.0
    assert cats["idle"].after_ms == 10.0
    assert cats["launch_overhead"].before_ms == 0.0
    assert cats["launch_overhead"].after_ms == 0.0


def test_v01_launch_overhead_carved_from_idle():
    """launch_overhead is carved out of idle; the four buckets sum to total."""
    from nsys_ai.diff import ProfileSummary, compute_category_attribution

    def _summary(co, nccl, ov, idle, launch):
        return ProfileSummary(
            path="p",
            gpu=0,
            schema_version=None,
            total_gpu_ns=0,
            kernel_rows=0,
            kernels=[],
            nvtx=[],
            overlap=_make_overlap_dict(co, nccl, ov, idle, launch_overhead_ms=launch),
        )

    # before idle=20 of which 8 is launch overhead; after idle=30 of which 5 is.
    before = _summary(100, 20, 10, 20, 8)
    after = _summary(120, 25, 15, 30, 5)
    cats = {c.category: c for c in compute_category_attribution(before, after)}

    assert cats["launch_overhead"].before_ms == 8.0
    assert cats["launch_overhead"].after_ms == 5.0
    # idle is reduced by the carved launch overhead (residual idle).
    assert cats["idle"].before_ms == 12.0  # 20 - 8
    assert cats["idle"].after_ms == 25.0  # 30 - 5

    # Sum invariant: the four buckets reproduce the original step time, so the
    # verdict is unaffected by adding the bucket.
    for side in ("before_ms", "after_ms"):
        four = sum(getattr(cats[c], side) for c in cats)
        three = (
            getattr(cats["compute"], side)
            + getattr(cats["communication"], side)
            + getattr(cats["launch_overhead"], side)
            + getattr(cats["idle"], side)
        )
        assert four == three  # tautology guard that all four are present
    assert sum(getattr(cats[c], "before_ms") for c in cats) == 100 + 20 + 10 + 20


def test_v01_launch_overhead_capped_at_idle():
    """A launch_overhead_ms exceeding idle is capped so residual idle stays >= 0."""
    from nsys_ai.diff import ProfileSummary, compute_category_attribution

    def _summary(launch):
        return ProfileSummary(
            path="p",
            gpu=0,
            schema_version=None,
            total_gpu_ns=0,
            kernel_rows=0,
            kernels=[],
            nvtx=[],
            overlap=_make_overlap_dict(100, 0, 0, 5, launch_overhead_ms=launch),
        )

    cats = {
        c.category: c
        for c in compute_category_attribution(_summary(9.0), _summary(9.0))
    }
    # launch capped at idle (5), residual idle clamped to 0.
    assert cats["launch_overhead"].before_ms == 5.0
    assert cats["idle"].before_ms == 0.0


def test_launch_overhead_ms_counts_only_exposed_idle():
    """launch_overhead_ms = GPU-idle time overlapping a kernel-launch API call."""
    import sqlite3

    from nsys_ai.connection import wrap_connection
    from nsys_ai.overlap import launch_overhead_ms

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            deviceId INTEGER, correlationId INTEGER, start INTEGER, end INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            correlationId INTEGER, start INTEGER, end INTEGER
        );
        """
    )
    # All on device 0; timestamps in ns (1e6 ns = 1 ms).
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)",
        [
            (0, 1, 1_000_000, 2_000_000),  # first kernel: no preceding idle -> 0
            (0, 2, 5_000_000, 6_000_000),  # idle gap (2e6,5e6)
            (0, 3, 10_000_000, 11_000_000),  # idle gap (6e6,10e6)
            (0, 4, 12_000_000, 13_000_000),  # idle gap (11e6,12e6)
        ],
    )
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?)",
        [
            (1, 900_000, 1_000_000),  # before first kernel — not attributed
            (2, 4_000_000, 4_500_000),  # fully inside gap -> 0.5 ms
            (3, 8_500_000, 10_200_000),  # (8.5e6,10.2e6) ∩ (6e6,10e6) -> 1.5 ms
            (4, 10_500_000, 10_900_000),  # ends before gap start (11e6) -> hidden, 0
        ],
    )
    conn.commit()

    class _Schema:
        kernel_table = "CUPTI_ACTIVITY_KIND_KERNEL"
        tables = ["CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME"]

    class _Prof:
        schema = _Schema()
        # A real Profile always wraps its connection; launch_overhead_ms resolves
        # the runtime table through the adapter, so a bare sqlite3.Connection
        # (which happens to satisfy .execute) is not a faithful stub.
        adapter = wrap_connection(conn)

    assert launch_overhead_ms(_Prof(), device=0) == 2.0  # 0.5 + 1.5
    assert launch_overhead_ms(_Prof(), device=99) == 0.0  # no kernels on device


def test_launch_overhead_ms_without_runtime_table_is_zero():
    """No runtime table → launch overhead is 0.0 (best-effort enrichment)."""
    import sqlite3

    from nsys_ai.connection import wrap_connection
    from nsys_ai.overlap import launch_overhead_ms

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
        "(deviceId INTEGER, correlationId INTEGER, start INTEGER, end INTEGER);"
    )
    conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 1, 0, 1000)")
    conn.commit()

    class _Schema:
        kernel_table = "CUPTI_ACTIVITY_KIND_KERNEL"
        tables = ["CUPTI_ACTIVITY_KIND_KERNEL"]  # no RUNTIME table

    class _Prof:
        schema = _Schema()
        adapter = wrap_connection(conn)

    assert launch_overhead_ms(_Prof(), device=0) == 0.0


def test_launch_overhead_through_real_profile_stack(tmp_path):
    """End-to-end: launch overhead flows through Profile (incl. DuckDB cache) and
    into the carved attribution bucket — guards against the best-effort
    try/except silently returning 0 on a backend the unit test doesn't exercise.
    """
    import sqlite3

    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import build_profile_summary, compute_category_attribution
    from nsys_ai.overlap import launch_overhead_ms

    db = tmp_path / "ctrl.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE TARGET_INFO_GPU (id INTEGER PRIMARY KEY, name TEXT,
          busLocation TEXT DEFAULT '', totalMemory INTEGER DEFAULT 0,
          smCount INTEGER DEFAULT 0, chipName TEXT DEFAULT '', memoryBandwidth INTEGER DEFAULT 0);
        CREATE TABLE TARGET_INFO_CUDA_DEVICE (gpuId INTEGER, cudaId INTEGER,
          pid INTEGER DEFAULT 0, uuid TEXT DEFAULT '', numMultiprocessors INTEGER DEFAULT 0);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
          globalPid INTEGER DEFAULT 0, deviceId INTEGER DEFAULT 0, streamId INTEGER DEFAULT 0,
          correlationId INTEGER DEFAULT 0, start INTEGER, end INTEGER, shortName INTEGER,
          demangledName INTEGER DEFAULT 0, gridX INTEGER DEFAULT 1, gridY INTEGER DEFAULT 1,
          gridZ INTEGER DEFAULT 1, blockX INTEGER DEFAULT 1, blockY INTEGER DEFAULT 1, blockZ INTEGER DEFAULT 1);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (globalTid INTEGER DEFAULT 0,
          correlationId INTEGER, start INTEGER, end INTEGER, nameId INTEGER DEFAULT 0);
        INSERT INTO StringIds VALUES (1, 'kernel_A');
        INSERT INTO TARGET_INFO_GPU VALUES (0, 'NVIDIA Test GPU', '', 0, 108, 'Chip', 0);
        INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (0, 0, 100, '', 108);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
          (deviceId, streamId, correlationId, start, end, shortName) VALUES
          (0, 7, 1, 1000000, 2000000, 1),
          (0, 7, 2, 5000000, 6000000, 1),
          (0, 7, 3, 10000000, 11000000, 1),
          (0, 7, 4, 12000000, 13000000, 1);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (correlationId, start, end) VALUES
          (1, 900000, 1000000), (2, 4000000, 4500000),
          (3, 8500000, 10200000), (4, 10500000, 10900000);
        """
    )
    conn.commit()
    conn.close()

    prof = profile_mod.open(str(db))
    try:
        # Same scenario as the unit test: 0.5 + 1.5 ms exposed.
        assert launch_overhead_ms(prof, 0) == 2.0
        summary = build_profile_summary(prof, 0, trim=None)
        assert summary.overlap["launch_overhead_ms"] == 2.0
        # Carved out of idle in attribution, and present as its own bucket.
        cats = {c.category: c for c in compute_category_attribution(summary, summary)}
        assert cats["launch_overhead"].before_ms == 2.0
    finally:
        prof.close()


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
        schema_version="3.24.14",
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
        schema_version="3.25.0",
        total_gpu_ns=100,
        kernel_rows=100,
        kernels=[],
        nvtx=[],
        overlap={},
    )
    warnings, conf = collect_sanity_warnings(matched, other)
    assert conf == 0.0
    assert any("schema" in w.lower() for w in warnings)


def _summary(**over):
    from nsys_ai.diff import ProfileSummary

    base = dict(
        path="x", gpu=0, schema_version="3.24.14", total_gpu_ns=100,
        kernel_rows=100, kernels=[], nvtx=[], overlap={},
    )
    return ProfileSummary(**{**base, **over})


def test_nsys_build_bump_alone_does_not_zero_comparability():
    """A different nsys build on the same export schema is still comparable.

    ``ProfileSummary.schema_version`` used to be fed the *product* version, so
    re-profiling after any toolkit update zeroed the comparability confidence
    and suppressed the verdict. Real case from the local corpus: 2026.1.1.204
    and 2026.1.2.63 both export schema 3.24.14.
    """
    from nsys_ai.diff import collect_sanity_warnings

    before = _summary(product_version="2026.1.1.204")
    after = _summary(product_version="2026.1.2.63")

    warnings, conf = collect_sanity_warnings(before, after)
    assert conf == 1.0, "same export schema must stay fully comparable"
    assert any("build differs" in w for w in warnings), "the bump is still worth reporting"
    assert not any("export schema differs" in w for w in warnings)


def test_export_schema_mismatch_still_zeros_comparability():
    """The hard gate must survive: a real schema change is incomparable."""
    from nsys_ai.diff import collect_sanity_warnings

    warnings, conf = collect_sanity_warnings(
        _summary(schema_version="3.24.14", product_version="2026.1.1.204"),
        _summary(schema_version="3.27.0", product_version="2026.1.1.204"),
    )
    assert conf == 0.0
    assert any("export schema differs" in w for w in warnings)


@pytest.mark.parametrize("empty_side", ["before", "after", "both"])
def test_empty_kernel_side_zeros_comparability(empty_side):
    """An empty side is comparability zero, not a 100% improvement.

    Every other ratio in collect_sanity_warnings guards against a zero
    denominator, so a side with no kernels used to leave confidence at 1.0 and
    the verdict at improvement_likely.
    """
    from nsys_ai.diff import collect_sanity_warnings, compute_verdict

    warnings, conf = collect_sanity_warnings(
        _summary(kernel_rows=0 if empty_side in ("before", "both") else 100),
        _summary(kernel_rows=0 if empty_side in ("after", "both") else 100),
    )
    assert conf == 0.0
    # Same wording the `profile` command refuses this condition with.
    assert any("no GPU kernel activity" in w for w in warnings)
    expected_subject = (
        "Both the before and after profiles contain"
        if empty_side == "both"
        else f"The {empty_side} profile contains"
    )
    assert any(w.startswith(expected_subject) for w in warnings), warnings
    assert compute_verdict(-100.0, conf) == "inconclusive"


def test_executive_summary_refuses_an_empty_capture(tmp_path):
    """The deterministic summary states the refusal instead of narrating deltas."""
    from nsys_ai.ai.diff_narrative import build_executive_summary
    from nsys_ai.diff import diff_profiles
    from nsys_ai.profile import Profile

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[])

    with Profile(str(before)) as bp, Profile(str(after)) as ap:
        summary = diff_profiles(bp, ap, gpu=0)

    text = build_executive_summary(summary)
    assert "no GPU kernel activity" in text
    assert "Total GPU time went" not in text
    assert "Largest improvement" not in text


def test_build_profile_summary_reads_export_schema_not_product_version():
    """Guards the assignment itself, end-to-end on a real export.

    The two are trivially told apart by shape — export schema is "3.24.14",
    the nsys build is "2026.1.1.204" — so this fails loudly if the fields are
    ever swapped back.
    """
    from nsys_ai.diff import build_profile_summary
    from nsys_ai.profile import Profile

    fixture = pathlib.Path(__file__).parent / "fixtures" / "mock.sqlite"
    if not fixture.exists():
        pytest.skip("mock.sqlite fixture not present")

    with Profile(str(fixture)) as prof:
        s = build_profile_summary(prof, gpu=None, trim=None)

    assert s.schema_version == prof.schema.schema_version == "3.24.14"
    assert s.product_version == prof.schema.version
    assert s.product_version != s.schema_version


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
    assert payload["before"]["profile_id"].startswith("nsys2:")
    assert payload["after"]["profile_id"].startswith("nsys2:")

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

    # category_attribution is a list of category bucket entries — all four
    # step-time buckets (launch_overhead carved from idle; see §1.6).
    cats = payload["category_attribution"]
    assert isinstance(cats, list)
    seen = {c["category"] for c in cats}
    assert seen == {"compute", "communication", "launch_overhead", "idle"}


def test_diff_json_includes_communication_and_idle_summary_axes(tmp_path):
    """Diff JSON exposes drillable communication and idle summary axes."""
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    ms = 1_000_000
    strings = {
        1: "compute_A",
        2: "compute_A_dem",
        3: "compute_B",
        4: "compute_B_dem",
        5: "ncclAllReduceKernel",
        6: "ncclAllReduceKernel_dem",
        7: "ncclAllGatherKernel",
        8: "ncclAllGatherKernel_dem",
    }
    _make_named_profile(
        str(before),
        strings=strings,
        kernels=[
            (0, 10 * ms, 0, 7, 1, 1, 2),
            (12 * ms, 17 * ms, 0, 17, 2, 5, 6),
            (18 * ms, 20 * ms, 0, 17, 3, 7, 8),
            (30 * ms, 40 * ms, 0, 7, 4, 3, 4),
        ],
    )
    _make_named_profile(
        str(after),
        strings=strings,
        kernels=[
            (0, 10 * ms, 0, 7, 1, 1, 2),
            (12 * ms, 20 * ms, 0, 17, 2, 5, 6),
            (21 * ms, 22 * ms, 0, 17, 3, 7, 8),
            (45 * ms, 55 * ms, 0, 7, 4, 3, 4),
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
            "--limit",
            "5",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    comm = payload["communication_summary"]
    assert comm["axis"] == "communication"
    assert comm["total_basis"] == "exposed comm"
    assert comm["before_ms"] == 7.0
    assert comm["after_ms"] == 9.0
    assert comm["delta_ms"] == 2.0
    comm_entries = {entry["key"]: entry for entry in comm["entries"]}
    assert comm_entries["allreduce"]["delta_ms"] == 3.0
    assert comm_entries["allreduce"]["classification"] == "regression"
    assert comm_entries["allreduce"]["selection"]["source"] == "diff:communication_summary"
    assert comm_entries["allreduce"]["selection"]["profile_id"] == payload["after"]["profile_id"]
    assert comm_entries["allreduce"]["selection"]["start_ns"] == 12 * ms
    assert comm_entries["allreduce"]["selection"]["end_ns"] == 20 * ms
    assert comm_entries["allreduce"]["selection"]["gpu_ids"] == [0]
    assert comm_entries["allgather"]["delta_ms"] == -1.0

    idle = payload["idle_summary"]
    assert idle["axis"] == "idle"
    assert idle["total_basis"] == "wall-clock idle"
    assert idle["before_ms"] == 13.0
    assert idle["after_ms"] == 26.0
    assert idle["delta_ms"] == 13.0
    assert len(idle["entries"]) == 1
    gap = idle["entries"][0]
    assert gap["delta_ms"] == 15.0
    assert gap["classification"] == "grown"
    assert gap["metadata"]["device_id"] == 0
    assert gap["metadata"]["stream_id"] == 7
    assert gap["selection"]["source"] == "diff:idle_summary"
    assert gap["selection"]["profile_id"] == payload["after"]["profile_id"]
    assert gap["selection"]["start_ns"] == 10 * ms
    assert gap["selection"]["end_ns"] == 45 * ms
    assert gap["selection"]["gpu_ids"] == [0]


def test_diff_compute_only_omits_empty_summary_axes(tmp_path):
    """A compute-only diff must not render empty communication/idle axis sections."""
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    ms = 1_000_000
    # One compute kernel each: no NCCL, no inter-kernel gaps -> both axes empty.
    _make_profile(str(before), kernels=[(0, 10 * ms, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 13 * ms, 0, 7, 1, 1, 2)])

    js = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff", str(before), str(after),
         "--gpu", "0", "--format", "json"],
        capture_output=True,
        text=True,
    )
    assert js.returncode == 0, js.stderr
    payload = json.loads(js.stdout)
    # Omitted (null), not an empty "Total 0 -> 0" object.
    assert payload["communication_summary"] is None
    assert payload["idle_summary"] is None

    term = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff", str(before), str(after),
         "--gpu", "0", "--no-ai"],
        capture_output=True,
        text=True,
    )
    assert term.returncode == 0, term.stderr
    assert "Communication/NCCL Summary" not in term.stdout
    assert "Idle Gap Summary" not in term.stdout


def test_v01_confidence_separates_schema_and_gpu_mismatch():
    """c_schema and c_gpu are independent factors; mismatching gpu alone zeros confidence."""
    from nsys_ai.diff import ProfileSummary, collect_sanity_warnings

    # Same schema, different gpu id → c_gpu = 0 → confidence = 0,
    # but the warning text mentions GPU (not schema).
    a = ProfileSummary(
        path="a",
        gpu=0,
        schema_version="3.24.14",
        total_gpu_ns=100,
        kernel_rows=100,
        kernels=[],
        nvtx=[],
        overlap={},
    )
    b = ProfileSummary(
        path="b",
        gpu=1,  # different GPU id, same schema
        schema_version="3.24.14",
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
        schema_version="3.24.14",
        total_gpu_ns=100,
        kernel_rows=100,
        kernels=[],
        nvtx=[],
        overlap=_make_overlap_dict(100, 20, 10, 5),
    )
    bad = ProfileSummary(
        path="a",
        gpu=0,
        schema_version="3.24.14",
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


# ---------------------------------------------------------------------------
# The verdict and comparability score are shown, and an incomparable pair is
# refused instead of decorated with delta tables.
# ---------------------------------------------------------------------------

_MFU_BEFORE = pathlib.Path(__file__).parent / "fixtures" / "mfu_2gpu_before.sqlite"
_MFU_AFTER = pathlib.Path(__file__).parent / "fixtures" / "mfu_2gpu_after.sqlite"


def _incomparable_pair(tmp_path, *, before_n=12, after_n=2):
    """A pair whose only difference is workload size, scoring below the gate.

    ``collect_sanity_warnings`` derives comparability from min/max kernel rows,
    so 12 vs 2 lands at 0.167 — well under MIN_COMPARABILITY_CONFIDENCE.
    """
    before = tmp_path / "incomp_before.sqlite"
    after = tmp_path / "incomp_after.sqlite"

    def _k(n):
        return [(i * 10_000_000, i * 10_000_000 + 5_000_000, 0, 7, i, 1, 2) for i in range(n)]

    _make_profile(str(before), kernels=_k(before_n))
    _make_profile(str(after), kernels=_k(after_n))
    return before, after


def _diff_summary(before, after, **kwargs):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        return diff_profiles(b, a, **kwargs)


def test_diff_terminal_states_the_verdict_and_comparability(tmp_path):
    """The judgement is shown, not left for the reader to infer from deltas.

    Both numbers were already computed, persisted to diff.json and gating CI;
    the terminal was the one consumer never told.
    """
    from nsys_ai.diff_render import format_diff_markdown, format_diff_terminal

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 12_000_000, 0, 7, 1, 1, 2)])
    summary = _diff_summary(before, after, gpu=0, limit=10)
    assert summary.comparability_confidence >= 0.5

    terminal = format_diff_terminal(summary)
    assert f"Verdict: {summary.verdict}" in terminal
    assert "(comparability 1.00)" in terminal

    markdown = format_diff_markdown(summary)
    assert f"**Verdict**: `{summary.verdict}`" in markdown
    assert "comparability `1.00`" in markdown


def test_diff_cli_shows_the_verdict_that_contradicts_the_headline():
    """The reported case: headline says faster, the verdict says neutral.

    Runs the issue's own invocation on the committed pair, so the two numbers
    have to appear together in the output a human actually reads.
    """
    if not _MFU_BEFORE.is_file() or not _MFU_AFTER.is_file():
        raise FileNotFoundError(f"missing fixture profiles: {_MFU_BEFORE} / {_MFU_AFTER}")
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff", str(_MFU_BEFORE), str(_MFU_AFTER), "--no-ai"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Verdict: neutral" in result.stdout, result.stdout[:800]
    assert "comparability 0.89" in result.stdout, result.stdout[:800]


def test_comparability_is_truncated_not_rounded_across_the_gate():
    """A score under the gate must never display as though it had reached it."""
    from nsys_ai.diff_render import _fmt_confidence

    assert _fmt_confidence(0.4996) == "0.49"
    assert _fmt_confidence(0.894) == "0.89"
    assert _fmt_confidence(1.0) == "1.00"
    assert _fmt_confidence(0.0) == "0.00"


def test_diff_terminal_refuses_an_incomparable_pair_instead_of_tabulating_it(tmp_path):
    """Below the gate the deltas are arithmetic; printed as tables they read as findings."""
    from nsys_ai.diff_render import format_diff_terminal

    before, after = _incomparable_pair(tmp_path)
    summary = _diff_summary(before, after, gpu=0, limit=10)
    assert summary.comparability_confidence < 0.5
    assert summary.verdict == "inconclusive"

    out = format_diff_terminal(summary)

    assert "Verdict: inconclusive" in out
    assert "No comparison was made" in out
    assert "Kernel row counts differ a lot" in out
    # The tables that would have been read as findings are gone, not merely
    # preceded by a caveat.
    for section in ("Executive Summary", "Overall", "Top regressions", "Top improvements"):
        assert section not in out, f"{section!r} survived the refusal:\n{out}"


def test_diff_markdown_refuses_an_incomparable_pair(tmp_path):
    from nsys_ai.diff_render import format_diff_markdown

    before, after = _incomparable_pair(tmp_path)
    summary = _diff_summary(before, after, gpu=0, limit=10)

    out = format_diff_markdown(summary)
    assert "### No comparison was made" in out
    assert "Kernel row counts differ a lot" in out
    for section in ("### Top regressions", "### Top improvements", "### Overall"):
        assert section not in out, f"{section!r} survived the refusal:\n{out}"


def test_incomparable_refusal_states_a_reason_when_no_warning_fired(tmp_path):
    """The refusal must never be a bare verdict word.

    A workload ratio between 0.33 and 0.5 drops confidence under the gate
    without tripping the 3x warning threshold, so ``warnings`` is empty and the
    score against the gate is the only reason available.
    """
    from nsys_ai.diff_render import format_diff_terminal, incomparable_reason, is_incomparable

    before, after = _incomparable_pair(tmp_path, before_n=12, after_n=5)
    summary = _diff_summary(before, after, gpu=0, limit=10)
    assert summary.warnings == [], summary.warnings
    assert is_incomparable(summary)

    reason = incomparable_reason(summary)
    assert reason.strip()
    assert "0.41" in reason and "0.50" in reason

    out = format_diff_terminal(summary)
    assert "No comparison was made" in out
    assert reason in out


def test_diff_terminal_multi_withholds_per_gpu_tables_when_incomparable(tmp_path):
    """Per-GPU tables are the same deltas sliced by device; they must not survive."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles_all_gpus
    from nsys_ai.diff_render import format_diff_terminal_multi

    before, after = _incomparable_pair(tmp_path)
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        global_summary, per_gpu = diff_profiles_all_gpus(b, a, limit=10)
    assert global_summary.comparability_confidence < 0.5

    out = format_diff_terminal_multi(global_summary, per_gpu)
    assert "No comparison was made" in out
    assert "Per-GPU Overview" not in out, out
    assert "Top regressions" not in out, out


def _mixed_comparability_pair(tmp_path):
    """A node whose GPU 0 slice is sound and whose GPU 1 slice is not.

    Comparability is derived per slice from min/max kernel rows, so GPU 0 at
    100 vs 100 clears the gate while GPU 1 at 3 vs 40 lands at 0.075. The
    node-wide score stays above the gate, which is the case that matters: the
    global refusal never fires, so nothing else stops GPU 1's deltas.
    """
    before = tmp_path / "mixed_before.sqlite"
    after = tmp_path / "mixed_after.sqlite"

    def _k(dev, n, base):
        return [
            (base + i * 10_000_000, base + i * 10_000_000 + 5_000_000, dev, 7, i, 1, 2)
            for i in range(n)
        ]

    _make_profile(str(before), kernels=_k(0, 100, 0) + _k(1, 3, 0))
    _make_profile(str(after), kernels=_k(0, 100, 0) + _k(1, 40, 0))
    return before, after


def test_diff_terminal_multi_withholds_only_the_devices_that_cannot_be_compared(tmp_path):
    """A node-wide pass must not carry an incomparable device's deltas with it.

    The global guard is not enough: a rank that dropped out, or one whose
    kernel set barely overlaps, scores below the gate on its own while the node
    total clears it. Its per-device rows are then the same arithmetic the
    refusal exists to withhold.
    """
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles_all_gpus
    from nsys_ai.diff_render import format_diff_terminal_multi, is_incomparable

    before, after = _mixed_comparability_pair(tmp_path)
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        global_summary, per_gpu = diff_profiles_all_gpus(b, a, limit=10)

    assert not is_incomparable(global_summary), global_summary.comparability_confidence
    assert not is_incomparable(per_gpu[0]), per_gpu[0].comparability_confidence
    assert is_incomparable(per_gpu[1]), per_gpu[1].comparability_confidence

    out = format_diff_terminal_multi(global_summary, per_gpu)
    assert "Per-GPU Overview" in out, out
    overview = out.split("Per-GPU Overview", 1)[1].split("\n\n", 1)[0]
    assert "\n  0 |" in overview, overview
    assert "\n  1 |" not in overview, overview
    assert "Top regressions (GPU 1)" not in out, out
    # Named, not silently dropped -- the device exists and its numbers are in JSON.
    assert "GPU 1 (0.07)" in out, out


def test_diff_markdown_multi_withholds_only_the_devices_that_cannot_be_compared(tmp_path):
    """The markdown renderer is the one multi-GPU path with no refusal test."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles_all_gpus
    from nsys_ai.diff_render import format_diff_markdown_multi

    before, after = _mixed_comparability_pair(tmp_path)
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        global_summary, per_gpu = diff_profiles_all_gpus(b, a, limit=10)

    out = format_diff_markdown_multi(global_summary, per_gpu)
    assert "| `0` |" in out, out
    assert "| `1` |" not in out, out
    assert "#### GPU 1" not in out, out
    assert "GPU 1 (0.07)" in out, out


def test_diff_markdown_multi_refuses_an_incomparable_pair(tmp_path):
    """The whole-node refusal, on the renderer the other tests did not cover."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles_all_gpus
    from nsys_ai.diff_render import format_diff_markdown_multi

    before, after = _incomparable_pair(tmp_path)
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        global_summary, per_gpu = diff_profiles_all_gpus(b, a, limit=10)

    out = format_diff_markdown_multi(global_summary, per_gpu)
    assert "No comparison was made" in out
    assert "Per-GPU Breakdown" not in out, out
    assert "Per-GPU Top Regressions" not in out, out


def test_comparability_display_never_reads_one_hundredth_low():
    """Truncation must not inherit the multiply's binary error.

    math.floor(0.29 * 100) is 28, because 0.29 * 100 is 28.999999999999996.
    Three of the hundred two-decimal values were displayed a hundredth below
    the score the gate actually used.
    """
    from decimal import Decimal

    from nsys_ai.diff_render import _fmt_confidence

    for hundredths in range(101):
        exact = Decimal(hundredths) / 100
        assert _fmt_confidence(float(exact)) == f"{exact:.2f}", hundredths


def test_no_model_is_asked_to_narrate_a_pair_that_cannot_be_compared(tmp_path):
    """The refusal discards the narrative, so requesting one is billed for nothing.

    An empty side was already refused here for the stronger reason that the
    model would narrate vanished kernels as a win. Low comparability is the
    same condition one step weaker.
    """
    from nsys_ai import chat_config
    from nsys_ai.ai import diff_narrative as narrative_mod

    before, after = _incomparable_pair(tmp_path)
    summary = _diff_summary(before, after, gpu=0, limit=10)
    assert summary.comparability_confidence < 0.5

    asked = []

    # generate_diff_narrative resolves the model lazily from chat_config, so
    # this is the last gate before any provider call. Reaching it at all is the
    # failure.
    def _refuse_to_be_asked(*args, **kwargs):
        asked.append(args)
        raise AssertionError("an incomparable pair must not reach the provider")

    original = chat_config._get_model_and_key
    chat_config._get_model_and_key = _refuse_to_be_asked
    try:
        result = narrative_mod.generate_diff_narrative(summary)
    finally:
        chat_config._get_model_and_key = original

    assert not asked
    assert result.ai_narrative is None
    assert result.executive_summary
def _kernels(dev, count, duration_ns, *, base_correlation=0):
    return [
        (i * 2_000_000, i * 2_000_000 + duration_ns, dev, 7, base_correlation + i, 1, 2)
        for i in range(count)
    ]


def test_a_device_holding_almost_no_time_does_not_refuse_the_comparison(tmp_path):
    """One microsecond on a second device is not a topology change.

    An NCCL or cuDNN bootstrap can touch a device for microseconds. Refusing on
    the device set alone turned that into comparability 0.00 and failed a CI
    gate on a pair whose totals agree to four decimal places.
    """
    before = tmp_path / "boot_before.sqlite"
    after = tmp_path / "boot_after.sqlite"
    _make_profile(
        str(before),
        kernels=_kernels(0, 200, 1_000_000) + [(0, 1_000, 1, 7, 9_999, 1, 2)],
    )
    _make_profile(str(after), kernels=_kernels(0, 200, 1_000_000))

    summary = _diff_summary(before, after, gpu=None, limit=10)
    assert summary.comparability_confidence >= 0.5, summary.warnings
    assert summary.verdict == "neutral", summary.verdict
    # Still said, because it is still true -- just not fatal.
    assert any("under the 1% that would move the totals" in w for w in summary.warnings), (
        summary.warnings
    )


def test_a_rank_that_dropped_out_still_refuses_the_comparison(tmp_path):
    """The immaterial-share exemption must not swallow the case it came from."""
    before = tmp_path / "rank_before.sqlite"
    after = tmp_path / "rank_after.sqlite"
    _make_profile(
        str(before),
        kernels=_kernels(0, 200, 1_000_000) + _kernels(1, 200, 1_000_000, base_correlation=1000),
    )
    _make_profile(str(after), kernels=_kernels(0, 200, 1_000_000))

    summary = _diff_summary(before, after, gpu=None, limit=10)
    assert summary.comparability_confidence == 0.0, summary.warnings
    assert summary.verdict == "inconclusive"
    assert any("GPU count differs" in w for w in summary.warnings), summary.warnings


def test_a_missing_gpu_selection_is_not_also_called_an_empty_capture(tmp_path):
    """One cause, one sentence.

    `--gpu 5` on two profiles that both ran used to print the device warning
    and then "the profile contains no GPU kernel activity" -- false about the
    profile, and it sends the reader off to check whether the run executed.
    """
    before = tmp_path / "sel_before.sqlite"
    after = tmp_path / "sel_after.sqlite"
    _make_profile(str(before), kernels=_kernels(0, 20, 1_000_000))
    _make_profile(str(after), kernels=_kernels(0, 20, 1_000_000))

    summary = _diff_summary(before, after, gpu=5, limit=10)
    assert summary.comparability_confidence == 0.0
    assert any("GPU 5 recorded no kernels" in w for w in summary.warnings), summary.warnings
    assert not any("no GPU kernel activity" in w for w in summary.warnings), summary.warnings


def test_two_different_captures_carrying_no_capture_metadata_are_not_called_one(tmp_path):
    """Equal ids mean equal row counts when the metadata that names a run is absent.

    `_make_profile` writes no TARGET_INFO_* / ANALYSIS_* tables, so the id
    rests on the kernel row count alone. Keying the identity exemption on the
    schema version let two unrelated captures of the same size be declared
    "the same capture, every delta below is self-noise".
    """
    import sqlite3

    from nsys_ai.fingerprint import get_profile_id, profile_id_is_capture_derived

    before = tmp_path / "anon_before.sqlite"
    after = tmp_path / "anon_after.sqlite"
    # Same row count and same total duration, different distribution.
    _make_profile(
        str(before),
        kernels=[(0, 40_000_000, 0, 7, 0, 1, 2), (50_000_000, 60_000_000, 0, 7, 1, 1, 2)],
    )
    _make_profile(
        str(after),
        kernels=[(0, 10_000_000, 0, 7, 0, 1, 2), (50_000_000, 90_000_000, 0, 7, 1, 1, 2)],
    )
    for path in (before, after):
        conn = sqlite3.connect(str(path))
        try:
            assert not profile_id_is_capture_derived(conn), path
            assert get_profile_id(conn, fallback_path=str(path))
        finally:
            conn.close()

    summary = _diff_summary(before, after, gpu=0, limit=10)
    assert not any("same capture" in w for w in summary.warnings), summary.warnings


def test_executive_summary_never_signs_a_magnitude_after_a_direction_word(tmp_path):
    """"went faster by -46.37ms" is a double negative that inverts the claim."""
    import re

    from nsys_ai.ai.diff_narrative import build_executive_summary

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 20_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    faster = build_executive_summary(_diff_summary(before, after, gpu=0, limit=10))
    assert "went faster by 10.00ms" in faster, faster
    assert not re.search(r"went (faster|slower) by -", faster), faster

    slower = build_executive_summary(_diff_summary(after, before, gpu=0, limit=10))
    assert "went slower by 10.00ms" in slower, slower
    assert not re.search(r"went (faster|slower) by -", slower), slower


def test_executive_summary_says_unchanged_rather_than_slower_by_zero(tmp_path):
    from nsys_ai.ai.diff_narrative import build_executive_summary

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    text = build_executive_summary(_diff_summary(before, after, gpu=0, limit=10))
    assert "Total GPU time was unchanged." in text, text
    assert "slower" not in text and "faster" not in text, text


# ---------------------------------------------------------------------------
# Inputs that cannot support a comparison: the same capture on both sides, a
# side that never recorded a device, a GPU selection that does not exist, two
# captures of different topologies, and a delta inside run-to-run noise.
# ---------------------------------------------------------------------------


def _lost_device_pair(tmp_path):
    """before ran on GPUs 0 and 1; after recorded GPU 0 only.

    Per-GPU work is identical, so every delta the pipeline finds is the device
    the after side never captured.
    """
    ms = 1_000_000
    before = tmp_path / "lost_before.sqlite"
    after = tmp_path / "lost_after.sqlite"
    _make_profile(
        str(before),
        kernels=[
            (0, 50 * ms, 0, 7, 1, 1, 2),
            (60 * ms, 110 * ms, 0, 7, 2, 3, 4),
            (0, 50 * ms, 1, 7, 3, 1, 2),
            (60 * ms, 110 * ms, 1, 7, 4, 3, 4),
        ],
    )
    _make_profile(
        str(after),
        kernels=[
            (0, 50 * ms, 0, 7, 1, 1, 2),
            (60 * ms, 110 * ms, 0, 7, 2, 3, 4),
        ],
    )
    return before, after


def _all_gpu_summary(before, after, **kwargs):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles_all_gpus

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        return diff_profiles_all_gpus(b, a, **kwargs)


def test_a_side_that_lost_a_gpu_is_not_a_clean_win(tmp_path):
    """Half the devices missing read as half the GPU time saved.

    The all-GPU invocation is the one that matters: with no ``--gpu`` both
    sides aggregate over whatever devices they happen to have, so the
    empty-capture check never fires and the missing rank lands in the total as
    an improvement.
    """
    before, after = _lost_device_pair(tmp_path)
    global_summary, _ = _all_gpu_summary(before, after, limit=10)

    assert global_summary.step_time_delta_pct == -50.0, "the fixture must show a large win"
    assert global_summary.verdict == "inconclusive", global_summary.verdict
    assert global_summary.comparability_confidence == 0.0
    assert any(
        "GPU count differs" in w and "[0, 1]" in w and "[0]" in w
        for w in global_summary.warnings
    ), global_summary.warnings


def test_a_gpu_missing_from_one_side_names_the_side_and_the_device(tmp_path):
    """Scoped to the lost device, the reason must not be "the profile is empty".

    The empty-capture warning already fires on this path, and on its own it
    sends the reader to check whether the run executed — when the run executed
    fine and only that device was never captured.
    """
    before, after = _lost_device_pair(tmp_path)
    summary = _diff_summary(before, after, gpu=1, limit=10)

    assert summary.verdict == "inconclusive"
    assert summary.comparability_confidence == 0.0
    named = [w for w in summary.warnings if "GPU 1 recorded kernels in the before" in w]
    assert named, summary.warnings
    assert "none in the after profile" in named[0]
    assert "not a change in the workload" in named[0]


def test_a_gpu_present_in_neither_profile_says_so(tmp_path):
    """``--gpu 5`` on two 2-GPU captures is an empty selection, not two empty captures."""
    before, after = _lost_device_pair(tmp_path)
    summary = _diff_summary(before, after, gpu=5, limit=10)

    assert summary.verdict == "inconclusive"
    assert summary.comparability_confidence == 0.0
    named = [w for w in summary.warnings if "GPU 5 recorded no kernels in either profile" in w]
    assert named, summary.warnings
    assert "no before/after claim can be made" in named[0]


def test_a_topology_mismatch_is_refused_rather_than_scored(tmp_path):
    """2 GPUs against 4 is not a before and an after; it is different hardware."""
    ms = 1_000_000
    before = tmp_path / "topo_before.sqlite"
    after = tmp_path / "topo_after.sqlite"
    _make_profile(str(before), kernels=[(0, 50 * ms, d, 7, i, 1, 2) for i, d in enumerate([0, 1])])
    _make_profile(
        str(after), kernels=[(0, 50 * ms, d, 7, i, 1, 2) for i, d in enumerate([0, 1, 2, 3])]
    )
    global_summary, _ = _all_gpu_summary(before, after, limit=10)

    assert global_summary.step_time_delta_pct == 100.0, "the fixture must show a large loss"
    assert global_summary.verdict == "inconclusive"
    assert global_summary.comparability_confidence == 0.0
    assert any("GPU count differs" in w for w in global_summary.warnings), global_summary.warnings


def test_matching_gpu_count_with_different_ids_warns_without_refusing(tmp_path):
    """Same amount of hardware, different ordinals: comparable, worth saying."""
    ms = 1_000_000
    before = tmp_path / "ids_before.sqlite"
    after = tmp_path / "ids_after.sqlite"
    _make_profile(str(before), kernels=[(0, 50 * ms, d, 7, i, 1, 2) for i, d in enumerate([0, 1])])
    _make_profile(str(after), kernels=[(0, 50 * ms, d, 7, i, 1, 2) for i, d in enumerate([2, 3])])
    global_summary, _ = _all_gpu_summary(before, after, limit=10)

    assert global_summary.verdict != "inconclusive", global_summary.warnings
    assert global_summary.comparability_confidence >= 0.5
    assert any(
        "different GPU ids" in w and "[0, 1]" in w and "[2, 3]" in w
        for w in global_summary.warnings
    ), global_summary.warnings


def test_matching_device_sets_add_no_warning(tmp_path):
    """The gate must stay silent on a pair that recorded the same GPUs."""
    ms = 1_000_000
    before = tmp_path / "same_before.sqlite"
    after = tmp_path / "same_after.sqlite"
    _make_profile(str(before), kernels=[(0, 50 * ms, d, 7, i, 1, 2) for i, d in enumerate([0, 1])])
    _make_profile(str(after), kernels=[(0, 52 * ms, d, 7, i, 1, 2) for i, d in enumerate([0, 1])])
    global_summary, _ = _all_gpu_summary(before, after, limit=10)

    assert global_summary.comparability_confidence == 1.0
    assert not [w for w in global_summary.warnings if "GPU" in w], global_summary.warnings


def test_a_profile_diffed_against_itself_is_self_noise_not_a_verdict(tmp_path):
    """Same capture on both sides: the answer is "nothing was rerun"."""
    before, _ = _lost_device_pair(tmp_path)
    global_summary, _ = _all_gpu_summary(before, before, limit=10)

    assert global_summary.verdict == "neutral"
    assert any("the same capture" in w for w in global_summary.warnings), global_summary.warnings
    assert any("self-noise" in w for w in global_summary.warnings), global_summary.warnings

    scoped = _diff_summary(before, before, gpu=0, limit=10)
    assert scoped.verdict == "neutral"
    assert any("the same capture" in w for w in scoped.warnings), scoped.warnings


def test_the_same_capture_is_recognised_through_a_copy_under_another_name(tmp_path):
    """The id is content-derived, so a renamed copy is still the same capture."""
    import shutil

    copy = tmp_path / "renamed.sqlite"
    shutil.copy(_MFU_BEFORE, copy)
    global_summary, _ = _all_gpu_summary(_MFU_BEFORE, copy, limit=10)

    assert global_summary.before.path != global_summary.after.path
    assert global_summary.before.profile_id == global_summary.after.profile_id
    assert global_summary.verdict == "neutral"
    assert any("the same capture" in w for w in global_summary.warnings), global_summary.warnings


def test_identity_is_not_claimed_from_a_profile_that_carries_no_capture_metadata(tmp_path):
    """Two unrelated captures must never be called one.

    A profile with no capture metadata contributes only its kernel row count to
    the profile id, so two different runs with the same number of kernels hash
    the same. The identity claim must not rest on that.
    """
    from nsys_ai.diff import same_capture

    ms = 1_000_000
    before = tmp_path / "meta_before.sqlite"
    after = tmp_path / "meta_after.sqlite"
    _make_profile(str(before), kernels=[(0, 50 * ms, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 50 * ms, 0, 7, 1, 3, 4)])
    summary = _diff_summary(before, after, gpu=0, limit=10)

    assert summary.before.profile_id == summary.after.profile_id, "the id collision is the setup"
    assert summary.before.schema_version is None
    assert not same_capture(summary.before, summary.after)
    assert not [w for w in summary.warnings if "the same capture" in w], summary.warnings


def test_two_windows_of_one_capture_are_still_compared(tmp_path):
    """An iteration diff reads one capture twice through different windows.

    Both sides share a path and an id, and it is a real comparison; only the
    identical window makes the two sides one measurement.
    """
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles

    ms = 1_000_000
    path = tmp_path / "windows.sqlite"
    _make_profile(
        str(path),
        kernels=[
            (0, 50 * ms, 0, 7, 1, 1, 2),
            (100 * ms, 190 * ms, 0, 7, 2, 1, 2),
        ],
    )
    with profile_mod.open(str(path)) as b, profile_mod.open(str(path)) as a:
        summary = diff_profiles(
            b, a, gpu=0, trim_before=(0, 60 * ms), trim_after=(100 * ms, 200 * ms), limit=10
        )

    assert not [w for w in summary.warnings if "the same capture" in w], summary.warnings
    assert summary.kernel_diffs[0].delta_ns != 0


def test_a_step_time_delta_inside_the_noise_floor_stays_neutral():
    """A gate tighter than run-to-run jitter cannot make a coin flip a verdict."""
    from nsys_ai.diff import NOISE_FLOOR_STEP_TIME_PCT, compute_verdict

    assert NOISE_FLOOR_STEP_TIME_PCT < 5.0, "the floor must not move the default verdict"
    assert compute_verdict(1.0, 1.0, regression_pct=0.5) == "neutral"
    assert compute_verdict(-1.0, 1.0, regression_pct=0.5) == "neutral"
    # A change past the floor is still judged against the caller's threshold.
    assert compute_verdict(3.0, 1.0, regression_pct=0.5) == "regression_likely"
    assert compute_verdict(-3.0, 1.0, regression_pct=0.5) == "improvement_likely"
    # And the floor never rescues an incomparable pair into neutral.
    assert compute_verdict(0.1, 0.2, regression_pct=0.5) == "inconclusive"


def test_the_noise_floor_says_why_it_overrode_the_threshold(tmp_path):
    """Neutral beside a delta past the requested gate has to explain itself."""
    ms = 1_000_000
    before = tmp_path / "floor_before.sqlite"
    after = tmp_path / "floor_after.sqlite"
    _make_profile(str(before), kernels=[(0, 100 * ms, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 101 * ms, 0, 7, 1, 1, 2)])
    summary = _diff_summary(before, after, gpu=0, limit=10, regression_pct=0.5)

    assert summary.step_time_delta_pct is not None
    assert 0.5 <= abs(summary.step_time_delta_pct) < 2.0, summary.step_time_delta_pct
    assert summary.verdict == "neutral"
    said = [w for w in summary.warnings if "noise floor" in w]
    assert said, summary.warnings
    assert "reported as neutral" in said[0]


def test_a_default_gate_never_mentions_the_noise_floor(tmp_path):
    """The floor sits under the default threshold, so it must stay invisible there."""
    ms = 1_000_000
    before = tmp_path / "quiet_before.sqlite"
    after = tmp_path / "quiet_after.sqlite"
    _make_profile(str(before), kernels=[(0, 100 * ms, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 101 * ms, 0, 7, 1, 1, 2)])
    summary = _diff_summary(before, after, gpu=0, limit=10)

    assert summary.verdict == "neutral"
    assert not [w for w in summary.warnings if "noise floor" in w], summary.warnings


def test_cli_refuses_a_pair_that_lost_a_gpu_and_fails_the_gate(tmp_path):
    """End to end, through the surface a human reads and a CI job exits on."""
    before, after = _lost_device_pair(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nsys_ai",
            "diff",
            str(before),
            str(after),
            "--no-ai",
            "--exit-on-regression",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, (result.returncode, result.stdout[-2000:], result.stderr[-2000:])
    assert "Verdict: inconclusive" in result.stdout, result.stdout[:1200]
    assert "No comparison was made" in result.stdout, result.stdout[:1200]
    assert "GPU count differs" in result.stdout, result.stdout[:1200]
    # The withheld tables must not come back through the per-GPU sections.
    assert "Per-GPU Overview" not in result.stdout, result.stdout[:2000]


def test_cli_self_diff_passes_the_gate_and_says_it_is_the_same_capture(tmp_path):
    """A capture compared with itself is not a regression, and says why."""
    before, _ = _lost_device_pair(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nsys_ai",
            "diff",
            str(before),
            str(before),
            "--no-ai",
            "--exit-on-regression",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (result.returncode, result.stderr[-2000:])
    assert "Verdict: neutral" in result.stdout, result.stdout[:1200]
    assert "the same capture" in result.stdout, result.stdout[:1200]
