"""Tests for the nccl_anomaly skill.

It detects NCCL collectives whose duration is an outlier for their op type.
The computation moved from a multi-CTE SQL self-join to a flat fetch + Python
(issue #251) — the SQL crashed/hung DuckDB's sqlite_scanner on a direct-attached
profile. These tests pin the anomaly logic, the op-type classification order,
and that both engines agree (the robustness the rewrite is for).
"""

import sqlite3

import pytest

from nsys_ai.skills.builtins import nccl_anomaly as skill_mod

_MS = 1_000_000
# (start_ns, dur_ms, streamId, name)
_SCHEMA_COLS = "start INT, [end] INT, deviceId INT, streamId INT, correlationId INT, shortName INT, demangledName INT"


def _rows(named_kernels):
    """Return (stringids, kernels) for the given named kernels."""
    names = {}
    for _, _, _, name in named_kernels:
        names.setdefault(name, len(names) + 1)
    stringids = [(sid, n) for n, sid in names.items()]
    kernels = [
        (start_ns, start_ns + int(dur_ms * _MS), 0, stream, i, names[name], names[name])
        for i, (start_ns, dur_ms, stream, name) in enumerate(named_kernels)
    ]
    return stringids, kernels


def _build_sqlite(named_kernels):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute(f"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL({_SCHEMA_COLS})")
    stringids, kernels = _rows(named_kernels)
    conn.executemany("INSERT INTO StringIds VALUES(?,?)", stringids)
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start, [end], deviceId, streamId, "
        "correlationId, shortName, demangledName) VALUES(?,?,?,?,?,?,?)",
        kernels,
    )
    conn.commit()
    return conn


def _build_duckdb(named_kernels):
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE StringIds(id BIGINT, value VARCHAR)")
    con.execute(
        'CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start BIGINT, "end" BIGINT, deviceId INT, '
        'streamId INT, correlationId INT, shortName BIGINT, demangledName BIGINT)'
    )
    stringids, kernels = _rows(named_kernels)
    for r in stringids:
        con.execute("INSERT INTO StringIds VALUES(?,?)", list(r))
    for r in kernels:
        con.execute(
            'INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start, "end", deviceId, streamId, '
            "correlationId, shortName, demangledName) VALUES(?,?,?,?,?,?,?)",
            list(r),
        )
    return con


# 5 AllReduce (4x 1ms + 1x 20ms) -> avg 4.8ms, the 20ms one is the anomaly at 3x.
# 2 ReduceScatter at 5ms (uniform, no anomaly). 1 non-NCCL gemm (excluded).
_KERNELS = [
    (0, 1.0, 8, "ncclDevKernel_AllReduce_x"),
    (2 * _MS, 1.0, 8, "ncclDevKernel_AllReduce_x"),
    (4 * _MS, 1.0, 8, "ncclDevKernel_AllReduce_x"),
    (6 * _MS, 1.0, 8, "ncclDevKernel_AllReduce_x"),
    (8 * _MS, 20.0, 8, "ncclDevKernel_AllReduce_x"),
    (40 * _MS, 5.0, 9, "ncclDevKernel_ReduceScatter_y"),
    (50 * _MS, 5.0, 9, "ncclDevKernel_ReduceScatter_y"),
    (60 * _MS, 3.0, 7, "ampere_sgemm_128x64"),
]


def test_flags_slow_collective_with_correct_ratio():
    rows = skill_mod._execute(_build_sqlite(_KERNELS), threshold=3.0, limit=20)
    assert len(rows) == 1, "only the 20ms AllReduce is >3x its type's average"
    r = rows[0]
    assert r["op_type"] == "AllReduce"
    assert r["dur_ms"] == 20.0
    assert r["avg_ms"] == 4.8  # (4*1 + 20)/5
    assert r["ratio_to_avg"] == 4.2  # 20 / 4.8
    assert r["total_count"] == 5  # all AllReduce ops of that type
    assert r["streamId"] == 8


def test_op_type_classification_order():
    """AllReduce and ReduceScatter must not be swallowed by the bare 'Reduce'
    substring — the classifier checks specific names first."""
    assert skill_mod._op_type("ncclDevKernel_AllReduce_x") == "AllReduce"
    assert skill_mod._op_type("ncclDevKernel_ReduceScatter_y") == "ReduceScatter"
    assert skill_mod._op_type("ncclDevKernel_Reduce_z") == "Reduce"
    assert skill_mod._op_type("ncclDevKernel_AllGather_w") == "AllGather"
    assert skill_mod._op_type("some_unlabelled_nccl_kernel") == "Other"


def test_uniform_durations_have_no_anomaly():
    kernels = [(i * 10 * _MS, 5.0, 8, "ncclDevKernel_AllReduce_x") for i in range(6)]
    assert skill_mod._execute(_build_sqlite(kernels), threshold=3.0, limit=20) == []


def test_no_nccl_returns_empty():
    kernels = [(0, 5.0, 7, "ampere_sgemm"), (10 * _MS, 50.0, 7, "ampere_sgemm")]
    assert skill_mod._execute(_build_sqlite(kernels), threshold=3.0, limit=20) == []


def test_threshold_controls_sensitivity():
    # Same data; a threshold above the 4.2x ratio drops the anomaly.
    assert len(skill_mod._execute(_build_sqlite(_KERNELS), threshold=3.0)) == 1
    assert skill_mod._execute(_build_sqlite(_KERNELS), threshold=5.0) == []


def test_engine_parity_sqlite_matches_duckdb():
    """The whole point of the rewrite: identical output on both engines, with no
    CTE for the sqlite_scanner to choke on."""
    lite = skill_mod._execute(_build_sqlite(_KERNELS), threshold=2.0, limit=20)
    duck = skill_mod._execute(_build_duckdb(_KERNELS), threshold=2.0, limit=20)
    assert lite == duck and lite  # non-empty and equal
