"""On-demand nvtx_kernel_map builder (issue #257).

The four skills that attribute kernels to NVTX regions depend on a precomputed
`nvtx_kernel_map`, built only during the parquet-cache build. On the direct-attach
no-cache path the map is absent and each skill falls to an in-file IEJoin that
hangs DuckDB's sqlite_scanner. `ensure_nvtx_kernel_map` materialises the map
on-demand via the shared Python sort-merge (`_sweep_nvtx_kernel_map`), fed by
flat fetches that run fine on every backend.
"""

import sqlite3

import pytest

from nsys_ai.parquet_cache import _sweep_nvtx_kernel_map, ensure_nvtx_kernel_map

# ── The sort-merge sweep (shared containment core) ──────────────────────────


def test_sweep_attributes_kernel_to_innermost_nvtx():
    """A kernel is credited to its innermost enclosing NVTX range, with the full
    outer>inner path and the nesting depth."""
    # nvtx rows sorted by (tid, start): train_step[0,100] ⊃ forward[10,50]
    nvtx = [(1, 0, 100, "train_step"), (1, 10, 50, "forward")]
    # kr rows: (tid, r_start, r_end, k_start, k_end, kernel_name)
    kr = [
        (1, 15, 20, 1000, 1005, "gemm"),  # inside forward (⊂ train_step)
        (1, 60, 65, 2000, 2010, "add"),  # inside train_step only (forward closed)
    ]
    out = {r["kernel_name"]: r for r in _sweep_nvtx_kernel_map(kr, nvtx)}

    assert out["gemm"]["nvtx_text"] == "forward"
    assert out["gemm"]["nvtx_depth"] == 1
    assert out["gemm"]["nvtx_path"] == "train_step > forward"
    assert out["gemm"]["k_dur_ns"] == 5

    assert out["add"]["nvtx_text"] == "train_step"
    assert out["add"]["nvtx_depth"] == 0
    assert out["add"]["nvtx_path"] == "train_step"
    assert out["add"]["k_dur_ns"] == 10


def test_sweep_drops_kernels_with_no_enclosing_range():
    nvtx = [(1, 0, 10, "phase")]
    kr = [(1, 50, 60, 100, 110, "orphan")]  # runtime after the range closed
    assert _sweep_nvtx_kernel_map(kr, nvtx) == []


def test_sweep_isolates_by_thread():
    """A range on one thread never encloses a kernel launched on another."""
    nvtx = [(1, 0, 100, "t1_range")]
    kr = [(2, 10, 20, 500, 505, "other_thread_kernel")]
    assert _sweep_nvtx_kernel_map(kr, nvtx) == []


# ── ensure_nvtx_kernel_map materialisation ──────────────────────────────────


def _duckdb_profile():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE StringIds(id BIGINT, value VARCHAR)")
    con.execute(
        'CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start BIGINT, "end" BIGINT, deviceId INT, '
        'streamId INT, correlationId BIGINT, shortName BIGINT, demangledName BIGINT)'
    )
    con.execute(
        'CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(globalTid BIGINT, correlationId BIGINT, '
        'start BIGINT, "end" BIGINT)'
    )
    con.execute(
        'CREATE TABLE NVTX_EVENTS(globalTid BIGINT, start BIGINT, "end" BIGINT, text VARCHAR, '
        "eventType INT, textId BIGINT)"
    )
    # gemm (short id 1 / demangled id 2), add (3/4)
    con.execute(
        "INSERT INTO StringIds VALUES (1,'gemm'),(2,'void gemm<float>'),(3,'add'),(4,'void add<float>')"
    )
    # kernels + their runtime launches on tid 1
    con.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
        "(1000,1005,0,7,101,1,2), (2000,2010,0,7,102,3,4)"
    )
    con.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1,101,15,20), (1,102,60,65)"
    )
    # NVTX PushPop ranges (eventType 59) on tid 1
    con.execute(
        "INSERT INTO NVTX_EVENTS VALUES (1,0,100,'train_step',59,NULL), (1,10,50,'forward',59,NULL)"
    )
    return con


def test_ensure_builds_map_with_demangled_names():
    con = _duckdb_profile()
    assert ensure_nvtx_kernel_map(con) is True

    rows = con.execute(
        "SELECT nvtx_text, nvtx_depth, kernel_name, k_dur_ns FROM nvtx_kernel_map "
        "ORDER BY k_start"
    ).fetchall()
    assert rows == [
        ("forward", 1, "void gemm<float>", 5),  # demangled name, not shortName
        ("train_step", 0, "void add<float>", 10),
    ]
    # path_dict populated and joinable
    paths = dict(
        con.execute(
            "SELECT d.nvtx_path, m.nvtx_text FROM nvtx_kernel_map m "
            "JOIN nvtx_path_dict d ON m.path_id = d.path_id"
        ).fetchall()
    )
    assert paths == {"train_step > forward": "forward", "train_step": "train_step"}


def test_ensure_is_noop_when_map_present():
    con = _duckdb_profile()
    assert ensure_nvtx_kernel_map(con) is True
    before = con.execute("SELECT COUNT(*) FROM nvtx_kernel_map").fetchone()[0]
    # Second call must see the existing table and not rebuild/duplicate.
    assert ensure_nvtx_kernel_map(con) is True
    after = con.execute("SELECT COUNT(*) FROM nvtx_kernel_map").fetchone()[0]
    assert before == after == 2


def test_ensure_returns_false_for_sqlite_connection():
    # Non-DuckDB connection: unchanged, caller keeps its own path.
    assert ensure_nvtx_kernel_map(sqlite3.connect(":memory:")) is False
