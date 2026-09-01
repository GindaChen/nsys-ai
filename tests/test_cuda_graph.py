"""CUDA Graph attribution contracts (#236)."""

import sqlite3

from nsys_ai.parquet_cache import build_cache, ensure_nvtx_kernel_map, open_cached_db
from nsys_ai.profile import NsightSchema, Profile
from nsys_ai.skills.builtins.kernel_launch_overhead import SKILL as OVERHEAD
from nsys_ai.skills.builtins.kernel_launch_pattern import SKILL as PATTERN


def _graph_connection(*, graph_kernel_columns=True):
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO StringIds VALUES
            (1, 'tiny_kernel'), (2, 'void tiny_kernel()'),
            (10, 'cudaLaunchKernel');
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
            globalPid INTEGER, deviceId INTEGER, streamId INTEGER,
            correlationId INTEGER, start INTEGER, "end" INTEGER,
            shortName INTEGER, demangledName INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
            globalTid INTEGER, correlationId INTEGER, start INTEGER,
            "end" INTEGER, nameId INTEGER
        );
        CREATE TABLE NVTX_EVENTS(
            globalTid INTEGER, start INTEGER, "end" INTEGER, text TEXT,
            eventType INTEGER, rangeId INTEGER, textId INTEGER
        );
        CREATE TABLE CUDA_GRAPH_EVENTS(
            start INTEGER, "end" INTEGER, globalTid INTEGER, nameId INTEGER,
            graphId INTEGER, originalGraphId INTEGER, graphExecId INTEGER
        );
        CREATE TABLE CUDA_GRAPH_NODE_EVENTS(
            start INTEGER, "end" INTEGER, eventClass INTEGER, globalTid INTEGER,
            nameId INTEGER, graphNodeId INTEGER, originalGraphNodeId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(
            start INTEGER, "end" INTEGER, deviceId INTEGER, contextId INTEGER,
            streamId INTEGER, correlationId INTEGER, globalPid INTEGER,
            graphId INTEGER, graphExecId INTEGER
        );
        """
    )
    if graph_kernel_columns:
        conn.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN graphNodeId INTEGER")
        conn.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN graphId INTEGER")
    return conn


def _insert_graph_replay(conn):
    rows = [
        (0x100000000, 0, 7, 7, 10_000, 10_005, 1, 2, 101, 42),
        (0x100000000, 0, 7, 7, 10_020, 10_025, 1, 2, 102, 42),
    ]
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
        "(globalPid, deviceId, streamId, correlationId, start, end, shortName, demangledName, graphNodeId, graphId) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
        (0x100000007, 7, 8_000, 9_000, 10),
    )
    conn.execute(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?, ?)",
        (0x100000007, 0, 20_000, "decode", 59, 1, None),
    )
    conn.commit()


def test_schema_detects_graph_tables_and_optional_kernel_columns():
    conn = _graph_connection()
    schema = NsightSchema(conn)

    assert schema.kernel_graph_columns == {
        "graph_node_id": "graphNodeId",
        "graph_id": "graphId",
    }
    assert schema.cuda_graph["kernel_attribution"] is True
    assert set(schema.cuda_graph["tables"]) == {
        "CUDA_GRAPH_EVENTS",
        "CUDA_GRAPH_NODE_EVENTS",
        "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
    }
    assert schema.missing_required_columns() == []


def test_graph_metadata_without_kernel_ids_is_a_clean_partial_capability():
    conn = _graph_connection(graph_kernel_columns=False)
    schema = NsightSchema(conn)

    assert schema.cuda_graph["available"] is True
    assert schema.cuda_graph["kernel_attribution"] is False
    # Optional graph support must never make an otherwise valid old-style
    # kernel export fail the hard schema contract.
    assert schema.missing_required_columns() == []


def test_profile_kernel_accessors_preserve_graph_attribution():
    conn = _graph_connection()
    _insert_graph_replay(conn)
    profile = Profile._from_conn(conn)

    rows = profile.kernels(0)
    assert [(row["graph_node_id"], row["graph_id"]) for row in rows] == [(101, 42), (102, 42)]
    mapped = profile.kernel_map(0)
    assert mapped[7]["graph_node_id"] == 102
    assert mapped[7]["graph_id"] == 42


def test_launch_overhead_charges_one_runtime_call_for_many_graph_nodes():
    conn = _graph_connection()
    _insert_graph_replay(conn)

    rows = OVERHEAD.execute_fn(conn, min_launches=1, device=0)

    assert len(rows) == 1
    assert rows[0]["launch_count"] == 2
    assert rows[0]["total_api_ms"] == 0.001
    assert rows[0]["avg_api_us"] == 1.0
    assert rows[0]["graph_node_count"] == 2
    assert rows[0]["graph_id_count"] == 1


def test_launch_pattern_reports_graph_fanout():
    conn = _graph_connection()
    _insert_graph_replay(conn)

    rows = PATTERN.execute(conn, limit=10, _skip_device_validation=True)

    assert len(rows) == 1
    assert rows[0]["kernel_count"] == 2
    assert rows[0]["graph_node_count"] == 2
    assert rows[0]["graph_id_count"] == 1


def test_cache_keeps_graph_tables_and_normalized_kernel_ids(tmp_path):
    source = tmp_path / "graph.sqlite"
    conn = _graph_connection()
    _insert_graph_replay(conn)
    disk = sqlite3.connect(source)
    conn.backup(disk)
    disk.close()
    conn.close()

    cache_dir = build_cache(str(source))
    db = open_cached_db(str(source))
    try:
        assert db.execute("SELECT COUNT(*) FROM cuda_graph_events").fetchone()[0] == 0
        ids = db.execute(
            "SELECT graph_node_id, graph_id FROM kernels ORDER BY start"
        ).fetchall()
        assert ensure_nvtx_kernel_map(db) is True
        mapped = db.execute(
            "SELECT graph_node_id, graph_id FROM nvtx_kernel_map ORDER BY k_start"
        ).fetchall()
    finally:
        db.close()

    assert cache_dir.joinpath("cuda_graph_events.parquet").is_file()
    assert cache_dir.joinpath("cuda_graph_node_events.parquet").is_file()
    assert cache_dir.joinpath("graph_trace.parquet").is_file()
    assert ids == [(101, 42), (102, 42)]
    assert mapped == [(101, 42), (102, 42)]


def test_sweep_carries_graph_ids_only_for_kernels_that_have_them():
    """The graph ids are sparse in the row dict, and that is load-bearing.

    ``_sweep_nvtx_kernel_map`` builds one dict per attributed kernel and holds
    every one of them for the length of the build, so the dict's key count is
    multiplied by the row count. CPython sizes a dict by its key count, and 11
    keys crosses the boundary that 9 keys sits under: 272 bytes becomes 464, a
    192-byte tax on every row for two fields that are null on every kernel that
    was not launched from a graph -- which, on a capture that uses no graphs, is
    all of them. That is what took the skills window from 6.63 MB to 7.54 MB in
    CI and broke ``test_running_skills_stays_under_the_python_heap_ceiling``.

    Absent therefore means "not a graph node", and the one consumer reads the
    fields with ``.get()``. Writing them unconditionally would restore the
    regression without failing anything else, so it is pinned here.
    """
    from nsys_ai.parquet_cache import _sweep_nvtx_kernel_map

    nvtx_rows = [(1, 0, 1000, "step")]
    kr_rows = [
        # (globalTid, r_start, r_end, k_start, k_end, name, node, graph, elig, used)
        (1, 10, 20, 30, 40, "plain_kernel", None, None, 0, 0),
        (1, 50, 60, 70, 80, "graph_kernel", 101, 42, 0, 0),
    ]

    rows = _sweep_nvtx_kernel_map(kr_rows, nvtx_rows)
    by_name = {r["kernel_name"]: r for r in rows}

    assert "graph_node_id" not in by_name["plain_kernel"]
    assert "graph_id" not in by_name["plain_kernel"]
    assert by_name["plain_kernel"].get("graph_node_id") is None

    assert by_name["graph_kernel"]["graph_node_id"] == 101
    assert by_name["graph_kernel"]["graph_id"] == 42


def test_the_arrow_writer_reads_the_sparse_graph_fields():
    """The consumer's contract: a missing key is a null column, not a KeyError."""
    from nsys_ai.parquet_cache import _nvtx_map_arrow_tables

    results = [
        {
            "nvtx_text": "step",
            "nvtx_depth": 0,
            "nvtx_path": "step",
            "kernel_name": "plain_kernel",
            "k_start": 30,
            "k_end": 40,
            "k_dur_ns": 10,
            "is_tc_eligible": 0,
            "uses_tc": 0,
        }
    ]

    map_tbl, _ = _nvtx_map_arrow_tables(results)

    assert map_tbl.column("graph_node_id").to_pylist() == [None]
    assert map_tbl.column("graph_id").to_pylist() == [None]
