"""Tests for communicator-aware NCCL analysis."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from nsys_ai.nccl_communicator import analyze_nccl_communicators
from nsys_ai.profile import Profile
from nsys_ai.skills.registry import get_skill

COMM_SCHEMA = 1001
ALLREDUCE_SCHEMA = 1002
ALLGATHER_SCHEMA = 1003
SENDRECV_SCHEMA = 1004
REDUCTION_ENUM = 9001


def _pack_u32(value: int) -> bytes:
    return int(value).to_bytes(4, "little", signed=False)


def _pack_u64(value: int) -> bytes:
    return int(value).to_bytes(8, "little", signed=False)


def _make_blob(domain_id: int, schema_id: int, payload: bytes) -> bytes:
    header = (
        _pack_u64(domain_id)
        + _pack_u64(schema_id)
        + _pack_u64(len(payload))
        + _pack_u64(32)
    )
    return header + payload


def _payload_comm_init(comm_id: int, num_ranks: int, rank: int, cuda_device: int) -> bytes:
    return _pack_u64(comm_id) + _pack_u32(num_ranks) + _pack_u32(rank) + _pack_u32(cuda_device) + b"\x00" * 4


def _payload_allreduce(comm_id: int, message_size: int, reduction_value: int = 0) -> bytes:
    return _pack_u64(comm_id) + _pack_u64(message_size) + _pack_u32(reduction_value) + b"\x00" * 4


def _payload_allgather(comm_id: int, message_size: int) -> bytes:
    return _pack_u64(comm_id) + _pack_u64(message_size)


def _payload_sendrecv_no_size(comm_id: int) -> bytes:
    return _pack_u64(comm_id)


def _create_blob_profile(db_path: Path, *, binary_decl: str = "BLOB") -> str:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.executescript(
        f"""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE TARGET_INFO_GPU (
            id INTEGER PRIMARY KEY,
            name TEXT,
            busLocation TEXT DEFAULT '',
            totalMemory INTEGER DEFAULT 0,
            smCount INTEGER DEFAULT 0,
            chipName TEXT DEFAULT '',
            memoryBandwidth INTEGER DEFAULT 0
        );
        CREATE TABLE TARGET_INFO_CUDA_DEVICE (
            gpuId INTEGER,
            cudaId INTEGER,
            pid INTEGER DEFAULT 0,
            uuid TEXT DEFAULT '',
            numMultiprocessors INTEGER DEFAULT 0
        );
        CREATE TABLE TARGET_INFO_NIC_INFO (
            GUID INTEGER,
            stateName TEXT,
            nicId INTEGER,
            name TEXT,
            deviceId INTEGER,
            vendorId INTEGER,
            linkLayer INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            globalPid INTEGER DEFAULT 0,
            deviceId INTEGER DEFAULT 0,
            streamId INTEGER DEFAULT 0,
            correlationId INTEGER DEFAULT 0,
            start INTEGER NOT NULL,
            end INTEGER NOT NULL,
            shortName INTEGER NOT NULL,
            demangledName INTEGER DEFAULT 0,
            gridX INTEGER DEFAULT 1,
            gridY INTEGER DEFAULT 1,
            gridZ INTEGER DEFAULT 1,
            blockX INTEGER DEFAULT 1,
            blockY INTEGER DEFAULT 1,
            blockZ INTEGER DEFAULT 1
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            globalTid INTEGER DEFAULT 0,
            correlationId INTEGER DEFAULT 0,
            start INTEGER NOT NULL,
            end INTEGER NOT NULL,
            nameId INTEGER DEFAULT 0
        );
        CREATE TABLE NVTX_EVENTS (
            start INTEGER NOT NULL,
            end INTEGER,
            eventType INTEGER NOT NULL,
            rangeId INTEGER,
            category INTEGER,
            color INTEGER,
            text TEXT,
            globalTid INTEGER,
            endGlobalTid INTEGER,
            textId INTEGER,
            domainId INTEGER,
            uint64Value INTEGER,
            int64Value INTEGER,
            doubleValue REAL,
            uint32Value INTEGER,
            int32Value INTEGER,
            floatValue REAL,
            jsonTextId INTEGER,
            jsonText TEXT,
            binaryData {binary_decl}
        );
        CREATE TABLE NVTX_PAYLOAD_SCHEMAS (
            domainId INTEGER,
            schemaId INTEGER,
            name TEXT,
            type INTEGER,
            flags INTEGER,
            numEntries INTEGER,
            payloadSize INTEGER,
            alignTo INTEGER
        );
        CREATE TABLE NVTX_PAYLOAD_SCHEMA_ENTRIES (
            domainId INTEGER NOT NULL,
            schemaId INTEGER NOT NULL,
            idx INTEGER NOT NULL,
            flags INTEGER,
            type INTEGER,
            name TEXT,
            description TEXT,
            arrayOrUnionDetail INTEGER,
            offset INTEGER
        );
        CREATE TABLE NVTX_PAYLOAD_ENUM_ENTRIES (
            domainId INTEGER NOT NULL,
            schemaId INTEGER NOT NULL,
            idx INTEGER NOT NULL,
            name TEXT,
            value INTEGER,
            isFlag INTEGER
        );
        """
    )

    cur.executemany(
        "INSERT INTO StringIds VALUES (?, ?)",
        [
            (1, "ncclDevKernel_AllReduce"),
            (2, "nccl_preprocess_helper"),
            (3, "ncclKernel_AllGather"),
            (4, "ncclSendKernel"),
            (100, "ncclCommInitRankConfig"),
            (101, "ncclAllReduce"),
            (102, "ncclAllGather"),
            (103, "ncclSendRecv"),
            (200, "cudaLaunchKernel"),
        ],
    )
    cur.executemany(
        "INSERT INTO TARGET_INFO_GPU VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (0, "NVIDIA H100 SXM", "0000:00:00.0", 0, 132, "GH100", 0),
            (1, "NVIDIA H100 SXM", "0000:00:00.1", 0, 132, "GH100", 0),
        ],
    )
    cur.executemany(
        "INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (?, ?, ?, ?, ?)",
        [(0, 0, 10, "", 132), (1, 1, 10, "", 132)],
    )
    cur.executemany(
        "INSERT INTO TARGET_INFO_NIC_INFO VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(1, "Local", 0, "mlx5_0", 1, 5555, 2)],
    )

    cur.executemany(
        "INSERT INTO NVTX_PAYLOAD_SCHEMAS VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, COMM_SCHEMA, None, 1, None, 4, 24, None),
            (1, ALLREDUCE_SCHEMA, None, 1, None, 3, 24, None),
            (1, ALLGATHER_SCHEMA, None, 1, None, 2, 16, None),
            (1, SENDRECV_SCHEMA, None, 1, None, 1, 8, None),
        ],
    )
    cur.executemany(
        "INSERT INTO NVTX_PAYLOAD_SCHEMA_ENTRIES VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, COMM_SCHEMA, 0, None, 18, "NCCL communicator ID", None, None, None),
            (1, COMM_SCHEMA, 1, None, 5, "No. of ranks", None, None, 8),
            (1, COMM_SCHEMA, 2, None, 5, "Rank", None, None, 12),
            (1, COMM_SCHEMA, 3, None, 5, "CUDA device", None, None, 16),
            (1, ALLREDUCE_SCHEMA, 0, None, 18, "NCCL communicator ID", None, None, None),
            (1, ALLREDUCE_SCHEMA, 1, None, 22, "Message size [bytes]", None, None, 8),
            (1, ALLREDUCE_SCHEMA, 2, None, REDUCTION_ENUM, "Reduction operation", None, None, 16),
            (1, ALLGATHER_SCHEMA, 0, None, 18, "NCCL communicator ID", None, None, None),
            (1, ALLGATHER_SCHEMA, 1, None, 22, "Message size [bytes]", None, None, 8),
            (1, SENDRECV_SCHEMA, 0, None, 18, "NCCL communicator ID", None, None, None),
        ],
    )
    cur.executemany(
        "INSERT INTO NVTX_PAYLOAD_ENUM_ENTRIES VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, REDUCTION_ENUM, 0, "Sum", 0, None),
            (1, REDUCTION_ENUM, 1, "Product", 1, None),
        ],
    )

    cur.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1000, 2000, 59, None, None, None, None, 100, None, 100, 1, None, None, None, None, None, None, None, None, sqlite3.Binary(_make_blob(1, COMM_SCHEMA, _payload_comm_init(0xABC, 2, 0, 0)))),
            (1000, 2000, 59, None, None, None, None, 101, None, 100, 1, None, None, None, None, None, None, None, None, sqlite3.Binary(_make_blob(1, COMM_SCHEMA, _payload_comm_init(0xABC, 2, 1, 1)))),
            (2100, 2200, 59, None, None, None, None, 100, None, 100, 1, None, None, None, None, None, None, None, None, sqlite3.Binary(_make_blob(1, COMM_SCHEMA, _payload_comm_init(0xDEF, 2, 0, 0)))),
            (3000, 8000, 59, None, None, None, None, 100, None, 101, 1, None, None, None, None, None, None, None, None, sqlite3.Binary(_make_blob(1, ALLREDUCE_SCHEMA, _payload_allreduce(0xABC, 4096)))),
            (3000, 8000, 59, None, None, None, None, 101, None, 101, 1, None, None, None, None, None, None, None, None, sqlite3.Binary(_make_blob(1, ALLREDUCE_SCHEMA, _payload_allreduce(0xABC, 4096)))),
            (9000, 13000, 59, None, None, None, None, 100, None, 103, 1, None, None, None, None, None, None, None, None, sqlite3.Binary(_make_blob(1, SENDRECV_SCHEMA, _payload_sendrecv_no_size(0xDEF)))),
            (14000, 18000, 59, None, None, None, None, 100, None, 102, 1, None, None, None, None, None, None, None, None, sqlite3.Binary(_make_blob(1, ALLGATHER_SCHEMA, _payload_allgather(0xABC, 8192)))),
        ],
    )

    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
        [
            (100, 10, 3500, 3600, 200),
            (100, 11, 3600, 3700, 200),
            (101, 20, 3500, 3600, 200),
            (100, 30, 9500, 9600, 200),
            (100, 40, 14500, 14600, 200),
        ],
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (10, 0, 7, 10, 4000, 5000, 1, 1, 1, 1, 1, 1, 1, 1),
            (10, 0, 7, 11, 4200, 4700, 2, 2, 1, 1, 1, 1, 1, 1),
            (10, 1, 7, 20, 4300, 5300, 1, 1, 1, 1, 1, 1, 1, 1),
            (10, 0, 8, 30, 10000, 11000, 4, 4, 1, 1, 1, 1, 1, 1),
            (10, 0, 9, 40, 15000, 16000, 3, 3, 1, 1, 1, 1, 1, 1),
        ],
    )
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture
def nccl_blob_db_path(tmp_path):
    return _create_blob_profile(tmp_path / "nccl_blob_profile.sqlite")


@pytest.fixture
def nccl_mixed_blob_db_path(tmp_path):
    return _create_blob_profile(tmp_path / "nccl_mixed_blob_profile.sqlite", binary_decl="TEXT")


@pytest.fixture
def nccl_blob_conn(nccl_blob_db_path):
    conn = sqlite3.connect(nccl_blob_db_path)
    yield conn
    conn.close()


def _rows_without_diagnostics(rows: list[dict]) -> list[dict]:
    return [row for row in rows if not row.get("_diagnostic")]


def test_sqlite_analysis_groups_by_communicator(nccl_blob_conn):
    prof = Profile._from_conn(nccl_blob_conn)
    rows = _rows_without_diagnostics(analyze_nccl_communicators(prof))
    assert rows

    allreduce = next(r for r in rows if r["collective_type"] == "allreduce")
    assert allreduce["communicator_hex"] == "0x0000000000000abc"
    assert allreduce["count"] == 2
    assert allreduce["num_ranks"] == 2
    assert allreduce["inferred_dimension"] == "data_parallel_or_global"
    assert allreduce["total_bytes"] == 8192
    assert allreduce["bandwidth_gbps"] is not None
    assert allreduce["reduction_op"] == "Sum"


def test_device_filter_excludes_other_devices(nccl_blob_conn):
    prof = Profile._from_conn(nccl_blob_conn)
    rows = _rows_without_diagnostics(analyze_nccl_communicators(prof, device=0))
    allreduce = next(r for r in rows if r["collective_type"] == "allreduce")
    assert allreduce["count"] == 1
    assert allreduce["devices"] == "0"


def test_multi_kernel_tiebreaker_ignores_helper_kernel(nccl_blob_conn):
    prof = Profile._from_conn(nccl_blob_conn)
    rows = _rows_without_diagnostics(analyze_nccl_communicators(prof, device=0))
    allreduce = next(r for r in rows if r["collective_type"] == "allreduce")
    assert allreduce["total_ms"] == pytest.approx(0.001, abs=1e-6)


def test_missing_message_size_omits_bandwidth(nccl_blob_conn):
    prof = Profile._from_conn(nccl_blob_conn)
    rows = _rows_without_diagnostics(analyze_nccl_communicators(prof, device=0))
    sendrecv = next(r for r in rows if r["collective_type"] == "sendrecv")
    assert sendrecv["total_bytes"] is None
    assert sendrecv["bandwidth_gbps"] is None


def test_skill_formats_grouped_output(nccl_blob_conn):
    skill = get_skill("nccl_communicator_analysis")
    assert skill is not None
    text = skill.run(nccl_blob_conn)
    assert "NCCL Communication by Communicator" in text
    assert "0x0000000000000abc" in text
    assert "allreduce" in text


def test_direct_and_cached_paths_match(nccl_blob_db_path):
    skill = get_skill("nccl_communicator_analysis")
    assert skill is not None

    direct = Profile(nccl_blob_db_path, cache_mode="direct")
    cached = Profile(nccl_blob_db_path, cache_mode="parquet")
    try:
        direct_rows = _rows_without_diagnostics(skill.execute(direct.db))
        cached_rows = _rows_without_diagnostics(skill.execute(cached.db))
    finally:
        direct.close()
        cached.close()

    def _key(rows: list[dict]) -> list[tuple]:
        return sorted(
            (
                row["communicator_hex"],
                row["collective_type"],
                row["count"],
                row["total_bytes"],
                row["devices"],
            )
            for row in rows
        )

    assert _key(direct_rows) == _key(cached_rows)


def test_direct_mode_uses_sqlite_fallback_for_mixed_text_blob_column(nccl_mixed_blob_db_path):
    skill = get_skill("nccl_communicator_analysis")
    assert skill is not None

    direct = Profile(nccl_mixed_blob_db_path, cache_mode="direct")
    try:
        rows = skill.execute(direct.db)
    finally:
        direct.close()

    data_rows = _rows_without_diagnostics(rows)
    diagnostics = [row for row in rows if row.get("_diagnostic")]
    assert data_rows
    assert any("SQLite side-connection fallback" in detail for row in diagnostics for detail in row.get("details", []))


def test_profile_health_manifest_includes_communicator_summary(nccl_blob_conn):
    skill = get_skill("profile_health_manifest")
    assert skill is not None

    rows = skill.execute(nccl_blob_conn, device=0)
    manifest = rows[0]

    assert manifest["communicators"]["communicators"] >= 1
    assert manifest["communicators"]["dominant_collective"] in {"allreduce", "allgather", "sendrecv"}


def test_root_cause_matcher_uses_communicator_evidence(nccl_blob_conn):
    skill = get_skill("root_cause_matcher")
    assert skill is not None

    rows = skill.execute(nccl_blob_conn, device=0)
    patterns = {row["pattern"] for row in rows}
    assert "Inefficient NCCL Communicator" in patterns
