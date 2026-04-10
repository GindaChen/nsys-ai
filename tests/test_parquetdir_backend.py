from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nsys_ai.profile import Profile
from nsys_ai.skills.registry import get_skill

COMM_SCHEMA = 1001
ALLREDUCE_SCHEMA = 1002
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
    return (
        _pack_u64(comm_id)
        + _pack_u32(num_ranks)
        + _pack_u32(rank)
        + _pack_u32(cuda_device)
        + b"\x00" * 4
    )


def _payload_allreduce(comm_id: int, message_size: int, reduction_value: int = 0) -> bytes:
    return _pack_u64(comm_id) + _pack_u64(message_size) + _pack_u32(reduction_value) + b"\x00" * 4


def _invalid_large_string(values: list[bytes | None]) -> pa.Array:
    binary = pa.array(values, type=pa.large_binary())
    null_bitmap, offsets, data = binary.buffers()
    return pa.LargeStringArray.from_buffers(
        len(binary),
        offsets,
        data,
        null_bitmap=null_bitmap,
        null_count=binary.null_count,
    )


def _write_table(parquet_dir: Path, name: str, table: pa.Table) -> None:
    pq.write_table(table, parquet_dir / f"{name}.parquet")


def _create_parquetdir_profile(parquet_dir: Path) -> str:
    parquet_dir.mkdir()
    schema_entries = [
        (1, COMM_SCHEMA, 0, None, 18, "NCCL communicator ID", None, None, 0),
        (1, COMM_SCHEMA, 1, None, 5, "No. of ranks", None, None, 8),
        (1, COMM_SCHEMA, 2, None, 5, "Rank", None, None, 12),
        (1, COMM_SCHEMA, 3, None, 5, "CUDA device", None, None, 16),
        (1, ALLREDUCE_SCHEMA, 0, None, 18, "NCCL communicator ID", None, None, 0),
        (1, ALLREDUCE_SCHEMA, 1, None, 22, "Message size [bytes]", None, None, 8),
        (1, ALLREDUCE_SCHEMA, 2, None, REDUCTION_ENUM, "Reduction operation", None, None, 16),
    ]
    _write_table(
        parquet_dir,
        "StringIds",
        pa.table(
            {
                "id": pa.array([1, 100, 101, 200], type=pa.int32()),
                "value": pa.array(
                    [
                        "ncclDevKernel_AllReduce",
                        "ncclCommInitRankConfig",
                        "ncclAllReduce",
                        "cudaLaunchKernel",
                    ]
                ),
            }
        ),
    )
    _write_table(
        parquet_dir,
        "TARGET_INFO_GPU",
        pa.table(
            {
                "id": pa.array([0, 1], type=pa.int64()),
                "name": pa.array(["NVIDIA H100 SXM", "NVIDIA H100 SXM"]),
                "busLocation": pa.array(["0000:00:00.0", "0000:00:00.1"]),
                "totalMemory": pa.array([0, 0], type=pa.int64()),
                "smCount": pa.array([132, 132], type=pa.int32()),
                "chipName": pa.array(["GH100", "GH100"]),
                "memoryBandwidth": pa.array([0, 0], type=pa.int64()),
            }
        ),
    )
    _write_table(
        parquet_dir,
        "TARGET_INFO_CUDA_DEVICE",
        pa.table(
            {
                "gpuId": pa.array([0, 1], type=pa.int64()),
                "cudaId": pa.array([0, 1], type=pa.int64()),
                "pid": pa.array([10, 10], type=pa.int64()),
                "uuid": pa.array(["", ""]),
                "numMultiprocessors": pa.array([132, 132], type=pa.int64()),
            }
        ),
    )
    _write_table(
        parquet_dir,
        "TARGET_INFO_NIC_INFO",
        pa.table(
            {
                "GUID": pa.array([], type=pa.int64()),
                "stateName": pa.array([], type=pa.string()),
                "nicId": pa.array([], type=pa.int64()),
                "name": pa.array([], type=pa.string()),
                "deviceId": pa.array([], type=pa.int64()),
                "vendorId": pa.array([], type=pa.int64()),
                "linkLayer": pa.array([], type=pa.int64()),
            }
        ),
    )
    _write_table(
        parquet_dir,
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        pa.table(
            {
                "globalTid": pa.array([100, 101], type=pa.int64()),
                "correlationId": pa.array([10, 20], type=pa.int64()),
                "start": pa.array([3500, 3500], type=pa.int64()),
                "end": pa.array([3600, 3600], type=pa.int64()),
                "nameId": pa.array([200, 200], type=pa.int64()),
            }
        ),
    )
    _write_table(
        parquet_dir,
        "CUPTI_ACTIVITY_KIND_KERNEL",
        pa.table(
            {
                "globalPid": pa.array([10, 10], type=pa.int64()),
                "deviceId": pa.array([0, 1], type=pa.int64()),
                "streamId": pa.array([7, 7], type=pa.int64()),
                "correlationId": pa.array([10, 20], type=pa.int64()),
                "start": pa.array([4000, 4300], type=pa.int64()),
                "end": pa.array([5000, 5300], type=pa.int64()),
                "shortName": pa.array([1, 1], type=pa.int64()),
                "demangledName": pa.array([1, 1], type=pa.int64()),
                "gridX": pa.array([1, 1], type=pa.int64()),
                "gridY": pa.array([1, 1], type=pa.int64()),
                "gridZ": pa.array([1, 1], type=pa.int64()),
                "blockX": pa.array([1, 1], type=pa.int64()),
                "blockY": pa.array([1, 1], type=pa.int64()),
                "blockZ": pa.array([1, 1], type=pa.int64()),
            }
        ),
    )
    _write_table(
        parquet_dir,
        "NVTX_PAYLOAD_SCHEMAS",
        pa.table(
            {
                "domainId": pa.array([1, 1], type=pa.int64()),
                "schemaId": pa.array([COMM_SCHEMA, ALLREDUCE_SCHEMA], type=pa.int64()),
                "name": pa.array([None, None], type=pa.string()),
                "type": pa.array([1, 1], type=pa.int64()),
                "flags": pa.array([None, None], type=pa.int64()),
                "numEntries": pa.array([4, 3], type=pa.int64()),
                "payloadSize": pa.array([24, 24], type=pa.int64()),
                "alignTo": pa.array([None, None], type=pa.int64()),
            }
        ),
    )
    _write_table(
        parquet_dir,
        "NVTX_PAYLOAD_SCHEMA_ENTRIES",
        pa.table(
            {
                "domainId": pa.array([row[0] for row in schema_entries], type=pa.int64()),
                "schemaId": pa.array([row[1] for row in schema_entries], type=pa.int64()),
                "idx": pa.array([row[2] for row in schema_entries], type=pa.int64()),
                "flags": pa.array([row[3] for row in schema_entries], type=pa.int64()),
                "type": pa.array([row[4] for row in schema_entries], type=pa.int64()),
                "name": pa.array([row[5] for row in schema_entries]),
                "description": pa.array([row[6] for row in schema_entries], type=pa.string()),
                "arrayOrUnionDetail": pa.array([row[7] for row in schema_entries], type=pa.int64()),
                "offset": pa.array([row[8] for row in schema_entries], type=pa.int64()),
            }
        ),
    )
    _write_table(
        parquet_dir,
        "NVTX_PAYLOAD_ENUM_ENTRIES",
        pa.table(
            {
                "domainId": pa.array([1, 1], type=pa.int64()),
                "schemaId": pa.array([REDUCTION_ENUM, REDUCTION_ENUM], type=pa.int64()),
                "idx": pa.array([0, 1], type=pa.int64()),
                "name": pa.array(["Sum", "Product"]),
                "value": pa.array([0, 1], type=pa.int64()),
                "isFlag": pa.array([None, None], type=pa.int64()),
            }
        ),
    )

    binary_values = [
        _make_blob(1, COMM_SCHEMA, _payload_comm_init(0xABC, 2, 0, 0)),
        _make_blob(1, COMM_SCHEMA, _payload_comm_init(0xABC, 2, 1, 1)),
        _make_blob(1, ALLREDUCE_SCHEMA, _payload_allreduce(0xABC, 4096)),
        _make_blob(1, ALLREDUCE_SCHEMA, _payload_allreduce(0xABC, 4096)),
    ]
    _write_table(
        parquet_dir,
        "NVTX_EVENTS",
        pa.Table.from_arrays(
            [
                pa.array([1000, 1000, 3000, 3000], type=pa.int64()),
                pa.array([2000, 2000, 8000, 8000], type=pa.int64()),
                pa.array([59, 59, 59, 59], type=pa.uint32()),
                pa.array([None, None, None, None], type=pa.uint64()),
                pa.array([None, None, None, None], type=pa.uint64()),
                pa.array([None, None, None, None], type=pa.uint32()),
                pa.array([None, None, None, None], type=pa.large_string()),
                pa.array([100, 101, 100, 101], type=pa.uint64()),
                pa.array([None, None, None, None], type=pa.uint64()),
                pa.array([100, 100, 101, 101], type=pa.uint32()),
                pa.array([1, 1, 1, 1], type=pa.uint64()),
                pa.array([None, None, None, None], type=pa.uint64()),
                pa.array([None, None, None, None], type=pa.int64()),
                pa.array([None, None, None, None], type=pa.float64()),
                pa.array([None, None, None, None], type=pa.uint32()),
                pa.array([None, None, None, None], type=pa.int32()),
                pa.array([None, None, None, None], type=pa.float32()),
                pa.array([None, None, None, None], type=pa.uint32()),
                pa.array([None, None, None, None], type=pa.large_string()),
                _invalid_large_string(binary_values),
            ],
            schema=pa.schema(
                [
                    ("start", pa.int64()),
                    ("end", pa.int64()),
                    ("eventType", pa.uint32()),
                    ("rangeId", pa.uint64()),
                    ("category", pa.uint64()),
                    ("color", pa.uint32()),
                    ("text", pa.large_string()),
                    ("globalTid", pa.uint64()),
                    ("endGlobalTid", pa.uint64()),
                    ("textId", pa.uint32()),
                    ("domainId", pa.uint64()),
                    ("uint64Value", pa.uint64()),
                    ("int64Value", pa.int64()),
                    ("doubleValue", pa.float64()),
                    ("uint32Value", pa.uint32()),
                    ("int32Value", pa.int32()),
                    ("floatValue", pa.float32()),
                    ("jsonTextId", pa.uint32()),
                    ("jsonText", pa.large_string()),
                    ("binaryData", pa.large_string()),
                ]
            ),
        ),
    )
    return str(parquet_dir)


def _rows_without_diagnostics(rows: list[dict]) -> list[dict]:
    return [row for row in rows if not row.get("_diagnostic")]


def test_profile_parquetdir_backend_opens_and_aliases(tmp_path):
    parquetdir = _create_parquetdir_profile(tmp_path / "synthetic.parquetdir")
    with Profile(parquetdir, backend="parquetdir") as prof:
        assert prof.meta.devices == [0, 1]
        assert prof.db.execute("SELECT COUNT(*) FROM NVTX_EVENTS WHERE binaryData IS NOT NULL").fetchone()[0] == 4
        assert prof.db.execute("SELECT COUNT(*) FROM gpu_info").fetchone()[0] == 2
        assert prof.db.execute("SELECT COUNT(*) FROM cuda_device").fetchone()[0] == 2


def test_schema_inspect_runs_on_parquetdir_backend(tmp_path):
    parquetdir = _create_parquetdir_profile(tmp_path / "synthetic.parquetdir")
    skill = get_skill("schema_inspect")
    assert skill is not None
    with Profile(parquetdir, backend="parquetdir") as prof:
        rows = skill.execute(prof.db)
    table_names = {row["table_name"] for row in rows}
    assert "NVTX_EVENTS" in table_names
    assert "NVTX_PAYLOAD_SCHEMAS" in table_names
    assert "string_ids" in table_names


def test_nccl_communicator_skill_runs_on_parquetdir_without_sqlite_fallback(tmp_path):
    parquetdir = _create_parquetdir_profile(tmp_path / "synthetic.parquetdir")
    skill = get_skill("nccl_communicator_analysis")
    assert skill is not None
    with Profile(parquetdir, backend="parquetdir") as prof:
        rows = skill.execute(prof.db)
    data_rows = _rows_without_diagnostics(rows)
    assert data_rows
    assert data_rows[0]["communicator_hex"] == "0x0000000000000abc"
    diagnostic_details = [
        detail for row in rows if row.get("_diagnostic") for detail in row.get("details", [])
    ]
    assert not any("SQLite side-connection fallback" in detail for detail in diagnostic_details)
