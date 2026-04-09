"""Communicator-aware NCCL analysis from NVTX extended payload blobs.

The enriched Nsight SQLite export (`nsys export --include-blobs=true`) stores
per-event NVTX payloads in `NVTX_EVENTS.binaryData`. For NCCL events, the blob
starts with a fixed 32-byte header:

    [domainId:u64][schemaId:u64][payloadSize:u64][payloadOffset:u64]

followed by the payload bytes referenced by `NVTX_PAYLOAD_SCHEMAS` and
`NVTX_PAYLOAD_SCHEMA_ENTRIES`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
import sqlite3

from .connection import DB_ERRORS
from .profile import Profile, get_first_gpu_name

_NVTX_RANGE_EVENT = 59
_BLOB_HEADER_SIZE = 32
_NCCL_KERNEL_PATTERNS = (
    "nccldevkernel",
    "ncclkernel",
    "ncclsend",
    "ncclrecv",
    "ncclallreduce",
)
_COLLECTIVE_NAME_HINTS = {
    "allgather": "allgather",
    "reducescatter": "reducescatter",
    "allreduce": "allreduce",
    "broadcast": "broadcast",
    "sendrecv": "sendrecv",
    "reduce": "reduce",
}
_NVLINK_PEAK_GBPS = {
    "B200": 1800.0,
    "GB200": 1800.0,
    "H200": 900.0,
    "H100": 900.0,
    "H800": 900.0,
    "A100": 600.0,
    "V100": 300.0,
}


@dataclass
class _DecodedEvent:
    nvtx_text: str
    global_tid: int
    start_ns: int
    end_ns: int
    domain_id: int
    schema_id: int
    communicator_id: int | None
    message_size_bytes: int | None
    num_ranks: int | None
    rank: int | None
    cuda_device: int | None
    reduction_op: str | None
    root_rank: int | None
    collective_type: str | None


def analyze_nccl_communicators(
    prof: Profile,
    device: int | None = None,
    trim: tuple[int, int] | None = None,
) -> list[dict]:
    """Aggregate NCCL communication by communicator ID and collective type."""
    events, diagnostics = _load_nccl_payload_events(prof, trim=trim)
    if not events:
        return [
            {
                "_diagnostic": True,
                "message": diagnostics[0]
                if diagnostics
                else (
                    "No NCCL payload blobs found. Re-export the profile with "
                    "`nsys export --include-blobs=true`."
                ),
            }
        ]

    communicator_meta = _build_communicator_metadata(events)
    attributed, dropped = _attribute_events_to_kernels(prof, events, device=device, trim=trim)
    diagnostics.extend(dropped)

    if not attributed:
        return [
            {
                "_diagnostic": True,
                "message": (
                    "Decoded NCCL payload events were found, but no NCCL kernels could be "
                    "attributed to them on the selected device scope."
                ),
                "details": diagnostics,
            }
        ]

    peak_gbps, peak_source = _estimate_peak_bandwidth_gbps(prof)
    world_size = len(prof.meta.devices or [])

    grouped: dict[tuple[int, str], dict] = {}
    for row in attributed:
        comm_id = row["communicator_id"]
        collective_type = row["collective_type"]
        key = (comm_id, collective_type)
        meta = communicator_meta.get(comm_id, {})
        bucket = grouped.setdefault(
            key,
            {
                "communicator_id": comm_id,
                "communicator_hex": f"0x{comm_id:016x}",
                "collective_type": collective_type,
                "durations_ns": [],
                "total_bytes": 0,
                "sized_count": 0,
                "count": 0,
                "devices": set(),
                "streams": set(),
                "num_ranks": meta.get("num_ranks"),
                "rank": meta.get("rank"),
                "cuda_device": meta.get("cuda_device"),
                "reduction_op": row.get("reduction_op") or meta.get("reduction_op"),
                "root_rank": row.get("root_rank") or meta.get("root_rank"),
            },
        )
        bucket["durations_ns"].append(row["duration_ns"])
        bucket["count"] += 1
        bucket["devices"].add(row["device_id"])
        bucket["streams"].add(row["stream_id"])
        if row.get("message_size_bytes") is not None:
            bucket["sized_count"] += 1
            bucket["total_bytes"] += int(row["message_size_bytes"])
        for key_name in ("num_ranks", "rank", "cuda_device", "reduction_op", "root_rank"):
            if bucket.get(key_name) is None and row.get(key_name) is not None:
                bucket[key_name] = row[key_name]

    results = []
    for bucket in grouped.values():
        durations = bucket.pop("durations_ns")
        total_ns = sum(durations)
        avg_ns = total_ns / len(durations)
        bandwidth_gbps = None
        total_bytes = bucket["total_bytes"]
        if bucket["sized_count"] == bucket["count"] and total_bytes > 0 and total_ns > 0:
            bandwidth_gbps = total_bytes / (total_ns / 1e9) / 1e9
        efficiency_pct = None
        if bandwidth_gbps is not None and peak_gbps:
            efficiency_pct = (bandwidth_gbps / peak_gbps) * 100.0
        num_ranks = bucket.get("num_ranks")
        devices = sorted(bucket["devices"])
        device_scope = "all" if device is None else str(device)
        results.append(
            {
                "communicator_id": bucket["communicator_id"],
                "communicator_hex": bucket["communicator_hex"],
                "collective_type": bucket["collective_type"],
                "count": bucket["count"],
                "total_ms": round(total_ns / 1e6, 3),
                "avg_ms": round(avg_ns / 1e6, 3),
                "min_ms": round(min(durations) / 1e6, 3),
                "max_ms": round(max(durations) / 1e6, 3),
                "total_bytes": total_bytes if bucket["sized_count"] == bucket["count"] else None,
                "avg_bytes": round(total_bytes / bucket["count"], 1)
                if bucket["sized_count"] == bucket["count"] and bucket["count"] > 0
                else None,
                "bandwidth_gbps": round(bandwidth_gbps, 3) if bandwidth_gbps is not None else None,
                "peak_gbps": round(peak_gbps, 3) if peak_gbps else None,
                "peak_source": peak_source,
                "efficiency_pct": round(efficiency_pct, 1) if efficiency_pct is not None else None,
                "num_ranks": num_ranks,
                "rank": bucket.get("rank"),
                "cuda_device": bucket.get("cuda_device"),
                "root_rank": bucket.get("root_rank"),
                "reduction_op": bucket.get("reduction_op"),
                "inferred_dimension": _infer_parallel_dimension(num_ranks, world_size),
                "device_scope": device_scope,
                "device_count": len(devices),
                "devices": ",".join(str(d) for d in devices),
                "stream_count": len(bucket["streams"]),
            }
        )

    results.sort(key=lambda r: (-r["total_ms"], r["communicator_hex"], r["collective_type"]))
    if diagnostics:
        results.append(
            {
                "_diagnostic": True,
                "message": "Additional communicator-analysis diagnostics",
                "details": diagnostics,
            }
        )
    return results


def format_nccl_communicator(rows: list[dict]) -> str:
    """Render communicator-aware NCCL rows as grouped text."""
    if not rows:
        return "No communicator-aware NCCL rows found"

    diagnostic = [r for r in rows if r.get("_diagnostic")]
    data_rows = [r for r in rows if not r.get("_diagnostic")]
    if not data_rows:
        if diagnostic:
            details = diagnostic[0].get("details") or []
            lines = [diagnostic[0].get("message", "No communicator-aware NCCL rows found")]
            for detail in details[:8]:
                lines.append(f"  - {detail}")
            return "\n".join(lines)
        return "No communicator-aware NCCL rows found"

    lines = ["NCCL Communication by Communicator"]
    current_comm = None
    for row in sorted(data_rows, key=lambda r: (r["communicator_hex"], -r["total_ms"])):
        comm = row["communicator_hex"]
        if comm != current_comm:
            current_comm = comm
            header = (
                f"  [{comm}] {row['inferred_dimension']}  "
                f"ranks={row['num_ranks'] if row['num_ranks'] is not None else '?'}  "
                f"devices={row['devices'] or row['device_scope']}"
            )
            lines.append(header)
        bandwidth = (
            f"  BW={row['bandwidth_gbps']:.2f}GB/s"
            if row.get("bandwidth_gbps") is not None
            else "  BW=n/a"
        )
        if row.get("efficiency_pct") is not None:
            bandwidth += f" ({row['efficiency_pct']:.1f}% of {row['peak_source']})"
        lines.append(
            f"    {row['collective_type']:16s} {row['total_ms']:8.3f}ms  "
            f"×{row['count']:<4d} avg={row['avg_ms']:.3f}ms{bandwidth}"
        )

    if diagnostic:
        lines.append("")
        lines.append("  Diagnostics:")
        for item in diagnostic:
            for detail in item.get("details") or [item.get("message")]:
                if detail:
                    lines.append(f"    - {detail}")
    return "\n".join(lines)


def _load_nccl_payload_events(
    prof: Profile,
    trim: tuple[int, int] | None = None,
) -> tuple[list[_DecodedEvent], list[str]]:
    schemas = _load_payload_schemas(prof)
    enum_maps = _load_payload_enums(prof)
    if not schemas:
        return [], [
            "Payload schema tables are unavailable. Use an enriched SQLite export with "
            "`--include-blobs=true`."
        ]

    trim_clause = ""
    params: list[object] = []
    if trim:
        trim_clause = "AND n.[end] >= ? AND n.start <= ?"
        params.extend([int(trim[0]), int(trim[1])])

    try:
        rows = prof._duckdb_query(
            f"""
            SELECT
                COALESCE(n.text, s.value) AS nvtx_text,
                n.start,
                n.[end],
                n.eventType,
                n.globalTid,
                n.domainId,
                n.binaryData
            FROM NVTX_EVENTS n
            LEFT JOIN StringIds s ON n.textId = s.id
            WHERE n.binaryData IS NOT NULL
              AND n.eventType = {_NVTX_RANGE_EVENT}
              AND n.[end] > n.start
              {trim_clause}
            ORDER BY n.globalTid, n.start
            """,
            params,
        )
    except DB_ERRORS as exc:
        msg = str(exc)
        if "Invalid type in column \"binaryData\"" in msg:
            sqlite_path = _discover_sqlite_path(prof)
            if sqlite_path:
                rows, sqlite_diagnostics = _load_nvtx_payload_rows_sqlite(sqlite_path, trim=trim)
                events, diagnostics = _decode_events_from_rows(rows, schemas, enum_maps)
                return events, [
                    "Used SQLite side-connection fallback for NVTX blob decoding in direct mode."
                ] + sqlite_diagnostics + diagnostics
            return [], [
                "Direct DuckDB-over-SQLite mode cannot read mixed TEXT/BLOB NVTX payload data "
                "from this export, and the underlying SQLite path could not be discovered for "
                "automatic fallback."
            ]
        return [], [f"Failed to read NVTX payload events: {msg}"]

    return _decode_events_from_rows(rows, schemas, enum_maps)


def _decode_events_from_rows(
    rows: list[dict],
    schemas: dict[tuple[int, int], list[dict]],
    enum_maps: dict[int, dict[int, str]],
) -> tuple[list[_DecodedEvent], list[str]]:
    diagnostics: list[str] = []
    events: list[_DecodedEvent] = []
    for row in rows:
        nvtx_text = (row.get("nvtx_text") or "").strip()
        if not nvtx_text.lower().startswith("nccl"):
            continue
        try:
            domain_id, schema_id, payload_bytes = _decode_blob_header(row["binaryData"])
        except ValueError as exc:
            diagnostics.append(f"Failed to decode NCCL blob header for '{nvtx_text}': {exc}")
            continue

        schema_entries = schemas.get((domain_id, schema_id))
        if not schema_entries:
            diagnostics.append(
                f"Missing payload schema definition for domain={domain_id} schema={schema_id}"
            )
            continue
        payload = _decode_payload(schema_entries, payload_bytes, enum_maps)
        event = _DecodedEvent(
            nvtx_text=nvtx_text,
            global_tid=int(row["globalTid"]),
            start_ns=int(row["start"]),
            end_ns=int(row["end"]),
            domain_id=domain_id,
            schema_id=schema_id,
            communicator_id=_lookup_field(payload, "communicator id"),
            message_size_bytes=_lookup_field(payload, "message size"),
            num_ranks=_lookup_field(payload, "no. of ranks"),
            rank=_lookup_exact_field(payload, "rank"),
            cuda_device=_lookup_field(payload, "cuda device"),
            reduction_op=_lookup_field(payload, "reduction operation"),
            root_rank=_lookup_exact_field(payload, "root"),
            collective_type=_classify_collective_name(nvtx_text),
        )
        events.append(event)
    return events, diagnostics


def _discover_sqlite_path(prof: Profile) -> str | None:
    path = getattr(prof, "path", "")
    if path and os.path.isfile(path) and path.lower().endswith(".sqlite"):
        return path
    try:
        rows = prof.adapter.execute(
            """
            SELECT path
            FROM duckdb_databases()
            WHERE type = 'sqlite' AND path IS NOT NULL
            ORDER BY database_name
            LIMIT 1
            """
        ).fetchall()
        if rows and rows[0][0]:
            return str(rows[0][0])
    except Exception:
        return None
    return None


def _load_nvtx_payload_rows_sqlite(
    sqlite_path: str,
    trim: tuple[int, int] | None = None,
) -> tuple[list[dict], list[str]]:
    diagnostics: list[str] = []
    trim_clause = ""
    params: list[object] = []
    if trim:
        trim_clause = "AND n.[end] >= ? AND n.start <= ?"
        params.extend([int(trim[0]), int(trim[1])])
    try:
        conn = sqlite3.connect(sqlite_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        result = [
            dict(row)
            for row in cur.execute(
                f"""
                SELECT
                    COALESCE(n.text, s.value) AS nvtx_text,
                    n.start,
                    n.[end],
                    n.eventType,
                    n.globalTid,
                    n.domainId,
                    n.binaryData
                FROM NVTX_EVENTS n
                LEFT JOIN StringIds s ON n.textId = s.id
                WHERE n.binaryData IS NOT NULL
                  AND n.eventType = {_NVTX_RANGE_EVENT}
                  AND n.[end] > n.start
                  {trim_clause}
                ORDER BY n.globalTid, n.start
                """,
                params,
            )
        ]
        conn.close()
        return result, diagnostics
    except sqlite3.Error as exc:
        diagnostics.append(f"SQLite fallback for NVTX blob decoding failed: {exc}")
        return [], diagnostics


def _build_communicator_metadata(events: list[_DecodedEvent]) -> dict[int, dict]:
    metadata: dict[int, dict] = {}
    for event in events:
        if event.communicator_id is None:
            continue
        entry = metadata.setdefault(event.communicator_id, {})
        for key in ("num_ranks", "rank", "cuda_device", "reduction_op", "root_rank"):
            value = getattr(event, key)
            if value is not None and entry.get(key) is None:
                entry[key] = value
    return metadata


def _attribute_events_to_kernels(
    prof: Profile,
    events: list[_DecodedEvent],
    *,
    device: int | None,
    trim: tuple[int, int] | None,
) -> tuple[list[dict], list[str]]:
    tables = prof.adapter.resolve_activity_tables()
    kernel_table = tables.get("kernel", prof.schema.kernel_table or "CUPTI_ACTIVITY_KIND_KERNEL")
    runtime_table = tables.get("runtime", "CUPTI_ACTIVITY_KIND_RUNTIME")

    params: list[object] = []
    where = ["(COALESCE(d.value, s.value, '') LIKE '%nccl%' OR COALESCE(d.value, s.value, '') LIKE '%NCCL%')"]
    if device is not None:
        where.append("k.deviceId = ?")
        params.append(int(device))
    if trim:
        where.append("k.[end] >= ? AND k.start <= ?")
        params.extend([int(trim[0]), int(trim[1])])

    kernel_rows = prof._duckdb_query(
        f"""
        SELECT
            r.globalTid,
            r.start AS rt_start,
            r.[end] AS rt_end,
            k.deviceId,
            k.streamId,
            k.correlationId,
            k.start AS k_start,
            k.[end] AS k_end,
            COALESCE(d.value, s.value, 'kernel_' || CAST(k.shortName AS VARCHAR)) AS kernel_name
        FROM {kernel_table} k
        JOIN {runtime_table} r ON r.correlationId = k.correlationId
        LEFT JOIN StringIds s ON k.shortName = s.id
        LEFT JOIN StringIds d ON k.demangledName = d.id
        WHERE {' AND '.join(where)}
        ORDER BY r.globalTid, r.start
        """,
        params,
    )

    events_by_tid: dict[int, list[_DecodedEvent]] = defaultdict(list)
    for event in events:
        if event.collective_type is None or event.communicator_id is None:
            continue
        events_by_tid[event.global_tid].append(event)
    for rows in events_by_tid.values():
        rows.sort(key=lambda e: (e.start_ns, e.end_ns))

    diagnostics: list[str] = []
    attributed: list[dict] = []
    for kernel in kernel_rows:
        kernel_name = str(kernel["kernel_name"] or "")
        if not _matches_nccl_kernel_allowlist(kernel_name):
            continue
        tid = int(kernel["globalTid"])
        candidates = events_by_tid.get(tid) or []
        if not candidates:
            continue
        rt_start = int(kernel["rt_start"])
        rt_end = int(kernel["rt_end"])
        best_event = None
        best_span = None
        for event in candidates:
            if event.start_ns > rt_start:
                break
            if event.end_ns >= rt_end:
                span = event.end_ns - event.start_ns
                if best_span is None or span < best_span:
                    best_span = span
                    best_event = event
        if best_event is None:
            continue
        attributed.append(
            {
                "communicator_id": best_event.communicator_id,
                "collective_type": best_event.collective_type,
                "duration_ns": int(kernel["k_end"]) - int(kernel["k_start"]),
                "device_id": int(kernel["deviceId"]),
                "stream_id": int(kernel["streamId"]),
                "kernel_name": kernel_name,
                "message_size_bytes": best_event.message_size_bytes,
                "num_ranks": best_event.num_ranks,
                "rank": best_event.rank,
                "cuda_device": best_event.cuda_device,
                "reduction_op": best_event.reduction_op,
                "root_rank": best_event.root_rank,
            }
        )

    if not attributed and kernel_rows:
        diagnostics.append(
            "NCCL kernels were present, but none matched the communicator-event allowlist "
            "after runtime/NVTX attribution."
        )
    return attributed, diagnostics


def _load_payload_schemas(prof: Profile) -> dict[tuple[int, int], list[dict]]:
    try:
        rows = prof._duckdb_query(
            """
            SELECT
                se.domainId,
                se.schemaId,
                se.idx,
                se.name,
                se.type,
                se.offset,
                ps.payloadSize
            FROM NVTX_PAYLOAD_SCHEMA_ENTRIES se
            JOIN NVTX_PAYLOAD_SCHEMAS ps
              ON se.domainId = ps.domainId
             AND se.schemaId = ps.schemaId
            ORDER BY se.domainId, se.schemaId, se.idx
            """
        )
    except DB_ERRORS:
        return {}

    deduped: dict[tuple[int, int, int], dict] = {}
    for row in rows:
        key = (int(row["domainId"]), int(row["schemaId"]), int(row["idx"]))
        deduped.setdefault(key, row)

    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for (_, _, _), row in deduped.items():
        grouped[(int(row["domainId"]), int(row["schemaId"]))].append(row)
    for entries in grouped.values():
        entries.sort(key=lambda r: int(r["idx"]))
    return grouped


def _load_payload_enums(prof: Profile) -> dict[int, dict[int, str]]:
    try:
        rows = prof._duckdb_query(
            """
            SELECT schemaId, idx, name, value
            FROM NVTX_PAYLOAD_ENUM_ENTRIES
            ORDER BY schemaId, idx
            """
        )
    except DB_ERRORS:
        return {}

    enums: dict[int, dict[int, str]] = defaultdict(dict)
    for row in rows:
        schema_id = int(row["schemaId"])
        value = int(row["value"]) if row["value"] is not None else int(row["idx"])
        enums[schema_id].setdefault(value, str(row["name"]))
    return enums


def _decode_blob_header(blob: bytes | bytearray | memoryview | str) -> tuple[int, int, bytes]:
    if isinstance(blob, str):
        raw = bytes.fromhex(blob)
    else:
        raw = bytes(blob)
    if len(raw) < _BLOB_HEADER_SIZE:
        raise ValueError(f"blob too short ({len(raw)} bytes)")
    domain_id = int.from_bytes(raw[0:8], "little", signed=False)
    schema_id = int.from_bytes(raw[8:16], "little", signed=False)
    payload_size = int.from_bytes(raw[16:24], "little", signed=False)
    payload_offset = int.from_bytes(raw[24:32], "little", signed=False)
    if payload_offset < _BLOB_HEADER_SIZE or payload_offset > len(raw):
        raise ValueError(f"unexpected payload offset {payload_offset}")
    payload = raw[payload_offset : payload_offset + payload_size]
    if len(payload) != payload_size:
        raise ValueError(
            f"payload truncated (expected {payload_size} bytes, got {len(payload)})"
        )
    return domain_id, schema_id, payload


def _decode_payload(
    entries: list[dict],
    payload: bytes,
    enum_maps: dict[int, dict[int, str]],
) -> dict[str, object]:
    decoded: dict[str, object] = {}
    offsets = []
    payload_size = int(entries[0]["payloadSize"]) if entries else len(payload)
    for entry in entries:
        start = int(entry["offset"]) if entry["offset"] is not None else 0
        offsets.append(start)

    for index, entry in enumerate(entries):
        start = int(entry["offset"]) if entry["offset"] is not None else 0
        end = payload_size
        if index + 1 < len(entries):
            next_offset = entries[index + 1]["offset"]
            if next_offset is not None:
                end = int(next_offset)
        chunk = payload[start:end]
        if not chunk:
            continue
        value = int.from_bytes(chunk, "little", signed=False)
        enum_type = int(entry["type"]) if entry["type"] is not None else None
        if enum_type in enum_maps:
            low32 = value & 0xFFFFFFFF
            decoded[str(entry["name"])] = enum_maps[enum_type].get(low32, str(low32))
        else:
            if len(chunk) == 4:
                value &= 0xFFFFFFFF
            decoded[str(entry["name"])] = value
    return decoded


def _lookup_field(payload: dict[str, object], needle: str) -> object | None:
    needle = _normalize_name(needle)
    for key, value in payload.items():
        if needle in _normalize_name(key):
            return value
    return None


def _lookup_exact_field(payload: dict[str, object], needle: str) -> object | None:
    needle = _normalize_name(needle)
    for key, value in payload.items():
        if _normalize_name(key) == needle:
            return value
    return None


def _normalize_name(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _classify_collective_name(text: str) -> str | None:
    lower = text.lower()
    if "groupstart" in lower or "groupend" in lower or "comminit" in lower or "commabort" in lower:
        return None
    for hint, label in _COLLECTIVE_NAME_HINTS.items():
        if hint in lower:
            return label
    return None


def _matches_nccl_kernel_allowlist(kernel_name: str) -> bool:
    lower = kernel_name.lower()
    return any(pattern in lower for pattern in _NCCL_KERNEL_PATTERNS)


def _infer_parallel_dimension(num_ranks: int | None, world_size: int | None) -> str:
    if not num_ranks or num_ranks <= 1:
        return "single_rank_or_unknown"
    if world_size and num_ranks == world_size:
        return "data_parallel_or_global"
    if world_size and 1 < num_ranks < world_size:
        return f"subgroup_parallelism({num_ranks})"
    return f"subgroup_parallelism({num_ranks})"


def _estimate_peak_bandwidth_gbps(prof: Profile) -> tuple[float | None, str | None]:
    # `TARGET_INFO_NIC_INFO` does not expose link speed in the checked exports,
    # so only use NIC info when an explicit speed-like column exists. Otherwise
    # fall back to a conservative NVLink estimate from the GPU model.
    try:
        nic_cols = set(prof.adapter.get_table_columns("TARGET_INFO_NIC_INFO"))
    except Exception:
        nic_cols = set()
    speed_cols = next(
        (col for col in nic_cols if col.lower() in {"speed", "linkspeed", "linkspeedgbps"}),
        None,
    )
    if speed_cols:
        try:
            row = prof.adapter.execute(
                f"SELECT MAX({speed_cols}) FROM TARGET_INFO_NIC_INFO"
            ).fetchone()
            if row and row[0]:
                return float(row[0]), f"NIC {speed_cols}"
        except Exception:
            pass

    gpu_name = get_first_gpu_name(prof.conn if prof.db is None else prof.db)
    normalized = (gpu_name or "").upper()
    for key, gbps in sorted(_NVLINK_PEAK_GBPS.items(), key=lambda item: len(item[0]), reverse=True):
        if key in normalized:
            return gbps, f"{key} NVLink"
    return None, None
