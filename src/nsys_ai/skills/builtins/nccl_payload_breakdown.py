"""NCCL payload breakdown — decodes NVTX typed-payload binaryData.

Nsight stores NCCL operation details (communicator ID, message size in
bytes, peer rank, world size) as typed NVTX payloads. The schemas are
declared in ``NVTX_PAYLOAD_SCHEMAS`` / ``NVTX_PAYLOAD_SCHEMA_ENTRIES``;
the actual values for each event live in ``NVTX_EVENTS.binaryData`` as a
raw byte blob.

This skill decodes that blob and aggregates per schema category. It is the
foundation for downstream NCCL-aware skills (``nccl_bandwidth_utilization``,
etc.) — without real message sizes from the profile, those have to estimate.

Binary layout (verified against Nsight Systems 2026.2.x profiles —
revalidate if a future Nsight major version changes the header):
  bytes 0-3   uint32  domainId
  bytes 4-7   uint32  (reserved / flags)
  bytes 8-11  uint32  schemaId
  bytes 12-15 uint32  (reserved / flags)
  bytes 16-23 uint64  payloadSize (matches NVTX_PAYLOAD_SCHEMAS.payloadSize)
  bytes 24-31 uint64  totalSize (= 32 + payloadSize)
  bytes 32+   payload (fields per NVTX_PAYLOAD_SCHEMA_ENTRIES, by `offset`)

Output convention:
  Row 0 is a SUMMARY row with ``_summary: True`` carrying profile-wide
  totals (``total_payload_events``, ``total_bytes_all``, etc.). Rows 1..N
  are per-schema breakdowns. Consumers should skip ``rows[0]`` (or check
  ``r.get('_summary')``) when iterating per-schema data.
"""

import logging
import struct
from collections import defaultdict

from nsys_ai.connection import DB_ERRORS, wrap_connection

from ..base import Skill

_log = logging.getLogger(__name__)

# NVTX payload field type code → byte width. Subset Nsight uses for NCCL.
# Type 5 = uint32 (e.g. rank count, rank id, device id, peer rank).
# Type 18 = uint64 (NCCL communicator ID).
# Type 22 = uint64 (message size in bytes).
_TYPE_SIZES = {5: 4, 18: 8, 22: 8}

# Event header layout (little-endian), 32 bytes total. Derive the size from
# the format so a future format change can't desync this constant.
_HEADER_FMT = "<IIIIQQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _load_schemas(adapter) -> dict[int, dict]:
    """Build {schemaId: {"payload_size": int, "fields": [{offset,type,name}, ...]}}.

    Accepts either a wrapped adapter (uniform .execute().fetchall() across
    sqlite/duckdb) or a raw sqlite3 connection — :func:`wrap_connection` is
    a no-op when given an adapter, so the public callsite can hand us
    whichever the CLI provided.
    """
    schemas: dict[int, dict] = {}
    try:
        for sid, plen in adapter.execute(
            "SELECT DISTINCT schemaId, payloadSize FROM NVTX_PAYLOAD_SCHEMAS"
        ).fetchall():
            schemas[sid] = {"payload_size": plen, "fields": []}
        # `offset` is a DuckDB reserved keyword and `type` is a soft keyword;
        # both must be quoted for the query to parse against the DuckDB cache
        # (sqlite tolerates either form). Without quoting, the query raises
        # `Parser Error: syntax error at or near "FROM"` and `DB_ERRORS`
        # catches it silently — leaving `fields=[]` for every schema and
        # making every binaryData row look undecodable.
        for sid, idx, ftype, fname, foff in adapter.execute(
            'SELECT DISTINCT schemaId, idx, "type", name, "offset" '
            "FROM NVTX_PAYLOAD_SCHEMA_ENTRIES ORDER BY schemaId, idx"
        ).fetchall():
            if sid in schemas:
                # First field's offset is NULL in the table (means 0).
                schemas[sid]["fields"].append(
                    {"idx": idx, "type": ftype, "name": fname, "offset": foff or 0}
                )
    except DB_ERRORS:
        _log.debug("No NVTX_PAYLOAD_SCHEMAS — profile lacks typed payloads", exc_info=True)

    # Safety net: if schemas loaded but every fields[] is empty, the entries
    # query silently failed (the original bug here was DuckDB rejecting
    # `offset` as a reserved keyword, swallowed by DB_ERRORS). Warn so the
    # next regression doesn't waste a debug round-trip.
    if schemas and all(not s["fields"] for s in schemas.values()):
        _log.warning(
            "NVTX_PAYLOAD_SCHEMAS loaded (%d) but NVTX_PAYLOAD_SCHEMA_ENTRIES "
            "returned no fields — entries query likely failed silently. "
            "Common cause: a column rename or reserved-keyword conflict in a "
            "newer Nsight version (e.g. `offset` became reserved in DuckDB). "
            "Inspect the table schema with PRAGMA table_info or DESCRIBE.",
            len(schemas),
        )
    return schemas


def _classify(field_names: list[str]) -> str:
    """Categorize a schema by its field set."""
    fs = set(field_names)
    has_msg = "Message size [bytes]" in fs
    has_peer = "Peer rank" in fs
    has_world = "No. of ranks" in fs
    if has_world:
        return "init"
    if has_msg and has_peer:
        return "p2p"           # ncclSend / ncclRecv with peer
    if has_msg:
        return "collective"    # ncclAllReduce / ncclAllGather etc.
    return "group_marker"      # ncclGroupStart / ncclGroupEnd (comm-id only)


def _decode_row(bd, schemas: dict[int, dict]) -> dict | None:
    """Decode one binaryData blob → {schema_id, fields: {name: value}}.

    Accepts either bytes (sqlite3 BLOB) or str (DuckDB returns BLOB columns
    hex-encoded — twice the byte length, only chars [0-9a-fA-F]).
    """
    if not bd:
        return None
    if isinstance(bd, str):
        # DuckDB path: hex-decode back to raw bytes.
        try:
            bd = bytes.fromhex(bd)
        except ValueError:
            return None
    if len(bd) < _HEADER_SIZE:
        return None
    try:
        _domain, _, schema_id, _, _hdr_payload_size, _ = struct.unpack_from(_HEADER_FMT, bd, 0)
    except struct.error:
        return None
    if schema_id not in schemas:
        return None
    # Trust the schema's declared payload size (authoritative) over the
    # header's payload_size field. Belt-and-braces: if either says the
    # buffer is too short, bail.
    schema_payload_size = schemas[schema_id]["payload_size"] or 0
    required = _HEADER_SIZE + max(schema_payload_size, _hdr_payload_size or 0)
    if len(bd) < required:
        return None
    out: dict = {"schema_id": schema_id}
    fields: dict[str, int] = {}
    for f in schemas[schema_id]["fields"]:
        sz = _TYPE_SIZES.get(f["type"])
        if sz is None:
            continue
        off = _HEADER_SIZE + f["offset"]
        # Guard against schema rows whose offsets exceed the declared
        # payload size — without this, a malformed schema could send us
        # reading past the buffer end.
        if off + sz > _HEADER_SIZE + schema_payload_size:
            continue
        fmt = "<I" if sz == 4 else "<Q"
        try:
            (val,) = struct.unpack_from(fmt, bd, off)
            fields[f["name"]] = val
        except struct.error:
            pass
    out["fields"] = fields
    return out


def _percentile(sorted_vals: list[int], p: float) -> int:
    """Linear-interpolation percentile on a pre-sorted list."""
    if not sorted_vals:
        return 0
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = p * (n - 1)
    lo = int(pos)
    if lo == n - 1:
        return sorted_vals[lo]
    frac = pos - lo
    return int(sorted_vals[lo] + frac * (sorted_vals[lo + 1] - sorted_vals[lo]))


def _execute(conn, **kwargs) -> list[dict]:
    adapter = wrap_connection(conn)
    schemas = _load_schemas(adapter)
    if not schemas:
        return [{"error": "No NVTX_PAYLOAD_SCHEMAS in this profile (NCCL typed payloads not captured)"}]

    # Probe for the two "no data" cases:
    #   (a) capture-time misconfiguration: schemas declared but binaryData
    #       column has zero non-null rows (NCCL ran but typed-payload NVTX
    #       wasn't emitted). Observed on real fastvideo/nano_vllm captures.
    #   (b) backend limitation: DuckDB's sqlite_scanner reports BLOB
    #       columns with declared TEXT type as a type mismatch, raising on
    #       any read. Hits when the CLI is run with --no-cache against a
    #       profile that DOES have binaryData. Without this branch we'd
    #       misreport (b) as (a) and tell the user to re-profile when the
    #       fix is "use the parquet cache instead".
    try:
        (has_any,) = adapter.execute(
            "SELECT EXISTS(SELECT 1 FROM NVTX_EVENTS WHERE binaryData IS NOT NULL)"
        ).fetchone()
        probe_failed = False
    except DB_ERRORS as exc:
        has_any = 0
        # Distinguish (b) from (a): if NVTX_EVENTS rows exist but the
        # binaryData filter errored, the column is unreadable through this
        # backend — not actually empty.
        probe_failed = False
        try:
            (nvtx_rows,) = adapter.execute("SELECT COUNT(*) FROM NVTX_EVENTS").fetchone()
            if nvtx_rows > 0:
                probe_failed = True
                probe_error_msg = str(exc)
        except DB_ERRORS:
            pass

    if probe_failed:
        return [{
            "error": (
                "Cannot read NVTX_EVENTS.binaryData through the current "
                "backend (DuckDB sqlite-scanner raises on BLOB columns "
                f"declared as TEXT in sqlite: {probe_error_msg!r}). "
                "Re-run without `--no-cache` — the parquet cache path "
                "stores binaryData as hex-encoded strings and works "
                "correctly."
            ),
            "schemas_present": len(schemas),
            "backend_limitation": True,
        }]
    if not has_any:
        return [{
            "error": (
                f"NVTX_PAYLOAD_SCHEMAS declares {len(schemas)} schemas but "
                "NVTX_EVENTS.binaryData is empty — NCCL kernels likely ran but "
                "typed-payload NVTX emission was not enabled at capture time. "
                "Re-profile with `nsys profile -t cuda,nvtx,...` and ensure NCCL "
                "is built with NVTX support (NCCL ≥ 2.18 emits typed payloads "
                "by default; older builds require NCCL_NVTX_DOMAIN_EVENTS=1)."
            ),
            "schemas_present": len(schemas),
            "binary_data_rows": 0,
        }]

    # Honor the standard trim convention documented in PRINCIPLES.md §9.
    # Either both bounds set or neither; partial trim falls through to
    # whole-profile (consistent with how iteration_detail / overlap_breakdown
    # handle the same kwargs). Partial-bound passes log a WARNING so users
    # know their intent was dropped rather than silently honored.
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    trim_clause = ""
    trim_params: list = []
    if trim_start is not None and trim_end is not None:
        # NVTX event overlaps the window if start <= trim_end AND end >= trim_start.
        trim_clause = ' AND start <= ? AND "end" >= ?'
        trim_params = [int(trim_end), int(trim_start)]
    elif trim_start is not None or trim_end is not None:
        _log.warning(
            "nccl_payload_breakdown: partial trim bounds dropped — both "
            "trim_start_ns and trim_end_ns must be set to scope the query "
            "(got start=%r, end=%r). Returning whole-profile aggregation.",
            trim_start, trim_end,
        )

    # Per-schema buckets + a cross-schema communicator set so the summary
    # can answer "are these schemas all on the same comm or different?"
    # (Without this, six schemas each reporting "distinct_communicators: 1"
    # is ambiguous between "same comm everywhere" and "six different
    # single-rank comms" — observed when reviewing the skill on
    # tgate05_perf.sqlite.)
    per_schema: dict[int, dict] = defaultdict(
        lambda: {
            "count": 0,
            "msg_sizes": [],
            "communicators": set(),
            "peer_ranks": defaultdict(int),
        }
    )
    global_communicators: set = set()
    decoded = 0
    skipped = 0

    for (bd,) in adapter.execute(
        f"SELECT binaryData FROM NVTX_EVENTS WHERE binaryData IS NOT NULL{trim_clause}",
        trim_params,
    ).fetchall():
        row = _decode_row(bd, schemas)
        if row is None:
            skipped += 1
            continue
        decoded += 1
        sid = row["schema_id"]
        bucket = per_schema[sid]
        bucket["count"] += 1
        f = row["fields"]
        if "Message size [bytes]" in f:
            bucket["msg_sizes"].append(f["Message size [bytes]"])
        if "NCCL communicator ID" in f:
            bucket["communicators"].add(f["NCCL communicator ID"])
            global_communicators.add(f["NCCL communicator ID"])
        if "Peer rank" in f:
            bucket["peer_ranks"][f["Peer rank"]] += 1

    rows: list[dict] = []
    total_bytes = 0
    message_carrying_events = 0
    for sid, bucket in sorted(per_schema.items()):
        field_names = [fl["name"] for fl in schemas[sid]["fields"]]
        category = _classify(field_names)
        sizes = sorted(bucket["msg_sizes"])
        msg_total = sum(sizes)
        total_bytes += msg_total
        row = {
            "schema_id": sid,
            "category": category,
            "field_set": field_names,
            "count": bucket["count"],
            "distinct_communicators": len(bucket["communicators"]),
        }
        # Surface peer-rank distribution for schemas that record it (p2p).
        # A uniformity score = 1.0 means traffic is split evenly across all
        # observed peers (e.g. symmetric all-to-all), < 1.0 means skewed
        # (e.g. ring-only or hub-spoke). Computed only when ≥ 1 peer
        # observed so init/collective/group_marker rows stay compact.
        if bucket["peer_ranks"]:
            peer_counts = dict(bucket["peer_ranks"])
            total_peer_events = sum(peer_counts.values())
            n_peers = len(peer_counts)
            # Uniformity score (Jain's fairness index simplified for our use):
            #   1.0 when all peers see the same traffic share
            #   1/n when one peer hogs everything
            if n_peers > 0 and total_peer_events > 0:
                mean = total_peer_events / n_peers
                ssq = sum((c - mean) ** 2 for c in peer_counts.values())
                # std/mean coefficient of variation → mapped to (0, 1] uniformity
                cv = (ssq / n_peers) ** 0.5 / mean if mean > 0 else 0.0
                uniformity = max(0.0, 1.0 - cv)
                row["distinct_peer_ranks"] = n_peers
                row["peer_rank_uniformity"] = round(uniformity, 3)
                row["peer_rank_distribution"] = peer_counts
        if sizes:
            message_carrying_events += len(sizes)
            row["msg_size_bytes_total"] = msg_total
            row["msg_size_bytes_min"] = sizes[0]
            row["msg_size_bytes_p50"] = _percentile(sizes, 0.50)
            row["msg_size_bytes_p99"] = _percentile(sizes, 0.99)
            row["msg_size_bytes_max"] = sizes[-1]
        rows.append(row)

    # Prepend a summary row so consumers see profile-wide totals.
    # `total_payload_events` counts every decoded NVTX payload event;
    # `message_carrying_events` is the subset whose schema declares a
    # "Message size [bytes]" field (collective + p2p). The gap is
    # init + group_marker events. Surfacing both means a profile with
    # 1000 group_markers and 0 collectives reads as "1000 events / 0
    # carrying / 0 bytes" instead of the ambiguous "1000 events / 0 bytes".
    summary = {
        "_summary": True,
        "total_payload_events": decoded,
        "message_carrying_events": message_carrying_events,
        "skipped_events": skipped,
        "total_bytes_all": total_bytes,
        "total_gib_all": round(total_bytes / (1024**3), 2),
        "distinct_schemas": len(per_schema),
        # Cross-schema communicator rollup. "1" here means every event in
        # every schema uses the same NCCL communicator (typical for a
        # single SP/TP group). Higher values indicate multi-group setups
        # (e.g. DP×TP×SP). Per-schema `distinct_communicators` is a subset
        # view of this set.
        "total_distinct_communicators": len(global_communicators),
    }
    return [summary, *rows]


def _format(rows: list[dict]) -> str:
    if not rows or "error" in rows[0]:
        return f"(NCCL payload: {rows[0].get('error', 'no data') if rows else 'no data'})"
    s = rows[0]
    carrying = s.get("message_carrying_events", s["total_payload_events"])
    lines = [
        "── NCCL Payload Breakdown ──",
        f"  Total payload events:  {s['total_payload_events']:,}"
        + (f"  ({s['skipped_events']:,} undecodable)" if s["skipped_events"] else ""),
        f"  Message-carrying:      {carrying:,}"
        f"   ({s['total_gib_all']} GiB total)",
        f"  Distinct schemas:      {s['distinct_schemas']}",
        "",
    ]
    for r in rows[1:]:
        sid = r["schema_id"]
        cat = r["category"]
        cnt = r["count"]
        comms = r["distinct_communicators"]
        head = f"  [{cat:<14}] schema={sid}  calls={cnt:>8,}  communicators={comms}"
        if "msg_size_bytes_total" in r:
            tot_gib = r["msg_size_bytes_total"] / (1024**3)
            p50_mib = r["msg_size_bytes_p50"] / (1024**2)
            p99_mib = r["msg_size_bytes_p99"] / (1024**2)
            head += (
                f"\n      msg_size: p50={p50_mib:.2f} MiB  p99={p99_mib:.2f} MiB"
                f"   total={tot_gib:.2f} GiB"
            )
        lines.append(head)
    return "\n".join(lines)


SKILL = Skill(
    name="nccl_payload_breakdown",
    title="NCCL Payload Breakdown (message sizes via NVTX typed payloads)",
    description=(
        "Decodes NCCL typed-payload NVTX events to extract real per-collective "
        "message sizes, peer ranks, and communicator IDs from NVTX_EVENTS.binaryData. "
        "Distinguishes init, group markers, collective, and p2p operations by their "
        "payload schema. Foundation for bandwidth-utilization analysis."
    ),
    category="communication",
    execute_fn=_execute,
    format_fn=_format,
    tags=[
        "nccl", "payload", "message-size", "bandwidth", "communication",
        "distributed", "p2p", "collective",
    ],
)
