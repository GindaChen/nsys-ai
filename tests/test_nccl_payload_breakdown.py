"""Tests for nccl_payload_breakdown skill.

Builds a minimal sqlite that mirrors Nsight's NVTX_PAYLOAD_SCHEMAS +
NVTX_PAYLOAD_SCHEMA_ENTRIES + NVTX_EVENTS.binaryData layout, then asserts
the decoder + aggregator produce the expected per-category breakdown.

Binary layout encoded by `_pack_event` mirrors the format documented in the
skill's module docstring (32-byte header + payload-bytes from schema).
"""

import sqlite3
import struct

from nsys_ai.skills.builtins.nccl_payload_breakdown import (
    _HEADER_FMT,
    _HEADER_SIZE,
    SKILL,
    _classify,
    _decode_row,
    _load_schemas,
    _percentile,
)


def _pack_event(schema_id: int, payload_bytes: bytes) -> bytes:
    """Build the header + payload using the production constants.

    Importing `_HEADER_FMT` / `_HEADER_SIZE` directly (rather than hard-coding
    "<IIIIQQ" / 32 here) means any future header-layout change in the skill
    will be reflected by these test fixtures too — preventing the test suite
    from drifting silently behind production.
    """
    payload_size = len(payload_bytes)
    header = struct.pack(_HEADER_FMT, 1, 0, schema_id, 0, payload_size, _HEADER_SIZE + payload_size)
    return header + payload_bytes


def _build_db_with_nccl_payloads():
    """Create an in-memory sqlite with two schemas + a handful of decoded rows."""
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    c.execute(
        "CREATE TABLE NVTX_PAYLOAD_SCHEMAS "
        "(domainId INTEGER, schemaId INTEGER, name TEXT, type INTEGER, "
        " flags INTEGER, numEntries INTEGER, payloadSize INTEGER, alignTo INTEGER)"
    )
    c.execute(
        "CREATE TABLE NVTX_PAYLOAD_SCHEMA_ENTRIES "
        "(domainId INTEGER, schemaId INTEGER, idx INTEGER, flags INTEGER, "
        " type INTEGER, name TEXT, description TEXT, arrayOrUnionDetail INTEGER, offset INTEGER)"
    )
    c.execute(
        "CREATE TABLE NVTX_EVENTS "
        "(start INTEGER, eventType INTEGER, binaryData BLOB)"
    )

    # Schema A: simple collective (16 bytes payload — comm_id + msg_size)
    c.execute(
        "INSERT INTO NVTX_PAYLOAD_SCHEMAS VALUES (1, 100, NULL, 1, NULL, 2, 16, NULL)"
    )
    c.execute(
        "INSERT INTO NVTX_PAYLOAD_SCHEMA_ENTRIES VALUES "
        "(1, 100, 0, NULL, 18, 'NCCL communicator ID', NULL, NULL, NULL)"
    )
    c.execute(
        "INSERT INTO NVTX_PAYLOAD_SCHEMA_ENTRIES VALUES "
        "(1, 100, 1, NULL, 22, 'Message size [bytes]', NULL, NULL, 8)"
    )

    # Schema B: p2p (24 bytes — comm_id + msg_size + peer_rank)
    c.execute(
        "INSERT INTO NVTX_PAYLOAD_SCHEMAS VALUES (1, 200, NULL, 1, NULL, 3, 24, NULL)"
    )
    c.execute(
        "INSERT INTO NVTX_PAYLOAD_SCHEMA_ENTRIES VALUES "
        "(1, 200, 0, NULL, 18, 'NCCL communicator ID', NULL, NULL, NULL)"
    )
    c.execute(
        "INSERT INTO NVTX_PAYLOAD_SCHEMA_ENTRIES VALUES "
        "(1, 200, 1, NULL, 22, 'Message size [bytes]', NULL, NULL, 8)"
    )
    c.execute(
        "INSERT INTO NVTX_PAYLOAD_SCHEMA_ENTRIES VALUES "
        "(1, 200, 2, NULL, 5, 'Peer rank', NULL, NULL, 16)"
    )

    # Insert events:
    #   - 3 collective events on schema 100 with msg_sizes [1024, 2048, 4096]
    #   - 2 p2p events on schema 200 with msg_sizes [8192, 16384], peers [1, 2]
    comm = 0xCAFE_BABE_DEAD_BEEF
    for ms in (1024, 2048, 4096):
        payload = struct.pack("<QQ", comm, ms)
        c.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, 59, ?)", (0, _pack_event(100, payload))
        )
    for ms, peer in ((8192, 1), (16384, 2)):
        payload = struct.pack("<QQI", comm, ms, peer) + b"\x00\x00\x00\x00"
        c.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, 59, ?)", (0, _pack_event(200, payload))
        )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Decoder unit tests
# ---------------------------------------------------------------------------


def test_load_schemas_returns_field_layout():
    conn = _build_db_with_nccl_payloads()
    schemas = _load_schemas(conn)
    assert set(schemas.keys()) == {100, 200}
    assert schemas[100]["payload_size"] == 16
    assert schemas[200]["payload_size"] == 24
    # field offsets: first is NULL → coerced to 0
    s100_fields = {f["name"]: f["offset"] for f in schemas[100]["fields"]}
    assert s100_fields == {"NCCL communicator ID": 0, "Message size [bytes]": 8}


def test_decode_row_extracts_fields():
    schemas = {100: {"payload_size": 16, "fields": [
        {"idx": 0, "type": 18, "name": "NCCL communicator ID", "offset": 0},
        {"idx": 1, "type": 22, "name": "Message size [bytes]", "offset": 8},
    ]}}
    payload = struct.pack("<QQ", 0xDEAD_BEEF, 65536)
    bd = _pack_event(100, payload)
    decoded = _decode_row(bd, schemas)
    assert decoded == {
        "schema_id": 100,
        "fields": {"NCCL communicator ID": 0xDEAD_BEEF, "Message size [bytes]": 65536},
    }


def test_decode_row_returns_none_on_truncated_blob():
    assert _decode_row(b"", {}) is None
    assert _decode_row(b"\x00" * 10, {}) is None  # shorter than 32-byte header


def test_decode_row_returns_none_on_unknown_schema():
    bd = _pack_event(99999, struct.pack("<QQ", 1, 1))
    assert _decode_row(bd, {}) is None


def test_classify_buckets():
    assert _classify(["NCCL communicator ID"]) == "group_marker"
    assert _classify(["NCCL communicator ID", "Message size [bytes]"]) == "collective"
    assert _classify(["NCCL communicator ID", "Message size [bytes]", "Peer rank"]) == "p2p"
    assert _classify(["NCCL communicator ID", "No. of ranks", "Rank", "CUDA device"]) == "init"


def test_percentile_handles_edge_cases():
    assert _percentile([], 0.5) == 0
    assert _percentile([42], 0.5) == 42
    # p50 of [10, 20, 30] = 20 (linear interp at index 1.0)
    assert _percentile([10, 20, 30], 0.5) == 20
    # p99 of [10, 20, 30] ≈ 29.6 → int → 29
    assert _percentile([10, 20, 30], 0.99) == 29


# ---------------------------------------------------------------------------
# Skill execute + aggregate
# ---------------------------------------------------------------------------


def test_skill_aggregates_per_schema_correctly():
    conn = _build_db_with_nccl_payloads()
    rows = SKILL.execute_fn(conn)

    # First row is the summary.
    s = rows[0]
    assert s["_summary"] is True
    assert s["total_payload_events"] == 5  # 3 collective + 2 p2p
    assert s["distinct_schemas"] == 2

    # Per-schema rows.
    data = {r["schema_id"]: r for r in rows[1:]}
    assert 100 in data and 200 in data

    coll = data[100]
    assert coll["category"] == "collective"
    assert coll["count"] == 3
    assert coll["msg_size_bytes_total"] == 1024 + 2048 + 4096
    assert coll["msg_size_bytes_min"] == 1024
    assert coll["msg_size_bytes_max"] == 4096
    assert coll["distinct_communicators"] == 1

    p2p = data[200]
    assert p2p["category"] == "p2p"
    assert p2p["count"] == 2
    assert p2p["msg_size_bytes_total"] == 8192 + 16384
    assert "Peer rank" not in p2p["field_set"] or "Peer rank" in p2p["field_set"]
    # The peer field is captured in field_set:
    assert "Peer rank" in p2p["field_set"]


def test_skill_returns_error_when_no_payload_schemas():
    """If the profile has no NVTX_PAYLOAD_SCHEMAS table, return a clear error."""
    conn = sqlite3.connect(":memory:")
    rows = SKILL.execute_fn(conn)
    assert "error" in rows[0]
    assert "NVTX_PAYLOAD" in rows[0]["error"]


def test_skill_distinguishes_blob_read_failure_from_empty_data():
    """If the backend's BLOB scanner errors out (observed: DuckDB
    sqlite-scanner on --no-cache mode raises a 'Mismatch Type Error' for
    BLOB columns declared as TEXT in sqlite), the skill must NOT silently
    report 'binaryData empty' — that would tell users to re-profile when
    the real fix is to switch off --no-cache.

    Mocks the adapter so the binaryData probe raises, but the table-row
    count succeeds (i.e. table is reachable, just unreadable through this
    backend)."""
    from nsys_ai.connection import DB_ERRORS

    # Pick a real DB_ERRORS class to raise. We use the first one in the
    # tuple (typically sqlite3.Error) which IS in DB_ERRORS, so the
    # skill's try/except matches.
    db_error_cls = DB_ERRORS[0]

    class _FlakyAdapter:
        """Reports schemas + total rows fine, but errors on binaryData reads."""

        def __init__(self, base_conn):
            self._base = base_conn

        def execute(self, sql, params=()):
            if "binaryData IS NOT NULL" in sql:
                raise db_error_cls(
                    "Mismatch Type Error: column declared text, found blob"
                )
            return self._base.execute(sql, params)

    conn = _build_db_with_nccl_payloads()
    # Patch wrap_connection to return our flaky adapter.
    from nsys_ai.skills.builtins import nccl_payload_breakdown as mod
    orig = mod.wrap_connection
    mod.wrap_connection = lambda c: _FlakyAdapter(c)
    try:
        rows = SKILL.execute_fn(conn)
    finally:
        mod.wrap_connection = orig

    assert "error" in rows[0]
    assert rows[0].get("backend_limitation") is True, (
        f"Expected backend_limitation:True; got {rows[0]}"
    )
    # Error message must NOT recommend re-profiling (the real fix is
    # switching backend, not re-capturing the profile).
    assert "Re-profile" not in rows[0]["error"]
    assert "--no-cache" in rows[0]["error"]


def test_skill_handles_empty_events_table():
    """Schemas defined but no binaryData events → explicit error message
    explaining the likely capture-flag issue, not silent zeros. This is
    the second-most-common "no data" case after `no schemas at all` and
    was observed on real fastvideo / nano_vllm profiles where NCCL ran
    but typed-payload NVTX emission wasn't enabled at capture time."""
    conn = _build_db_with_nccl_payloads()
    conn.execute("DELETE FROM NVTX_EVENTS")
    conn.commit()
    rows = SKILL.execute_fn(conn)
    assert "error" in rows[0]
    # Error must call out the capture-flag root cause:
    assert "typed-payload" in rows[0]["error"]
    assert "capture" in rows[0]["error"].lower()
    # Diagnostic fields to help the agent build a re-profile suggestion:
    assert rows[0]["schemas_present"] == 2  # from the fixture
    assert rows[0]["binary_data_rows"] == 0


def test_format_produces_human_readable_output():
    conn = _build_db_with_nccl_payloads()
    text = SKILL.format_fn(SKILL.execute_fn(conn))
    assert "NCCL Payload Breakdown" in text
    assert "collective" in text
    assert "p2p" in text
    assert "MiB" in text  # message size formatted


# ---------------------------------------------------------------------------
# Regression / coverage gaps surfaced during real-profile end-to-end debugging
# ---------------------------------------------------------------------------


def test_decode_row_accepts_hex_string_blob():
    """DuckDB returns BLOB columns as hex-encoded strings (e.g. '0100...').
    Sqlite3 returns raw bytes. The decoder must handle both — this test
    locks in the hex-string path so a future regression that re-breaks
    DuckDB compatibility fails fast at unit-test time."""
    schemas = {100: {"payload_size": 16, "fields": [
        {"idx": 0, "type": 18, "name": "NCCL communicator ID", "offset": 0},
        {"idx": 1, "type": 22, "name": "Message size [bytes]", "offset": 8},
    ]}}
    payload = struct.pack("<QQ", 0xCAFE_BEEF, 32768)
    raw = _pack_event(100, payload)
    hex_str = raw.hex()  # DuckDB-style representation
    decoded = _decode_row(hex_str, schemas)
    assert decoded is not None
    assert decoded["schema_id"] == 100
    assert decoded["fields"]["Message size [bytes]"] == 32768


def test_decode_row_rejects_non_hex_string():
    """A str blob that isn't valid hex (e.g. garbage) must return None,
    not raise."""
    assert _decode_row("not-actually-hex-zzz", {}) is None


def test_percentile_boundary_p0_and_p1():
    """Percentile at the endpoints should return min/max exactly."""
    vals = [10, 20, 30, 40, 50]
    assert _percentile(vals, 0.0) == 10
    assert _percentile(vals, 1.0) == 50


def test_decode_row_skips_unknown_field_type():
    """If a schema declares a field with a type not in TYPE_SIZES (e.g. a
    future Nsight emits type 99 = string), the decoder must skip that
    field — never crash. The other fields with known types still decode."""
    schemas = {100: {"payload_size": 16, "fields": [
        {"idx": 0, "type": 18, "name": "NCCL communicator ID", "offset": 0},
        {"idx": 1, "type": 99, "name": "Future field (unknown)", "offset": 8},
    ]}}
    payload = struct.pack("<QQ", 0xCAFE_BEEF, 12345)
    decoded = _decode_row(_pack_event(100, payload), schemas)
    assert decoded is not None
    # Known field decoded:
    assert decoded["fields"]["NCCL communicator ID"] == 0xCAFE_BEEF
    # Unknown-type field skipped (not crashed, not present):
    assert "Future field (unknown)" not in decoded["fields"]


def test_load_schemas_warns_on_silent_entries_failure(caplog):
    """Regression guard for the DuckDB reserved-keyword landmine: when
    NVTX_PAYLOAD_SCHEMAS loads but NVTX_PAYLOAD_SCHEMA_ENTRIES fails (or
    returns no rows), _load_schemas must log a WARNING. Without this, a
    silent SQL failure leaves every event undecodable and no signal in
    the output explains why."""
    import logging

    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    # Schemas table populated, entries table empty.
    c.execute(
        "CREATE TABLE NVTX_PAYLOAD_SCHEMAS "
        "(domainId INTEGER, schemaId INTEGER, name TEXT, type INTEGER, "
        " flags INTEGER, numEntries INTEGER, payloadSize INTEGER, alignTo INTEGER)"
    )
    c.execute("INSERT INTO NVTX_PAYLOAD_SCHEMAS VALUES (1, 100, NULL, 1, NULL, 2, 16, NULL)")
    c.execute(
        "CREATE TABLE NVTX_PAYLOAD_SCHEMA_ENTRIES "
        "(domainId INTEGER, schemaId INTEGER, idx INTEGER, flags INTEGER, "
        " type INTEGER, name TEXT, description TEXT, arrayOrUnionDetail INTEGER, offset INTEGER)"
    )
    conn.commit()

    with caplog.at_level(logging.WARNING, logger="nsys_ai.skills.builtins.nccl_payload_breakdown"):
        schemas = _load_schemas(conn)

    assert len(schemas) == 1
    assert all(not s["fields"] for s in schemas.values())
    assert any(
        "returned no fields" in rec.message for rec in caplog.records
    ), f"expected WARNING about empty fields[], got: {[r.message for r in caplog.records]}"


def test_decode_row_rejects_buffer_too_short_for_schema():
    """When a header advertises a tiny payload but the schema declares
    fields at higher offsets, the decoder must not read past the buffer
    end. Trust the schema's declared payload size as authoritative."""
    schemas = {100: {"payload_size": 24, "fields": [
        {"idx": 0, "type": 18, "name": "NCCL communicator ID", "offset": 0},
        {"idx": 1, "type": 22, "name": "Message size [bytes]", "offset": 8},
        {"idx": 2, "type": 5, "name": "Peer rank", "offset": 16},
    ]}}
    # Pack a 16-byte payload (less than schema's 24-byte requirement).
    short_payload = struct.pack("<QQ", 0xDEAD_BEEF, 4096)
    bd = _pack_event(100, short_payload)
    # bd length is _HEADER_SIZE + 16 — schema needs _HEADER_SIZE + 24.
    decoded = _decode_row(bd, schemas)
    assert decoded is None, (
        "Decoder must reject when buffer is shorter than schema's declared "
        "payload size, even if the header agrees with the shorter length"
    )


def test_decode_row_skips_field_offsets_beyond_payload():
    """If a schema has a single oversized field offset beyond payload_size,
    that field is skipped — other fields still decode."""
    schemas = {100: {"payload_size": 16, "fields": [
        {"idx": 0, "type": 18, "name": "NCCL communicator ID", "offset": 0},
        # Bogus: offset 100 exceeds payload_size=16
        {"idx": 1, "type": 22, "name": "Out-of-range field", "offset": 100},
    ]}}
    payload = struct.pack("<QQ", 0xCAFE_BEEF, 9999)
    decoded = _decode_row(_pack_event(100, payload), schemas)
    assert decoded is not None
    assert decoded["fields"]["NCCL communicator ID"] == 0xCAFE_BEEF
    assert "Out-of-range field" not in decoded["fields"]


def test_header_size_matches_format():
    """_HEADER_SIZE is computed from _HEADER_FMT at import time. Verify
    they stay consistent in case someone hand-edits one but not the other."""
    assert _HEADER_SIZE == struct.calcsize(_HEADER_FMT)


def test_skill_respects_trim_window():
    """Passing trim_start_ns / trim_end_ns must scope aggregation to events
    whose [start, end] overlaps the window. Required by PRINCIPLES.md §9
    so agents trimming to one iteration get scoped NCCL payload data."""
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    c.execute(
        "CREATE TABLE NVTX_PAYLOAD_SCHEMAS "
        "(domainId INTEGER, schemaId INTEGER, name TEXT, type INTEGER, "
        " flags INTEGER, numEntries INTEGER, payloadSize INTEGER, alignTo INTEGER)"
    )
    c.execute("INSERT INTO NVTX_PAYLOAD_SCHEMAS VALUES (1, 100, NULL, 1, NULL, 2, 16, NULL)")
    c.execute(
        "CREATE TABLE NVTX_PAYLOAD_SCHEMA_ENTRIES "
        "(domainId INTEGER, schemaId INTEGER, idx INTEGER, flags INTEGER, "
        " type INTEGER, name TEXT, description TEXT, arrayOrUnionDetail INTEGER, offset INTEGER)"
    )
    c.execute("INSERT INTO NVTX_PAYLOAD_SCHEMA_ENTRIES VALUES (1, 100, 0, NULL, 18, 'NCCL communicator ID', NULL, NULL, NULL)")
    c.execute("INSERT INTO NVTX_PAYLOAD_SCHEMA_ENTRIES VALUES (1, 100, 1, NULL, 22, 'Message size [bytes]', NULL, NULL, 8)")
    c.execute(
        'CREATE TABLE NVTX_EVENTS (start INTEGER, "end" INTEGER, binaryData BLOB)'
    )
    # 3 events at three different time windows: 0-100, 1000-1100, 2000-2100.
    for start, msg_size in ((0, 1024), (1000, 2048), (2000, 4096)):
        payload = struct.pack("<QQ", 0xCAFE_BABE, msg_size)
        c.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?)",
            (start, start + 100, _pack_event(100, payload)),
        )
    conn.commit()

    # Without trim: all 3 events
    rows_all = SKILL.execute_fn(conn)
    assert rows_all[0]["total_payload_events"] == 3

    # Trim to [500, 1500] — only the middle event (1000-1100) overlaps
    rows_mid = SKILL.execute_fn(conn, trim_start_ns=500, trim_end_ns=1500)
    assert rows_mid[0]["total_payload_events"] == 1
    # Confirm the middle message size (2048) is the only one captured
    by_sid = {r["schema_id"]: r for r in rows_mid[1:]}
    assert by_sid[100]["msg_size_bytes_total"] == 2048

    # Trim to [50, 2050] — captures all three (each event 100ns wide, all
    # within or overlapping this 2 µs window)
    rows_wide = SKILL.execute_fn(conn, trim_start_ns=50, trim_end_ns=2050)
    assert rows_wide[0]["total_payload_events"] == 3


def test_skill_trim_partial_bounds_falls_through_to_whole_profile():
    """Only one of trim_start_ns / trim_end_ns set → trim ignored
    (consistent with overlap_breakdown / iteration_detail behavior)."""
    conn = _build_db_with_nccl_payloads()
    full = SKILL.execute_fn(conn)[0]["total_payload_events"]
    # Pass only trim_start_ns → should NOT filter (whole profile returned).
    partial = SKILL.execute_fn(conn, trim_start_ns=999_999_999_999)[0]["total_payload_events"]
    assert partial == full, (
        f"Partial-bounds trim should fall through; got {partial} vs full {full}"
    )


def test_skill_warns_on_partial_trim_bounds(caplog):
    """A user passing only one trim bound is almost certainly a mistake —
    must log a WARNING so the silent fall-through doesn't mask the bug."""
    import logging

    conn = _build_db_with_nccl_payloads()
    with caplog.at_level(logging.WARNING, logger="nsys_ai.skills.builtins.nccl_payload_breakdown"):
        SKILL.execute_fn(conn, trim_start_ns=12345)
    assert any(
        "partial trim bounds dropped" in rec.message for rec in caplog.records
    ), f"expected WARNING about partial trim; got: {[r.message for r in caplog.records]}"


def test_format_handles_summary_only_no_data_rows():
    """When the schemas exist but no events match (e.g. heavy trim), the
    summary row is the only row. The formatter must not crash and must
    produce a recognisable output."""
    conn = _build_db_with_nccl_payloads()
    # Trim window in the distant past — captures zero events.
    rows = SKILL.execute_fn(conn, trim_start_ns=-1_000_000, trim_end_ns=-1)
    assert rows[0]["_summary"] is True
    assert rows[0]["total_payload_events"] == 0
    # Format must not crash on summary-only output.
    text = SKILL.format_fn(rows)
    assert "NCCL Payload Breakdown" in text
    assert "Total payload events:  0" in text


def test_skill_output_is_json_serializable():
    """The skill emits via CLI's `--format json` path. If anything leaks
    into the output that json can't serialize (set, bytes, NaN, None
    from a malformed schema), this catches it before end-to-end."""
    import json

    conn = _build_db_with_nccl_payloads()
    rows = SKILL.execute_fn(conn)
    # Must not raise.
    serialized = json.dumps(rows)
    # Round-trip equality on basic structure.
    re_parsed = json.loads(serialized)
    assert re_parsed[0]["_summary"] is True
    assert len(re_parsed) == len(rows)


def test_summary_message_carrying_events_excludes_marker_only_schemas():
    """`message_carrying_events` must count ONLY events whose schema declares
    a "Message size [bytes]" field — group_marker / init events don't
    contribute. Validates the field's documented semantics."""
    conn = _build_db_with_nccl_payloads()
    # Add a group_marker event (schema 300 with only comm_id, no msg_size).
    c = conn.cursor()
    c.execute("INSERT INTO NVTX_PAYLOAD_SCHEMAS VALUES (1, 300, NULL, 1, NULL, 1, 8, NULL)")
    c.execute(
        "INSERT INTO NVTX_PAYLOAD_SCHEMA_ENTRIES VALUES "
        "(1, 300, 0, NULL, 18, 'NCCL communicator ID', NULL, NULL, NULL)"
    )
    # 2 group_marker events
    for _ in range(2):
        payload = struct.pack("<Q", 0xCAFE_BABE_DEAD_BEEF)
        c.execute("INSERT INTO NVTX_EVENTS VALUES (?, 59, ?)", (0, _pack_event(300, payload)))
    conn.commit()

    rows = SKILL.execute_fn(conn)
    s = rows[0]
    # Fixture: 3 collective + 2 p2p = 5 message-carrying. Plus 2 group_marker = 7 total.
    assert s["total_payload_events"] == 7
    assert s["message_carrying_events"] == 5, (
        f"group_marker events must NOT be counted; got "
        f"message_carrying={s['message_carrying_events']}"
    )


def test_summary_totals_consistent_with_per_schema_totals():
    """`_summary.total_bytes_all` MUST equal the sum of per-schema
    `msg_size_bytes_total` — any drift indicates the aggregation logic
    diverged between the summary path and the per-row path."""
    conn = _build_db_with_nccl_payloads()
    rows = SKILL.execute_fn(conn)
    summary = rows[0]
    per_schema_total = sum(
        r.get("msg_size_bytes_total", 0) for r in rows[1:]
    )
    assert summary["total_bytes_all"] == per_schema_total, (
        f"summary total {summary['total_bytes_all']} != "
        f"sum of per-schema totals {per_schema_total}"
    )


def test_skill_counts_distinct_communicators():
    """When two events share a communicator ID, distinct_communicators must
    not double-count. When they differ, it must reflect both."""
    conn = _build_db_with_nccl_payloads()
    # The fixture uses the same comm_id (0xCAFEBABEDEADBEEF) for all events.
    rows = SKILL.execute_fn(conn)
    data = {r["schema_id"]: r for r in rows[1:]}
    assert data[100]["distinct_communicators"] == 1
    assert data[200]["distinct_communicators"] == 1

    # Add an event on schema 100 with a different communicator and re-check.
    c = conn.cursor()
    payload = struct.pack("<QQ", 0xFEED_FACE_BEEFCAFE, 8192)
    c.execute("INSERT INTO NVTX_EVENTS VALUES (?, 59, ?)", (0, _pack_event(100, payload)))
    conn.commit()
    rows = SKILL.execute_fn(conn)
    data = {r["schema_id"]: r for r in rows[1:]}
    assert data[100]["distinct_communicators"] == 2


def test_summary_total_distinct_communicators_is_cross_schema():
    """`distinct_communicators` per row is a per-schema view; we also need a
    cross-schema rollup so users can answer 'is this a single-comm or
    multi-comm setup?'. On a fresh fixture both schemas share the same
    comm so the cross-schema total is 1."""
    conn = _build_db_with_nccl_payloads()
    rows = SKILL.execute_fn(conn)
    assert rows[0]["total_distinct_communicators"] == 1

    # Now add a second-comm event on schema 200 (p2p). Per-schema counts
    # update; cross-schema rollup should bump to 2.
    c = conn.cursor()
    payload = struct.pack("<QQI", 0xDEAD_FACE_F00D_C0DE, 4096, 1) + b"\x00\x00\x00\x00"
    c.execute("INSERT INTO NVTX_EVENTS VALUES (?, 59, ?)", (0, _pack_event(200, payload)))
    conn.commit()

    rows2 = SKILL.execute_fn(conn)
    assert rows2[0]["total_distinct_communicators"] == 2, (
        f"expected cross-schema 2 comm IDs; got {rows2[0]['total_distinct_communicators']}"
    )


def test_p2p_peer_rank_distribution_uniform():
    """When all peer ranks see equal traffic share (the all-to-all common
    case), `peer_rank_uniformity` should be 1.0."""
    conn = _build_db_with_nccl_payloads()
    # Add 4 more p2p events with distinct peer ranks 0,1,2,3 — uniform 50/50.
    # Existing fixture already has peers [1, 2] one each. Add peers [0, 3]
    # so all four ranks see exactly one event each (perfectly uniform).
    c = conn.cursor()
    comm = 0xCAFE_BABE_DEAD_BEEF
    for peer in (0, 3):
        payload = struct.pack("<QQI", comm, 8192, peer) + b"\x00\x00\x00\x00"
        c.execute("INSERT INTO NVTX_EVENTS VALUES (?, 59, ?)", (0, _pack_event(200, payload)))
    conn.commit()

    rows = SKILL.execute_fn(conn)
    p2p_row = next(r for r in rows[1:] if r["schema_id"] == 200)
    assert p2p_row["distinct_peer_ranks"] == 4
    assert p2p_row["peer_rank_uniformity"] == 1.0, (
        f"4 peers × 1 event each → uniform; got {p2p_row['peer_rank_uniformity']}"
    )
    assert p2p_row["peer_rank_distribution"] == {0: 1, 1: 1, 2: 1, 3: 1}


def test_p2p_peer_rank_distribution_skewed():
    """A skewed distribution (one peer dominates) gets uniformity < 1.0.
    Catches future regressions where the score computation might lock to
    1.0 for any non-empty distribution."""
    conn = _build_db_with_nccl_payloads()
    c = conn.cursor()
    comm = 0xCAFE_BABE_DEAD_BEEF
    # Pile 8 more events all on peer rank 1 (existing fixture: 1 each on
    # peer 1 and 2). After: peer 1 has 9, peer 2 has 1 — highly skewed.
    for _ in range(8):
        payload = struct.pack("<QQI", comm, 8192, 1) + b"\x00\x00\x00\x00"
        c.execute("INSERT INTO NVTX_EVENTS VALUES (?, 59, ?)", (0, _pack_event(200, payload)))
    conn.commit()

    rows = SKILL.execute_fn(conn)
    p2p_row = next(r for r in rows[1:] if r["schema_id"] == 200)
    assert p2p_row["distinct_peer_ranks"] == 2
    assert p2p_row["peer_rank_uniformity"] < 1.0, (
        f"skewed distribution should not score 1.0; got {p2p_row['peer_rank_uniformity']}"
    )


def test_collective_schemas_have_no_peer_rank_fields():
    """Non-p2p schemas (no Peer rank in field_set) must not emit
    peer-rank fields. Locks in compactness for init/collective rows."""
    conn = _build_db_with_nccl_payloads()
    rows = SKILL.execute_fn(conn)
    collective_row = next(r for r in rows[1:] if r["schema_id"] == 100)
    assert "distinct_peer_ranks" not in collective_row
    assert "peer_rank_uniformity" not in collective_row
    assert "peer_rank_distribution" not in collective_row
