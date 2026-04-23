import sqlite3

import pytest

from nsys_ai.skills.builtins.sync_cost_analysis import _sync_result_cache
from nsys_ai.skills.registry import get_skill


@pytest.fixture(autouse=True)
def _clear_sync_cache():
    """Prevent id(conn) reuse of in-memory connections from poisoning results."""
    _sync_result_cache.clear()


@pytest.fixture
def sync_skill():
    skill = get_skill("sync_cost_analysis")
    assert skill is not None, "sync_cost_analysis skill not registered"
    return skill


def test_sync_analysis_without_tables(sync_skill):
    conn = sqlite3.connect(":memory:")

    rows = sync_skill.execute(conn)
    assert len(rows) == 1
    assert "error" in rows[0]
    assert "not found" in rows[0]["error"]


def test_sync_analysis_math(sync_skill):
    conn = sqlite3.connect(":memory:")

    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION (start INTEGER, [end] INTEGER, globalPid INTEGER, syncType INTEGER)"
    )
    conn.execute("CREATE TABLE ENUM_CUPTI_SYNC_TYPE (id INTEGER, name TEXT)")

    conn.execute("INSERT INTO ENUM_CUPTI_SYNC_TYPE VALUES (1, 'Event sync'), (2, 'Stream sync')")

    # Thread 1 does Event Sync 100-200 ms
    # Thread 2 does Event Sync 150-300 ms (Overlap union logic should yield 100-300 = 200ms)
    # Thread 1 does Stream Sync 500-600 ms
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES (100000000, 200000000, 1, 1)"
    )
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES (150000000, 300000000, 2, 1)"
    )
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES (500000000, 600000000, 1, 2)"
    )

    rows = sync_skill.execute(conn)
    assert len(rows) == 1
    m = rows[0]

    # Global Union: [100, 300] and [500, 600] -> 200 + 100 = 300ms total
    assert m["total_sync_wall_ms"] == 300.0

    # Event Union: [100, 300] -> 200ms
    # Stream Union: [500, 600] -> 100ms
    assert m["sync_by_type_ms"]["Event sync"] == 200.0
    assert m["sync_by_type_ms"]["Stream sync"] == 100.0


def test_sync_analysis_trimming(sync_skill):
    conn = sqlite3.connect(":memory:")

    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION (start INTEGER, [end] INTEGER, globalPid INTEGER, syncType INTEGER)"
    )
    conn.execute("CREATE TABLE ENUM_CUPTI_SYNC_TYPE (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO ENUM_CUPTI_SYNC_TYPE VALUES (1, 'Event sync')")

    # [100, 300] ms event
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES (100000000, 300000000, 1, 1)"
    )

    # Trim to [200, 400] ms
    # Expectation: start is clamped to 200, end remains 300 -> 100ms duration
    rows = sync_skill.execute(conn, trim_start_ns=200000000, trim_end_ns=400000000)
    m = rows[0]

    assert m["total_sync_wall_ms"] == 100.0
    assert m["profile_span_ms"] == 200.0
    assert m["sync_density_pct"] == 50.0


# ---------------------------------------------------------------------------
# Per-device aggregation (CUPTI_ACTIVITY_KIND_SYNCHRONIZATION.deviceId)
# ---------------------------------------------------------------------------


def _seed_multi_device(conn):
    """Seed a sync table with two devices of unequal sync load (rank-asymmetry case)."""
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION "
        "(start INTEGER, [end] INTEGER, deviceId INTEGER, globalPid INTEGER, syncType INTEGER)"
    )
    conn.execute("CREATE TABLE ENUM_CUPTI_SYNC_TYPE (id INTEGER, name TEXT)")
    conn.execute(
        "INSERT INTO ENUM_CUPTI_SYNC_TYPE VALUES (1, 'Event sync'), (3, 'Stream sync')"
    )
    # Device 0: 300ms stream sync + 100ms event sync = 400ms total (union)
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES "
        "(100000000, 400000000, 0, 1, 3),"  # stream sync device 0: 300ms
        "(500000000, 600000000, 0, 1, 1)"   # event sync device 0: 100ms
    )
    # Device 1: 50ms event sync only — rank asymmetry
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES "
        "(100000000, 150000000, 1, 2, 1)"   # event sync device 1: 50ms
    )


def test_sync_analysis_per_device_breakdown(sync_skill):
    conn = sqlite3.connect(":memory:")
    _seed_multi_device(conn)

    rows = sync_skill.execute(
        conn, trim_start_ns=0, trim_end_ns=1_000_000_000
    )
    m = rows[0]
    assert m["sync_by_device"] is not None, "sync_by_device must appear when deviceId present"
    by_dev = {d["device"]: d for d in m["sync_by_device"]}
    assert set(by_dev) == {0, 1}
    assert by_dev[0]["total_sync_wall_ms"] == 400.0
    assert by_dev[1]["total_sync_wall_ms"] == 50.0
    # sync_by_type_ms per device
    assert by_dev[0]["sync_by_type_ms"]["Stream sync"] == 300.0
    assert by_dev[0]["sync_by_type_ms"]["Event sync"] == 100.0
    assert by_dev[1]["sync_by_type_ms"]["Event sync"] == 50.0
    # density should reflect 1000ms span
    assert by_dev[0]["sync_density_pct"] == 40.0
    assert by_dev[1]["sync_density_pct"] == 5.0


def test_sync_analysis_device_filter(sync_skill):
    conn = sqlite3.connect(":memory:")
    _seed_multi_device(conn)

    rows = sync_skill.execute(
        conn, trim_start_ns=0, trim_end_ns=1_000_000_000, device=1
    )
    m = rows[0]
    assert m["total_sync_wall_ms"] == 50.0
    # sync_by_device still present but filtered to only device 1
    by_dev = {d["device"]: d for d in m["sync_by_device"]}
    assert set(by_dev) == {1}
    assert by_dev[1]["total_sync_wall_ms"] == 50.0


def test_sync_analysis_legacy_schema_without_deviceid(sync_skill):
    """Old Nsight exports lack deviceId; skill must fall back, not crash."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION "
        "(start INTEGER, [end] INTEGER, globalPid INTEGER, syncType INTEGER)"
    )
    conn.execute("CREATE TABLE ENUM_CUPTI_SYNC_TYPE (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO ENUM_CUPTI_SYNC_TYPE VALUES (1, 'Event sync')")
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES (0, 100000000, 1, 1)"
    )
    rows = sync_skill.execute(conn)
    m = rows[0]
    assert m["total_sync_wall_ms"] == 100.0
    assert m["sync_by_device"] is None, "legacy schema: sync_by_device must be null"


def test_sync_analysis_device_filter_on_legacy_schema_errors(sync_skill):
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION "
        "(start INTEGER, [end] INTEGER, globalPid INTEGER, syncType INTEGER)"
    )
    conn.execute("CREATE TABLE ENUM_CUPTI_SYNC_TYPE (id INTEGER, name TEXT)")
    rows = sync_skill.execute(conn, device=0)
    assert "error" in rows[0]
    assert "deviceId" in rows[0]["error"]


def test_sync_analysis_invalid_device_param(sync_skill):
    conn = sqlite3.connect(":memory:")
    _seed_multi_device(conn)
    rows = sync_skill.execute(conn, device="not-a-number")
    assert "error" in rows[0]
    assert "device" in rows[0]["error"].lower()


def test_sync_analysis_device_filter_on_missing_tables_reports_not_found(sync_skill):
    """Regression: if sync tables are absent entirely, supplying `device=N`
    must still report 'Synchronization tables not found', not 'no deviceId column'.
    The latter message is misleading when the table itself is missing."""
    conn = sqlite3.connect(":memory:")
    rows = sync_skill.execute(conn, device=0)
    assert "error" in rows[0]
    assert "not found" in rows[0]["error"].lower()
    assert "deviceid" not in rows[0]["error"].lower()
