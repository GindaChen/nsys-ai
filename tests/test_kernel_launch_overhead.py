"""Focused contracts for structured small-kernel launch findings."""

import json
import sqlite3

import pytest

from nsys_ai.annotation import Finding, TraceSelection
from nsys_ai.skills.builtins.kernel_launch_overhead import SKILL
from nsys_ai.skills.builtins.root_cause_matcher import _small_kernel_launch_summary


def _connection(*, with_kernel=True, with_runtime=True, with_strings=True):
    conn = sqlite3.connect(":memory:")
    if with_strings:
        conn.executescript(
            """
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
            INSERT INTO StringIds VALUES
                (1, 'a_tiny_kernel'), (2, 'b_tiny_kernel'),
                (3, 'large_kernel'), (10, 'cudaLaunchKernel'),
                (11, 'cudaDeviceSynchronize');
            """
        )
    if with_kernel:
        conn.execute(
            'CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL('
            'globalPid INTEGER, deviceId INTEGER, streamId INTEGER, '
            'correlationId INTEGER, start INTEGER, "end" INTEGER, shortName INTEGER)'
        )
    if with_runtime:
        conn.execute(
            'CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME('
            'globalTid INTEGER, correlationId INTEGER, start INTEGER, '
            '"end" INTEGER, nameId INTEGER)'
        )
    return conn


def _insert_launch(
    conn,
    *,
    process=0x100000000,
    thread=7,
    device=0,
    stream=1,
    correlation=1,
    kernel_start=10_000,
    kernel_duration=5_000,
    api_start=8_000,
    api_duration=1_000,
    kernel_name_id=1,
    api_name_id=10,
):
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
        (
            process, device, stream, correlation, kernel_start,
            kernel_start + kernel_duration, kernel_name_id,
        ),
    )
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
        (process | thread, correlation, api_start, api_start + api_duration, api_name_id),
    )


def _run(conn, **kwargs):
    return SKILL.execute_fn(conn, min_launches=1, device=0, **kwargs)


def test_uses_api_duration_and_counts_only_strictly_positive_queue_time():
    conn = _connection()
    _insert_launch(conn, kernel_start=10_000, api_start=8_000, api_duration=1_000)
    _insert_launch(
        conn, correlation=2, kernel_start=20_000, api_start=19_000,
        api_duration=1_000,
    )
    _insert_launch(
        conn, correlation=3, kernel_start=30_000, api_start=29_500,
        api_duration=1_000,
    )

    rows = _run(conn)

    assert rows[0]["launch_count"] == 3
    assert rows[0]["total_api_ms"] == pytest.approx(0.003)
    assert rows[0]["avg_api_us"] == pytest.approx(1.0)
    assert rows[0]["queue_count"] == 1
    assert rows[0]["total_queue_ms"] == pytest.approx(0.001)
    assert rows[0]["avg_queue_us"] == pytest.approx(1.0)


def test_device_filter_prevents_multi_gpu_cross_counting():
    conn = _connection()
    _insert_launch(conn, device=0, correlation=7)
    _insert_launch(conn, device=1, stream=2, correlation=7)

    rows = _run(conn)

    assert len(rows) == 1
    assert rows[0]["device_id"] == 0
    assert rows[0]["launch_count"] == 1


def test_process_identity_prevents_correlation_id_cross_join():
    conn = _connection()
    _insert_launch(conn, process=0x100000000, correlation=7, api_duration=1_000)
    _insert_launch(
        conn, process=0x200000000, thread=9, stream=2, correlation=7,
        kernel_start=30_000, api_start=27_000, api_duration=2_000,
    )

    rows = _run(conn)

    assert rows[0]["launch_count"] == 2
    assert rows[0]["total_api_ms"] == pytest.approx(0.003)


def test_negative_signed_global_tid_uses_official_process_mask():
    conn = _connection()
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
        (-4_294_967_296, 0, 1, 7, 10_000, 15_000, 1),
    )
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
        (-4_294_967_289, 7, 8_000, 9_000, 10),
    )

    rows = _run(conn)

    assert rows[0]["launch_count"] == 1
    assert rows[0]["total_api_ms"] == pytest.approx(0.001)


def test_compact_unencoded_ids_do_not_bypass_process_mask():
    conn = _connection()
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
        (100, 0, 1, 7, 10_000, 15_000, 1),
    )
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
        (100, 7, 8_000, 9_000, 10),
    )

    assert _run(conn) == []


def test_equal_timestamp_launches_remain_distinct():
    conn = _connection()
    _insert_launch(conn, stream=1, correlation=1)
    _insert_launch(conn, stream=2, correlation=2)

    rows = _run(conn)

    assert rows[0]["launch_count"] == 2


def test_reused_correlation_id_uses_nearest_preceding_launch():
    conn = _connection()
    _insert_launch(conn, correlation=7, api_start=8_000, api_duration=1_000)
    _insert_launch(
        conn, correlation=7, stream=2, kernel_start=30_000,
        api_start=27_000, api_duration=2_000,
    )

    rows = _run(conn)

    assert rows[0]["launch_count"] == 2
    assert rows[0]["total_api_ms"] == pytest.approx(0.003)


def test_nonlaunch_runtime_row_cannot_steal_match():
    conn = _connection()
    _insert_launch(conn, correlation=7, api_start=8_000, api_duration=1_000)
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
        (0x100000007, 7, 9_100, 9_500, 11),
    )

    rows = _run(conn)

    assert rows[0]["launch_count"] == 1
    assert rows[0]["avg_api_us"] == pytest.approx(1.0)


def test_duckdb_execution_path():
    import duckdb

    conn = duckdb.connect()
    conn.execute("CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value VARCHAR)")
    conn.execute("INSERT INTO StringIds VALUES (1, 'tiny'), (10, 'cudaLaunchKernel')")
    conn.execute(
        'CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL('
        'globalPid BIGINT, deviceId INTEGER, streamId INTEGER, '
        'correlationId INTEGER, start BIGINT, "end" BIGINT, shortName INTEGER)'
    )
    conn.execute(
        'CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME('
        'globalTid BIGINT, correlationId INTEGER, start BIGINT, '
        '"end" BIGINT, nameId INTEGER)'
    )
    _insert_launch(conn)

    rows = SKILL.execute_fn(conn, min_launches=1, device=0)

    assert rows
    assert rows[0]["avg_kernel_us"] == pytest.approx(5.0)
    conn.close()


@pytest.mark.parametrize(
    ("with_kernel", "with_runtime", "with_strings", "missing"),
    [
        (False, True, True, "CUPTI_ACTIVITY_KIND_KERNEL"),
        (True, False, True, "CUPTI_ACTIVITY_KIND_RUNTIME"),
        (True, True, False, "StringIds"),
    ],
)
def test_missing_required_table_abstains(
    with_kernel, with_runtime, with_strings, missing,
):
    conn = _connection(
        with_kernel=with_kernel,
        with_runtime=with_runtime,
        with_strings=with_strings,
    )

    rows = _run(conn)

    assert rows[0]["_abstained"] is True
    assert missing in rows[0]["missing_tables"]


def test_missing_process_identity_column_abstains():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE StringIds(id INTEGER, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
            deviceId INTEGER, streamId INTEGER, correlationId INTEGER,
            start INTEGER, "end" INTEGER, shortName INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
            globalTid INTEGER, correlationId INTEGER, start INTEGER,
            "end" INTEGER, nameId INTEGER
        );
        """
    )

    rows = _run(conn)

    assert rows[0]["_abstained"] is True
    assert rows[0]["missing_columns"]["CUPTI_ACTIVITY_KIND_KERNEL"] == ["globalPid"]


def test_candidate_filter_runs_before_limit():
    conn = _connection()
    _insert_launch(
        conn, correlation=1, kernel_name_id=3, kernel_duration=50_000,
        api_duration=50_000,
    )
    _insert_launch(
        conn, correlation=2, kernel_name_id=1, kernel_duration=5_000,
        api_duration=1_000,
    )

    rows = _run(conn, limit=1)

    assert [row["kernel_name"] for row in rows] == ["a_tiny_kernel"]


def test_ranking_uses_api_cost_not_queue_time():
    conn = _connection()
    _insert_launch(
        conn, correlation=1, kernel_name_id=1, kernel_start=20_000,
        api_start=18_000, api_duration=1_500,
    )
    _insert_launch(
        conn, correlation=2, kernel_name_id=2, kernel_start=1_000_000,
        api_start=1_000, api_duration=1_000,
    )

    rows = _run(conn, limit=1)

    assert rows[0]["kernel_name"] == "a_tiny_kernel"
    assert rows[0]["avg_queue_us"] < 10


def test_equal_ranking_is_deterministic_by_kernel_name():
    conn = _connection()
    _insert_launch(conn, correlation=1, kernel_name_id=2)
    _insert_launch(conn, correlation=2, kernel_name_id=1, kernel_start=20_000)

    first = _run(conn, limit=2)
    second = _run(conn, limit=2)

    assert [row["kernel_name"] for row in first] == ["a_tiny_kernel", "b_tiny_kernel"]
    assert first == second


def test_trim_bounds_do_not_require_profile_discovery(monkeypatch):
    from nsys_ai.profile import Profile

    conn = _connection()
    _insert_launch(conn)
    monkeypatch.setattr(
        Profile,
        "_from_conn",
        classmethod(lambda cls, conn: pytest.fail("Profile discovery must not run")),
    )

    rows = _run(conn, trim_start_ns=0, trim_end_ns=100_000)

    assert rows[0]["span_start_ns"] == 10_000
    assert rows[0]["span_end_ns"] == 15_000


def _finding_row(**overrides):
    row = {
        "kernel_name": "tiny_op",
        "device_id": 3,
        "launch_count": 1000,
        "total_api_ms": 3.0,
        "avg_api_us": 3.0,
        "max_api_us": 5.0,
        "queue_count": 1000,
        "total_queue_ms": 20.0,
        "avg_queue_us": 20.0,
        "total_kernel_ms": 5.0,
        "avg_kernel_us": 5.0,
        "span_start_ns": 100,
        "span_end_ns": 200,
    }
    row.update(overrides)
    return row


def test_finding_has_valid_highlight_semantics_and_scoped_actions():
    finding = SKILL.to_findings_fn(
        [_finding_row()], context={"profile_id": "profile"},
    )[0]

    assert finding.type == "highlight"
    assert finding.start_ns == 0
    assert finding.end_ns is None
    assert finding.category == "launch_overhead"
    assert finding.gpu_id == 3
    assert finding.selection.start_ns == 100
    assert finding.selection.end_ns == 200
    assert finding.selection.gpu_ids == [3]
    assert all("sync" not in action.lower() for action in finding.suggested_actions)
    assert finding.evidence[0].values["avg_api_us"] == 3.0
    assert finding.evidence[0].values["queue_count"] == 1000
    assert finding.evidence[0].values["avg_queue_us"] == 20.0
    assert finding.evidence[0].values["total_queue_ms"] == 20.0


def test_finding_json_round_trip():
    finding = SKILL.to_findings_fn(
        [_finding_row()], context={"profile_id": "profile"},
    )[0]

    restored = Finding.from_dict(json.loads(json.dumps(finding.to_dict())))

    assert restored.category == "launch_overhead"
    assert restored.type == "highlight"
    assert isinstance(restored.selection, TraceSelection)


def test_matcher_counts_aggregated_launch_occurrences_and_checks_average():
    occurrences, kernel_types = _small_kernel_launch_summary([
        {"launch_count": 100_000, "total_kernel_ms": 500.0},
        {"launch_count": 10_000, "total_kernel_ms": 200.0},
    ])

    assert occurrences == 100_000
    assert kernel_types == 1


def test_matcher_accepts_explicit_avg_kernel_metric():
    assert _small_kernel_launch_summary([
        {"launch_count": 6, "avg_kernel_us": 9.999},
        {"launch_count": 100, "avg_kernel_us": 10.0},
    ]) == (6, 1)


def test_skill_is_enrolled_in_evidence_pipeline():
    from nsys_ai.evidence_builder import EvidenceBuilder

    assert EvidenceBuilder._SKILL_PIPELINE["kernel_launch_overhead"] == (
        "kernel_launch_overhead", {}
    )
