"""Every execute_fn table dependency must use the abstention contract."""

import sqlite3

import pytest

from nsys_ai.skills.base import is_abstention
from nsys_ai.skills.registry import get_skill


@pytest.mark.parametrize(
    "skill_name",
    ["sync_cost_analysis", "host_sync_parent_ranges"],
)
def test_missing_required_execute_fn_table_abstains(skill_name):
    conn = sqlite3.connect(":memory:")

    rows = get_skill(skill_name).execute(conn)

    assert is_abstention(rows), f"{skill_name} returned data: {rows[:1]}"
    assert rows[0]["missing_tables"]
    assert "no" in rows[0]["reason"].lower()


def test_sync_cost_analysis_requires_sync_type_table():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION "
        "(start INTEGER, [end] INTEGER, syncType INTEGER)"
    )

    rows = get_skill("sync_cost_analysis").execute(conn)

    assert is_abstention(rows)
    assert rows[0]["missing_tables"] == ["ENUM_CUPTI_SYNC_TYPE"]


def test_pipeline_bubble_keeps_partial_result_without_memset(minimal_nsys_conn):
    """Memset is optional: kernels and memcpy still produce a useful metric."""
    minimal_nsys_conn.execute("DROP TABLE CUPTI_ACTIVITY_KIND_MEMSET")

    rows = get_skill("pipeline_bubble_metrics").execute(minimal_nsys_conn)

    assert rows
    assert not is_abstention(rows)
    assert "bubble_pct" in rows[0]


def test_pipeline_bubble_does_not_fabricate_sync_zero(minimal_nsys_conn):
    minimal_nsys_conn.execute(
        "DROP TABLE IF EXISTS CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"
    )
    minimal_nsys_conn.execute("DROP TABLE IF EXISTS ENUM_CUPTI_SYNC_TYPE")

    rows = get_skill("pipeline_bubble_metrics").execute(minimal_nsys_conn)

    assert rows
    assert "sync_ms" not in rows[0]


def test_gc_keeps_runtime_result_without_nvtx():
    """NVTX enrichment is optional; runtime GC rows remain answerable."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            correlationId INTEGER, start INTEGER, [end] INTEGER, nameId INTEGER
        );
        INSERT INTO StringIds VALUES (1, 'cudaFree');
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1, 100, 200, 1);
        """
    )

    rows = get_skill("gc_impact").execute(conn)

    assert not is_abstention(rows)
    assert rows[0]["event_name"] == "cudaFree"


def test_root_cause_memset_checker_abstains_without_required_tables():
    from nsys_ai.skills.builtins.root_cause_matcher import _check_sync_memset

    rows = _check_sync_memset(sqlite3.connect(":memory:"))

    assert is_abstention(rows)
    assert "CUPTI_ACTIVITY_KIND_MEMSET" in rows[0]["missing_tables"]
