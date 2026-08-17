"""Registry boundaries for profile-backed chat tools."""

import json
import sqlite3

import pytest


def test_run_skill_can_return_raw_rows_without_changing_cli_default(minimal_nsys_conn):
    from nsys_ai.skills.registry import run_skill

    raw = run_skill("top_kernels", minimal_nsys_conn, raw=True, limit=1)
    formatted = run_skill("top_kernels", minimal_nsys_conn, limit=1)

    assert isinstance(raw, list)
    assert isinstance(formatted, str)
    assert raw


def test_profile_tool_handlers_delegate_to_registered_skills(minimal_nsys_conn, monkeypatch):
    from nsys_ai.skills import registry
    from nsys_ai.tool_dispatch import ToolDispatcher

    calls = []

    def fake_run_skill(skill_name, conn, *, raw=False, **kwargs):
        calls.append((skill_name, conn, raw, kwargs))
        if skill_name == "overlap_breakdown":
            return [{"total_ms": 12.0, "compute_only_ms": 8.0}]
        if skill_name == "nccl_breakdown":
            return [{"collective": "AllReduce", "total_ms": 4.0}]
        if skill_name == "region_mfu":
            return [{"matched_text": "train_step", "mfu_pct_kernel_union": 42.0}]
        raise AssertionError(f"unexpected skill: {skill_name}")

    monkeypatch.setattr(registry, "run_skill", fake_run_skill)
    dispatcher = ToolDispatcher(conn=minimal_nsys_conn, sqlite_path="/profile.sqlite")

    region = dispatcher.dispatch(
        "compute_region_mfu",
        json.dumps({"name": "train_step", "theoretical_flops": 1e12}),
    )
    overlap = dispatcher.dispatch("get_gpu_overlap_stats", json.dumps({}))
    nccl = dispatcher.dispatch("get_nccl_breakdown", json.dumps({"device_id": 0}))

    assert json.loads(region.content)["matched_text"] == "train_step"
    assert json.loads(overlap.content)["per_gpu"][0]["total_ms"] == 12.0
    assert json.loads(nccl.content)["collectives"][0]["collective"] == "AllReduce"
    assert [call[0] for call in calls] == [
        "region_mfu",
        "overlap_breakdown",
        "nccl_breakdown",
    ]
    assert all(call[2] is True for call in calls)
    assert calls[0][3]["profile_path"] == "/profile.sqlite"


@pytest.mark.parametrize("skill_name", ["region_mfu", "overlap_breakdown", "nccl_breakdown"])
def test_profile_skills_abstain_when_kernel_activity_is_missing(skill_name):
    from nsys_ai.skills.base import is_abstention
    from nsys_ai.skills.registry import run_skill

    conn = sqlite3.connect(":memory:")
    try:
        rows = run_skill(skill_name, conn, raw=True)
    finally:
        conn.close()

    assert is_abstention(rows)
    assert "KERNEL" in rows[0]["reason"]


def test_query_profile_db_remains_exploratory_tool(minimal_nsys_conn):
    from nsys_ai.tool_dispatch import ToolDispatcher

    dispatcher = ToolDispatcher(
        conn=minimal_nsys_conn,
        query_runner=lambda sql: f"exploratory:{sql}",
    )
    result = dispatcher.dispatch("query_profile_db", '{"sql_query":"SELECT 1"}')

    assert result.content == "exploratory:SELECT 1"


def test_submit_finding_uses_session_sink_and_preserves_confidence():
    from nsys_ai.tool_dispatch import ToolDispatcher

    persisted = []
    dispatcher = ToolDispatcher(
        finding_counter=lambda: 7,
        finding_sink=persisted.append,
        finding_provenance={
            "source": "llm",
            "model": "test-model",
            "prompt_sha256": "a" * 64,
        },
    )
    result = dispatcher.dispatch(
        "submit_finding",
        json.dumps(
            {
                "type": "region",
                "label": "NCCL stall",
                "start_ns": 10,
                "end_ns": 20,
                "severity": "warning",
                "confidence": 0.75,
            }
        ),
    )

    assert json.loads(result.content)["index"] == 7
    assert result.events[0]["finding"]["index"] == 7
    assert persisted[0]["confidence"] == 0.75
    assert persisted[0]["provenance"] == {
        "source": "llm",
        "model": "test-model",
        "prompt_sha256": "a" * 64,
    }
