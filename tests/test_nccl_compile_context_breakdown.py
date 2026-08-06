"""Tests for nccl_compile_context_breakdown skill.

Verifies the classification of NCCL kernels by their **leaf** NVTX label
into eager / inductor_captured / temporal_only buckets — the lesson from
§4 B audit gap #14 (NVTX path containment is temporal, not lexical).
"""

import sqlite3
from types import SimpleNamespace

import pytest

# Minimal Nsight-like schema for in-memory tests (mirrors tests/conftest.py).
_SCHEMA_SQL = """
CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE TARGET_INFO_GPU (
    id INTEGER PRIMARY KEY, name TEXT, busLocation TEXT DEFAULT '',
    totalMemory INTEGER DEFAULT 0, smCount INTEGER DEFAULT 0,
    chipName TEXT DEFAULT '', memoryBandwidth INTEGER DEFAULT 0
);
CREATE TABLE TARGET_INFO_CUDA_DEVICE (
    gpuId INTEGER, cudaId INTEGER, pid INTEGER DEFAULT 0,
    uuid TEXT DEFAULT '', numMultiprocessors INTEGER DEFAULT 0
);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
    globalPid INTEGER DEFAULT 0, deviceId INTEGER DEFAULT 0,
    streamId INTEGER DEFAULT 0, correlationId INTEGER DEFAULT 0,
    start INTEGER NOT NULL, end INTEGER NOT NULL,
    shortName INTEGER NOT NULL, demangledName INTEGER DEFAULT 0,
    gridX INTEGER DEFAULT 1, gridY INTEGER DEFAULT 1, gridZ INTEGER DEFAULT 1,
    blockX INTEGER DEFAULT 1, blockY INTEGER DEFAULT 1, blockZ INTEGER DEFAULT 1
);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
    globalTid INTEGER DEFAULT 0, correlationId INTEGER DEFAULT 0,
    start INTEGER NOT NULL, end INTEGER NOT NULL, nameId INTEGER DEFAULT 0
);
CREATE TABLE NVTX_EVENTS (
    globalTid INTEGER DEFAULT 0, start INTEGER NOT NULL,
    end INTEGER DEFAULT -1, text TEXT DEFAULT '',
    eventType INTEGER DEFAULT 59, rangeId INTEGER DEFAULT 0,
    textId INTEGER DEFAULT NULL
);
"""

# Three NCCL kernels each under a distinct innermost NVTX scope:
#   kernel A (1_000_000 ns)  inside leaf "c10d::all_reduce"
#   kernel B (4_000_000 ns)  inside leaf "## Call CompiledFxGraph"
#   kernel C (10_000_000 ns) inside leaf "backward"     (no deeper scope)
# Plus one non-NCCL compute kernel under the eager leaf — must be filtered out.
_SEED_SQL = """
INSERT INTO StringIds VALUES
    (10, 'ncclKernel_AllReduce_RING'),
    (20, 'sm80_gemm_f16f16'),
    (24, 'cudaLaunchKernel');

INSERT INTO TARGET_INFO_GPU VALUES
    (0, 'NVIDIA Test', '', 8589934592, 108, 'TestChip', 0);
INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (0, 0, 100, '', 108);

INSERT INTO NVTX_EVENTS (globalTid, start, end, text, eventType, rangeId) VALUES
    (100,  0,        100000000, 'iteration',                  59, 0),
    (100,  1000000,   30000000, 'forward',                    59, 1),
    (100,  5000000,    8000000, 'c10d::all_reduce',           59, 2),
    (100, 15000000,   25000000, '## Call CompiledFxGraph',    59, 3),
    (100, 40000000,   90000000, 'backward',                   59, 4);

-- NCCL kernels (shortName 10) on stream 8
INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
    (100, 0, 8,  1,  6000000,  7000000, 10, 10, 1,1,1, 512,1,1),
    (100, 0, 8,  2, 18000000, 22000000, 10, 10, 1,1,1, 512,1,1),
    (100, 0, 8,  3, 50000000, 60000000, 10, 10, 1,1,1, 512,1,1);

-- One compute kernel under the eager leaf — must NOT be counted as NCCL.
INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
    (100, 0, 7,  4,  6200000,  6800000, 20, 20, 32,1,1, 256,1,1);

-- Runtime entries tie each kernel launch to thread 100 so attribution can
-- locate the open NVTX scope at launch time.
INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
    (100, 1,  5500000,  6000000, 24),
    (100, 2, 17000000, 18000000, 24),
    (100, 3, 45000000, 50000000, 24),
    (100, 4,  5700000,  6200000, 24);
"""


@pytest.fixture
def three_bucket_conn():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA_SQL)
    conn.executescript(_SEED_SQL)
    yield conn
    conn.close()


def _classification_rows(bucket_ns):
    bucket_order = ("eager", "inductor_captured", "temporal_only")
    counts = {name: int(bucket_ns.get(name, 0) > 0) for name in bucket_order}
    total_count = sum(counts.values())
    total_ns = sum(bucket_ns.values())
    return [
        {
            "_summary": True,
            "total_nccl_kernels": total_count,
            "total_nccl_ns": total_ns,
            "total_nccl_ms": round(total_ns / 1e6, 3),
            "span_start_ns": 10,
            "span_end_ns": 20,
            "device_scope": "all_captured_devices",
        },
        *[
            {
                "bucket": name,
                "count": counts[name],
                "ns": bucket_ns.get(name, 0),
                "ms": round(bucket_ns.get(name, 0) / 1e6, 3),
                "pct": round(counts[name] / total_count * 100, 1),
                "ms_pct": round(bucket_ns.get(name, 0) / total_ns * 100, 1),
            }
            for name in bucket_order
        ],
    ]


# ---------------------------------------------------------------------------
# Unit tests on the classification helpers (no DB needed)
# ---------------------------------------------------------------------------


def test_classify_leaf_eager_prefixes():
    from nsys_ai.skills.builtins.nccl_compile_context_breakdown import _classify_leaf

    assert _classify_leaf("c10d::all_reduce") == "eager"
    assert _classify_leaf("c10d::reduce_scatter_tensor") == "eager"
    assert _classify_leaf("nccl:all_reduce") == "eager"
    assert _classify_leaf("ncclAllReduce") == "eager"


def test_classify_leaf_inductor_marker():
    from nsys_ai.skills.builtins.nccl_compile_context_breakdown import _classify_leaf

    assert _classify_leaf("## Call CompiledFxGraph") == "inductor_captured"
    # Substring also matches (the marker can appear with a suffix in real traces).
    assert _classify_leaf("## Call CompiledFxGraph forward_0") == "inductor_captured"


def test_classify_leaf_temporal_only_fallback():
    from nsys_ai.skills.builtins.nccl_compile_context_breakdown import _classify_leaf

    assert _classify_leaf("") == "temporal_only"
    assert _classify_leaf("backward") == "temporal_only"
    assert _classify_leaf("forward") == "temporal_only"
    assert _classify_leaf("Torch-Compiled Region") == "temporal_only"  # ancestor, not leaf


def test_is_nccl_kernel_filter():
    from nsys_ai.skills.builtins.nccl_compile_context_breakdown import _is_nccl_kernel

    assert _is_nccl_kernel("ncclKernel_AllReduce_RING_LL_Sum")
    assert _is_nccl_kernel("nccl_AllReduce_kernel")
    assert _is_nccl_kernel("ncclDevKernel_Generic")
    assert not _is_nccl_kernel("sm80_gemm_f16f16")
    assert not _is_nccl_kernel("")


# ---------------------------------------------------------------------------
# End-to-end skill execution on the three-bucket fixture
# ---------------------------------------------------------------------------


def test_skill_classifies_three_buckets(three_bucket_conn):
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("nccl_compile_context_breakdown")
    rows = skill.execute(three_bucket_conn)

    # rows[0] is the summary; rows[1:] are the buckets.
    summary = rows[0]
    assert summary["_summary"] is True
    assert summary["total_nccl_kernels"] == 3  # 3 NCCL kernels, compute kernel excluded

    buckets = {r["bucket"]: r for r in rows[1:]}
    assert set(buckets) == {"eager", "inductor_captured", "temporal_only"}

    # Each leaf NVTX scope contains exactly one NCCL kernel.
    assert buckets["eager"]["count"] == 1
    assert buckets["inductor_captured"]["count"] == 1
    assert buckets["temporal_only"]["count"] == 1
    assert buckets["eager"]["ns"] == 1_000_000
    assert buckets["inductor_captured"]["ns"] == 4_000_000
    assert buckets["temporal_only"]["ns"] == 10_000_000

    # Durations come from the seeded kernel spans.
    assert buckets["eager"]["ms"] == 1.0           # 6_000_000 → 7_000_000
    assert buckets["inductor_captured"]["ms"] == 4.0  # 18_000_000 → 22_000_000
    assert buckets["temporal_only"]["ms"] == 10.0     # 50_000_000 → 60_000_000

    # Percentages sum to ~100 (count-pct and ms-pct independently).
    assert sum(b["pct"] for b in buckets.values()) == pytest.approx(100.0, abs=0.3)
    assert sum(b["ms_pct"] for b in buckets.values()) == pytest.approx(100.0, abs=0.3)


def test_evidence_builder_reports_the_observed_classification(three_bucket_conn):
    from nsys_ai.evidence_builder import EvidenceBuilder

    class ProfileStub:
        path = "three-bucket.sqlite"
        conn = three_bucket_conn
        meta = SimpleNamespace(time_range=(0, 100_000_000))

        def query_conn(self):
            return self.conn

    report = EvidenceBuilder(ProfileStub()).build(only=["nccl_compile_context_breakdown"])

    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.category == "communication"
    assert finding.label == "NCCL Call Mode (dominant: temporal_only)"
    assert finding.start_ns == 6_000_000
    assert finding.end_ns == 60_000_000
    assert finding.headroom_ms is None
    assert finding.selection is not None
    assert finding.selection.profile_id == report.profile_id
    assert finding.selection.source == "skill:nccl_compile_context_breakdown"
    assert finding.selection.gpu_ids is None
    assert "all captured devices" in finding.selection.label
    assert finding.suggested_actions == [
        "For temporal-only attribution, run nccl_payload_breakdown and overlap_breakdown "
        "to identify the collective and exposed communication."
    ]
    assert any("device selection is not applied" in note for note in finding.false_positive_notes)
    assert finding.provenance["device_scope"] == "all_captured_devices"

    assert finding.evidence is not None
    evidence = finding.evidence[0]
    assert evidence.source_skill == "nccl_compile_context_breakdown"
    assert evidence.values["eager_count"] == 1
    assert evidence.values["eager_ns"] == 1_000_000
    assert evidence.values["inductor_captured_count"] == 1
    assert evidence.values["inductor_captured_ns"] == 4_000_000
    assert evidence.values["temporal_only_count"] == 1
    assert evidence.values["temporal_only_ns"] == 10_000_000
    assert evidence.values["temporal_only_ms_pct"] == pytest.approx(66.7)
    assert evidence.values["dominant_basis"] == "duration"
    assert evidence.values["device_scope"] == "all_captured_devices"
    assert evidence.provenance["classification_basis"] == "leaf_pushpop_nvtx"
    assert evidence.provenance["device_scope"] == "all_captured_devices"
    assert evidence.provenance["limitations"]


@pytest.mark.parametrize(
    ("dominant", "action_fragment"),
    [
        ("eager", "async_op=True"),
        ("inductor_captured", "functional collectives"),
        ("temporal_only", "nccl_payload_breakdown and overlap_breakdown"),
    ],
)
def test_finding_action_follows_dominant_bucket(dominant, action_fragment):
    from nsys_ai.skills.registry import get_skill

    rows = _classification_rows({dominant: 1_000_000})

    finding = get_skill("nccl_compile_context_breakdown").to_findings_fn(rows)[0]

    assert finding.suggested_actions is not None
    assert action_fragment in finding.suggested_actions[0]


def test_finding_uses_exact_ns_when_display_ms_tie():
    from nsys_ai.skills.registry import get_skill

    rows = _classification_rows({"eager": 1_000_400, "inductor_captured": 1_000_499})
    assert rows[1]["ms"] == rows[2]["ms"] == 1.0

    finding = get_skill("nccl_compile_context_breakdown").to_findings_fn(rows)[0]

    assert finding.label == "NCCL Call Mode (dominant: inductor_captured)"
    assert finding.evidence[0].values["dominant_bucket"] == "inductor_captured"
    assert finding.evidence[0].values["eager_ns"] == 1_000_400
    assert finding.evidence[0].values["inductor_captured_ns"] == 1_000_499
    assert "functional collectives" in finding.suggested_actions[0]


def test_finding_discloses_exact_tie_without_prescribing_one_bucket():
    from nsys_ai.skills.registry import get_skill

    rows = _classification_rows({"eager": 1_000_499, "inductor_captured": 1_000_499})
    finding = get_skill("nccl_compile_context_breakdown").to_findings_fn(rows)[0]
    evidence = finding.evidence[0]

    assert finding.label == "NCCL Call Mode (dominant: mixed)"
    assert evidence.values["dominant_bucket"] == "mixed"
    assert evidence.values["dominant_buckets"] == ["eager", "inductor_captured"]
    assert "tied across eager, inductor_captured" in finding.explanation
    assert "tied across eager, inductor_captured" in finding.suggested_actions[0]
    assert "async_op=True" not in finding.suggested_actions[0]
    assert "functional collectives" not in finding.suggested_actions[0]


def test_agent_analyze_renders_the_observed_classification(three_bucket_conn, monkeypatch):
    from nsys_ai.agent import loop as agent_loop
    from nsys_ai.skills.registry import get_skill

    classifier = get_skill("nccl_compile_context_breakdown")
    monkeypatch.setattr(
        agent_loop,
        "get_skill",
        lambda name: classifier if name == "nccl_compile_context_breakdown" else None,
    )

    agent = agent_loop.Agent.__new__(agent_loop.Agent)
    agent.profile = SimpleNamespace(query_conn=lambda: three_bucket_conn)
    agent._fallback_conn = None
    agent._trim_kwargs = {}
    agent._try_llm_synthesis = lambda *args, **kwargs: None

    report = agent.analyze()

    assert report.count("NCCL Call-Mode Breakdown (by leaf NVTX)") == 1
    assert "eager" in report
    assert "inductor_captured" in report
    assert "temporal_only" in report
    assert "Total NCCL kernels: 3" in report
    assert "Scope: all captured devices" in report


def test_skill_no_nccl_kernels_returns_error():
    """When the profile has no NCCL kernels, the skill returns an explicit error
    row instead of producing a misleading 0/0/0 breakdown."""
    from nsys_ai.skills.registry import get_skill

    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA_SQL)
    conn.executescript("""
        INSERT INTO StringIds VALUES (20, 'sm80_gemm_f16f16'), (24, 'cudaLaunchKernel');
        INSERT INTO TARGET_INFO_GPU VALUES (0, 'NVIDIA Test', '', 0, 108, 'TestChip', 0);
        INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (0, 0, 100, '', 108);
        INSERT INTO NVTX_EVENTS (globalTid, start, end, text, eventType, rangeId) VALUES
            (100, 0, 10000000, 'forward', 59, 0);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
            (100, 0, 7, 1, 1000000, 2000000, 20, 20, 32,1,1, 256,1,1);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
            (100, 1, 500000, 1000000, 24);
    """)
    try:
        skill = get_skill("nccl_compile_context_breakdown")
        rows = skill.execute(conn)
        assert len(rows) == 1
        assert "error" in rows[0]
        assert "No NCCL kernels" in rows[0]["error"]
    finally:
        conn.close()
