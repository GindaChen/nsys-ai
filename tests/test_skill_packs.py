"""Contracts for the canonical skill-selection packs."""

from nsys_ai.agent.loop import Agent
from nsys_ai.evidence_builder import EvidenceBuilder
from nsys_ai.skill_packs import (
    ASK_FALLBACK,
    ASK_KEYWORD_MAP,
    DIAGNOSE_DEFAULT,
    EVIDENCE_OVERLAY,
)
from nsys_ai.skills.registry import list_skills


def test_canonical_packs_reference_registered_skills():
    registered = set(list_skills())
    selected = set(DIAGNOSE_DEFAULT) | set(ASK_FALLBACK)
    selected.update(skill for skills in ASK_KEYWORD_MAP.values() for skill in skills)
    selected.update(skill for skill, _params in EVIDENCE_OVERLAY.values())

    assert selected <= registered, sorted(selected - registered)


def test_consumers_use_the_canonical_pack_objects():
    assert Agent._KEYWORD_MAP is ASK_KEYWORD_MAP
    assert EvidenceBuilder._SKILL_PIPELINE is EVIDENCE_OVERLAY


def test_default_packs_have_stable_order_and_expected_fallback():
    assert DIAGNOSE_DEFAULT == [
        "top_kernels",
        "gpu_idle_gaps",
        "memory_transfers",
        "memory_bandwidth",
        "nccl_breakdown",
        "nccl_communicator_analysis",
        "nccl_anomaly",
        "kernel_launch_overhead",
        "kernel_launch_pattern",
        "stream_concurrency",
        "overlap_breakdown",
        "kernel_overlap_matrix",
        "iteration_timing",
        "nvtx_layer_breakdown",
        "nccl_compile_context_breakdown",
    ]
    assert ASK_FALLBACK == ["top_kernels", "gpu_idle_gaps"]
