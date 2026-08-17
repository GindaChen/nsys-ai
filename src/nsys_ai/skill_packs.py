"""Canonical skill packs shared by the analysis entry points.

The packs describe which registered skills each entry point runs.  Execution
still belongs to :mod:`nsys_ai.skills.registry`; this module only owns the
selection and ordering so CLI, evidence generation, and ask fallback cannot
silently drift apart.
"""

from __future__ import annotations

from typing import Final

# The default Agent.analyze roster.  Keep this order stable: the textual
# report and downstream evidence consumers use it as a deterministic order.
DIAGNOSE_DEFAULT: Final[list[str]] = [
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


# Analyzer name -> (registered skill name, skill parameters) used to produce
# timeline findings.  Analyzer names are part of the EvidenceReport contract.
EVIDENCE_OVERLAY: Final[dict[str, tuple[str, dict]]] = {
    "slow_iterations": ("iteration_timing", {}),
    "idle_gaps": ("gpu_idle_gaps", {"limit": 5, "min_gap_ns": 1000000}),
    "nccl_stalls": ("kernel_instances", {"name": "nccl", "limit": 3}),
    "kernel_hotspots": ("kernel_instances", {"limit": 3}),
    "top_kernel_aggregates": ("top_kernels", {"limit": 15}),
    "overlap_ratio": ("overlap_breakdown", {}),
    "memory_anomalies": ("memory_bandwidth", {"limit": 5}),
    "h2d_spikes": ("h2d_distribution", {}),
    "kernel_launch_overhead": ("kernel_launch_overhead", {}),
    "nccl_breakdown": ("nccl_breakdown", {}),
    "nccl_compile_context_breakdown": ("nccl_compile_context_breakdown", {}),
    # Profile-level bound class.  It contributes the verdict but reports no
    # headroom by design.
    "bound_class": ("critical_path", {}),
    # Roll-up characterization of the whole profile.
    "profile_health": ("profile_health_manifest", {}),
}


# Used when triage and keyword selection produce no usable skill list.
ASK_FALLBACK: Final[list[str]] = ["top_kernels", "gpu_idle_gaps"]


# Deterministic fallback routing for questions without an LLM triage result.
ASK_KEYWORD_MAP: Final[dict[str, list[str]]] = {
    "kernel": ["top_kernels", "kernel_launch_overhead"],
    "hotspot": ["top_kernels"],
    "slow": ["top_kernels", "gpu_idle_gaps"],
    "bubble": ["gpu_idle_gaps"],
    "idle": ["gpu_idle_gaps"],
    "gap": ["gpu_idle_gaps"],
    "stall": ["gpu_idle_gaps", "nccl_anomaly"],
    "memory": ["memory_transfers", "memory_bandwidth"],
    "transfer": ["memory_transfers", "memory_bandwidth"],
    "h2d": ["memory_transfers", "memory_bandwidth"],
    "copy": ["memory_transfers", "memory_bandwidth"],
    "bandwidth": ["memory_bandwidth"],
    "nccl": [
        "nccl_breakdown",
        "nccl_communicator_analysis",
        "overlap_breakdown",
        "kernel_overlap_matrix",
        "nccl_anomaly",
    ],
    "allreduce": ["nccl_breakdown", "nccl_communicator_analysis", "nccl_anomaly"],
    "collective": ["nccl_breakdown", "nccl_communicator_analysis", "nccl_anomaly"],
    "distributed": [
        "nccl_breakdown",
        "nccl_communicator_analysis",
        "overlap_breakdown",
        "kernel_overlap_matrix",
        "nccl_anomaly",
    ],
    "multi-gpu": [
        "nccl_breakdown",
        "nccl_communicator_analysis",
        "overlap_breakdown",
        "kernel_overlap_matrix",
    ],
    "communicator": ["nccl_communicator_analysis", "nccl_breakdown"],
    "rank": ["nccl_communicator_analysis", "nccl_breakdown"],
    "tensor parallel": ["nccl_communicator_analysis", "nccl_breakdown"],
    "pipeline parallel": ["nccl_communicator_analysis", "nccl_breakdown"],
    "data parallel": ["nccl_communicator_analysis", "nccl_breakdown"],
    "anomaly": ["nccl_anomaly"],
    "outlier": ["nccl_anomaly"],
    "overlap": ["overlap_breakdown", "kernel_overlap_matrix"],
    "matrix": ["kernel_overlap_matrix"],
    "contention": ["kernel_overlap_matrix", "stream_concurrency"],
    "hidden": ["kernel_overlap_matrix", "overlap_breakdown"],
    "nvtx": ["nvtx_kernel_map", "nvtx_layer_breakdown"],
    "source": ["nvtx_kernel_map"],
    "attribution": ["nvtx_kernel_map"],
    "mapping": ["nvtx_kernel_map"],
    "layer": ["nvtx_layer_breakdown"],
    "launch": ["kernel_launch_overhead", "kernel_launch_pattern"],
    "overhead": ["kernel_launch_overhead"],
    "dispatch": ["kernel_launch_pattern"],
    "pattern": ["kernel_launch_pattern"],
    "burst": ["kernel_launch_pattern"],
    "stream": ["stream_concurrency"],
    "concurrency": ["stream_concurrency"],
    "parallel": ["stream_concurrency"],
    "serial": ["stream_concurrency"],
    "cpu": ["thread_utilization", "cpu_gpu_pipeline"],
    "thread": ["thread_utilization"],
    "utilization": ["thread_utilization", "stream_concurrency"],
    "pipeline": ["cpu_gpu_pipeline"],
    "starvation": ["cpu_gpu_pipeline"],
    "queue": ["cpu_gpu_pipeline"],
    "schema": ["schema_inspect"],
    "table": ["schema_inspect"],
    "mfu": ["region_mfu", "theoretical_flops"],
    "flops": ["theoretical_flops"],
    "efficiency": ["region_mfu"],
    "iteration": ["iteration_timing"],
    "iter": ["iteration_timing"],
    "training": ["iteration_timing"],
    "step": ["iteration_timing"],
    "diagnosis": ["root_cause_matcher"],
    "root-cause": ["root_cause_matcher"],
    "why": ["root_cause_matcher"],
    "speedup": ["speedup_estimator"],
    "estimate": ["speedup_estimator"],
    "projection": ["speedup_estimator"],
}


__all__ = [
    "ASK_FALLBACK",
    "ASK_KEYWORD_MAP",
    "DIAGNOSE_DEFAULT",
    "EVIDENCE_OVERLAY",
]
