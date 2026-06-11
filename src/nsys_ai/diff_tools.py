"""
diff_tools.py — Phase C spatio-temporal retrieval toolchain for profile diff.

Provides discovery and slicing tools so an AI agent (or power user) can answer
fine-grained questions like "For the 1st iteration, how did Attention change?"
without re-running full SQL each time. All tools accept target_gpu: int | None
(None = aggregate all GPUs).

Tool catalog:
  - search_nvtx_regions: fuzzy NVTX name discovery; call before any region diff
  - get_iteration_boundaries: per-iter windows for both profiles + is_aligned
  - explore_nvtx_hierarchy: step-by-step NVTX tree navigation
  - get_top_nvtx_diffs: hotspot radar: top N regions by absolute time change
  - get_iteration_diff: macro diff for one iteration (Stage 2)
  - get_region_diff: micro diff for a code region (Stage 3)
  - summarize_nvtx_subtree, get_launch_config_diff, get_source_code_context,
  - get_gpu_imbalance_stats, get_global_diff, get_memory_profile_diff, get_gpu_peak_tflops, compute_mfu (pure; call twice for compare)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .ai.backend.chat_tools import TOOL_COMPUTE_MFU, TOOL_GET_GPU_PEAK_TFLOPS
from .diff import ProfileDiffSummary, diff_profiles
from .nvtx_tree import build_nvtx_tree
from .overlap import detect_iterations, overlap_analysis
from .profile import Profile


def _top_k_payload(summary: ProfileDiffSummary, top_n: int = 5) -> tuple[list, list, float]:
    """Build top_regressions, top_improvements with impact ratio and others_aggregated_delta_ms (Payload Contract)."""
    total_pos = sum(k.delta_ns for k in summary.kernel_diffs if k.delta_ns > 0) or 1
    total_neg = sum(k.delta_ns for k in summary.kernel_diffs if k.delta_ns < 0) or -1

    def impact(k):
        pct = (
            (k.after_total_ns / (summary.after.total_gpu_ns or 1)) * 100
            if summary.after.total_gpu_ns
            else 0
        )
        contrib = (
            (k.delta_ns / total_pos * 100) if k.delta_ns > 0 else (k.delta_ns / total_neg * 100)
        )
        return {
            "pct_of_iteration_time": round(pct, 2),
            "contribution_to_total_delta_pct": round(contrib, 2),
        }

    top_keys = {k.key for k in summary.top_regressions[:top_n]} | {
        k.key for k in summary.top_improvements[:top_n]
    }
    others_ns = sum(k.delta_ns for k in summary.kernel_diffs if k.key not in top_keys)

    regressions = [
        {
            "name": k.name,
            "delta_ns": k.delta_ns,
            "delta_ms": round(k.delta_ns / 1e6, 3),
            "selection": k.selection.to_dict() if k.selection else None,
            **impact(k),
        }
        for k in summary.top_regressions[:top_n]
    ]
    improvements = [
        {
            "name": k.name,
            "delta_ns": k.delta_ns,
            "delta_ms": round(k.delta_ns / 1e6, 3),
            "selection": k.selection.to_dict() if k.selection else None,
            **impact(k),
        }
        for k in summary.top_improvements[:top_n]
    ]
    return regressions, improvements, round(others_ns / 1e6, 2)


@dataclass
class DiffContext:
    """Holds before/after profiles and optional cached diff for Phase C tools."""

    before: Profile
    after: Profile
    trim: tuple[int, int] | None
    marker: str
    _cached_summary: ProfileDiffSummary | None = field(default=None, repr=False)

    def ensure_summary(self, gpu: int | None) -> ProfileDiffSummary:
        if self._cached_summary is None:
            self._cached_summary = diff_profiles(
                self.before,
                self.after,
                gpu=gpu,
                trim=self.trim,
                limit=500,
                sort="delta",
                nvtx_limit=500,
            )
        return self._cached_summary

    @property
    def summary(self) -> ProfileDiffSummary | None:
        return self._cached_summary


def search_nvtx_regions(
    ctx: DiffContext,
    query: str,
    limit: int = 50,
    use_glob: bool = False,
) -> dict:
    """
    Fuzzy NVTX name discovery (LIKE/GLOB). Call before any region diff.

    Returns:
        regions: list of {text, total_ns_before, total_ns_after, count_before, count_after, in_before, in_after}
        query, limit
    """
    before_rows = ctx.before.search_nvtx_names(query, limit=limit, use_glob=use_glob, trim=ctx.trim)
    after_rows = ctx.after.search_nvtx_names(query, limit=limit, use_glob=use_glob, trim=ctx.trim)
    by_text: dict[str, dict] = {}
    for r in before_rows:
        text = str(r.get("text") or "")
        by_text.setdefault(
            text,
            {
                "text": text,
                "total_ns_before": 0,
                "total_ns_after": 0,
                "count_before": 0,
                "count_after": 0,
            },
        )
        by_text[text]["total_ns_before"] = int(r.get("total_ns") or 0)
        by_text[text]["count_before"] = int(r.get("count") or 0)
    for r in after_rows:
        text = str(r.get("text") or "")
        by_text.setdefault(
            text,
            {
                "text": text,
                "total_ns_before": 0,
                "total_ns_after": 0,
                "count_before": 0,
                "count_after": 0,
            },
        )
        by_text[text]["total_ns_after"] = int(r.get("total_ns") or 0)
        by_text[text]["count_after"] = int(r.get("count") or 0)
    for v in by_text.values():
        v["in_before"] = v["count_before"] > 0
        v["in_after"] = v["count_after"] > 0
    regions = sorted(by_text.values(), key=lambda x: -(x["total_ns_before"] + x["total_ns_after"]))[
        :limit
    ]
    return {
        "regions": regions,
        "query": query,
        "limit": limit,
        "use_glob": use_glob,
    }


def get_iteration_boundaries(
    ctx: DiffContext,
    marker: str | None = None,
    target_gpu: int | None = None,
) -> dict:
    """
    Per-iteration time windows for both profiles plus is_aligned flag.

    Uses detect_iterations on both profiles. target_gpu is used for kernel
    correlation; if None, use first device from each profile for detection.
    """
    m = marker or ctx.marker
    devices_before = ctx.before.meta.devices or [0]
    devices_after = ctx.after.meta.devices or [0]
    gpu_b = target_gpu if target_gpu is not None else devices_before[0]
    gpu_a = target_gpu if target_gpu is not None else devices_after[0]
    iters_before = detect_iterations(ctx.before, gpu_b, trim=ctx.trim, marker=m)
    iters_after = detect_iterations(ctx.after, gpu_a, trim=ctx.trim, marker=m)
    boundaries = []
    for i in range(max(len(iters_before), len(iters_after))):
        b = iters_before[i] if i < len(iters_before) else None
        a = iters_after[i] if i < len(iters_after) else None
        start_ns_b = int(b["gpu_start_s"] * 1e9) if b else None
        end_ns_b = int(b["gpu_end_s"] * 1e9) if b else None
        start_ns_a = int(a["gpu_start_s"] * 1e9) if a else None
        end_ns_a = int(a["gpu_end_s"] * 1e9) if a else None
        boundaries.append(
            {
                "iteration_index": i,
                "before": {
                    "start_ns": start_ns_b,
                    "end_ns": end_ns_b,
                    "duration_ms": b["duration_ms"] if b else None,
                },
                "after": {
                    "start_ns": start_ns_a,
                    "end_ns": end_ns_a,
                    "duration_ms": a["duration_ms"] if a else None,
                },
                "has_before": b is not None,
                "has_after": a is not None,
            }
        )
    return {
        "marker": m,
        "target_gpu": target_gpu,
        "iteration_count_before": len(iters_before),
        "iteration_count_after": len(iters_after),
        "is_aligned": len(iters_before) == len(iters_after) and len(iters_before) > 0,
        "boundaries": boundaries,
        "workload_warning": abs(len(iters_before) - len(iters_after)) > 0,
    }


def _tree_children_at_path(roots: list[dict], path_parts: list[str], depth: int) -> list[dict]:
    """Walk NVTX tree by path (e.g. ['sample_0', 'Attention']) and return child nodes up to depth."""
    if not path_parts and depth <= 0:
        return [
            {
                "name": n["name"],
                "type": n["type"],
                "duration_ms": round((n["end"] - n["start"]) / 1e6, 3),
                "path": n["name"],
            }
            for n in roots
        ]
    current = list(roots)
    for part in path_parts:
        next_level = []
        for node in current:
            if node.get("type") != "nvtx":
                continue
            for ch in node.get("children", []):
                if ch.get("type") == "nvtx" and ch.get("name") == part:
                    next_level.append(ch)
                    break
        current = next_level
        if not current:
            return []
    if depth <= 0:
        return [
            {
                "name": n["name"],
                "type": n["type"],
                "duration_ms": round((n["end"] - n["start"]) / 1e6, 3),
            }
            for n in current
        ]
    out = []
    for n in current:
        if n.get("type") != "nvtx":
            continue
        path_str = " > ".join(path_parts + [n["name"]])
        out.append(
            {
                "name": n["name"],
                "type": "nvtx",
                "duration_ms": round((n["end"] - n["start"]) / 1e6, 3),
                "path": path_str,
                "child_count": len([c for c in n.get("children", []) if c.get("type") == "nvtx"]),
            }
        )
    return out


def explore_nvtx_hierarchy(
    ctx: DiffContext,
    parent_path: str = "",
    depth: int = 1,
    target_gpu: int | None = None,
    profile_side: str = "after",
) -> dict:
    """
    Step-by-step NVTX tree navigation. Use to discover exact paths before get_region_diff.

    parent_path: NVTX path like "sample_0 > Attention" or "" for roots.
    depth: how many levels of children to return (1 = immediate children).
    profile_side: "before" or "after" — which profile to walk.
    """
    prof = ctx.after if profile_side == "after" else ctx.before
    devices = prof.meta.devices or [0]
    gpu = target_gpu if target_gpu is not None else devices[0]
    trim = ctx.trim or prof.meta.time_range
    roots = build_nvtx_tree(prof, gpu, trim)
    if not roots:
        return {
            "parent_path": parent_path,
            "depth": depth,
            "target_gpu": target_gpu,
            "profile_side": profile_side,
            "children": [],
            "note": "No NVTX tree (empty profile or trim).",
        }
    path_parts = [p.strip() for p in parent_path.split(">") if p.strip()]
    children = _tree_children_at_path(roots, path_parts, depth)
    return {
        "parent_path": parent_path,
        "depth": depth,
        "target_gpu": target_gpu,
        "profile_side": profile_side,
        "children": children,
    }


def get_top_nvtx_diffs(
    ctx: DiffContext,
    limit: int = 20,
    target_gpu: int | None = None,
) -> dict:
    """
    Hotspot radar: top NVTX regions by absolute time change (regression + improvement).

    Requires cached diff; pass gpu when building context or call ensure_summary(target_gpu) first.
    """
    gpu = target_gpu
    summary = ctx.ensure_summary(gpu)
    nvtx_diffs = summary.nvtx_diffs
    by_abs = sorted(nvtx_diffs, key=lambda n: abs(n.delta_ns), reverse=True)
    top = by_abs[: max(0, limit)]
    return {
        "limit": limit,
        "target_gpu": target_gpu,
        "top_nvtx_diffs": [
            {
                "text": nd.text,
                "before_total_ns": nd.before_total_ns,
                "after_total_ns": nd.after_total_ns,
                "delta_ns": nd.delta_ns,
                "delta_ms": round(nd.delta_ns / 1e6, 3),
                "classification": nd.classification,
            }
            for nd in top
        ],
    }


def get_iteration_diff(
    ctx: DiffContext,
    iteration_index: int,
    marker: str | None = None,
    target_gpu: int | None = None,
) -> dict:
    """
    Macro diff for one iteration (time-window trim from detect_iterations).

    Returns defensive payload: wall_clock_ms, sum_of_kernels_ms, memcpy_ms,
    top_k_regressions, top_k_improvements, overlap_pct, etc.
    """
    bounds = get_iteration_boundaries(ctx, marker=marker, target_gpu=target_gpu)
    if iteration_index >= len(bounds["boundaries"]):
        return {
            "error": f"iteration_index {iteration_index} out of range (max {len(bounds['boundaries']) - 1})",
            "iteration_index": iteration_index,
        }
    bnd = bounds["boundaries"][iteration_index]
    trim_before = (
        (bnd["before"]["start_ns"], bnd["before"]["end_ns"])
        if bnd["before"]["start_ns"] is not None
        else None
    )
    trim_after = (
        (bnd["after"]["start_ns"], bnd["after"]["end_ns"])
        if bnd["after"]["start_ns"] is not None
        else None
    )
    if not trim_before or not trim_after:
        return {
            "error": "Missing before or after window for this iteration",
            "iteration_index": iteration_index,
            "has_before": bnd["has_before"],
            "has_after": bnd["has_after"],
        }
    summary = diff_profiles(
        ctx.before,
        ctx.after,
        gpu=target_gpu,
        trim_before=trim_before,
        trim_after=trim_after,
        limit=10,
        sort="delta",
    )
    wall_b = (trim_before[1] - trim_before[0]) / 1e6
    wall_a = (trim_after[1] - trim_after[0]) / 1e6
    sum_k_b = sum(k.total_ns for k in summary.before.kernels) / 1e6
    sum_k_a = sum(k.total_ns for k in summary.after.kernels) / 1e6
    # Unique stream count (Payload Contract); target_gpu=None aggregates all devices
    # so these detail fields match the all-GPU scope of `summary`.
    kerns_b = ctx.before.kernels(target_gpu, trim_before)
    kerns_a = ctx.after.kernels(target_gpu, trim_after)
    # streamId is device-scoped, so count distinct (deviceId, streamId) pairs;
    # otherwise an all-GPU window collapses each device's matching stream ids.
    unique_streams_b = len(
        {(k.get("deviceId"), k.get("streamId")) for k in kerns_b if k.get("streamId") is not None}
    )
    unique_streams_a = len(
        {(k.get("deviceId"), k.get("streamId")) for k in kerns_a if k.get("streamId") is not None}
    )
    # Memcpy H2D/D2H in window
    mc_b = ctx.before.memcpy_in_window(target_gpu, trim_before)
    mc_a = ctx.after.memcpy_in_window(target_gpu, trim_after)
    memcpy_ms = {
        "h2d": {
            "before": round(mc_b["h2d_ns"] / 1e6, 2),
            "after": round(mc_a["h2d_ns"] / 1e6, 2),
            "delta": round((mc_a["h2d_ns"] - mc_b["h2d_ns"]) / 1e6, 2),
        },
        "d2h": {
            "before": round(mc_b["d2h_ns"] / 1e6, 2),
            "after": round(mc_a["d2h_ns"] / 1e6, 2),
            "delta": round((mc_a["d2h_ns"] - mc_b["d2h_ns"]) / 1e6, 2),
        },
    }
    # top3_global_categories: Compute / Memcpy / Idle (Payload Contract)
    idle_b = summary.before.overlap.get("idle_ms") or 0
    idle_a = summary.after.overlap.get("idle_ms") or 0
    mem_b = (mc_b["h2d_ns"] + mc_b["d2h_ns"] + mc_b["d2d_ns"]) / 1e6
    mem_a = (mc_a["h2d_ns"] + mc_a["d2h_ns"] + mc_a["d2d_ns"]) / 1e6
    top3_global_categories = {
        "Compute": {
            "before": round(sum_k_b, 2),
            "after": round(sum_k_a, 2),
            "delta": round(sum_k_a - sum_k_b, 2),
        },
        "Memcpy": {
            "before": round(mem_b, 2),
            "after": round(mem_a, 2),
            "delta": round(mem_a - mem_b, 2),
        },
        "Idle": {
            "before": round(idle_b, 2),
            "after": round(idle_a, 2),
            "delta": round(idle_a - idle_b, 2),
        },
    }
    top_regressions_payload, top_improvements_payload, others_ms = _top_k_payload(summary, 5)
    return {
        "iteration_index": iteration_index,
        "is_aligned": bounds["is_aligned"],
        "wall_clock_ms": {
            "before": round(wall_b, 2),
            "after": round(wall_a, 2),
            "delta": round(wall_a - wall_b, 2),
        },
        "sum_of_kernels_ms": {
            "before": round(sum_k_b, 2),
            "after": round(sum_k_a, 2),
            "delta": round(sum_k_a - sum_k_b, 2),
        },
        "memcpy_ms": memcpy_ms,
        "memcpy_compute_overlap_pct": None,  # Optional: requires interval overlap; agent can ignore when None
        "unique_streams_count_before": unique_streams_b,
        "unique_streams_count_after": unique_streams_a,
        "top_regressions": top_regressions_payload,
        "top_improvements": top_improvements_payload,
        "others_aggregated_delta_ms": others_ms,
        "top3_global_categories": top3_global_categories,
        "overlap_pct": {
            "before": summary.before.overlap.get("overlap_pct"),
            "after": summary.after.overlap.get("overlap_pct"),
        },
        "workload_warning": bounds.get("workload_warning", False),
        "Hardware_Warning": _hardware_warning(ctx),
        "JIT_Compilation_Warning": (
            iteration_index == 0
            or _has_overhead_in_window(ctx.before, trim_before)
            or _has_overhead_in_window(ctx.after, trim_after)
        ),
    }


def get_global_diff(
    ctx: DiffContext,
    skip_first_ms: float = 0,
    duration_ms: float | None = None,
    target_gpu: int | None = None,
) -> dict:
    """
    Time-based fallback when no iteration marker. Compare [skip_first_ms, skip_first_ms + duration_ms].

    If duration_ms is None, use the rest of the profile after skip.
    """
    trim_before = _time_window_ns(ctx.before, skip_first_ms, duration_ms)
    trim_after = _time_window_ns(ctx.after, skip_first_ms, duration_ms)
    if trim_before is None or trim_after is None:
        return {"error": "Could not compute time window (empty profile or invalid skip/duration)"}
    summary = diff_profiles(
        ctx.before, ctx.after, gpu=target_gpu, trim=trim_before, limit=15, sort="delta"
    )
    summary2 = diff_profiles(
        ctx.before, ctx.after, gpu=target_gpu, trim=trim_after, limit=15, sort="delta"
    )
    wall_b = (trim_before[1] - trim_before[0]) / 1e6
    wall_a = (trim_after[1] - trim_after[0]) / 1e6
    return {
        "skip_first_ms": skip_first_ms,
        "duration_ms": duration_ms,
        "target_gpu": target_gpu,
        "wall_clock_ms": {
            "before": round(wall_b, 2),
            "after": round(wall_a, 2),
            "delta": round(wall_a - wall_b, 2),
        },
        "top_regressions": [
            {
                "name": k.name,
                "delta_ms": round(k.delta_ns / 1e6, 3),
                "selection": k.selection.to_dict() if k.selection else None,
            }
            for k in summary2.top_regressions[:10]
        ],
        "top_improvements": [
            {
                "name": k.name,
                "delta_ms": round(k.delta_ns / 1e6, 3),
                "selection": k.selection.to_dict() if k.selection else None,
            }
            for k in summary2.top_improvements[:10]
        ],
        "warnings": summary.warnings,
    }


def get_region_diff(
    ctx: DiffContext,
    nvtx_exact_match: str | list[str],
    iteration_index: int | None = None,
    target_gpu: int | None = None,
) -> dict:
    """
    Micro diff for a code region (time window + NVTX filter).
    Call search_nvtx_regions or explore_nvtx_hierarchy first to get exact NVTX name(s).
    """
    if isinstance(nvtx_exact_match, str):
        nvtx_exact_match = [nvtx_exact_match]
    trim_before = ctx.trim
    trim_after = ctx.trim
    if iteration_index is not None:
        bounds = get_iteration_boundaries(ctx, target_gpu=target_gpu)
        if iteration_index >= len(bounds["boundaries"]):
            return {
                "error": f"iteration_index {iteration_index} out of range",
                "iteration_index": iteration_index,
            }
        bnd = bounds["boundaries"][iteration_index]
        trim_before = (
            (bnd["before"]["start_ns"], bnd["before"]["end_ns"])
            if bnd["before"]["start_ns"] is not None
            else ctx.trim
        )
        trim_after = (
            (bnd["after"]["start_ns"], bnd["after"]["end_ns"])
            if bnd["after"]["start_ns"] is not None
            else ctx.trim
        )
    if trim_before is None:
        trim_before = ctx.before.meta.time_range
    if trim_after is None:
        trim_after = ctx.after.meta.time_range
    if trim_before is None or trim_after is None:
        return {"error": "Time window not available (empty profile)"}
    # Use iteration-scoped diff when trim was set from boundaries
    summary = diff_profiles(
        ctx.before,
        ctx.after,
        gpu=target_gpu,
        trim_before=trim_before,
        trim_after=trim_after,
        limit=10,
        sort="delta",
        nvtx_limit=500,
    )
    matching = [n for n in summary.nvtx_diffs if n.text in nvtx_exact_match]
    if not matching:
        return {
            "error": f"No NVTX region matching {nvtx_exact_match!r}. Call search_nvtx_regions first.",
            "nvtx_exact_match": nvtx_exact_match,
        }
    nd = matching[0]
    # Payload Contract: same defensive fields as get_iteration_diff; target_gpu=None
    # aggregates all devices so these match the all-GPU scope of `summary`.
    mc_b = ctx.before.memcpy_in_window(target_gpu, trim_before)
    mc_a = ctx.after.memcpy_in_window(target_gpu, trim_after)
    memcpy_ms = {
        "h2d": {
            "before": round(mc_b["h2d_ns"] / 1e6, 2),
            "after": round(mc_a["h2d_ns"] / 1e6, 2),
            "delta": round((mc_a["h2d_ns"] - mc_b["h2d_ns"]) / 1e6, 2),
        },
        "d2h": {
            "before": round(mc_b["d2h_ns"] / 1e6, 2),
            "after": round(mc_a["d2h_ns"] / 1e6, 2),
            "delta": round((mc_a["d2h_ns"] - mc_b["d2h_ns"]) / 1e6, 2),
        },
    }
    kerns_b = ctx.before.kernels(target_gpu, trim_before)
    kerns_a = ctx.after.kernels(target_gpu, trim_after)
    # streamId is device-scoped, so count distinct (deviceId, streamId) pairs;
    # otherwise an all-GPU window collapses each device's matching stream ids.
    unique_streams_b = len(
        {(k.get("deviceId"), k.get("streamId")) for k in kerns_b if k.get("streamId") is not None}
    )
    unique_streams_a = len(
        {(k.get("deviceId"), k.get("streamId")) for k in kerns_a if k.get("streamId") is not None}
    )
    idle_b = summary.before.overlap.get("idle_ms") or 0
    idle_a = summary.after.overlap.get("idle_ms") or 0
    mem_b = (mc_b["h2d_ns"] + mc_b["d2h_ns"] + mc_b["d2d_ns"]) / 1e6
    mem_a = (mc_a["h2d_ns"] + mc_a["d2h_ns"] + mc_a["d2d_ns"]) / 1e6
    sum_k_b = summary.before.total_gpu_ns / 1e6
    sum_k_a = summary.after.total_gpu_ns / 1e6
    top3_global_categories = {
        "Compute": {
            "before": round(sum_k_b, 2),
            "after": round(sum_k_a, 2),
            "delta": round(sum_k_a - sum_k_b, 2),
        },
        "Memcpy": {
            "before": round(mem_b, 2),
            "after": round(mem_a, 2),
            "delta": round(mem_a - mem_b, 2),
        },
        "Idle": {
            "before": round(idle_b, 2),
            "after": round(idle_a, 2),
            "delta": round(idle_a - idle_b, 2),
        },
    }
    top_regressions_payload, top_improvements_payload, others_ms = _top_k_payload(summary, 5)

    return {
        "nvtx_exact_match": nd.text,
        "wall_clock_ms": {
            "before": round(nd.before_total_ns / 1e6, 2),
            "after": round(nd.after_total_ns / 1e6, 2),
            "delta": round(nd.delta_ns / 1e6, 2),
        },
        "sum_of_kernels_ms": {
            "before": round(nd.before_total_ns / 1e6, 2),
            "after": round(nd.after_total_ns / 1e6, 2),
            "delta": round(nd.delta_ns / 1e6, 2),
        },
        "memcpy_ms": memcpy_ms,
        "memcpy_compute_overlap_pct": None,
        "unique_streams_count_before": unique_streams_b,
        "unique_streams_count_after": unique_streams_a,
        "classification": nd.classification,
        "top_regressions": top_regressions_payload,
        "top_improvements": top_improvements_payload,
        "others_aggregated_delta_ms": others_ms,
        "top3_global_categories": top3_global_categories,
        "overlap_pct": {
            "before": summary.before.overlap.get("overlap_pct"),
            "after": summary.after.overlap.get("overlap_pct"),
        },
        "workload_warning": False,
        "Hardware_Warning": _hardware_warning(ctx),
        "JIT_Compilation_Warning": (
            _has_overhead_in_window(ctx.before, trim_before)
            or _has_overhead_in_window(ctx.after, trim_after)
        ),
    }


def summarize_nvtx_subtree(
    ctx: DiffContext,
    parent_path: str,
    iteration_index: int | None = None,
    target_gpu: int | None = None,
    top_n: int = 3,
) -> dict:
    """
    Roll up all depths under parent; return top N by delta (macro→micro in one call).
    """
    summary = ctx.ensure_summary(target_gpu)
    # Filter nvtx_diffs to those whose text equals parent_path or starts with parent_path + " >"
    prefix = (parent_path.rstrip(" >") + " >") if parent_path.strip() else ""
    candidates = [
        n
        for n in summary.nvtx_diffs
        if not prefix or n.text == parent_path.strip() or n.text.startswith(prefix)
    ]
    by_abs = sorted(candidates, key=lambda n: abs(n.delta_ns), reverse=True)
    top_deltas = [
        {
            "text": n.text,
            "delta_ns": n.delta_ns,
            "delta_ms": round(n.delta_ns / 1e6, 3),
            "classification": n.classification,
        }
        for n in by_abs[:top_n]
    ]
    return {
        "parent_path": parent_path,
        "iteration_index": iteration_index,
        "target_gpu": target_gpu,
        "top_by_delta": top_deltas,
        "total_matching": len(candidates),
    }


_LAUNCH_GRID_COLS = ("gridX", "gridY", "gridZ")
_LAUNCH_BLOCK_COLS = ("blockX", "blockY", "blockZ")
_LAUNCH_REGISTER_COLS = ("registersPerThread",)
_LAUNCH_STATIC_SHARED_COLS = ("staticSharedMemory", "staticSharedMemoryBytes")
_LAUNCH_DYNAMIC_SHARED_COLS = ("dynamicSharedMemory", "dynamicSharedMemoryBytes")


def _kernel_table_columns(prof: Profile) -> list[str]:
    kt = prof.schema.kernel_table
    if not kt:
        return []
    try:
        return prof.adapter.get_table_columns(kt)
    except Exception:
        # Keep the older explicit probe paths as a fallback for any adapter
        # that does not expose get_table_columns correctly.
        try:
            if prof.db is not None:
                col_rows = prof._duckdb_query(f"DESCRIBE {kt}")
                return [r.get("column_name") or r.get("name") or "" for r in col_rows]
            with prof._lock:
                return [r[1] for r in prof.conn.execute(f"PRAGMA table_info({kt})").fetchall()]
        except Exception:
            return []


def _first_existing(cols: set[str], candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in cols:
            return col
    return None


def _launch_config_columns(cols: list[str]) -> dict[str, str]:
    colset = set(cols)
    selected: dict[str, str] = {}
    for col in _LAUNCH_GRID_COLS + _LAUNCH_BLOCK_COLS:
        if col in colset:
            selected[col] = col
    reg_col = _first_existing(colset, _LAUNCH_REGISTER_COLS)
    if reg_col:
        selected["registersPerThread"] = reg_col
    static_col = _first_existing(colset, _LAUNCH_STATIC_SHARED_COLS)
    if static_col:
        selected["staticSharedMemory"] = static_col
    dynamic_col = _first_existing(colset, _LAUNCH_DYNAMIC_SHARED_COLS)
    if dynamic_col:
        selected["dynamicSharedMemory"] = dynamic_col
    return selected


def _product(values: list[int | None]) -> int | None:
    if any(v is None for v in values):
        return None
    out = 1
    for value in values:
        out *= int(value or 0)
    return out


def _format_dims(values: list[int | None] | None) -> str:
    if not values:
        return "?"
    return "x".join("?" if v is None else str(v) for v in values)


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "?"
    if value < 0:
        return "-" + _format_bytes(-value)
    if value == 0:
        return "0B"
    if value % 1024 == 0:
        kib = value // 1024
        if kib % 1024 == 0:
            mib = kib // 1024
            if mib % 1024 == 0:
                return f"{mib // 1024}GiB"
            return f"{mib}MiB"
        return f"{kib}KiB"
    return f"{value}B"


def _format_signed_bytes(value: int | None) -> str:
    if value is None:
        return "?"
    if value == 0:
        return "0B"
    sign = "+" if value > 0 else "-"
    return sign + _format_bytes(abs(value))


def _scalar_delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return after - before


def _delta_entry(before, after):
    if before is None or after is None:
        delta = None
    elif isinstance(before, list) and isinstance(after, list):
        delta = [
            _scalar_delta(b, a)
            for b, a in zip(before, after, strict=False)
        ]
    else:
        delta = after - before
    return {"before": before, "after": after, "delta": delta}


def _shared_total(row: dict) -> int | None:
    static = row.get("staticSharedMemory")
    dynamic = row.get("dynamicSharedMemory")
    if static is None and dynamic is None:
        return None
    return int(static or 0) + int(dynamic or 0)


def _normalize_launch_row(row: dict | None, selected_cols: dict[str, str]) -> dict | None:
    if row is None:
        return None
    config: dict = {
        "matched_name": row.get("matched_name"),
        "sample_count": int(row.get("sample_count") or 0),
        "total_ms": round((int(row.get("total_ns") or 0)) / 1e6, 3),
        "avg_ms": round((float(row.get("avg_ns") or 0.0)) / 1e6, 6),
    }
    distinct_configs = row.get("distinct_configs")
    if distinct_configs is not None:
        config["distinct_configs"] = int(distinct_configs)
    total_invocations = int(row.get("total_invocations") or 0)
    if total_invocations > 0:
        config["total_invocations"] = total_invocations
        config["dominant_share"] = round(config["sample_count"] / total_invocations, 4)

    for col in _LAUNCH_GRID_COLS + _LAUNCH_BLOCK_COLS:
        if col in selected_cols:
            value = row.get(col)
            config[col] = int(value) if value is not None else None

    if any(col in config for col in _LAUNCH_GRID_COLS):
        config["grid"] = [config.get(col) for col in _LAUNCH_GRID_COLS]
        config["total_blocks"] = _product(config["grid"])
    if any(col in config for col in _LAUNCH_BLOCK_COLS):
        config["block"] = [config.get(col) for col in _LAUNCH_BLOCK_COLS]
        config["threads_per_block"] = _product(config["block"])
    if config.get("total_blocks") is not None and config.get("threads_per_block") is not None:
        config["total_threads"] = int(config["total_blocks"]) * int(config["threads_per_block"])

    for col in ("registersPerThread", "staticSharedMemory", "dynamicSharedMemory"):
        if col in selected_cols:
            value = row.get(col)
            config[col] = int(value) if value is not None else None
    shared = _shared_total(config)
    if shared is not None:
        config["sharedMemoryBytes"] = shared
    return config


def _query_launch_config(
    prof: Profile,
    kernel_name: str,
    selected_cols: dict[str, str],
    trim: tuple[int, int] | None,
    target_gpu: int | None,
) -> dict | None:
    if not prof.schema.kernel_table or not selected_cols:
        return None

    select_parts = ["COALESCE(d.value, s.value, ?) AS matched_name"]
    group_parts = []
    for output_name, source_col in selected_cols.items():
        select_parts.append(f"k.{source_col} AS {output_name}")
        group_parts.append(f"k.{source_col}")
    select_parts.extend(
        [
            "COUNT(*) AS sample_count",
            "SUM(k.[end] - k.start) AS total_ns",
            "AVG(k.[end] - k.start) AS avg_ns",
        ]
    )

    sql = f"""
        SELECT {", ".join(select_parts)}
        FROM {prof.schema.kernel_table} k
        LEFT JOIN StringIds s ON k.shortName = s.id
        LEFT JOIN StringIds d ON k.demangledName = d.id
        WHERE (s.value = ? OR d.value = ?)
    """
    params: list = [kernel_name, kernel_name, kernel_name]
    if target_gpu is not None:
        sql += " AND k.deviceId = ?"
        params.append(target_gpu)
    if trim is not None:
        sql += " AND k.start >= ? AND k.[end] <= ?"
        params.extend([int(trim[0]), int(trim[1])])
    if group_parts:
        sql += " GROUP BY " + ", ".join(group_parts) + ", matched_name"
    sql += " ORDER BY total_ns DESC, sample_count DESC"

    rows = prof._duckdb_query(sql, params)
    if not rows:
        return None
    # One row per distinct launch config; the first (by total GPU time) is the
    # dominant config we report. Carry the distinct-config count + total
    # invocations so the caller can express how representative the dominant
    # config is — an honesty signal for kernels launched with several shapes.
    dominant = dict(rows[0])
    dominant["distinct_configs"] = len(rows)
    dominant["total_invocations"] = sum(int(r.get("sample_count") or 0) for r in rows)
    return dominant


def _resolve_diff_windows(
    ctx: DiffContext,
    iteration_index: int | None,
    target_gpu: int | None,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None, dict | None]:
    if iteration_index is None:
        return ctx.trim, ctx.trim, None
    bounds = get_iteration_boundaries(ctx, marker=ctx.marker, target_gpu=target_gpu)
    # Reject negatives explicitly: Python would otherwise let -1 silently index
    # the last iteration, contradicting the "out of range" contract.
    if iteration_index < 0 or iteration_index >= len(bounds["boundaries"]):
        return (
            None,
            None,
            {
                "error": f"iteration_index {iteration_index} out of range (max {len(bounds['boundaries']) - 1})",
                "iteration_index": iteration_index,
            },
        )
    bnd = bounds["boundaries"][iteration_index]
    trim_before = (
        (bnd["before"]["start_ns"], bnd["before"]["end_ns"])
        if bnd["before"]["start_ns"] is not None
        else None
    )
    trim_after = (
        (bnd["after"]["start_ns"], bnd["after"]["end_ns"])
        if bnd["after"]["start_ns"] is not None
        else None
    )
    if trim_before is None or trim_after is None:
        return (
            trim_before,
            trim_after,
            {
                "error": "Missing before or after window for this iteration",
                "iteration_index": iteration_index,
                "has_before": bnd["has_before"],
                "has_after": bnd["has_after"],
            },
        )
    return trim_before, trim_after, None


def _resolve_launch_config_windows(
    ctx: DiffContext,
    iteration_index: int | None,
    target_gpu: int | None,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None, dict | None]:
    return _resolve_diff_windows(ctx, iteration_index, target_gpu)


def _build_launch_delta(before: dict | None, after: dict | None) -> dict:
    delta: dict = {}
    for config_field in (
        "gridX",
        "gridY",
        "gridZ",
        "grid",
        "blockX",
        "blockY",
        "blockZ",
        "block",
        "total_blocks",
        "threads_per_block",
        "total_threads",
        "registersPerThread",
        "staticSharedMemory",
        "dynamicSharedMemory",
        "sharedMemoryBytes",
    ):
        b = before.get(config_field) if before else None
        a = after.get(config_field) if after else None
        if b is not None or a is not None:
            delta[config_field] = _delta_entry(b, a)
    return delta


def _changed(delta: dict, field: str) -> bool:
    entry = delta.get(field)
    if not entry:
        return False
    d = entry.get("delta")
    if isinstance(d, list):
        return any(x not in (None, 0) for x in d)
    return d not in (None, 0)


def _launch_config_explanation(kernel_name: str, before: dict | None, after: dict | None, delta: dict) -> str:
    if before is None and after is None:
        return f"No launch configuration rows found for kernel '{kernel_name}'."
    if before is None:
        return f"Kernel '{kernel_name}' only appears after; launch configuration has no before baseline."
    if after is None:
        return f"Kernel '{kernel_name}' only appears before; launch configuration has no after comparison."

    parts: list[str] = []
    if _changed(delta, "grid"):
        parts.append(f"grid {_format_dims(before.get('grid'))} -> {_format_dims(after.get('grid'))}")
    if _changed(delta, "block"):
        parts.append(f"block {_format_dims(before.get('block'))} -> {_format_dims(after.get('block'))}")
    if _changed(delta, "registersPerThread"):
        parts.append(
            f"registers/thread {before.get('registersPerThread')} -> {after.get('registersPerThread')}"
        )
    if _changed(delta, "sharedMemoryBytes"):
        parts.append(
            f"shared mem {_format_bytes(before.get('sharedMemoryBytes'))} -> {_format_bytes(after.get('sharedMemoryBytes'))}"
        )
    if not parts:
        return "Launch config is unchanged; slowdown is likely from data, scheduling, memory behavior, or lower-level kernel execution."

    occupancy_reasons = []
    reg_delta = (delta.get("registersPerThread") or {}).get("delta")
    shared_delta = (delta.get("sharedMemoryBytes") or {}).get("delta")
    threads_delta = (delta.get("threads_per_block") or {}).get("delta")
    blocks_delta = (delta.get("total_blocks") or {}).get("delta")
    if isinstance(reg_delta, int) and reg_delta > 0:
        occupancy_reasons.append("higher register pressure")
    if isinstance(shared_delta, int) and shared_delta > 0:
        occupancy_reasons.append("more shared memory per block")
    if isinstance(threads_delta, int) and threads_delta > 0:
        occupancy_reasons.append("larger blocks")
    if isinstance(blocks_delta, int) and blocks_delta < 0:
        occupancy_reasons.append("fewer blocks")

    if occupancy_reasons:
        return "; ".join(parts) + " -> likely lower occupancy/parallelism from " + ", ".join(occupancy_reasons) + "."
    return "; ".join(parts) + " -> launch geometry changed; inspect occupancy and memory behavior."


def get_launch_config_diff(
    ctx: DiffContext,
    kernel_name: str,
    iteration_index: int | None = None,
    target_gpu: int | None = None,
) -> dict:
    """
    Grid/block/registers before vs after; explains 'why'. Returns error when columns missing (BETA).
    """
    before_cols = _kernel_table_columns(ctx.before)
    after_cols = _kernel_table_columns(ctx.after)
    before_selected = _launch_config_columns(before_cols)
    after_selected = _launch_config_columns(after_cols)
    common_keys = sorted(set(before_selected).intersection(after_selected))
    before_common = {key: before_selected[key] for key in common_keys}
    after_common = {key: after_selected[key] for key in common_keys}

    if not ("gridX" in before_common or "blockX" in before_common):
        return {
            "error": "not available",
            "reason": "Launch-config columns (gridX/blockX, etc.) not in export",
            "available_columns": {
                "before": before_cols,
                "after": after_cols,
            },
        }

    trim_before, trim_after, window_error = _resolve_launch_config_windows(
        ctx, iteration_index, target_gpu
    )
    if window_error is not None:
        return window_error

    before_row = _query_launch_config(ctx.before, kernel_name, before_common, trim_before, target_gpu)
    after_row = _query_launch_config(ctx.after, kernel_name, after_common, trim_after, target_gpu)
    before_config = _normalize_launch_row(before_row, before_common)
    after_config = _normalize_launch_row(after_row, after_common)
    delta = _build_launch_delta(before_config, after_config)
    explanation = _launch_config_explanation(kernel_name, before_config, after_config, delta)

    base = {
        "kernel_name": kernel_name,
        "target_gpu": target_gpu,
        "iteration_index": iteration_index,
        "trim_before_ns": list(trim_before) if trim_before else None,
        "trim_after_ns": list(trim_after) if trim_after else None,
        "columns_used": {
            "before": before_common,
            "after": after_common,
        },
        "before": before_config,
        "after": after_config,
        "delta": delta,
        "explanation": explanation,
        "uses_tensor_core_likely": any(
            p in kernel_name
            for p in ("sm80_xmma_", "volta_fp16_s884gemm", "h884gemm", "xmma", "wmma")
        ),
    }
    if before_config is None or after_config is None:
        base["error"] = "not comparable"
        base["reason"] = explanation
    return base


def get_source_code_context(ctx: DiffContext, nvtx_path: str) -> dict:
    """
    filename:line for Coder Agent handoff. Returns error when SOURCE_LOCATOR missing.
    """
    if "CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR" not in ctx.before.schema.tables:
        return {"error": "not available", "reason": "SOURCE_LOCATOR table not in export"}
    return {
        "nvtx_path": nvtx_path,
        "before_profile": ctx.before.path,
        "after_profile": ctx.after.path,
        "file_line": None,
        "semantic_fingerprint": None,
    }


def get_gpu_imbalance_stats(
    ctx: DiffContext,
    iteration_index: int,
    marker: str | None = None,
) -> dict:
    """
    Per-GPU compute/nccl/idle for NCCL straggler proof. Call when NCCL time spiked.
    """
    bounds = get_iteration_boundaries(ctx, marker=marker, target_gpu=None)
    if iteration_index >= len(bounds["boundaries"]):
        return {"error": f"iteration_index {iteration_index} out of range"}
    bnd = bounds["boundaries"][iteration_index]
    trim_b = (
        (bnd["before"]["start_ns"], bnd["before"]["end_ns"]) if bnd["before"]["start_ns"] else None
    )
    trim_a = (
        (bnd["after"]["start_ns"], bnd["after"]["end_ns"]) if bnd["after"]["start_ns"] else None
    )
    devices = sorted(set(ctx.before.meta.devices or [0]) | set(ctx.after.meta.devices or [0]))
    per_gpu = []
    for dev in devices:
        ob = (
            overlap_analysis(ctx.before, dev, trim_b)
            if trim_b
            else {"compute_only_ms": 0, "nccl_only_ms": 0, "idle_ms": 0}
        )
        oa = (
            overlap_analysis(ctx.after, dev, trim_a)
            if trim_a
            else {"compute_only_ms": 0, "nccl_only_ms": 0, "idle_ms": 0}
        )
        if isinstance(ob, dict) and "error" in ob:
            ob = {"compute_only_ms": 0, "nccl_only_ms": 0, "idle_ms": 0}
        if isinstance(oa, dict) and "error" in oa:
            oa = {"compute_only_ms": 0, "nccl_only_ms": 0, "idle_ms": 0}
        per_gpu.append(
            {
                "gpu_id": dev,
                "before": {
                    "compute_ms": ob.get("compute_only_ms", 0),
                    "nccl_ms": ob.get("nccl_only_ms", 0),
                    "idle_ms": ob.get("idle_ms", 0),
                },
                "after": {
                    "compute_ms": oa.get("compute_only_ms", 0),
                    "nccl_ms": oa.get("nccl_only_ms", 0),
                    "idle_ms": oa.get("idle_ms", 0),
                },
            }
        )
    return {"iteration_index": iteration_index, "per_gpu": per_gpu}


_MEMORY_USAGE_TABLE = "CUDA_GPU_MEMORY_USAGE_EVENTS"
_MEMORY_REQUIRED_COLS = ("start", "bytes", "memoryOperationType")
_MEMORY_DELTA_FIELDS = (
    "peak_vram_bytes",
    "baseline_vram_bytes",
    "final_vram_bytes",
    "allocated_bytes",
    "freed_bytes",
    "net_delta_bytes",
    "alloc_count",
    "free_count",
    "event_count",
    "host_event_count",
    "unknown_event_count",
)
# memKind 0/1 are Pageable / Pinned host memory and never consume VRAM.
_HOST_MEM_KINDS = (0, 1)


def _memory_table_columns(prof: Profile) -> list[str]:
    if _MEMORY_USAGE_TABLE not in prof.schema.tables:
        return []
    try:
        return prof.adapter.get_table_columns(_MEMORY_USAGE_TABLE)
    except Exception:
        try:
            if prof.db is not None:
                rows = prof._duckdb_query(f"DESCRIBE {_MEMORY_USAGE_TABLE}")
                return [r.get("column_name") or r.get("name") or "" for r in rows]
            with prof._lock:
                return [
                    r[1]
                    for r in prof.conn.execute(
                        f"PRAGMA table_info({_MEMORY_USAGE_TABLE})"
                    ).fetchall()
                ]
        except Exception:
            return []


def _missing_memory_columns(cols: list[str]) -> list[str]:
    colset = set(cols)
    return [col for col in _MEMORY_REQUIRED_COLS if col not in colset]


def _normalize_memory_profile_row(
    row: dict,
    *,
    trim: tuple[int, int] | None,
    target_gpu: int | None,
    mem_kind_available: bool,
    distinct_contexts: int | None,
    mem_kind_breakdown: list,
) -> dict:
    baseline = int(row.get("baseline_vram_bytes") or 0)
    peak_after_event = row.get("peak_after_event_bytes")
    peak_after_event_int = int(peak_after_event) if peak_after_event is not None else baseline
    peak = max(baseline, peak_after_event_int)
    net_delta = int(row.get("net_delta_bytes") or 0)
    first_event = row.get("first_event_ns")
    last_event = row.get("last_event_ns")

    out = {
        "target_gpu": target_gpu,
        "trim_ns": list(trim) if trim else None,
        "mem_kind_available": mem_kind_available,
        "peak_vram_bytes": peak,
        "baseline_vram_bytes": baseline,
        "final_vram_bytes": baseline + net_delta,
        "allocated_bytes": int(row.get("allocated_bytes") or 0),
        "freed_bytes": int(row.get("freed_bytes") or 0),
        "net_delta_bytes": net_delta,
        "alloc_count": int(row.get("alloc_count") or 0),
        "free_count": int(row.get("free_count") or 0),
        "event_count": int(row.get("event_count") or 0),
        "host_event_count": int(row.get("host_event_count") or 0),
        "unknown_event_count": int(row.get("unknown_event_count") or 0),
    }
    if distinct_contexts is not None:
        out["distinct_contexts"] = distinct_contexts
    if mem_kind_breakdown:
        out["mem_kind_breakdown"] = mem_kind_breakdown
    if first_event is not None and last_event is not None:
        out["event_window_ns"] = [int(first_event), int(last_event)]
    return out


def _query_memory_profile(
    prof: Profile,
    trim: tuple[int, int] | None,
    target_gpu: int | None,
    cols: list[str],
) -> dict:
    colset = set(cols)
    has_mem_kind = "memKind" in colset
    has_context = "contextId" in colset

    scope_where = "1=1"
    scope_params: list = []
    if target_gpu is not None:
        scope_where = "deviceId = ?"
        scope_params = [int(target_gpu)]

    # Events CTE pulls everything up to the window end so the running balance and
    # the pre-window baseline are correct; the window filter is applied afterwards.
    events_where = scope_where
    events_params = list(scope_params)
    if trim is not None:
        events_where += " AND start <= ?"
        events_params.append(int(trim[1]))

    # In-window raw filter, for the per-kind / per-context breakdown sub-queries.
    inwin_where = scope_where
    inwin_params = list(scope_params)
    if trim is not None:
        inwin_where += " AND start >= ? AND start <= ?"
        inwin_params += [int(trim[0]), int(trim[1])]

    if trim is not None:
        window_where = "event_start >= ? AND event_start <= ?"
        window_params: list = [int(trim[0]), int(trim[1])]
        baseline_expr = (
            "(SELECT COALESCE(SUM(delta_bytes), 0) FROM events WHERE event_start < ?) "
            "AS baseline_vram_bytes"
        )
        baseline_params: list = [int(trim[0])]
    else:
        window_where = "1=1"
        window_params = []
        baseline_expr = "0 AS baseline_vram_bytes"
        baseline_params = []

    # memKind 0/1 = host (pageable/pinned), never VRAM; 2/3 device, 4/6 managed,
    # 5 device-static are device-resident (documented Nsight ENUM_CUDA_MEM_KIND).
    # Without the column we cannot tell, so count every event (best effort).
    is_device = (
        "CASE WHEN CAST(memKind AS INTEGER) IN (0, 1) THEN 0 ELSE 1 END" if has_mem_kind else "1"
    )

    # memoryOperationType: 0 = alloc, 1 = free. Allocs sort before frees at equal
    # timestamps so a same-ts alloc+free still registers the high-water mark.
    sql = f"""
        WITH events AS (
            SELECT
                CAST(start AS BIGINT) AS event_start,
                CAST(memoryOperationType AS INTEGER) AS op,
                CAST(COALESCE(bytes, 0) AS BIGINT) AS bytes,
                {is_device} AS is_device,
                CASE
                    WHEN {is_device} = 0 THEN 0
                    WHEN memoryOperationType = 0 THEN CAST(COALESCE(bytes, 0) AS BIGINT)
                    WHEN memoryOperationType = 1 THEN -CAST(COALESCE(bytes, 0) AS BIGINT)
                    ELSE 0
                END AS delta_bytes
            FROM {_MEMORY_USAGE_TABLE}
            WHERE {events_where}
        ),
        running AS (
            SELECT
                event_start,
                op,
                bytes,
                is_device,
                delta_bytes,
                SUM(delta_bytes) OVER (
                    ORDER BY event_start, CASE WHEN op = 0 THEN 0 ELSE 1 END
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS balance_bytes
            FROM events
        ),
        windowed AS (
            SELECT * FROM running WHERE {window_where}
        )
        SELECT
            COUNT(*) AS event_count,
            COALESCE(SUM(CASE WHEN is_device = 0 THEN 1 ELSE 0 END), 0) AS host_event_count,
            COALESCE(SUM(CASE WHEN is_device = 1 AND op = 0 THEN 1 ELSE 0 END), 0) AS alloc_count,
            COALESCE(SUM(CASE WHEN is_device = 1 AND op = 1 THEN 1 ELSE 0 END), 0) AS free_count,
            COALESCE(SUM(CASE WHEN op IS NULL OR op NOT IN (0, 1) THEN 1 ELSE 0 END), 0)
                AS unknown_event_count,
            COALESCE(SUM(CASE WHEN is_device = 1 AND op = 0 THEN bytes ELSE 0 END), 0)
                AS allocated_bytes,
            COALESCE(SUM(CASE WHEN is_device = 1 AND op = 1 THEN bytes ELSE 0 END), 0)
                AS freed_bytes,
            COALESCE(SUM(delta_bytes), 0) AS net_delta_bytes,
            MIN(event_start) AS first_event_ns,
            MAX(event_start) AS last_event_ns,
            MAX(balance_bytes) AS peak_after_event_bytes,
            {baseline_expr}
        FROM windowed
    """
    row = prof._duckdb_query(sql, events_params + window_params + baseline_params)[0]

    distinct_contexts = None
    if has_context:
        ctx_rows = prof._duckdb_query(
            f"SELECT COUNT(DISTINCT contextId) AS n FROM {_MEMORY_USAGE_TABLE} WHERE {inwin_where}",
            list(inwin_params),
        )
        distinct_contexts = int(ctx_rows[0].get("n") or 0) if ctx_rows else 0

    mem_kind_breakdown: list = []
    if has_mem_kind:
        bk_rows = prof._duckdb_query(
            f"""
            SELECT
                CAST(memKind AS INTEGER) AS mem_kind,
                COALESCE(SUM(CASE WHEN memoryOperationType = 0 THEN 1 ELSE 0 END), 0) AS alloc_count,
                COALESCE(SUM(CASE WHEN memoryOperationType = 1 THEN 1 ELSE 0 END), 0) AS free_count
            FROM {_MEMORY_USAGE_TABLE}
            WHERE {inwin_where}
            GROUP BY CAST(memKind AS INTEGER)
            ORDER BY mem_kind
            """,
            list(inwin_params),
        )
        for r in bk_rows:
            mk = r.get("mem_kind")
            mk_int = int(mk) if mk is not None else None
            mem_kind_breakdown.append(
                {
                    "mem_kind": mk_int,
                    "is_host": mk_int in _HOST_MEM_KINDS if mk_int is not None else False,
                    "alloc_count": int(r.get("alloc_count") or 0),
                    "free_count": int(r.get("free_count") or 0),
                }
            )

    return _normalize_memory_profile_row(
        row,
        trim=trim,
        target_gpu=target_gpu,
        mem_kind_available=has_mem_kind,
        distinct_contexts=distinct_contexts,
        mem_kind_breakdown=mem_kind_breakdown,
    )


def _build_memory_delta(before: dict, after: dict) -> dict:
    delta: dict = {}
    for metric in _MEMORY_DELTA_FIELDS:
        delta[metric] = _delta_entry(before.get(metric), after.get(metric))
    return delta


def _memory_columns_used(cols: list[str]) -> list[str]:
    used = [col for col in _MEMORY_REQUIRED_COLS if col in cols]
    for opt in ("deviceId", "memKind", "contextId"):
        if opt in cols:
            used.append(opt)
    return used


def _memory_profile_explanation(before: dict, after: dict, delta: dict) -> str:
    if before["event_count"] == 0 and after["event_count"] == 0:
        if before["baseline_vram_bytes"] or after["baseline_vram_bytes"]:
            return (
                "No GPU alloc/free events occurred in the selected window; peak VRAM "
                "stayed at the pre-window balance."
            )
        return (
            "Memory capture is present, but no GPU alloc/free events were found "
            "in the selected window."
        )

    parts: list[str] = []
    peak_delta = delta["peak_vram_bytes"]["delta"]
    net_delta_change = delta["net_delta_bytes"]["delta"]
    alloc_count_delta = delta["alloc_count"]["delta"]
    free_count_delta = delta["free_count"]["delta"]

    if peak_delta:
        parts.append(
            "peak VRAM "
            f"{_format_bytes(before['peak_vram_bytes'])} -> "
            f"{_format_bytes(after['peak_vram_bytes'])} "
            f"({_format_signed_bytes(peak_delta)})"
        )
    if alloc_count_delta or free_count_delta:
        parts.append(
            "alloc/free events "
            f"{before['alloc_count']}/{before['free_count']} -> "
            f"{after['alloc_count']}/{after['free_count']}"
        )
    if net_delta_change or before["net_delta_bytes"] or after["net_delta_bytes"]:
        parts.append(
            "net allocation "
            f"{_format_signed_bytes(before['net_delta_bytes'])} -> "
            f"{_format_signed_bytes(after['net_delta_bytes'])}"
        )

    if not parts:
        return (
            "GPU memory profile is unchanged; this diff does not explain the regression "
            "via allocation footprint or allocator churn."
        )

    if isinstance(peak_delta, int) and peak_delta > 0:
        if isinstance(alloc_count_delta, int) and alloc_count_delta > 0:
            suffix = (
                " -> higher peak plus more allocation churn suggests memory pressure "
                "or allocator churn."
            )
        else:
            suffix = " -> higher peak VRAM suggests increased memory pressure."
    elif isinstance(peak_delta, int) and peak_delta < 0:
        suffix = " -> lower peak VRAM reduces memory pressure."
    elif isinstance(net_delta_change, int) and net_delta_change > 0:
        suffix = (
            " -> more memory remains allocated by window end; check retained tensors "
            "or allocator caching."
        )
    elif (
        isinstance(alloc_count_delta, int)
        and isinstance(free_count_delta, int)
        and (alloc_count_delta > 0 or free_count_delta > 0)
    ):
        suffix = " -> allocation/free churn changed; inspect cudaMalloc/cudaFree behavior."
    else:
        suffix = " -> memory behavior changed; inspect allocation lifetime and caching."
    return "; ".join(parts) + suffix


def get_memory_profile_diff(
    ctx: DiffContext,
    iteration_index: int | None = None,
    target_gpu: int | None = None,
) -> dict:
    """
    Peak VRAM + alloc/free count (Stage 7). Returns error when memory capture not present.
    """
    before_has = _MEMORY_USAGE_TABLE in ctx.before.schema.tables
    after_has = _MEMORY_USAGE_TABLE in ctx.after.schema.tables
    before_cols = _memory_table_columns(ctx.before) if before_has else []
    after_cols = _memory_table_columns(ctx.after) if after_has else []
    if not before_has or not after_has:
        return {
            "error": "not available",
            "reason": "Memory profiling not enabled in both profiles (table missing)",
            "tables_present": {"before": before_has, "after": after_has},
            "available_columns": {"before": before_cols, "after": after_cols},
        }

    missing_before = _missing_memory_columns(before_cols)
    missing_after = _missing_memory_columns(after_cols)
    if missing_before or missing_after:
        return {
            "error": "not available",
            "reason": "Memory profiling table is missing required columns",
            "missing_columns": {"before": missing_before, "after": missing_after},
            "available_columns": {"before": before_cols, "after": after_cols},
        }

    if target_gpu is not None and ("deviceId" not in before_cols or "deviceId" not in after_cols):
        return {
            "error": "not available",
            "reason": "target_gpu filtering requires deviceId in memory profiling table",
            "available_columns": {"before": before_cols, "after": after_cols},
        }

    trim_before, trim_after, window_error = _resolve_diff_windows(
        ctx, iteration_index, target_gpu
    )
    if window_error is not None:
        return window_error

    before_stats = _query_memory_profile(ctx.before, trim_before, target_gpu, before_cols)
    after_stats = _query_memory_profile(ctx.after, trim_after, target_gpu, after_cols)
    delta = _build_memory_delta(before_stats, after_stats)
    explanation = _memory_profile_explanation(before_stats, after_stats, delta)
    if not (before_stats["mem_kind_available"] and after_stats["mem_kind_available"]):
        explanation += (
            " Note: the memKind column is absent, so host vs device memory could not be "
            "separated — these figures may include non-VRAM (host) allocations."
        )
    return {
        "target_gpu": target_gpu,
        "iteration_index": iteration_index,
        "trim_before_ns": list(trim_before) if trim_before else None,
        "trim_after_ns": list(trim_after) if trim_after else None,
        "table": _MEMORY_USAGE_TABLE,
        "columns_used": {
            "before": _memory_columns_used(before_cols),
            "after": _memory_columns_used(after_cols),
        },
        "before": before_stats,
        "after": after_stats,
        "delta": delta,
        "explanation": explanation,
    }


# ── Phase C system prompt and tool metadata for agent integration ─────────────

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_diff_system_prompt() -> str:
    """Load the diff system prompt, falling back to a minimal prompt if file is missing."""
    _fallback = (
        "You are a performance analysis assistant for Nsight Systems profile diffs. "
        "Provide concise, technically accurate analysis based solely on the supplied tool outputs."
    )
    prompt_path = _PROMPTS_DIR / "diff_system.txt"
    try:
        return prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        _log_dt = logging.getLogger(__name__)
        _log_dt.warning(
            "diff_tools: diff_system.txt not found at %s; using minimal fallback prompt",
            prompt_path,
        )
        return _fallback
    except OSError as _exc:
        _log_dt = logging.getLogger(__name__)
        _log_dt.warning(
            "diff_tools: could not read diff_system.txt at %s: %s; using minimal fallback prompt",
            prompt_path,
            _exc,
        )
        return _fallback


DIFF_SYSTEM_PROMPT = _load_diff_system_prompt()

# Dynamic override: load skills/diff.md when available (falls back to hardcoded prompt).
try:
    from .prompt_loader import load_skill as _load_skill_diff

    _diff_md = _load_skill_diff("skills/diff.md")
    if _diff_md:
        # Preserve the fallback for debugging; expose loaded version as DIFF_SYSTEM_PROMPT
        _DIFF_SYSTEM_PROMPT_FALLBACK = DIFF_SYSTEM_PROMPT
        DIFF_SYSTEM_PROMPT = _diff_md
except (ImportError, OSError) as _exc:
    import logging as _logging

    _logging.getLogger(__name__).debug("diff_tools: could not load skills/diff.md: %s", _exc)


TOOL_DESCRIPTIONS = {
    "search_nvtx_regions": "Fuzzy NVTX name discovery (LIKE/GLOB). Call BEFORE any region diff to get exact strings.",
    "get_iteration_boundaries": "Per-iteration time windows for both profiles + is_aligned. Use to pick iteration_index.",
    "explore_nvtx_hierarchy": "Step-by-step NVTX tree navigation. Use to discover exact paths before get_region_diff.",
    "get_top_nvtx_diffs": "Hotspot radar: top NVTX regions by absolute time change.",
    "get_iteration_diff": "Macro diff for one iteration (time-window from detect_iterations).",
    "get_region_diff": "Micro diff for a code region. WARNING: Call search_nvtx_regions first to get exact NVTX name.",
    "summarize_nvtx_subtree": "Roll up all depths under parent path; return top N by delta.",
    "get_launch_config_diff": "Grid/block/registers before vs after; explains 'why'. Returns error when columns missing.",
    "get_source_code_context": "filename:line for Coder Agent handoff. Returns error when SOURCE_LOCATOR missing.",
    "get_gpu_imbalance_stats": "Only call when NCCL time spiked. Per-GPU compute/nccl/idle for straggler proof.",
    "get_global_diff": "Time-based fallback when no iteration marker.",
    "get_memory_profile_diff": "Peak VRAM + alloc/free (optional). Returns error when memory capture not present.",
}

# OpenAI-style tool list for diff-chat (Stage 6). Used by chat.py when diff_context is set.
TOOLS_DIFF_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "search_nvtx_regions",
            "description": TOOL_DESCRIPTIONS["search_nvtx_regions"],
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Substring or pattern for NVTX name search",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 50)",
                        "default": 50,
                    },
                    "use_glob": {
                        "type": "boolean",
                        "description": "Use GLOB instead of LIKE",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_iteration_boundaries",
            "description": TOOL_DESCRIPTIONS["get_iteration_boundaries"],
            "parameters": {
                "type": "object",
                "properties": {
                    "marker": {
                        "type": "string",
                        "description": "NVTX marker for iteration (e.g. %sample_0%)",
                    },
                    "target_gpu": {
                        "type": "integer",
                        "description": "GPU ID for iteration detection; omit to use the first device",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explore_nvtx_hierarchy",
            "description": TOOL_DESCRIPTIONS["explore_nvtx_hierarchy"],
            "parameters": {
                "type": "object",
                "properties": {
                    "parent_path": {
                        "type": "string",
                        "description": "Parent NVTX path (empty for root)",
                        "default": "",
                    },
                    "depth": {"type": "integer", "description": "Depth to expand", "default": 1},
                    "target_gpu": {
                        "type": "integer",
                        "description": "GPU ID for the NVTX tree; omit to use the first device",
                    },
                    "profile_side": {
                        "type": "string",
                        "enum": ["before", "after"],
                        "default": "after",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_nvtx_diffs",
            "description": TOOL_DESCRIPTIONS["get_top_nvtx_diffs"],
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20},
                    "target_gpu": {
                        "type": "integer",
                        "description": "GPU ID; omit to aggregate all GPUs",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_iteration_diff",
            "description": TOOL_DESCRIPTIONS["get_iteration_diff"],
            "parameters": {
                "type": "object",
                "properties": {
                    "iteration_index": {
                        "type": "integer",
                        "description": "0-based iteration index (from get_iteration_boundaries)",
                    },
                    "marker": {"type": "string", "description": "Iteration marker"},
                    "target_gpu": {
                        "type": "integer",
                        "description": "GPU ID; omit to aggregate all GPUs",
                    },
                },
                "required": ["iteration_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_region_diff",
            "description": TOOL_DESCRIPTIONS["get_region_diff"],
            "parameters": {
                "type": "object",
                "properties": {
                    "nvtx_exact_match": {
                        "description": "Exact NVTX name or list of path segments",
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                    },
                    "iteration_index": {
                        "type": "integer",
                        "description": "0-based iteration (optional)",
                    },
                    "target_gpu": {
                        "type": "integer",
                        "description": "GPU ID; omit to aggregate all GPUs",
                    },
                },
                "required": ["nvtx_exact_match"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_nvtx_subtree",
            "description": TOOL_DESCRIPTIONS["summarize_nvtx_subtree"],
            "parameters": {
                "type": "object",
                "properties": {
                    "parent_path": {"type": "string"},
                    "iteration_index": {"type": "integer"},
                    "target_gpu": {
                        "type": "integer",
                        "description": "GPU ID; omit to aggregate all GPUs",
                    },
                    "top_n": {"type": "integer", "default": 3},
                },
                "required": ["parent_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_launch_config_diff",
            "description": TOOL_DESCRIPTIONS["get_launch_config_diff"],
            "parameters": {
                "type": "object",
                "properties": {
                    "kernel_name": {"type": "string"},
                    "iteration_index": {"type": "integer"},
                    "target_gpu": {
                        "type": "integer",
                        "description": "GPU ID; omit to aggregate all GPUs",
                    },
                },
                "required": ["kernel_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_source_code_context",
            "description": TOOL_DESCRIPTIONS["get_source_code_context"],
            "parameters": {
                "type": "object",
                "properties": {"nvtx_path": {"type": "string"}},
                "required": ["nvtx_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_gpu_imbalance_stats",
            "description": TOOL_DESCRIPTIONS["get_gpu_imbalance_stats"],
            "parameters": {
                "type": "object",
                "properties": {
                    "iteration_index": {"type": "integer"},
                    "marker": {"type": "string"},
                },
                "required": ["iteration_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_global_diff",
            "description": TOOL_DESCRIPTIONS["get_global_diff"],
            "parameters": {
                "type": "object",
                "properties": {
                    "skip_first_ms": {"type": "number", "default": 0},
                    "duration_ms": {
                        "type": "number",
                        "description": "Window length in ms or omit for full",
                    },
                    "target_gpu": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_memory_profile_diff",
            "description": TOOL_DESCRIPTIONS["get_memory_profile_diff"],
            "parameters": {
                "type": "object",
                "properties": {
                    "iteration_index": {"type": "integer"},
                    "target_gpu": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    TOOL_GET_GPU_PEAK_TFLOPS,
    TOOL_COMPUTE_MFU,
]


def run_diff_tool(ctx: DiffContext, name: str, arguments: dict) -> dict:
    """Execute a Phase C tool by name with parsed arguments. Returns a JSON-serialisable dict."""
    args_str = json.dumps(arguments) if arguments else "{}"
    from .tool_dispatch import ToolDispatcher

    dispatcher = ToolDispatcher(mode="diff", diff_context=ctx)
    try:
        res = dispatcher.dispatch(name, args_str)
    except (TypeError, ValueError, KeyError) as e:
        return {"error": "invalid_arguments", "message": str(e)}
    except Exception as e:
        return {"error": "internal_error", "message": "Tool execution failed", "detail": str(e)}
    try:
        j = json.loads(res.content)
        return j if isinstance(j, dict) else {"result": j}
    except json.JSONDecodeError:
        return {"error": res.content}


def build_diff_system_prompt(
    ctx: DiffContext,
    before_path: str,
    after_path: str,
    snapshot: dict | None = None,
) -> str:
    """Build system prompt for diff-chat: Phase C rules + paths + optional proactive snapshot."""
    parts = [
        DIFF_SYSTEM_PROMPT.strip(),
        "",
        f"Before profile: {before_path}",
        f"After profile: {after_path}",
    ]
    if ctx.marker:
        parts.append(f"Iteration marker: {ctx.marker}")
    if snapshot:
        parts.append("")
        parts.append("Proactive snapshot (first iteration or global):")
        parts.append(json.dumps(snapshot, indent=2))
    return "\n".join(parts)


def _gpu_clock_mhz(prof: Profile) -> float | None:
    """Best-effort read of GPU clock (MHz) from META_DATA. Returns None if not available."""
    for table in ("META_DATA_CAPTURE", "META_DATA_EXPORT"):
        if table not in prof.schema.tables:
            continue
        kv = prof.schema._read_kv_table(table)
        for key, val in kv.items():
            k = key.lower()
            if "clock" in k or "frequency" in k:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
    return None


def _hardware_warning(ctx: DiffContext) -> bool:
    """True when both profiles have GPU clock and relative diff >5% (Stage 5)."""
    c_b = _gpu_clock_mhz(ctx.before)
    c_a = _gpu_clock_mhz(ctx.after)
    if c_b is None or c_a is None or c_b <= 0:
        return False
    return abs(c_a - c_b) / max(c_b, c_a) > 0.05


def _has_overhead_in_window(prof: Profile, trim: tuple[int, int]) -> bool:
    """True if CUPTI_ACTIVITY_KIND_OVERHEAD has any row overlapping the time window (Stage 5)."""
    if "CUPTI_ACTIVITY_KIND_OVERHEAD" not in prof.schema.tables or not trim:
        return False
    try:
        rows = prof._duckdb_query(
            "SELECT 1 FROM CUPTI_ACTIVITY_KIND_OVERHEAD WHERE start < ? AND [end] > ? LIMIT 1",
            [trim[1], trim[0]],
        )
        return len(rows) > 0
    except Exception:
        return False


def _time_window_ns(
    prof: Profile, skip_first_ms: float, duration_ms: float | None
) -> tuple[int, int] | None:
    t0, t1 = prof.meta.time_range
    if t0 is None or t1 is None:
        return None
    skip_ns = int(skip_first_ms * 1e6)
    start_ns = t0 + skip_ns
    if start_ns >= t1:
        return None
    if duration_ms is not None:
        end_ns = start_ns + int(duration_ms * 1e6)
        end_ns = min(end_ns, t1)
    else:
        end_ns = t1
    return (start_ns, end_ns)
