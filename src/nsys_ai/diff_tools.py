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
from dataclasses import dataclass, field

from .ai.backend.chat_tools import TOOL_COMPUTE_MFU, TOOL_GET_GPU_PEAK_TFLOPS
from .diff import ProfileDiffSummary, diff_profiles
from .hardware import get_peak_tflops
from .mfu import compute_mfu_from_args
from .nvtx_tree import build_nvtx_tree
from .overlap import detect_iterations, overlap_analysis
from .profile import Profile


def _top_k_payload(summary: ProfileDiffSummary, top_n: int = 5) -> tuple[list, list, float]:
    """Build top_regressions, top_improvements with impact ratio and others_aggregated_delta_ms (Payload Contract)."""
    total_pos = sum(k.delta_ns for k in summary.kernel_diffs if k.delta_ns > 0) or 1
    total_neg = sum(k.delta_ns for k in summary.kernel_diffs if k.delta_ns < 0) or -1

    def impact(k):
        pct = (k.after_total_ns / (summary.after.total_gpu_ns or 1)) * 100 if summary.after.total_gpu_ns else 0
        contrib = (k.delta_ns / total_pos * 100) if k.delta_ns > 0 else (k.delta_ns / total_neg * 100)
        return {"pct_of_iteration_time": round(pct, 2), "contribution_to_total_delta_pct": round(contrib, 2)}

    top_keys = {k.key for k in summary.top_regressions[:top_n]} | {k.key for k in summary.top_improvements[:top_n]}
    others_ns = sum(k.delta_ns for k in summary.kernel_diffs if k.key not in top_keys)

    regressions = [
        {"name": k.name, "delta_ns": k.delta_ns, "delta_ms": round(k.delta_ns / 1e6, 3), **impact(k)}
        for k in summary.top_regressions[:top_n]
    ]
    improvements = [
        {"name": k.name, "delta_ns": k.delta_ns, "delta_ms": round(k.delta_ns / 1e6, 3), **impact(k)}
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
        by_text.setdefault(text, {"text": text, "total_ns_before": 0, "total_ns_after": 0, "count_before": 0, "count_after": 0})
        by_text[text]["total_ns_before"] = int(r.get("total_ns") or 0)
        by_text[text]["count_before"] = int(r.get("count") or 0)
    for r in after_rows:
        text = str(r.get("text") or "")
        by_text.setdefault(text, {"text": text, "total_ns_before": 0, "total_ns_after": 0, "count_before": 0, "count_after": 0})
        by_text[text]["total_ns_after"] = int(r.get("total_ns") or 0)
        by_text[text]["count_after"] = int(r.get("count") or 0)
    for v in by_text.values():
        v["in_before"] = v["count_before"] > 0
        v["in_after"] = v["count_after"] > 0
    regions = sorted(by_text.values(), key=lambda x: -(x["total_ns_before"] + x["total_ns_after"]))[:limit]
    return {
        "regions": regions,
        "query": query,
        "limit": limit,
        "use_glob": use_glob,
    }


def get_iteration_boundaries(
    ctx: DiffContext,
    marker: str | None = None,
    target_gpu: int | None = 0,
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
        boundaries.append({
            "iteration_index": i,
            "before": {"start_ns": start_ns_b, "end_ns": end_ns_b, "duration_ms": b["duration_ms"] if b else None},
            "after": {"start_ns": start_ns_a, "end_ns": end_ns_a, "duration_ms": a["duration_ms"] if a else None},
            "has_before": b is not None,
            "has_after": a is not None,
        })
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
        return [{"name": n["name"], "type": n["type"], "duration_ms": round((n["end"] - n["start"]) / 1e6, 3), "path": n["name"]} for n in roots]
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
        return [{"name": n["name"], "type": n["type"], "duration_ms": round((n["end"] - n["start"]) / 1e6, 3)} for n in current]
    out = []
    for n in current:
        if n.get("type") != "nvtx":
            continue
        path_str = " > ".join(path_parts + [n["name"]])
        out.append({
            "name": n["name"],
            "type": "nvtx",
            "duration_ms": round((n["end"] - n["start"]) / 1e6, 3),
            "path": path_str,
            "child_count": len([c for c in n.get("children", []) if c.get("type") == "nvtx"]),
        })
    return out


def explore_nvtx_hierarchy(
    ctx: DiffContext,
    parent_path: str = "",
    depth: int = 1,
    target_gpu: int | None = 0,
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
    target_gpu: int | None = 0,
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
    trim_before = (bnd["before"]["start_ns"], bnd["before"]["end_ns"]) if bnd["before"]["start_ns"] is not None else None
    trim_after = (bnd["after"]["start_ns"], bnd["after"]["end_ns"]) if bnd["after"]["start_ns"] is not None else None
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
    # Unique stream count (Payload Contract)
    gpu = target_gpu if target_gpu is not None else (ctx.before.meta.devices or [0])[0]
    kerns_b = ctx.before.kernels(gpu, trim_before)
    kerns_a = ctx.after.kernels(gpu, trim_after)
    unique_streams_b = len(set(k.get("streamId") for k in kerns_b if k.get("streamId") is not None))
    unique_streams_a = len(set(k.get("streamId") for k in kerns_a if k.get("streamId") is not None))
    # Memcpy H2D/D2H in window
    mc_b = ctx.before.memcpy_in_window(gpu, trim_before)
    mc_a = ctx.after.memcpy_in_window(gpu, trim_after)
    memcpy_ms = {
        "h2d": {"before": round(mc_b["h2d_ns"] / 1e6, 2), "after": round(mc_a["h2d_ns"] / 1e6, 2), "delta": round((mc_a["h2d_ns"] - mc_b["h2d_ns"]) / 1e6, 2)},
        "d2h": {"before": round(mc_b["d2h_ns"] / 1e6, 2), "after": round(mc_a["d2h_ns"] / 1e6, 2), "delta": round((mc_a["d2h_ns"] - mc_b["d2h_ns"]) / 1e6, 2)},
    }
    # top3_global_categories: Compute / Memcpy / Idle (Payload Contract)
    idle_b = summary.before.overlap.get("idle_ms") or 0
    idle_a = summary.after.overlap.get("idle_ms") or 0
    mem_b = (mc_b["h2d_ns"] + mc_b["d2h_ns"] + mc_b["d2d_ns"]) / 1e6
    mem_a = (mc_a["h2d_ns"] + mc_a["d2h_ns"] + mc_a["d2d_ns"]) / 1e6
    top3_global_categories = {
        "Compute": {"before": round(sum_k_b, 2), "after": round(sum_k_a, 2), "delta": round(sum_k_a - sum_k_b, 2)},
        "Memcpy": {"before": round(mem_b, 2), "after": round(mem_a, 2), "delta": round(mem_a - mem_b, 2)},
        "Idle": {"before": round(idle_b, 2), "after": round(idle_a, 2), "delta": round(idle_a - idle_b, 2)},
    }
    top_regressions_payload, top_improvements_payload, others_ms = _top_k_payload(summary, 5)
    return {
        "iteration_index": iteration_index,
        "is_aligned": bounds["is_aligned"],
        "wall_clock_ms": {"before": round(wall_b, 2), "after": round(wall_a, 2), "delta": round(wall_a - wall_b, 2)},
        "sum_of_kernels_ms": {"before": round(sum_k_b, 2), "after": round(sum_k_a, 2), "delta": round(sum_k_a - sum_k_b, 2)},
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
    summary = diff_profiles(ctx.before, ctx.after, gpu=target_gpu, trim=trim_before, limit=15, sort="delta")
    summary2 = diff_profiles(ctx.before, ctx.after, gpu=target_gpu, trim=trim_after, limit=15, sort="delta")
    wall_b = (trim_before[1] - trim_before[0]) / 1e6
    wall_a = (trim_after[1] - trim_after[0]) / 1e6
    return {
        "skip_first_ms": skip_first_ms,
        "duration_ms": duration_ms,
        "target_gpu": target_gpu,
        "wall_clock_ms": {"before": round(wall_b, 2), "after": round(wall_a, 2), "delta": round(wall_a - wall_b, 2)},
        "top_regressions": [{"name": k.name, "delta_ms": round(k.delta_ns / 1e6, 3)} for k in summary2.top_regressions[:10]],
        "top_improvements": [{"name": k.name, "delta_ms": round(k.delta_ns / 1e6, 3)} for k in summary2.top_improvements[:10]],
        "warnings": summary.warnings,
    }


def get_region_diff(
    ctx: DiffContext,
    nvtx_exact_match: str | list[str],
    iteration_index: int | None = None,
    target_gpu: int | None = 0,
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
            return {"error": f"iteration_index {iteration_index} out of range", "iteration_index": iteration_index}
        bnd = bounds["boundaries"][iteration_index]
        trim_before = (bnd["before"]["start_ns"], bnd["before"]["end_ns"]) if bnd["before"]["start_ns"] is not None else ctx.trim
        trim_after = (bnd["after"]["start_ns"], bnd["after"]["end_ns"]) if bnd["after"]["start_ns"] is not None else ctx.trim
    if trim_before is None:
        trim_before = ctx.before.meta.time_range
    if trim_after is None:
        trim_after = ctx.after.meta.time_range
    if trim_before is None or trim_after is None:
        return {"error": "Time window not available (empty profile)"}
    # Use iteration-scoped diff when trim was set from boundaries
    summary = diff_profiles(
        ctx.before, ctx.after,
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
    gpu = target_gpu if target_gpu is not None else (ctx.before.meta.devices or [0])[0]
    # Payload Contract: same defensive fields as get_iteration_diff
    mc_b = ctx.before.memcpy_in_window(gpu, trim_before)
    mc_a = ctx.after.memcpy_in_window(gpu, trim_after)
    memcpy_ms = {
        "h2d": {"before": round(mc_b["h2d_ns"] / 1e6, 2), "after": round(mc_a["h2d_ns"] / 1e6, 2), "delta": round((mc_a["h2d_ns"] - mc_b["h2d_ns"]) / 1e6, 2)},
        "d2h": {"before": round(mc_b["d2h_ns"] / 1e6, 2), "after": round(mc_a["d2h_ns"] / 1e6, 2), "delta": round((mc_a["d2h_ns"] - mc_b["d2h_ns"]) / 1e6, 2)},
    }
    kerns_b = ctx.before.kernels(gpu, trim_before)
    kerns_a = ctx.after.kernels(gpu, trim_after)
    unique_streams_b = len(set(k.get("streamId") for k in kerns_b if k.get("streamId") is not None))
    unique_streams_a = len(set(k.get("streamId") for k in kerns_a if k.get("streamId") is not None))
    idle_b = summary.before.overlap.get("idle_ms") or 0
    idle_a = summary.after.overlap.get("idle_ms") or 0
    mem_b = (mc_b["h2d_ns"] + mc_b["d2h_ns"] + mc_b["d2d_ns"]) / 1e6
    mem_a = (mc_a["h2d_ns"] + mc_a["d2h_ns"] + mc_a["d2d_ns"]) / 1e6
    sum_k_b = summary.before.total_gpu_ns / 1e6
    sum_k_a = summary.after.total_gpu_ns / 1e6
    top3_global_categories = {
        "Compute": {"before": round(sum_k_b, 2), "after": round(sum_k_a, 2), "delta": round(sum_k_a - sum_k_b, 2)},
        "Memcpy": {"before": round(mem_b, 2), "after": round(mem_a, 2), "delta": round(mem_a - mem_b, 2)},
        "Idle": {"before": round(idle_b, 2), "after": round(idle_a, 2), "delta": round(idle_a - idle_b, 2)},
    }
    top_regressions_payload, top_improvements_payload, others_ms = _top_k_payload(summary, 5)

    return {
        "nvtx_exact_match": nd.text,
        "wall_clock_ms": {"before": round(nd.before_total_ns / 1e6, 2), "after": round(nd.after_total_ns / 1e6, 2), "delta": round(nd.delta_ns / 1e6, 2)},
        "sum_of_kernels_ms": {"before": round(nd.before_total_ns / 1e6, 2), "after": round(nd.after_total_ns / 1e6, 2), "delta": round(nd.delta_ns / 1e6, 2)},
        "memcpy_ms": memcpy_ms,
        "memcpy_compute_overlap_pct": None,
        "unique_streams_count_before": unique_streams_b,
        "unique_streams_count_after": unique_streams_a,
        "classification": nd.classification,
        "top_regressions": top_regressions_payload,
        "top_improvements": top_improvements_payload,
        "others_aggregated_delta_ms": others_ms,
        "top3_global_categories": top3_global_categories,
        "overlap_pct": {"before": summary.before.overlap.get("overlap_pct"), "after": summary.after.overlap.get("overlap_pct")},
        "workload_warning": False,
        "Hardware_Warning": _hardware_warning(ctx),
        "JIT_Compilation_Warning": (
            _has_overhead_in_window(ctx.before, trim_before) or _has_overhead_in_window(ctx.after, trim_after)
        ),
    }


def summarize_nvtx_subtree(
    ctx: DiffContext,
    parent_path: str,
    iteration_index: int | None = None,
    target_gpu: int | None = 0,
    top_n: int = 3,
) -> dict:
    """
    Roll up all depths under parent; return top N by delta (macro→micro in one call).
    """
    summary = ctx.ensure_summary(target_gpu)
    # Filter nvtx_diffs to those whose text equals parent_path or starts with parent_path + " >"
    prefix = (parent_path.rstrip(" >") + " >") if parent_path.strip() else ""
    candidates = [n for n in summary.nvtx_diffs if not prefix or n.text == parent_path.strip() or n.text.startswith(prefix)]
    by_abs = sorted(candidates, key=lambda n: abs(n.delta_ns), reverse=True)
    top_deltas = [
        {"text": n.text, "delta_ns": n.delta_ns, "delta_ms": round(n.delta_ns / 1e6, 3), "classification": n.classification}
        for n in by_abs[:top_n]
    ]
    return {
        "parent_path": parent_path,
        "iteration_index": iteration_index,
        "target_gpu": target_gpu,
        "top_by_delta": top_deltas,
        "total_matching": len(candidates),
    }


def get_launch_config_diff(
    ctx: DiffContext,
    kernel_name: str,
    iteration_index: int | None = None,
    target_gpu: int | None = 0,
) -> dict:
    """
    Grid/block/registers before vs after; explains 'why'. Returns error when columns missing (BETA).
    """
    # Launch-config columns are BETA; detect with PRAGMA table_info
    kt = ctx.before.schema.kernel_table
    with ctx.before._lock:
        cols = [r[1] for r in ctx.before.conn.execute(f"PRAGMA table_info({kt})").fetchall()]
    if not ("gridX" in cols or "blockX" in cols):
        return {"error": "not available", "reason": "Launch-config columns (gridX/blockX, etc.) not in export"}
    # Optional: query one kernel by name in trim window and return before/after config
    return {
        "kernel_name": kernel_name,
        "before": {},
        "after": {},
        "uses_tensor_core_likely": any(
            p in kernel_name for p in ("sm80_xmma_", "volta_fp16_s884gemm", "h884gemm", "xmma", "wmma")
        ),
        "error": "not available",
        "reason": "Launch-config diff not yet implemented; schema present",
    }


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
    trim_b = (bnd["before"]["start_ns"], bnd["before"]["end_ns"]) if bnd["before"]["start_ns"] else None
    trim_a = (bnd["after"]["start_ns"], bnd["after"]["end_ns"]) if bnd["after"]["start_ns"] else None
    devices = sorted(set(ctx.before.meta.devices or [0]) | set(ctx.after.meta.devices or [0]))
    per_gpu = []
    for dev in devices:
        ob = overlap_analysis(ctx.before, dev, trim_b) if trim_b else {"compute_only_ms": 0, "nccl_only_ms": 0, "idle_ms": 0}
        oa = overlap_analysis(ctx.after, dev, trim_a) if trim_a else {"compute_only_ms": 0, "nccl_only_ms": 0, "idle_ms": 0}
        if isinstance(ob, dict) and "error" in ob:
            ob = {"compute_only_ms": 0, "nccl_only_ms": 0, "idle_ms": 0}
        if isinstance(oa, dict) and "error" in oa:
            oa = {"compute_only_ms": 0, "nccl_only_ms": 0, "idle_ms": 0}
        per_gpu.append({
            "gpu_id": dev,
            "before": {"compute_ms": ob.get("compute_only_ms", 0), "nccl_ms": ob.get("nccl_only_ms", 0), "idle_ms": ob.get("idle_ms", 0)},
            "after": {"compute_ms": oa.get("compute_only_ms", 0), "nccl_ms": oa.get("nccl_only_ms", 0), "idle_ms": oa.get("idle_ms", 0)},
        })
    return {"iteration_index": iteration_index, "per_gpu": per_gpu}


def get_memory_profile_diff(
    ctx: DiffContext,
    iteration_index: int | None = None,
    target_gpu: int | None = None,
) -> dict:
    """
    Peak VRAM + alloc/free count (Stage 7). Returns error when memory capture not present.
    """
    if "CUDA_GPU_MEMORY_USAGE_EVENTS" not in ctx.before.schema.tables:
        return {"error": "not available", "reason": "Memory profiling not enabled (table missing)"}
    return {"error": "not available", "reason": "Memory diff not yet implemented"}


# ── Phase C system prompt and tool metadata for agent integration ─────────────

DIFF_SYSTEM_PROMPT = """
You are a Senior MLSys Performance Engineer analyzing Nsight Systems (nsys) profile diffs.
You have access to before/after SQLite profiles and MUST use the following tools in order.

1. **Never guess names** — Call search_nvtx_regions or explore_nvtx_hierarchy to get exact NVTX/kernel strings before any diff call.
2. **Wall-clock vs kernel sum** — If they diverge, conclude stream serialization or external sync, not kernel regression.
3. **Explain "why"** — For regressed kernels, call get_launch_config_diff when available; if kernel sped up with no config change, check uses_tensor_core_likely.
4. **Strict modality (nsys, not ncu)** — No cache hit rate, bandwidth, or bank-conflict claims; tell user to use Nsight Compute for those.
5. **NCCL spike → imbalance first** — Not "network"; use get_gpu_imbalance_stats to prove; if within-node GPUs are balanced but NCCL still high, conclude cross-node delay.
6. **Idle spike → CPU starvation** — If iteration slower but sum_of_kernels_ms unchanged and idle spiked, steer toward DataLoader / Python overhead.
7. **Overlap caution** — If a kernel or region got faster but overlap_pct is high, warn that E2E speedup may be smaller or zero.
8. **Hardware_Warning present** — Prefer thermal/power explanation before software regression.
9. **Workload_Mismatch_Warning** — Do not draw a performance conclusion; tell user the input dimensions may differ.
10. **Impact ratio** — Check pct_of_iteration_time and contribution_to_total_delta_pct; if regression is <1% of iteration time, classify as Negligible Variance.
11. **MFU** — (1) Get step_time_s from get_iteration_diff (wall_clock_ms/1000) or get_global_diff. (2) Call get_gpu_peak_tflops to get peak_tflops from the profile GPU; if it returns an error, ask the user for peak_tflops. (3) Ask the user for model_flops_per_step (nsys does not store it). **Do NOT call compute_mfu until the user has provided model_flops_per_step** — after asking, end your response and wait for their next message; only then call compute_mfu with the value they provided. (4) For before/after: call compute_mfu twice (before_s, flops, peak) and (after_s, flops, peak); synthesize (e.g. \"MFU before 35%, after 75%, +40%\").
"""

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
                    "query": {"type": "string", "description": "Substring or pattern for NVTX name search"},
                    "limit": {"type": "integer", "description": "Max results (default 50)", "default": 50},
                    "use_glob": {"type": "boolean", "description": "Use GLOB instead of LIKE", "default": False},
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
                    "marker": {"type": "string", "description": "NVTX marker for iteration (e.g. %sample_0%)"},
                    "target_gpu": {"type": "integer", "description": "GPU ID or omit for default"},
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
                    "parent_path": {"type": "string", "description": "Parent NVTX path (empty for root)", "default": ""},
                    "depth": {"type": "integer", "description": "Depth to expand", "default": 1},
                    "target_gpu": {"type": "integer", "description": "GPU ID"},
                    "profile_side": {"type": "string", "enum": ["before", "after"], "default": "after"},
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
                    "target_gpu": {"type": "integer", "description": "GPU ID or omit for all GPUs"},
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
                    "iteration_index": {"type": "integer", "description": "0-based iteration index (from get_iteration_boundaries)"},
                    "marker": {"type": "string", "description": "Iteration marker"},
                    "target_gpu": {"type": "integer", "default": 0},
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
                        "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
                    },
                    "iteration_index": {"type": "integer", "description": "0-based iteration (optional)"},
                    "target_gpu": {"type": "integer", "default": 0},
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
                    "target_gpu": {"type": "integer", "default": 0},
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
                    "target_gpu": {"type": "integer", "default": 0},
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
                    "duration_ms": {"type": "number", "description": "Window length in ms or omit for full"},
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
    args = dict(arguments) if arguments else {}
    try:
        if name == "search_nvtx_regions":
            return search_nvtx_regions(ctx, args.get("query", ""), args.get("limit", 50), args.get("use_glob", False))
        if name == "get_iteration_boundaries":
            return get_iteration_boundaries(ctx, args.get("marker"), args.get("target_gpu", 0))
        if name == "explore_nvtx_hierarchy":
            return explore_nvtx_hierarchy(
                ctx,
                args.get("parent_path", ""),
                args.get("depth", 1),
                args.get("target_gpu", 0),
                args.get("profile_side", "after"),
            )
        if name == "get_top_nvtx_diffs":
            return get_top_nvtx_diffs(ctx, args.get("limit", 20), args.get("target_gpu"))
        if name == "get_iteration_diff":
            return get_iteration_diff(
                ctx,
                int(args["iteration_index"]),
                args.get("marker"),
                args.get("target_gpu", 0),
            )
        if name == "get_region_diff":
            nvtx = args.get("nvtx_exact_match")
            if isinstance(nvtx, list):
                pass
            elif isinstance(nvtx, str):
                nvtx = nvtx
            else:
                nvtx = ""
            return get_region_diff(
                ctx,
                nvtx,
                args.get("iteration_index"),
                args.get("target_gpu", 0),
            )
        if name == "summarize_nvtx_subtree":
            return summarize_nvtx_subtree(
                ctx,
                args.get("parent_path", ""),
                args.get("iteration_index"),
                args.get("target_gpu", 0),
                args.get("top_n", 3),
            )
        if name == "get_launch_config_diff":
            return get_launch_config_diff(
                ctx,
                args.get("kernel_name", ""),
                args.get("iteration_index"),
                args.get("target_gpu", 0),
            )
        if name == "get_source_code_context":
            return get_source_code_context(ctx, args.get("nvtx_path", ""))
        if name == "get_gpu_imbalance_stats":
            return get_gpu_imbalance_stats(ctx, int(args.get("iteration_index", 0)), args.get("marker"))
        if name == "get_global_diff":
            return get_global_diff(
                ctx,
                args.get("skip_first_ms", 0),
                args.get("duration_ms"),
                args.get("target_gpu"),
            )
        if name == "get_memory_profile_diff":
            return get_memory_profile_diff(
                ctx,
                args.get("iteration_index"),
                args.get("target_gpu"),
            )
        if name == "get_gpu_peak_tflops":
            devs = ctx.after.meta.devices or [0]
            gpu_name = ""
            if devs:
                gi = ctx.after.meta.gpu_info.get(devs[0])
                if gi:
                    gpu_name = gi.name or ""
            return get_peak_tflops(gpu_name)
        if name == "compute_mfu":
            return compute_mfu_from_args(args)
    except (KeyError, TypeError, ValueError) as e:
        return {"error": "invalid arguments", "detail": str(e)}
    except Exception as e:
        return {"error": str(e)}
    return {"error": "unknown tool", "name": name}


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
        with prof._lock:
            r = prof.conn.execute(
                "SELECT 1 FROM CUPTI_ACTIVITY_KIND_OVERHEAD WHERE start < ? AND [end] > ? LIMIT 1",
                (trim[1], trim[0]),
            ).fetchone()
        return r is not None
    except Exception:
        return False


def _time_window_ns(prof: Profile, skip_first_ms: float, duration_ms: float | None) -> tuple[int, int] | None:
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
