"""Per-NVTX-region GPU time breakdown.

Attributes GPU kernels to their parent NVTX regions via the efficient
nvtx_attribution module (nsys recipe primary, sort-merge fallback),
producing a flat table of "which code region spent the most GPU time".

This enables the agent to say "Layer 12 Attention backward has 15ms
NCCL stall" instead of "some stall at timestamp X".

Features:
  - Auto-detects layer-level NVTX depth (numbered layers, repeated ops)
  - Compute vs NCCL split per region
  - Top-3 hotspot kernels per region (embedded in JSON output)
  - Cross-layer outlier detection (IQR + median dual threshold)
"""

from collections import defaultdict

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    """Execute NVTX region GPU time breakdown via attribution module."""
    from ...nvtx_attribution import attribute_kernels_to_nvtx
    from ...nvtx_layer_detect import detect_layer_depth, is_outlier
    from ...overlap import classify_kernel

    limit = int(kwargs.get("limit", 20))
    depth = kwargs.get("depth")
    if depth is not None:
        depth = int(depth)

    raw_auto_depth = kwargs.get("auto_depth", True)
    if isinstance(raw_auto_depth, bool):
        auto_depth = raw_auto_depth
    elif isinstance(raw_auto_depth, str):
        token = raw_auto_depth.strip().lower()
        true_tokens = {"true", "1", "yes", "y", "on"}
        false_tokens = {"false", "0", "no", "n", "off"}
        if token in true_tokens:
            auto_depth = True
        elif token in false_tokens:
            auto_depth = False
        else:
            return [{"error": f"Invalid value for auto_depth: {raw_auto_depth!r}. "
                              f"Expected one of {sorted(true_tokens | false_tokens)}."}]
    else:
        auto_depth = bool(raw_auto_depth)

    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    trim = (trim_start, trim_end) if trim_start is not None and trim_end is not None else None

    sqlite_path = kwargs.get("_sqlite_path")

    rows = attribute_kernels_to_nvtx(conn, sqlite_path=sqlite_path, trim=trim)

    if not rows:
        return []

    # Auto-detect layer depth if not explicitly specified
    detection_meta = None
    auto_group_depth = None  # depth level to use for grouping by path component
    if depth is None and auto_depth:
        detection = detect_layer_depth(rows)
        detection_meta = detection  # always emit, even on fallback
        if detection["layer_depth"] is not None:
            auto_group_depth = detection["layer_depth"]
        else:
            # Provide available depths so the agent can retry with explicit -p depth=N
            available_depths = sorted({r.get("nvtx_depth", 0) for r in rows if r.get("nvtx_depth") is not None})
            # Sample region names at each depth for context
            depth_samples = {}
            for d in available_depths[:5]:
                samples = list({r.get("nvtx_text", "") for r in rows if r.get("nvtx_depth") == d})[:3]
                depth_samples[d] = samples
            detection_meta["available_depths"] = available_depths
            detection_meta["depth_samples"] = depth_samples

    # Explicit depth filtering: only keep rows at exactly the requested depth
    if depth is not None:
        if depth < 0:
            return [{"error": "Invalid depth <0 requested. Depth must be >= 0."}]
        rows = [r for r in rows if r.get("nvtx_depth") == depth]
        if not rows:
            return []

    # Python GROUP BY on nvtx_text with kernel classification for
    # compute/NCCL split, plus per-region kernel time tracking.
    groups: dict[str, dict] = defaultdict(
        lambda: {
            "total_ns": 0,
            "compute_ns": 0,
            "nccl_ns": 0,
            "count": 0,
            "max_ns": 0,
            "nvtx_depth": -1,
            "nvtx_path": "",
            "nvtx_region": "",
            "kernel_times": defaultdict(int),  # kernel_name → total_ns
        }
    )
    _class_cache: dict[str, str] = {}
    for r in rows:
        text = r["nvtx_text"]
        if not text:
            continue
        dur_ns = r["k_dur_ns"]
        k_name = r["kernel_name"]
        if k_name not in _class_cache:
            _class_cache[k_name] = classify_kernel(k_name)
        kernel_class = _class_cache[k_name]

        path = r.get("nvtx_path", text)

        # When auto-depth detected a layer level, group by the path prefix
        # up to that depth (e.g. "encoder > layer_0") instead of only the
        # single component. This preserves enough context to avoid collisions
        # between unrelated scopes that share the same leaf name.
        if auto_group_depth is not None:
            parts = path.split(" > ")
            if auto_group_depth < len(parts):
                prefix = parts[: auto_group_depth + 1]
                group_key = " > ".join(prefix)
                region_name = group_key
                group_depth = auto_group_depth
            else:
                group_key = path
                region_name = text
                group_depth = r.get("nvtx_depth", 0)
        else:
            group_key = path
            region_name = text
            group_depth = r.get("nvtx_depth", 0)

        stats = groups[group_key]
        stats["total_ns"] += dur_ns
        stats["count"] += 1
        if dur_ns > stats["max_ns"]:
            stats["max_ns"] = dur_ns
        # Per-kernel time tracking
        stats["kernel_times"][k_name] += dur_ns
        # Compute/NCCL split
        if kernel_class.startswith("nccl_"):
            stats["nccl_ns"] += dur_ns
        else:
            stats["compute_ns"] += dur_ns
        # Capture depth/path from first seen entry
        if stats["nvtx_depth"] < 0:
            stats["nvtx_depth"] = group_depth
            stats["nvtx_path"] = group_key
            stats["nvtx_region"] = region_name

    # Build aggregated results (defer top-kernel extraction until after limit)
    results = []
    for path_key, stats in groups.items():
        total_ns = stats["total_ns"]
        count = stats["count"]
        max_ns = stats["max_ns"]
        compute_ns = stats["compute_ns"]
        nccl_ns = stats["nccl_ns"]

        results.append(
            {
                "_raw_total_ns": total_ns,
                "_kernel_times": stats["kernel_times"],  # kept for deferred top-k
                "nvtx_region": stats["nvtx_region"],
                "nvtx_depth": stats["nvtx_depth"],
                "nvtx_path": stats["nvtx_path"],
                "kernel_count": count,
                "total_gpu_ms": round(total_ns / 1e6, 2),
                "compute_ms": round(compute_ns / 1e6, 2),
                "nccl_ms": round(nccl_ns / 1e6, 2),
                "nccl_pct": round(100 * nccl_ns / total_ns, 1) if total_ns > 0 else 0,
                "avg_kernel_ms": round(total_ns / count / 1e6, 3),
                "max_kernel_ms": round(max_ns / 1e6, 3),
            }
        )

    # Sort by total GPU time descending
    results.sort(key=lambda r: -r["_raw_total_ns"])

    # Cross-layer outlier detection — delegate to shared helper
    all_times = [r["_raw_total_ns"] / 1e6 for r in results]
    for r in results:
        t = r["_raw_total_ns"] / 1e6
        r["is_outlier"] = is_outlier(t, all_times)

    # Apply limit, then extract top-kernels only for kept results
    limited = results[:limit]
    for r in limited:
        kernel_times = r.pop("_kernel_times")
        top_k = sorted(kernel_times.items(), key=lambda x: -x[1])[:3]
        r["top_kernels"] = [
            {"kernel_name": k, "total_ms": round(v / 1e6, 3)}
            for k, v in top_k
        ]
        r.pop("_raw_total_ns", None)

    # Prepend detection metadata if auto-detection was used
    if detection_meta is not None:
        meta_entry = {
            "_detection_meta": True,
            "layer_depth": detection_meta["layer_depth"],
            "layer_names": detection_meta["layer_names"],
            "detection_method": detection_meta["detection_method"],
            "grouping_type": detection_meta.get("grouping_type", "flat"),
            "confidence": detection_meta["confidence"],
        }
        # Include depth hints when auto-detection falls back
        if "available_depths" in detection_meta:
            meta_entry["available_depths"] = detection_meta["available_depths"]
            meta_entry["depth_samples"] = detection_meta["depth_samples"]
        limited.insert(0, meta_entry)
    return limited


def _format(rows):
    if not rows:
        return "(No NVTX regions with attributed kernels found)"
    if "error" in rows[0]:
        return f"Error: {rows[0]['error']}"

    lines = []

    # Handle detection metadata header
    if rows and rows[0].get("_detection_meta"):
        meta = rows[0]
        method = meta["detection_method"]
        if method == "numbered_pattern":
            n = len(meta["layer_names"])
            lines.append(f"✅ Detected {n} layers at depth {meta['layer_depth']}")
            names_preview = ", ".join(meta["layer_names"][:5])
            if n > 5:
                names_preview += f", … ({n - 5} more)"
            lines.append(f"   Layers: {names_preview}")
        elif method == "repeated_siblings":
            lines.append(f"ℹ️  Grouped by repeated operations at depth {meta['layer_depth']}")
            lines.append("   (No numbered layer hierarchy found — showing per-operation breakdown)")
        else:
            lines.append("⚠️  No layer hierarchy detected — showing per-operation breakdown")
        lines.append("")
        rows = rows[1:]

    if not rows:
        return "\n".join(lines) + "\n(No NVTX regions with attributed kernels found)"

    lines.extend([
        "── NVTX Region GPU Time Breakdown ──",
        f"{'NVTX Region':<40s}  {'Depth':>5s}  {'Kernels':>7s}  {'Total(ms)':>10s}"
        f"  {'Compute':>9s}  {'NCCL':>9s}  {'NCCL%':>6s}  {'Outlier':>7s}",
        "─" * 106,
    ])
    for r in rows:
        # Favor nvtx_path over nvtx_region for disambiguation
        name = r.get("nvtx_path") or r.get("nvtx_region") or "(unnamed)"
        if len(name) > 38:
            name = "..." + name[-35:]
        outlier_flag = "  ⚠️" if r.get("is_outlier") else ""
        lines.append(
            f"{name:<40s}  {r['nvtx_depth']:>5d}  {r['kernel_count']:>7d}  {r['total_gpu_ms']:>10.2f}"
            f"  {r['compute_ms']:>9.2f}  {r['nccl_ms']:>9.2f}  {r['nccl_pct']:>5.1f}%{outlier_flag:>7s}"
        )
        # Show top kernels indented below each region
        for tk in r.get("top_kernels", []):
            k_name = tk["kernel_name"]
            if len(k_name) > 50:
                k_name = k_name[:47] + "..."
            lines.append(f"    └─ {k_name}  ({tk['total_ms']:.3f}ms)")
    return "\n".join(lines)


SKILL = Skill(
    name="nvtx_layer_breakdown",
    title="NVTX Region GPU Time Breakdown",
    description=(
        "Attributes GPU kernels to their parent NVTX regions (e.g. layers, "
        "forward/backward passes) and ranks them by total GPU time. "
        "Shows compute vs NCCL split per region, top-3 hotspot kernels, "
        "and flags outlier layers. Auto-detects layer-level NVTX depth "
        "when depth is not specified. "
        "Use to identify which code region is the bottleneck."
    ),
    category="nvtx",
    execute_fn=_execute,
    params=[
        SkillParam("limit", "Max number of NVTX regions to return", "int", False, 20),
        SkillParam(
            "depth",
            "Filter to specific NVTX nesting depth (0=top-level). "
            "When not specified, auto-detection finds the layer level "
            "via numbered patterns or repeated siblings.",
            "int",
            False,
            None,
        ),
        SkillParam(
            "auto_depth",
            "Enable auto-detection of layer depth (default True). "
            "Set to False to disable auto-detection and use all depths.",
            "bool",
            False,
            True,
        ),
    ],
    format_fn=_format,
    tags=["nvtx", "layer", "breakdown", "attribution", "region", "nccl", "compute", "outlier"],
)
