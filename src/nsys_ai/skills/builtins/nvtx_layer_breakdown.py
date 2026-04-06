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

from nsys_ai.connection import DB_ERRORS

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    """Execute NVTX region GPU time breakdown via fast DuckDB SQL."""
    from ...nvtx_layer_detect import detect_layer_depth, is_outlier

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
            return [
                {
                    "error": f"Invalid value for auto_depth: {raw_auto_depth!r}. "
                    f"Expected one of {sorted(true_tokens | false_tokens)}."
                }
            ]
    else:
        auto_depth = bool(raw_auto_depth)

    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")

    # Check if nvtx_kernel_map exists or can be built
    try:
        conn.execute("SELECT 1 FROM nvtx_kernel_map LIMIT 1")
        has_nkm = True
    except DB_ERRORS:
        has_nkm = False

    if not has_nkm:
        from ...nvtx_attribution import attribute_kernels_to_nvtx
        from ...overlap import classify_kernel

        trim = (trim_start, trim_end) if trim_start is not None and trim_end is not None else None
        sqlite_path = kwargs.get("_sqlite_path")
        rows = attribute_kernels_to_nvtx(conn, sqlite_path=sqlite_path, trim=trim)

        if not rows:
            return []

        detection_meta = None
        auto_group_depth = None
        if depth is None and auto_depth:
            detection_meta = detect_layer_depth(rows)
            if detection_meta["layer_depth"] is not None:
                auto_group_depth = detection_meta["layer_depth"]
            else:
                available_depths = sorted(
                    {r.get("nvtx_depth", 0) for r in rows if r.get("nvtx_depth") is not None}
                )
                depth_samples = {}
                for d in available_depths[:5]:
                    depth_samples[d] = [
                        r.get("nvtx_text", "") for r in rows if r.get("nvtx_depth") == d
                    ][:3]
                detection_meta["available_depths"] = available_depths
                detection_meta["depth_samples"] = depth_samples

        if depth is not None:
            if depth < 0:
                return [{"error": "Invalid depth <0 requested. Depth must be >= 0."}]

        groups_leaf = defaultdict(
            lambda: {
                "leaf_depth": -1,
                "leaf_text": "",
                "total_ns": 0,
                "nccl_ns": 0,
                "compute_ns": 0,
                "tc_elig": None,
                "tc_act": None,
                "count": 0,
                "max_ns": 0,
            }
        )

        k_times_by_group = defaultdict(lambda: defaultdict(int))
        _class_cache = {}
        for r in rows:
            text = r["nvtx_text"]
            if not text:
                continue
            path = r.get("nvtx_path", text)
            k_name = r["kernel_name"]
            dur_ns = r["k_dur_ns"]

            if k_name not in _class_cache:
                _class_cache[k_name] = classify_kernel(k_name)
            is_nccl = _class_cache[k_name].startswith("nccl_")

            stats = groups_leaf[path]
            if stats["leaf_depth"] < 0:
                stats["leaf_depth"] = r.get("nvtx_depth", 0)
                stats["leaf_text"] = text
            stats["total_ns"] += dur_ns
            if is_nccl:
                stats["nccl_ns"] += dur_ns
            else:
                stats["compute_ns"] += dur_ns
            stats["count"] += 1
            if dur_ns > stats["max_ns"]:
                stats["max_ns"] = dur_ns

            k_times_by_group[path][k_name] += dur_ns

        leaf_rows = [
            (
                path,
                v["leaf_depth"],
                v["leaf_text"],
                v["total_ns"],
                v["nccl_ns"],
                v["compute_ns"],
                v["tc_elig"],
                v["tc_act"],
                v["count"],
                v["max_ns"],
            )
            for path, v in groups_leaf.items()
        ]

        if depth is not None:
            leaf_rows = [r for r in leaf_rows if r[1] == depth]

    else:
        # DB has nvtx_kernel_map, use fast SQL
        trim_clause = ""
        params = []
        if trim_start is not None and trim_end is not None:
            trim_clause = "AND n.k_start >= ? AND n.k_end <= ?"
            params = [trim_start, trim_end]

        # Quick dummy rows for Auto-depth detection using distinct paths
        detection_meta = None
        auto_group_depth = None
        if depth is None and auto_depth:
            paths = [
                r[0]
                for r in conn.execute("SELECT DISTINCT nvtx_path FROM nvtx_kernel_map").fetchall()
            ]
            dummy_rows = [
                {
                    "nvtx_path": p,
                    "nvtx_text": p.split(" > ")[-1] if p else "",
                    "nvtx_depth": p.count(" > "),
                }
                for p in paths
            ]
            detection_meta = detect_layer_depth(dummy_rows)
            if detection_meta["layer_depth"] is not None:
                auto_group_depth = detection_meta["layer_depth"]
            else:
                available_depths = sorted({r["nvtx_depth"] for r in dummy_rows})
                depth_samples = {}
                for d in available_depths[:5]:
                    depth_samples[d] = [r["nvtx_text"] for r in dummy_rows if r["nvtx_depth"] == d][
                        :3
                    ]
                detection_meta["available_depths"] = available_depths
                detection_meta["depth_samples"] = depth_samples

        if depth is not None:
            if depth < 0:
                return [{"error": "Invalid depth <0 requested. Depth must be >= 0."}]

        # Phase 1: Pure SQL aggregation across all leaf paths
        sql_agg = f"""
            SELECT
                n.nvtx_path,
                FIRST(n.nvtx_depth) AS leaf_depth,
                FIRST(n.nvtx_text) AS leaf_text,
                SUM(n.k_dur_ns) AS total_ns,
                SUM(CASE WHEN lower(k.name) ILIKE '%nccl%' THEN n.k_dur_ns ELSE 0 END) AS nccl_ns,
                SUM(CASE WHEN lower(k.name) NOT ILIKE '%nccl%' THEN n.k_dur_ns ELSE 0 END) AS compute_ns,
                SUM(CASE WHEN k.is_tc_eligible = 1 THEN n.k_dur_ns ELSE 0 END) AS tc_eligible_ns,
                SUM(CASE WHEN k.uses_tc = 1 AND k.is_tc_eligible = 1 THEN n.k_dur_ns ELSE 0 END) AS tc_active_ns,
                COUNT(*) AS count,
                MAX(n.k_dur_ns) AS max_ns
            FROM nvtx_kernel_map n
            JOIN kernels k ON n.k_start = k.start AND n.k_end = k."end"
            WHERE 1=1 {trim_clause}
            GROUP BY n.nvtx_path
        """
        leaf_rows = conn.execute(sql_agg, params).fetchall()

        if not leaf_rows:
            return []

        if depth is not None:
            leaf_rows = [r for r in leaf_rows if r[1] == depth]

    # Phase 2: Python rollover for auto_group_depth truncation
    groups = defaultdict(
        lambda: {
            "total_ns": 0,
            "compute_ns": 0,
            "nccl_ns": 0,
            "tc_eligible_ns": 0,
            "tc_active_ns": 0,
            "count": 0,
            "max_ns": 0,
            "nvtx_depth": -1,
            "nvtx_path": "",
            "nvtx_region": "",
        }
    )

    for row in leaf_rows:
        (
            nvtx_path,
            leaf_depth,
            leaf_text,
            total_ns,
            nccl_ns,
            compute_ns,
            tc_elig,
            tc_act,
            count,
            max_ns,
        ) = row

        path_parts = nvtx_path.split(" > ") if nvtx_path else [""]
        if auto_group_depth is not None:
            if auto_group_depth < len(path_parts):
                group_key = " > ".join(path_parts[: auto_group_depth + 1])
                region_name = group_key
                group_depth = auto_group_depth
            else:
                group_key = nvtx_path
                region_name = leaf_text
                group_depth = leaf_depth
        else:
            group_key = nvtx_path
            region_name = leaf_text
            group_depth = leaf_depth

        stats = groups[group_key]
        stats["total_ns"] += total_ns or 0
        stats["compute_ns"] += compute_ns or 0
        stats["nccl_ns"] += nccl_ns or 0
        stats["tc_eligible_ns"] += tc_elig or 0
        stats["tc_active_ns"] += tc_act or 0
        stats["count"] += count
        if (max_ns or 0) > stats["max_ns"]:
            stats["max_ns"] = max_ns or 0
        if stats["nvtx_depth"] < 0:
            stats["nvtx_depth"] = group_depth
            stats["nvtx_path"] = group_key
            stats["nvtx_region"] = region_name

    # Phase 2.5: Re-group fallback k_times_by_group for consistent iteration
    if not has_nkm:
        new_k_times = defaultdict(lambda: defaultdict(int))
        for old_path, k_dict in k_times_by_group.items():
            path_parts = old_path.split(" > ") if old_path else [""]
            if auto_group_depth is not None and auto_group_depth < len(path_parts):
                new_key = " > ".join(path_parts[: auto_group_depth + 1])
            else:
                new_key = old_path
            for k_name, dura in k_dict.items():
                new_k_times[new_key][k_name] += dura
        k_times_by_group = new_k_times

    # Phase 3: Fast SQL grouping for top kernels (if natively supported)
    if has_nkm:
        sql_kernels = f"""
            SELECT n.nvtx_path, n.kernel_name, SUM(n.k_dur_ns) AS k_total
            FROM nvtx_kernel_map n
            WHERE 1=1 {trim_clause}
            GROUP BY n.nvtx_path, n.kernel_name
        """
        kernel_rows = conn.execute(sql_kernels, params).fetchall()
        k_times_by_group = defaultdict(lambda: defaultdict(int))
        for row in kernel_rows:
            nvtx_path, k_name, k_total = row
            path_parts = nvtx_path.split(" > ") if nvtx_path else [""]
            if auto_group_depth is not None:
                if auto_group_depth < len(path_parts):
                    group_key = " > ".join(path_parts[: auto_group_depth + 1])
                else:
                    group_key = nvtx_path
            else:
                group_key = nvtx_path
            k_times_by_group[group_key][k_name] += k_total
    # Else: k_times_by_group is already seeded by Fallback logic!

    # Final aggregation & sorting
    results = []
    for path_key, stats in groups.items():
        total_ns = stats["total_ns"]
        count = stats["count"]
        compute_ns = stats["compute_ns"]
        nccl_ns = stats["nccl_ns"]
        tc_elig = stats["tc_eligible_ns"]
        tc_act = stats["tc_active_ns"]

        # Sort and pick top kernels
        ktimes = k_times_by_group[path_key]
        top_k = sorted(ktimes.items(), key=lambda x: -x[1])[:3]
        top_kernels = [{"kernel_name": k, "total_ms": round(v / 1e6, 3)} for k, v in top_k]

        results.append(
            {
                "_raw_total_ns": total_ns,
                "nvtx_region": stats["nvtx_region"],
                "nvtx_depth": stats["nvtx_depth"],
                "nvtx_path": stats["nvtx_path"],
                "kernel_count": count,
                "total_gpu_ms": round(total_ns / 1e6, 2),
                "compute_ms": round(compute_ns / 1e6, 2),
                "nccl_ms": round(nccl_ns / 1e6, 2),
                "nccl_pct": round(100 * nccl_ns / total_ns, 1) if total_ns > 0 else 0,
                "tc_achieved_pct": round(100 * tc_act / tc_elig, 1)
                if tc_elig
                else (None if tc_elig is None else 0.0),
                "avg_kernel_ms": round(total_ns / count / 1e6, 3) if count else 0,
                "max_kernel_ms": round(stats["max_ns"] / 1e6, 3),
                "top_kernels": top_kernels,
            }
        )

    # Sort descending
    results.sort(key=lambda r: -r["_raw_total_ns"])

    # Outlier detection
    all_times = [r["_raw_total_ns"] / 1e6 for r in results]
    for r in results:
        r["is_outlier"] = is_outlier(r["_raw_total_ns"] / 1e6, all_times)
        r.pop("_raw_total_ns", None)

    limited = results[:limit]

    # Prepend metadata
    if detection_meta is not None:
        meta_entry = {
            "_detection_meta": True,
            "layer_depth": detection_meta["layer_depth"],
            "layer_names": detection_meta["layer_names"],
            "detection_method": detection_meta["detection_method"],
            "grouping_type": detection_meta.get("grouping_type", "flat"),
            "confidence": detection_meta["confidence"],
        }
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

    lines.extend(
        [
            "── NVTX Region GPU Time Breakdown ──",
            f"{'NVTX Region':<40s}  {'Depth':>5s}  {'Kernels':>7s}  {'Total(ms)':>10s}"
            f"  {'Compute':>9s}  {'NCCL':>9s}  {'NCCL%':>6s}  {'TC Ops%':>7s}  {'Outlier':>7s}",
            "─" * 116,
        ]
    )
    for r in rows:
        # Favor nvtx_path over nvtx_region for disambiguation
        name = r.get("nvtx_path") or r.get("nvtx_region") or "(unnamed)"
        if len(name) > 38:
            name = "..." + name[-35:]
        outlier_flag = "  ⚠️" if r.get("is_outlier") else ""
        tc_pct_str = (
            f"{r['tc_achieved_pct']:>6.1f}%" if r["tc_achieved_pct"] is not None else "    N/A"
        )
        lines.append(
            f"{name:<40s}  {r['nvtx_depth']:>5d}  {r['kernel_count']:>7d}  {r['total_gpu_ms']:>10.2f}"
            f"  {r['compute_ms']:>9.2f}  {r['nccl_ms']:>9.2f}  {r['nccl_pct']:>5.1f}%"
            f"  {tc_pct_str:>7s}{outlier_flag:>7s}"
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
