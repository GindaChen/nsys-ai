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

from nsys_ai.connection import (
    DB_ERRORS,
    cached_nvtx_map_has_embedded_tc,
    cached_nvtx_map_uses_path_id,
)

from ..base import Skill, SkillParam

# Compact vs full output (CLI --report removed; skills still accept -p report=…).
REPORT_FULL = "full"
REPORT_COMPACT = "compact"

# Above this many IDs, expanding IN (?,?,…) becomes slower than a full scan
# and can hit SQLite's 999-parameter limit.  We drop the IN filter and let
# leaf_to_group post-filter in Python instead.
_INLINE_FILTER_LIMIT = 900


def coerce_report_param(raw: object | None) -> tuple[str, str | None]:
    """Return (mode, error_message). Default full; error_message if invalid."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return REPORT_FULL, None
    token = str(raw).strip().lower()
    if token in (REPORT_FULL, "default"):
        return REPORT_FULL, None
    if token in (REPORT_COMPACT, "slim", "minimal"):
        return REPORT_COMPACT, None
    return REPORT_FULL, (
        f"Invalid report mode {raw!r}; expected '{REPORT_FULL}' or '{REPORT_COMPACT}'."
    )

# Leaf rows are 11-tuples:
#   (path_id, nvtx_path, leaf_depth, leaf_text, total_ns, nccl_ns, compute_ns,
#    tc_elig, tc_act, count, max_ns)
# path_id is int for DuckDB paths that GROUP BY surrogate key; None on SQLite/Python.
_LR_ID = 0
_LR_PATH = 1
_LR_DEPTH = 2
_LR_TEXT = 3
_LR_TOTAL = 4
_LR_NCCL = 5
_LR_COMPUTE = 6
_LR_TC_ELIG = 7
_LR_TC_ACT = 8
_LR_COUNT = 9
_LR_MAX = 10


def _execute(conn, **kwargs):
    """Execute NVTX region GPU time breakdown via fast DuckDB SQL."""
    from ...connection import DuckDBAdapter, wrap_connection
    from ...nvtx_layer_detect import detect_layer_depth

    report, report_err = coerce_report_param(kwargs.get("report"))
    if report_err:
        return [{"error": report_err}]

    wrapped = wrap_connection(conn)

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
    adapter = None
    try:
        conn.execute("SELECT 1 FROM nvtx_kernel_map LIMIT 1")
        has_nkm = True
    except DB_ERRORS:
        has_nkm = False

    if not has_nkm:
        adapter = wrapped
        # Fast path: when running on DuckDB but nvtx_kernel_map is deferred,
        # aggregate via SQL directly instead of materializing full Python maps.
        if isinstance(adapter, DuckDBAdapter):
            tables = adapter.resolve_activity_tables()
            kernel_table = tables.get("kernel", "CUPTI_ACTIVITY_KIND_KERNEL")
            runtime_table = tables.get("runtime", "CUPTI_ACTIVITY_KIND_RUNTIME")
            nvtx_table = tables.get("nvtx", "NVTX_EVENTS")
            has_textid = adapter.detect_nvtx_text_id()
            if has_textid:
                text_expr = "COALESCE(n.text, ns.value)"
                text_join = "LEFT JOIN StringIds ns ON n.textId = ns.id"
            else:
                text_expr = "n.text"
                text_join = ""

            trim_clause = ""
            params: list[int] = []
            kr_where = ""
            nvtx_where = ""
            if trim_start is not None and trim_end is not None:
                trim_clause = "WHERE m.k_start >= ? AND m.k_end <= ?"
                t_start = int(trim_start)
                t_end = int(trim_end)
                kr_where = 'WHERE k.start >= ? AND k."end" <= ?'
                nvtx_where = 'AND n.start <= ? AND n."end" >= ?'
                params = [t_start, t_end, t_end, t_start, t_start, t_end]

            sql_agg = f"""
                WITH kr AS (
                    SELECT r.globalTid, r.start AS r_start, r."end" AS r_end,
                           k.start AS k_start, k."end" AS k_end,
                           COALESCE(kd.value, ks.value, 'kernel_' || CAST(k.shortName AS VARCHAR)) AS kernel_name,
                           r.correlationId
                    FROM {kernel_table} k
                    JOIN {runtime_table} r ON r.correlationId = k.correlationId
                    LEFT JOIN StringIds ks ON k.shortName = ks.id
                    LEFT JOIN StringIds kd ON k.demangledName = kd.id
                    {kr_where}
                ),
                enclosing AS (
                    SELECT kr.k_start, kr.k_end, kr.kernel_name,
                           kr.r_start, kr.r_end, kr.globalTid, kr.correlationId,
                           {text_expr} AS nvtx_text,
                           (n."end" - n.start) AS n_dur, n.start AS n_start
                    FROM kr
                    JOIN {nvtx_table} n
                      ON n.globalTid = kr.globalTid
                      AND n.eventType = 59
                      AND n."end" > n.start
                      AND n.start <= kr.r_start
                      AND n."end" >= kr.r_end
                      {nvtx_where}
                    {text_join}
                    WHERE {text_expr} IS NOT NULL
                ),
                mapped AS (
                    SELECT
                        FIRST(nvtx_text ORDER BY n_dur ASC, n_start ASC) AS nvtx_text,
                        CAST(COUNT(*) - 1 AS INTEGER) AS nvtx_depth,
                        string_agg(nvtx_text, ' > ' ORDER BY n_dur DESC, n_start ASC) AS nvtx_path,
                        kernel_name, k_start, k_end, (k_end - k_start) AS k_dur_ns
                    FROM enclosing
                    GROUP BY k_start, k_end, globalTid, kernel_name, correlationId
                ),
                path_dict AS (
                    SELECT nvtx_path, ROW_NUMBER() OVER (ORDER BY nvtx_path)::BIGINT AS path_id
                    FROM (SELECT DISTINCT nvtx_path FROM mapped)
                ),
                mapped_pid AS (
                    SELECT m.*, d.path_id
                    FROM mapped m
                    JOIN path_dict d ON m.nvtx_path = d.nvtx_path
                )
                SELECT
                    m.path_id,
                    MAX(m.nvtx_path) AS nvtx_path,
                    FIRST(m.nvtx_depth) AS leaf_depth,
                    FIRST(m.nvtx_text) AS leaf_text,
                    SUM(m.k_dur_ns) AS total_ns,
                    SUM(CASE WHEN m.kernel_name ILIKE '%nccl%' THEN m.k_dur_ns ELSE 0 END) AS nccl_ns,
                    SUM(CASE WHEN lower(m.kernel_name) NOT ILIKE '%nccl%' THEN m.k_dur_ns ELSE 0 END) AS compute_ns,
                    SUM(CASE WHEN k.is_tc_eligible = 1 THEN m.k_dur_ns ELSE 0 END) AS tc_eligible_ns,
                    SUM(CASE WHEN k.uses_tc = 1 AND k.is_tc_eligible = 1 THEN m.k_dur_ns ELSE 0 END) AS tc_active_ns,
                    COUNT(*) AS count,
                    MAX(m.k_dur_ns) AS max_ns
                FROM mapped_pid m
                JOIN kernels k ON m.k_start = k.start AND m.k_end = k."end"
                {trim_clause}
                GROUP BY m.path_id
            """
            leaf_rows = conn.execute(sql_agg, params).fetchall()
            if not leaf_rows:
                return []
            detection_meta = None
            auto_group_depth = None
            if depth is None and auto_depth:
                dummy_rows = [
                    {
                        "nvtx_path": r[_LR_PATH],
                        "nvtx_text": (r[_LR_PATH].split(" > ")[-1] if r[_LR_PATH] else ""),
                        "nvtx_depth": int(r[_LR_DEPTH] or 0),
                    }
                    for r in leaf_rows
                ]
                detection_meta = detect_layer_depth(dummy_rows)
                if detection_meta["layer_depth"] is not None:
                    auto_group_depth = detection_meta["layer_depth"]
            if depth is not None:
                if depth < 0:
                    return [{"error": "Invalid depth <0 requested. Depth must be >= 0."}]
                leaf_rows = [r for r in leaf_rows if int(r[_LR_DEPTH]) == depth]
            k_times_by_group = defaultdict(lambda: defaultdict(int))
        else:
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
                    None,
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
                leaf_rows = [r for r in leaf_rows if int(r[_LR_DEPTH]) == depth]

    else:
        # DB has nvtx_kernel_map, use fast SQL
        trim_clause = ""
        params = []
        if trim_start is not None and trim_end is not None:
            trim_clause = "AND n.k_start >= ? AND n.k_end <= ?"
            params = [trim_start, trim_end]

        if depth is not None and depth < 0:
            return [{"error": "Invalid depth <0 requested. Depth must be >= 0."}]

        # Phase 1: aggregate per leaf path. Prefer map-only scan when TC flags are
        # embedded in nvtx_kernel_map (avoids a wide join to kernels — DuckDB perf).
        # When path_id is available, GROUP BY integer key (join dict for nvtx_path string output).
        pid = cached_nvtx_map_uses_path_id(conn)
        join_dict = (
            "JOIN nvtx_path_dict d ON n.path_id = d.path_id\n                "
            if pid
            else ""
        )
        path_col = "MAX(d.nvtx_path)" if pid else "n.nvtx_path"
        group_col = "n.path_id" if pid else "n.nvtx_path"
        head_select = (
            "n.path_id,\n                    MAX(d.nvtx_path) AS nvtx_path,"
            if pid
            else f"{path_col} AS nvtx_path,"
        )

        if cached_nvtx_map_has_embedded_tc(conn):
            sql_agg = f"""
                SELECT
                    {head_select}
                    FIRST(n.nvtx_depth) AS leaf_depth,
                    FIRST(n.nvtx_text) AS leaf_text,
                    SUM(n.k_dur_ns) AS total_ns,
                    SUM(CASE WHEN n.kernel_name ILIKE '%nccl%' THEN n.k_dur_ns ELSE 0 END) AS nccl_ns,
                    SUM(CASE WHEN lower(n.kernel_name) NOT ILIKE '%nccl%' THEN n.k_dur_ns ELSE 0 END) AS compute_ns,
                    SUM(CASE WHEN n.is_tc_eligible = 1 THEN n.k_dur_ns ELSE 0 END) AS tc_eligible_ns,
                    SUM(CASE WHEN n.uses_tc = 1 AND n.is_tc_eligible = 1 THEN n.k_dur_ns ELSE 0 END) AS tc_active_ns,
                    COUNT(*) AS count,
                    MAX(n.k_dur_ns) AS max_ns
                FROM nvtx_kernel_map n
                {join_dict}WHERE 1=1 {trim_clause}
                GROUP BY {group_col}
            """
        else:
            sql_agg = f"""
                SELECT
                    {head_select}
                    FIRST(n.nvtx_depth) AS leaf_depth,
                    FIRST(n.nvtx_text) AS leaf_text,
                    SUM(n.k_dur_ns) AS total_ns,
                    SUM(CASE WHEN k.name ILIKE '%nccl%' THEN n.k_dur_ns ELSE 0 END) AS nccl_ns,
                    SUM(CASE WHEN lower(k.name) NOT ILIKE '%nccl%' THEN n.k_dur_ns ELSE 0 END) AS compute_ns,
                    SUM(CASE WHEN k.is_tc_eligible = 1 THEN n.k_dur_ns ELSE 0 END) AS tc_eligible_ns,
                    SUM(CASE WHEN k.uses_tc = 1 AND k.is_tc_eligible = 1 THEN n.k_dur_ns ELSE 0 END) AS tc_active_ns,
                    COUNT(*) AS count,
                    MAX(n.k_dur_ns) AS max_ns
                FROM nvtx_kernel_map n
                {join_dict}JOIN kernels k ON n.k_start = k.start AND n.k_end = k."end"
                WHERE 1=1 {trim_clause}
                GROUP BY {group_col}
            """
        leaf_rows = conn.execute(sql_agg, params).fetchall()
        if not pid:
            leaf_rows = [(None,) + tuple(r) for r in leaf_rows]

        if not leaf_rows:
            return []

        detection_meta = None
        auto_group_depth = None
        if depth is None and auto_depth:
            dummy_rows = [
                {
                    "nvtx_path": r[_LR_PATH],
                    "nvtx_text": (r[_LR_TEXT] or "") or ((r[_LR_PATH] or "").split(" > ")[-1]),
                    "nvtx_depth": int(r[_LR_DEPTH] or 0),
                }
                for r in leaf_rows
            ]
            detection_meta = detect_layer_depth(dummy_rows)
            if detection_meta["layer_depth"] is not None:
                auto_group_depth = detection_meta["layer_depth"]
            else:
                available_depths = sorted({row["nvtx_depth"] for row in dummy_rows})
                depth_samples = {}
                for d in available_depths[:5]:
                    depth_samples[d] = [
                        row["nvtx_text"] for row in dummy_rows if row["nvtx_depth"] == d
                    ][:3]
                detection_meta["available_depths"] = available_depths
                detection_meta["depth_samples"] = depth_samples

        if depth is not None:
            leaf_rows = [r for r in leaf_rows if int(r[_LR_DEPTH]) == depth]

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

    leaf_to_group: dict[str, str] = {}
    path_id_to_nvtx_path: dict[int, str] = {}
    nvtx_path_to_pid: dict[str, int] = {}
    for row in leaf_rows:
        path_id = row[_LR_ID]
        nvtx_path = row[_LR_PATH]
        leaf_depth = row[_LR_DEPTH]
        leaf_text = row[_LR_TEXT]
        total_ns = row[_LR_TOTAL]
        nccl_ns = row[_LR_NCCL]
        compute_ns = row[_LR_COMPUTE]
        tc_elig = row[_LR_TC_ELIG]
        tc_act = row[_LR_TC_ACT]
        count = row[_LR_COUNT]
        max_ns = row[_LR_MAX]
        if path_id is not None:
            ipid = int(path_id)
            path_id_to_nvtx_path[ipid] = nvtx_path
            nvtx_path_to_pid[nvtx_path] = ipid

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

        leaf_to_group[nvtx_path] = group_key
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
    elif has_nkm:
        # Leaf rows came from nvtx_kernel_map SQL; per-kernel histogram is filled in Phase 3.
        k_times_by_group = defaultdict(lambda: defaultdict(int))

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
                "_group_key": path_key,
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

    # Sort descending with deterministic tie-break (matches JSON sort_spec).
    results.sort(key=lambda r: (-r["_raw_total_ns"], r["_group_key"]))

    # Outlier detection — O(N log N) total, not O(N² log N).
    # Sort once and derive median + quartiles from indices to avoid a second
    # sort inside statistics.quantiles().
    all_times = sorted(r["_raw_total_ns"] / 1e6 for r in results)
    _n = len(all_times)
    if _n < 2:
        _fence = float("inf")
        _med = 0.0
    elif _n < 4:
        mid = _n // 2
        _med = all_times[mid] if _n % 2 else (all_times[mid - 1] + all_times[mid]) / 2
        _fence = _med * 2.0
    else:
        # Linear interpolation at p*(N-1) — matches statistics.quantiles "inclusive" method
        def _q(p: float) -> float:
            pos = p * (_n - 1)
            lo = int(pos)
            frac = pos - lo
            return all_times[lo] if frac == 0 else all_times[lo] + frac * (all_times[lo + 1] - all_times[lo])

        _med = _q(0.5)
        _q1 = _q(0.25)
        _q3 = _q(0.75)
        _iqr = _q3 - _q1
        _fence = (_q3 + 1.5 * _iqr) if _iqr > 0 else (_med * 2.0)

    for r in results:
        v = r["_raw_total_ns"] / 1e6
        r["is_outlier"] = _n >= 2 and v > _fence and v > _med * 1.5
        r.pop("_raw_total_ns", None)

    limited = results[:limit]

    # Phase 3 (full report only): accurate top-kernels per visible group.
    # Compact omits top_kernels entirely — skip this work for wall-clock savings.
    if report == REPORT_COMPACT:
        for r in limited:
            r.pop("_group_key", None)
            r.pop("top_kernels", None)
    else:
        selected_group_keys = {r["_group_key"] for r in limited}
        if selected_group_keys:
            if has_nkm:
                selected_leaf_paths = [
                    p for p, g in leaf_to_group.items() if g in selected_group_keys
                ]
                if selected_leaf_paths:
                    uses_pid = cached_nvtx_map_uses_path_id(conn)
                    selected_pids = [
                        nvtx_path_to_pid[p]
                        for p in selected_leaf_paths
                        if p in nvtx_path_to_pid
                    ]
                    selected_pids = list(dict.fromkeys(selected_pids))
                    if uses_pid and selected_pids:
                        if len(selected_pids) <= _INLINE_FILTER_LIMIT:
                            ph = ",".join("?" for _ in selected_pids)
                            sql_k = (
                                f"SELECT n.path_id, n.kernel_name, SUM(n.k_dur_ns) AS k_total "
                                f"FROM nvtx_kernel_map n WHERE n.path_id IN ({ph})"
                            )
                            params_k = list(selected_pids)
                        else:
                            # Too many IDs to inline; full scan — leaf_to_group filters output
                            sql_k = (
                                "SELECT n.path_id, n.kernel_name, SUM(n.k_dur_ns) AS k_total "
                                "FROM nvtx_kernel_map n WHERE 1=1"
                            )
                            params_k = []
                        if trim_start is not None and trim_end is not None:
                            sql_k += " AND n.k_start >= ? AND n.k_end <= ?"
                            params_k.extend([int(trim_start), int(trim_end)])
                        sql_k += " GROUP BY n.path_id, n.kernel_name"
                        kernel_rows = conn.execute(sql_k, params_k).fetchall()
                        k_times_by_group = defaultdict(lambda: defaultdict(int))
                        for path_id_k, k_name, k_total in kernel_rows:
                            nvtx_p = path_id_to_nvtx_path.get(int(path_id_k), "")
                            gk = leaf_to_group.get(nvtx_p, nvtx_p)
                            k_times_by_group[gk][k_name] += k_total
                    else:
                        if len(selected_leaf_paths) <= _INLINE_FILTER_LIMIT:
                            placeholders = ",".join("?" for _ in selected_leaf_paths)
                            sql_k = (
                                f"SELECT n.nvtx_path, n.kernel_name, SUM(n.k_dur_ns) AS k_total "
                                f"FROM nvtx_kernel_map n "
                                f"WHERE n.nvtx_path IN ({placeholders})"
                            )
                            params_k = list(selected_leaf_paths)
                        else:
                            # Too many paths to inline; full scan — leaf_to_group filters output
                            sql_k = (
                                "SELECT n.nvtx_path, n.kernel_name, SUM(n.k_dur_ns) AS k_total "
                                "FROM nvtx_kernel_map n WHERE 1=1"
                            )
                            params_k = []
                        if trim_start is not None and trim_end is not None:
                            sql_k += " AND n.k_start >= ? AND n.k_end <= ?"
                            params_k.extend([int(trim_start), int(trim_end)])
                        sql_k += " GROUP BY n.nvtx_path, n.kernel_name"
                        kernel_rows = conn.execute(sql_k, params_k).fetchall()
                        k_times_by_group = defaultdict(lambda: defaultdict(int))
                        for nvtx_path, k_name, k_total in kernel_rows:
                            gk = leaf_to_group.get(nvtx_path, nvtx_path)
                            k_times_by_group[gk][k_name] += k_total
            elif adapter is not None and isinstance(adapter, DuckDBAdapter):
                selected_leaf_paths = [
                    p for p, g in leaf_to_group.items() if g in selected_group_keys
                ]
                if selected_leaf_paths:
                    has_textid = adapter.detect_nvtx_text_id()
                    if has_textid:
                        text_expr = "COALESCE(n.text, ns.value)"
                        text_join = "LEFT JOIN StringIds ns ON n.textId = ns.id"
                    else:
                        text_expr = "n.text"
                        text_join = ""
                    tables = adapter.resolve_activity_tables()
                    kernel_table = tables.get("kernel", "CUPTI_ACTIVITY_KIND_KERNEL")
                    runtime_table = tables.get("runtime", "CUPTI_ACTIVITY_KIND_RUNTIME")
                    nvtx_table = tables.get("nvtx", "NVTX_EVENTS")
                    kr_where = ""
                    nvtx_where = ""
                    params_k = []
                    if trim_start is not None and trim_end is not None:
                        t_start = int(trim_start)
                        t_end = int(trim_end)
                        kr_where = 'WHERE k.start >= ? AND k."end" <= ?'
                        nvtx_where = 'AND n.start <= ? AND n."end" >= ?'
                        params_k.extend([t_start, t_end, t_end, t_start])
                    sel_pids = [
                        nvtx_path_to_pid[p]
                        for p in selected_leaf_paths
                        if p in nvtx_path_to_pid
                    ]
                    sel_pids = list(dict.fromkeys(sel_pids))
                    if sel_pids:
                        if len(sel_pids) <= _INLINE_FILTER_LIMIT:
                            ph_pids = ",".join("?" for _ in sel_pids)
                            _path_filter = f"WHERE mp.path_id IN ({ph_pids})"
                            params_k.extend(sel_pids)
                        else:
                            # Too many IDs to inline; full scan — leaf_to_group filters output
                            _path_filter = ""
                        sql_k = f"""
                        WITH kr AS (
                            SELECT r.globalTid, r.start AS r_start, r."end" AS r_end,
                                   k.start AS k_start, k."end" AS k_end,
                                   COALESCE(kd.value, ks.value, 'kernel_' || CAST(k.shortName AS VARCHAR)) AS kernel_name,
                                   r.correlationId
                            FROM {kernel_table} k
                            JOIN {runtime_table} r ON r.correlationId = k.correlationId
                            LEFT JOIN StringIds ks ON k.shortName = ks.id
                            LEFT JOIN StringIds kd ON k.demangledName = kd.id
                            {kr_where}
                        ),
                        enclosing AS (
                            SELECT kr.k_start, kr.k_end, kr.kernel_name,
                                   kr.r_start, kr.r_end, kr.globalTid, kr.correlationId,
                                   {text_expr} AS nvtx_text, (n."end" - n.start) AS n_dur, n.start AS n_start
                            FROM kr
                            JOIN {nvtx_table} n
                              ON n.globalTid = kr.globalTid
                              AND n.eventType = 59
                              AND n."end" > n.start
                              AND n.start <= kr.r_start
                              AND n."end" >= kr.r_end
                              {nvtx_where}
                            {text_join}
                            WHERE {text_expr} IS NOT NULL
                        ),
                        mapped AS (
                            SELECT
                                string_agg(nvtx_text, ' > ' ORDER BY n_dur DESC, n_start ASC) AS nvtx_path,
                                kernel_name,
                                k_start,
                                k_end,
                                (k_end - k_start) AS k_dur_ns
                            FROM enclosing
                            GROUP BY k_start, k_end, globalTid, kernel_name, correlationId
                        ),
                        path_dict AS (
                            SELECT nvtx_path, ROW_NUMBER() OVER (ORDER BY nvtx_path)::BIGINT AS path_id
                            FROM (SELECT DISTINCT nvtx_path FROM mapped)
                        ),
                        mp AS (
                            SELECT m.*, d.path_id
                            FROM mapped m
                            JOIN path_dict d ON m.nvtx_path = d.nvtx_path
                        )
                        SELECT mp.nvtx_path, mp.kernel_name, SUM(mp.k_dur_ns) AS k_total
                        FROM mp
                        {_path_filter}
                        GROUP BY mp.nvtx_path, mp.kernel_name
                    """
                    else:
                        if len(selected_leaf_paths) <= _INLINE_FILTER_LIMIT:
                            placeholders = ",".join("?" for _ in selected_leaf_paths)
                            _path_filter2 = f"WHERE nvtx_path IN ({placeholders})"
                            params_k.extend(selected_leaf_paths)
                        else:
                            # Too many paths to inline; full scan — leaf_to_group filters output
                            _path_filter2 = ""
                        sql_k = f"""
                        WITH kr AS (
                            SELECT r.globalTid, r.start AS r_start, r."end" AS r_end,
                                   k.start AS k_start, k."end" AS k_end,
                                   COALESCE(kd.value, ks.value, 'kernel_' || CAST(k.shortName AS VARCHAR)) AS kernel_name,
                                   r.correlationId
                            FROM {kernel_table} k
                            JOIN {runtime_table} r ON r.correlationId = k.correlationId
                            LEFT JOIN StringIds ks ON k.shortName = ks.id
                            LEFT JOIN StringIds kd ON k.demangledName = kd.id
                            {kr_where}
                        ),
                        enclosing AS (
                            SELECT kr.k_start, kr.k_end, kr.kernel_name,
                                   kr.r_start, kr.r_end, kr.globalTid, kr.correlationId,
                                   {text_expr} AS nvtx_text, (n."end" - n.start) AS n_dur, n.start AS n_start
                            FROM kr
                            JOIN {nvtx_table} n
                              ON n.globalTid = kr.globalTid
                              AND n.eventType = 59
                              AND n."end" > n.start
                              AND n.start <= kr.r_start
                              AND n."end" >= kr.r_end
                              {nvtx_where}
                            {text_join}
                            WHERE {text_expr} IS NOT NULL
                        ),
                        mapped AS (
                            SELECT
                                string_agg(nvtx_text, ' > ' ORDER BY n_dur DESC, n_start ASC) AS nvtx_path,
                                kernel_name,
                                k_start,
                                k_end,
                                (k_end - k_start) AS k_dur_ns
                            FROM enclosing
                            GROUP BY k_start, k_end, globalTid, kernel_name, correlationId
                        )
                        SELECT nvtx_path, kernel_name, SUM(k_dur_ns) AS k_total
                        FROM mapped
                        {_path_filter2}
                        GROUP BY nvtx_path, kernel_name
                    """
                    kernel_rows = conn.execute(sql_k, params_k).fetchall()
                    k_times_by_group = defaultdict(lambda: defaultdict(int))
                    for nvtx_path, k_name, k_total in kernel_rows:
                        gk = leaf_to_group.get(nvtx_path, nvtx_path)
                        k_times_by_group[gk][k_name] += k_total

        # Fill top kernels and strip internal group keys
        for r in limited:
            gk = r["_group_key"]
            ktimes = k_times_by_group.get(gk, {})
            top_k = sorted(ktimes.items(), key=lambda x: -x[1])[:3]
            r["top_kernels"] = [{"kernel_name": k, "total_ms": round(v / 1e6, 3)} for k, v in top_k]
            r.pop("_group_key", None)

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

    if report == REPORT_COMPACT:
        for row in limited:
            if row.get("_detection_meta"):
                names = row.pop("layer_names", None)
                if isinstance(names, list) and names:
                    row["layer_names_sample"] = names[:5]
                    row["layer_names_total"] = len(names)
                    row["layer_names_omitted"] = True
            else:
                row.pop("top_kernels", None)  # no-op if already removed in compact fast path
                row["kernels_included"] = False

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
            names_full = meta.get("layer_names")
            if isinstance(names_full, list):
                n = len(names_full)
                preview_src = names_full
            else:
                n = int(meta.get("layer_names_total", 0))
                preview_src = list(meta.get("layer_names_sample") or [])
            lines.append(f"✅ Detected {n} layers at depth {meta['layer_depth']}")
            names_preview = ", ".join(preview_src[:5])
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
        # Show top kernels indented below each region (omitted in compact report)
        if r.get("kernels_included") is False:
            lines.append("    └─ (top kernels omitted; use report=full or -p report=full)")
        else:
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
        SkillParam(
            "report",
            "Output shape: 'full' (default) includes top_kernels and full layer_names; "
            "'compact' keeps the same per-region metrics for returned rows but omits "
            "top_kernels and truncates layer name lists (see layer_names_sample / *_total).",
            "str",
            False,
            "full",
        ),
    ],
    format_fn=_format,
    tags=["nvtx", "layer", "breakdown", "attribution", "region", "nccl", "compute", "outlier"],
)
