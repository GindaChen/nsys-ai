"""
region_mfu.py — NVTX-region and kernel-level MFU computation for nsys profiles.

This module implements a higher-level "region MFU" tool that:

- Finds NVTX ranges by name (supports both NVTX_EVENTS.text and textId → StringIds).
- Finds CUDA kernels by name directly (for profiles without custom NVTX labels).
- Attributes CUDA kernels to the chosen NVTX range via CUPTI_ACTIVITY_KIND_RUNTIME
  correlationId (same semantics as nvtx_tree.py, but specialized for a single
  region instead of building a full tree).
- Aggregates time in three ways:
    * wall_time_ns: NVTX range span (end − start), or total kernel span for kernel mode
    * kernel_sum_ns: sum of child kernel durations
    * kernel_union_ns: union-of-intervals over child kernels (no double-counting)
- Computes MFU for the region given theoretical FLOPs and peak TFLOPS.

Two modes via the ``source`` parameter:

    source="nvtx"   (default) — match an NVTX range, then attribute kernels inside it.
    source="kernel" — match kernels by name directly, use their aggregate time.

Exports a single public entry point for callers and tools:

    compute_region_mfu(profile_path, name, theoretical_flops, source="nvtx", ...)

Error handling is explicit and structured. All public functions return dicts
with either data fields or an "error" block:

    {"error": {"code": "...", "message": "..."}}.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .hardware import get_peak_tflops
from .profile import NsightSchema, get_first_gpu_name, resolve_profile_path

ErrorDict = dict[str, Any]
RowDict = dict[str, Any]


def _compat_execute(conn, sql, params=None):
    """Execute SQL, translating bracket syntax for DuckDB connections."""
    import duckdb as _ddb
    if isinstance(conn, _ddb.DuckDBPyConnection):
        from .sql_compat import sqlite_to_duckdb
        sql = sqlite_to_duckdb(sql)
    return conn.execute(sql, params or [])


def _escape_like(value: str) -> str:
    """Escape SQL LIKE wildcards so ``%`` and ``_`` are treated literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _error(code: str, message: str) -> ErrorDict:
    return {"error": {"code": code, "message": message}}


# ---------------------------------------------------------------------------
# Theoretical FLOPs calculator — reliable arithmetic for the LLM agent
# ---------------------------------------------------------------------------

_VALID_OPERATIONS = {
    "attention",  # QK^T + softmax*V: 4 * S^2 * H
    "qkv_proj",  # Q/K/V linear projections: 6 * S * H^2
    "output_proj",  # Output projection: 2 * S * H^2
    "mlp",  # MLP/FFN up + down: 4 * S * H * ffn
    "full_layer",  # attention + qkv_proj + output_proj + mlp
    "full_model",  # full_layer * num_layers (alias for convenience)
    "linear",  # Generic: 2 * M * N * K
}


def compute_theoretical_flops(
    operation: str,
    *,
    hidden_dim: int = 0,
    seq_len: int = 0,
    num_layers: int = 1,
    ffn_dim: int | None = None,
    batch_size: int = 1,
    multiplier: int = 1,
    # For "linear" operation only:
    M: int = 0,
    N: int = 0,
    K: int = 0,
) -> RowDict | ErrorDict:
    """Compute theoretical FLOPs for transformer operations.

    This tool exists because LLMs cannot reliably multiply large numbers
    like ``131072² × 4096 × 32``.  The agent selects the operation and
    provides model parameters; Python does the exact arithmetic.

    Args:
        operation: One of ``attention``, ``qkv_proj``, ``output_proj``,
            ``mlp``, ``full_layer``, ``full_model``, ``linear``.
        hidden_dim (H): Model hidden dimension.
        seq_len (S): Sequence length.
        num_layers (L): Number of transformer layers. Applied to per-layer
            operations automatically.
        ffn_dim: FFN intermediate dimension. Defaults to ``4 * hidden_dim``.
        batch_size: Batch size. Default 1.
        multiplier: 1 = forward only, 3 = fwd+bwd (no ckpt),
            4 = fwd+bwd with activation checkpointing.
        M, N, K: Dimensions for generic ``linear`` (2*M*N*K).

    Returns:
        Dict with ``theoretical_flops``, ``formula``, and ``breakdown``.
    """
    op = operation.lower().strip()
    if op not in _VALID_OPERATIONS:
        return _error(
            "INVALID_ARGUMENT",
            f"operation must be one of {sorted(_VALID_OPERATIONS)}, got '{operation}'.",
        )

    H = int(hidden_dim)
    S = int(seq_len)
    L = max(1, int(num_layers))
    B = max(1, int(batch_size))
    mul = max(1, int(multiplier))
    ffn = int(ffn_dim) if ffn_dim is not None else 4 * H

    if op == "linear":
        Mi, Ni, Ki = int(M), int(N), int(K)
        if Mi <= 0 or Ni <= 0 or Ki <= 0:
            return _error("INVALID_ARGUMENT", "M, N, K must all be positive for 'linear'.")
        flops = 2 * Mi * Ni * Ki
        formula = f"2 * {Mi} * {Ni} * {Ki}"
        breakdown = {"per_op": flops}
        total = flops * B * mul
        formula_full = f"{formula} * batch({B}) * mul({mul})"
    else:
        if H <= 0 or S <= 0:
            return _error("INVALID_ARGUMENT", "hidden_dim and seq_len must be positive.")

        components: dict[str, int] = {}
        if op in ("attention", "full_layer", "full_model"):
            components["attention"] = 4 * S * S * H
        if op in ("qkv_proj", "full_layer", "full_model"):
            components["qkv_proj"] = 6 * S * H * H
        if op in ("output_proj", "full_layer", "full_model"):
            components["output_proj"] = 2 * S * H * H
        if op in ("mlp", "full_layer", "full_model"):
            components["mlp"] = 4 * S * H * ffn

        per_layer = sum(components.values())
        total = per_layer * L * B * mul
        breakdown = {
            "per_layer_flops": per_layer,
            "components": {k: v for k, v in components.items()},
            "num_layers": L,
            "batch_size": B,
            "multiplier": mul,
        }
        parts = " + ".join(f"{k}={v:.4e}" for k, v in components.items())
        formula = f"({parts}) * L({L}) * batch({B}) * mul({mul})"
        formula_full = formula

    return {
        "theoretical_flops": total,
        "theoretical_flops_str": f"{total:.6e}",
        "operation": op,
        "formula": formula_full,
        "breakdown": breakdown,
    }


def _detect_nvtx_text_id(conn: sqlite3.Connection) -> bool:
    """Return True if NVTX_EVENTS uses textId -> StringIds."""
    try:
        import duckdb
        if isinstance(conn, duckdb.DuckDBPyConnection):
            cols = [r[0] for r in conn.execute("DESCRIBE NVTX_EVENTS").fetchall()]
        else:
            cur = conn.execute("PRAGMA table_info(NVTX_EVENTS)")
            cols = [r[1] for r in cur.fetchall()]
        return "textId" in cols
    except Exception:
        return False


def find_nvtx_ranges(
    conn: sqlite3.Connection,
    nvtx_name: str,
    *,
    match_mode: str = "contains",
) -> list[RowDict]:
    """
    Find NVTX ranges whose resolved text matches ``nvtx_name``.

    Returns a list of dicts with:
        text, start_ns, end_ns, global_tid, duration_ns
    ordered by start_ns ascending.
    """
    if not nvtx_name:
        return []

    has_text_id = _detect_nvtx_text_id(conn)
    if match_mode not in ("contains", "exact"):
        match_mode = "contains"

    if has_text_id:
        base_sql = (
            "SELECT COALESCE(n.text, s.value) AS text, "
            "n.start AS start_ns, n.[end] AS end_ns, n.globalTid AS global_tid "
            "FROM NVTX_EVENTS n "
            "LEFT JOIN StringIds s ON n.textId = s.id "
            "WHERE (n.text IS NOT NULL OR s.value IS NOT NULL) "
            "AND n.[end] > n.start "
        )
    else:
        base_sql = (
            "SELECT n.text AS text, n.start AS start_ns, n.[end] AS end_ns, "
            "n.globalTid AS global_tid "
            "FROM NVTX_EVENTS n "
            "WHERE n.text IS NOT NULL AND n.[end] > n.start "
        )

    params: list[Any] = []
    if has_text_id:
        text_expr = "COALESCE(n.text, s.value)"
    else:
        text_expr = "n.text"

    if match_mode == "exact":
        base_sql += f"AND {text_expr} = ? "
        params.append(nvtx_name)
    else:
        base_sql += f"AND {text_expr} LIKE ? ESCAPE '\\' "
        params.append(f"%{_escape_like(nvtx_name)}%")

    base_sql += "ORDER BY start_ns"

    cur = _compat_execute(conn, base_sql, params)
    rows: list[RowDict] = []
    for text, start_ns, end_ns, global_tid in cur.fetchall():
        if text is None:
            continue
        d = int(end_ns) - int(start_ns)
        if d <= 0:
            continue
        rows.append(
            {
                "text": str(text),
                "start_ns": int(start_ns),
                "end_ns": int(end_ns),
                "global_tid": int(global_tid) if global_tid is not None else None,
                "duration_ns": d,
            }
        )
    return rows


def _resolve_string_ids(
    conn: sqlite3.Connection,
    pattern: str,
    *,
    match_mode: str = "contains",
) -> dict[int, str]:
    """Resolve a name pattern to {id: value} from the StringIds table.

    StringIds is small (typically <1000 rows), so this is always fast —
    even on multi-GB profiles where the kernel table has millions of rows.
    """
    if match_mode == "exact":
        sql = "SELECT id, value FROM StringIds WHERE value = ?"
        params: list[Any] = [pattern]
    else:
        sql = "SELECT id, value FROM StringIds WHERE value LIKE ? ESCAPE '\\'"
        params = [f"%{_escape_like(pattern)}%"]

    cur = _compat_execute(conn, sql, params)
    return {int(row[0]): str(row[1]) for row in cur.fetchall()}


def _ensure_kernel_indexes(conn: sqlite3.Connection) -> None:
    """Create indexes on kernel/runtime tables if the DB is writable.

    Delegates to the centralized :func:`~nsys_ai.indexing.ensure_performance_indexes`.
    """
    from .indexing import ensure_performance_indexes

    ensure_performance_indexes(conn)


def find_kernels_by_name(
    conn: sqlite3.Connection,
    kernel_name: str,
    *,
    match_mode: str = "contains",
    device_id: int | None = None,
) -> list[RowDict]:
    """Find CUDA kernels whose shortName matches *kernel_name*.

    Uses a **two-step strategy** for performance on large profiles:

    1. Resolve ``kernel_name`` → integer id(s) via ``StringIds`` (tiny table).
    2. Query the kernel table with ``shortName IN (…)`` — no JOIN needed.

    Returns a list of dicts ordered by ``start_ns``.
    """
    if not kernel_name:
        return []

    schema = NsightSchema(conn)
    if not schema.kernel_table:
        return []

    # Step 1 — resolve name to StringIds (instant, <1000 rows)
    id_map = _resolve_string_ids(conn, kernel_name, match_mode=match_mode)
    if not id_map:
        return []

    # Optionally create indexes (no-op on read-only connections)
    _ensure_kernel_indexes(conn)

    # Step 2 — query kernel table by integer id(s)
    kernel_table = schema.kernel_table
    placeholders = ",".join("?" * len(id_map))
    params: list[Any] = list(id_map.keys())

    dev_filter = ""
    if device_id is not None:
        dev_filter = "AND k.deviceId = ?"
        params.append(int(device_id))

    sql = (
        f"SELECT k.shortName, k.start AS start_ns, k.[end] AS end_ns, "
        f"(k.[end] - k.start) AS duration_ns, k.deviceId, k.streamId "
        f"FROM {kernel_table} k "
        f"WHERE k.shortName IN ({placeholders}) "
        f"AND k.[end] > k.start {dev_filter} "
        f"ORDER BY k.start"
    )

    cur = _compat_execute(conn, sql, params)
    kernels: list[RowDict] = []
    for short_id, start_ns, end_ns, duration_ns, dev, stream_id in cur.fetchall():
        d = int(duration_ns) if duration_ns is not None else int(end_ns) - int(start_ns)
        if d <= 0:
            continue
        kernels.append(
            {
                "name": id_map.get(int(short_id), ""),
                "start_ns": int(start_ns),
                "end_ns": int(end_ns),
                "duration_ns": d,
                "device_id": int(dev) if dev is not None else None,
                "stream_id": int(stream_id) if stream_id is not None else None,
            }
        )
    return kernels


def select_nvtx_occurrence(matches: list[RowDict], occurrence_index: int) -> RowDict | ErrorDict:
    """
    Pick the N-th NVTX match (1-based). Returns the row dict or an error.
    """
    if not matches:
        return _error("NVTX_NOT_FOUND", "No NVTX range matched the requested name.")
    if occurrence_index <= 0:
        return _error("INVALID_ARGUMENT", "occurrence_index must be >= 1.")
    if occurrence_index > len(matches):
        return _error(
            "NVTX_OCCURRENCE_OUT_OF_RANGE",
            f"Requested occurrence_index {occurrence_index}, but only {len(matches)} match(es) were found.",
        )
    row = dict(matches[occurrence_index - 1])
    row["occurrence_index"] = occurrence_index
    return row


def get_region_kernels(
    conn: sqlite3.Connection,
    *,
    nvtx_start_ns: int,
    nvtx_end_ns: int,
    global_tid: int | None,
    device_id: int | None,
) -> list[RowDict]:
    """
    Attribute kernels to an NVTX region via CUPTI_ACTIVITY_KIND_RUNTIME.correlationId.

    Only runtime calls whose CPU span lies fully inside the NVTX CPU span are
    considered. If ``global_tid`` is provided, restricts to that thread to
    mirror nvtx_tree.py semantics. If ``device_id`` is provided, restricts to
    that GPU.
    """
    schema = NsightSchema(conn)
    if not schema.kernel_table:
        return []

    kernel_table = schema.kernel_table

    where_clauses = ["r.start >= ?", "r.[end] <= ?"]
    params: list[Any] = [int(nvtx_start_ns), int(nvtx_end_ns)]

    if global_tid is not None:
        where_clauses.append("r.globalTid = ?")
        params.append(int(global_tid))

    if device_id is not None:
        dev_filter = "AND k.deviceId = ?"
        params.append(int(device_id))
    else:
        dev_filter = ""

    sql = (
        "SELECT r.correlationId AS correlation_id, "
        "k.deviceId AS device_id, k.streamId AS stream_id, "
        "k.start AS start_ns, k.[end] AS end_ns, "
        "(k.[end] - k.start) AS duration_ns, "
        "s.value AS kernel_name "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        f"JOIN {kernel_table} k ON r.correlationId = k.correlationId "
        "LEFT JOIN StringIds s ON k.shortName = s.id "
        "WHERE " + " AND ".join(where_clauses) + f" {dev_filter} "
        "ORDER BY start_ns"
    )

    cur = _compat_execute(conn, sql, params)
    kernels: list[RowDict] = []
    for row in cur.fetchall():
        (
            correlation_id,
            dev,
            stream_id,
            start_ns,
            end_ns,
            duration_ns,
            kernel_name,
        ) = row
        if duration_ns is None:
            duration_ns = int(end_ns) - int(start_ns)
        d = int(duration_ns)
        if d <= 0:
            continue
        kernels.append(
            {
                "correlation_id": int(correlation_id),
                "device_id": int(dev) if dev is not None else None,
                "stream_id": int(stream_id) if stream_id is not None else None,
                "start_ns": int(start_ns),
                "end_ns": int(end_ns),
                "duration_ns": d,
                "name": str(kernel_name) if kernel_name is not None else "",
            }
        )
    return kernels


def _merge_intervals(intervals: list[tuple[int, int]]) -> int:
    """Return total union length for a list of [start, end] intervals in ns."""
    if not intervals:
        return 0
    intervals_sorted = sorted(intervals, key=lambda x: x[0])
    total = 0
    cur_start, cur_end = intervals_sorted[0]
    for start, end in intervals_sorted[1:]:
        if start <= cur_end:
            if end > cur_end:
                cur_end = end
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def summarize_region_kernel_times(kernels: list[RowDict]) -> RowDict:
    """
    Summarize kernel timings and coverage for a region.

    Returns:
        {
            "kernel_count": int,
            "kernel_sum_ns": int,
            "kernel_union_ns": int,
            "device_ids": [int, ...],
            "stream_ids": [int, ...],
        }
    """
    if not kernels:
        return {
            "kernel_count": 0,
            "kernel_sum_ns": 0,
            "kernel_union_ns": 0,
            "device_ids": [],
            "stream_ids": [],
        }

    kernel_sum_ns = sum(int(k["duration_ns"]) for k in kernels)
    intervals = [(int(k["start_ns"]), int(k["end_ns"])) for k in kernels]
    kernel_union_ns = _merge_intervals(intervals)

    devs = sorted({int(k["device_id"]) for k in kernels if k.get("device_id") is not None})
    streams = sorted({int(k["stream_id"]) for k in kernels if k.get("stream_id") is not None})

    return {
        "kernel_count": len(kernels),
        "kernel_sum_ns": int(kernel_sum_ns),
        "kernel_union_ns": int(kernel_union_ns),
        "device_ids": devs,
        "stream_ids": streams,
    }


def compute_mfu_metrics_for_region(
    *,
    theoretical_flops: float,
    peak_tflops: float,
    wall_time_s: float,
    kernel_sum_s: float,
    kernel_union_s: float,
) -> RowDict | ErrorDict:
    """
    Compute MFU metrics for a region given FLOPs and various time bases.
    """
    if theoretical_flops <= 0:
        return _error(
            "INVALID_ARGUMENT",
            "theoretical_flops must be positive (e.g. model_flops_per_step).",
        )
    if peak_tflops <= 0:
        return _error(
            "INVALID_ARGUMENT",
            "peak_tflops must be positive (GPU peak TFLOPS for the chosen precision).",
        )
    if wall_time_s <= 0 or kernel_union_s <= 0 or kernel_sum_s <= 0:
        # We allow caller to decide whether missing kernels is an error; this
        # helper only validates positive times.
        return _error(
            "INVALID_ARGUMENT",
            "All time bases (wall_time_s, kernel_sum_s, kernel_union_s) must be positive.",
        )

    def _achieved(time_s: float) -> float:
        return (theoretical_flops / time_s) / 1e12

    achieved_wall = _achieved(wall_time_s)
    achieved_sum = _achieved(kernel_sum_s)
    achieved_union = _achieved(kernel_union_s)

    mfu_wall = 100.0 * achieved_wall / peak_tflops
    mfu_sum = 100.0 * achieved_sum / peak_tflops
    mfu_union = 100.0 * achieved_union / peak_tflops

    return {
        "theoretical_flops": float(theoretical_flops),
        "peak_tflops": float(peak_tflops),
        "achieved_tflops_wall": round(achieved_wall, 2),
        "achieved_tflops_kernel_sum": round(achieved_sum, 2),
        "achieved_tflops_kernel_union": round(achieved_union, 2),
        "mfu_pct_wall": round(mfu_wall, 2),
        "mfu_pct_kernel_sum": round(mfu_sum, 2),
        "mfu_pct_kernel_union": round(mfu_union, 2),
    }


def _auto_peak_tflops(conn: sqlite3.Connection, explicit_peak: float | None) -> RowDict | ErrorDict:
    """
    Resolve peak_tflops either from explicit value or from profile GPU name.
    """
    if explicit_peak is not None:
        if explicit_peak <= 0:
            return _error(
                "INVALID_ARGUMENT",
                "peak_tflops must be positive when provided explicitly.",
            )
        return {"peak_tflops": float(explicit_peak), "source": "explicit"}

    gpu_name = get_first_gpu_name(conn)
    if not gpu_name:
        return _error(
            "GPU_PEAK_UNKNOWN",
            "Could not determine GPU name from profile; provide peak_tflops explicitly.",
        )
    info = get_peak_tflops(gpu_name)
    if "error" in info or "peak_tflops" not in info:
        return _error(
            "GPU_PEAK_UNKNOWN",
            f"Unknown GPU for peak_tflops lookup: {gpu_name}. Provide peak_tflops explicitly.",
        )
    return {"peak_tflops": float(info["peak_tflops"]), "source": "profile"}


def compute_region_mfu_from_conn(
    conn: sqlite3.Connection,
    profile_path: str | None,
    name: str,
    theoretical_flops: float,
    *,
    source: str = "nvtx",
    peak_tflops: float | None = None,
    num_gpus: int = 1,
    occurrence_index: int = 1,
    device_id: int | None = None,
    match_mode: str = "contains",
) -> RowDict | ErrorDict:
    """
    Compute MFU for a named region using an existing SQLite connection.

    Args:
        name:   NVTX range text (when ``source="nvtx"``) or kernel short name
                (when ``source="kernel"``).
        source: ``"nvtx"`` — find an NVTX range and attribute kernels inside it.
                ``"kernel"`` — find CUDA kernels by name directly.
    """
    if not name:
        return _error("INVALID_ARGUMENT", "name must be a non-empty string.")
    if theoretical_flops <= 0:
        return _error(
            "INVALID_ARGUMENT",
            "theoretical_flops must be positive (e.g. model_flops_per_step).",
        )
    if source not in ("nvtx", "kernel"):
        return _error("INVALID_ARGUMENT", "source must be 'nvtx' or 'kernel'.")

    # ---------------------------------------------------------------
    # Branch: source="kernel" — query kernels directly by name
    # ---------------------------------------------------------------
    if source == "kernel":
        kernels = find_kernels_by_name(
            conn,
            name,
            match_mode=match_mode,
            device_id=device_id,
        )
        if not kernels:
            return _error(
                "KERNEL_NOT_FOUND",
                f"No kernels matching '{name}' found in the profile.",
            )
        summary = summarize_region_kernel_times(kernels)
        matched_name = kernels[0]["name"]
        kernel_sum_s = summary["kernel_sum_ns"] / 1e9
        kernel_union_s = summary["kernel_union_ns"] / 1e9
        # wall_time = span from first kernel start to last kernel end
        wall_time_ns = max(k["end_ns"] for k in kernels) - min(k["start_ns"] for k in kernels)
        wall_time_s = max(wall_time_ns / 1e9, kernel_union_s)

    # ---------------------------------------------------------------
    # Branch: source="nvtx" (default) — NVTX range → kernel attribution
    # ---------------------------------------------------------------
    else:
        matches = find_nvtx_ranges(conn, name, match_mode=match_mode)
        chosen = select_nvtx_occurrence(matches, occurrence_index)
        if "error" in chosen:
            return chosen

        nvtx_start_ns = int(chosen["start_ns"])
        nvtx_end_ns = int(chosen["end_ns"])
        global_tid = chosen.get("global_tid")

        wall_time_ns = max(0, nvtx_end_ns - nvtx_start_ns)
        if wall_time_ns <= 0:
            return _error(
                "NO_KERNELS_IN_REGION",
                "Selected NVTX range has non-positive duration.",
            )

        kernels = get_region_kernels(
            conn,
            nvtx_start_ns=nvtx_start_ns,
            nvtx_end_ns=nvtx_end_ns,
            global_tid=global_tid,
            device_id=device_id,
        )
        summary = summarize_region_kernel_times(kernels)
        if summary["kernel_count"] == 0:
            return _error(
                "NO_KERNELS_IN_REGION",
                "No kernels found inside the selected NVTX region (after thread/device filters).",
            )
        matched_name = chosen.get("text", "")
        wall_time_s = wall_time_ns / 1e9
        kernel_sum_s = summary["kernel_sum_ns"] / 1e9 if summary["kernel_sum_ns"] > 0 else 0.0
        kernel_union_s = summary["kernel_union_ns"] / 1e9 if summary["kernel_union_ns"] > 0 else 0.0

    # ---------------------------------------------------------------
    # Common: resolve peak, compute MFU, return result
    # ---------------------------------------------------------------
    peak_info = _auto_peak_tflops(conn, peak_tflops)
    if "error" in peak_info:
        return peak_info
    peak_per_gpu = float(peak_info["peak_tflops"])
    effective_num_gpus = max(1, int(num_gpus))
    effective_peak = peak_per_gpu * effective_num_gpus

    metrics = compute_mfu_metrics_for_region(
        theoretical_flops=theoretical_flops,
        peak_tflops=effective_peak,
        wall_time_s=wall_time_s,
        kernel_sum_s=kernel_sum_s or wall_time_s,
        kernel_union_s=kernel_union_s or wall_time_s,
    )
    if "error" in metrics:
        return metrics

    return {
        "source": source,
        "name": name,
        "matched_text": matched_name,
        "match_mode": match_mode,
        "occurrence_index": int(chosen.get("occurrence_index", occurrence_index))
        if source == "nvtx"
        else None,
        "device_id": int(device_id) if device_id is not None else None,
        "profile_path": profile_path,
        "num_gpus": effective_num_gpus,
        "peak_tflops_per_gpu": peak_per_gpu,
        "effective_peak_tflops": effective_peak,
        "wall_time_s": round(wall_time_s, 6),
        "gpu_kernel_sum_s": round(kernel_sum_s, 6),
        "gpu_kernel_union_s": round(kernel_union_s, 6),
        "time_basis_used": "wall",
        "kernel_count": summary["kernel_count"],
        "device_ids": summary["device_ids"],
        "stream_ids": summary["stream_ids"],
        **metrics,
    }


def compute_region_mfu(
    profile_path: str,
    name: str,
    theoretical_flops: float,
    *,
    source: str = "nvtx",
    peak_tflops: float | None = None,
    num_gpus: int = 1,
    occurrence_index: int = 1,
    device_id: int | None = None,
    match_mode: str = "contains",
) -> RowDict | ErrorDict:
    """
    Convenience wrapper that opens the profile for MFU computation.

    Uses ``open_profile_readonly`` to leverage the DuckDB Parquet cache when
    available. The chat agent should prefer :func:`compute_region_mfu_from_conn`
    to reuse its existing connection.
    """
    try:
        sqlite_path = resolve_profile_path(profile_path)
    except RuntimeError as e:
        return _error("PROFILE_NOT_LOADED", f"Profile error: {e}")

    from nsys_ai.ai.backend.profile_db_tool import open_profile_readonly
    conn = open_profile_readonly(sqlite_path)
    try:
        return compute_region_mfu_from_conn(
            conn,
            sqlite_path,
            name,
            theoretical_flops,
            source=source,
            peak_tflops=peak_tflops,
            num_gpus=num_gpus,
            occurrence_index=occurrence_index,
            device_id=device_id,
            match_mode=match_mode,
        )
    finally:
        conn.close()
