"""Rank likely code/config regions for a selected profile window.

This skill is a deterministic first step toward profile-to-code tracing:
it uses NVTX timing context to map a selected finding window back to likely
model/training regions, without claiming exact source-line attribution.
"""

from collections import Counter, defaultdict

from ..base import Skill, SkillParam

_LIMITATIONS = [
    "NVTX attribution is temporal context, not exact source-line attribution",
    "Use finer NVTX labels, stack traces, or explicit file/function labels for exact source mapping",
]


def _overlap_ns(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _is_nccl_kernel(name: str) -> bool:
    return "nccl" in name.lower()


def _confidence(
    overlap_ratio: float,
    contains_selection: bool,
    kernel_count: int,
    depth: int,
) -> float:
    score = 0.55 * overlap_ratio
    if contains_selection:
        score += 0.15
    if kernel_count > 0:
        score += 0.20
    if depth > 0:
        score += min(depth, 4) * 0.025
    return round(min(score, 0.95), 2)


def _resolve_text_expr(adapter, alias: str = "n") -> tuple[str, str]:
    if adapter.detect_nvtx_text_id():
        return (
            f"COALESCE({alias}.text, s.value)",
            f"LEFT JOIN StringIds s ON {alias}.textId = s.id",
        )
    return f"{alias}.text", ""


def _load_nvtx_ranges(adapter, start_ns: int, end_ns: int) -> list[dict]:
    tables = adapter.resolve_activity_tables()
    nvtx_table = tables.get("nvtx", "NVTX_EVENTS")
    text_expr, text_join = _resolve_text_expr(adapter)

    rows = adapter.execute(
        f"""
        SELECT n.globalTid, n.start, n."end", {text_expr} AS text
        FROM {nvtx_table} n
        {text_join}
        WHERE n.eventType = 59
          AND n."end" > n.start
          AND n.start <= ?
          AND n."end" >= ?
          AND {text_expr} IS NOT NULL
        ORDER BY n.globalTid, n.start, n."end" DESC
        """,
        (end_ns, start_ns),
    ).fetchall()

    by_tid: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for tid, ns_start, ns_end, text in rows:
        if text:
            by_tid[int(tid)].append((int(ns_start), int(ns_end), str(text)))

    ranges: list[dict] = []
    for tid, tid_rows in by_tid.items():
        stack: list[tuple[int, int, str]] = []
        for ns_start, ns_end, text in tid_rows:
            while stack and stack[-1][1] <= ns_start:
                stack.pop()

            enclosing = [
                item for item in stack if item[0] <= ns_start and item[1] >= ns_end
            ]
            path_parts = [item[2] for item in enclosing] + [text]
            depth = len(path_parts) - 1
            ranges.append(
                {
                    "global_tid": tid,
                    "nvtx_text": text,
                    "nvtx_path": " > ".join(path_parts),
                    "nvtx_depth": depth,
                    "start_ns": ns_start,
                    "end_ns": ns_end,
                }
            )

            stack = enclosing + [(ns_start, ns_end, text)]

    return ranges


def _load_kernels(adapter, start_ns: int, end_ns: int, device: int | None) -> list[dict]:
    tables = adapter.resolve_activity_tables()
    kernel_table = tables.get("kernel", "CUPTI_ACTIVITY_KIND_KERNEL")
    runtime_table = tables.get("runtime", "CUPTI_ACTIVITY_KIND_RUNTIME")

    device_clause = ""
    params: list[object] = [end_ns, start_ns]
    if device is not None:
        device_clause = "AND k.deviceId = ?"
        params.append(device)

    rows = adapter.execute(
        f"""
        SELECT r.globalTid,
               r.start AS runtime_start_ns,
               r."end" AS runtime_end_ns,
               k.start AS kernel_start_ns,
               k."end" AS kernel_end_ns,
               k.deviceId AS device_id,
               k.streamId AS stream_id,
               COALESCE(d.value, s.value, 'kernel_' || CAST(k.shortName AS VARCHAR)) AS kernel_name
        FROM {kernel_table} k
        JOIN {runtime_table} r ON r.correlationId = k.correlationId
        LEFT JOIN StringIds s ON k.shortName = s.id
        LEFT JOIN StringIds d ON k.demangledName = d.id
        WHERE k.start <= ?
          AND k."end" >= ?
          {device_clause}
        """,
        params,
    ).fetchall()

    return [
        {
            "global_tid": int(r[0]),
            "runtime_start_ns": int(r[1]),
            "runtime_end_ns": int(r[2]),
            "kernel_start_ns": int(r[3]),
            "kernel_end_ns": int(r[4]),
            "device_id": int(r[5]),
            "stream_id": int(r[6]),
            "kernel_name": str(r[7]),
        }
        for r in rows
    ]


def _top_kernels(kernels: list[dict], limit: int = 3) -> list[dict]:
    by_name: dict[str, dict] = {}
    for k in kernels:
        name = k["kernel_name"]
        item = by_name.setdefault(name, {"kernel_name": name, "count": 0, "total_ms": 0.0})
        item["count"] += 1
        item["total_ms"] += (k["kernel_end_ns"] - k["kernel_start_ns"]) / 1e6

    ranked = sorted(by_name.values(), key=lambda r: (-r["total_ms"], r["kernel_name"]))
    for row in ranked:
        row["total_ms"] = round(row["total_ms"], 3)
    return ranked[:limit]


def _execute(conn, **kwargs):
    from ...connection import wrap_connection

    start_ns = int(kwargs["start_ns"])
    end_ns = int(kwargs["end_ns"])
    if end_ns <= start_ns:
        return [{"error": "end_ns must be greater than start_ns"}]

    device_arg = kwargs.get("device")
    device = None if device_arg in (None, "", "all") else int(device_arg)
    limit = int(kwargs.get("limit", 5))
    if limit < 1:
        return [{"error": "limit must be >= 1"}]
    min_overlap_pct = float(kwargs.get("min_overlap_pct", 0.0))

    adapter = wrap_connection(conn)
    selection_duration = end_ns - start_ns
    nvtx_ranges = _load_nvtx_ranges(adapter, start_ns, end_ns)
    if not nvtx_ranges:
        return [
            {
                "error": "No overlapping NVTX ranges found for the selected window",
                "selection": {
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "device_id": device,
                },
                "limitations": _LIMITATIONS,
            }
        ]

    kernels = _load_kernels(adapter, start_ns, end_ns, device)
    kernel_tids = Counter(k["global_tid"] for k in kernels)

    candidates = []
    for r in nvtx_ranges:
        overlap = _overlap_ns(start_ns, end_ns, r["start_ns"], r["end_ns"])
        if overlap <= 0:
            continue
        overlap_pct = round(100.0 * overlap / selection_duration, 1)
        if overlap_pct < min_overlap_pct:
            continue

        child_kernels = [
            k
            for k in kernels
            if k["global_tid"] == r["global_tid"]
            and r["start_ns"] <= k["runtime_start_ns"]
            and r["end_ns"] >= k["runtime_end_ns"]
        ]
        total_kernel_ms = sum(
            (k["kernel_end_ns"] - k["kernel_start_ns"]) / 1e6 for k in child_kernels
        )
        nccl_count = sum(1 for k in child_kernels if _is_nccl_kernel(k["kernel_name"]))
        compute_count = len(child_kernels) - nccl_count
        contains_selection = r["start_ns"] <= start_ns and r["end_ns"] >= end_ns
        overlap_ratio = overlap / selection_duration

        confidence = _confidence(
            overlap_ratio=overlap_ratio,
            contains_selection=contains_selection,
            kernel_count=len(child_kernels),
            depth=int(r["nvtx_depth"]),
        )
        reasons = [
            f"NVTX range overlaps {overlap_pct:.1f}% of the selected window",
        ]
        if contains_selection:
            reasons.append("NVTX range fully contains the selected window")
        if child_kernels:
            reasons.append(f"Contains {len(child_kernels)} GPU kernel launches in this window")
        if nccl_count:
            reasons.append(f"Includes {nccl_count} NCCL kernel launch(es)")

        candidates.append(
            {
                "candidate": r["nvtx_path"],
                "candidate_type": "nvtx_path",
                "nvtx_text": r["nvtx_text"],
                "nvtx_depth": r["nvtx_depth"],
                "confidence": confidence,
                "reason": "; ".join(reasons),
                "evidence": {
                    "global_tid": r["global_tid"],
                    "selection_start_ns": start_ns,
                    "selection_end_ns": end_ns,
                    "selection_device_id": device,
                    "nvtx_start_ns": r["start_ns"],
                    "nvtx_end_ns": r["end_ns"],
                    "overlap_ns": overlap,
                    "overlap_pct": overlap_pct,
                    "contains_selection": contains_selection,
                    "kernel_count": len(child_kernels),
                    "total_kernel_ms": round(total_kernel_ms, 3),
                    "compute_kernel_count": compute_count,
                    "nccl_kernel_count": nccl_count,
                    "top_kernels": _top_kernels(child_kernels),
                },
                "limitations": _LIMITATIONS,
            }
        )

    if not candidates:
        return [
            {
                "error": "No NVTX ranges met the overlap threshold for the selected window",
                "selection": {
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "device_id": device,
                    "min_overlap_pct": min_overlap_pct,
                },
                "limitations": _LIMITATIONS,
            }
        ]

    # Prefer candidates on threads that launched selected kernels when possible.
    candidates.sort(
        key=lambda c: (
            0
            if not kernel_tids
            else (0 if kernel_tids.get(c["evidence"]["global_tid"], 0) else 1),
            -float(c["confidence"]),
            -float(c["evidence"]["overlap_pct"]),
            -int(c["evidence"]["kernel_count"]),
            -int(c["nvtx_depth"]),
            c["candidate"],
        )
    )

    return candidates[:limit]


def _format(rows):
    if not rows:
        return "(No code attribution candidates found)"
    if "error" in rows[0]:
        msg = rows[0]["error"]
        limitations = rows[0].get("limitations", [])
        if limitations:
            msg += "\nLimitations:\n" + "\n".join(f"  - {item}" for item in limitations)
        return msg

    lines = ["── Code Attribution Candidates ──"]
    for idx, r in enumerate(rows, start=1):
        ev = r["evidence"]
        lines.append(f"{idx}. {r['candidate']}")
        lines.append(f"   Confidence: {r['confidence']:.2f}")
        lines.append(
            "   Evidence: "
            f"overlap={ev['overlap_pct']:.1f}%, "
            f"kernels={ev['kernel_count']}, "
            f"gpu_time={ev['total_kernel_ms']:.3f}ms, "
            f"compute={ev['compute_kernel_count']}, nccl={ev['nccl_kernel_count']}"
        )
        top = ev.get("top_kernels", [])
        if top:
            labels = [f"{k['kernel_name']} ({k['total_ms']:.3f}ms x{k['count']})" for k in top]
            lines.append(f"   Top kernels: {', '.join(labels)}")
        lines.append(f"   Reason: {r['reason']}")
    lines.append("")
    lines.append("Limitations:")
    for item in _LIMITATIONS:
        lines.append(f"  - {item}")
    return "\n".join(lines)


SKILL = Skill(
    name="code_attribution_candidates",
    title="Code Attribution Candidates",
    description=(
        "Ranks likely model/training code or config regions for a selected profile "
        "window using NVTX timing context, kernel evidence, confidence, and explicit "
        "limitations. This is profile-to-code-region attribution, not exact source-line tracing."
    ),
    category="nvtx",
    execute_fn=_execute,
    format_fn=_format,
    params=[
        SkillParam("start_ns", "Selected window start timestamp in ns", "int", True),
        SkillParam("end_ns", "Selected window end timestamp in ns", "int", True),
        SkillParam("device", "GPU device ID, or 'all'", "str", False, "all"),
        SkillParam("limit", "Max candidates to return", "int", False, 5),
        SkillParam("min_overlap_pct", "Minimum selected-window overlap percentage", "float", False, 0.0),
    ],
    tags=["code", "source", "nvtx", "attribution", "candidate", "trace"],
)
