"""
summary.py — Generate concise profile summary reports.

Produces a structured overview of a GPU profile: hardware, kernel stats,
NVTX breakdown, stream utilization, and NCCL timing. Designed for both
agent consumption and human reading.
"""
from typing import Optional
from .profile import Profile


def gpu_summary(prof: Profile, device: int,
                trim: Optional[tuple[int, int]] = None) -> dict:
    """
    Generate a summary report for one GPU.

    Returns a dict with: hardware, top_kernels, streams, nvtx_top_level, timing.
    """
    info = prof.meta.gpu_info.get(device)
    kernels = prof.kernels(device, trim)

    if not kernels:
        return {"device": device, "error": "no kernels found"}

    # Timing
    k_starts = [k["start"] for k in kernels]
    k_ends = [k["end"] for k in kernels]
    span_ns = max(k_ends) - min(k_starts)

    # Top kernels by total duration
    from collections import Counter, defaultdict
    dur_by_name = defaultdict(float)
    count_by_name = Counter()
    for k in kernels:
        dur_by_name[k["name"]] += (k["end"] - k["start"]) / 1e6  # ms
        count_by_name[k["name"]] += 1

    top = sorted(dur_by_name.items(), key=lambda x: -x[1])[:10]

    # Stream breakdown
    stream_kernels = defaultdict(int)
    stream_dur = defaultdict(float)
    for k in kernels:
        stream_kernels[k["streamId"]] += 1
        stream_dur[k["streamId"]] += (k["end"] - k["start"]) / 1e6

    # Idle gaps (periods with no kernel active)
    sorted_kernels = sorted(kernels, key=lambda k: k["start"])
    idle_ns = 0
    max_end = sorted_kernels[0]["end"]
    for k in sorted_kernels[1:]:
        if k["start"] > max_end:
            idle_ns += k["start"] - max_end
        max_end = max(max_end, k["end"])

    total_compute_ns = sum(k["end"] - k["start"] for k in kernels)

    return {
        "device": device,
        "hardware": {
            "name": info.name if info else "unknown",
            "pci_bus": info.pci_bus if info else "",
            "sm_count": info.sm_count if info else 0,
            "memory_gb": round(info.memory_bytes / 1e9, 1) if info else 0,
        },
        "timing": {
            "span_ms": round(span_ns / 1e6, 2),
            "compute_ms": round(total_compute_ns / 1e6, 2),
            "idle_ms": round(idle_ns / 1e6, 2),
            "utilization_pct": round(100 * total_compute_ns / span_ns, 1) if span_ns else 0,
        },
        "kernel_count": len(kernels),
        "top_kernels": [
            {"name": name, "total_ms": round(ms, 3), "count": count_by_name[name],
             "pct": round(100 * ms / (total_compute_ns / 1e6), 1)}
            for name, ms in top
        ],
        "streams": {
            sid: {"kernels": stream_kernels[sid], "total_ms": round(stream_dur[sid], 2)}
            for sid in sorted(stream_kernels)
        },
    }


def format_text(summary: dict) -> str:
    """Format a GPU summary as readable text."""
    if "error" in summary:
        return f"GPU {summary['device']}: {summary['error']}"

    hw = summary["hardware"]
    t = summary["timing"]
    lines = [
        f"GPU {summary['device']}: {hw['name']} ({hw['pci_bus']}) — {hw['sm_count']} SMs, {hw['memory_gb']}GB",
        f"  Span: {t['span_ms']:.1f}ms | Compute: {t['compute_ms']:.1f}ms | Idle: {t['idle_ms']:.1f}ms | Util: {t['utilization_pct']}%",
        f"  Kernels: {summary['kernel_count']}",
        "",
        "  Top kernels:",
    ]
    for k in summary["top_kernels"]:
        lines.append(f"    {k['pct']:5.1f}%  {k['total_ms']:8.1f}ms  ×{k['count']:<4d}  {k['name']}")

    lines.append("")
    lines.append("  Streams:")
    snames = {21: "Compute", 56: "NCCL"}
    for sid, s in summary["streams"].items():
        label = snames.get(sid, "Aux")
        lines.append(f"    Stream {sid} ({label}): {s['kernels']} kernels, {s['total_ms']:.1f}ms")

    return "\n".join(lines)


def auto_commentary(summary: dict) -> str:
    """
    Generate a natural language paragraph summarizing a GPU profile.

    Designed for LLM agent consumption — highlights the top bottleneck,
    compute/NCCL split, and overall utilization in plain English.
    """
    if "error" in summary:
        return f"GPU {summary['device']}: {summary['error']}"

    hw = summary["hardware"]
    t = summary["timing"]
    top = summary["top_kernels"]

    sentences = []
    sentences.append(
        f"GPU {summary['device']} ({hw['name']}) ran {summary['kernel_count']} kernels "
        f"over {t['span_ms']:.0f}ms with {t['utilization_pct']}% utilization."
    )

    if top:
        sentences.append(
            f"The top bottleneck is **{top[0]['name']}** at {top[0]['pct']}% "
            f"of compute time ({top[0]['total_ms']:.0f}ms across {top[0]['count']} calls)."
        )

    # Compute vs NCCL split from streams
    nccl_ms = sum(s["total_ms"] for sid, s in summary["streams"].items()
                  if sid in (56,))
    compute_ms = sum(s["total_ms"] for sid, s in summary["streams"].items()
                     if sid not in (56,))
    total_stream_ms = nccl_ms + compute_ms
    if total_stream_ms > 0 and nccl_ms > 0:
        nccl_pct = 100 * nccl_ms / total_stream_ms
        sentences.append(
            f"NCCL collectives account for {nccl_pct:.0f}% of kernel time "
            f"({nccl_ms:.0f}ms), with compute at {compute_ms:.0f}ms."
        )

    if t["idle_ms"] > 10:
        idle_pct = 100 * t["idle_ms"] / t["span_ms"]
        sentences.append(
            f"There are {t['idle_ms']:.0f}ms of idle gaps ({idle_pct:.0f}% of span)."
        )

    return " ".join(sentences)

