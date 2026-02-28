"""
report.py — Full auto-generated performance report (analyze command).

Combines summary, overlap, NCCL breakdown, iteration detection, and NVTX
hierarchy into one "just tell me what's wrong" report for terminal and
optional markdown output. Aligns with Nsight Systems practice of focused
profiling (trim window) and post-processing SQLite results into a report.
"""
import statistics

from .profile import Profile
from .summary import gpu_summary, format_text, auto_commentary
from .overlap import (
    overlap_analysis,
    format_overlap,
    nccl_breakdown,
    format_nccl,
    detect_iterations,
    format_iterations,
)
from .tree import build_nvtx_tree


def _nvtx_hierarchy_summary(prof: Profile, device: int, trim: tuple[int, int]) -> str:
    """Short summary of top-level NVTX regions (name + duration)."""
    roots = build_nvtx_tree(prof, device, trim)
    if not roots:
        return "NVTX hierarchy: (none or no kernels in trim window)"
    lines = ["NVTX hierarchy (top-level regions)"]
    for node in roots:
        dur_ms = (node["end"] - node["start"]) / 1e6
        lines.append(f"  {node['name']}: {dur_ms:.1f}ms")
    nvtx_count = sum(1 for n in _walk_nodes(roots) if n.get("type") == "nvtx")
    kern_count = sum(1 for n in _walk_nodes(roots) if n.get("type") == "kernel")
    lines.append(f"  ({len(roots)} top-level, {nvtx_count} NVTX, {kern_count} kernels)")
    return "\n".join(lines)


def _walk_nodes(nodes):
    """Yield all nodes in tree (roots + descendants)."""
    for node in nodes:
        yield node
        for child in _walk_nodes(node.get("children", [])):
            yield child


def _iteration_regression_flags(iters: list[dict]) -> list[str]:
    """Flag iterations that are unusually slow (e.g. >1.5x median)."""
    if len(iters) < 2:
        return []
    durs = [it["duration_ms"] for it in iters]
    med = statistics.median(durs)
    if med <= 0:
        return []
    flags = []
    for it in iters:
        if it["duration_ms"] > 1.5 * med:
            pct = 100 * it["duration_ms"] / med
            flags.append(
                f"  ⚠ iter {it['iteration']}: {it['duration_ms']:.1f}ms "
                f"(~{pct:.0f}% of median {med:.1f}ms)"
            )
    return flags


def run_analyze(prof: Profile, device: int, trim: tuple[int, int]) -> dict:
    """
    Run all analyses and return a structured dict for formatting.

    Keys: summary, overlap, nccl_breakdown, iters, iters_regression, nvtx_summary.
    """
    summary = gpu_summary(prof, device, trim)
    overlap = overlap_analysis(prof, device, trim)
    nccl = nccl_breakdown(prof, device, trim)
    iters = detect_iterations(prof, device, trim)
    iters_regression = _iteration_regression_flags(iters) if iters else []
    nvtx_summary = _nvtx_hierarchy_summary(prof, device, trim)

    return {
        "summary": summary,
        "overlap": overlap,
        "nccl_breakdown": nccl,
        "iters": iters,
        "iters_regression": iters_regression,
        "nvtx_summary": nvtx_summary,
    }


def format_report_terminal(data: dict) -> str:
    """Format the full report for terminal (plain text)."""
    sections = []

    # 1. Top bottlenecks + AI-style commentary
    s = data["summary"]
    if "error" in s:
        sections.append(f"GPU {s['device']}: {s['error']}")
    else:
        sections.append(format_text(s))
        sections.append("")
        sections.append("Summary: " + auto_commentary(s))
        sections.append("")

    # 2. NVTX hierarchy summary
    sections.append(data["nvtx_summary"])
    sections.append("")

    # 3. Compute vs NCCL overlap
    ov = data["overlap"]
    if "error" in ov:
        sections.append("Overlap: " + ov["error"])
    else:
        sections.append(format_overlap(ov))
    sections.append("")

    # 4. NCCL breakdown
    sections.append(format_nccl(data["nccl_breakdown"]))
    sections.append("")

    # 5. Iteration timings + regression flags
    sections.append(format_iterations(data["iters"]))
    if data["iters_regression"]:
        sections.append("")
        sections.append("Possible regression (slow iters):")
        sections.extend(data["iters_regression"])
    sections.append("")

    return "\n".join(sections)


def format_report_markdown(data: dict, profile_path: str, trim: tuple[int, int]) -> str:
    """Format the full report as markdown (for -o file.md)."""
    s = data["summary"]
    trim_label = f"{trim[0]/1e9:.1f}s – {trim[1]/1e9:.1f}s" if trim else "full range"
    lines = [
        "# nsys-ai analyze report",
        "",
        f"- **Profile:** `{profile_path}`",
        f"- **GPU:** {s.get('device', '?')}",
        f"- **Trim:** {trim_label}",
        "",
        "---",
        "",
        "## 1. Top bottlenecks & summary",
        "",
    ]
    if "error" in s:
        lines.append(f"GPU {s['device']}: {s['error']}")
    else:
        hw = s["hardware"]
        t = s["timing"]
        lines.append(f"**GPU {s['device']}:** {hw['name']} ({hw['pci_bus']}) — {hw['sm_count']} SMs, {hw['memory_gb']}GB")
        lines.append("")
        lines.append(f"Span: {t['span_ms']:.1f}ms | Compute: {t['compute_ms']:.1f}ms | Idle: {t['idle_ms']:.1f}ms | Util: {t['utilization_pct']}%")
        lines.append("")
        lines.append("| % | Total ms | Count | Kernel |")
        lines.append("|---|----------|-------|-------|")
        for k in s["top_kernels"]:
            name_esc = (k["name"] or "").replace("|", "\\|")
            lines.append(f"| {k['pct']:.1f} | {k['total_ms']:.1f} | {k['count']} | {name_esc} |")
        lines.append("")
        lines.append(auto_commentary(s))
    lines.extend(["", "---", "", "## 2. NVTX hierarchy (top-level)", "", "```", data["nvtx_summary"], "```", ""])

    lines.extend(["---", "", "## 3. Compute vs NCCL overlap", "", "```", format_overlap(data["overlap"]), "```", ""])
    lines.extend(["---", "", "## 4. NCCL collective breakdown", "", "```", format_nccl(data["nccl_breakdown"]), "```", ""])
    lines.extend(["---", "", "## 5. Iteration timings", "", "```", format_iterations(data["iters"]), "```", ""])
    if data["iters_regression"]:
        lines.append("### Possible regression (slow iters)")
        lines.append("")
        lines.extend(data["iters_regression"])
        lines.append("")

    return "\n".join(lines)
