"""
diff_render.py — Render structured diff payloads for CLI outputs.
"""

from __future__ import annotations

import json

from .ai.diff_narrative import DiffNarrative
from .diff import ProfileDiffSummary


def _fmt_ns(ns: int) -> str:
    ms = ns / 1e6
    if abs(ms) >= 1000:
        return f"{ms / 1000:.2f}s"
    if abs(ms) >= 1:
        return f"{ms:.2f}ms"
    us = ns / 1e3
    if abs(us) >= 1:
        return f"{us:.2f}us"
    return f"{ns}ns"


def _fmt_delta_ns(ns: int) -> str:
    """Format an nanosecond delta with an explicit +/- sign."""
    s = _fmt_ns(ns)
    if ns > 0:
        return f"+{s}"
    # negative numbers already formatting with a leading '-', and 0 is fine without sign.
    return s


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def format_diff_terminal(
    data: ProfileDiffSummary,
    narrative: DiffNarrative | None = None,
) -> str:
    lines: list[str] = []
    lines.append("Profile Diff")
    lines.append("─" * 60)
    lines.append(f"Before: {data.before.path}")
    lines.append(f"After:  {data.after.path}")
    lines.append(f"GPU:    {data.before.gpu}")
    lines.append("")

    if narrative:
        lines.append("Executive Summary")
        lines.append("─" * 60)
        lines.append(narrative.executive_summary)
        lines.append("")

    if data.warnings:
        lines.append("Warnings:")
        for w in data.warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("Overall")
    lines.append("─" * 60)
    lines.append(
        f"Total GPU: {_fmt_ns(data.before.total_gpu_ns)} → {_fmt_ns(data.after.total_gpu_ns)}  (Δ {_fmt_delta_ns(data.after.total_gpu_ns - data.before.total_gpu_ns)})"
    )
    if data.overlap_delta:
        b = data.overlap_before
        a = data.overlap_after
        lines.append("Overlap (ms):")
        for k in ("compute_only_ms", "nccl_only_ms", "overlap_ms", "idle_ms", "total_ms"):
            if k in b and k in a:
                d = data.overlap_delta.get(k, 0)
                sign = "+" if d > 0 else ""
                lines.append(f"- {k:<15} {b[k]:>9.2f} → {a[k]:>9.2f}  (Δ {sign}{d})")
        if "overlap_pct" in b and "overlap_pct" in a:
            d = data.overlap_delta.get("overlap_pct", 0)
            sign = "+" if d > 0 else ""
            lines.append(
                f"- {'overlap_pct':<15} {b['overlap_pct']:>8.1f}% → {a['overlap_pct']:>8.1f}%  (Δ {sign}{d}%)"
            )
    lines.append("")

    def add_kernel_section(title: str, rows):
        lines.append(title)
        lines.append("─" * 60)
        if not rows:
            lines.append("(none)\n")
            return
        # Header
        lines.append(f"{'Δ Time':>10}  | {'Count Change':>13} | Kernel")
        lines.append(f"{'-' * 10}--+-{'-' * 13}-+-{'-' * 30}")
        for kd in rows:
            count_str = f"{kd.before_count}->{kd.after_count}"
            lines.append(f"{_fmt_delta_ns(kd.delta_ns):>10}  | {count_str:>13} | {kd.name}")
            # Detail row
            lines.append(
                f"{' ':>10}  | {' ':>13} |   before: {_fmt_ns(kd.before_total_ns):>7} ({_fmt_pct(kd.before_share):>6})  after: {_fmt_ns(kd.after_total_ns):>7} ({_fmt_pct(kd.after_share):>6})"
            )
        lines.append("")

    add_kernel_section("Top regressions (kernels)", data.top_regressions)
    add_kernel_section("Top improvements (kernels)", data.top_improvements)

    # NVTX (lightweight)
    nvtx_reg = [n for n in data.nvtx_diffs if n.delta_ns > 0][:10]
    nvtx_imp = [n for n in data.nvtx_diffs if n.delta_ns < 0][:10]
    lines.append("Top NVTX regressions (range wall time; v1 signal)")
    lines.append("─" * 60)
    if nvtx_reg:
        for n in nvtx_reg:
            lines.append(
                f"{_fmt_delta_ns(n.delta_ns):>10}  {n.text}  (count {n.before_count}->{n.after_count})"
            )
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("Top NVTX improvements (range wall time; v1 signal)")
    lines.append("─" * 60)
    if nvtx_imp:
        for n in nvtx_imp:
            lines.append(
                f"{_fmt_delta_ns(n.delta_ns):>10}  {n.text}  (count {n.before_count}->{n.after_count})"
            )
    else:
        lines.append("(none)")
    lines.append("")

    if narrative:
        if narrative.ai_narrative:
            lines.append("AI Narrative")
            lines.append("─" * 60)
            lines.append(narrative.ai_narrative)
            lines.append("")
        if narrative.warning:
            lines.append("Note")
            lines.append("─" * 60)
            lines.append(narrative.warning)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def format_diff_terminal_multi(
    global_summary: ProfileDiffSummary,
    per_gpu: dict[int, ProfileDiffSummary],
    narrative: DiffNarrative | None = None,
) -> str:
    """Render a multi-GPU diff: global view + per-GPU overview + per-GPU top-k."""
    parts: list[str] = []

    # 1) Global section (reuse single-GPU formatter but tweak header)
    header = "Profile Diff (All GPUs)\n" + "─" * 60 + "\n"
    global_block = format_diff_terminal(global_summary, narrative=narrative)
    # Drop the first two lines ("Profile Diff" + underline) from the reused block.
    gl_lines = global_block.splitlines()
    if len(gl_lines) >= 2:
        gl_lines = gl_lines[2:]
        # Normalize "GPU: None" to "all" when rendering the all-GPU summary.
        processed: list[str] = []
        for line in gl_lines:
            stripped = line.lstrip()
            if stripped.startswith("GPU:") and "None" in line:
                processed.append(line.replace("None", "all"))
            else:
                processed.append(line)
        gl_lines = processed
    parts.append(header + "\n".join(gl_lines).rstrip() + "\n")

    # 2) Per-GPU overview table
    if per_gpu:
        parts.append("Per-GPU Overview")
        parts.append("─" * 60)
        parts.append(
            f"{'GPU':>3} | {'Before Total':>13} | {'After Total':>13} | {'Δ':>10} | {'Overlap % (B→A)':>18}"
        )
        parts.append(f"{'-' * 3}-+-{'-' * 13}-+-{'-' * 13}-+-{'-' * 10}-+-{'-' * 18}")
        for dev, summary in sorted(per_gpu.items()):
            b = summary.before
            a = summary.after
            delta_ns = a.total_gpu_ns - b.total_gpu_ns
            ov_b = b.overlap.get("overlap_pct")
            ov_a = a.overlap.get("overlap_pct")
            if isinstance(ov_b, (int, float)) and isinstance(ov_a, (int, float)):
                ov_str = f"{ov_b:4.1f}% → {ov_a:4.1f}%"
            else:
                ov_str = "n/a"
            parts.append(
                f"{dev:>3} | {_fmt_ns(b.total_gpu_ns):>13} | {_fmt_ns(a.total_gpu_ns):>13} | "
                f"{_fmt_delta_ns(delta_ns):>10} | {ov_str:>18}"
            )
        parts.append("")

    # 3) Per-GPU top regressions (short)
    for dev, summary in sorted(per_gpu.items()):
        if not summary.top_regressions:
            continue
        parts.append(f"Top regressions (GPU {dev})")
        parts.append("─" * 60)
        parts.append(f"{'Δ Time':>10}  | {'Count Change':>13} | Kernel")
        parts.append(f"{'-' * 10}--+-{'-' * 13}-+-{'-' * 30}")
        for kd in summary.top_regressions:
            count_str = f"{kd.before_count}->{kd.after_count}"
            parts.append(f"{_fmt_delta_ns(kd.delta_ns):>10}  | {count_str:>13} | {kd.name}")
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def format_diff_markdown(
    data: ProfileDiffSummary,
    narrative: DiffNarrative | None = None,
) -> str:
    md: list[str] = []
    md.append("## Profile Diff")
    md.append("")
    md.append(f"- **Before**: `{data.before.path}`")
    md.append(f"- **After**: `{data.after.path}`")
    md.append(f"- **GPU**: `{data.before.gpu}`")
    md.append("")

    if narrative:
        md.append("### Executive Summary")
        md.append("")
        md.append(narrative.executive_summary)
        md.append("")

    if data.warnings:
        md.append("### Warnings")
        for w in data.warnings:
            md.append(f"- {w}")
        md.append("")

    md.append("### Overall")
    md.append("")
    md.append(
        f"- **Total GPU**: `{_fmt_ns(data.before.total_gpu_ns)}` → `{_fmt_ns(data.after.total_gpu_ns)}` (Δ `{_fmt_delta_ns(data.after.total_gpu_ns - data.before.total_gpu_ns)}`)"
    )
    if data.overlap_delta:
        b = data.overlap_before
        a = data.overlap_after
        md.append("- **Overlap (ms)**:")
        for k in ("compute_only_ms", "nccl_only_ms", "overlap_ms", "idle_ms", "total_ms"):
            if k in b and k in a:
                d = data.overlap_delta.get(k, 0)
                sign = "+" if d > 0 else ""
                md.append(f"  - `{k}`: `{b[k]}` → `{a[k]}` (Δ `{sign}{d}`)")
        if "overlap_pct" in b and "overlap_pct" in a:
            d = data.overlap_delta.get("overlap_pct", 0)
            sign = "+" if d > 0 else ""
            md.append(
                f"  - `overlap_pct`: `{b['overlap_pct']}%` → `{a['overlap_pct']}%` (Δ `{sign}{d}%`)"
            )
    md.append("")

    def add_table(title: str, rows, is_regression: bool):
        md.append(f"### {title}")
        md.append("")
        if not rows:
            md.append("_none_")
            md.append("")
            return

        icon = "🚨 " if is_regression else "✅ "
        md.append("| Δ | kernel | before | after | count |")
        md.append("|---:|---|---:|---:|---:|")
        for kd in rows:
            # Emphasize removed/new counts
            c_str = f"{kd.before_count} → {kd.after_count}"
            if kd.classification == "new":
                c_str = f"**0 → {kd.after_count}**"
            elif kd.classification == "removed":
                c_str = f"**{kd.before_count} → 0**"

            md.append(
                f"| {icon}`{_fmt_delta_ns(kd.delta_ns)}` | `{kd.name}` | `{_fmt_ns(kd.before_total_ns)}` ({_fmt_pct(kd.before_share)}) | `{_fmt_ns(kd.after_total_ns)}` ({_fmt_pct(kd.after_share)}) | {c_str} |"
            )
        md.append("")

    add_table("Top regressions (kernels)", data.top_regressions, is_regression=True)
    add_table("Top improvements (kernels)", data.top_improvements, is_regression=False)

    if narrative:
        if narrative.ai_narrative:
            md.append("### AI Narrative")
            md.append("")
            md.append(narrative.ai_narrative)
            md.append("")
        if narrative.warning:
            md.append("### Note")
            md.append("")
            md.append(narrative.warning)
            md.append("")

    return "\n".join(md).rstrip() + "\n"


def format_diff_markdown_multi(
    global_summary: ProfileDiffSummary,
    per_gpu: dict[int, ProfileDiffSummary],
    narrative: DiffNarrative | None = None,
) -> str:
    """Render a multi-GPU markdown report: global + per-GPU overview + per-GPU top-k."""
    md: list[str] = []
    g = global_summary
    md.append("## Profile Diff (All GPUs)")
    md.append("")
    md.append(f"- **Before**: `{g.before.path}`")
    md.append(f"- **After**: `{g.after.path}`")
    md.append("")

    if narrative:
        md.append("### Executive Summary")
        md.append("")
        md.append(narrative.executive_summary)
        md.append("")

    # Warnings (global)
    if g.warnings:
        md.append("### Warnings")
        for w in g.warnings:
            md.append(f"- {w}")
        md.append("")

    # 1) Global overall
    md.append("### 1. Global Overall (Aggregated)")
    md.append("")
    md.append(
        f"- **Total GPU**: `{_fmt_ns(g.before.total_gpu_ns)}` → "
        f"`{_fmt_ns(g.after.total_gpu_ns)}` (Δ `{_fmt_delta_ns(g.after.total_gpu_ns - g.before.total_gpu_ns)}`)"
    )
    if g.overlap_delta:
        b = g.overlap_before
        a = g.overlap_after
        md.append("- **Overlap (ms)**:")
        for k in ("compute_only_ms", "nccl_only_ms", "overlap_ms", "idle_ms", "total_ms"):
            if k in b and k in a:
                d = g.overlap_delta.get(k, 0)
                sign = "+" if d > 0 else ""
                md.append(f"  - `{k}`: `{b[k]}` → `{a[k]}` (Δ `{sign}{d}`)")
        if "overlap_pct" in b and "overlap_pct" in a:
            d = g.overlap_delta.get("overlap_pct", 0)
            sign = "+" if d > 0 else ""
            md.append(
                f"  - `overlap_pct`: `{b['overlap_pct']}%` → `{a['overlap_pct']}%` (Δ `{sign}{d}%`)"
            )
    md.append("")

    # 2) Per-GPU breakdown table
    if per_gpu:
        md.append("### 2. Per-GPU Breakdown (Load Balancing)")
        md.append("")
        md.append(
            "| GPU | Total Time (Before) | Total Time (After) | Δ | Overlap % (Before → After) |"
        )
        md.append("|---:|---:|---:|---:|---:|")
        for dev, summary in sorted(per_gpu.items()):
            b = summary.before
            a = summary.after
            delta_ns = a.total_gpu_ns - b.total_gpu_ns
            ov_b = b.overlap.get("overlap_pct")
            ov_a = a.overlap.get("overlap_pct")
            if isinstance(ov_b, (int, float)) and isinstance(ov_a, (int, float)):
                ov_str = f"{ov_b:.1f}% → {ov_a:.1f}%"
            else:
                ov_str = "n/a"
            md.append(
                f"| `{dev}` | `{_fmt_ns(b.total_gpu_ns)}` | `{_fmt_ns(a.total_gpu_ns)}` | "
                f"`{_fmt_delta_ns(delta_ns)}` | `{ov_str}` |"
            )
        md.append("")

    # 3) Global top regressions / improvements (reuse existing tables)
    md.append("---")
    md.append("")
    md.append("### 3. Global Top Regressions (Kernels)")
    md.append("")
    global_md = format_diff_markdown(g)
    marker = "### Top regressions"
    section = ""
    if marker in global_md:
        tail = global_md.split(marker, 1)[1]
        lines_after = tail.splitlines()
        body_lines: list[str] = []
        for line in lines_after[1:]:
            if line.strip().startswith("### "):
                break
            body_lines.append(line)
        section = "\n".join(body_lines).strip()
    if section:
        md.append(section)

    # 4) Per-GPU top regressions (short)
    if per_gpu:
        md.append("---")
        md.append("")
        md.append("### 4. Per-GPU Top Regressions")
        md.append("")
        for dev, summary in sorted(per_gpu.items()):
            if not summary.top_regressions:
                continue
            md.append(f"#### GPU {dev}")
            md.append("")
            md.append("| Δ | kernel | before | after | count |")
            md.append("|---:|---|---:|---:|---:|")
            for kd in summary.top_regressions:
                c_str = f"{kd.before_count} → {kd.after_count}"
                if kd.classification == "new":
                    c_str = f"**0 → {kd.after_count}**"
                elif kd.classification == "removed":
                    c_str = f"**{kd.before_count} → 0**"
                delta_str = _fmt_delta_ns(kd.delta_ns)
                md.append(
                    f"| `{delta_str}` | `{kd.name}` | "
                    f"`{_fmt_ns(kd.before_total_ns)}` ({_fmt_pct(kd.before_share)}) | "
                    f"`{_fmt_ns(kd.after_total_ns)}` ({_fmt_pct(kd.after_share)}) | {c_str} |"
                )
            md.append("")

    if narrative:
        if narrative.ai_narrative:
            md.append("### AI Narrative")
            md.append("")
            md.append(narrative.ai_narrative)
            md.append("")
        if narrative.warning:
            md.append("### Note")
            md.append("")
            md.append(narrative.warning)
            md.append("")

    return "\n".join(md).rstrip() + "\n"


def to_diff_json(data: ProfileDiffSummary) -> str:
    # Keep this relatively stable; tests can snapshot it.
    payload = {
        "before": {
            "path": data.before.path,
            "gpu": data.before.gpu,
            "schema_version": data.before.schema_version,
            "total_gpu_ns": data.before.total_gpu_ns,
        },
        "after": {
            "path": data.after.path,
            "gpu": data.after.gpu,
            "schema_version": data.after.schema_version,
            "total_gpu_ns": data.after.total_gpu_ns,
        },
        "warnings": list(data.warnings),
        "top_regressions": [
            {
                "key": k.key,
                "name": k.name,
                "demangled": k.demangled,
                "before_total_ns": k.before_total_ns,
                "after_total_ns": k.after_total_ns,
                "delta_ns": k.delta_ns,
                "before_count": k.before_count,
                "after_count": k.after_count,
                "classification": k.classification,
                "before_share": k.before_share,
                "after_share": k.after_share,
                "delta_share": k.delta_share,
            }
            for k in data.top_regressions
        ],
        "top_improvements": [
            {
                "key": k.key,
                "name": k.name,
                "demangled": k.demangled,
                "before_total_ns": k.before_total_ns,
                "after_total_ns": k.after_total_ns,
                "delta_ns": k.delta_ns,
                "before_count": k.before_count,
                "after_count": k.after_count,
                "classification": k.classification,
                "before_share": k.before_share,
                "after_share": k.after_share,
                "delta_share": k.delta_share,
            }
            for k in data.top_improvements
        ],
        "nvtx_regressions": [
            {
                "text": n.text,
                "before_total_ns": n.before_total_ns,
                "after_total_ns": n.after_total_ns,
                "delta_ns": n.delta_ns,
                "before_count": n.before_count,
                "after_count": n.after_count,
                "classification": n.classification,
            }
            for n in sorted(
                (n for n in data.nvtx_diffs if n.delta_ns > 0),
                key=lambda x: x.delta_ns,
                reverse=True,
            )[:20]
        ],
        "nvtx_improvements": [
            {
                "text": n.text,
                "before_total_ns": n.before_total_ns,
                "after_total_ns": n.after_total_ns,
                "delta_ns": n.delta_ns,
                "before_count": n.before_count,
                "after_count": n.after_count,
                "classification": n.classification,
            }
            for n in sorted(
                (n for n in data.nvtx_diffs if n.delta_ns < 0),
                key=lambda x: x.delta_ns,
            )[:20]
        ],
        "overlap": {
            "before": data.overlap_before,
            "after": data.overlap_after,
            "delta": data.overlap_delta,
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
