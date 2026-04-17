"""Generate a CUTracer instrumentation shell script from nsys profile analysis.

Workflow
--------
1. ``nsys-ai cutracer plan profile.sqlite``
   → Reads top GPU kernels from the profile.
   → Prints a ready-to-run bash script to stdout (or saves with ``--output``).

2. User edits the ``LAUNCH_CMD`` line and runs the script on a GPU machine.
   → CUTracer writes one ``*_hist.csv`` per kernel into ``OUTPUT_DIR``.

3. ``nsys-ai cutracer analyze profile.sqlite OUTPUT_DIR``
   → Correlates instruction data with NVTX attribution and reports bottlenecks.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class KernelTarget:
    """One kernel chosen for CUTracer instrumentation."""

    name: str
    """Full demangled name as stored in the nsys profile."""
    total_ms: float
    pct_of_gpu: float
    invocations: int
    category: str = "unknown"
    """Classifier category (custom_gemm, flash_attn, nccl_comms, …)."""
    cutracer_value: str = "HIGH"
    """Instrumentation value: HIGH, MEDIUM, or LOW."""
    skip_reason: str = ""
    """Non-empty when this kernel was considered but de-prioritised."""


@dataclass
class CutracerPlan:
    """Instrumentation plan derived from a single nsys profile."""

    profile_path: str
    targets: list[KernelTarget] = field(default_factory=list)
    """Top kernels selected for instrumentation, sorted by GPU time desc."""
    skipped: list[KernelTarget] = field(default_factory=list)
    """High-GPU-time kernels that were de-prioritised (LOW value)."""
    device: int = 0
    trim: tuple[float, float] | None = None
    """Trim window in seconds (start, end), matching ``--trim`` args."""


# ---------------------------------------------------------------------------
# Plan builder
# ---------------------------------------------------------------------------


def build_plan(
    conn,
    profile_path: str,
    *,
    top_n: int = 5,
    device: int = 0,
    trim: tuple[float, float] | None = None,
) -> CutracerPlan:
    """Query the nsys profile and return a :class:`CutracerPlan`.

    Fetches a wider candidate pool (``top_n * 3``), classifies each kernel
    by instrumentation value (HIGH / MEDIUM / LOW), and selects the best
    ``top_n`` targets.  LOW-value kernels (NCCL, elementwise, vendor libs)
    are moved to :attr:`CutracerPlan.skipped` so the user knows they were
    considered but de-prioritised.

    Parameters
    ----------
    conn:
        Open nsys profile connection (SQLite or DuckDB adapter).
    profile_path:
        Path to the profile file (stored for display in the generated script).
    top_n:
        How many top kernels to include in the final plan (default 5).
    device:
        GPU device index to filter on (default 0).
    trim:
        Optional (start_s, end_s) window to restrict kernel selection.
    """
    from nsys_ai.cutracer.kernel_classifier import (
        classify_kernel,
        instrumentation_priority,
    )
    from nsys_ai.skills.builtins.top_kernels import SKILL as top_kernels_skill

    # Fetch a wider pool so we can afford to skip LOW-value kernels and still
    # fill top_n slots with HIGH/MEDIUM candidates.
    candidate_limit = max(top_n * 3, 15)
    kwargs: dict = {"limit": candidate_limit, "device": device}
    if trim:
        kwargs["trim_start_ns"] = int(trim[0] * 1e9)
        kwargs["trim_end_ns"] = int(trim[1] * 1e9)

    try:
        rows = top_kernels_skill.execute_fn(conn, **kwargs)
    except Exception as exc:
        _log.warning("build_plan: top_kernels query failed: %s", exc)
        return CutracerPlan(profile_path=profile_path, device=device, trim=trim)

    if rows and "error" in rows[0]:
        return CutracerPlan(profile_path=profile_path, device=device, trim=trim)

    # Compute percentage relative to the full candidate pool total.
    total_ms = sum(r.get("total_ms", 0) for r in rows) or 1.0

    # Classify and build KernelTarget objects.
    all_targets: list[KernelTarget] = []
    for r in rows:
        ms = r.get("total_ms", 0.0)
        category, value, reason = classify_kernel(r["kernel_name"])
        all_targets.append(
            KernelTarget(
                name=r["kernel_name"],
                total_ms=ms,
                pct_of_gpu=round(ms / total_ms * 100, 1),
                invocations=r.get("invocations", 0),
                category=category,
                cutracer_value=value,
                skip_reason="" if value != "LOW" else reason,
            )
        )

    # Sort by (value priority, gpu_time desc) so HIGH kernels fill the plan first.
    all_targets.sort(key=lambda t: (instrumentation_priority(t.cutracer_value), -t.total_ms))

    selected: list[KernelTarget] = []
    skipped: list[KernelTarget] = []
    seen_names: set[str] = set()

    for t in all_targets:
        if t.name in seen_names:
            continue
        seen_names.add(t.name)
        if t.cutracer_value == "LOW":
            skipped.append(t)
        elif len(selected) < top_n:
            selected.append(t)
        else:
            # Filled top_n — remaining HIGH/MEDIUM go to skipped with a note.
            t.skip_reason = f"top-{top_n} slots already filled by higher-priority kernels"
            skipped.append(t)

    # Re-sort selected by GPU time descending for display.
    selected.sort(key=lambda t: -t.total_ms)
    skipped.sort(key=lambda t: -t.total_ms)

    return CutracerPlan(
        profile_path=profile_path,
        targets=selected,
        skipped=skipped,
        device=device,
        trim=trim,
    )


# ---------------------------------------------------------------------------
# Script generator
# ---------------------------------------------------------------------------


def format_plan_script(
    plan: CutracerPlan,
    *,
    output_dir: str = "./cutracer_out",
    launch_cmd: str = "",
    mode: str = "proton_instr_histogram",
) -> str:
    """Render the plan as a runnable bash script.

    Parameters
    ----------
    plan:
        A :class:`CutracerPlan` produced by :func:`build_plan`.
    output_dir:
        Directory where CUTracer will write histogram CSVs.
    launch_cmd:
        The training command to wrap (e.g. ``python train.py``).
        If empty, a ``TODO`` placeholder is inserted.
    mode:
        CUTracer instrumentation mode (default: ``proton_instr_histogram``).
    """
    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines += [
        "#!/usr/bin/env bash",
        "# Auto-generated by: nsys-ai cutracer plan",
        f"# Profile: {plan.profile_path}",
        "# Edit LAUNCH_CMD, then run on a machine with the target GPU.",
        "#",
        "# Workflow:",
        "#   1. Edit LAUNCH_CMD below to match your training invocation.",
        "#   2. Run: nsys-ai cutracer install   (build cutracer.so once)",
        "#   3. Run this script on a GPU machine.",
        f"#   4. Run: nsys-ai cutracer analyze {shlex.quote(plan.profile_path)} {shlex.quote(output_dir)}",
        "#      (auto-detects *_hist.csv or raw ndjson+cubin outputs)",
        "#",
    ]

    # ── Top kernels table ────────────────────────────────────────────────────
    if plan.targets:
        lines.append("# Top kernels selected for instrumentation:")
        for i, t in enumerate(plan.targets, 1):
            short = t.name if len(t.name) <= 70 else t.name[:67] + "…"
            lines.append(f"#   {i}. {t.total_ms:8.2f} ms  {t.pct_of_gpu:5.1f}%  {short}")
    else:
        lines.append("# (No kernels found in profile — check device/trim args)")
    lines.append("#")

    lines += [
        "set -euo pipefail",
        "",
    ]

    # ── Configuration ────────────────────────────────────────────────────────
    escaped_output_dir = (
        output_dir.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )
    lines += [
        '# Path to CUTracer NVBit .so — override with: export CUTRACER_SO=/your/path',
        'CUTRACER_SO="${CUTRACER_SO:-$HOME/.nsys-ai/cutracer/lib/cutracer.so}"',
        "",
        "# Output directory for histogram CSVs",
        f'OUTPUT_DIR="${{1:-{escaped_output_dir}}}"',
        'mkdir -p "$OUTPUT_DIR"',
        "",
    ]

    # ── Kernel filter ────────────────────────────────────────────────────────
    if plan.targets:
        lines.append("# Kernel name filter — CUTracer will only instrument these kernels.")
        lines.append("# CUTracer matches against the short (non-demangled) SASS name;")
        lines.append("# these are the best-guess prefixes extracted from the nsys profile.")
        lines.append("KERNEL_FILTER=\"$(cat <<'KERNELS'")
        for t in plan.targets:
            # Use normalised short name as filter token — CUTracer uses prefix matching
            from nsys_ai.cutracer.correlator import normalize_kernel_name
            lines.append(normalize_kernel_name(t.name))
        lines.append("KERNELS")
        lines.append(')\"')
        lines.append("# Join with commas for the env var")
        lines.append('KERNEL_FILTER_CSV="$(echo "$KERNEL_FILTER" | paste -sd,)"')
        lines.append("")
    else:
        lines += [
            "# No kernel filter — instruments all kernels (may be slow).",
            'KERNEL_FILTER_CSV=""',
            "",
        ]

    # ── Launch command ───────────────────────────────────────────────────────
    cmd = launch_cmd.strip() if launch_cmd.strip() else "python train.py"
    escaped_cmd = (
        cmd.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )
    lines += [
        "# Your training / inference command",
        f'LAUNCH_CMD="{escaped_cmd}"',
        '# TODO: replace LAUNCH_CMD with your real command (supports quotes/spaces).',
        "",
    ]

    # ── Safety check + run ───────────────────────────────────────────────────
    analyze_cmd = f"nsys-ai cutracer analyze {shlex.quote(plan.profile_path)} \"$OUTPUT_DIR\""
    if plan.trim:
        start_s, end_s = plan.trim
        analyze_cmd += f" --trim {start_s} {end_s}"

    lines += [
        '# Safety check',
        'if [[ ! -f "$CUTRACER_SO" ]]; then',
        '  echo "ERROR: cutracer.so not found at $CUTRACER_SO" >&2',
        '  echo "       Run: nsys-ai cutracer install" >&2',
        '  exit 1',
        'fi',
        'if ! command -v cutracer &>/dev/null; then',
        '  echo "ERROR: cutracer CLI not found. Run: pip install cutracer" >&2',
        '  exit 1',
        'fi',
        "",
        'echo "==> Running with CUTracer instrumentation"',
        'echo "    .so    : $CUTRACER_SO"',
        'echo "    filter : $KERNEL_FILTER_CSV"',
        'echo "    output : $OUTPUT_DIR"',
        'echo ""',
        "",
        "# Build cutracer trace command",
        'CUTRACER_ARGS=(',
        '  cutracer trace',
        '  --cutracer-so "$CUTRACER_SO"',
        f'  --analysis {shlex.quote(mode)}',
        '  --output-dir "$OUTPUT_DIR"',
        ')',
        'if [[ -n "$KERNEL_FILTER_CSV" ]]; then',
        '  CUTRACER_ARGS+=(--kernel-filters "$KERNEL_FILTER_CSV")',
        'fi',
        'CUTRACER_ARGS+=(-- bash -lc "$LAUNCH_CMD")',
        "",
        '"${CUTRACER_ARGS[@]}"',
        "",
        'echo ""',
        'echo "==> CUTracer trace complete."',
        'echo "    Raw traces written to: $OUTPUT_DIR"',
        'echo "    Analyze with (SASS resolution + instruction mix):"',
        f'echo "      {analyze_cmd}"',
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Human-readable plan summary (for --format table)
# ---------------------------------------------------------------------------


def format_plan_summary(plan: CutracerPlan) -> str:
    """Return a short human-readable summary of the plan (not the full script)."""
    from nsys_ai.cutracer.kernel_classifier import VALUE_HIGH, VALUE_LOW, VALUE_MEDIUM

    if not plan.targets and not plan.skipped:
        return "(No kernels found — nothing to instrument)"

    value_badge = {VALUE_HIGH: "HIGH  ", VALUE_MEDIUM: "MEDIUM", VALUE_LOW: "LOW   "}

    lines = [
        f"CUTracer Plan — {len(plan.targets)} kernel(s) selected from: {plan.profile_path}",
    ]

    if plan.targets:
        lines += [
            "",
            "  Selected for instrumentation:",
            f"  {'ms':>10}  {'%':>6}  {'calls':>7}  {'value':6}  {'category':<14}  Kernel",
            "  " + "─" * 90,
        ]
        for t in plan.targets:
            short = t.name if len(t.name) <= 42 else t.name[:39] + "…"
            badge = value_badge.get(t.cutracer_value, t.cutracer_value)
            lines.append(
                f"  {t.total_ms:>10.2f}  {t.pct_of_gpu:>5.1f}%  {t.invocations:>7d}"
                f"  {badge}  {t.category:<14}  {short}"
            )

    if plan.skipped:
        lines += [
            "",
            "  Skipped (LOW instrumentation value):",
            f"  {'ms':>10}  {'%':>6}  {'category':<14}  Reason",
            "  " + "─" * 72,
        ]
        for t in plan.skipped:
            lines.append(
                f"  {t.total_ms:>10.2f}  {t.pct_of_gpu:>5.1f}%  {t.category:<14}  {t.skip_reason}"
            )

    # Context-aware next steps
    has_high = any(t.cutracer_value == VALUE_HIGH for t in plan.targets)
    all_low = not plan.targets

    lines.append("")
    if all_low:
        lines += [
            "Note: All top kernels are LOW value for CUTracer (NCCL / vendor / elementwise).",
            "      CUTracer is unlikely to yield actionable results for this profile.",
            "      Consider: nsys-ai skill tensor_core_usage  OR  nsys-ai skill nccl_breakdown",
        ]
    else:
        lines += ["Next steps (choose a path):", ""]
        if has_high:
            lines += [
                "  Local GPU:",
                "    nsys-ai cutracer install                         # build cutracer.so (once)",
                "    nsys-ai cutracer plan <profile> --script         # generate run script",
                "    # edit LAUNCH_CMD, then run the script",
                "    nsys-ai cutracer analyze <profile> <output_dir>  # bottleneck report",
                "",
                "  Modal (cloud GPU, no local GPU needed):",
                "    nsys-ai cutracer run <profile> --launch-cmd '...' --backend modal",
                "    modal run <output>_cutracer.py",
                "    nsys-ai cutracer analyze <profile> <output_dir>  # bottleneck report",
                "",
                "  Agent interpretation (after analyze):",
                "    nsys-ai agent <profile>  → mention trace_dir to trigger cutracer_analysis skill",
            ]
        else:
            lines += [
                "  Note: Selected kernels are MEDIUM value — useful to verify Tensor Core usage.",
                "    nsys-ai cutracer plan <profile> --script  # generate run script",
                "    nsys-ai cutracer analyze <profile> <output_dir>",
            ]

    return "\n".join(lines)
