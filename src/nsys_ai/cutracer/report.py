"""Format CUTracer analysis results for human and machine consumption.

Produces:
- A structured dict (machine-readable, suitable for LLM agent context)
- A terminal-friendly text report
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .parser import KernelHistogram
from .sass_ops import classify_opcode, stall_score

# ---------------------------------------------------------------------------
# Bottleneck classification thresholds
# ---------------------------------------------------------------------------

_STALL_SCORE_MIN: float = 0.05        # minimum stall score to include in top_stalls list
_TOP_STALLS_N: int = 5                # how many top stalls to surface per kernel
_MEM_STALL_THRESHOLD: float = 0.25    # sum of memory stall scores → memory-bound verdict
_MEMORY_PCT_THRESHOLD: float = 30.0   # memory instruction % → memory-bound verdict
_SYNC_PCT_THRESHOLD: float = 15.0     # sync instruction % → sync-bound verdict
_COMPUTE_PCT_THRESHOLD: float = 60.0  # compute+tensor % → compute-bound verdict
# Bank conflict: high LDS/STS stall AND meaningful compute activity
_BANK_CONFLICT_LDS_STALL: float = 0.30   # combined LDS+STS stall score threshold
_BANK_CONFLICT_COMPUTE_MIN: float = 20.0 # minimum compute+tensor % to flag conflict


# ---------------------------------------------------------------------------
# Instruction mix
# ---------------------------------------------------------------------------


@dataclass
class InstrMix:
    """Categorised instruction mix for one kernel."""

    kernel_name: str
    total_count: int
    category_pct: dict[str, float] = field(default_factory=dict)
    """Category → percentage of total instruction count (0–100)."""

    top_stalls: list[tuple[str, float]] = field(default_factory=list)
    """Top stalling opcodes: [(opcode, stall_score), …] sorted descending."""

    tc_active: bool = False
    """True when Tensor Core opcodes (HMMA/IMMA/…) are present."""

    cycles_per_instr: dict[str, float] = field(default_factory=dict)
    """Opcode → avg cycles/instruction (includes stall cycles)."""

    bottleneck: str = "unknown"
    """Coarse bottleneck label: 'compute', 'memory', 'sync', or 'balanced'."""

    bank_conflict_hint: bool = False
    """True when LDS/STS stall scores suggest shared memory bank conflicts."""


def compute_mix(hist: KernelHistogram) -> InstrMix:
    """Derive :class:`InstrMix` from a :class:`KernelHistogram`."""
    total = hist.total_count
    if total == 0:
        return InstrMix(kernel_name=hist.kernel_name, total_count=0)

    # Aggregate by category
    cat_counts: dict[str, int] = {}
    stall_scores: list[tuple[str, float]] = []
    cpi: dict[str, float] = {}

    for opcode, count in hist.instruction_counts.items():
        cat = classify_opcode(opcode)
        cat_counts[cat] = cat_counts.get(cat, 0) + count
        cycles = hist.instruction_cycles.get(opcode, 0)
        ss = stall_score(opcode, cycles, count)
        if ss > _STALL_SCORE_MIN:
            stall_scores.append((opcode, round(ss, 3)))
        if count:
            cpi[opcode] = round(cycles / count, 1)

    cat_pct = {cat: round(cnt / total * 100, 1) for cat, cnt in cat_counts.items()}
    stall_scores.sort(key=lambda x: -x[1])

    tc_active = any(
        count > 0 and classify_opcode(opcode) == "tensor"
        for opcode, count in hist.instruction_counts.items()
    )

    # Coarse bottleneck heuristic
    compute_pct = cat_pct.get("compute", 0) + cat_pct.get("tensor", 0)
    memory_pct = cat_pct.get("memory", 0)
    sync_pct = cat_pct.get("sync", 0)
    mem_stall = sum(s for op, s in stall_scores if classify_opcode(op) == "memory")

    if mem_stall > _MEM_STALL_THRESHOLD or memory_pct > _MEMORY_PCT_THRESHOLD:
        bottleneck = "memory"
    elif sync_pct > _SYNC_PCT_THRESHOLD:
        bottleneck = "sync"
    elif compute_pct > _COMPUTE_PCT_THRESHOLD:
        bottleneck = "compute"
    else:
        bottleneck = "balanced"

    # Bank conflict hint: LDS + STS stall scores together are high while
    # the kernel is doing meaningful compute — classic symptom of shared
    # memory bank conflicts (consecutive threads accessing the same bank).
    lds_sts_stall = sum(
        s for op, s in stall_scores
        if op.startswith("LDS") or op.startswith("STS")
    )
    bank_conflict_hint = (
        lds_sts_stall > _BANK_CONFLICT_LDS_STALL
        and compute_pct > _BANK_CONFLICT_COMPUTE_MIN
    )

    return InstrMix(
        kernel_name=hist.kernel_name,
        total_count=total,
        category_pct=cat_pct,
        top_stalls=stall_scores[:_TOP_STALLS_N],
        tc_active=tc_active,
        cycles_per_instr=cpi,
        bottleneck=bottleneck,
        bank_conflict_hint=bank_conflict_hint,
    )


# ---------------------------------------------------------------------------
# Enriched per-kernel report (adds nsys context)
# ---------------------------------------------------------------------------


@dataclass
class KernelReport:
    """Combined CUTracer instruction data + nsys NVTX context."""

    mix: InstrMix
    nsys_kernel_name: str | None = None
    nvtx_path: str | None = None
    """NVTX attribution path from nsys (e.g. 'Layer 12 > Attn Backward')."""
    total_ms: float | None = None
    """GPU time from nsys profile (ms)."""
    pct_of_gpu: float | None = None
    """Percentage of total GPU time."""
    achieved_warps: int | None = None
    """Number of distinct warp IDs seen — proxy for warp occupancy."""


def format_kernel_report(report: KernelReport) -> str:
    """Return a terminal-friendly text block for one kernel."""
    mix = report.mix
    lines: list[str] = []

    sep = "━" * 60
    lines.append(sep)

    # Header — show CUTracer name, then nsys match if different
    ct_name = mix.kernel_name
    nsys_name = report.nsys_kernel_name
    display_name = nsys_name or ct_name
    if len(display_name) > 55:
        display_name = display_name[:52] + "…"
    lines.append(f"Kernel: {display_name}")

    # Show CUTracer source name when it differs from the nsys match
    if nsys_name and nsys_name != ct_name:
        lines.append(f"  CUTracer: {ct_name}")

    # Correlation / nsys enrichment
    if nsys_name is None:
        lines.append("  Match:  (not found in nsys profile — instruction mix only)")
    if report.nvtx_path:
        lines.append(f"  NVTX:   {report.nvtx_path}")
    if report.total_ms is not None:
        pct_str = (
            f"  ({report.pct_of_gpu:.1f}% of GPU time)"
            if report.pct_of_gpu is not None
            else ""
        )
        lines.append(f"  nsys:   {report.total_ms:.2f} ms{pct_str}")
    if report.achieved_warps is not None:
        lines.append(f"  Warps:  {report.achieved_warps} distinct warp IDs observed")

    lines.append("")

    # Instruction mix bar chart
    lines.append("Instruction Mix:")
    order = ["tensor", "compute", "memory", "sync", "control", "special", "other"]
    bar_width = 20
    for cat in order:
        pct = mix.category_pct.get(cat, 0)
        if pct == 0:
            continue
        filled = round(pct / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        suffix = " ← Tensor Core" if cat == "tensor" and mix.tc_active else ""
        lines.append(f"  {bar}  {cat:<8s} {pct:5.1f}%{suffix}")

    lines.append("")

    # Stall analysis
    if mix.top_stalls:
        lines.append("Top Stalls (stall score = fraction of cycles wasted):")
        for opcode, score in mix.top_stalls:
            cpi = mix.cycles_per_instr.get(opcode, 0)
            bar = "▓" * round(score * 20)
            lines.append(f"  {opcode:<10s} {bar:<20s} {score*100:4.0f}%  ({cpi:.0f} cyc/instr)")

    lines.append("")

    # Bank conflict warning (shown before bottleneck verdict)
    if mix.bank_conflict_hint:
        lines.append("  ⚠ Shared memory bank conflict suspected (high LDS/STS stall score).")
        lines.append("    Fix: pad shared memory arrays by 1 element per row to scatter accesses.")

    # Bottleneck verdict
    bottleneck_msgs = {
        "memory": "MEMORY-BOUND — high LDG/STG stall. Consider tiling, prefetch, or algorithm changes.",
        "compute": "COMPUTE-BOUND — high arithmetic utilisation. Consider mixed-precision or algorithmic improvements.",
        "sync": "SYNC-BOUND — many warp barriers. Consider reducing synchronisation or restructuring the kernel.",
        "balanced": "BALANCED — no single dominant bottleneck detected.",
        "unknown": "UNKNOWN — insufficient data.",
    }
    lines.append(f"Bottleneck: {mix.bottleneck.upper()}")
    lines.append(f"  {bottleneck_msgs.get(mix.bottleneck, '')}")
    lines.append(sep)

    return "\n".join(lines)


def to_dict(report: KernelReport) -> dict:
    """Return a JSON-serialisable dict for LLM agent consumption."""
    mix = report.mix
    return {
        "kernel_name": report.nsys_kernel_name or mix.kernel_name,
        "cutracer_kernel_name": mix.kernel_name,
        "nvtx_path": report.nvtx_path,
        "total_ms": report.total_ms,
        "pct_of_gpu": report.pct_of_gpu,
        "total_instructions": mix.total_count,
        "instruction_mix_pct": mix.category_pct,
        "tensor_core_active": mix.tc_active,
        "bottleneck": mix.bottleneck,
        "bank_conflict_hint": mix.bank_conflict_hint,
        "achieved_warps": report.achieved_warps,
        "top_stalls": [
            {"opcode": op, "stall_score": s, "cycles_per_instr": mix.cycles_per_instr.get(op)}
            for op, s in mix.top_stalls
        ],
    }


def summarize_all(reports: list[KernelReport]) -> str:
    """Return a concatenated text summary of all kernel reports."""
    if not reports:
        return "(No CUTracer kernel data found)"
    return "\n\n".join(format_kernel_report(r) for r in reports)
