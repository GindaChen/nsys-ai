"""Root cause pattern matcher.

Programmatically detects known GPU performance anti-patterns from
the Book of Root Causes using existing skill data.

Each pattern has:
  - name: canonical root cause name
  - check: function that examines skill outputs and returns match info
  - severity: critical / warning / info
  - recommendation: actionable fix suggestion

This is a Python-level skill that runs other skills internally
to gather evidence, then matches against known patterns.
"""

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    """Run all pattern matchers against the profile."""
    findings = []

    # Gather evidence from skills (forward trim kwargs)

    # Top kernels (request a larger slice to reduce bias in hotspot detection)
    top_kernels_data = _safe_execute("top_kernels", conn, limit=1000, **kwargs)
    # GPU idle gaps
    idle_gaps_data = _safe_execute("gpu_idle_gaps", conn, **kwargs)
    # Overlap
    overlap_data = _safe_execute("overlap_breakdown", conn, **kwargs)
    # Kernel launch overhead
    launch_data = _safe_execute("kernel_launch_overhead", conn, **kwargs)

    # --- Pattern 1: GPU Bubbles (Pipeline Stalls) ---
    if idle_gaps_data:
        large_gaps = [g for g in idle_gaps_data
                      if g.get("gap_ms", 0) > 1.0]
        if len(large_gaps) >= 3:
            findings.append({
                "pattern": "GPU Bubbles (Pipeline Stalls)",
                "id": 1,
                "severity": "warning",
                "evidence": f"{len(large_gaps)} gaps > 1ms detected",
                "recommendation": (
                    "Use CUDA graphs, overlap data loading with compute, "
                    "or replace explicit cudaDeviceSynchronize with events."
                ),
            })

    # --- Pattern 3: NCCL Serialization ---
    if overlap_data and len(overlap_data) > 0:
        ov = overlap_data[0]
        if "error" not in ov:
            overlap_pct = ov.get("overlap_pct", 100)
            nccl_only = ov.get("nccl_only_ms", 0)
            total = ov.get("total_ms", 1)
            if nccl_only > 0 and overlap_pct < 30:
                findings.append({
                    "pattern": "NCCL Serialization",
                    "id": 3,
                    "severity": "critical",
                    "evidence": (
                        f"NCCL overlap only {overlap_pct}%, "
                        f"NCCL-only time: {nccl_only:.1f}ms / {total:.1f}ms"
                    ),
                    "recommendation": (
                        "Tune DDP bucket sizes (bucket_cap_mb), "
                        "ensure NCCL runs on separate stream, "
                        "consider gradient compression or FSDP."
                    ),
                })

    # --- Pattern 4: Excessive H2D Transfers ---
    mem_data = _safe_execute("memory_bandwidth", conn, **kwargs)
    if mem_data:
        h2d = [r for r in mem_data if r.get("copyKind") == 1]
        if h2d:
            h2d_ms = h2d[0].get("total_dur_ms", 0)
            if h2d_ms > 50:  # > 50ms of H2D is suspicious
                findings.append({
                    "pattern": "Excessive H2D Transfers",
                    "id": 4,
                    "severity": "warning",
                    "evidence": (
                        f"H2D transfers: {h2d_ms:.1f}ms total, "
                        f"{h2d[0].get('total_mb', 0):.1f}MB, "
                        f"{h2d[0].get('op_count', 0)} ops, "
                        f"avg bandwidth {h2d[0].get('avg_bandwidth_gbps', 0):.1f} GB/s"
                    ),
                    "recommendation": (
                        "Use pin_memory=True in DataLoader, keep "
                        "model params on GPU, accumulate metrics on GPU."
                    ),
                })

    # --- Pattern 5: Small Kernel Overhead ---
    if launch_data:
        # overhead_us > kernel_ms*1000 means overhead > kernel duration
        high_overhead = [e for e in launch_data
                         if e.get("kernel_ms", 0) > 0
                         and e.get("overhead_us", 0) > e["kernel_ms"] * 1000]
        if len(high_overhead) >= 5:
            findings.append({
                "pattern": "Small Kernel Overhead",
                "id": 5,
                "severity": "warning",
                "evidence": f"{len(high_overhead)} kernels with launch overhead > kernel duration",
                "recommendation": (
                    "Use torch.compile() to fuse element-wise ops, "
                    "enable cudnn.benchmark, or use CUDA graphs."
                ),
            })

    # --- Pattern 6: Kernel Hotspot ---
    if top_kernels_data and len(top_kernels_data) >= 2:
        # Compute percentage from total_ms since top_kernels doesn't have pct
        total_all_ms = sum(k.get("total_ms", 0) for k in top_kernels_data)
        if total_all_ms > 0:
            top_k = top_kernels_data[0]
            pct = (top_k.get("total_ms", 0) / total_all_ms) * 100
            if pct > 50:
                findings.append({
                    "pattern": "Kernel Hotspot",
                    "id": 6,
                    "severity": "info",
                    "evidence": (
                        f"'{top_k.get('kernel_name', '?')}' accounts for {pct:.0f}% "
                        f"of time in the profiled top kernels "
                        f"({top_k.get('total_ms', 0):.1f}ms)"
                    ),
                    "recommendation": (
                        "Ensure shapes are multiples of 128 (H100) / 64 (A100), "
                        "use FlashAttention, or profile with NCU for details."
                    ),
                })

    # --- Pattern 8: Compute-Communication Imbalance ---
    if overlap_data and len(overlap_data) > 0:
        ov = overlap_data[0]
        if "error" not in ov:
            compute_ms = ov.get("compute_only_ms", 0)
            nccl_ms_total = ov.get("nccl_only_ms", 0) + ov.get("overlap_ms", 0)
            if nccl_ms_total > 0 and compute_ms > 0:
                ratio = compute_ms / nccl_ms_total
                if ratio < 0.5:
                    findings.append({
                        "pattern": "Compute-Communication Imbalance",
                        "id": 8,
                        "severity": "critical",
                        "evidence": (
                            f"Compute/NCCL ratio = {ratio:.2f} (healthy > 2.0). "
                            f"Compute: {compute_ms:.1f}ms, NCCL: {nccl_ms_total:.1f}ms"
                        ),
                        "recommendation": (
                            "Reduce tensor parallel degree (e.g. TP=4 → TP=1 "
                            "if model fits on one GPU), rebalance pipeline stages, "
                            "or pad sequences to uniform length."
                        ),
                    })

    if not findings:
        findings.append({
            "pattern": "No Known Anti-Patterns Detected",
            "id": 0,
            "severity": "info",
            "evidence": "All checks passed — profile looks healthy",
            "recommendation": "Consider deep-diving with NCU for fine-grained analysis.",
        })

    return findings


def _safe_execute(skill_name, conn, **kwargs):
    """Execute a skill, returning [] on any error."""
    from ...skills.registry import get_skill
    try:
        skill = get_skill(skill_name)
        if skill is None:
            return []
        return skill.execute(conn, **kwargs)
    except Exception:
        return []


def _format(rows):
    if not rows:
        return "(No patterns checked)"
    lines = ["── Root Cause Pattern Analysis ──"]
    for f in rows:
        icon = {"critical": "🔴", "warning": "🟡", "info": "🟢"}.get(f["severity"], "⚪")
        lines.append(f"\n{icon} [{f['id']}] {f['pattern']}")
        lines.append(f"  Evidence: {f['evidence']}")
        lines.append(f"  Fix: {f['recommendation']}")
    return "\n".join(lines)


SKILL = Skill(
    name="root_cause_matcher",
    title="Root Cause Pattern Matcher",
    description=(
        "Automatically detects known GPU performance anti-patterns from the "
        "Book of Root Causes: GPU bubbles, NCCL serialization, kernel hotspots, "
        "small kernel overhead, compute-communication imbalance. "
        "Returns matched patterns with evidence and fix recommendations."
    ),
    category="analysis",
    execute_fn=_execute,
    format_fn=_format,
    params=[SkillParam("device", "GPU device ID", "int", False, 0)],
    tags=["root-cause", "pattern", "diagnosis", "analysis", "recommendation"],
)
