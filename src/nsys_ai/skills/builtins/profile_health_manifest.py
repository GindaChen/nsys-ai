"""Profile Health Manifest — one-shot summary for external AI agents.

Returns a compact JSON summary (~500 bytes) that captures the essential
profile characteristics in a single tool call, eliminating the need for
5-8 sequential skill invocations during agent exploration.

Internally orchestrates: top_kernels, overlap_breakdown, nccl_breakdown,
gpu_idle_gaps, and root_cause_matcher.
"""

import logging

from ..base import Skill, SkillParam

_log = logging.getLogger(__name__)


def _safe_skill_run(skill_name: str, conn, **kwargs):
    """Run a skill by name, returning [] on any error."""
    from ..registry import get_skill

    skill = get_skill(skill_name)
    if skill is None:
        return []
    try:
        return skill.execute(conn, **kwargs)
    except Exception as exc:
        _log.debug("manifest: %s failed: %s", skill_name, exc, exc_info=True)
        return []


def _execute(conn, **kwargs):
    """Build a compact profile health manifest."""
    from ...profile import Profile

    device = int(kwargs.get("device", 0))

    # Forward trim kwargs if present
    trim_kwargs = {}
    for k in ("trim_start_ns", "trim_end_ns"):
        if kwargs.get(k) is not None:
            trim_kwargs[k] = kwargs[k]

    # ── 1. Profile metadata ──────────────────────────────────────
    prof = Profile._from_conn(conn)
    from ...profile import get_first_gpu_name
    # NOTE: get_first_gpu_name returns the GPU name for cudaId=0.
    # On multi-GPU nodes with identical GPUs this is fine; for
    # heterogeneous setups the caller should interpret accordingly.
    gpu_name = get_first_gpu_name(conn) or "unknown"
    start_ns, end_ns = prof.meta.time_range
    # Use the trim window (if provided), clamped to the profile range,
    # so the reported span matches the analysis window used by sub-skills.
    effective_start_ns = start_ns
    effective_end_ns = end_ns
    if trim_kwargs.get("trim_start_ns") is not None:
        effective_start_ns = max(effective_start_ns, trim_kwargs["trim_start_ns"])
    if trim_kwargs.get("trim_end_ns") is not None:
        effective_end_ns = min(effective_end_ns, trim_kwargs["trim_end_ns"])
    profile_span_ns = (
        effective_end_ns - effective_start_ns
        if effective_end_ns > effective_start_ns
        else 0
    )
    profile_span_ms = round(profile_span_ns / 1e6, 1) if profile_span_ns > 0 else 0

    # ── 2. Top kernels (compact: top 5 only) ─────────────────────
    top_k_rows = _safe_skill_run(
        "top_kernels", conn, limit=5, device=device, **trim_kwargs
    )
    top_kernels = []
    for r in top_k_rows[:5]:
        name = r.get("kernel_name", "?")
        if len(name) > 60:
            name = name[:57] + "..."
        top_kernels.append({
            "name": name,
            "total_ms": r.get("total_ms", 0),
            "count": r.get("invocations", 0),
        })

    # Compute total kernel time over all kernels, not just the top 5.
    all_kernel_rows = _safe_skill_run(
        "top_kernels", conn, device=device, **trim_kwargs
    )
    total_kernel_ms = sum(r.get("total_ms", 0) for r in all_kernel_rows)

    # ── 3. Compute/NCCL overlap ──────────────────────────────────
    overlap_rows = _safe_skill_run(
        "overlap_breakdown", conn, device=device, **trim_kwargs
    )
    overlap = {}
    if overlap_rows:
        ov = overlap_rows[0]
        if "error" in ov:
            # Preserve error details from overlap_breakdown instead of dropping them.
            # At minimum, surface the primary error message; keep any additional
            # fields that may provide context for callers.
            overlap = dict(ov)
        else:
            overlap = {
                "compute_only_ms": ov.get("compute_only_ms", 0),
                "nccl_only_ms": ov.get("nccl_only_ms", 0),
                "overlap_pct": ov.get("overlap_pct", 0),
                "idle_ms": ov.get("idle_ms", 0),
            }

    # ── 4. NCCL breakdown (compact summary) ──────────────────────
    nccl_rows = _safe_skill_run(
        "nccl_breakdown", conn, device=device, **trim_kwargs
    )
    nccl_summary = {"streams": 0, "collectives": 0}
    if nccl_rows:
        stream_ids = {r.get("stream_id") for r in nccl_rows}
        nccl_summary["streams"] = len(stream_ids)
        nccl_summary["collectives"] = len(nccl_rows)
        # Dominant collective by total_ms
        dominant = max(nccl_rows, key=lambda r: r.get("total_ms", 0))
        nccl_summary["dominant_type"] = dominant.get("type", "?")
        nccl_summary["dominant_pct"] = dominant.get("pct", 0)
        nccl_summary["total_nccl_ms"] = round(
            sum(r.get("total_ms", 0) for r in nccl_rows), 1
        )

    # ── 5. GPU idle gaps (summary only) ──────────────────────────
    idle_rows = _safe_skill_run(
        "gpu_idle_gaps", conn, device=device, limit=1, **trim_kwargs
    )
    idle_summary = {"gap_count": 0, "idle_pct": 0}
    summary_row = next((r for r in idle_rows if r.get("_summary")), None)
    if summary_row:
        idle_summary["gap_count"] = summary_row.get("gap_count", 0)
        idle_summary["idle_pct"] = summary_row.get("pct_of_profile", 0)
        idle_summary["total_idle_ms"] = summary_row.get("total_idle_ms", 0)

    # ── 6. Root cause findings (count + top severity) ────────────
    rc_rows = _safe_skill_run("root_cause_matcher", conn, device=device, **trim_kwargs)
    root_causes = []
    for r in rc_rows:
        pattern = r.get("pattern", "")
        if pattern == "No Known Anti-Patterns Detected":
            continue
        root_causes.append({
            "pattern": pattern,
            "severity": r.get("severity", "info"),
        })

    # ── Assemble manifest ────────────────────────────────────────
    manifest = {
        "gpu": gpu_name,
        "profile_span_ms": profile_span_ms,
        "top_kernels": top_kernels,
        "total_kernel_ms": round(total_kernel_ms, 1),
        "overlap": overlap,
        "nccl": nccl_summary,
        "idle": idle_summary,
        "root_cause_count": len(root_causes),
        "root_causes": root_causes[:5],  # Cap at 5 to keep output compact
    }

    # Infer suspected bottleneck
    bottleneck = _infer_bottleneck(manifest)
    if bottleneck:
        manifest["suspected_bottleneck"] = bottleneck

    return [manifest]


def _infer_bottleneck(m: dict) -> str:
    """Heuristic bottleneck inference from manifest data."""
    overlap = m.get("overlap", {})
    idle = m.get("idle", {})
    nccl = m.get("nccl", {})

    # Check NCCL serialization first (most impactful)
    if overlap.get("overlap_pct", 100) < 30 and overlap.get("nccl_only_ms", 0) > 0:
        return "NCCL serialization (overlap < 30%)"

    # Check idle gaps
    if idle.get("idle_pct", 0) > 15:
        return f"GPU idle bubbles ({idle['idle_pct']}% of profile)"

    # Check if one kernel dominates
    top_k = m.get("top_kernels", [])
    total = m.get("total_kernel_ms", 0)
    if top_k and total > 0:
        top_pct = top_k[0]["total_ms"] / total * 100
        if top_pct > 60:
            return f"Kernel hotspot: {top_k[0]['name']} ({top_pct:.0f}%)"

    # Check NCCL dominance
    if nccl.get("total_nccl_ms", 0) > overlap.get("compute_only_ms", float("inf")):
        return "Communication-bound (NCCL > compute)"

    return ""


def _format(rows):
    if not rows:
        return "(No manifest data)"
    m = rows[0]
    lines = ["══ Profile Health Manifest ══"]
    lines.append(f"  GPU:          {m.get('gpu', '?')}")
    lines.append(f"  Profile span: {m.get('profile_span_ms', 0):.1f}ms")

    # Top kernels
    lines.append("")
    lines.append("  Top Kernels:")
    for k in m.get("top_kernels", []):
        lines.append(f"    {k['name'][:50]:<52s}  {k['total_ms']:>8.1f}ms  ×{k['count']}")

    # Overlap
    ov = m.get("overlap", {})
    if ov:
        lines.append("")
        lines.append("  Compute/NCCL Overlap:")
        lines.append(f"    Compute only: {ov.get('compute_only_ms', 0):.1f}ms")
        lines.append(f"    NCCL only:    {ov.get('nccl_only_ms', 0):.1f}ms")
        lines.append(f"    Overlap:      {ov.get('overlap_pct', 0)}%")

    # NCCL
    nccl = m.get("nccl", {})
    if nccl.get("streams", 0) > 0:
        lines.append("")
        lines.append("  NCCL Summary:")
        lines.append(f"    Streams: {nccl['streams']}, Collectives: {nccl['collectives']}")
        lines.append(f"    Dominant: {nccl.get('dominant_type', '?')} ({nccl.get('dominant_pct', 0)}%)")
        lines.append(f"    Total: {nccl.get('total_nccl_ms', 0):.1f}ms")

    # Idle
    idle = m.get("idle", {})
    if idle.get("gap_count", 0) > 0:
        lines.append("")
        lines.append(f"  GPU Idle: {idle['gap_count']} gaps, {idle.get('idle_pct', 0)}% of profile")

    # Root causes
    rcs = m.get("root_causes", [])
    if rcs:
        lines.append("")
        lines.append(f"  Root Causes ({m.get('root_cause_count', 0)} findings):")
        for rc in rcs:
            sev_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(rc["severity"], "⚪")
            lines.append(f"    {sev_icon} {rc['pattern']}")

    # Bottleneck
    bn = m.get("suspected_bottleneck", "")
    if bn:
        lines.append("")
        lines.append(f"  ⚡ Suspected bottleneck: {bn}")

    return "\n".join(lines)


SKILL = Skill(
    name="profile_health_manifest",
    title="Profile Health Manifest",
    description=(
        "One-shot profile health summary for AI agents. Returns a compact JSON manifest "
        "covering GPU info, top kernels, compute/NCCL overlap and NCCL summary, "
        "idle gaps, and root cause findings — all in a single call. "
        "Use this as the FIRST skill to call on any new profile."
    ),
    category="utility",
    execute_fn=_execute,
    format_fn=_format,
    params=[
        SkillParam("device", "GPU device ID", "int", False, 0),
    ],
    tags=["manifest", "summary", "health", "overview", "agent", "triage"],
)
