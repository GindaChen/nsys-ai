"""Per-iteration timing analysis.

Uses detect_iterations() from overlap.py to find repeating training iterations
via a top-level NVTX marker and report per-iteration GPU timing and kernel counts.
This is a Python-level skill (execute_fn).
"""

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    from ...overlap import detect_iterations
    from ...profile import Profile

    prof = Profile._from_conn(conn)
    device = int(kwargs.get("device", 0))
    marker = kwargs.get("marker", "sample_0")
    # Support --trim passthrough from agent analyze
    trim = None
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    if trim_start is not None and trim_end is not None:
        trim = (int(trim_start), int(trim_end))
    iters = detect_iterations(prof, device, trim=trim, marker=marker)
    for it in iters:
        it["device_id"] = device
    return iters


def _to_findings(rows: list[dict]) -> list:
    import statistics

    from nsys_ai.annotation import Finding

    findings = []
    # Only judge variance over genuine iterations: real (not the sub-ms op ranges
    # a loose NVTX marker matches, whose contaminated median makes every real
    # iteration look thousands of percent slow) AND NVTX-marker-based (not the
    # heuristic gap fallback, which on a non-looping capture emits pauses that are
    # not iterations, so a "slow iteration" finding over them is phantom — #215).
    real = [
        it
        for it in rows
        if it.get("is_real_iteration", True) and not it.get("heuristic", False)
    ]
    if len(real) < 3:
        return findings

    durs = [it["duration_ms"] for it in real if "duration_ms" in it]
    if not durs:
        return findings

    med = statistics.median(durs)
    if med <= 0:
        return findings

    slow = [it for it in real if it.get("duration_ms", 0) > 1.5 * med]

    for it in slow:
        pct = 100 * it["duration_ms"] / med
        # Prefer nanosecond-precision timestamps if available; fall back to seconds-based values.
        start_ns = it.get("gpu_start_ns")
        if start_ns is None:
            start_ns = int(it.get("gpu_start_s", 0) * 1e9)
        else:
            start_ns = int(start_ns)
        end_ns = it.get("gpu_end_ns")
        if end_ns is None:
            end_ns = int(it.get("gpu_end_s", 0) * 1e9)
        else:
            end_ns = int(end_ns)

        note = (
            f"{it['duration_ms']:.1f}ms "
            f"({pct:.0f}% of median {med:.1f}ms), "
            f"{it.get('kernel_count', 0)} kernels"
        )

        findings.append(
            Finding(
                type="region",
                label=f"Slow Iteration {it.get('iteration', '?')}",
                start_ns=start_ns,
                end_ns=end_ns,
                gpu_id=it.get(
                    "device_id", 0
                ),  # Assuming device_id is available or defaults to 0
                severity="warning",
                # No headroom, by design. A slow iteration's excess over the
                # median is wall-clock time whose idle and exposed-comm parts are
                # already claimed in full by gpu_idle_gaps (device idle) and
                # overlap_breakdown (exposed NCCL); any remainder is extra
                # compute the iteration genuinely did, not recoverable. Claiming
                # the slack here would double-count that time — the same
                # deferral critical_path and nccl_breakdown make. The finding
                # stays as the variance locator: which iteration to investigate.
                headroom_ms=None,
                headroom_basis=None,
                note=note,
            )
        )
    return findings


def _format(rows):
    if not rows:
        return "(No iterations detected — NVTX marker not found and no large gaps found)"
    is_heuristic = any(it.get("heuristic", False) for it in rows)
    title = (
        "── Iteration Timings (Heuristic Fallback) ──"
        if is_heuristic
        else "── Iteration Timings ──"
    )
    # Show real iterations only — a loose NVTX marker can match hundreds of
    # sub-iteration op ranges, and listing all of them buries the signal.
    real = [it for it in rows if it.get("is_real_iteration", True)]
    display = real if real else rows
    noise_count = len(rows) - len(real)

    lines = [title]
    for it in display:
        lines.append(
            f"  iter {it['iteration']:2d}  "
            f"{it['duration_ms']:8.1f}ms  "
            f"({it['kernel_count']} kernels, {it['nccl_count']} NCCL)  "
            f"compute={it['compute_ms']:.1f}ms"
        )
    if noise_count:
        lines.append(
            f"  … {noise_count} sub-iteration NVTX range(s) filtered as noise "
            f"(marker matched op-level annotations)"
        )
    if len(display) > 1:
        durs = [it["duration_ms"] for it in display]
        avg = sum(durs) / len(durs)
        lines.append(f"\n  Average: {avg:.1f}ms  Min: {min(durs):.1f}ms  Max: {max(durs):.1f}ms")
    return "\n".join(lines)


SKILL = Skill(
    name="iteration_timing",
    title="Per-Iteration Timing Analysis",
    description=(
        "Detects repeating training iterations using a top-level NVTX marker "
        "and reports per-iteration GPU timing, kernel counts, and NCCL counts. "
        "Use to identify slow iterations and iteration variance."
    ),
    category="nvtx",
    execute_fn=_execute,
    format_fn=_format,
    to_findings_fn=_to_findings,
    params=[
        SkillParam("device", "GPU device ID", "int", False, 0),
        SkillParam("marker", "NVTX text pattern for iteration boundary", "str", False, "sample_0"),
    ],
    tags=["iteration", "timing", "variance", "nvtx", "training"],
)
