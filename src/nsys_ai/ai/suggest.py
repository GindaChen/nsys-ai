"""suggest.py — AI-powered NVTX annotation suggestions.

Analyzes GPU kernel patterns (gaps, bursts, lack of NVTX coverage) to
suggest where adding Nsight ranges would improve profile clarity.
"""

from __future__ import annotations
import logging

_log = logging.getLogger(__name__)

def analyze_patterns(prof, gpu: int, trim: tuple[int, int] | None = None) -> list[dict]:
    """Analyze profile for patterns that benefit from NVTX."""
    from nsys_ai.skills.builtins.gpu_idle_gaps import _execute as run_idle_gaps
    from nsys_ai.skills.builtins.kernel_launch_pattern import SKILL as launch_pattern_skill
    from nsys_ai.skills.registry import run_skill
    
    findings = []
    
    # 1. Find large GPU idle gaps (bubbles)
    # These often indicate CPU-side bottlenecks that lack NVTX coverage
    # to explain what the CPU was doing.
    gaps = run_idle_gaps(prof.connection, device=gpu, trim_start_ns=trim[0] if trim else None, trim_end_ns=trim[1] if trim else None, min_gap_ns=500_000)
    for gap in gaps:
        if gap.get("_summary"): continue
        if gap.get("gap_ns", 0) > 1_000_000: # > 1ms
            findings.append({
                "type": "gap",
                "duration_ms": gap["gap_ns"] / 1e6,
                "before": gap.get("before_kernel"),
                "after": gap.get("after_kernel"),
                "stream": gap.get("streamId")
            })

    # 2. Find high-frequency small kernel bursts (trickle launch)
    # These often benefit from being wrapped in a single NVTX range.
    # We can't easily use the SQL skill directly for "burst" detection without more logic,
    # but we can look for high dispatch rates.
    try:
        launch_stats = launch_pattern_skill.execute(prof.connection, trim_start_ns=trim[0] if trim else None, trim_end_ns=trim[1] if trim else None)
        for s in launch_stats:
            if s.get("dispatch_rate_per_ms", 0) > 10.0: # > 10 kernels/ms
                findings.append({
                    "type": "burst",
                    "rate": s["dispatch_rate_per_ms"],
                    "stream": s["streamId"],
                    "count": s["kernel_count"]
                })
    except Exception as e:
        _log.debug(f"Failed to run launch_pattern: {e}")

    return findings

def generate_suggestions(findings: list[dict]) -> list[dict]:
    """Convert findings into actionable NVTX suggestions."""
    suggestions = []
    seen_funcs = set()

    for f in findings:
        if f["type"] == "gap":
            # Heuristic: if we have a gap after 'after_kernel', suggest wrapping the caller
            # This is hard without source mapping, so we'll suggest a generic label
            # based on the kernel name parts.
            kernel = f["after"]
            if not kernel: continue
            
            # Simple heuristic: take the first part of the kernel name
            func_guess = kernel.split("<")[0].split("(")[0].strip()
            if func_guess and func_guess not in seen_funcs:
                suggestions.append({
                    "reason": f"Significant GPU idle gap ({f['duration_ms']:.2f}ms) before this kernel burst.",
                    "label": f"Region.{func_guess}",
                    "func": func_guess,
                    "type": "wrap_call"
                })
                seen_funcs.add(func_guess)
        
        elif f["type"] == "burst":
            # High rate suggests many small calls that should be grouped
            suggestions.append({
                "reason": f"High kernel dispatch rate ({f['rate']:.1f} kernels/ms) detected on stream {f['stream']}.",
                "label": "HighFreqLoop",
                "type": "generic"
            })

    return suggestions

def format_suggestions(suggestions: list[dict]) -> str:
    """Format suggestions for terminal output."""
    if not suggestions:
        return "No suggestions."
        
    lines = ["\n── NVTX Annotation Suggestions ──", ""]
    for i, s in enumerate(suggestions, 1):
        lines.append(f"{i}. {s['reason']}")
        if s.get("label"):
            lines.append(f"   Suggested Label: {s['label']}")
        if s.get("func"):
            lines.append(f"   Code Example:")
            lines.append(f"     with nsight_range(\"{s['label']}\"):")
            lines.append(f"         {s['func']}(...)")
        lines.append("")
    
    lines.append("💡 TIP: Add these ranges to your Python code to see them in the timeline.")
    lines.append("   Use --insert to attempt auto-annotation (requires exact function name match).")
    return "\n".join(lines)
