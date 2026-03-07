"""
diff_narrative.py — Executive summary and optional AI narrative for profile diff reports.

Produces a deterministic one-line summary and, when an LLM is configured, a short
narrative explaining what changed. All numeric claims are derived from the
computed ProfileDiffSummary; the LLM is instructed not to invent causality.
"""

from __future__ import annotations

from dataclasses import dataclass

from nsys_ai.diff import ProfileDiffSummary


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
    s = _fmt_ns(ns)
    if ns > 0:
        return f"+{s}"
    return s


@dataclass(frozen=True)
class DiffNarrative:
    """Augmentation for diff reports: deterministic summary + optional AI narrative."""

    executive_summary: str
    ai_narrative: str | None
    model: str | None
    warning: str | None


def build_executive_summary(summary: ProfileDiffSummary) -> str:
    """
    Generate a 1–2 sentence deterministic summary from the diff numbers.

    Does not call any LLM; safe to use when --no-ai or when no API key is set.
    """
    delta_ns = summary.after.total_gpu_ns - summary.before.total_gpu_ns
    direction = "faster" if delta_ns < 0 else "slower"
    total_str = f"Total GPU time went {direction} by {_fmt_delta_ns(delta_ns)}."

    parts = [total_str]

    if summary.top_regressions:
        top = summary.top_regressions[0]
        parts.append(
            f"Largest regression: {top.name} ({_fmt_delta_ns(top.delta_ns)})."
        )
    if summary.top_improvements:
        top = summary.top_improvements[0]
        parts.append(
            f"Largest improvement: {top.name} ({_fmt_delta_ns(top.delta_ns)})."
        )

    return " ".join(parts)


def _build_narrative_prompt_payload(summary: ProfileDiffSummary) -> str:
    """Build a compact, numeric-only payload for the LLM (no raw kernel lists)."""
    lines: list[str] = []
    delta_ns = summary.after.total_gpu_ns - summary.before.total_gpu_ns
    lines.append(
        f"Total GPU time: before {_fmt_ns(summary.before.total_gpu_ns)}, "
        f"after {_fmt_ns(summary.after.total_gpu_ns)}, delta {_fmt_delta_ns(delta_ns)}."
    )
    if summary.overlap_delta and "overlap_pct" in summary.overlap_before:
        b = summary.overlap_before.get("overlap_pct")
        a = summary.overlap_after.get("overlap_pct")
        if b is not None and a is not None:
            d = summary.overlap_delta.get("overlap_pct", 0)
            sign = "+" if d > 0 else ""
            lines.append(f"Overlap %: before {b:.1f}%, after {a:.1f}%, delta {sign}{d}%.")
    lines.append("Top regressions (name, delta time):")
    for k in summary.top_regressions[:5]:
        lines.append(f"  - {k.name}: {_fmt_delta_ns(k.delta_ns)}")
    lines.append("Top improvements (name, delta time):")
    for k in summary.top_improvements[:5]:
        lines.append(f"  - {k.name}: {_fmt_delta_ns(k.delta_ns)}")
    if summary.warnings:
        lines.append("Warnings: " + "; ".join(summary.warnings[:3]))
    return "\n".join(lines)


def generate_diff_narrative(
    summary: ProfileDiffSummary,
    *,
    preferred_model: str | None = None,
) -> DiffNarrative:
    """
    Build executive summary and, when an LLM is available, an AI narrative.

    On any LLM failure or when --no-ai is used, ai_narrative is None and
    warning may be set; the report still includes the deterministic summary.
    """
    executive_summary = build_executive_summary(summary)

    from nsys_ai.chat_config import _get_model_and_key

    model, _ = _get_model_and_key(preferred_model)
    if not model:
        return DiffNarrative(
            executive_summary=executive_summary,
            ai_narrative=None,
            model=None,
            warning="No LLM configured (set API key or use --no-ai to suppress).",
        )

    payload = _build_narrative_prompt_payload(summary)
    system = (
        "You are a GPU performance analyst. Summarize the profile comparison below in 2–4 short sentences. "
        "Use only the numbers provided; do not invent causes or speculate beyond the data. "
        "Mention the main regressions, improvements, and overlap change if relevant."
    )
    user = f"Profile diff (before vs after):\n\n{payload}"

    try:
        import litellm

        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as e:
        return DiffNarrative(
            executive_summary=executive_summary,
            ai_narrative=None,
            model=model,
            warning=f"AI narrative unavailable: {e!s}",
        )

    choice = response.choices[0] if response.choices else None
    if not choice:
        return DiffNarrative(
            executive_summary=executive_summary,
            ai_narrative=None,
            model=model,
            warning="AI returned no content.",
        )
    message = choice.message
    content = (
        (message.get("content") if isinstance(message, dict) else getattr(message, "content", None))
        or ""
    )
    content = content.strip()
    return DiffNarrative(
        executive_summary=executive_summary,
        ai_narrative=content if content else None,
        model=model,
        warning=None,
    )
