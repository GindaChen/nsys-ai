"""
nsys_ai.agent — The nsys-ai agent: a CUDA ML systems performance expert.

This package provides:
    persona.py  — Agent identity, system prompt, knowledge layers
    loop.py     — Compatibility wrapper for the public Agent API
    runner.py   — Shared profile evidence, selection, and synthesis runner
"""

from .loop import Agent
from .persona import AGENT_IDENTITY, SYSTEM_PROMPT
from .runner import (
    format_evidence_first_answer,
    run_diagnose_pack,
    select_skills_for_question,
    synthesize_evidence,
)

__all__ = [
    "SYSTEM_PROMPT",
    "AGENT_IDENTITY",
    "Agent",
    "format_evidence_first_answer",
    "run_diagnose_pack",
    "select_skills_for_question",
    "synthesize_evidence",
]
