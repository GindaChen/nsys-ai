"""
nsys_tui.agent — The nsys-ai agent: a CUDA ML systems performance expert.

This package provides:
    persona.py  — Agent identity, system prompt, knowledge layers
    loop.py     — Core analysis loop: profile → skill selection → execution → report
"""
from .loop import Agent
from .persona import AGENT_IDENTITY, SYSTEM_PROMPT

__all__ = ["SYSTEM_PROMPT", "AGENT_IDENTITY", "Agent"]
