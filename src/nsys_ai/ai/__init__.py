"""
nsys_ai.ai — LLM-adjacent helpers used by the agent and the diff narrative.

Modules:
    backend/           — read-only profile database tooling the agent queries
    diff_narrative.py  — LLM narrative over a computed diff

This package once also held a self-annotation pipeline — an env-gated NVTX
context manager, a source rewriter that inserted it, and an NVTX-tree
convergence analyzer. Nothing ever called them and no surface exposed them;
they were removed rather than left reading as a shipped feature. Annotating a
PyTorch workload that has none is now a capture-time option, ``nsys-ai profile
--pytorch``, which uses Nsight's own annotation instead of editing source.
"""
