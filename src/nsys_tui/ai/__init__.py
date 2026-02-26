"""
nsight.ai — Agentic kernel-to-source mapping via iterative NVTX annotation.

Modules:
    gate.py       — Env-var gated NVTX (zero-overhead when NSIGHT_AI is not set)
    annotator.py  — Insert NVTX annotations into Python source files
    analyzer.py   — Find multi-kernel NVTX regions, measure convergence
"""
