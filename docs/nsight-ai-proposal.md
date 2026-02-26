# Nsight AI: Agentic Kernel-to-Source Mapping via Iterative NVTX Annotation

## Problem

When analyzing GPU profiles, users want to know **which line of source code** produced a given GPU kernel. Today, the options are:

1. **CPU stack trace profiling** — Nsight can capture call stacks, but this adds significant overhead and slows profiling
2. **Manual NVTX annotation** — Developers manually add `nvtx.range()` around code regions, which is tedious and requires domain knowledge
3. **Guessing from kernel names** — e.g. `flash_fwd_kernel` probably comes from FlashAttention, but `vectorized_elementwise_kernel` could be anywhere

None of these are satisfactory for large codebases like Megatron-LM (100k+ LOC).

## Proposed Solution: Agentic NVTX Annotation

An AI agent that **iteratively instruments source code with NVTX annotations**, profiles, and refines until every kernel maps to exactly one source location.

### The Loop

```
┌─────────────────────────────────────────────────┐
│ 1. Agent reads source code + existing profile   │
│ 2. Agent inserts NVTX annotations at key points │
│ 3. Profile runs on GPU (Modal / local)          │
│ 4. Agent analyzes: which NVTX covers which      │
│    kernels?                                      │
│ 5. If an NVTX covers multiple kernels:          │
│    → Add finer-grained NVTX inside that region  │
│    → Go to step 3                               │
│ 6. If each innermost NVTX → 1 kernel:           │
│    → Done. We have kernel→source mapping.        │
└─────────────────────────────────────────────────┘
```

### Convergence Criterion

**Goal:** Each innermost NVTX annotation contains exactly one GPU kernel launch. At that point, the NVTX name tells us the exact source location (file, function, line range).

### Human-Guided Focus

Humans don't always want full-codebase mapping. They can specify:

- **Region of interest**: "I care about the attention mechanism"
- **Entry point**: "Start from `TransformerLayer.forward()`"
- **Kernel of interest**: "Where does `nvjet_tst_128x320` come from?"

The agent would then:
1. Only instrument the relevant code path
2. Skip irrelevant branches (data loading, logging, etc.)
3. Potentially extract and profile just that code path in isolation

### Agent Capabilities

The agent needs to:
1. **Parse Python/CUDA source** — understand function boundaries, class hierarchy
2. **Insert NVTX annotations** — wrap function calls and code blocks with `torch.cuda.nvtx.range()`
3. **Run profiling** — execute on GPU via Modal or local, with nsys
4. **Analyze results** — use the `nsight` library to build NVTX trees and count kernels per NVTX
5. **Decide refinement** — which NVTX regions need finer scoping

### Optimization: Branch Extraction

For focused profiling, the agent can:
- Extract a specific function and its dependencies
- Create a minimal script that runs just that function with synthetic inputs
- Profile only that script → faster iteration, fewer unrelated kernels

This is harder when functions need complex setup (attention needs Q/K/V tensors, model config, etc.), but often feasible with `torch.randn()` inputs.

## Example Scenario

**Target:** DISTCA codebase with Megatron-LM profiling on Modal H200s

Round 1 — Agent instruments `TransformerLayer.forward()`:
```python
with nvtx.range("TransformerLayer.forward"):
    # ... entire forward pass
```
Result: 50+ kernels under one NVTX. Too coarse.

Round 2 — Agent splits into sub-regions:
```python
with nvtx.range("TransformerLayer.self_attention"):
    hidden = self.self_attention(hidden)
with nvtx.range("TransformerLayer.mlp"):
    hidden = self.mlp(hidden)
with nvtx.range("TransformerLayer.layernorm"):
    hidden = self.layernorm(hidden)
```
Result: self_attention has 12 kernels, mlp has 8, layernorm has 2.

Round 3 — Agent drills into self_attention:
```python
with nvtx.range("self_attention.qkv_proj"):
    qkv = self.qkv_projection(hidden)
with nvtx.range("self_attention.flash_attn"):
    out = flash_attn(q, k, v)
with nvtx.range("self_attention.output_proj"):
    out = self.output_projection(out)
```
Result: flash_attn → 1 kernel (`flash_fwd_kernel`). ✅ Mapped.

## Test Codebases

| Codebase | Profile Source | Expected Difficulty |
|----------|---------------|-------------------|
| DISTCA (Megatron-LM) | Modal H200 cluster | Medium — well-structured, existing NVTX |
| FastVideo | Local / Modal | Hard — complex video pipeline |
| VLM | Modal | Medium — standard transformer |
| HGL | Modal | Unknown |

## Architecture

```
nsight-ai/
├── annotator.py     # Insert NVTX annotations into source code
├── profiler.py      # Run profiling on Modal / local
├── analyzer.py      # Analyze NVTX tree, find multi-kernel NVTX
├── planner.py       # Decide where to add finer NVTX
├── mapper.py        # Final kernel → source location mapping
└── cli.py           # Orchestration loop
```

## Open Questions

1. **How to handle dynamic dispatch?** — Some kernels only fire for certain inputs
2. **How to handle Triton kernels?** — Names like `triton_poi_fused_add_0` don't map to source easily
3. **How deep to go?** — Should the agent drill into PyTorch internals or stop at user code?
4. **Multi-rank alignment** — When using `torchrun`, different ranks start at different times. How to align clocks across nodes?
5. **Overhead budget** — How many NVTX annotations before profiling overhead becomes significant?

## Related Work

- NVIDIA's `torch.cuda.nvtx` integration
- PyTorch Profiler's `record_function()` context manager
- Nsight Systems' built-in `nvtx_gpu_proj_trace` report
- Megatron-LM's existing NVTX annotations in `core/transformer/`
