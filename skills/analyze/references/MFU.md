# Skill: MFU Analysis

Read this when the user asks about GPU efficiency, MFU, or TFLOPS utilization.
**Read `PRINCIPLES.md` first** for rules, error handling, and tool definitions.

---

## Common Models Reference

| Model | H (hidden) | L (layers) | ffn | Notes |
|-------|-----------|-----------|-----|-------|
| LLaMA-7B | 4096 | 32 | 11008 | SwiGLU, ffn≈2.69×H |
| LLaMA-13B | 5120 | 40 | 13824 | |
| LLaMA-70B / 65B | 8192 | 80 | 28672 | |
| LLaMA-3.1-8B | 4096 | 32 | 14336 | |
| Mistral-7B | 4096 | 32 | 14336 | GQA |
| GPT-3 175B | 12288 | 96 | 49152 | Dense transformer |
| Falcon-7B | 4544 | 32 | 18176 | MQA |
| Falcon-40B | 8192 | 60 | 32768 | |

---

## Operation → FLOPs Mapping

| Target kernel / region | `operation` arg | FLOPs (per layer) |
|------------------------|-----------------|-------------------|
| `flash_fwd`, `flash_bwd` | `"attention"` | 4·S²·H |
| QKV linear projection | `"qkv_proj"` | 6·S·H² |
| Output projection | `"output_proj"` | 2·S·H² |
| FFN / MLP (up+gate+down) | `"mlp"` | 4·S·H·ffn |
| One full transformer layer | `"full_layer"` | sum of above |
| NVTX wrapping fwd pass | `"full_model"`, multiplier=1 | per_layer × L |
| NVTX wrapping fwd+bwd | `"full_model"`, multiplier=3 | (no checkpointing) |
| NVTX fwd+bwd+recompute | `"full_model"`, multiplier=4 | (gradient ckpt) |
| Generic GEMM | `"linear"` | 2·M·N·K |

**CRITICAL**: If the target is a SINGLE kernel type (e.g. `flash_fwd`), use the narrow
per-kernel operation. Using `full_model` FLOPs for one kernel ALWAYS gives MFU > 100%.

---

## Workflow 1: MFU for a Specific NVTX Region or Kernel

```
Step 1  Use `get_gpu_peak_tflops()`
        If error → ask user for GPU model and consult hardware.py

Step 2  Discover the target name (mandatory — never guess):
        [Check NVTX first]
        SELECT DISTINCT COALESCE(n.text, s.value) AS name
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
        WHERE COALESCE(n.text, s.value) IS NOT NULL LIMIT 30

        [If no NVTX or targeting a kernel directly]
        SELECT DISTINCT s.value FROM StringIds s
        JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.shortName=s.id
        WHERE s.value LIKE '%<keyword>%' LIMIT 20

Step 3  Resolve model architecture — in this order, stop when resolved:
        A. User named the model → use the table above
        B. Infer H from kernel timing: flash_fwd avg_ms ≈ 4·S²·H / (peak_tflops·1e12)
        C. Ask: "What model are you training? (e.g. LLaMA-7B = H=4096, L=32)"
           → End message. Wait.

Step 4  Use `compute_theoretical_flops`:
            operation=<see table above>, hidden_dim=H, seq_len=S, num_layers=L
        → {theoretical_flops: <value>}

Step 5  Use `compute_region_mfu`:
            name="<from step 2>", theoretical_flops=<from step 4>,
            source="nvtx" (or "kernel"), peak_tflops=<from step 1>, num_gpus=<world_size>
        → {mfu_pct_wall, mfu_pct_kernel_union, wall_time_s, kernel_count, ...}

Step 6  Sanity check — MANDATORY before reporting:
        • mfu_pct_wall > 100% → operation scope too wide; use narrower operation from table
        • mfu_pct_wall < 1%   → name match failed (check kernel_count) or FLOPs too low
        • 40–80%              → healthy compute-bound
        • < 30%               → possible memory bandwidth bottleneck

Step 7  Interpret in context:
        • flash_fwd: typical vendor benchmark is 60–70% for standard LLM configs
        • Full training step: PaLM/Chinchilla ~46%, LLaMA ~38–45%
        Always state whether MFU is "forward-only" or "forward+backward" to avoid confusion.
```

---

## Workflow 2: Step-Level MFU (Whole Training Throughput)

```
Step 1  Use `get_gpu_peak_tflops()`

Step 2  Get a SINGLE representative step time (not the whole profile span):
        Option A — NVTX marker (preferred):
          SELECT COALESCE(n.text, s.value) AS name, ([end]-start)/1e6 AS ms
          FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
          WHERE name LIKE '%sample%' OR name LIKE '%step%' OR name LIKE '%iter%'
          ORDER BY start LIMIT 10
          → Skip index 0 (JIT); use median duration across iterations

        Option B — No NVTX (rough upper bound):
          SELECT (MAX([end]) - MIN(start)) / 1e9 AS total_span_s FROM CUPTI_ACTIVITY_KIND_KERNEL
          ⚠ This includes warmup + all iterations combined. Tell user it will understate MFU.

Step 3  Ask for model architecture:
        "I need: model name (or hidden_dim, num_layers, seq_len), batch_size per GPU,
         and whether this is forward-only, forward+backward, or with gradient checkpointing?"
        → End message. Wait. Use model table above if name is known.

Step 4  Use `compute_theoretical_flops`:
            operation="full_model", hidden_dim=H, seq_len=S, num_layers=L,
            batch_size=B, multiplier=<1/3/4>

Step 5  Use `compute_mfu`:
            step_time_s=<step 2>, model_flops_per_step=<step 4 value>, peak_tflops=<step 1>
        → {MFU_pct, achieved_model_TFLOPS}

Step 6  Contextualise:
        < 20%  → significant problem; check NCCL, batch size, mixed precision
        30–45% → typical well-tuned single-node training
        45–60% → good; Flash Attention likely in use
        > 60%  → excellent; publication-grade efficiency
```

---

## Acceptance Criteria

Before delivering MFU results, verify:

- [ ] MFU between 0% and 100% (if > 100%, recompute with narrower operation)
- [ ] `theoretical_flops` came from a theoretical FLOPs calculator, not estimated by hand
- [ ] NVTX or kernel name came from a query (not guessed)
- [ ] Step time is from a SINGLE representative iteration (not full profile kernel span)
- [ ] `operation` matches the exact scope of the measured region/kernel
- [ ] Result states forward-only or forward+backward to avoid ambiguity

Also run global checklist in `PRINCIPLES.md`.

---

## Error Handling

| Signal | Action |
|--------|--------|
| `mfu_pct_wall > 100%` | Recompute with narrower `operation`; explain to user |
| `kernel_count = 0` | Report KERNEL_NOT_FOUND; try `source="kernel"` or fix name substring |
| GPU unknown | Ask user for `peak_tflops` (BF16/FP16, dense, no sparsity) |
| Model name not in table | Ask user for `hidden_dim`, `num_layers`, `seq_len` |
