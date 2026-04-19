# Mode 3 — Compute (kernels / tensor core / MFU)

Reference for `/nsys-ai` Mode 3. **Read `PRINCIPLES.md` first** — §4 guards, §5 evidence,
§7 fail template, §10 checklist.

---

## 1. Precondition gate

§4.1 row 3 is the global guard (no kernel table → abort before any mode). No additional gate
for Mode 3. Soft note if `profile_health_manifest` `nccl.collectives > 0` AND
`overlap.overlap_pct < 30`: "NCCL overlap is critical — Mode 2 may address more time than
compute tuning. Proceed with Mode 3 or switch? Reply `mode2` or press on."

MFU sub-focus additionally needs an NVTX region name. If `nvtx_layer_breakdown` returns empty,
print: "MFU sub-focus needs an NVTX region name; falling back to `arithmetic_intensity` for a
roofline estimate (no per-region breakdown)."

---

## 2. Stages

| # | Question | Condition/Default |
|---|----------|-------------------|
| 1 | Profile path | Only if not supplied |
| 2 | Sub-focus: `top-n` / `tensor-core` / `mfu` / `specific-kernel` | Optional; default `top-n` |
| 3 | Model arch (llama-7b / llama-13b / llama-70b / mistral-7b / gpt3-175b) OR raw `hidden_dim seq_len` | **Only if sub-focus = `mfu`** AND arch is ambiguous from kernel names |

Skip Stage 3 if arch is already clear from top kernel names (e.g. `flash_fwd_hdim128` → H=128
per head; `sm90_xmma_gemm_*_4096x` → H=4096). Ask only when kernel names give no signal.

---

## 3. Skills

| Sub-focus | Skills (in order) |
|-----------|-------------------|
| `top-n` (default) | `top_kernels -p limit=20` → `kernel_launch_pattern` |
| `tensor-core` | `tensor_core_usage` → `top_kernels -p limit=10` |
| `mfu` | `theoretical_flops -p operation=<op> -p hidden_dim=<H> -p seq_len=<S>` → `region_mfu -p name=<region> -p theoretical_flops=<N>` → `arithmetic_intensity -p theoretical_flops=<N>` |
| `specific-kernel` | `kernel_instances -p name=<name>` |

Device propagation: `region_mfu` uses **`-p device_id=N`** (NOT `-p device=N`).
`top_kernels`, `tensor_core_usage`, and `kernel_launch_pattern` do not accept device.
`arithmetic_intensity` and `kernel_instances` accept `-p device=N`. See PRINCIPLES.md §6.

Mode 3 is the **sole caller of `MFU.md`**. Never call `region_mfu` from Mode 1 — always
redirect via "want an MFU number? → Mode 3 mfu sub-focus". See `MFU.md` for FLOPs formulas
and model architecture table.

---

## 4. Signals

| Field / condition | Diagnosis | Action |
|-------------------|-----------|--------|
| `top_kernels[0].total_ms` / `profile_span_ms` > 60% | Single hotspot dominates | tune or replace that kernel |
| `tensor_core_usage[*].tc_achieved_pct` = 0 for eligible kernels | Tensor cores inactive (FP32 fallback) | force BF16 / TF32; check `torch.set_default_dtype` |
| `tensor_core_usage[*].is_outlier` = true | Kernel has < 50% TC time (FP32 fallback) | inspect matmul shape; pad to multiple of 8/16 |
| `region_mfu.mfu.mfu_pct < 30` | Severely underutilizing GPU | widen batch; improve memory access pattern |
| `region_mfu.mfu.mfu_pct > 100` | FLOPs scope too broad — **always wrong** | narrow `operation` to match the actual kernel target (see MFU.md) |
| `arithmetic_intensity.classification` starts with `"Memory-bound"` | Below ridge point | fuse ops; increase tile sizes; reduce redundant loads |
| `arithmetic_intensity.classification` starts with `"Compute-bound"` with low MFU | Above ridge but inefficient | occupancy issue; check register pressure / block size |
| `kernel_launch_pattern.sync_stalls` > 0 on a stream | CPU dispatch bottleneck | use CUDA graphs; batch launches |
| `top_kernels[0]` is custom / Triton AND > 60% | SASS-level investigation warranted | suggest Mode 7 (coming Stage C1) |

---

## 5. Cross-mode exits

After delivery, suggest a second mode only if a distinct critical finding exists.
Soft cap: avoid suggesting >2 modes in one delivery.

- Top custom/Triton kernel > 60% AND `tensor_core_usage` / `arithmetic_intensity` ambiguous
  → suggest **Mode 7** (coming Stage C1; fall through to Mode 1 until then)
- Per-layer attribution needed to identify which model component owns the hotspot
  → suggest **Mode 5** (`layer` sub-focus)

---

## 6. Delivery

Follow `PRINCIPLES.md §5` for evidence build + timeline URL. Then 3-part summary:

1. **Root cause** — hotspot identity + quantified share:
   > "Top kernel `ampere_bf16_s16816gemm_bf16_128x128x32_ldg8_f2f_stages_32x3_nn`
   > accounts for 68% of GPU time. Tensor core utilization is 91% — compute is the bottleneck,
   > not memory."

2. **Specific fix** — matching the finding:
   - FP32 fallback: `model = model.to(torch.bfloat16)` or `torch.set_float32_matmul_precision('high')`
   - Low MFU, memory-bound: increase batch size; fuse attention via `scaled_dot_product_attention`
   - Custom kernel: profiling alone can't fix — suggest Mode 7 for SASS-level diagnosis
   - Small matmul shapes: pad sequence length / batch to GEMM-friendly multiples (128, 256)

3. **Expected gain** — MFU % if computed; `speedup_estimator` if NVTX present; omit otherwise.

See `ROOT_CAUSE.md §1` for the cross-mode cause→fix matrix.
