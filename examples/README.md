# 🏛️ Nsight Exhibition

Showcase of analysis tools for NVIDIA Nsight Systems profiles.

## Exhibits

### NVTX Stack Trace Viewer

Interactive HTML views showing GPU kernel execution organized as a function call stack,
with NVTX annotations forming the hierarchy and kernels as leaves.

| File | Description |
|------|-------------|
| [nvtx_stack_gpu4_one_iteration.html](nvtx_stack_gpu4_one_iteration.html) | One transformer layer iteration (39.98-40.5s) — see the full call stack from `sample_0` → `TransformerLayer` → `FlashAttention` → `flash_fwd_kernel` |
| [nvtx_stack_gpu4_full.html](nvtx_stack_gpu4_full.html) | Full 3-second window (39-42s) — multiple iterations with all kernels and NVTX nesting |

**Features:**
- Collapsible tree nodes (click to expand/collapse)
- Search/filter by kernel or NVTX name
- Expand to depth controls (depth 2, 4, or all)
- Dark theme, monospaced font for readability

### Example hierarchy from one iteration:

```
📦 sample_0(repeat=5)  (1628ms)
  📦 TransformerLayer._forward_attention.self_attention  (43.5ms)
    📦 TELayerNormColumnParallelLinear forward  (7.7ms)
      📦 transformer_engine._LayerNormLinear.forward.norm
        📦 nvte_rmsnorm_fwd
          ⚡ rmsnorm_fwd_tuned_kernel [stream 21]  (0.138ms)
      📦 transformer_engine._LayerNormLinear.forward.gemm
        📦 nvte_cublas_gemm
          ⚡ nvjet_tst_192x192 [stream 21]  (2.1ms)
    📦 TEDotProductAttention.forward  (26.3ms)
      📦 ... → te.FlashAttention.run_attention
        ⚡ flash_fwd_kernel [stream 21]  (25.9ms)
```

## Data Source

Profile: `baseline.t128k.host-fs-mbz-gpu-899.nsys-rep`
- 7× NVIDIA H200 (132 SMs, 150GB, GH100 chip)
- Megatron-LM Transformer Engine workload
- Profiled with NVIDIA Nsight Systems

**Dataset:** [GindaChen/nsys-hero](https://huggingface.co/datasets/GindaChen/nsys-hero) on HuggingFace

## How to reproduce

```bash
# Download profile data (if not already present)
python3 ../hf_data.py distca-0

# Generate exhibition HTMLs
python3 -c "
import sys; sys.path.insert(0, '../lib/python')
from nsight import profile
from nsight.viewer import write_html

prof = profile.open('../playground/distca-0/files/profile/baseline.t128k.host-fs-mbz-gpu-899.sqlite')
write_html(prof, device=4, trim=(int(39.98e9), int(40.5e9)), path='nvtx_stack_gpu4_one_iteration.html')
write_html(prof, device=4, trim=(int(39e9), int(42e9)), path='nvtx_stack_gpu4_full.html')
prof.close()
"
```
