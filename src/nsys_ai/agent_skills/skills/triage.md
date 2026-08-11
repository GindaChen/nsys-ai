# Skill: Profile Triage

Read this when the user opens a profile with no specific question,
or asks "what's the bottleneck?" / "why is it slow?".
**Read `PRINCIPLES.md` first** for rules, error handling, and tool definitions.

---

This is a **top-down, elimination-based** pipeline. From the macro-level to the micro-level, utilize these 6 stages to systematically eliminate bottlenecks and pinpoint root causes. 

You must intelligently combine these stages via the explicitly listed `nsys-ai` CLI tools.

### Fast Path: Profile Health Manifest
**Before diving into the 6 stages**, run the one-shot manifest which executes Stages 1-2-5-6 internally:
```bash
nsys-ai skill run profile_health_manifest profile.sqlite --format json
```
This returns GPU info, top 5 kernels, compute/NCCL overlap, an NCCL summary (streams, counts, dominant type), idle gaps, root cause findings, and `suspected_bottleneck` — all in a single call. Use `suspected_bottleneck` to decide which stage to drill into next. Use `--max-rows N` on any subsequent skill call to control JSON output size.

> **Note on Profiler Overhead**: If the manifest reports high **Profiler Overhead** (>1%), this is Nsight Systems instrumentation latency. The two biggest contributors are `CUDA profiling initialization` (up to ~1s when CUPTI attaches to PyTorch streams) and `CUDA profiling data flush` (~300ms when dumping buffers to disk on exit). Because this overhead is heavily concentrated at the start and end of a script, running `nsys profile python train.py` from start-to-finish severely skews short profiles. **Recommendation**: Tell the user to use `torch.cuda.profiler.start/stop()` inside the training loop and profile with `nsys profile --capture-range=cudaProfilerApi` to exclude cold-start/teardown overhead.

### Stage 1: Orient — Establish the Workload Context
**Goal**: Determine the basic context to avoid blindly guessing.
**Actions**:
- What is the profile duration? How many GPUs?
- What is the workload (training, inference, fine-tuning)?
- What parallelism strategy is used (TP, PP, DP, FSDP, Single GPU)?
- Are NVTX annotations present? (Determines granularity of downstream analysis).
_How to detect parallelism_: Look at NCCL op types. Majority AllReduce → DP/FSDP. AllGather + ReduceScatter → FSDP/TP. Send/Recv → PP. Mixed → Hybrid.
**Suggested Approach**: Use `nsys-ai info <profile>` to get basic hardware facts. For advanced schema or parallelism detection, explore the [commands/skill.md](../commands/skill.md) catalog for relevant skills (e.g., `schema_inspect`, `nccl_anomaly`).

---

### Stage 2: Temporal Breakdown — The Time Budget
**Goal**: Establish a "time budget" to determine where GPU time is spent (compute, communication, memory, idle).
**Actions**:
- Calculate GPU time breakdown: Compute %, NCCL %, Memcpy %, Idle %.
- Identify the **primary bottleneck class**:
  - *Compute-bound*: Idle < 10%, NCCL < 20%
  - *Communication-bound*: NCCL > 30%, compute/NCCL ratio < 1
  - *Sync/idle-bound*: Idle > 30%
  - *Memory-bound*: H2D/D2H > 15%, or massive memcpy interleaved with compute
  - *Pipeline-bubble-bound* (PP only): Regular, long idle gaps near step boundaries
**Suggested Approach**: Query the aggregate compute/NCCL breakdown (e.g., via the `overlap_breakdown` skill). Use `pipeline_bubble_metrics` for exact GPU idle % per device. Refer to [commands/skill.md](../commands/skill.md) to discover other skills tailored for checking idle gaps or memory bandwidth.

---

### Stage 3: Kernel Deep-Dive — The Heaviest Kernels
**Goal**: Identify the top-N heaviest kernels and determine if their duration is justified.
**Actions**:
- Find the top 10 kernels by total duration.
- Classify and evaluate each top kernel:
  - Compute (GEMM, FlashAttention, Conv) → Normal, actual work being done.
  - NCCL (AllReduce, AllGather) → Check if the proportion is reasonable.
  - Memcpy → Abnormal. Investigate why data is moving.
  - Sync/Wait → Abnormal. Points to synchronization/idle issues.
- Analyze **launch patterns**: Are they executed in bursts or evenly spaced? Is there serialization (kernels executing sequentially on the same stream without concurrency)?
**Suggested Approach**: Identify the top consumers first (e.g., via the `top_kernels` skill). Use `kernel_instances -p name=<kernel>` to get exact ns timestamps for evidence overlay. See [commands/skill.md](../commands/skill.md) for deeper analysis tools to investigate launch patterns or stream concurrency.

---

### Stage 4: NVTX → Code Mapping — Who is Calling What?
**Goal**: Attribute GPU kernels back to the Python source code level operations.
**Actions** (Requires NVTX annotations):
- Inspect NVTX hierarchy: Usually Step > Forward/Backward/Optimizer > Layer > Op.
- Perform Top Kernel NVTX attribution: Which NVTX range is calling `flash_bwd_kernel`?
- Identify **which layer or operation** consumes the most GPU time.
- If NVTX is absent: Perform heuristic kernel-to-PyTorch mapping (e.g., `ampere_sgemm` → linear layer, `flash_fwd` → attention).
**Suggested Approach**: View the full NVTX tree (e.g., using `nsys-ai tui <profile> --depth 3`). For specific kernel mapping, consult the [commands/skill.md](../commands/skill.md) catalog for NVTX-related skills like `nvtx_layer_breakdown`.

---

### Stage 5: Cross-GPU Analysis — Multi-GPU Bottlenecks
**Goal**: Identify inter-GPU dependencies and imbalances.
**Actions**:
- **Compute/NCCL overlap**: Are NCCL and compute overlapping? Low overlap (<30%) means GPUs are idle during NCCL (no computation hiding).
- **NCCL Anomaly Detection**: Do specific NCCL op durations exceed the median significantly? (Suggests a specific rank dragging down collective comms).
- **Per-GPU idle variance**: Is idle % consistent across all GPUs? E.g., if GPU 0 idles at 5% but GPU 3 idles at 30%, suspect load imbalance or PP bubbles.
- **Pipeline bubbles** (PP only): Look for fixed-pattern long idle gaps near training step boundaries (Send/Recv waits between micro-batches).
**Suggested Approach**: Analyze cross-GPU overlaps (e.g., via the `overlap_breakdown` skill). Check [commands/skill.md](../commands/skill.md) for multi-GPU anomaly detectors like `nccl_anomaly` or per-GPU idle variance checkers.

---

### Stage 6: Root Cause & Recommendations
**Goal**: Synthesize findings into a causal chain to produce actionable recommendations.
**Actions**:
- **Source Code Correlation**: If you have access to the user's local source code alongside the profile, map your NVTX findings directly to the corresponding Python files. Inspect the exact code block (e.g., the dataloader setup, the explicit `torch.cuda.synchronize()` call) to verify the bottleneck and propose a direct code fix.

**Common Root Cause Matrix**:

| Root Cause | Evidence Pattern | Recommendation |
|-----------|-----------------|----------------|
| Explicit sync stalls | `gpu_idle_gaps` attributed to `cudaDeviceSynchronize` | Remove explicit syncs; only sync at iteration boundaries. |
| DataLoader bottleneck | Unchanging idle gap length across steps; high CPU util | Increase `num_workers`, use `pin_memory=True`, add prefetching. |
| Low compute/NCCL overlap | Overlap < 30% | Enable async comms in `torch.distributed`; check backward pass for implicit syncs. |
| NCCL outlier | Specific NCCL op duration >> median | Inspect NVLink topology; verify no PCIe fallback. |
| PP bubble | >10ms idle gap strictly at step boundaries | Increase micro-batch count, or use interleaved 1F1B scheduling. |
| Unnecessary H2D transfer | Massive H2D memory copies inside the training loop | Ensure tensors stay on device; check for rogue `.item()` or `.cpu()` calls. |
| Kernel serialization | Stream concurrency = 1 (kernels operate serially) | Verify ops aren't all defaulting to Stream 0; overlap via custom CUDA streams. |
| JIT / module loading | `module_loading` shows repeated `cuModuleLoad*` outside init phase | Pre-compile with `torch.compile()` warmup; use `CUDA_MODULE_LOADING=EAGER`. |
| GC / memory churn | `gc_impact` shows frequent `cudaFree` with long max stall | Pre-allocate tensors; set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; reduce Python object churn. |

**Suggested Approach**: Match findings against known patterns (e.g., via the `root_cause_matcher` skill). Consult [commands/skill.md](../commands/skill.md) for estimation tools like `speedup_estimator` to quantify potential fixes.

> **Output Requirement**: After eliminating hypotheses via Stages 1-6, generate visual evidence:
> ```bash
> # Option A: Automatic — run heuristic analyzers
> nsys-ai evidence build profile.sqlite --format json -o /tmp/findings.json
>
> # Option B: Manual — encode your conclusions as findings.json
> # (use kernel_instances skill to get exact ns timestamps)
>
> # Then visualize:
> nsys-ai timeline-web profile.sqlite --findings /tmp/findings.json
> ```
