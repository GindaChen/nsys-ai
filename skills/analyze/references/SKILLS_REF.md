# Skills Quick Reference

**Always start with**: `profile_health_manifest` (one-shot triage, ~2 s, returns everything)

**Token budget**: add `--max-rows N` to limit JSON output size.
Default cap: 50 rows. Use 10–20 for large profiles.

**Trim**: add `--trim START_S END_S` or `--iteration N` to all skills.

**Run syntax**:
```bash
nsys-ai skill run <skill_name> <profile> --format json [-p key=value ...]
```

---

## Utility (start here)

| Skill | Description |
|-------|-------------|
| `profile_health_manifest` | One-shot: GPU, top-5 kernels, overlap %, NCCL summary, idle %, root causes, `suspected_bottleneck` |
| `schema_inspect` | List all tables + columns (run if unsure about schema) |

---

## Kernels

| Skill | Params | Description |
|-------|--------|-------------|
| `top_kernels` | `limit=15` | Heaviest kernels by total time |
| `tensor_core_usage` | — | FP32 fallback detection — are tensor cores active? |
| `kernel_instances` | `name=<kernel>` ✅ | Exact ns timestamps per kernel instance → use for findings.json |
| `kernel_launch_overhead` | — | CPU dispatch latency per kernel |
| `kernel_launch_pattern` | — | Per-stream burst density, sync-stall detection |
| `stream_concurrency` | — | Multi-stream concurrency analysis |
| `gpu_idle_gaps` | `min_gap_ns=1000000` | Idle gaps with CPU attribution |
| `region_mfu` | `name` ✅, `theoretical_flops` ✅ | MFU for a named NVTX region |
| `theoretical_flops` | `operation` ✅, `hidden_dim` ✅, `seq_len` ✅ | Exact FLOPs for transformer ops |
| `arithmetic_intensity` | `theoretical_flops` ✅ | Roofline: compute-bound vs memory-bound |

✅ = required parameter

---

## Communication

| Skill | Description |
|-------|-------------|
| `nccl_breakdown` | NCCL ops by stream × collective (infer TP/PP/DP from stream) |
| `overlap_breakdown` | Compute/NCCL overlap %. >60% = NCCL well-hidden |
| `nccl_communicator_analysis` | Communicator-level NCCL efficiency (needs NVTX payload tables) |
| `nccl_anomaly` | Outlier NCCL ops vs median [`threshold=3.0`] |
| `kernel_overlap_matrix` | Pairwise overlap: comm×comm, comm×compute |

---

## Memory

| Skill | Description |
|-------|-------------|
| `memory_bandwidth` | Per-direction throughput, peak vs sustained |
| `memory_transfers` | H2D/D2H/D2D/P2P breakdown with counts |
| `h2d_distribution` | H2D transfer pattern: init-heavy / spread-out / spike |

---

## NVTX / Code Attribution

| Skill | Params | Description |
|-------|--------|-------------|
| `nvtx_layer_breakdown` | — | NVTX region GPU time breakdown with top kernels per region |
| `nvtx_kernel_map` | — | NVTX annotation → GPU kernel mapping (source attribution) |
| `iteration_timing` | `marker=sample_0` | Per-iteration timing |
| `iteration_detail` | `iteration` ✅ | Kernel breakdown for one iteration |

---

## System

| Skill | Description |
|-------|-------------|
| `cpu_gpu_pipeline` | CPU dispatch lead time, GPU starvation events |
| `thread_utilization` | CPU % per thread (DataLoader / GIL detection) |

---

## Bottlenecks

| Skill | Description |
|-------|-------------|
| `module_loading` | JIT compilation + cuModuleLoad stalls |
| `gc_impact` | cudaMalloc/cudaFree stalls + Python GC events |
| `sync_cost_analysis` | CPU-side sync density + blocking cost (cudaStreamSynchronize, etc.) |
| `pipeline_bubble_metrics` | True GPU idle % (interval merge, O(n log n)) |

---

## Analysis

| Skill | Params | Description |
|-------|--------|-------------|
| `root_cause_matcher` | — | Auto-detect 12+ anti-patterns with evidence + fix |
| `speedup_estimator` | `iteration_ms` ✅ | What-if: idle→X speedup, perfect overlap→Y |
| `cutracer_analysis` | `trace_dir` ✅ | Instruction-mix classifier (memory/compute/sync-bound) |

---

## Common patterns

```bash
# Full triage starting point
nsys-ai skill run profile_health_manifest profile.sqlite --format json

# Top kernels, budget 20 rows
nsys-ai skill run top_kernels profile.sqlite --format json --max-rows 20

# NCCL deep dive
nsys-ai skill run nccl_breakdown profile.sqlite --format json
nsys-ai skill run overlap_breakdown profile.sqlite --format json

# NVTX attribution
nsys-ai skill run nvtx_layer_breakdown profile.sqlite --format json
nsys-ai skill run nvtx_kernel_map profile.sqlite --format json

# Root cause auto-detect
nsys-ai skill run root_cause_matcher profile.sqlite --format json

# Speedup estimate (needs iteration_ms from iteration_timing first)
nsys-ai skill run iteration_timing profile.sqlite --format json
nsys-ai skill run speedup_estimator profile.sqlite --format json -p iteration_ms=<value>
```
