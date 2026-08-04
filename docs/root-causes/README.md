# Book of Root Causes — Quick Reference

> Performance problems that cost millions of dollars in GPU hours, distilled into a single table.

| # | Root Cause | Symptom | Detection Skill | Severity |
|---|-----------|---------|----------------|----------|
| 1 | **GPU Bubbles** | Idle gaps between kernels on a stream | `gpu_idle_gaps` | High |
| 2 | **CPU Bottleneck** | Low GPU utilization despite available work | `thread_utilization` | High |
| 3 | **NCCL Serialization** | AllReduce not overlapping with compute | `nccl_breakdown` | High |
| 4 | **Excessive H2D Transfers** | Large memory copies in the critical path | `memory_transfers` | Medium |
| 5 | **Small Kernel Overhead** | Thousands of tiny kernels with high launch cost | `kernel_launch_overhead` | Medium |
| 6 | **Kernel Hotspot** | Single kernel dominates >50% of total time | `top_kernels` | High |
| 7 | **Missing NVTX** | Cannot attribute kernels to source code | `nvtx_kernel_map` | Low |
| 8 | **GC Pauses** | Python garbage collection stalls GPU pipeline | `gpu_idle_gaps` | Medium |
| 9 | **Module Loading** | Import/JIT compilation during forward pass | `gpu_idle_gaps` | Low |
| 10 | **Compute-Comm Imbalance** | Some ranks finish early, wait at barrier | `nccl_breakdown` | High |
| 11 | **Stream Serialization** | Streams that should overlap but run sequentially | `gpu_idle_gaps` | Medium |
| 12 | **Excessive Synchronization** | `cudaDeviceSynchronize` in the loop | `kernel_launch_overhead` | Medium |
| 13 | **FP32 Fallback** | Tensor Core eligible kernels falling back to FP32/SIMT | `tensor_core_usage` | High |

---

## How to Use This

1. **Run `nsys-ai agent analyze <profile>`.** It runs a fixed set of skills and prints one
   report. Adding `--evidence` also runs the evidence pipeline and writes a findings JSON.
2. **Check the top hits** — start with the High severity rows.
3. **Drill down** with `nsys-ai skill run <skill> <profile>` for the detail behind a hit,
   or ask a question directly with `nsys-ai ask <profile> "..."`.
4. **Read the [full writeup](book.md)** for remediation guidance.

## What `agent analyze` covers

Five of the detection skills named above run on every `agent analyze`: `gpu_idle_gaps`,
`top_kernels`, `nccl_breakdown`, `memory_transfers`, and `kernel_launch_overhead`. Rows 1,
3, 4, 5, 6, 8, 9, 10, 11 and 12 are therefore reported without any extra command.

The other three named skills are not part of that run. Two of the three root causes are
still surfaced in the report anyway, by a different skill; the third needs one explicit
command:

- **Row 13, FP32 Fallback.** `tensor_core_usage` does not run, but `top_kernels` carries
  Tensor Core columns: the report marks each kernel that is TC eligible but not using
  Tensor Cores, and the evidence pipeline emits a `tc_eligible_inactive` finding for it.
  This signal comes from the Parquet cache; on the pure-SQLite fallback path TC
  eligibility is unknown and no such finding is emitted.
- **Row 7, Missing NVTX.** `nvtx_kernel_map` does not run, but a profile without usable
  NVTX is reported anyway: `iteration_timing` and `nvtx_layer_breakdown` abstain with an
  explicit "no NVTX annotation — re-capture with NVTX enabled" message, and the evidence
  pipeline emits `profile_insufficient_nvtx_coverage`.
- **Row 2, CPU Bottleneck.** Not covered by a plain `agent analyze`. The skill the table
  names, `thread_utilization`, needs CPU sampling in the capture (a `COMPOSITE_EVENTS`
  table); without it it abstains and says so. Per-thread attribution does not require that
  capture option, though: `nsys-ai skill run cpu_gpu_pipeline <profile>` derives it from
  the CUDA runtime trace joined to kernels by `correlationId`, and reports per-thread
  dispatch counts, launch-queue depth and GPU starvation events on a profile with no CPU
  sampling at all. `agent analyze --evidence` reaches that data too — `critical_path` runs
  there and folds the starvation count into its CPU attribution line alongside a bound
  verdict — but neither `critical_path` nor `cpu_gpu_pipeline` runs without `--evidence`.

See also: [veteran-questions.md](veteran-questions.md) — diagnostic questions a performance expert would ask.
