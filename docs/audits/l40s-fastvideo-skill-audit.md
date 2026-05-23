# L40S FastVideo skill audit

Profile: [`perf_compile.sqlite`](https://huggingface.co/datasets/rich7421/fastvideo-wan-l40s-nsys) (FastVideo Wan2.1, 4× L40S, `torch.compile` on)  
Skill list: `tests/test_skills.py::test_list_skills` (35 built-ins)  
Reproduce: [README.md](README.md) · Follow-ups: [l40s-fastvideo-gaps.md](l40s-fastvideo-gaps.md)

## Ground truth (device 0)

Compute and NCCL run on **CUDA stream 7** with **~0% overlap** (~311k ms compute-only, ~272k ms NCCL-only on a ~677 s span). NCCL payload **~8,936 GiB**. Expected fix: dedicated NCCL stream in the app, not a single-kernel tune.

## Grades

| Grade | Meaning |
|-------|---------|
| PASS | Output matches the ground truth above (or correctly reports no issue) |
| PARTIAL | Runs but misses the main story, weak defaults, or incomplete output |
| SKIP | Needs extra inputs (`theoretical_flops`, `trace_dir`, etc.) or N/A here |
| FAIL | Wrong or broken vs ground truth (none on this profile) |

**Batch** = `python scripts/batch_audit_skills.py "$L40S_PROFILE"` → `_batch_summary.json` (local, gitignored).

## Results

| Skill | Batch | Render | Grade | Gap |
|-------|-------|--------|-------|-----|
| `overlap_breakdown` | OK | summary | PASS | — |
| `nccl_payload_breakdown` | OK | rows | PASS | — |
| `nccl_breakdown` | OK | rows | PASS | Defaults to device 0 |
| `nccl_communicator_analysis` | OK | summary | PASS | — |
| `nccl_anomaly` | OK | rows | PARTIAL | Does not identify dominant collectives by name |
| `kernel_overlap_matrix` | OK | rows | PASS | — |
| `stream_concurrency` | OK | rows | PARTIAL | Does not call out stream 7 serialization |
| `sync_cost_analysis` | OK | summary | PASS | High sync % pushes manifest to rank sync first |
| `profile_health_manifest` | OK | summary | PARTIAL | 20 s auto-trim; `suspected_bottleneck` is CPU sync, not stream 7 |
| `root_cause_matcher` | OK | rows | PARTIAL | May not rank same-stream NCCL first |
| `gpu_idle_gaps` | OK | rows | PARTIAL | Idle on stream 7 not summarized as ~13.9% headline |
| `pipeline_bubble_metrics` | OK | rows | PARTIAL | Same |
| `iteration_timing` | OK | rows | PARTIAL | Generic step labels |
| `iteration_detail` | OK | summary | PARTIAL | Batch uses `iteration=0`; slow iter needs `--iteration` |
| `top_kernels` | OK | rows | PARTIAL | Aggregates all GPUs |
| `nvtx_layer_breakdown` | OK | rows | PARTIAL | Compile NVTX noise; Ulysses layer names not top |
| `nvtx_kernel_map` | OK | rows | PARTIAL | Noisy regions on compile profile |
| `memory_transfers` | OK | rows | PASS | — |
| `h2d_distribution` | OK | rows | PASS | — |
| `memory_bandwidth` | OK | rows | PASS | — |
| `kernel_launch_overhead` | OK | rows | PASS | — |
| `kernel_launch_pattern` | OK | rows | PARTIAL | Does not link streams to overlap story |
| `kernel_instances` | OK | rows | PASS | — |
| `thread_utilization` | EMPTY | EMPTY | PARTIAL | `COMPOSITE_EVENTS` missing or empty on this export — no CPU thread utilization rows |
| `cpu_gpu_pipeline` | OK | rows | PASS | — |
| `host_sync_parent_ranges` | OK | rows | PARTIAL | ~34 s on full profile |
| `gc_impact` | OK | rows | PASS | — |
| `module_loading` | OK | rows | PASS | — |
| `arithmetic_intensity` | ERROR_JSON | error | SKIP | Requires `theoretical_flops` |
| `theoretical_flops` | ERROR_JSON | error | SKIP | Requires model dimensions |
| `region_mfu` | ERROR_JSON | error | SKIP | Requires `theoretical_flops` + region name |
| `speedup_estimator` | ERROR_JSON | error | SKIP | Requires `iteration_ms` |
| `tensor_core_usage` | OK | rows | PASS | — |
| `cutracer_analysis` | SKIP | SKIP | SKIP | Requires `trace_dir` |
| `schema_inspect` | OK | rows | PASS | — |

**PASS 16 · PARTIAL 14 · SKIP 5 · FAIL 0**

## Notes

- **`overlap_breakdown`** is the skill that states stream 7 and 0% overlap clearly on the full profile.
- **`profile_health_manifest`** reports 0% overlap in its overlap block but names **High CPU Synchronization Blocking** as `suspected_bottleneck` on a 20 s trimmed window (`idle_pct` 1.9% vs ~13.9% on the full span).
- **`evidence build --format json`** produces 401 findings. The stream-7 serialization finding (`"Low Compute/NCCL Overlap (0.0%)"`, with stream=7, 0% overlap, 271,563 ms NCCL-only out of 677,344 ms span, same-stream diagnosis) appears at position **390/401** — buried below ~377 "Slow Iteration" regions and several idle-gap entries. It has no NVTX scope context (no Ulysses `all_to_all_4D` mention). This confirms P0 #3 in [gaps.md](l40s-fastvideo-gaps.md): the finding exists but is not a top finding.
