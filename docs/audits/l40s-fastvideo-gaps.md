# L40S FastVideo — follow-up gaps

Profile: `perf_compile.sqlite` · [Dataset](https://huggingface.co/datasets/rich7421/fastvideo-wan-l40s-nsys) · Audit: [l40s-fastvideo-skill-audit.md](l40s-fastvideo-skill-audit.md)

## P0 — Top finding / diagnose path

1. **`profile_health_manifest`** — `_infer_bottleneck()` treats high CPU sync before low overlap. On this profile it returns “High CPU Synchronization Blocking” on a 20 s trim even though overlap is 0% and NCCL/compute share stream 7. Should surface same-stream NCCL when `same_stream_diagnosis` is set.

2. **Manifest auto-trim** — Default 20 s window understates idle (~1.9% vs ~13.9% full span) and diverges from overlap numbers in `analysis_measurements.md` (full ~677 s run).

3. **`evidence build` finding rank** — `evidence build --format json` produces 401 findings. The stream-7 serialization finding (`"Low Compute/NCCL Overlap (0.0%)"`) is at position **390/401**, buried below ~377 "Slow Iteration" regions and several idle-gap entries. The finding does contain stream ID (7), overlap pct (0%), and NCCL-only ms (271,563 ms of 677,344 ms span), but no NVTX scope (Ulysses `all_to_all_4D` not mentioned). Fix: elevate the `overlap_breakdown` same-stream finding to a top-ranked composite finding above per-iteration slow regions.

## P1 — Defaults and coverage

4. **Multi-GPU** — `overlap_breakdown` analyzes device 0 by default; devices 1–3 use stream 17, not stream 7. Need per-device overlap (or one summary) without four manual `-p device=N` runs.

5. **`iteration_timing`** — Emits `heuristic_step_*` labels when NVTX markers do not match; output should say when labels are synthesized.

6. **Parameter-gated skills** — `arithmetic_intensity`, `theoretical_flops`, `region_mfu`, `speedup_estimator` error without required `-p` values. Errors should list required params and an example for inference profiles.

## P2 — Output shape

7. **Comm skills** — No note that this node is PCIe (no NVLink) when overlap is near zero and payload is large.

8. **`top_kernels`** — Totals span all GPUs; per-device or per-rank breakdown would match how the dataset reports kernel share.

9. **`nvtx_layer_breakdown`** — Top regions on the compile profile are torch compile / `aten::to`, not Ulysses `all_to_all_4D`. Filter or depth option for denoising-stage children would help.
