# Command: `nsys-ai skill`

Manage and run builtin analysis skills against Nsight Systems profiles.
**Read `PRINCIPLES.md` first** for rules, error handling, and tool definitions.

---

## Subcommands

### `nsys-ai skill list`

List all registered skills with name, category, and description.

```bash
nsys-ai skill list                   # human-readable table
nsys-ai skill list --format json     # JSON array (for programmatic use)
```

### `nsys-ai skill run`

Execute a skill against a profile database.

```bash
nsys-ai skill run <skill_name> <profile.sqlite> [options]

# Examples:
nsys-ai skill run top_kernels profile.sqlite
nsys-ai skill run gpu_idle_gaps profile.sqlite --param min_gap_ns=5000000
nsys-ai skill run nccl_breakdown profile.sqlite --format json
nsys-ai skill run top_kernels profile.sqlite --trim 1.5 3.0   # seconds
```

**Options**:
- `--format {text,json}` — output format (default: text)
- `--param KEY=VALUE` — pass a parameter (repeatable, shorthand: `-p`)
- `--trim START_S END_S` — restrict analysis to time range (seconds, space-separated)
- `--max-rows N` — limit JSON output to at most N data rows (for token budget control). When clipping occurs, the JSON array contains up to N data rows plus a final truncation metadata object: `{"_truncated": true, "_total_rows": <original>, "_shown_rows": <shown>}`.
- `--iteration N` — auto-trim to the N-th training iteration (0-based). Cannot be combined with `--trim`. Requires NVTX markers (or falls back to heuristic kernel-gap detection).
- `--marker TEXT` — NVTX marker for iteration boundary detection (default: `sample_0`). Used with `--iteration`.

### `nsys-ai skill [--skills-dir <dir>] add <path.md>`

Register a custom skill from a markdown file.
Requires `--skills-dir <dir>` (before the subcommand) or the `NSYS_AI_CUSTOM_SKILLS_DIR` environment variable.
The file must use heading-based format (not YAML frontmatter):
- `# <name>` — top-level heading with the skill name
- `## Description` — short summary
- `## Category` — e.g. `kernels`, `memory`, `custom`
- `## SQL` — fenced ```sql code block with the query

See `skills/SKILL_TEMPLATE.md` for a complete example.

### `nsys-ai skill [--skills-dir <dir>] remove <name>`

Unregister a previously added custom skill.
Requires `--skills-dir <dir>` (before the subcommand) or the `NSYS_AI_CUSTOM_SKILLS_DIR` environment variable.

### `nsys-ai skill save`

Export a skill definition to a markdown file.

```bash
nsys-ai skill save <name> -o <output.md>
```

---

## Builtin Skills Catalog

The `--trim` flag is accepted for all skills, but only restricts the analysis window
for skills whose SQL uses `{trim_clause}` or whose `execute_fn` explicitly reads
`trim_start_ns`/`trim_end_ns`. Skills like `theoretical_flops`, `speedup_estimator`,
and `schema_inspect` ignore it.
Parameters marked **required** must be provided via `--param`.

---

### Category: `kernels`

| Skill | Title | Description |
|-------|-------|-------------|
| `top_kernels` | Top GPU Kernels by Total Time | Lists the heaviest GPU kernels ranked by cumulative execution time. Use to identify hotspots. |
| `tensor_core_usage` | Tensor Core Utilization | Ranks GPU kernels based on Tensor Core eligibility vs actual utilization. Detects FP32 fallbacks. |
| `kernel_instances` | Kernel Instance Details | Returns individual kernel instances with exact **ns timestamps** (`start_ns`, `end_ns`). Use to get precise time ranges for `findings.json` evidence overlay. |
| `kernel_launch_overhead` | Kernel Launch Overhead | Measures the gap between CUDA Runtime API call and GPU kernel execution. High overhead = CPU-side bottleneck. |
| `kernel_launch_pattern` | Kernel Launch Pattern Analysis | Per-stream dispatch rate, burst density, inter-launch gaps, and sync-stall detection. |
| `stream_concurrency` | Stream Concurrency Analysis | Multi-stream GPU concurrency: active streams, kernel packing, compute/memory overlap. |
| `gpu_idle_gaps` | GPU Idle Gaps (Bubbles) | Finds idle gaps between consecutive GPU kernels with aggregation stats and CPU attribution. |
| `region_mfu` | Region-Level MFU | Computes MFU for an NVTX region or kernel. Requires `name` and `theoretical_flops`. |
| `theoretical_flops` | Theoretical FLOPs Calculator | Exact FLOPs for transformer operations (attention, mlp, full_model, etc.). LLMs should use this instead of manual multiplication. |
| `arithmetic_intensity` | Arithmetic Intensity vs. GPU Peak (Roofline) | Aggregate roofline assessment: combines GPU hardware specs (peak TFLOPS, HBM BW) with kernel time and user-provided FLOPs. Reports MFU and classifies workload as compute-bound or memory-bound. |

#### `top_kernels` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `limit` | int | | 15 | Max number of kernels to return |

#### `kernel_instances` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `device` | int | | 0 | GPU device ID |
| `name` | str | | (all) | Kernel name substring filter (demangled) |
| `limit` | int | | 10 | Max instances to return |

#### `kernel_launch_overhead` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `limit` | int | | 20 | Max results |

#### `kernel_launch_pattern` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `limit` | int | | 10 | Max streams to show |

#### `stream_concurrency` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `limit` | int | | 10 | Max streams to show |

#### `gpu_idle_gaps` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `min_gap_ns` | int | | 1000000 | Minimum gap in nanoseconds to report |
| `limit` | int | | 20 | Max results |
| `device` | int | | 0 | GPU device ID |

#### `region_mfu` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `name` | str | ✅ | — | NVTX region or kernel name to analyze |
| `theoretical_flops` | float | ✅ | — | Model FLOPs per step (from user) |
| `source` | str | | `nvtx` | Match source: `nvtx` or `kernel` |
| `peak_tflops` | float | | auto | GPU peak TFLOPS (auto-detected if omitted) |
| `num_gpus` | int | | 1 | Number of GPUs (for DP/TP adjustment) |
| `occurrence_index` | int | | 1 | Which occurrence to analyze (1-based) |
| `device_id` | int | | — | GPU device ID filter |
| `match_mode` | str | | `contains` | `contains`, `exact`, or `startswith` |

#### `theoretical_flops` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `operation` | str | ✅ | — | `attention` / `qkv_proj` / `output_proj` / `mlp` / `full_layer` / `full_model` / `linear` |
| `hidden_dim` | int | ✅* | 0 | Model hidden dimension (H) |
| `seq_len` | int | ✅* | 0 | Sequence length (S) |
| `num_layers` | int | | 1 | Number of transformer layers |
| `ffn_dim` | int | | 4×H | FFN intermediate dimension |
| `batch_size` | int | | 1 | Batch size |
| `multiplier` | int | | 1 | 1=fwd, 3=fwd+bwd, 4=fwd+bwd+ckpt |
| `M` / `N` / `K` | int | | 0 | For `linear` operation: matrix dimensions |

\* `hidden_dim` and `seq_len` must be > 0 for all operations except `linear` (which uses `M`/`N`/`K` instead). Passing 0 returns an `INVALID_ARGUMENT` error.

#### `arithmetic_intensity` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `theoretical_flops` | float | ✅ | — | Total FLOPs for the profiled workload (use `theoretical_flops` skill to compute) |
| `bytes_moved` | float | | — | Total bytes moved to/from HBM. If provided, computes true arithmetic intensity. |
| `device` | int | | 0 | GPU device ID |
| `peak_tflops` | float | | auto | Override GPU peak FP16 TFLOPS (auto-detected from chipName if omitted) |
| `hbm_bw_gbps` | float | | auto | Override HBM bandwidth in GB/s (auto-detected if omitted) |

---

### Category: `memory`

| Skill | Title | Description |
|-------|-------|-------------|
| `memory_transfers` | Memory Transfer Summary | Breaks down memory copy operations by direction (H2D, D2H, D2D, P2P). |
| `h2d_distribution` | H2D Transfer Time Distribution | Groups H2D transfers by second. Classifies pattern as `init_heavy`, `spread_out`, or `spike`. |
| `memory_bandwidth` | Memory Bandwidth & Utilization | Per-direction throughput (GB/s), peak vs sustained bandwidth, large-transfer identification. |

`memory_transfers` and `memory_bandwidth` have no required parameters.

#### `h2d_distribution` Parameters

No required parameters. Supports `--trim`.

---

### Category: `communication`

| Skill | Title | Description |
|-------|-------|-------------|
| `nccl_breakdown` | NCCL Collective Breakdown | Summarizes NCCL ops (AllReduce, AllGather, etc.) by stream × collective type with count, time, variability. Use stream_id to infer TP/PP/DP. |
| `nccl_anomaly` | NCCL Anomaly Detection | Detects outlier NCCL operations exceeding a threshold relative to their op type average. |
| `overlap_breakdown` | Compute/Communication Overlap | Quantifies compute vs NCCL overlap. `overlap_pct > 60%` = NCCL well-hidden. |
| `kernel_overlap_matrix` | Kernel Overlap Matrix | Pairwise overlap between kernel categories: comm×comm, comm×compute, compute×compute. |

#### `nccl_anomaly` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `threshold` | float | | 3.0 | Anomaly threshold: ratio to average duration |
| `limit` | int | | 20 | Max anomalies to return |

#### `overlap_breakdown` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `device` | int | | 0 | GPU device ID |

---

### Category: `nvtx`

| Skill | Title | Description |
|-------|-------|-------------|
| `nvtx_kernel_map` | NVTX → Kernel Mapping | Maps NVTX annotation ranges to the GPU kernels within them. Core of source-code attribution. |
| `nvtx_layer_breakdown` | NVTX Region GPU Time Breakdown | Attributes GPU kernels to NVTX regions, ranked by GPU time. Shows compute/NCCL split, top kernels, outlier detection. Auto-detects layer depth. |
| `iteration_timing` | Per-Iteration Timing Analysis | Detects repeating training iterations via NVTX markers. Reports per-iteration GPU timing and kernel counts. |
| `iteration_detail` | Per-Iteration Kernel Breakdown | Drill into a specific iteration: top kernels, NCCL stats, compute time, vs-median comparison. Use after `iteration_timing` identifies a slow iteration. |

#### `nvtx_kernel_map` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `limit` | int | | 50 | Max results |

#### `nvtx_layer_breakdown` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `limit` | int | | 20 | Max NVTX regions to return |
| `depth` | int | | auto | Filter to specific NVTX nesting depth (0=top-level) |
| `auto_depth` | bool | | true | Auto-detect layer depth via numbered patterns |

#### `iteration_timing` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `device` | int | | 0 | GPU device ID |
| `marker` | str | | `sample_0` | NVTX text pattern for iteration boundary |

#### `iteration_detail` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `iteration` | int | ✅ | | Iteration index (0-based) |
| `device` | int | | 0 | GPU device ID |
| `marker` | str | | `sample_0` | NVTX marker for iteration boundary detection |

---

### Category: `system`

| Skill | Title | Description |
|-------|-------|-------------|
| `cpu_gpu_pipeline` | CPU-GPU Pipeline Analysis | Measures CPU dispatch lead time, GPU starvation events, per-thread launch contribution. |
| `thread_utilization` | CPU Thread Utilization | CPU % by thread. Identifies CPU-bound threads starving the GPU (DataLoader, GIL). Requires `COMPOSITE_EVENTS` table. |

#### `cpu_gpu_pipeline` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `limit` | int | | 10 | Max threads to show |

#### `thread_utilization` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `limit` | int | | 10 | Max threads to show |

---

### Category: `bottlenecks`

| Skill | Title | Description |
|-------|-------|-------------|
| `module_loading` | JIT Compilation & Module Loading Stalls | Detects CUDA Driver/Runtime API calls for JIT compilation (`cuModuleLoadData`, `cuModuleLoad`, `cudaCompilePTX`) and module loading. Reports occurrence count, total/max/avg durations. Stalls during training (not init) indicate repeated JIT — a major pipeline bubble source. |
| `gc_impact` | Garbage Collection & Memory Stalls | Quantifies CPU stalls from `cudaMalloc`/`cudaFree` (with versioned API names) and Python GC events (via NVTX). Dual-layer detection: CUDA Runtime API clustering + NVTX GC markers. Supports both `text` and `textId` NVTX schemas. |

`module_loading` and `gc_impact` have no required parameters. Both support `--trim`.

---

### Category: `utilization`

| Skill | Title | Description |
|-------|-------|-------------|
| `pipeline_bubble_metrics` | Pipeline Bubble Metrics (True Idle %) | Merges all overlapping kernel, memcpy, and memset intervals per GPU to compute exact active time vs. true idle time. Reports `Bubble %` — the fraction of GPU wall-time with zero activity. Essential input for MFU calculation. Uses O(n log n) Python interval merging. |

`pipeline_bubble_metrics` has no required parameters. Supports `--trim`.

---

### Category: `analysis`

| Skill | Title | Description |
|-------|-------|-------------|
| `root_cause_matcher` | Root Cause Pattern Matcher | Automatically detects known GPU anti-patterns: bubbles, NCCL serialization, kernel hotspots, excessive sync, sync memcpy/memset, pageable memory. Returns findings with evidence and fix recommendations. |
| `speedup_estimator` | Speedup Estimation Framework | Estimates potential speedup from: eliminating idle gaps, perfect NCCL overlap, reducing TP degree. Pure computation — no DB access. |

#### `root_cause_matcher` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `device` | int | | 0 | GPU device ID |

#### `speedup_estimator` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `iteration_ms` | float | ✅ | — | Current iteration time in ms |
| `compute_ms` | float | | 0 | Total compute time in ms |
| `nccl_ms` | float | | 0 | Total NCCL time in ms |
| `idle_ms` | float | | 0 | Total GPU idle time in ms |
| `overlap_pct` | float | | 0 | Current compute/NCCL overlap % |
| `tp_degree` | int | | 1 | Current tensor parallel degree |
| `model_params_b` | float | | 0 | Model size in billions of params |
| `gpu_memory_gb` | float | | 80 | GPU memory in GB |

---

### Category: `utility`

| Skill | Title | Description |
|-------|-------|-------------|
| `schema_inspect` | Database Schema Inspector | Lists all tables and columns in the profile database (DuckDB or SQLite). Run this to understand available data. |
| `profile_health_manifest` | Profile Health Manifest | **Start here.** One-shot summary: GPU info, top 5 kernels, overlap stats, NCCL breakdown, idle gaps, root causes, and suspected bottleneck — all in a single call. |

#### `profile_health_manifest` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `device` | int | | 0 | GPU device ID |

`schema_inspect` has no required parameters.

---

## Skill Categories Quick Reference

| Category | Count | What it covers |
|----------|:-----:|----------------|
| `kernels` | 10 | GPU kernel timing, instance details, launch overhead, MFU, FLOPs, Tensor Cores, roofline |
| `memory` | 3 | H2D/D2H transfers, bandwidth, distribution |
| `communication` | 4 | NCCL breakdown, anomalies, overlap, overlap matrix |
| `nvtx` | 4 | NVTX→kernel mapping, layer breakdown, iterations, iteration detail |
| `system` | 2 | CPU→GPU pipeline, thread utilization |
| `bottlenecks` | 2 | JIT/module loading stalls, GC/memory allocation stalls |
| `utilization` | 1 | Pipeline bubble metrics (true GPU idle %) |
| `analysis` | 2 | Root cause patterns, speedup estimates |
| `utility` | 2 | Schema inspection, profile health manifest |
| **Total** | **30** | |

> **Note**: `memory_transfers.py` registers 2 skills (`memory_transfers` + `h2d_distribution`).
> There are exactly 30 unique Python builtin skills.

---

## Related Commands

| Command | Purpose |
|---------|--------|
| `nsys-ai evidence build <profile>` | Run heuristic analyzers → generate timeline-ready `findings.json` with ns timestamps |
| `nsys-ai timeline-web <profile> --findings <file>` | Visualize evidence findings on the timeline |

See [evidence-build.md](evidence-build.md) for details.
