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

### `nsys-ai skill add <path.md>`

Register a custom skill from a markdown file. The file must contain
a fenced SQL code block and frontmatter with `name`, `description`, `category`.
See `skills/SKILL_TEMPLATE.md` for the format.

### `nsys-ai skill remove <name>`

Unregister a previously added custom skill.

### `nsys-ai skill save`

Export a skill definition to a markdown file.

```bash
nsys-ai skill save <name> -o <output.md>
```

---

## Builtin Skills Catalog

All skills support the `--trim` flag for time-range restriction.
Parameters marked **required** must be provided via `--param`.

---

### Category: `kernels`

| Skill | Title | Description |
|-------|-------|-------------|
| `top_kernels` | Top GPU Kernels by Total Time | Lists the heaviest GPU kernels ranked by cumulative execution time. Use to identify hotspots. |
| `kernel_launch_overhead` | Kernel Launch Overhead | Measures the gap between CUDA Runtime API call and GPU kernel execution. High overhead = CPU-side bottleneck. |
| `kernel_launch_pattern` | Kernel Launch Pattern Analysis | Per-stream dispatch rate, burst density, inter-launch gaps, and sync-stall detection. |
| `stream_concurrency` | Stream Concurrency Analysis | Multi-stream GPU concurrency: active streams, kernel packing, compute/memory overlap. |
| `gpu_idle_gaps` | GPU Idle Gaps (Bubbles) | Finds idle gaps between consecutive GPU kernels with aggregation stats and CPU attribution. |
| `region_mfu` | Region-Level MFU | Computes MFU for an NVTX region or kernel. Requires `name` and `theoretical_flops`. |
| `theoretical_flops` | Theoretical FLOPs Calculator | Exact FLOPs for transformer operations (attention, mlp, full_model, etc.). LLMs should use this instead of manual multiplication. |

#### `top_kernels` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|:--------:|---------|-------------|
| `limit` | int | | 15 | Max number of kernels to return |

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
| `hidden_dim` | int | | 0 | Model hidden dimension (H) |
| `seq_len` | int | | 0 | Sequence length (S) |
| `num_layers` | int | | 1 | Number of transformer layers |
| `ffn_dim` | int | | 4×H | FFN intermediate dimension |
| `batch_size` | int | | 1 | Batch size |
| `multiplier` | int | | 1 | 1=fwd, 3=fwd+bwd, 4=fwd+bwd+ckpt |
| `M` / `N` / `K` | int | | 0 | For `linear` operation: matrix dimensions |

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
| `nccl_breakdown` | NCCL Collective Breakdown | Summarizes NCCL ops (AllReduce, AllGather, etc.) by type with count, time, variability. |
| `nccl_anomaly` | NCCL Anomaly Detection | Detects outlier NCCL operations exceeding a threshold relative to their op type average. |
| `overlap_breakdown` | Compute/Communication Overlap | Quantifies compute vs NCCL overlap. `overlap_pct > 60%` = NCCL well-hidden. |

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
| `schema_inspect` | Database Schema Inspector | Lists all tables and columns in the Nsight SQLite database. Run this first to understand available data. |

No required parameters.

---

## Skill Categories Quick Reference

| Category | Count | What it covers |
|----------|:-----:|----------------|
| `kernels` | 7 | GPU kernel timing, launch overhead, MFU, FLOPs |
| `memory` | 3 | H2D/D2H transfers, bandwidth, distribution |
| `communication` | 3 | NCCL breakdown, anomalies, overlap |
| `nvtx` | 3 | NVTX→kernel mapping, layer breakdown, iterations |
| `system` | 2 | CPU→GPU pipeline, thread utilization |
| `analysis` | 2 | Root cause patterns, speedup estimates |
| `utility` | 1 | Schema inspection |
| **Total** | **21** | |

> **Note**: `memory_transfers.py` registers 2 skills (`memory_transfers` + `h2d_distribution`).
> There are exactly 21 unique Python builtin skills derived from 20 modules.
