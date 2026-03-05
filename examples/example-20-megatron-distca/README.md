# Example 20 — Megatron-LM DistCA Profiling

Analyze a real Megatron-LM Transformer Engine training profile captured on 8× NVIDIA H200 GPUs.

**Profile:** `baseline.t128k.host-fs-mbz-gpu-899` from [GindaChen/nsys-hero](https://huggingface.co/datasets/GindaChen/nsys-hero)

---

## Quick Start

### Step 1: Install nsys-ai

```bash
pip install nsys-ai
```

### Step 2: Download the profile

```bash
python download_data.py
```

This downloads the `.sqlite` profile (~700 MB) from HuggingFace into `output/megatron_distca.sqlite`.

### Step 3: Explore the profile

Run these commands in order to explore the profile:

```bash
# 3a. Profile overview — see GPUs, tables, time range
nsys-ai info output/megatron_distca.sqlite

# 3b. Kernel summary — top kernels by GPU time
nsys-ai summary output/megatron_distca.sqlite --gpu 4

# 3c. NVTX tree — hierarchical view of one training iteration
nsys-ai tree output/megatron_distca.sqlite --gpu 4 --trim 39 42

# 3d. Compute / NCCL overlap analysis
nsys-ai overlap output/megatron_distca.sqlite --gpu 4 --trim 39 42
```

### Step 4: Interactive TUIs

```bash
# 4a. Timeline TUI — Perfetto-style horizontal timeline
#     Keys: ←→ pan, ↑↓ select stream, +/- zoom, Tab snap to kernel
nsys-ai timeline output/megatron_distca.sqlite --gpu 4 --trim 39 42

# 4b. Tree TUI — interactive NVTX hierarchy browser
#     Keys: ↑↓ navigate, Enter expand, / search, q quit
nsys-ai tui output/megatron_distca.sqlite --gpu 4 --trim 39 42
```

### Step 5: Web UI & Exports

```bash
# 5a. Web viewer — opens interactive HTML in browser
nsys-ai web output/megatron_distca.sqlite --gpu 4 --trim 39 42

# 5b. Timeline web viewer
nsys-ai timeline-web output/megatron_distca.sqlite --gpu 4 --trim 39 42

# 5c. Perfetto trace — opens in ui.perfetto.dev
nsys-ai perfetto output/megatron_distca.sqlite --gpu 4 --trim 39 42

# 5d. Export HTML report
nsys-ai viewer output/megatron_distca.sqlite --gpu 4 --trim 39 42 -o output/report.html

# 5e. Export CSV for scripting
nsys-ai export-csv output/megatron_distca.sqlite --gpu 4 --trim 39 42 -o output/kernels.csv
```

---

## What You'll See

This profile contains a Megatron-LM training run with Transformer Engine on 8 GPUs. Key things to look for:

- **FlashAttention kernels** (`flash_fwd_splitkv_kernel`) — the dominant GPU kernels
- **NCCL collectives** — AllReduce operations between training iterations
- **NVTX hierarchy** — `Iteration` → `forward`/`backward` → `TransformerLayer` → individual ops
- **Compute vs communication overlap** — how well GPU compute overlaps with NCCL

### Recommended GPU & Time Range

- **GPU 4** — good representative GPU with clear NVTX annotations
- **Trim 39–42** — captures ~3 complete training iterations
- **Trim 39.98–40.5** — zoomed into a single transformer layer forward pass

---

## Files

| File | Purpose |
|------|---------|
| `download_data.py` | Downloads `.sqlite` from HuggingFace |
| `benchmark_timeline_web.py` | Benchmarks timeline-web cache/tile phases |
| `timeline_web_perf_budget.json` | Performance budget for regression checks |
| `.gitignore` | Ignores `output/` directory |
| `output/` | Downloaded profile data (gitignored) |

## Benchmark & Regression

Run benchmark locally (prints JSON timings):

```bash
python benchmark_timeline_web.py --runs 1
```

Run with budget enforcement:

```bash
python benchmark_timeline_web.py --check
```

Run pytest regression test:

```bash
pytest tests/test_timeline_web_distca_benchmark.py -q
```
