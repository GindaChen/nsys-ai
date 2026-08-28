# Example 02 — FastVideo Inference Profiling

Profile FastVideo (FastWan2.1-T2V-1.3B) video generation inference on 4× H100 GPUs using Modal.

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

> **Note:** This example requires a pre-captured profile, and one is not published on HuggingFace
> yet — `download_data.py` will report the failure rather than produce a file. Capture your own with
> the Modal profiling script below (requires a Modal account + GPU access), or start with
> [example 20](../example-20-megatron-distca/), whose capture does download.

### Step 3: Explore the profile

```bash
# Profile overview
nsys-ai info output/fastvideo_inference.sqlite

# Kernel summary — see which kernels dominate inference
nsys-ai summary output/fastvideo_inference.sqlite --gpu 0

# NVTX tree — hierarchical view of the inference pipeline (--gpu and --trim required)
nsys-ai tree output/fastvideo_inference.sqlite --gpu 0 --trim 0 5

# Interactive timeline TUI (--trim required)
nsys-ai timeline output/fastvideo_inference.sqlite --gpu 0 --trim 0 5
```

### Step 4: Web UI & Exports

```bash
# Web viewer
nsys-ai web output/fastvideo_inference.sqlite --gpu 0

# Chrome Trace Event JSON — open the written file in ui.perfetto.dev
nsys-ai export output/fastvideo_inference.sqlite --gpu 0 --trim 0 5 -o output/trace-export/
```

---

## Capturing Your Own Profile (Advanced)

If you have a Modal account with GPU access, you can capture fresh profiles:

```bash
# Install Modal
pip install modal
modal setup

# Profile FastVideo inference on 4× H100
modal run profile_inference.py
```

This will:
1. Install FastVideo in a Modal container
2. Run video generation under `nsys profile`
3. Export to `.sqlite`
4. Download the result to `output/`

---

## Profile Details

- **Model:** FastWan2.1-T2V-1.3B-Diffusers (1.3B params)
- **GPUs:** 4× H100 (Modal)
- **Workload:** Single 5-second video generation
- **Tracing:** `nsys profile --trace cuda,nvtx`

---

## Files

| File | Purpose |
|------|---------|
| `download_data.py` | Downloads profile from HuggingFace |
| `output/` | Downloaded/captured profile data (gitignored) |
| `README.md` | This file |
