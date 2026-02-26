# Focused Profiling

> **Source:** [Nsight Systems User Guide — Preparing Your Application for Profiling](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#preparing-your-application-for-profiling)

---

## Why Focus Your Profile?

By default, Nsight Systems profiles the **entire application run**. This is wasteful for ML training because:
- Warmup iterations aren't representative
- Training loops are repetitive — profiling 1000 identical iterations is pointless
- Large profiles are slow to export and analyze

**Rule of thumb:** Profile 2-5 representative iterations after warmup.

## Method 1: `cudaProfilerApi` (Recommended for ML)

Use CUDA's built-in start/stop mechanism. The profiler instruments everything but only **saves data** during the marked range.

### Python/PyTorch Usage

```python
import torch

# Warmup
for step in range(10):
    train_step()
torch.cuda.synchronize()

# Profile these iterations
torch.cuda.cudart().cudaProfilerStart()
for step in range(3):
    train_step()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()

# Continue training (not profiled)
for step in range(1000):
    train_step()
```

### CLI Command
```bash
nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx python train.py
```

### How it Works
1. `nsys` starts and attaches to the process
2. Data collection is **paused** until `cudaProfilerStart()` is called
3. Data is collected until `cudaProfilerStop()` is called
4. Process continues running normally

## Method 2: NVTX Capture Range

Use an NVTX range to trigger collection:

### Python Usage
```python
import nvtx

for step in range(100):
    if step == 10:
        rng = nvtx.start_range("profiler", domain="training")

    train_step()

    if step == 15:
        nvtx.end_range(rng)
```

### CLI Command
```bash
nsys profile -c nvtx -p profiler@training python train.py
```

### Capture Range Specification

| Format | Meaning |
|--------|---------|
| `MESSAGE@DOMAIN` | Range with message in specific domain |
| `MESSAGE@*` | Range with message in any domain |
| `MESSAGE` | Range with message in default domain |

**Important:** By default, only NVTX **registered strings** are considered. For unregistered strings:
```bash
nsys profile -c nvtx -p message@domain -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 <app>
```

## Method 3: Duration + Delay

Simple but imprecise — good for quick checks:

```bash
# Skip first 30s of warmup, profile for 10s
nsys profile -y 30 -d 10 --trace=cuda,nvtx python train.py
```

## Method 4: Hotkey

For interactive applications:

```bash
nsys profile -c hotkey python train.py
# Press Ctrl+F11 to start, Ctrl+F11 again to stop
```

## Comparison

| Method | Precision | Code Changes | Best For |
|--------|-----------|-------------|----------|
| `cudaProfilerApi` | Exact iteration | Yes (minimal) | ML training loops |
| NVTX capture range | Exact range | Yes | Complex multi-phase apps |
| Duration + delay | Time-based | None | Quick checks |
| Hotkey | Manual | None | Interactive debugging |

## Recommendations for ML Training

```python
# Minimal boilerplate for focused profiling
import torch
import os

PROFILE_STEP = int(os.environ.get("PROFILE_STEP", 10))
PROFILE_COUNT = int(os.environ.get("PROFILE_COUNT", 3))

for step in range(total_steps):
    if step == PROFILE_STEP:
        torch.cuda.cudart().cudaProfilerStart()

    train_step()

    if step == PROFILE_STEP + PROFILE_COUNT:
        torch.cuda.cudart().cudaProfilerStop()
```

```bash
# Profile steps 10-12
nsys profile --capture-range=cudaProfilerApi \
  --pytorch=autograd-shapes-nvtx \
  --trace=cuda,nvtx \
  -o focused_profile \
  python train.py

# Profile steps 50-52 instead
PROFILE_STEP=50 nsys profile --capture-range=cudaProfilerApi ...
```
