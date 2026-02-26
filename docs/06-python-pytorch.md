# Python & PyTorch Profiling

> **Source:** [Nsight Systems User Guide — Python Profiling](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#python-profiling) and [PyTorch Profiling](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#pytorch-profiling)

---

## Python Profiling Features

Nsight Systems provides several Python-specific profiling capabilities:

| Feature | CLI Flag | Description |
|---------|----------|-------------|
| Python backtrace sampling | `--python-backtrace=cuda` | Captures Python stack traces |
| Python functions trace | `--python-functions-trace=<file>` | Traces specific Python functions |
| Python GIL tracing | `--trace=python-gil` | Tracks GIL acquisition/contention |

### Important: Python Output Buffering

Python changes stdout buffering when its output is redirected (which Nsight Systems does). Use:
```bash
nsys profile python -u train.py
# or
PYTHONUNBUFFERED=1 nsys profile python train.py
```

## PyTorch Profiling

Nsight Systems can **automatically annotate** PyTorch operations without any code changes. Requires CPython ≥ 3.8.

### `--pytorch` Options

| Value | Effect |
|-------|--------|
| `autograd-nvtx` | Enable `torch.autograd.profiler.emit_nvtx(record_shapes=False)` |
| `autograd-shapes-nvtx` | Enable `torch.autograd.profiler.emit_nvtx(record_shapes=True)` — records tensor shapes |
| `functions-trace` | Annotate forward/backward/step operations with tensor shapes |

Options can be combined: `--pytorch=autograd-shapes-nvtx,functions-trace`

### Example Commands

```bash
# Basic PyTorch profiling with NVTX autograd
nsys profile --pytorch=autograd-nvtx --trace=cuda,nvtx python train.py

# Full PyTorch profiling with tensor shapes
nsys profile \
  --pytorch=autograd-shapes-nvtx,functions-trace \
  --trace=cuda,nvtx,osrt \
  --cudabacktrace=all \
  python train.py

# Profile specific training iterations using cudaProfilerApi
nsys profile \
  --capture-range=cudaProfilerApi \
  --pytorch=autograd-shapes-nvtx \
  --trace=cuda,nvtx \
  -o pytorch_profile \
  python train.py
```

### What `functions-trace` Captures

- Forward operations (`torch.nn.Module.forward`)
- Backward operations (`torch.autograd.backward`)
- Optimizer step operations (`optimizer.step`)
- Input/output tensor shapes

### Programmatic Control with cudaProfilerApi

The recommended way to profile specific training iterations:

```python
import torch

# Warm up
for step in range(10):
    train_step()

# Profile these iterations
torch.cuda.cudart().cudaProfilerStart()
for step in range(3):
    train_step()
torch.cuda.cudart().cudaProfilerStop()
```

Then profile with:
```bash
nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx python train.py
```

### Using NVTX for Custom Annotations

```python
import torch.cuda.nvtx as nvtx

def train_step(model, data, target, optimizer, criterion):
    nvtx.range_push("forward")
    output = model(data)
    nvtx.range_pop()

    nvtx.range_push("loss")
    loss = criterion(output, target)
    nvtx.range_pop()

    nvtx.range_push("backward")
    loss.backward()
    nvtx.range_pop()

    nvtx.range_push("optimizer_step")
    optimizer.step()
    optimizer.zero_grad()
    nvtx.range_pop()

    return loss.item()
```

## Dask Profiling

```bash
nsys profile --dask=functions-trace python dask_script.py
```

Annotates Dask functions and renames threads to 'Dask Worker' / 'Dask Scheduler'.

## Recommended ML Profiling Workflow

1. **Quick sweep** — broad tracing, few iterations:
   ```bash
   nsys profile --trace=cuda,nvtx -d 30 python train.py
   ```

2. **Targeted profile** — specific iterations with shapes:
   ```bash
   nsys profile \
     --capture-range=cudaProfilerApi \
     --pytorch=autograd-shapes-nvtx \
     --trace=cuda,nvtx,nccl \
     python train.py
   ```

3. **Deep dive** — full tracing with backtraces:
   ```bash
   nsys profile \
     --capture-range=cudaProfilerApi \
     --pytorch=functions-trace,autograd-shapes-nvtx \
     --trace=cuda,nvtx,osrt,cudnn,cublas,nccl \
     --cudabacktrace=all \
     --gpu-metrics-device=all \
     python train.py
   ```
