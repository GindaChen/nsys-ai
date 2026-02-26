# nsys CLI Reference

> **Source:** [Nsight Systems User Guide — Profiling from the CLI](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#profiling-from-the-cli)

---

## Command Structure

```
nsys [global_option]
nsys [command_switch] [optional options] [application] [optional app_options]
```

All options are **case-sensitive**. Short options use a space (`-s process-tree`), long options use `=` (`--sample=process-tree`).

## Global Options

| Short | Long | Description |
|-------|------|-------------|
| `-h` | `--help` | Print help |
| `-v` | `--version` | Print version |

## Core Commands

| Command | Description |
|---------|-------------|
| `profile` | Collect a profile of the target application |
| `launch` | Launch application in paused state for later attachment |
| `start` | Start collection on a running session |
| `stop` | Stop collection on a running session |
| `cancel` | Cancel a running collection without saving |
| `shutdown` | Shut down a session |
| `export` | Export `.nsys-rep` to other formats (SQLite, JSON) |
| `stats` | Generate stats from `.nsys-rep` files |
| `analyze` | Run automated analysis |
| `recipe` | Run predefined analysis recipes |
| `sessions` | List active sessions |

## Essential `profile` Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--trace` / `-t` | `cuda,opengl,nvtx,osrt` | Trace APIs to collect. Options: `cuda`, `nvtx`, `osrt`, `cublas`, `cudnn`, `nccl`, `mpi`, `opengl`, `vulkan`, `syscall`, `python-gil` |
| `--output` / `-o` | `report#` | Output file path (without extension) |
| `--duration` / `-d` | unlimited | Collection duration in seconds |
| `--delay` / `-y` | 0 | Delay before collection starts (seconds) |
| `--sample` / `-s` | `cpu` | CPU sampling mode: `cpu`, `process-tree`, `system-wide`, `none` |
| `--cpuctxsw` | `process-tree` | CPU context switch tracing |
| `--capture-range` / `-c` | `none` | Capture range trigger: `none`, `cudaProfilerApi`, `nvtx`, `hotkey` |
| `--capture-range-end` | `stop` | What happens when capture range ends |
| `--stats` | false | Print stats summary after collection |
| `--force-overwrite` | false | Overwrite existing output files |
| `--kill` | `sigterm` | How to handle target process at end: `sigterm`, `sigkill`, `none` |
| `--gpu-metrics-device` | `none` | GPU metrics collection: device index or `all` |
| `--cudabacktrace` | `none` | CUDA backtrace collection: `all`, `kernel`, `memory`, `sync`, `none` |
| `--pytorch` | none | PyTorch profiling: `autograd-nvtx`, `autograd-shapes-nvtx`, `functions-trace` |
| `--python-backtrace` | `none` | Python backtrace sampling |
| `--python-sampling` | `none` | Python sampling frequency |
| `--run-as` | none | Run target as specified user |
| `-e` | none | Set environment variable: `-e KEY=VALUE` |
| `-w` / `--show-output` | true | Show target's stdout/stderr |

## Example Commands

### Version check
```bash
nsys -v
```

### Default analysis run
```bash
nsys profile <application> [app-args]
```
Traces CUDA, OpenGL, NVTX, OSRT. Collects CPU sampling + thread scheduling.

### Minimal trace (fast, focused)
```bash
nsys profile --trace=cuda,nvtx -d 20 --sample=none --cpuctxsw=none -o my_test <app> [args]
```
CUDA + NVTX only, 20 second duration, no CPU sampling.

### Delayed start
```bash
nsys profile -y 20 <app> [args]
```
Wait 20 seconds before collecting (skip warmup).

### NVTX-triggered capture range
```bash
nsys profile -c nvtx -w true -p MESSAGE@DOMAIN <app> [args]
```
Only collect data while the specified NVTX range is open.

### PyTorch with autograd annotations
```bash
nsys profile --pytorch=autograd-shapes-nvtx --trace=cuda,nvtx python train.py
```

### Full ML training profile
```bash
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas,nccl \
  --pytorch=autograd-shapes-nvtx \
  --cudabacktrace=all \
  --gpu-metrics-device=all \
  --stats=true \
  -d 60 \
  -o training_profile \
  python train.py
```

### MPI / Distributed (multi-process)
```bash
mpirun -np 4 nsys profile --trace=cuda,nvtx,nccl -o rank_%q{OMPI_COMM_WORLD_RANK} python train.py
```

### Export to SQLite
```bash
nsys export --type sqlite report.nsys-rep -o report.sqlite
```

### Generate stats
```bash
nsys stats report.nsys-rep
```

### Interactive session
```bash
nsys launch --trace=cuda,nvtx python train.py  # Launch paused
nsys start                                       # Start collection
nsys stop                                        # Stop collection
nsys shutdown                                    # End session
```

## Handling Application Launchers

When profiling with `mpirun`, `deepspeed`, or `torchrun`, wrap `nsys` around each process:

```bash
# Option 1: Use nsys with mpirun
mpirun -np N nsys profile --trace=cuda,nvtx,nccl -o rank_%q{OMPI_COMM_WORLD_RANK} <app>

# Option 2: Use --trace-fork-before-exec (deprecated in some versions)
nsys profile --trace-fork-before-exec=true mpirun -np N <app>
```

## Sessions

Multiple profiling sessions can run concurrently on the same system. A session starts with `start`/`launch`/`profile` and ends with `shutdown` or when `profile` terminates.

## Key Notes

- By default, launched processes are **terminated** when collection completes (unless `--kill none`)
- The `--stats` flag causes `nsys` to print a summary table after collection — useful for quick checks
- For Python applications, use `python -u` or `PYTHONUNBUFFERED=1` to avoid output buffering issues
- The `--capture-range=cudaProfilerApi` flag is the recommended way to profile specific iterations in training loops
