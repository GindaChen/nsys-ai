# CUDA Trace

> **Source:** [Nsight Systems User Guide — CUDA Trace](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-trace)

---

## Overview

CUDA tracing captures GPU kernel executions, memory operations, CUDA API calls, and CUDA graphs. It is enabled by default with `--trace=cuda`.

## What Gets Traced

| Category | Captured Data |
|----------|--------------|
| **CUDA Runtime API** | `cudaLaunch*`, `cudaMalloc*`, `cudaMemcpy*`, `cudaSync*`, etc. |
| **CUDA Driver API** | `cuLaunchKernel`, `cuMemAlloc`, etc. |
| **GPU Kernels** | Kernel name, duration, grid/block config, stream, device |
| **Memory Copies** | Direction (H2D/D2H/D2D/P2P), size, duration |
| **Memory Sets** | `cudaMemset` operations |
| **CUDA Graphs** | Graph creation, instantiation, and execution |
| **Unified Memory** | Page faults, migrations, throttling events |
| **CUDA Events** | `cudaEventRecord`, `cudaEventSynchronize` |

## Key CLI Options

```bash
# Basic CUDA trace
nsys profile --trace=cuda <app>

# CUDA + cuDNN + cuBLAS
nsys profile --trace=cuda,cudnn,cublas <app>

# With kernel backtraces (shows CPU call stack that launched each kernel)
nsys profile --trace=cuda --cudabacktrace=all <app>

# Specific backtrace types
nsys profile --cudabacktrace=kernel,memory,sync <app>
```

## GPU Memory Allocation Graph

Nsight Systems tracks CUDA memory allocations over time, showing:
- Allocation/deallocation events
- Peak memory usage
- Memory fragmentation patterns

## CUDA Graphs

CUDA Graph tracing shows:
- Graph node creation (which API calls created which nodes)
- Graph instantiation
- Graph execution (which kernels ran as part of which graph)

## Unified Memory

When enabled, tracks:
- Page fault events
- CPU → GPU and GPU → CPU page migrations
- Throttling events when migration bandwidth is saturated

## Python CUDA Backtrace

For Python applications, Nsight Systems can capture Python backtraces for CUDA API calls, showing the Python source location that triggered GPU operations.

Enable with: `--cudabacktrace=all --python-backtrace=cuda`

## Important SQLite Tables for CUDA Analysis

| Table | Key Columns |
|-------|-------------|
| `CUPTI_ACTIVITY_KIND_KERNEL` | `demangledName`→StringIds, `start`, `end`, `streamId`, `deviceId`, `gridX/Y/Z`, `blockX/Y/Z` |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | `nameId`→StringIds, `start`, `end`, `correlationId`, `globalTid` |
| `CUPTI_ACTIVITY_KIND_MEMCPY` | `copyKind`, `bytes`, `start`, `end`, `srcKind`, `dstKind` |
| `CUPTI_ACTIVITY_KIND_MEMSET` | `bytes`, `start`, `end`, `value` |

## Skipped CUDA Functions

By default, some high-frequency CUDA functions are **not traced** to reduce overhead. These include:
- `cudaGetDevice`, `cudaGetDeviceCount`
- `cudaGetLastError`, `cudaPeekAtLastError`
- `cudaDeviceGetAttribute`
- Various query functions

To trace all functions: use `--cuda-flush-interval=0` (increases overhead significantly).
