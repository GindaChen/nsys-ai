# NCCL Tracing

> **Source:** [Nsight Systems User Guide — NVIDIA NCCL Trace](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#nvidia-nccl-trace)

---

## Overview

Nsight Systems provides two methods for tracing NCCL (NVIDIA Collective Communications Library) operations:

### 1. Legacy NCCL Tracing
- Based on NVTX annotations within NCCL itself
- Enabled by default when NVTX tracing is active
- Traces API calls on the CPU
- Provides limited GPU-projection of ranges in the GUI
- Available in all recent Nsight Systems versions

### 2. Advanced NCCL Tracing (Nsight Systems 2025.6.1+)
- More detailed tracing mechanism
- Requires **NCCL ≥ 2.28** (limited support for 2.27.4+)
- Copy Engine (CE) collectives require **NCCL ≥ 2.29**
- Provides detailed GPU operations and async runtime scheduling
- Enhances event correlation
- Less precise CPU API timestamps compared to legacy

## CLI Usage

```bash
# Basic NCCL tracing (legacy, via NVTX)
nsys profile --trace=cuda,nvtx,nccl <app>

# Multi-GPU profiling with MPI
mpirun -np 4 nsys profile \
  --trace=cuda,nvtx,nccl \
  -o rank_%q{OMPI_COMM_WORLD_RANK} \
  python train.py

# With torchrun
torchrun --nproc-per-node=4 \
  nsys profile --trace=cuda,nvtx,nccl \
  -o rank_%q{LOCAL_RANK} \
  train.py
```

## NCCL Execution Model

A NCCL collective operation has multiple steps:

1. **Application calls NCCL API** (e.g., `ncclAllReduce`)
2. **NCCL runtime schedules** the operation in GPU queues
3. **CUDA kernel launched**
4. **Operation executes** within the CUDA kernel

### Group Operations
- Operations within `ncclGroupStart`...`ncclGroupEnd` are typically fused into a single CUDA kernel per rank/device
- With legacy tracing, `ncclGroupEnd` is projected to the fused kernel on the GPU
- Implicit groups exist for each individual API call (not shown in legacy tracing)

### Blocking vs Non-blocking Communicators
- **Blocking:** All operations happen on the calling thread
- **Non-blocking:** CUDA calls happen in different threads — legacy tracing can't track cross-thread operations, but advanced tracing can

### CUDA Graph Capture
- API calls and kernel launches are captured once
- Runtime scheduling and GPU operations happen once per graph launch

## Advanced Tracing Details

When using advanced NCCL tracing (2025.6.1+), the trace shows:

### API Calls
- `API Group` range spans all API calls within a group
- Individual API calls shown below the group range
- `nccl` prefix omitted (e.g., `ncclAllReduce` → `AllReduce`)
- `GroupLaunch` range = `ncclGroupEnd` function call
- `KernelLaunch` ranges show actual CUDA kernel launches

### Runtime Scheduling
- `GroupRuntime` ranges show CPU-side scheduling for GPU operations
- Can occur:
  - At end of API Group (blocking communicators)
  - In separate thread (non-blocking communicators)
  - In host function on special thread (graph launches)

## Key Analysis Patterns

### NCCL Overlap Analysis
The most important distributed training metric: how much of NCCL communication overlaps with computation?

```python
# Using nsys recipes
nsys recipe nccl_gpu_overlap_trace -i report.nsys-rep
```

### Straggler Detection
Compare per-collective timing across ranks to find slow GPUs:

```sql
-- Extract NCCL events from NVTX
SELECT text, start, end, (end - start) / 1e6 AS duration_ms,
       globalTid
FROM NVTX_EVENTS
WHERE text LIKE '%AllReduce%' OR text LIKE '%AllGather%'
ORDER BY start;
```

### Communication-Computation Ratio
For each training step, measure:
- Total GPU compute time (kernels NOT from NCCL)
- Total NCCL kernel time
- Overlap between the two

## Common NCCL Collective Operations

| Operation | Description |
|-----------|-------------|
| `AllReduce` | Sum/max/min across all ranks, result to all |
| `AllGather` | Gather from all ranks, result to all |
| `ReduceScatter` | Reduce + scatter across ranks |
| `Broadcast` | One rank sends to all |
| `Send`/`Recv` | Point-to-point communication |
| `AllToAll` | Full exchange between all ranks |

## Multi-Rank Profiling Tips

1. **Use `%q{RANK}` in output filename** to separate per-rank reports
2. **Multi-report timeline view** in GUI can align ranks for comparison
3. **Profile 2-3 iterations only** — NCCL patterns are repetitive
4. **Look for skew** — if one rank starts a collective later than others, that's the bottleneck
