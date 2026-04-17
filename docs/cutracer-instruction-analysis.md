# CUTracer Instruction-Level Analysis

This guide explains how to use the new `cutracer` workflow in `nsys-ai` to go from:

- "This kernel is slow"

to:

- "This kernel is memory-bound / compute-bound / sync-bound, and here is what to change next."

## Where this fits

Use `cutracer` after standard Nsight triage:

1. Find heavy kernels with `top_kernels` (or `nsys-ai summary`).
2. If the hot kernel is a custom kernel (for example `nvjet_*`, custom GEMM, Triton), run CUTracer.
3. Correlate instruction-level output back to NVTX + nsys timing via `cutracer_analysis`.

Do **not** start with CUTracer for everything. It is best used as a targeted drill-down tool.

## Installation

`cutracer` is now optional and intentionally not part of the core install.

```bash
# Core development install
pip install -e '.[dev]'

# Add CUTracer support
pip install -e '.[dev,cutracer]'
```

## Command overview

`nsys-ai` now provides:

- `nsys-ai cutracer check` — verify Python package + `.so` availability
- `nsys-ai cutracer install` — build/install `cutracer.so`
- `nsys-ai cutracer plan` — pick high-value kernels and generate a script
- `nsys-ai cutracer run` — execute local or Modal trace run
- `nsys-ai cutracer analyze` — parse traces and report bottlenecks

## End-to-end workflow (local GPU)

### 1) Verify environment

```bash
nsys-ai cutracer check
```

### 2) Generate an instrumentation plan

```bash
nsys-ai cutracer plan profile.sqlite --top-n 5
```

To generate a runnable script:

```bash
nsys-ai cutracer plan profile.sqlite --script > run_cutracer.sh
chmod +x run_cutracer.sh
```

### 3) Run instrumented workload

Edit `LAUNCH_CMD` in the generated script, then run:

```bash
./run_cutracer.sh ./cutracer_out_real
```

### 4) Analyze traces

```bash
nsys-ai cutracer analyze profile.sqlite ./cutracer_out_real
```

JSON output:

```bash
nsys-ai cutracer analyze profile.sqlite ./cutracer_out_real --format json
```

## Skill usage (agent or direct CLI)

The new skill is `cutracer_analysis`.

```bash
nsys-ai skill run cutracer_analysis profile.sqlite --format json -p trace_dir=./cutracer_out_real
```

Optional trim window (in ns):

```bash
nsys-ai skill run cutracer_analysis profile.sqlite --format json \
  -p trace_dir=./cutracer_out_real \
  -p trim_start_ns=0 \
  -p trim_end_ns=1000000000
```

## How to read results

- `MEMORY-BOUND`: prioritize data movement and memory access pattern fixes (tiling, reuse, prefetch).
- `COMPUTE-BOUND`: prioritize algorithmic flop reduction or mixed precision where valid.
- `SYNC-BOUND`: reduce barriers/synchronization hotspots and restructure reductions.
- `bank_conflict_hint: true`: inspect shared memory layout; consider padding per row.
- `tensor_core_active: false` on TC-eligible kernels: check dtype/alignment path.

## Modal workflow (no local GPU execution)

Generate or run a Modal app via:

```bash
nsys-ai cutracer run profile.sqlite --launch-cmd "python train.py" --backend modal
```

Or execute directly:

```bash
nsys-ai cutracer run profile.sqlite --launch-cmd "python train.py" --backend modal-run
```

## Agent guide / auto-guide note

- The canonical command is `nsys-ai agent-guide`.
- There is no separate `auto-guide` command right now.
- If your team internally says "auto-guide", map it to `agent-guide` in docs and scripts.

Recommended handoff pattern:

1. Run `nsys-ai agent-guide` to seed external agent context.
2. Run normal triage skills.
3. When kernel-level uncertainty remains, run `cutracer_analysis`.
4. Feed result JSON back to the agent for final recommendations.

## When this is complete enough to merge

Before merging, confirm:

- `nsys-ai cutracer plan` works on a real profile.
- `nsys-ai skill run cutracer_analysis ...` returns non-error output for a real trace directory.
- Tests pass for parser/planner/runner/installer/classifier/CLI/skills.
- Generated trace directories are ignored by git.
