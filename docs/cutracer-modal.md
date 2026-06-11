# Running CUTracer on Modal

This guide shows how to run CUTracer instruction-level kernel analysis on
[Modal](https://modal.com) — a serverless GPU platform — when you do **not**
have a local NVIDIA GPU.

It complements [cutracer-instruction-analysis.md](./cutracer-instruction-analysis.md),
which covers the local-GPU workflow and how to read the results. Read that
first if you are new to the `cutracer` subcommand.

## When to use the Modal backend

Use Modal when **all** of these are true:

- Nsight triage (`top_kernels` / `nsys-ai summary`) points at a heavy custom
  kernel — a Triton kernel, a hand-written GEMM, `nvjet_*`, etc. — that the
  high-level trace cannot explain.
- You want the SASS-level instruction mix (memory- vs compute- vs sync-bound),
  which requires re-running the workload under instrumentation.
- You have no local GPU, or you want to instrument on a different GPU
  architecture than the one on your machine.

CUTracer is a **targeted drill-down**, not a first pass. Do not Modal-trace a
whole training run — instrument a short, reproducible slice (see
[Keep the run short](#keep-the-run-short)).

## How it works

`nsys-ai cutracer run ... --backend modal` does not call Modal directly.
It **generates a complete, self-contained Modal app** (a Python file) from
your profile. The generated app:

1. Builds a CUDA devel image that clones and compiles the pinned CUTracer
   release (`cutracer.so`) into a **cached image layer** — the build only
   runs the first time, then Modal reuses the layer.
2. Mounts a `modal.Volume` so the output CSVs survive the container.
3. Runs `cutracer trace -- <your launch command>` on the GPU, filtered to the
   top-N kernels from your profile.
4. Post-processes the raw trace into histogram CSVs (resolving SASS opcodes
   inside the container, where `nvdisasm` is available).
5. Downloads the `*_hist.csv` files back to a local directory.

You then analyze them locally with `nsys-ai cutracer analyze`.

The generated script is intentionally human-readable and meant to be edited —
see [Add your training code](#add-your-training-code).

## Prerequisites

```bash
pip install nsys-ai modal      # nsys-ai generates the script; modal runs it
modal token new                # first time only — authenticates the Modal CLI
```

You do **not** need a local CUDA toolkit or `cutracer.so`: the `.so` is built
inside the Modal image. You only need a profile (`.sqlite` or `.nsys-rep`) to
pick the target kernels.

## The three Modal modes

| Command | What it does | Use when |
|---|---|---|
| `--backend modal` | Prints the generated app to stdout | You want to inspect or redirect it |
| `--modal-save FILE` | Writes the app to `FILE` (chmod +x) | **Recommended** — you will edit it |
| `--backend modal-run` | Writes a temp script and runs `modal run` immediately | Your launch command needs no extra code/deps in the image |

In almost all real cases you want **`--modal-save`**, because the generated
image does not yet contain your training code or its dependencies (see below).

## Quick start

```bash
# 1. Generate and save an editable Modal app from your profile.
nsys-ai cutracer run profile.sqlite \
  --launch-cmd "python train.py --steps 10" \
  --modal-save run_cutracer.py \
  --modal-gpu H100 \
  --top-n 5

# 2. Edit run_cutracer.py to add your training code + deps (see next section).

# 3. Run it on Modal.
modal run run_cutracer.py

# 4. Analyze the downloaded CSVs locally.
nsys-ai cutracer analyze profile.sqlite ./cutracer_out --format json
```

If your launch command genuinely needs nothing beyond `nsys-ai` and CUTracer
(rare — e.g. a self-contained microbenchmark installed via pip), you can skip
the edit step and run in one shot:

```bash
nsys-ai cutracer run profile.sqlite \
  --launch-cmd "python -m my_published_pkg.bench" \
  --backend modal-run --modal-gpu A100
```

## Add your training code

This is the step most people miss. The generated image installs `nsys-ai` and
`cutracer` and builds the `.so`, but it does **not** contain your `train.py` or
your training dependencies (PyTorch, your package, datasets, etc.). When the
container runs `cutracer trace -- python train.py`, that `train.py` must exist
inside the container.

Open the saved `run_cutracer.py` and edit the `image = (...)` block:

```python
image = (
    modal.Image.from_registry("nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "make", "g++", "curl", "libzstd-dev")
    .pip_install("nsys-ai", "cutracer>=0.2.0")
    # --- add your training dependencies ---
    .pip_install("torch", "transformers", "triton")
    # --- add your training code ---
    .add_local_dir(".", remote_path="/root")     # or .add_local_file("train.py", "/root/train.py")
    .run_commands(
        "git clone --depth=1 --branch v0.2.1 https://github.com/facebookexperimental/CUTracer /opt/CUTracer",
        "cd /opt/CUTracer && ./install_third_party.sh",
        "cd /opt/CUTracer && make -j$(nproc)",   # upstream's recommended build
        "mkdir -p /root/.nsys-ai/cutracer/lib && cp /opt/CUTracer/lib/cutracer.so /root/.nsys-ai/cutracer/lib/",
    )
)
```

Notes:

- `add_local_dir` / `add_local_file` are Modal's mechanisms for shipping local
  code into the image. See the [Modal images docs](https://modal.com/docs/guide/images).
- The container's working directory is `/root`; your `--launch-cmd` runs there.
  Make sure the path in your launch command matches where you mounted the code.
- If your code is an installed package, prefer `.pip_install(...)` over mounting.
- You can pre-bundle deps by setting the base image to your own training image
  instead of the stock CUDA devel image — just keep the `apt_install` packages
  (`git make g++ curl libzstd-dev`) and the CUTracer build commands.

### Match the CUDA toolkit to your framework

**This is the most common cause of empty/partial SASS output.** SASS resolution
runs `nvdisasm` (from the image's CUDA toolkit) on the cubins CUTracer dumps. If
your framework was built against a **newer** CUDA than the base image,
`nvdisasm` cannot parse those cubins and fails with:

```text
nvdisasm fatal : <kernel>.cubin is not a supported Elf file
```

When this happens the run still exits 0, but the affected kernel produces **no
histogram** — you silently get results only for kernels whose cubins the older
`nvdisasm` could read. In a real drill-down that often means your hot library
kernel (e.g. a cuBLAS `*_s16816gemm_*`) is the one that gets dropped.

A current `pip install torch` pulls **CUDA 13** wheels, which do **not** match
the stock `cuda:12.4.0-devel` base image. To keep them aligned, either:

- bump the base image to match (e.g. `nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu22.04`), or
- pin a framework build for the image's CUDA. On the stock 12.4 image, install a
  cu124 torch wheel:

  ```python
  .pip_install("torch", index_url="https://download.pytorch.org/whl/cu124")
  ```

  (Verified on Modal: with `torch==2.6.0+cu124` on the 12.4 image, a cuBLAS
  `ampere_*_s16816gemm_*` cubin disassembles cleanly — `tensor_core_active: true`,
  `HMMA`-dominated mix — whereas the default CUDA-13 torch fails `nvdisasm` on it.)

After a run, confirm the kernel you care about actually has a CSV:

```bash
ls cutracer_out/*_hist.csv      # the hot kernel's name should appear here
```

## GPU types and cost

`--modal-gpu` accepts any Modal GPU string. Common choices:

| `--modal-gpu` | Notes |
|---|---|
| `H100` | Default. Fastest; highest $/hr. |
| `A100` | Good middle ground. |
| `A10G` | Cheapest; fine for small kernels / smoke runs. |
| `L4`, `L40S`, `T4` | Also supported by Modal. |

CUTracer instrumentation adds roughly **1.5×** runtime overhead. A rough cost
estimate, where `span_s` is your instrumented run's wall time:

```text
gpu_seconds = span_s × 1.5
cost_usd    = gpu_seconds × per_second_rate
```

Approximate per-second rates (verify against
[Modal's current pricing](https://modal.com/pricing)):

```text
H100  ≈ $0.00127/s  (~$4.56/hr)
A100  ≈ $0.00071/s  (~$2.55/hr)
A10G  ≈ $0.00031/s  (~$1.10/hr)
```

Because you pay per GPU-second, keeping the instrumented run short matters more
than the GPU choice.

## Keep the run short

You pay per GPU-second, so instrument only enough to capture the hot kernel's
instruction mix — a few iterations, not the whole job. Levers, most effective first:

1. **Short launch command.** Make `--launch-cmd` stop early (e.g.
   `python train.py --steps 3`). This is the main control over how much is traced.
2. **Kernel filter (kept by default).** The generated `kernel_filter` is derived
   from your profile's top-N kernels — keep it (see the size warning below).
3. **Hard size cap.** `--trace-size-limit-mb N` makes CUTracer stop writing trace
   once the on-disk size reaches N MB (the running kernel is unaffected).

```bash
nsys-ai cutracer run profile.sqlite \
  --launch-cmd "python train.py --steps 3" \
  --trace-size-limit-mb 1000 \
  --modal-save run_cutracer.py
```

`--trim START_S END_S` only restricts which kernels are picked from the
*profile*; it does not shorten the Modal run. `--max-iters` is **deprecated and a
no-op** — upstream CUTracer has no `CUTRACER_MAX_ITERS` variable, so use the levers
above instead.

> **The histogram analysis writes a large raw trace.** `proton_instr_histogram`
> emits an uncompressed per-instruction NDJSON file **per kernel, per iteration** —
> commonly **200–460 MB each**. Tracing all kernels for several iterations can pile
> up multiple GB on the Volume before post-processing. Keep the kernel filter, keep
> the run short, and set `--trace-size-limit-mb`.

## Output and analysis

The generated `local_entrypoint` downloads every `*_hist.csv` from the Modal
Volume into `--output-dir` (default `./cutracer_out`). Then:

```bash
# Human-readable report
nsys-ai cutracer analyze profile.sqlite ./cutracer_out

# JSON (for agents / scripting)
nsys-ai cutracer analyze profile.sqlite ./cutracer_out --format json

# Or run the skill directly
nsys-ai skill run cutracer_analysis profile.sqlite --format json \
  -p trace_dir=./cutracer_out
```

See [cutracer-instruction-analysis.md §How to read results](./cutracer-instruction-analysis.md#how-to-read-results)
for interpreting `bottleneck`, `instruction_mix_pct`, `bank_conflict_hint`,
`tensor_core_active`, and `top_stalls`.

## The Modal Volume

Output is written to a `modal.Volume` named `cutracer-histograms` by default
(override with `--modal-volume NAME`). The volume **persists across runs** and
CSVs accumulate in it. Implications:

- Re-running with the same volume keeps older CSVs around; clear the volume or
  use a fresh `--modal-volume` name per experiment if you want isolation.
- Inspect or clean it with the Modal CLI:

  ```bash
  modal volume ls cutracer-histograms
  modal volume rm cutracer-histograms <path>
  ```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no *_hist.csv files found in volume` | Training command failed, or wrote no traced kernels | Check the Modal run logs; confirm the launch command runs standalone; verify the kernel filter matched real kernels (`nsys-ai cutracer plan profile.sqlite`) |
| `nvdisasm fatal : <kernel>.cubin is not a supported Elf file` | Image CUDA toolkit is older than the toolkit your framework was built with | [Match the CUDA toolkit to your framework](#match-the-cuda-toolkit-to-your-framework) — bump the base image or pin a matching framework wheel |
| Got a CSV, but not for the kernel I targeted | That kernel's cubin failed SASS resolution (see the row above); the run still exits 0 with other kernels' CSVs | Fix the CUDA version mismatch, then re-run; confirm with `ls cutracer_out/*_hist.csv` |
| `analyze` shows `top_stalls: []` and no CPI | `proton_instr_histogram` does not emit a `cycles` column, so stall/CPI scoring is unavailable | Expected with histogram mode; use the instruction mix + `bottleneck` instead |
| `WARNING: cutracer.so not found ... kernel logger mode only` | The `.so` build did not land in the image | Confirm the `run_commands` build steps are intact and `libzstd-dev` is installed; rebuild the image (`modal run` re-triggers a layer rebuild if changed) |
| `python: command not found` / module import errors in the container | Your code or deps are not in the image | [Add your training code](#add-your-training-code) via `.add_local_dir` / `.pip_install` |
| `modal: command not found` | Modal CLI not installed | `pip install modal` |
| Auth / token errors | Not authenticated | `modal token new` |
| First run is slow (~5 min before training starts) | First-time CUTracer image build | Expected; the layer is cached and reused on subsequent runs |
| `bottleneck = "unknown"`, `total_instructions = 0` in analyze | Ran in logger mode (no `.so`) | Same as the `.so not found` row above |

## See also

- [cutracer-instruction-analysis.md](./cutracer-instruction-analysis.md) — local
  workflow, command overview, reading results.
- [Modal documentation](https://modal.com/docs) — images, GPU types, volumes.
- `nsys-ai cutracer run --help` — full flag reference.
