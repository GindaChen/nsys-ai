<div align="center">

# nsys-ai

**AI-powered analysis for NVIDIA Nsight Systems profiles**

Navigate GPU kernel timelines, diff two runs, and diagnose performance
bottlenecks with an evidence-first agent — from your browser or terminal.

> **Mission:** Build an agent that understands GPU performance from first
> principles — one that can identify pipeline bubbles, calculate MFU, assess
> arithmetic intensity, and diagnose the root causes that cost millions of GPU
> hours, turning months of expert debugging into minutes.

[![CI](https://github.com/GindaChen/nsys-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/GindaChen/nsys-ai/actions)
[![PyPI](https://img.shields.io/pypi/v/nsys-ai)](https://pypi.org/project/nsys-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

nsys-ai reads `.nsys-rep` or `.sqlite` exports from
[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) and turns
them into something you can navigate and reason about: a web timeline, terminal
viewers, a before/after diff that reports whether a change actually helped, and
a set of deterministic analysis skills an LLM agent can drive. `.nsys-rep` files
are opened directly — nsys-ai exports them to SQLite for you on first use.

## Installation

```bash
pip install nsys-ai
```

No CUDA and no Nsight install are required to analyze a profile. Python 3.10+
only. (Capturing a new `.nsys-rep`, or converting one, needs the `nsys` CLI on
your machine; analyzing an existing `.sqlite` does not.)

## Quick start

### 1. Capture a profile

For ML training, capture a few representative iterations rather than the whole
run — it keeps the profile small and the profiler overhead low. Mark the region
with the CUDA profiler API and trace CUDA plus NVTX:

```python
import torch

for step in range(warmup):
    train_step()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for step in range(3):            # profile these iterations
    train_step()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
```

```bash
nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx \
  -o my_training python train.py
# -> my_training.nsys-rep
```

`--trace=cuda` is what every skill relies on (GPU kernels, memory copies, CUDA
API). `nvtx` adds the annotation hierarchy that drives the iteration, region,
and layer views. To use the iteration tools (`iters`, `diff --iteration`),
annotate each step with a consistent NVTX marker — see
[Focused Profiling](docs/08-focused-profiling.md) and
[NVTX Annotations](docs/03-nvtx-annotations.md).

No workload handy? Download an example profile:

```bash
cd examples/example-20-megatron-distca && python download_data.py
# -> output/megatron_distca.nsys-rep
```

### 2. Open it

```bash
# Default: open the web timeline in your browser
nsys-ai my_training.nsys-rep

# Metadata and GPU info
nsys-ai info my_training.nsys-rep

# GPU kernel summary
nsys-ai summary my_training.nsys-rep --gpu 0
```

Prefer the terminal? The TUIs work the same way:

```bash
nsys-ai timeline my_training.nsys-rep --gpu 0   # Perfetto-style horizontal timeline
nsys-ai tui my_training.nsys-rep --gpu 0        # NVTX tree browser
```

### 3. Compare two runs

```bash
nsys-ai diff before.sqlite after.sqlite
```

## Web timeline

A browser-based multi-GPU viewer with progressive rendering — no `--trim`
required. This is the default view when you run `nsys-ai <profile>`.

```bash
nsys-ai my_training.nsys-rep                       # opens in your browser
nsys-ai timeline-web my_training.nsys-rep --gpu 0 1 2 3
```

- Multi-GPU stacked view with color-coded separators
- Progressive rendering — pre-builds the NVTX tree at startup, then serves tiles
  in about a millisecond each
- NVTX hierarchy bars (L0-L5) per GPU
- AI chat sidebar (press `a`) and kernel search (press `/`)

| Input | Action |
|:-----:|--------|
| Swipe / `h` `l` / arrows | Pan through time |
| Swipe up-down / `j` `k` | Select stream |
| Pinch / `Shift+scroll` / `+` `-` | Zoom |
| `f` or `0` | Fit full time range |
| `Tab` | Next kernel |
| `/` | Search kernels |
| `n` | Toggle NVTX |
| `a` | AI chat |
| `?` | Help overlay |

## Timeline TUI

A Perfetto-style horizontal viewer with per-stream kernels, NVTX hierarchy
bars, and a time-cursor navigation model.

| Key | Action |
|:---:|--------|
| arrows | Pan time / select stream |
| `Shift+arrows` | Page pan (quarter viewport) |
| `Tab` | Snap to next kernel |
| `+` `-` | Zoom |
| `/` | Filter kernels by name |
| `m` | Minimum-duration threshold |
| `d` | Toggle demangled names |
| `B` | Save bookmark (with kernel + NVTX context) |
| `C` | Config panel (stream rows, tick density, NVTX depth) |
| `h` | Full help overlay |

## Profile diff

Comparing two profiles is the point of nsys-ai: it reports not just what changed
but whether the change is a likely regression or improvement.

```bash
# Terminal report
nsys-ai diff before.sqlite after.sqlite

# Interactive side-by-side web comparison
nsys-ai diff-web before.sqlite after.sqlite

# A specific device or time window
nsys-ai diff before.sqlite after.sqlite --gpu 0 --trim 39 42

# Compare one aligned iteration
nsys-ai diff before.sqlite after.sqlite --iteration 0

# Markdown (for a PR or issue) or JSON (for scripting)
nsys-ai diff before.sqlite after.sqlite --format markdown -o diff.md
nsys-ai diff before.sqlite after.sqlite --format json

# Gate CI: exit non-zero when the verdict is a likely regression
nsys-ai diff before.sqlite after.sqlite --exit-on-regression
```

The report covers top regressions and improvements, new and removed kernels,
NVTX region deltas, compute/NCCL overlap and idle changes, and a step-time
category rollup (compute / communication / idle). With `--format json` it adds a
top-level `verdict`, a `comparability_confidence` score, and a stable
content-derived `profile_id` per side. With no `--gpu`, the diff aggregates
across every device.

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu N` | all GPUs | Restrict to one device |
| `--trim START END` | full span | Compare only this window (seconds) |
| `--iteration N` | — | Compare one aligned iteration (needs an NVTX marker) |
| `--format` | `terminal` | `terminal` \| `markdown` \| `json` |
| `--limit N` | 15 | Top regressions/improvements to show |
| `--sort` | `delta` | `delta` \| `percent` \| `total` |
| `--exit-on-regression` | — | Exit 1 when the verdict is `regression_likely` |

## Commands

| Command | Description |
|---------|-------------|
| `open` | Quick-open a profile in the web UI, Perfetto, or TUI |
| `timeline-web` | Web multi-GPU timeline (progressive rendering) |
| `timeline` | Timeline TUI |
| `tui` | NVTX tree TUI |
| `web` | Web viewer server |
| `info` | Profile metadata and GPU hardware |
| `summary` | Top kernels and stream breakdown |
| `analyze` | Full auto-report (`--format json` emits evidence findings) |
| `overlap` | Compute / NCCL overlap analysis |
| `nccl` | NCCL collective breakdown |
| `iters` | Auto-detect training iterations |
| `tree` / `markdown` | NVTX hierarchy as text / markdown |
| `search` | Search kernels and NVTX by name |
| `report` | Generate a performance report |
| `diff` | Before/after profile comparison |
| `diff-web` | Side-by-side comparison web viewer |
| `chat` | AI chat TUI for a profile |
| `ask` | One-shot AI question about a profile |
| `agent` | Agent auto-analysis (`analyze`, `ask`) |
| `skill` | List and run analysis skills |
| `evidence` | Build evidence findings for the timeline overlay |
| `root-cause` | Browse and submit root-cause patterns |
| `cutracer` | Instruction-level drill-down (`check`, `install`, `plan`, `run`, `analyze`) |
| `export` / `export-csv` / `export-json` | Perfetto JSON, flat CSV, flat JSON |
| `viewer` / `timeline-html` | Interactive HTML report / timeline |
| `perfetto` | Open the trace in the Perfetto UI |

Run `nsys-ai <command> --help` for flags.

## Skills

Skills are self-contained analysis units that run without an LLM. nsys-ai ships
37 of them (kernels, memory, NCCL/communicators, NVTX, MFU, idle, root-cause,
profile health, and more).

```bash
nsys-ai skill list                                 # full catalog
nsys-ai skill run top_kernels profile.sqlite
nsys-ai skill run nccl_breakdown profile.sqlite
nsys-ai skill run profile_health_manifest profile.sqlite --format json
```

A few common ones:

| Skill | What it does |
|-------|-------------|
| `top_kernels` | Heaviest GPU kernels by total time |
| `gpu_idle_gaps` | Pipeline bubbles between kernels |
| `memory_transfers` | H2D / D2H / D2D transfer breakdown |
| `nccl_breakdown` | NCCL collective summary by type |
| `nccl_communicator_analysis` | Per-communicator NCCL topology and efficiency |
| `overlap_breakdown` | Compute / communication overlap |
| `kernel_launch_overhead` | CPU-to-GPU dispatch latency |
| `region_mfu` | Model FLOPs utilization for an NVTX region |
| `profile_health_manifest` | One-shot health summary (run this first) |

Skills are extensible — add one by dropping a Python file that exports a `SKILL`
constant. See [`skill list`](docs/agent_skills/commands/skill.md) for the full
catalog.

## AI analysis (optional)

The agent is a CUDA performance expert that runs the skills and cites the
evidence — kernel names, durations, timestamps — behind each diagnosis rather
than guessing.

```bash
nsys-ai agent analyze profile.sqlite
nsys-ai agent ask profile.sqlite "why are there bubbles in the pipeline?"
nsys-ai ask profile.sqlite "is NCCL overlapping with compute?"
nsys-ai chat profile.sqlite                        # interactive chat TUI
```

The AI features need a provider API key. Set one of:

```bash
export ANTHROPIC_API_KEY=...      # or
export OPENAI_API_KEY=...         # or
export GEMINI_API_KEY=...
export NSYS_AI_MODEL=...          # optional: pick a specific model
```

Install the dependencies with the `agent` extra:

```bash
pip install 'nsys-ai[agent]'
```

If no key is set, the agent still runs the deterministic skills and prints their
results; it just skips the natural-language synthesis.

## Claude Code plugin

nsys-ai ships as a [Claude Code](https://claude.com/claude-code) plugin: the
`/nsys-ai` slash command turns a profile into a root cause, a proposed fix, and
an annotated timeline. See
[docs/claude-plugin-quickstart.md](docs/claude-plugin-quickstart.md) to install
and [docs/claude-plugin.md](docs/claude-plugin.md) for the full reference.

## Documentation

The `docs/` directory mirrors the relevant NVIDIA Nsight Systems reference
(capture, schema, NVTX, CUDA/NCCL trace) plus nsys-ai project guides:

| Guide | Topic |
|-------|-------|
| [NVIDIA nsys CLI](docs/01-cli-reference.md) | The upstream `nsys` profiler CLI (capture-time) |
| [SQLite schema](docs/02-sqlite-schema.md) | Nsight export tables and queries |
| [NVTX annotations](docs/03-nvtx-annotations.md) | Annotating your code (and iteration markers) |
| [CUDA trace](docs/04-cuda-trace.md) | GPU kernel and memory tracing |
| [NCCL tracing](docs/05-nccl-tracing.md) | Multi-GPU collective analysis |
| [Python / PyTorch](docs/06-python-pytorch.md) | Profiling PyTorch workloads |
| [Containers](docs/07-container-profiling.md) | Profiling inside Docker / Slurm |
| [Focused profiling](docs/08-focused-profiling.md) | Capturing representative iterations |
| [CUTracer](docs/cutracer-instruction-analysis.md) | Instruction-level drill-down for top kernels |

The [`docs/sqlite-explorer/`](docs/sqlite-explorer/) directory holds an
interactive HTML explorer for the Nsight SQLite schema — open
`docs/sqlite-explorer/index.html` in a browser.

## Install tiers

```bash
pip install nsys-ai              # core: CLI, TUIs, skills, web/diff viewers
pip install 'nsys-ai[agent]'     # + LLM-backed agent (anthropic + litellm)
pip install 'nsys-ai[chat]'      # + chat TUI
pip install 'nsys-ai[cutracer]'  # + CUTracer instruction-level workflow
pip install 'nsys-ai[all]'       # everything
```

The `ai` extra is kept as an alias of `agent` for backward compatibility.

## Development

```bash
git clone https://github.com/GindaChen/nsys-ai.git
cd nsys-ai
pip install -e '.[dev]'
pytest tests/ -v
```

**Guided optimization loop** (diagnose → propose → re-profile → diff → accept): see [docs/guided-loop-setup.md](docs/guided-loop-setup.md).

---

## License

MIT — see [LICENSE](LICENSE).

<div align="center">
<sub>Built for GPU performance engineers.</sub>
</div>
