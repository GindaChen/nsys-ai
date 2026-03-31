<div align="center">

# 🔬 nsys-ai

**AI-powered analysis for NVIDIA Nsight Systems profiles**

Navigate GPU kernel timelines, diagnose performance bottlenecks with AI, and explore NVTX hierarchies — from your browser or terminal.

> **Mission:** Build an intelligent agent that truly understands GPU performance from first principles. An agent that can identify pipeline bubbles, calculate MFU, assess arithmetic intensity, and diagnose the root causes that cost millions of dollars in GPU hours — turning months of expert debugging into minutes.

[![CI](https://github.com/GindaChen/nsys-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/GindaChen/nsys-ai/actions)
[![PyPI](https://img.shields.io/pypi/v/nsys-ai)](https://pypi.org/project/nsys-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## ⚡ Install

```bash
pip install nsys-ai
```

No CUDA required. Just Python 3.10+. Core dependencies (`duckdb`, `rich`, `textual`) install automatically.

---

## 🌐 Web UI First (Default)

`nsys-ai` is web-first. The default command opens the timeline UI in your browser.

```bash
# Default: open web timeline UI
nsys-ai my_training.nsys-rep

# Explicit command (same web UI)
nsys-ai timeline-web my_training.nsys-rep
```

Use TUI/CLI modes when you specifically want terminal workflows.

---

## 🎯 What It Does

nsys-ai reads `.nsys-rep` or `.sqlite` profile exports from [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) and gives you a **web-first workflow** plus terminal and export tools:

<table>
<tr>
<td width="25%" align="center">

### 🌐 Web Timeline
Multi-GPU browser viewer with progressive rendering

</td>
<td width="25%" align="center">

### 🖥️ Timeline TUI
Perfetto-style horizontal timeline in your terminal

</td>
<td width="25%" align="center">

### 🌲 Tree TUI
Interactive NVTX hierarchy browser with kernel details

</td>
<td width="25%" align="center">

### 📄 HTML Export
Exportable interactive visualizations for sharing

</td>
</tr>
<tr>
<td>

Browser-based viewer:<br>
• Multi-GPU stacked streams<br>
• NVTX hierarchy bars<br>
• Pinch-to-zoom, trackpad pan<br>
• AI chat sidebar

</td>
<td>

```
S21 ████░██████░███
S56 ██████░░░███████
S60 ░░░██████░░░░░██
    |         │
    39.1s   39.5s
```

</td>
<td>

```
▼ Iteration (324ms)
  ▼ forward (180ms)
    ▼ Attention (89ms)
      ■ flash_fwd  26ms
      ■ flash_bwd  63ms
```

</td>
<td>

Interactive HTML exports:<br>
• NVTX stack viewer<br>
• SQLite schema explorer<br>
• Perfetto JSON traces

</td>
</tr>
</table>

---

## 🚀 Quick Start

### 1. Get a profile

```bash
# Option A: Profile your own PyTorch training
nsys profile -o my_training python train.py
# → produces my_training.nsys-rep  (or .sqlite via --export sqlite)

# Option B: Download an example profile
cd examples/example-20-megatron-distca
python download_data.py
# → downloads output/megatron_distca.nsys-rep
```

### 2. Explore it

```bash
# Start here: one command opens the web timeline in your browser
nsys-ai my_training.nsys-rep

# Or explicitly:
nsys-ai timeline-web my_training.nsys-rep

# Then use overview/summaries as needed
nsys-ai info my_training.nsys-rep

# GPU kernel summary
nsys-ai summary my_training.nsys-rep --gpu 0
```

> **Prefer a terminal?** nsys-ai also has full TUI support:
> ```bash
> nsys-ai timeline my_training.nsys-rep --gpu 0 --trim 39 42  # horizontal timeline
> nsys-ai tui my_training.nsys-rep --gpu 0 --trim 39 42       # tree browser
> ```

### 3. Export & share

```bash
# Perfetto JSON (open in ui.perfetto.dev)
nsys-ai export my_training.sqlite -o traces/

# Interactive HTML viewer
nsys-ai viewer my_training.sqlite --gpu 0 --trim 39 42 -o report.html

# Flat CSV/JSON for scripting
nsys-ai export-csv my_training.sqlite --gpu 0 --trim 39 42 -o kernels.csv
```

---

## 🌐 Web Timeline

The web timeline is a **browser-based multi-GPU viewer** with progressive rendering — no `--trim` required. This is the **default view** when you run `nsys-ai <profile>`.

```bash
# Just give it a profile — opens in your browser
nsys-ai my_training.nsys-rep

# Or explicitly with GPU selection:
nsys-ai timeline-web my_training.nsys-rep --gpu 0 1 2 3
```

### Features

- **Multi-GPU stacked view** — all GPUs shown simultaneously with color-coded separators
- **Progressive rendering** — pre-builds full NVTX tree at startup, then serves tiles instantly (~1ms per tile)
- **NVTX hierarchy** — layered bars (L0–L5) showing annotation nesting per GPU
- **AI chat sidebar** — press `A` to ask questions about the profile
- **Kernel search** — press `/` to search by kernel name

### Navigation

| Input | Action |
|:-----:|--------|
| **Swipe left/right** | Pan through time |
| **Swipe up/down** | Scroll through GPU streams |
| **Pinch** | Zoom in / out |
| `Shift+scroll` | Zoom in / out |
| `h` `l` or `←` `→` | Pan left / right |
| `j` `k` or `↑` `↓` | Select stream |
| `+` `-` | Zoom in / out |
| `f` or `0` | Fit all (full time range) |
| `Tab` | Next kernel |
| `/` | Search kernels |
| `n` | Toggle NVTX |
| `a` | AI Chat |
| `?` | Help overlay |

---

## ⌨️ Timeline TUI

Prefer working in the terminal? The timeline TUI is a **Perfetto-style** horizontal viewer with per-stream kernel visualization, NVTX hierarchy bars, and a time-cursor navigation model.

### Navigation

| Key | Action |
|:---:|--------|
| `←` `→` | Pan through time |
| `Shift+←/→` | Page pan (1/4 viewport) |
| `↑` `↓` | Select stream |
| `Tab` | Snap to next kernel |
| `+` `-` | Zoom in / out |
| `a` | Toggle absolute ↔ relative time |

### Analysis

| Key | Action |
|:---:|--------|
| `/` | Filter kernels by name |
| `m` | Set minimum duration threshold |
| `d` | Toggle demangled kernel names |
| `C` | Open config panel |
| `h` | Full help overlay |

### Bookmarks

| Key | Action |
|:---:|--------|
| `B` | Save bookmark (with kernel + NVTX context) |
| `'` | Bookmark list — press 1-9 to jump |
| `,` `.` | Cycle through bookmarks |
| `` ` `` | Jump back to previous position |
| `[` `]` | Set range start / end |

### Config Panel (`C`)

Tweak settings live with ↑/↓ to select and ←/→ to adjust:

- Selected stream rows (1-6)
- Other stream rows (1-4)
- Time tick density (2-20)
- NVTX depth levels (0-8)
- Min kernel duration filter

---

## 📚 Documentation

### Nsight Systems Guides

| Guide | Topic |
|-------|-------|
| [CLI Reference](docs/01-cli-reference.md) | Full `nsys` command reference |
| [SQLite Schema](docs/02-sqlite-schema.md) | Database tables & relationships |
| [NVTX Annotations](docs/03-nvtx-annotations.md) | Adding markers to your code |
| [CUDA Trace](docs/04-cuda-trace.md) | GPU kernel tracing |
| [NCCL Tracing](docs/05-nccl-tracing.md) | Multi-GPU collective analysis |
| [Python/PyTorch](docs/06-python-pytorch.md) | Profiling PyTorch workloads |
| [Containers](docs/07-container-profiling.md) | Profiling inside Docker/Slurm |
| [Focused Profiling](docs/08-focused-profiling.md) | Targeted profiling strategies |

### AI Agent Documentation
The [`docs/agent_skills/`](docs/agent_skills/) directory contains the full agent documentation — [quickstart](docs/agent_skills/QUICKSTART.md), [skill index](docs/agent_skills/INDEX.md), [design principles](docs/agent_skills/PRINCIPLES.md), 11 command references, and 8 skill methodology guides.

### Other Resources
- [**Book of Root Causes**](docs/root-causes/) — common GPU performance anti-patterns and their diagnostic signatures
- [**SQLite Schema Explorer**](docs/sqlite-explorer/) — interactive HTML tool for exploring Nsight SQLite tables, foreign keys, and example queries

---

## 🔀 Profile Diff

Compare two profiles side-by-side — spot regressions and improvements from a single command.

```bash
# Compare before and after a code change
nsys-ai diff before.sqlite after.sqlite

# Open interactive side-by-side web comparison
nsys-ai diff-web before.sqlite after.sqlite

# Focus on a specific GPU
nsys-ai diff before.sqlite after.sqlite --gpu 0

# Compare a specific time window
nsys-ai diff before.sqlite after.sqlite --trim 39 42

# Export as markdown (for GitHub issues)
nsys-ai diff before.sqlite after.sqlite --format markdown -o diff.md

# JSON output for scripting
nsys-ai diff before.sqlite after.sqlite --format json --no-ai
```

The report shows:

- **Top regressions** — kernels that got slower (by Δ time, %, or total)
- **Top improvements** — kernels that got faster
- **New / removed kernels** — workload changes across runs
- **NVTX region diff** — wall-time delta for annotated regions
- **Overlap diff** — compute/NCCL overlap and idle gap changes
- **Per-GPU breakdown** — when no `--gpu` is specified, shows every device

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu N` | all GPUs | Focus on a specific device |
| `--trim START END` | — | Compare only this time window (seconds) |
| `--format` | `terminal` | `terminal` \| `markdown` \| `json` |
| `-o / --output` | stdout | Write output to file |
| `--limit N` | 15 | Top regressions/improvements to show |
| `--sort` | `delta` | `delta` \| `percent` \| `total` |
| `--no-ai` | — | Skip AI narration (numeric diff only) |

---

## 🛠️ All Commands

| Command | Description |
|---------|-------------|
| `open` | **Quick profile opener (auto-selects UI)** |
| `timeline-web` | **Web-based multi-GPU timeline** (default) |
| `web` | Interactive web viewer with NVTX + kernel detail |
| `chat` / `ask` | Interactive AI chat / Single AI question |
| `agent analyze` | Full auto-analysis with LLM-backed diagnostics |
| `evidence build`| Run heuristic analyzers → findings JSON |
| `analyze` | Full auto-report: bottlenecks, overlap, iters, NVTX |
| `info` | Profile metadata & GPU hardware |
| `summary` | Top kernels, stream breakdown, auto-commentary |
| `overlap` | Compute / NCCL overlap analysis |
| `nccl` | NCCL collective breakdown by type |
| `iters` | Auto-detect training iterations |
| `diff` | **Before/after profile comparison (CLI)** |
| `diff-web` | **Side-by-side comparison web viewer** |
| `tui` | **Interactive tree TUI** |
| `timeline` | **Interactive timeline TUI** |
| `tree` | NVTX hierarchy as text |
| `search` | Search kernels / NVTX by name |
| `viewer` | Interactive HTML report |
| `export` | Perfetto JSON traces |
| `export-csv` | Flat CSV for spreadsheets |
| `export-json` | Flat JSON for scripting |

---

## 🧩 Skills (Analysis Building Blocks)

nsys-ai ships with **25 built-in analysis skills** — self-contained SQL-powered units that work without any LLM:

```bash
# List all available skills
nsys-ai skill list

# Run a specific skill
nsys-ai skill run top_kernels profile.sqlite
nsys-ai skill run region_mfu profile.sqlite -p operation=full_model
nsys-ai skill run root_cause_matcher profile.sqlite --format json
```

| Category | Skills |
|----------|--------|
| **Kernel** | `top_kernels`, `kernel_instances`, `kernel_launch_overhead`, `kernel_launch_pattern`, `kernel_overlap_matrix` |
| **Memory** | `memory_transfers`, `memory_bandwidth` |
| **Pipeline** | `gpu_idle_gaps`, `cpu_gpu_pipeline`, `stream_concurrency`, `overlap_breakdown` |
| **NVTX** | `nvtx_kernel_map`, `nvtx_layer_breakdown` |
| **NCCL** | `nccl_breakdown`, `nccl_anomaly` |
| **Iteration** | `iteration_timing`, `iteration_detail` |
| **Performance** | `region_mfu`, `theoretical_flops`, `speedup_estimator` |
| **Diagnostics** | `root_cause_matcher`, `profile_health_manifest`, `thread_utilization`, `schema_inspect` |

Skills are extensible — add your own with `nsys-ai skill add`.

---

## 🤖 AI Agent & Analysis

The 25 built-in skills run SQL queries directly — **no API key needed**:

```bash
nsys-ai analyze profile.sqlite                                  # full auto-report
nsys-ai evidence build profile.sqlite -o findings.json           # heuristic diagnostics
nsys-ai timeline-web profile.sqlite --findings findings.json     # overlay findings on timeline
```

With `pip install nsys-ai[agent]`, add LLM-backed analysis:

```bash
nsys-ai agent analyze profile.sqlite                             # LLM-narrated auto-analysis
nsys-ai agent ask profile.sqlite "why are there pipeline bubbles?"
nsys-ai chat profile.sqlite                                      # interactive AI chat TUI
```

The agent auto-selects skills, diagnoses root causes, calculates MFU, and generates evidence findings with nanosecond-precision timestamps.

---

## 📦 Install Tiers

```bash
pip install nsys-ai            # Core: CLI + TUI + skills (duckdb, rich, textual)
pip install nsys-ai[agent]     # + LLM agent (anthropic, litellm)
pip install nsys-ai[chat]      # + AI chat TUI (litellm)
pip install nsys-ai[all]       # Everything
pip install nsys-ai[dev]       # + dev tools (pytest, ruff)
```

---

## 🧑‍💻 Development

```bash
git clone https://github.com/GindaChen/nsys-ai.git
cd nsys-ai
pip install -e '.[dev]'
pytest tests/ -v
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).

<div align="center">
<sub>Built for GPU performance engineers.</sub>
</div>
