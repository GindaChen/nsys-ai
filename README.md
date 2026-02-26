<div align="center">

# 🔬 nsys-tui

**Terminal UI for NVIDIA Nsight Systems profiles**

Navigate GPU kernel timelines, explore NVTX hierarchies, and analyze performance — all from your terminal.

[![CI](https://github.com/GindaChen/nsys-tui/actions/workflows/ci.yml/badge.svg)](https://github.com/GindaChen/nsys-tui/actions)
[![PyPI](https://img.shields.io/pypi/v/nsys-tui)](https://pypi.org/project/nsys-tui/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## ⚡ Install

```bash
pip install nsys-tui
```

That's it. No system dependencies, no CUDA required. Just Python 3.10+.

---

## 🎯 What It Does

nsys-tui reads `.sqlite` profile exports from [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) and gives you **three ways** to explore them:

<table>
<tr>
<td width="33%" align="center">

### 🖥️ Timeline TUI
Perfetto-style horizontal timeline in your terminal

</td>
<td width="33%" align="center">

### 🌲 Tree TUI
Interactive NVTX hierarchy browser with kernel details

</td>
<td width="33%" align="center">

### 🌐 HTML Viewer
Exportable interactive visualizations for sharing

</td>
</tr>
<tr>
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
# Profile your PyTorch training
nsys profile -o my_training python train.py
# → produces my_training.sqlite
```

### 2. Explore it

```bash
# Quick overview
nsys-tui info my_training.sqlite

# Interactive timeline (the main attraction)
nsys-tui timeline my_training.sqlite --gpu 0 --trim 39 42

# Interactive tree browser
nsys-tui tui my_training.sqlite --gpu 0 --trim 39 42

# GPU kernel summary
nsys-tui summary my_training.sqlite --gpu 0
```

### 3. Export & share

```bash
# Perfetto JSON (open in ui.perfetto.dev)
nsys-tui export my_training.sqlite -o traces/

# Interactive HTML viewer
nsys-tui viewer my_training.sqlite --gpu 0 --trim 39 42 -o report.html

# Flat CSV/JSON for scripting
nsys-tui export-csv my_training.sqlite --gpu 0 --trim 39 42 -o kernels.csv
```

---

## ⌨️ Timeline TUI

The timeline is a **Perfetto-style** horizontal viewer with per-stream kernel visualization, NVTX hierarchy bars, and a time-cursor navigation model.

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

The `docs/` directory includes comprehensive guides for Nsight Systems profiling:

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

### 🔍 Interactive SQLite Schema Explorer

The [`docs/sqlite-explorer/`](docs/sqlite-explorer/) contains an **interactive HTML tool** for exploring the Nsight SQLite schema — tables, foreign keys, example queries, and key concepts. Open `docs/sqlite-explorer/index.html` in a browser:

- Browse all Nsight SQLite tables with column types
- See foreign key relationships visualized
- Copy-paste ready SQL query examples
- Cross-highlighted concept explanations

---

## 🛠️ All Commands

| Command | Description |
|---------|-------------|
| `info` | Profile metadata & GPU hardware |
| `summary` | Top kernels, stream breakdown, auto-commentary |
| `overlap` | Compute / NCCL overlap analysis |
| `nccl` | NCCL collective breakdown by type |
| `iters` | Auto-detect training iterations |
| `tree` | NVTX hierarchy as text |
| `tui` | **Interactive tree TUI** |
| `timeline` | **Interactive timeline TUI** |
| `search` | Search kernels / NVTX by name |
| `export` | Perfetto JSON traces |
| `export-csv` | Flat CSV for spreadsheets |
| `export-json` | Flat JSON for scripting |
| `viewer` | Interactive HTML report |
| `markdown` | NVTX hierarchy as markdown |

---

## 🤖 AI Analysis (Optional)

nsys-tui includes an optional AI module that can analyze your profiles:

```bash
pip install nsys-tui[ai]
```

- **Auto-commentary** on kernel distributions and performance patterns
- **NVTX annotation suggestions** for un-annotated code regions
- **Performance bottleneck detection** with actionable recommendations

---

## 🧑‍💻 Development

```bash
git clone https://github.com/GindaChen/nsys-tui.git
cd nsys-tui
pip install -e '.[dev]'
pytest tests/ -v
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).

<div align="center">
<sub>Built for GPU performance engineers who live in the terminal.</sub>
</div>
