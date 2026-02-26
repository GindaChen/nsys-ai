# nsys-tui

Terminal UI for NVIDIA Nsight Systems profiles — timeline viewer, kernel navigator, NVTX hierarchy.

[![CI](https://github.com/GindaChen/nsys-tui/actions/workflows/ci.yml/badge.svg)](https://github.com/GindaChen/nsys-tui/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Show profile info
nsys-tui info profile.sqlite

# Interactive timeline TUI (Perfetto-style)
nsys-tui timeline profile.sqlite --gpu 0 --trim 39 42

# Interactive tree TUI
nsys-tui tui profile.sqlite --gpu 0 --trim 39 42

# GPU kernel summary
nsys-tui summary profile.sqlite --gpu 0

# Export to Perfetto JSON
nsys-tui export profile.sqlite -o traces/
```

## Timeline TUI

A curses-based horizontal timeline viewer with:

- **Time-cursor navigation** — ←/→ pans through time, ↑/↓ selects stream
- **Per-stream colors** — 7-color palette for visual stream differentiation
- **NVTX hierarchy** — stacked NVTX span bars above stream swimlanes
- **Kernel details** — inline names, duration labels, heat-based styling
- **Bookmarks** — save/jump to positions and ranges
- **Config panel** — tweak stream rows, tick density, NVTX depth live

### Keybindings

| Key | Action |
|-----|--------|
| ←/→ | Pan through time |
| Shift+←/→ | Page pan (1/4 viewport) |
| ↑/↓ | Select stream |
| Tab / Shift+Tab | Snap to next/prev kernel |
| +/- | Zoom in/out |
| a | Toggle absolute/relative time |
| T | Cycle time tick density |
| B | Save bookmark |
| ' | Show bookmark list (1-9 to jump) |
| ` | Jump back to previous position |
| C | Config panel |
| h | Help overlay |
| / | Filter kernels by name |
| m | Set min duration threshold |
| q | Quit |

## Commands

| Command | Description |
|---------|-------------|
| `info` | Profile metadata and GPU info |
| `summary` | GPU kernel summary with top kernels |
| `overlap` | Compute/NCCL overlap analysis |
| `nccl` | NCCL collective breakdown |
| `iters` | Detect training iterations |
| `tree` | NVTX hierarchy as text |
| `tui` | Interactive tree TUI |
| `timeline` | Horizontal timeline TUI |
| `search` | Search kernels/NVTX by name |
| `export` | Export Perfetto JSON traces |
| `export-csv` | Flat CSV export |
| `export-json` | Flat JSON export |
| `viewer` | Generate interactive HTML viewer |

## Development

```bash
pip install -e '.[dev]'
pytest tests/ -v
```

## License

MIT
