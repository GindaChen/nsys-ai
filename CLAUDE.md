# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

nsys-ai is an AI-powered terminal UI for analyzing NVIDIA Nsight Systems GPU profiles (`.sqlite` files). It provides Textual-based TUI viewers, a local web timeline, HTML export, a skill-based analysis system, and an LLM agent for automated GPU performance diagnosis.

**Naming:** The PyPI package is `nsys-ai`, but the internal Python module is `nsys_ai` (historical). Both `nsys-ai` and `nsys-tui` CLI commands work.

## Build & Development Commands

```bash
# Install (pick one tier)
pip install -e '.[dev]'      # Core + pytest (for development)
pip install -e '.[agent]'    # Core + anthropic + litellm (for agent work)
pip install -e '.[all]'      # Everything

# Test
pytest tests/ -v --tb=short

# Smoke test
python -m nsys_ai --help

# Run the app
nsys-ai <command> <profile.sqlite>
```

Core runtime dependencies are `duckdb` + `pyarrow` (Parquet cache acceleration) and `rich` + `textual` (TUI). SQL profile analysis and the web server stay on the stdlib (`sqlite3`, `http.server`), so a profile can still be read and analyzed without the cache. AI features (`ask`/`chat`/`agent`) add `litellm` (and `anthropic`) via the `[agent]` / `[chat]` extras.

## Testing

- CI runs on Python 3.10, 3.11, 3.12
- Tests live in `tests/` — `test_cli.py` (smoke), `test_agent.py` (agent/persona), `test_skills.py` (skill system)
- New CLI subcommands need a test in `test_cli.py`
- AI-related changes need `pip install -e '.[ai]'` before testing

## Architecture

### Entry Point

`src/nsys_ai/__main__.py` delegates to `nsys_ai.cli.app:main`; the argparse CLI is built in `cli/parsers.py` (~30 subcommands) and dispatched to `cli/handlers.py`. Two entry points registered in pyproject.toml: `nsys-ai` and `nsys-tui`, both point to `nsys_ai.__main__:main`.

### Core Data Model

Profiles are `.sqlite` files from NVIDIA Nsight Systems. Key tables: `CUPTI_ACTIVITY_KIND_KERNEL`, `NVTX_EVENTS`, `CUPTI_ACTIVITY_KIND_RUNTIME`. The `Profile` class in `profile.py` handles loading and metadata discovery.

### Key Modules

- `profile.py` — SQLite profile loader, `Profile`/`ProfileMeta`/`GpuInfo` classes
- `tree/` — Textual NVTX tree TUI (`run_tui`) plus the NVTX tree data model and formatters (`build_nvtx_tree`, `format_text`/`format_markdown`)
- `timeline/` — Textual Perfetto-style horizontal timeline TUI (`run_timeline`)
- `overlap.py` — Compute/NCCL overlap analysis (and `launch_overhead_ms`)
- `export.py` / `export_flat.py` — HTML viewer and CSV/JSON export
- `viewer.py` — Perfetto JSON trace export
- `web.py` — Local HTTP server (stdlib `http.server` + custom `_ThreadPoolMixIn`; no Flask/Jinja2)
- `diff.py` / `diff_tools.py` / `diff_render.py` / `diff_web.py` — before/after profile comparison + verdict

### Skill System (`src/nsys_ai/skills/`)

Skills are self-contained SQL-based analysis units that don't require an LLM. Each skill in `skills/builtins/` defines a SQL query template + formatter:

- `top_kernels` — Heaviest GPU kernels by time
- `gpu_idle_gaps` — Pipeline bubbles between kernels
- `memory_transfers` — H2D/D2H/D2D breakdown
- `nccl_breakdown` — NCCL collective summary
- `nvtx_kernel_map` — NVTX annotation → kernel mapping
- `kernel_launch_overhead` — CPU→GPU dispatch latency
- `thread_utilization` — CPU thread bottleneck detection
- `schema_inspect` — Database tables and columns

`skills/base.py` defines the `Skill` dataclass; `skills/registry.py` handles auto-discovery.

### Agent System (`src/nsys_ai/agent/`)

- `persona.py` — System prompt defining the agent as a CUDA ML Systems Performance Expert
- `loop.py` — `Agent` class that orchestrates skill selection and LLM-based analysis
- Workflow: ORIENT → IDENTIFY → HYPOTHESIZE → INVESTIGATE → DIAGNOSE → RECOMMEND → VERIFY
- Requires `anthropic` SDK (`pip install -e '.[agent]'`)

### AI Module (`src/nsys_ai/ai/`)

- `analyzer.py` — LLM-based NVTX analysis
- `annotator.py` — NVTX annotation utilities
- `gate.py` — Cost gating for LLM API calls

## Release Process

1. Bump `version` in `pyproject.toml`
2. Commit and tag: `git tag vX.Y.Z`
3. Push: `git push origin main --tags`
4. GitHub Actions auto-publishes to PyPI via trusted publisher (no tokens needed)

## Project Labels & Workflow

- **Pillars:** `pillar/ai` (analysis, NLP), `pillar/ui` (TUI, web, viewer)
- **Priority:** `P0-critical` through `P3-low`
- **Agent workflow:** `agent-ready` → `agent-in-progress` → `agent-review` → merged
