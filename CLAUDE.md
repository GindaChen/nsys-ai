# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

nsys-ai is an AI-powered terminal UI for analyzing NVIDIA Nsight Systems GPU profiles (`.sqlite` files). It provides Textual-based TUI viewers, a local web timeline, HTML export, a skill-based analysis system, and an LLM agent for automated GPU performance diagnosis.

**Naming:** The PyPI package is `nsys-ai`, but the internal Python module is `nsys_ai` (historical). The wheel exposes two console entry points: `nsys-ai` for the main CLI and `nsys-ai-mcp` for the optional stdio MCP transport. `nsys-tui` is not a 0.3.0 entry point; tree and timeline TUIs are subcommands of `nsys-ai`.

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

`src/nsys_ai/__main__.py` delegates to `nsys_ai.cli.app:main`; the argparse CLI is built in `cli/parsers.py` (~30 subcommands) and dispatched to `cli/handlers.py`. `pyproject.toml` registers `nsys-ai` (the main CLI, pointing at `nsys_ai.__main__:main`) and `nsys-ai-mcp` (the stdio MCP server, pointing at `nsys_ai.mcp_server:main`). The MCP entry point is optional at runtime and requires the `mcp` extra when it is used.

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
- `baseline.py` — local baseline snapshot store (tag/list/show + `diff --against` resolution)

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

- `backend/` — read-only profile database tooling the agent queries
- `diff_narrative.py` — LLM narrative over a computed diff

## Release Process

Follow the complete [release guide](docs/dev/release.md) for scope/freeze,
candidate verification, packaging, publishing, PyPI verification, and handoff.
It is the source of truth for every release; do not replace its gates with a
shorter tag-only procedure.

For a quick orientation:

1. Create the release tracking issue and record the exact candidate commit.
2. Run the guide's source, full-suite, packaging, and product smoke checks.
3. After review, push only the tested version tag to the canonical repository.
4. Monitor the publish workflow and verify the package from a fresh PyPI
   environment before closing the release issue.

In the normal contributor checkout, `origin` is the fork and `upstream` is
`GindaChen/nsys-ai`; verify that mapping with `git remote -v`. Push release
branches and tags to the canonical remote (`upstream` in that setup), never
by pushing the fork's `main` branch together with every tag. GitHub Actions
publishes to PyPI through the trusted publisher; no package token belongs in
the repository.

## Project Labels & Workflow

- **Pillars:** `pillar/ai` (analysis, NLP), `pillar/ui` (TUI, web, viewer)
- **Priority:** `P0-critical` through `P3-low`
- **Agent workflow:** `agent-ready` → `agent-in-progress` → `agent-review` → merged
