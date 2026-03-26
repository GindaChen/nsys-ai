# `nsys-ai agent-guide` — External Agent Onboarding

Prints a self-contained, machine-readable guide for **external AI agents** (Claude, GPT, Cursor, Copilot)
that operate nsys-ai via CLI subprocess calls.

## Usage

```bash
nsys-ai agent-guide
```

No arguments required. No profile needed. Prints to stdout.

## What it outputs

The output is a structured System Prompt payload containing:

1. **Identity** — one-line role assignment ("You are an AI performance tuning agent...")
2. **Performance Note** — cold start warning (~60-90s for first skill run on large profiles)
3. **Core Principles** — 3 non-negotiable rules (no guessing, units, source code correlation)
4. **6-Stage Top-Down Triage Workflow** — the recommended analysis sequence:
   - Orient → Temporal Breakdown → Kernel Deep-Dive → NVTX Mapping → Cross-GPU → Root Cause
5. **CLI Execution syntax** — exact command templates for `nsys-ai info` and `nsys-ai skill run`
6. **Full Skill Catalog** — dynamically generated from the Python skill registry (always up-to-date)

## When to use

- **External agent bootstrap**: Inject the output into an AI agent's system prompt before analysis
- **Tool discovery**: List all available skills with their parameters without reading source code
- **Workflow reference**: Quick reminder of the 6-stage triage sequence

## Example integration

```bash
# Inject into an AI agent's context window
GUIDE=$(nsys-ai agent-guide)
echo "$GUIDE" | your_agent_framework --system-prompt -
```

## Implementation

- **Handler**: `_cmd_agent_guide` in `src/nsys_ai/cli/handlers.py`
- **Skill catalog**: Generated dynamically by `skill_catalog()` from `src/nsys_ai/skills/registry.py`
- **No profile required**: This command has zero dependencies — no DuckDB, no SQLite, no cache
