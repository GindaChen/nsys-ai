# nsys-ai Agent Design

> Design docs for AI agents that work on or with nsys-ai.

## Contents

| File | Description |
|------|-------------|
| [agent-design.md](./agent-design.md) | Agent identity, mission, knowledge hierarchy, task taxonomy |
| [problem-taxonomy.md](./problem-taxonomy.md) | 5 categories of GPU profiling problems the tool addresses |

## Quick Context

nsys-ai is a terminal-based tool for analyzing NVIDIA Nsight Systems GPU profiles. It reads `.sqlite` exports and provides:
- Interactive timeline TUI (Perfetto-style)
- NVTX hierarchy browser
- Kernel summary and overlap analysis
- Export to Perfetto JSON, CSV, HTML
- Optional AI-powered analysis (Claude)

## Related

| Resource | Path |
|----------|------|
| Agent playbook | [`../AGENTS.md`](../AGENTS.md) |
| Roadmap | [`../ROADMAP.md`](../ROADMAP.md) |
| Nsight reference docs | [`../docs/`](../docs/) |
| AI module source | [`../src/nsys_tui/ai/`](../src/nsys_tui/ai/) |
