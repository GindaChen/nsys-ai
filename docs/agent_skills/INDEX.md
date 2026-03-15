# nsys-ai Agent Skills — Index

**Read `PRINCIPLES.md` first.** It contains the rules, error handling, and tool architecture.

This file serves two audiences:
- **External agents** (e.g. OpenClaw) → use the [CLI Quick Reference](#cli-quick-reference) below to invoke `nsys-ai` from a terminal.
- **Internal agent** (nsys-ai's own LLM) → use the [Skill Router](#skill-router) to load the right `.md` file.

---

## CLI Quick Reference (for external agents)

> **Install**: `pip install nsys-ai[ai]`
> **Prerequisite**: An `.sqlite` profile exported from NVIDIA Nsight Systems.
> Convert `.nsys-rep` to `.sqlite`: `nsys export --type=sqlite --output=profile.sqlite report.nsys-rep`

### Ask — One-shot question

```bash
nsys-ai ask <profile.sqlite> "<question>"
```

| Example question | What it does |
|------------------|-------------|
| `"what's the bottleneck?"` | Triage: top kernels, GPU utilization, NCCL presence |
| `"what's the MFU of flash_fwd_kernel?"` | Compute region MFU for a specific kernel |
| `"how many kernels in this profile?"` | SQL query via the profile database |

**Output**: Markdown text on stdout. Parse the answer directly.

### Diff — Compare two profiles

```bash
# Terminal summary (human-readable)
nsys-ai diff before.sqlite after.sqlite

# Machine-readable JSON
nsys-ai diff before.sqlite after.sqlite --format json

# Specific iteration (skip warmup iteration 0)
nsys-ai diff before.sqlite after.sqlite --iteration 1 --format json

# Limit to top N regressions, sorted by absolute delta
nsys-ai diff before.sqlite after.sqlite --limit 10 --sort delta --format json

# Interactive AI chat for deep-dive
nsys-ai diff before.sqlite after.sqlite --chat
```

**Output formats**: `terminal` (default), `markdown`, `json`
**Key flags**: `--gpu N`, `--trim START END` (seconds), `--iteration N`, `--marker NAME`

### Report — Full analysis report

```bash
nsys-ai report profile.sqlite --gpu 0 --trim 1.0 5.0
nsys-ai report profile.sqlite --gpu 0 --trim 1.0 5.0 -o report.md
```

**Output**: Markdown performance report (stdout or file).

### Other commands

| Command | Usage | Purpose |
|---------|-------|---------|
| `nsys-ai open <profile>` | Opens in Perfetto/web/TUI viewer | Quick visual inspection |
| `nsys-ai web <profile> --gpu 0 --trim 1 5` | Serves interactive web viewer | Browser-based timeline exploration |
| `nsys-ai timeline-web <profile>` | Serves timeline-focused web UI | Full timeline + AI chat + evidence sidebar |
| `nsys-ai chat <profile>` | Interactive AI chat TUI | Multi-turn analysis session |
| `nsys-ai export <profile> --gpu 0 --trim 1 5` | Export Perfetto JSON | Post-processing / sharing |
| `nsys-ai diff-web <before> <after>` | Web diff viewer | Visual side-by-side comparison |
| `nsys-ai agent analyze <profile>` | Full auto-analysis report | CLI auto-analysis (no LLM needed) |
| `nsys-ai agent ask <profile> "<question>"` | Ask a targeted question | Keyword-based skill selection |

### Common agent workflows

```bash
# Workflow 1: Quick triage → MFU
nsys-ai ask profile.sqlite "what's the bottleneck?"
nsys-ai ask profile.sqlite "what's the MFU of flash_fwd_kernel with H=4096 S=2048 L=32?"

# Workflow 2: Regression root cause
nsys-ai diff before.sqlite after.sqlite --format json --iteration 1
nsys-ai diff before.sqlite after.sqlite --chat  # deep-dive if needed

# Workflow 3: Full report to file
nsys-ai report profile.sqlite --gpu 0 --trim 1 5 -o analysis.md
cat analysis.md

# Workflow 4: Visual evidence for human verification
# See docs/agent_skills/commands/evidence_schema.md for Finding JSON format

# 4a: Agent-driven evidence (recommended for external agents)
#     Agent queries data, reasons about it, writes findings.json with only
#     the time ranges that support its conclusions, then opens viewer.
nsys-ai skill run gpu_idle_gaps profile.sqlite --format json   # get timestamps
nsys-ai skill run nccl_breakdown profile.sqlite --format json  # get NCCL data
# ... agent writes /tmp/findings.json based on its conclusions ...
nsys-ai timeline-web profile.sqlite --findings /tmp/findings.json

# 4b: Auto-analyze evidence (built-in heuristics, no agent reasoning)
nsys-ai agent analyze profile.sqlite --evidence -o findings.json
nsys-ai timeline-web profile.sqlite --findings findings.json

# 4c: One-step auto-analyze (quickest)
nsys-ai timeline-web profile.sqlite --auto-analyze
```

---

## Slash Commands (for nsys-ai's internal agent)

| Command | When to use |
|---------|------------|
| [`/nsys:analyze`](commands/analyze.md) | Single profile: bottleneck, MFU, NCCL |
| [`/nsys:diff`](commands/diff.md) | Two profiles: regression root cause |
| [`/nsys:mfu`](commands/mfu.md) | Compute MFU / efficiency for a region or step |
| [`/nsys:refine`](commands/refine.md) | Add or improve a skill |
| [`/nsys:validate`](commands/validate.md) | Verify a previous analysis claim |
| [`/nsys:skilldoc`](commands/skilldoc.md) | Validate skill system doc completeness (35 checks) |
| [`/nsys:test`](commands/test.md) | Run pytest suite and update TEST.md |

---

## Skill Router

| User asks about… | Load this file |
|-----------------|----------------|
| "What's my MFU?" / "How efficient is flash attention?" | [`skills/mfu.md`](skills/mfu.md) |
| "What's in my profile?" / "What's the bottleneck?" | [`skills/triage.md`](skills/triage.md) |
| "Why did it slow down?" / "Compare before vs after" | [`skills/diff.md`](skills/diff.md) |
| "NCCL too slow" / "Multi-GPU not scaling" / "GPU imbalance?" | [`skills/distributed.md`](skills/distributed.md) |
| "Some steps spike" / "High iteration variance" | [`skills/variance.md`](skills/variance.md) |
| SQL syntax / writing a query | [`skills/sql.md`](skills/sql.md) |

---

## Tool Sets

### Set A — Single Profile
`query_profile_db` · `get_gpu_peak_tflops` · `compute_theoretical_flops` · `compute_region_mfu` · `compute_mfu` · `navigate_to_kernel` · `zoom_to_time_range` · `fit_nvtx_range` · `get_gpu_overlap_stats` · `get_nccl_breakdown`

### Set B — Diff Mode (two profiles loaded)
All of Set A, plus:
`search_nvtx_regions` · `get_iteration_boundaries` · `explore_nvtx_hierarchy` · `get_top_nvtx_diffs` · `get_iteration_diff` · `get_region_diff` · `summarize_nvtx_subtree` · `get_launch_config_diff` · `get_source_code_context` · `get_gpu_imbalance_stats` · `get_global_diff` · `get_memory_profile_diff`

---

## Adding a New Skill

See [`skills/SKILL_TEMPLATE.md`](skills/SKILL_TEMPLATE.md) and the SOP in `PRINCIPLES.md`.
