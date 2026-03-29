# nsys-ai Agent Skills — Index

**Read `PRINCIPLES.md` first.** It contains the rules, error handling, and tool architecture.

This file serves as your routing guide:
- Use the [CLI Quick Reference](#cli-quick-reference) to invoke `nsys-ai` analysis commands.
- Use the [Skill Router](#skill-router-llm-workflow-guides) to load the right `.md` reasoning workflow.

---

## CLI Quick Reference

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
| `nsys-ai open <profile> [--gpu N] [--trim S E] [--viewer V] [--port P]` | Opens in Perfetto/web/TUI viewer | Quick visual inspection |
| `nsys-ai web <profile> [--gpu N] [--trim S E] [--port P]` | Serves interactive web viewer | Browser-based timeline exploration |
| `nsys-ai timeline-web <profile> [--gpu N] [--trim S E] [--port P] [--findings F] [--auto-analyze]` | Serves timeline-focused web UI | Full timeline + AI chat + evidence sidebar |
| `nsys-ai chat <profile>` | Interactive AI chat TUI | Multi-turn analysis session |
| `nsys-ai timeline <profile> [--gpu N] [--trim S E] [--min-ms M]` | Horizontal timeline TUI | Terminal-based trace viewer |
| `nsys-ai tui <profile> [--gpu N] [--trim S E] [--depth D] [--min-ms M]` | Tree view TUI | Hierarchical NVTX exploration |
| `nsys-ai tree <profile> [--gpu N] [--trim S E]` | NVTX hierarchy as text | Console output of NVTX tree |
| `nsys-ai markdown <profile> [--gpu N] [--trim S E]` | NVTX hierarchy as markdown | Markdown format of NVTX tree |
| `nsys-ai info <profile>` | Show profile metadata and GPU info | Hardware specs and iteration counts |
| `nsys-ai summary <profile> [--gpu N] [--trim S E]` | GPU kernel summary with top kernels | Quick hotspot overview without AI |
| `nsys-ai overlap <profile> [--gpu N] [--trim S E]` | Compute/NCCL overlap analysis | Concurrent execution stats |
| `nsys-ai nccl <profile> [--gpu N] [--trim S E]` | NCCL collective breakdown | Network communication profiling |
| `nsys-ai iters <profile> [--gpu N] [--trim S E]` | Detect training iterations | Find stable steps and warmup |
| `nsys-ai export <profile> [--gpu N] [--trim S E] [-o DIR]` | Export Perfetto JSON | Post-processing / sharing |
| `nsys-ai perfetto <profile> [--gpu N] [--trim S E] [--port P]` | Open trace in Perfetto UI | Direct Perfetto trace viewer |
| `nsys-ai export-json <profile> [--gpu N] [--trim S E] [-o out.json] [--summary]` | Export kernel data as flat JSON | Easy programmatic ingestion |
| `nsys-ai export-csv <profile> [--gpu N] [--trim S E] [-o out.csv]` | Export kernel data as flat CSV | Statistical analysis pipelines |
| `nsys-ai viewer <profile> [--gpu N] [--trim S E] [-o out.html]` | Generate interactive HTML viewer | Shareable web report |
| `nsys-ai timeline-html <profile> [--gpu N] [--trim S E] [-o out.html]`| Generate horizontal timeline HTML | Static trace visualization |
| `nsys-ai search <profile> -q <query> [--gpu N] [--trim S E] [--parent P] [--type T] [--limit L]` | Search kernels/NVTX by name | Fast exact name discovery |
| `nsys-ai diff-web <before> <after> [--gpu N] [--trim S E] [--port P]` | Web diff viewer | Visual side-by-side comparison |
| `nsys-ai agent analyze <profile> [--trim S E] [--evidence] [-o out]`| Full auto-analysis report | CLI auto-analysis (no LLM needed) |
| `nsys-ai agent ask <profile> "<question>"` | Ask a targeted question | Keyword-based skill selection |
| `nsys-ai skill list [--format F]` | List all builtin analysis skills | Discover available skills |
| `nsys-ai skill run <name> <profile> [--trim S E] [--format F] [-p K=V]` | Run a skill against a profile | Targeted analysis |
| `nsys-ai skill add <path.md>`* | Add a custom skill | Extend the skill system |
| `nsys-ai skill remove <name>`* | Remove a custom skill | Manage skill system |
| `nsys-ai skill save <name> -o <out.md>`* | Export a skill to .md file | Modify/eject builtin skills |
| `nsys-ai agent-guide` | Print machine-readable agent guide | External agent onboarding |

> **Builtin Skills Catalog**: See [`commands/skill.md`](commands/skill.md) for the complete
> list of builtin skills with names, categories, descriptions, and parameters.
>
> *\* `skill add` and `skill remove` require `--skills-dir <dir>` (before the subcommand, e.g. `nsys-ai skill --skills-dir <dir> add <path.md>`) or the `NSYS_AI_CUSTOM_SKILLS_DIR` environment variable.*

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

# 4a: Agent-driven evidence (recommended)
#     Full agent loop: collect → reason → conclude → write findings → view
#
#   Step 1: COLLECT — query multiple skills for raw data
nsys-ai skill run gpu_idle_gaps profile.sqlite --format json > /tmp/gaps.json
nsys-ai skill run nccl_breakdown profile.sqlite --format json > /tmp/nccl.json
nsys-ai skill run top_kernels profile.sqlite --format json > /tmp/kernels.json
#
#   Step 2: REASON — agent (AI) analyzes the collected data:
#     - Cross-references gaps, NCCL, and kernel data
#     - Identifies root causes (e.g. "21s bubble caused by serialized NCCL")
#     - Draws conclusions with supporting evidence
#     (This is the agent's own LLM reasoning — not a nsys-ai command)
#
#   Step 3: WRITE — agent writes conclusions + supporting time ranges
#     Each finding in findings.json must:
#     - Have a conclusion LABEL (not just raw data)
#     - Reference the specific start_ns/end_ns that proves the conclusion
#     - Include a NOTE explaining WHY this time range matters
cat > /tmp/findings.json << 'EOF'
{
  "title": "Pipeline Parallelism Bubble Analysis",
  "findings": [
    {
      "type": "region",
      "label": "PP Bubble: 21s idle caused by serialized NCCL",
      "start_ns": 89886440111, "end_ns": 110951683466,
      "severity": "critical",
      "note": "GPU idle for 21s after AllReduce — NCCL not overlapping with compute"
    }
  ]
}
EOF
#
#   Step 4: VIEW — open timeline with evidence overlay
nsys-ai timeline-web profile.sqlite --findings /tmp/findings.json

# 4b: Auto-analyze evidence (built-in heuristics, no agent reasoning)
nsys-ai agent analyze profile.sqlite --evidence -o findings.json
nsys-ai timeline-web profile.sqlite --findings findings.json

# 4c: One-step auto-analyze (quickest)
nsys-ai timeline-web profile.sqlite --auto-analyze

# Workflow 5: External agent onboarding
# Print a machine-readable guide for external AI agents
nsys-ai agent-guide
# → Outputs: identity, 6-stage workflow, CLI syntax, full skill catalog
```

---

## Slash Commands

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

## Skill Router (LLM Workflow Guides)

These are **agent reasoning workflows** — step-by-step procedures for the LLM
to follow when analyzing a profile. They tell the agent *how to think*, not just
what tools to call.

> **Not the same as builtin skills.** The 21 Python builtin skills
> (invoked via `nsys-ai skill run`) are documented in [`commands/skill.md`](commands/skill.md).
> Those are executable analysis modules. The files below are reasoning guides.

| User asks about… | Load this file |
|-----------------|----------------|
| "What's my MFU?" / "How efficient is flash attention?" | [`skills/mfu.md`](skills/mfu.md) |
| "What's in my profile?" / "What's the bottleneck?" | [`skills/triage.md`](skills/triage.md) |
| "Why did it slow down?" / "Compare before vs after" | [`skills/diff.md`](skills/diff.md) |
| "NCCL too slow" / "Multi-GPU not scaling" / "GPU imbalance?" | [`skills/distributed.md`](skills/distributed.md) |
| "Some steps spike" / "High iteration variance" | [`skills/variance.md`](skills/variance.md) |
| SQL syntax / writing a query | [`skills/sql.md`](skills/sql.md) |

---


## Adding a New Skill

See [`skills/SKILL_TEMPLATE.md`](skills/SKILL_TEMPLATE.md) and the SOP in `PRINCIPLES.md`.
