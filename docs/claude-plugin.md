# nsys-ai Claude Skill

A Claude Code plugin that turns NVIDIA Nsight Systems GPU profiles into actionable
performance fixes — without knowing any CLI commands.

## What it does

Point it at a `.nsys-rep` or `.sqlite` profile and get:

- **Root cause** — NCCL serialization, GPU idle bubbles, kernel hotspot, memory bottleneck
- **Code fix** — specific Python change with line reference
- **Expected speedup** — quantified via `speedup_estimator`
- **Interactive timeline** — bottleneck highlighted in browser

## Modes

| Mode | Use when |
|------|----------|
| **Diagnose** | "Why is my training slow?" — auto-routes to the right analysis |
| **Compare** | "Did my change help?" — before/after regression diff |
| **Efficiency** | "What is my MFU?" — requires model architecture |
| **CUTracer** | "Why is this kernel slow at the instruction level?" — requires GPU re-run |

## Install

```bash
# Install nsys-ai CLI
pip install "nsys-ai[agent]"

# Install this skill
git clone https://github.com/GindaChen/nsys-ai ~/.claude/skills/nsys-ai
```

See [QUICKSTART.md](QUICKSTART.md) for a 2-minute walkthrough.

## Usage

```
/analyze                                  # show mode menu
/analyze profile.nsys-rep                 # with profile
/analyze profile.nsys-rep why slow?       # skip menu, direct question
/analyze before.sqlite after.sqlite diff  # compare mode
```

## Requirements

- `nsys-ai >= 0.9.0` (`pip install "nsys-ai[agent]"`)
- NVIDIA Nsight Systems (for profiling — not needed for analysis)
- CUTracer mode additionally requires: `cutracer.so` + `nvdisasm` (optional)

## What's in this repo

```
skills/analyze/
  SKILL.md              ← main skill instructions (Claude reads this)
  references/
    PRINCIPLES.md       ← non-negotiable analysis rules
    TRIAGE.md           ← 6-stage triage workflow
    DISTRIBUTED.md      ← NCCL / multi-GPU analysis
    DIFF.md             ← regression comparison workflow
    MFU.md              ← MFU / efficiency workflow
    VARIANCE.md         ← iteration variance analysis
    SKILLS_REF.md       ← 33-skill quick reference card
```
