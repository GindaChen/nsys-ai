# Quick Start — nsys-ai Claude Skill

## 1. Install

```bash
pip install "nsys-ai[agent]"
```

## 2. Add the plugin to Claude Code

**Option A — marketplace (recommended).** In Claude Code, run:

```
/plugin marketplace add GindaChen/nsys-ai
/plugin install nsys-ai@GindaChen
```

The first command registers this repo as a marketplace (reads
`.claude-plugin/marketplace.json`). The second installs the `nsys-ai` plugin
from it. Both commands are typed into the Claude Code prompt, not a shell.

To pick up marketplace changes later: `/plugin marketplace update GindaChen`.

**Option B — local dev (session-scoped).**

```bash
git clone https://github.com/GindaChen/nsys-ai
claude --plugin-dir ./nsys-ai
```

## 3. Profile your workload

```bash
# Recommended: capture only the training loop (smaller file, faster analysis)
nsys profile --capture-range=cudaProfilerApi python train.py
# (add torch.cuda.profiler.start() / .stop() around your training loop)

# Or: full run
nsys profile python train.py
```

This produces `report1.nsys-rep` (or whatever name you pass with `-o`). The skill accepts both `.nsys-rep` and `.sqlite` — no manual export needed.

## 4. Run the skill

In Claude Code, type:

```
/nsys-ai
```

The skill scans your CWD for profiles and shows the mode menu. Pick a number, or just
ask a question — keywords auto-route to the right mode.

## 5. Common invocations

```
# Auto-triage: "why is this slow?"
/nsys-ai report1.nsys-rep

# Skip menu with a direct question
/nsys-ai report1.nsys-rep why is my training slow?
/nsys-ai report1.nsys-rep nccl overlap
/nsys-ai report1.nsys-rep gemm kernel hotspot

# Compare two runs (Mode 8 — Diff)
/nsys-ai before.sqlite after.sqlite
/nsys-ai before.sqlite after.sqlite did my change help?

# Iteration spikes (Mode 9 — Variance)
/nsys-ai report.sqlite some iterations are much slower

# SASS instruction analysis (Mode 7 — CUTracer, requires re-run)
/nsys-ai report.sqlite cutracer flash attention
```

## 6. What you get

Within 3 turns:
- **Root cause** with mechanism and magnitude
- **Specific fix** matched to the diagnosis
- **Expected gain** (quantified)
- **Interactive timeline URL** with the bottleneck highlighted

Example output:
```
Root cause: flash_bwd accounts for 82% share after the change (+58 ms, 62%→82%).
Classification: regression. Likely cause: sequence length increase.

Fix: Revert sequence length or use gradient checkpointing to reduce activation memory.

Expected gain: ~58 ms per step recovered.

Timeline: http://127.0.0.1:8742  (bottleneck annotated)
```

## Modes at a glance

| # | Ask about | Example trigger |
|---|-----------|----------------|
| 1 | Why is it slow? (default) | `/nsys-ai profile.sqlite` |
| 2 | NCCL / multi-GPU overlap | `nccl`, `allreduce`, `overlap` |
| 3 | Hot kernels / tensor core / MFU | `gemm`, `kernel`, `mfu` |
| 4 | Memory transfers / bandwidth | `memory`, `h2d`, `bandwidth` |
| 5 | Which layer/step is slow? | `nvtx`, `layer`, `per-layer` |
| 6 | GPU idle, CPU stalls | `idle`, `sync`, `dataloader` |
| 7 | Instruction-level (SASS) | `cutracer`, `sass` |
| 8 | Before vs after diff | two profile paths, `diff`, `regression` |
| 9 | Some iterations are outliers | `spike`, `variance`, `jitter` |

See [claude-plugin.md](claude-plugin.md) for full reference.
