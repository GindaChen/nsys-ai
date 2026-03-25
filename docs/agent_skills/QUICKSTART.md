# nsys-ai Agent Skills — Quick Start

**You are a new agent. Read this in 2 minutes, then start.**

---

## What is nsys-ai?

`nsys-ai` is a Python tool for analyzing NVIDIA Nsight Systems GPU profiles (`.sqlite` files).
You are the AI agent that operates it. You have function-call tools that query the profile,
compute efficiency metrics, and compare runs.

---

## 5-Minute Orientation

### Step 1 — Know your two docs

| File | Read when |
|------|-----------|
| `PRINCIPLES.md` | **Always, before anything.** Rules, error handling, tools. |
| `INDEX.md` | Routing: which slash command or skill to load. |

**Never read all skill files upfront.** They are loaded on demand.

### Step 2 — Know your two tool sets

**Set A** (one profile loaded): `query_profile_db`, `get_gpu_peak_tflops`,
`compute_theoretical_flops`, `compute_region_mfu`, `compute_mfu`, `submit_finding`,
`get_gpu_overlap_stats`, `get_nccl_breakdown`,
navigation tools (UI-only: `navigate_to_kernel`, `zoom_to_time_range`, `fit_nvtx_range`).

**Set B** (two profiles loaded, diff mode): `get_gpu_peak_tflops`, `compute_mfu` from Set A,
plus diff tools: `get_iteration_boundaries`, `get_top_nvtx_diffs`, `get_iteration_diff`,
`get_region_diff`, `search_nvtx_regions`, `get_gpu_imbalance_stats`, and more.
Set B does NOT include `query_profile_db`, `compute_theoretical_flops`, `compute_region_mfu`,
`submit_finding`, `get_gpu_overlap_stats`, `get_nccl_breakdown`, or navigation tools.

> **Note**: Navigation tools are only available in `timeline-web` / `chat` TUI.
> Via CLI (`nsys-ai agent ask`), provide timestamp references in text instead.

### Step 3 — Know the 3 non-negotiable rules

1. **MFU > 100% = bug.** Stop, recompute with narrower `operation`.
2. **Never guess names.** Always query `NVTX_EVENTS` / `StringIds` first.
3. **`theoretical_flops` must come from `compute_theoretical_flops`.** Never estimate.

### Step 4 — Know the entry points

| User wants | Use |
|-----------|-----|
| Single profile analysis | `/nsys:analyze` |
| Compare two profiles | `/nsys:diff` |
| MFU / efficiency metric | `/nsys:mfu` |
| Add/improve a skill | `/nsys:refine` |
| Verify your output | `/nsys:validate` |

**External agents (CLI)**: Use `nsys-ai skill run <name> profile.sqlite` to run
specific analysis skills. See [`commands/skill.md`](commands/skill.md) for the full
catalog of 21 builtin skills.

---

## Typical First 60 Seconds

When a user loads a profile with no specific question:

```
1. Load skills/triage.md
2. Run get_gpu_peak_tflops()
3. Query top kernels by GPU time
4. Check NVTX and NCCL presence
5. Give a 4-line summary + ask what to investigate
```

When a user asks "what's my MFU?":

```
1. Load skills/mfu.md
2. Run get_gpu_peak_tflops()
3. Discover NVTX/kernel names (never guess)
4. Resolve model architecture (lookup table before asking user)
5. compute_theoretical_flops → compute_region_mfu or compute_mfu
6. Sanity check: MFU must be 0–100%
```

---

## Common Pitfalls (and How to Avoid)

| ❌ Wrong | ✅ Right |
|---------|---------|
| Compute FLOPs yourself | Call `compute_theoretical_flops` |
| Use full profile span as step_time | Use single NVTX iteration duration |
| Use `SELECT *` | Name specific columns |
| Guess NVTX name | Query `NVTX_EVENTS` first |
| Divide by 1000 for ms | Divide ns by 1e6 for ms, 1e9 for s |
| Report MFU > 100% | Recompute with narrower `operation` |
| Skip iteration 0 check in diff | Always skip index 0 (JIT warmup) |

---

## File Map

```
docs/agent_skills/
├── QUICKSTART.md      ← you are here
├── PRINCIPLES.md      ← rules + error handling + acceptance checklist
├── INDEX.md           ← routing table (loads in ~3 seconds of context)
├── commands/          ← slash command SOPs + CLI reference
│   ├── analyze.md     /nsys:analyze
│   ├── diff.md        /nsys:diff
│   ├── mfu.md         /nsys:mfu
│   ├── refine.md      /nsys:refine
│   ├── test.md        /nsys:test
│   ├── validate.md    /nsys:validate
│   ├── skilldoc.md    /nsys:skilldoc (documentation audit)
│   ├── evidence_schema.md  Finding JSON schema
│   └── skill.md       nsys-ai skill CLI + builtin catalog (21 Python skills)
└── skills/            ← LLM workflow guides (load on demand)
    ├── SKILL_TEMPLATE.md
    ├── TEST.md         ← test plan + results (like CLI-Anything TEST.md)
    ├── mfu.md          agent reasoning workflow for MFU analysis
    ├── triage.md       agent reasoning workflow for profile triage
    ├── diff.md         agent reasoning workflow for diff/regression
    ├── distributed.md  agent reasoning workflow for NCCL/multi-GPU
    ├── variance.md     agent reasoning workflow for iteration variance
    └── sql.md          SQL reference + query recipes
```
