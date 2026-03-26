# nsys-ai Agent Skills — Quick Start

**You are a new agent. Read this in 2 minutes, then start.**

---

## What is nsys-ai?

`nsys-ai` is a Python tool for analyzing NVIDIA Nsight Systems GPU profiles (`.sqlite` files).
You are the AI agent that operates it. You have function-call tools that query the profile,
compute efficiency metrics, and compare runs.

> **Backend**: On first open, `nsys-ai` exports the SQLite profile into a
> DuckDB + Parquet cache (`<profile>.nsys-cache/`) for fast analysis.
> All subsequent queries run over DuckDB views on cached Parquet files.
> If DuckDB is unavailable, it falls back to direct SQLite automatically.
> Use `SHOW TABLES` and `DESCRIBE <table>` for schema discovery — these
> work portably across both backends.

---

## 5-Minute Orientation

### Step 1 — Know your two docs

| File | Read when |
|------|-----------|
| `PRINCIPLES.md` | **Always, before anything.** Rules, error handling, tools. |
| `INDEX.md` | Routing: which slash command or skill to load. |

**Never read all skill files upfront.** They are loaded on demand.

### Step 2 — Know your tools

You have two ways to interact with the profile:
1. **High-level Analysis**: `/nsys:analyze`, `/nsys:diff`, `/nsys:mfu`
2. **Specific Skills**: Run `nsys-ai skill run <name> profile.sqlite`.

> **Note**: Navigation tools (`navigate_to_kernel`, `zoom_to_time_range`) are UI-only.
> Via CLI, provide timestamp references in text instead.

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
| External agent onboarding | `nsys-ai agent-guide` |

**Analysis Skills**: Use `nsys-ai skill run <name> profile.sqlite` to run specific
analysis modules. See [`commands/skill.md`](commands/skill.md) for the full catalog of 21 builtin skills.

> **After analysis**: Encode your conclusions as `findings.json` and open
> `nsys-ai timeline-web profile.sqlite --findings findings.json` so the user
> can visually verify your claims. See [`commands/evidence_schema.md`](commands/evidence_schema.md)
> for the JSON schema and end-to-end workflow.

> **Performance Note**: Running stateless CLI commands (`nsys-ai skill run`) on multi-gigabyte `.sqlite` profiles can take 30–60+ seconds per invocation due to DuckDB/SQLite cold starts and Parquet conversion. For heavy analysis workflows, batch your queries or expect high latency.

---

## Typical First 60 Seconds

When a user loads a profile with no specific question:

```
1. Load skills/triage.md (Read the workflow guide)
2. Parse NVTX hierarchy (e.g. via `tree` or `search`). Note: NVTX markers might be well-structured, poorly structured, or entirely missing.
3. Query top kernels by GPU time (e.g. via `top_kernels` skill)
4. Get GPU peak TFLOPS and NCCL presence
5. Give a 4-line summary + ask what to investigate
```

When a user asks "what's my MFU?":

```
1. Load skills/mfu.md (Read the workflow guide)
2. Get GPU peak TFLOPS
3. Discover NVTX/kernel names (never guess)
4. Resolve model architecture (lookup table before asking user)
5. Compute theoretical FLOPs → compute region MFU or standard MFU
6. Sanity check: MFU must be 0–100%
```

---

## Common Pitfalls (and How to Avoid)

| ❌ Wrong | ✅ Right |
|---------|---------|
| Compute FLOPs yourself | Use theoretical FLOPs calculator |
| Guess skill parameters (e.g. `num_heads`) | Read `commands/skill.md` strictly for required keys/Enums |
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
│   ├── skill.md       nsys-ai skill CLI + builtin catalog (21 Python skills)
│   └── agent-guide.md nsys-ai agent-guide (external agent onboarding)
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
