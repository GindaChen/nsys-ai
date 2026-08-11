# nsys-ai Agent Principles

**Source of truth for all nsys-ai agents.** Read this file before acting.
All skill files and slash commands reference this; never duplicate its content.

---

## Purpose

This harness provides a standard operating procedure (SOP) and toolkit for AI agents
performing GPU performance analysis using `nsys-ai`. The goal: let agents reliably
analyze NVIDIA Nsight Systems profiles (`.sqlite` files), compute efficiency metrics,
and diagnose performance issues without requiring a display or manual inspection.

---

## Key Principles

- **Use the real tools** — Never compute FLOPs, MFU, or time deltas yourself.
  Always call `compute_theoretical_flops`, `compute_mfu`, or `compute_region_mfu`.
  Manual arithmetic on large numbers is unreliable.
- **Discover before acting** — Never guess NVTX region names or kernel names.
  Always query `NVTX_EVENTS` or `StringIds` first.
- **The profile is a hard dependency** — If no profile is loaded, the tools will error.
  Fail clearly: tell the user what profile is needed and why.
- **Fail loudly** — Unambiguous error messages let the agent self-correct.
  Never silently return a wrong answer (e.g. MFU > 100% must be flagged as an error).
- **Provide introspection first** — Triage before computing. Run `skills/triage.md`
  to understand what's in the profile before running MFU or diff analysis.
- **JSON is ground truth** — Tool outputs are JSON. Parse them; do not summarize
  numbers from prose. Pass `theoretical_flops` directly from tool output to the next tool.
- **Time is always in nanoseconds** in the profile database. Divide by 1e6 for ms, 1e9 for s.
- **`theoretical_flops` is never in the profile.** It must be computed from model
  architecture parameters using `compute_theoretical_flops`.
- **Correlate with Source Code** — If the user's local source code is available alongside the profile, always cross-reference NVTX regions and anomalous kernels with the actual Python files to propose highly specific, line-level code fixes.

---

## Performance

- **DuckDB + Parquet is the default backend.** On first open, `nsys-ai` exports the
  SQLite profile into `<profile>.nsys-cache/` as ZSTD-compressed Parquet files.
  Subsequent opens are sub-second. SQL queries run via DuckDB views over cached Parquet.
  If DuckDB is unavailable, `nsys-ai` falls back to direct SQLite access automatically.
- **Large profiles (>100 MB) still benefit from trimming.** Even with DuckDB, full-table
  scans on 250 MB+ profiles can be slow. Narrow the analysis window:
  - **CLI**: add `--trim START_S END_S` to `skill run` or other commands.
  - **Manual SQL**: add `WHERE k.start >= <start_ns> AND k.[end] <= <end_ns>`.
  - **Best practice**: profile 1–2 representative iterations, not the entire run.
- **Costliest skills on big files**: `nvtx_kernel_map` / `nvtx_layer_breakdown`
  (sort-merge attribution across NVTX×Runtime×Kernel) and
  `gpu_idle_gaps` (window function over all kernels). Trim before running these.
- **Auto-indexing**: `nsys-ai` automatically creates indexes (`_nsysai_*`) on
  first skill execution. This is a one-time cost (~30 s for 250 MB) that speeds
  up all subsequent queries.

---

## Non-Negotiable Rules

1. **MFU > 100% is always wrong.** The FLOPs scope is too wide. Stop, recompute with
   narrower `operation`. Do not report a value > 100%.
2. **Never use `SELECT *`.** Always name the specific columns you need.
3. **In diff mode: `search_nvtx_regions` before `get_region_diff`.** Never pass a
   guessed NVTX name to `get_region_diff`.
4. **Never use whole-profile kernel span as single-step time** when multiple iterations
   exist. Use NVTX iteration markers instead.
5. **Skip iteration index 0 in diff analysis.** JIT compilation inflates the first
   iteration. Use index ≥ 1.
6. **Root cause statements are required.** Never say "it got slower." Always state:
   what changed, what the evidence is, and what to do about it.
7. **End your message before waiting for user input.** When you need model architecture
   parameters, ask → end message → wait. Do not proceed without the answer.

---

## Error Handling

| Error signal | Required action |
|-------------|-----------------|
| `MFU > 100%` | Recompute with narrower operation; explain the error |
| `kernel_count = 0` | Report KERNEL_NOT_FOUND; suggest trying `source="kernel"` |
| `is_aligned=false` in diff | Warn user; switch to `get_global_diff` |
| `Hardware_Warning=true` | Note thermal throttle; advise re-run before concluding |
| `JIT_Compilation_Warning=true` | Skip iteration index 0; re-run with index ≥ 1 |
| `iteration_count = 1` | Profile too short; ask user to re-profile with ≥ 3 iterations |
| GPU unknown in `get_gpu_peak_tflops` | Ask user for peak_tflops (BF16, dense, no sparsity) |
| SQLite error | Check that profile is loaded; verify column names with `PRAGMA table_info` |

---

## UI vs CLI Tooling

If you are operating within `timeline-web` or the `chat` TUI, you can use **Navigation tools** (`navigate_to_kernel`, `zoom_to_time_range`, `fit_nvtx_range`) to control the user's viewport.
These are NOT available via CLI (`nsys-ai agent ask` or `nsys-ai skill run`). When using CLI, omit UI navigation
commands and provide precise timestamp references in your output text instead.

All agents (UI or CLI) are expected to emit structured findings for the timeline viewer (via `submit_finding` tool if internal, or `findings.json` file if external).
See `commands/evidence_schema.md` for the JSON schema.

---

## Skill File List (LLM Workflow Guides)

Load skill files on demand. Do not pre-load all of them.
These are **reasoning workflows** for the LLM agent, not executable code.

> **Builtin analysis skills** (executable via `nsys-ai skill run`) are documented
> separately in [`commands/skill.md`](commands/skill.md). Those are 30 Python builtin skills
> for targeted analysis (e.g. `top_kernels`, `gpu_idle_gaps`, `root_cause_matcher`).

| When the user asks… | Load this file |
|--------------------|----------------|
| "What's my MFU?" / "How efficient is my flash attention?" | `skills/mfu.md` |
| "What's in my profile?" / "What's the bottleneck?" | `skills/triage.md` |
| "Why did it slow down?" / "Compare before vs after" | `skills/diff.md` |
| "How is my NCCL?" / "Distributed training is slow" | `skills/distributed.md` |
| "Why do some steps spike?" / "High step-to-step variance" | `skills/variance.md` |
| SQL syntax / query help | `skills/sql.md` |

---

## Acceptance Criteria (Verification Checklist)

After completing any analysis, verify:

- [ ] MFU values are between 0% and 100%
- [ ] `theoretical_flops` was computed by `compute_theoretical_flops`, not estimated
- [ ] NVTX or kernel names came from a query, not guessed
- [ ] Step time came from a single representative iteration (not full profile span)
- [ ] Diff analysis skipped iteration 0
- [ ] Any diff root cause statement includes: cause, evidence field+value, recommendation
- [ ] Root cause recommendations include specific Python file names and target code blocks when local source code is accessible
- [ ] No `SELECT *` was used
- [ ] Time values were converted from ns (÷ 1e6 for ms, ÷ 1e9 for s)

---

## Adding a New Skill

To add a new analysis capability:

1. Create `skills/<name>.md` following the **Skill Template** in `skills/SKILL_TEMPLATE.md`
2. Add an entry to the **Skill File List** table above
3. Add a slash command in `commands/<name>.md` if the skill warrants a top-level entry
4. Update `INDEX.md` routing table

**The skill file MUST include:**
- A one-sentence purpose statement at the top
- A complete step-by-step workflow with exact SQL and tool calls
- An Acceptance Criteria section with verifiable checks
