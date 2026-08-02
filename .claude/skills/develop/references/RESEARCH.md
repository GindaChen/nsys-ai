# Research sources, and facts already settled

Two purposes: point at sources that have actually paid off, and record conclusions so
they are not re-derived (or re-derived *wrong*) every session.

## Look it up rather than infer

Nsight Systems changes its SQLite schema between releases, and GPU-performance
accounting has conventions that are easy to invert. Both are areas where a confident
guess produces output that looks right and is wrong. When a change depends on either,
verify against a real profile or a primary source before building on it.

```bash
python -m nsys_ai skill run schema_inspect <profile.sqlite>   # what this export actually has
sqlite3 <profile.sqlite> ".tables"
```

## Primary sources

| Source | Use for |
|---|---|
| NVIDIA Nsight Systems docs + release notes | Export schema changes, table renames, CLI flags |
| `docs/02-sqlite-schema.md`, `docs/sqlite-explorer/` | This repo's own map of the schema |
| `docs/root-causes/` | The Book of Root Causes — existing cause→fix write-ups |
| NCCL docs | Collective semantics, communicator/topology fields |
| PyTorch Profiler / HTA (HolisticTraceAnalysis) | Accounting conventions, metric definitions |

## Prior art already cited by ROADMAP.md

Read the implementation, not the README, and say explicitly what you are taking and
what you are rejecting:

- [NAV](https://github.com/eshama1/NSYS-Analyzer-and-Visualizer) — comparative analysis
- [nsys_recipes](https://github.com/hyxcl/nsys_recipes) — overlap matrix, per-stream NCCL
- [nsys_easy](https://github.com/harrism/nsys_easy) — ergonomics of the capture wrapper
- [Profiling-AI-Software-Bootcamp](https://github.com/openhackathons-org/Profiling-AI-Software-Bootcamp)

## Settled facts — do not re-derive

- `overlap_ms` counts as **compute**, per HTA. Exposed communication is
  `exposed_comm_ms`. There is no `communication_ms`.
- Diff verdict gate: ±5% on step_time, with comparability ≥ 0.5. Per-bucket defaults
  are already defined in `diff.py` — read them before proposing new thresholds.
- Package naming: PyPI `nsys-ai`, Python module `nsys_ai`. `nsys_tui` was a pre-rename
  name and is gone; both `nsys-ai` and `nsys-tui` CLI entry points remain.
- Runtime deps are `duckdb` + `pyarrow` (Parquet cache) and `rich` + `textual` (TUI).
  SQL analysis and the web server are stdlib. "Zero runtime dependencies" is stale and
  was corrected — do not reintroduce the claim.
- CUTracer: v0.2.1 is pinned and latest; output is byte-identical v0.1.0 → main. Two
  known parser gaps remain (no `cycles` column; `iterN_` infix).

## Writing the research note

Four headings, short. It exists to be read by the reviewer and by the next session:

```markdown
## Problem
<what breaks, for whom, restated from the issue — not copied>

## What already exists
<files/skills/commands found, and why they do not cover it>

## What I had to look up
<schema facts, conventions, API behavior — with the source>

## Prior art
<project → what it does → taking / not taking, and why>
```

If the work spans sessions, save it as `docs/notes/<topic>.md`. Follow the repo doc
style: reader-first, no emoji in new docs, no enumeration of PR numbers, and the
content must match the code as it is now.
