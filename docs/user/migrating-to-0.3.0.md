# Migrating to nsys-ai 0.3.0

This page is for users upgrading from 0.2.3 or an older checkout. It is an action guide, not a
second changelog: start with the first section that matches how you use `nsys-ai`, make the small
change if one is required, and then run the confirmation command.

The short version is reassuring:

- You do **not** need to re-capture an existing workload.
- Existing `.sqlite` profiles continue to work.
- Existing session directories continue to be readable.
- The main migration for most users is allowing the first `.nsys-rep` query to create a
  `.parquetdir` beside the capture.

The examples below use `PROFILE` for an existing profile path and `SESSION_ROOT` for a directory
you own for handoff artifacts. Replace those placeholders in your shell; they are not literal file
names.

## 1. If you have an `.nsys-rep`, allow the Parquet directory

### What changed

`.nsys-rep` is now the primary input for a new analysis. With the default `auto` ingest policy,
nsys-ai runs the Nsight export step once and creates a sibling directory:

```text
PROFILE                 input capture
PROFILE.parquetdir      derived Parquet tables used by analysis
```

For example, `run/profile.nsys-rep` produces `run/profile.parquetdir`. The first command can be
slower and needs enough disk space for the derived tables; later commands reuse that directory.
The directory is derived state, not a replacement for the capture. Keep both if you want to move or
archive the profile.

### Action

Usually there is nothing to change in a command line. Make sure:

1. the `nsys` executable that exported the capture is available on `PATH` when conversion is
   needed; and
2. the directory containing the capture is writable, or use a pre-built `.parquetdir` directly.

Do not add a manual `nsys export` step to every script unless you need to control the export
yourself. nsys-ai validates and reuses a complete sibling directory.

### Confirm

Run the health check first. It exercises the same input selection without starting a long analysis:

```console
$ nsys-ai doctor PROFILE --format json
```

Look for successful `.nsys-rep conversion`, `Parquet cache`, and `Schema compatibility` sections.
For a human-readable summary, omit `--format json`:

```console
$ nsys-ai doctor PROFILE
```

If conversion is not possible, the error names the missing `nsys` executable and the export step.
You can also hand the derived directory to later commands:

```console
$ nsys-ai diagnose PROFILE.parquetdir
```

See [what to hand nsys-ai](profile-inputs.md) for completeness checks, rebuild conditions, and the
supported input matrix.

## 2. If your automation requires SQLite, select the compatibility path

### What changed

The storage policy is now explicit. `auto` uses the Parquet path for `.nsys-rep` and opens an
existing `.sqlite` directly. `NSYS_AI_INGEST=sqlite` reproduces the older `.nsys-rep` to `.sqlite`
workflow.

The SQLite policy also has a deliberate side effect: it selects direct SQLite reads and disables
the Parquet query cache for that run, even if `NSYS_AI_CACHE_MODE=parquet` is set. This is useful
for comparing old and new results or for handing a SQLite export to another tool, but it is not the
fast path for repeated analysis.

### Action

Only scripts that require a SQLite artifact or intentionally reproduce the pre-0.3.0 execution
path need an environment change:

```console
$ NSYS_AI_INGEST=sqlite nsys-ai diagnose PROFILE.nsys-rep
```

Do not leave this variable exported globally. A command prefix limits compatibility mode to the
one invocation:

```console
$ NSYS_AI_INGEST=sqlite nsys-ai info PROFILE.nsys-rep
```

`NSYS_AI_INGEST=sqlite` refuses a `.parquetdir` input because that input has already selected the
Parquet representation. Use `NSYS_AI_INGEST=parquetdir` when you want to require a Parquet input and
fail fast on accidental SQLite paths.

### Confirm

The following should still print profile metadata, including the kernel and NVTX counts:

```console
$ NSYS_AI_INGEST=sqlite nsys-ai info PROFILE.sqlite
```

The normal `.sqlite` path also remains valid without the variable:

```console
$ nsys-ai info PROFILE.sqlite
```

The distinction between ingest selection and cache selection is documented in [environment
variables](environment-variables.md#storage-and-ingest).

## 3. If another process must continue the analysis, pass the session directory

### What changed

The session directory is now the handoff unit. `diagnose`, `propose`, `diff`, `review`, Web, TUI,
and MCP can publish or resume artifacts from the same directory. The directory contains the state
and the artifacts, rather than requiring the next process to reconstruct analysis from terminal
output:

```text
SESSION_ROOT/run-001/
├── session.json
├── findings.json       # after diagnose
├── proposal.json       # after propose
├── runspec.json        # when a run specification is available
├── diff.json           # after diff
└── logs/
```

### Action

For a portable handoff, pass an explicit directory path. Put it below a directory that the process
can create and own; a dedicated subdirectory under `/tmp` is suitable for a short-lived handoff:

```console
$ SESSION_ROOT=/tmp/nsys-ai
$ nsys-ai diagnose PROFILE.sqlite --session "$SESSION_ROOT/run-001"
$ nsys-ai review --session "$SESSION_ROOT/run-001"
```

The same directory can be handed to a later command from a different working directory:

```console
$ nsys-ai ask --session "$SESSION_ROOT/run-001" "what is the bottleneck?"
```

For a pair comparison, publish the diff and decision to the session:

```console
$ nsys-ai diff before.sqlite after.sqlite --session "$SESSION_ROOT/run-001"
$ nsys-ai diff --session "$SESSION_ROOT/run-001" --accept \
    --reason "verified on the steady-state iteration"
```

The bare form remains supported for local sessions. It treats the value as an id below
`.nsys-ai/sessions` in the current working directory:

```console
$ nsys-ai diagnose PROFILE.sqlite --session run-001
```

Use an explicit path when the producer and consumer run in different working directories or
containers. An older session directory is not a database migration target: open it with `review`
or the Web/TUI surface and let the current command read its existing artifacts.

### Confirm

After `diagnose`, verify that the handoff contains `session.json` and `findings.json`; after a diff,
verify `diff.json` as well:

```console
$ nsys-ai review --session "$SESSION_ROOT/run-001"
```

The review output names the current phase and the next command when the loop is not complete. See
[the user guide](../user-guide.md#4-diagnose) for a complete handoff walkthrough.

## 4. If a script uses `--trim`, handle an out-of-range window

### What changed

Most profile-backed commands now reject a window that selects no part of the capture. Older code
could receive an empty result and continue as if the profile had no matching events. 0.3.0 returns
`TRIM_OUT_OF_RANGE` instead, including the profile's capture-clock range.

`--trim` is measured on the Nsight capture clock, not from zero at the first kernel. The capture may
start at 60 seconds or 155 seconds, so a hard-coded `--trim 0 1` is often outside the data.

### Action

If your script treats an empty result as a valid answer, update it to handle exit status 2 and the
`TRIM_OUT_OF_RANGE` error. Then discover the actual range before choosing a window:

```console
$ nsys-ai info PROFILE.sqlite
$ nsys-ai skill run top_kernels PROFILE.sqlite --trim START_S END_S
```

For paired profiles, prefer a logical iteration when the capture has matching NVTX markers. Each
side resolves its own capture-clock range:

```console
$ nsys-ai diff before.sqlite after.sqlite --iteration 0
```

Do not replace a partially overlapping window: it is still valid and is clipped to the available
data. Only a window that selects nothing is rejected.

### Confirm

Use `info` to read the range, then run a window inside it. An intentionally invalid window should
fail loudly:

```console
$ nsys-ai skill run top_kernels PROFILE.sqlite --trim 0 1
Error [TRIM_OUT_OF_RANGE]: --trim ... selects no part of this profile...
```

The exact message includes the measured start and end seconds. See [time windows](time-windows.md)
for command coverage and the difference between `--trim` and `--iteration`.

## 5. If a script invokes `nsys-tui`, use the packaged CLI

### What changed

The old standalone `nsys-tui` entry point is not the 0.3.0 command surface. The TUI remains
available through the main executable, which keeps the CLI, TUI, Web, and MCP paths on the same
session and analysis contracts.

### Action

Replace the executable in scripts and documentation:

```console
$ nsys-ai tui PROFILE
```

For the horizontal timeline mode, use:

```console
$ nsys-ai timeline PROFILE
```

For browser automation or a remote workstation, use `timeline-web` instead. `nsys-ai open PROFILE`
still opens the default Web viewer.

### Confirm

Check the installed command surface without opening a viewer:

```console
$ nsys-ai --help
```

The help output lists `open`, `tui` and `timeline` as available surfaces. The aliases are not a
second implementation; they delegate into the shared runtime.

## 6. If you depended on the old Perfetto server

### What changed

The `nsys-ai perfetto` service and `open --viewer perfetto` were removed. They started a local
server for an external `ui.perfetto.dev` page, which required a separate network-connected UI and
could not participate in the session loop.

The export format remains. `nsys-ai export` still writes Chrome Trace Event JSON, which can be
opened in Perfetto or another compatible trace viewer.

### Action

Replace service startup with a file export:

```console
$ nsys-ai export PROFILE.sqlite --trim START_S END_S -o trace-export/
```

Then open `trace.json` in the trace viewer of your choice. If you need analysis and decisions,
use `diagnose`, `diff`, or `review` instead of a read-only trace service.

### Confirm

Check that the export file is created and contains JSON before handing it to a viewer:

```console
$ nsys-ai export PROFILE.sqlite --trim START_S END_S -o trace-export/
$ python -m json.tool trace-export/trace_gpu0.json >/dev/null
```

## 7. Adopt the new diagnostics and comparison commands when useful

These commands are additions, not mandatory migration steps. They are useful replacements for
scripts that previously assembled several lower-level commands and parsed human-readable output.

| Need | 0.3.0 command | Stable output to automate on |
|---|---|---|
| Check the machine and profile before analysis | `nsys-ai doctor PROFILE --format json` | JSON health sections and exit status |
| Build expensive derived state before a batch | `nsys-ai warm PROFILE` | cache and NVTX-map status |
| Run deterministic default analysis | `nsys-ai diagnose PROFILE --format json` | findings envelope and skipped analyses |
| Compare before/after profiles | `nsys-ai diff BEFORE AFTER --format json --no-ai` | verdict, confidence, profile IDs, deltas |
| Gate a regression in CI | `nsys-ai diff BEFORE AFTER --exit-on-regression` | exit status |
| Keep a named baseline | `nsys-ai baseline tag NAME PROFILE` | stable `baseline:NAME` reference |
| Resume a recorded decision | `nsys-ai review --session SESSION` | phase and next action |

For a CI job, prefer JSON or exit status over parsing terminal colours and prose:

```console
$ nsys-ai doctor PROFILE --format json > doctor.json
$ nsys-ai diff BEFORE AFTER --format json --no-ai -o diff.json \
    --exit-on-regression
```

If the profiles cannot be compared, 0.3.0 treats an inconclusive result as a failed gate rather
than silently passing an empty comparison. Read [reading a diff](reading-a-diff.md) before changing
the threshold.

### Named baselines in CI

The baseline store defaults to `.nsys-ai-baselines` in the current working directory. CI should
set an absolute root so separate steps agree even when their working directories differ:

```console
$ export NSYS_AI_BASELINE_ROOT=/var/lib/nsys-ai/baselines
$ nsys-ai baseline tag main BEFORE.sqlite --reason "release baseline"
$ nsys-ai diff --against baseline:main AFTER.sqlite --format json --no-ai
```

Use an artifact or cache policy for that directory; it is a local store, not a remote registry.

## What did not change

These are compatibility guarantees for this release:

### Existing `.sqlite` inputs remain supported

You can keep passing a `.sqlite` to `info`, `doctor`, `diagnose`, `skill`, `diff`, `export`, and the
viewer commands. No conversion or re-capture is required. The default `.sqlite` path may build a
query cache beside the file; select `NSYS_AI_INGEST=sqlite` when you specifically need direct
SQLite behaviour.

### Existing captures do not need to be re-recorded

The ingest change is an analysis-time representation choice. A `.nsys-rep` from an older Nsight
Systems run is still an input; 0.3.0 converts it when needed. Re-capture only when the profile
itself lacks the trace data needed for the question.

### Existing session directories remain readable

Session artifacts are versioned and validated. A session from an earlier run can be passed to
`review --session SESSION` or opened by the matching Web/TUI workflow. If an artifact is too old for
the current reader, the command reports the artifact and schema version instead of silently
reinterpreting it. Keep the original directory when investigating a migration issue.

### Core analysis concepts remain the same

The profile still contains Nsight Systems events, skills still produce evidence, and `diff` still
compares before and after captures. The release changes the storage and handoff paths around that
analysis; it does not require changing the workload or its NVTX instrumentation.

## Upgrade checklist

Use this checklist in a workstation or CI image upgrade:

- [ ] Install or pin `nsys-ai==0.3.0`.
- [ ] Confirm `nsys --version` is available when `.nsys-rep` conversion is expected.
- [ ] Run `nsys-ai doctor PROFILE --format json` and archive the result for CI.
- [ ] Confirm the profile directory has room for `PROFILE.parquetdir`.
- [ ] Remove any global `NSYS_AI_INGEST=sqlite` unless compatibility mode is intentional.
- [ ] Replace `nsys-tui` with `nsys-ai tui` or `nsys-ai timeline`.
- [ ] Replace `open --viewer perfetto` with `nsys-ai export` if a trace file is needed.
- [ ] Update scripts that treat an empty `--trim` result as success.
- [ ] Give cross-process jobs an explicit session directory below a shared writable root.
- [ ] Use `--format json` and exit-status gates for machine consumers.

## Further reading

- [What to hand nsys-ai](profile-inputs.md) — input and ingest semantics
- [Time windows](time-windows.md) — capture-clock trimming and iteration selection
- [Environment variables](environment-variables.md) — ingest, cache, baseline, and tuning controls
- [The user guide](../user-guide.md) — one complete diagnose-to-decision workflow
- [Reading a diff](reading-a-diff.md) — verdicts, confidence, and CI interpretation
- [Changelog](../../CHANGELOG.md) — the complete 0.3.0 release record
