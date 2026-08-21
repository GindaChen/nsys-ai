# User guide

This guide takes one workload from a capture to a recorded decision, entirely on the command
line. Every command below was run against a real capture; the outputs are real, trimmed only for
length.

If you would rather drive the same workflow in a browser, see
[Guided loop setup](guided-loop-setup.md). That page covers the web UI; this one covers the CLI.

## Install

```bash
pip install nsys-ai
```

Analysis and the web server use only the standard library plus `duckdb`, `pyarrow`, `rich` and
`textual`. The AI features (`ask`, `chat`, `agent`) are optional extras and are not used anywhere in
this guide:

```bash
pip install 'nsys-ai[agent]'    # only if you want the AI commands
```

## 1. Check the environment first

`doctor` tells you what works before you spend a capture finding out.

```console
$ nsys-ai doctor
```

It reports three groups: required dependencies, optional features, and — when you pass a profile —
that profile's health.

```console
$ nsys-ai doctor myprofile.sqlite
Profile health
  Schema compatibility           [OK  ]  export schema 3.25.0
  Duration                       [OK  ]  0.3s
  GPUs                           [OK  ]  2
  GPU model                      [WARN]  unknown
                                    -> GPU model missing from CUPTI TARGET_INFO; MFU / efficiency cannot be computed.
  NVTX events                    [OK  ]  19118 events
  NCCL events                    [OK  ]  66 kernels
  Profiler overhead              [OK  ]  0.7%

Summary: 15 ok, 1 warning, 0 failed, 3 not configured, 1 skipped
```

Read the warnings before trusting a number downstream. "GPU model unknown" above is the reason MFU
cannot be computed for that capture — the tool says so rather than reporting a fabricated figure.

Use `--strict` to make warnings a non-zero exit, which is what you want in CI.

## 2. Capture

`nsys-ai profile` wraps `nsys` with defaults suited to ML workloads. Put wrapper options before the
`--`, and your workload after it:

```console
$ nsys-ai profile -o run-before -- ./my-workload --steps 20
[preparing] 0.0s
[capturing] 0.0s
[exporting] 2.4s
[validating] 2.4s
[finished] 2.4s
Report: run-before/profile.nsys-rep
SQLite: run-before/profile.sqlite
RunSpec: run-before/runspec.json
Profile ID: nsys2:sha256:<64 hex chars>
Export schema: 3.25.0
Nsight version: <your nsys version>
Kernels: 20
```

Four things land in `run-before/`:

| File | What it is |
|------|------------|
| `profile.nsys-rep` | The capture itself. Hand this to any command. |
| `profile.sqlite` | A SQLite export of the same capture, for tools that want one. |
| `runspec.json` | Exactly how this run was launched. Section 5 explains why this matters. |
| `stdout.log`, `stderr.log` | Your workload's own output. |

The **Profile ID** is a content hash. It is how the tool knows two artifacts describe the same
capture, so you never have to name sessions yourself.

`nsys-ai profile` validates the export before reporting success: a capture that produced no kernel
activity is an error, not a silent empty file. Add `--dry-run` to print the plan without running
anything.

If you already have a `.nsys-rep`, a `.sqlite` or a `parquetdir` from `nsys` directly, skip this step.
Every command below accepts any of them; see [what to hand nsys-ai](./user/profile-inputs.md) for how
each is read.

## 3. Look at it

```console
$ nsys-ai info run-before/profile.sqlite
Profile: run-before/profile.sqlite
  Nsight version (heuristic): <version>
  GPUs: [0, 1]
  Kernels: 4010  |  NVTX: 19118
  Time: 60.112s - 60.411s

  GPU 0:  | Kernels=1993 | Streams=[7]
  GPU 1:  | Kernels=2017 | Streams=[15]
```

For an interactive view, `nsys-ai open <profile>` starts the web timeline; `nsys-ai tui` and
`nsys-ai timeline` are terminal equivalents.

### Warming the cache

The first query against a profile builds a Parquet cache next to it (`<name>.nsys-cache/`, which is
gitignored). Later commands open in a fraction of a second instead of re-reading SQLite.

You can pay that cost up front rather than inside your first real query:

```console
$ nsys-ai warm run-before/profile.sqlite
Profile: run-before/profile.sqlite
Cache:   run-before/profile.nsys-cache
  base tables: 14 parquet files (already built; opened in 0.04s)
  nvtx kernel map: 4270 rows (built, 0.34s)
warmed in 0.38s
```

This is worth doing before a demo, before a batch of skill runs, or on a large capture where the
first build would otherwise appear as an unexplained pause. It is never required — every command
builds what it needs on demand.

## 4. Diagnose

`diagnose` runs the local analyzers — SQL and heuristics, no LLM — publishes the findings to a
session, and tells you the exact command to run next.

```console
$ nsys-ai diagnose run-before/profile.sqlite
  [highlight] Kernel hotspot: axpy(float, float *, float *, int) (98%)
      Kernel 'axpy' is 98% of total kernel time (0ms over 20 invocations). Threshold 60%.
  [highlight] Insufficient NVTX coverage: no NVTX annotations
      Profile cannot anchor iteration / region analysis. Downstream skills will have reduced fidelity.
-- Limitations / skipped analyses (4) --
  slow_iterations (iteration_timing) - skipped: This profile has no NVTX_EVENTS table, so it carries
    no NVTX annotation. Iteration detection needs annotated ranges - re-capture with NVTX enabled,
    or annotate the workload, to use this skill.
Next: nsys-ai propose --session <id> --finding-id kernel_instance_gpu0_stream7_562419326 --runspec <runspec.json>
Or open the same session: nsys-ai diagnose --session <id> --web
```

Three things to notice, because they are deliberate:

- **Skipped analyses are reported, not hidden.** A skill that cannot run says so and says why. An
  empty result and "this did not apply to your capture" are different answers, and the tool keeps
  them different. They are listed under limitations, not buried.
- **The session id is derived from the profile's content hash**, so you never name one yourself and
  re-running against the same capture reopens the same session. It lives under `.nsys-ai/sessions/<id>/`.
- **The next command is printed with its arguments filled in**, including the finding id. You do not
  have to read `findings.json` to find one.

Add `--web` to open the same session in the browser without recomputing anything.

For a cross-process handoff, pass the session directory itself instead of an id:

```console
$ nsys-ai diagnose run-before/profile.sqlite --session /tmp/nsys-run-001
$ nsys-ai review --session /tmp/nsys-run-001
$ nsys-ai ask --session /tmp/nsys-run-001 "what is the bottleneck?"
```

The directory is the contract: its `session.json` and artifacts are opened by the CLI, Web, and
TUI from any working directory. A bare value such as `--session run-001` remains supported and
continues to mean `.nsys-ai/sessions/run-001` relative to the current directory.

The Web viewer exposes the same ask workflow as JSON at `POST /api/ask`. Pass either a profile path
or a session handoff directory; the response includes the selected registered skills, structured
evidence, and the evidence-first answer:

```console
$ curl -s http://127.0.0.1:8144/api/ask \
    -H 'Content-Type: application/json' \
    -d '{"session_id":"run-001","session_root":"/tmp/nsys-run-001","question":"what is the bottleneck?","use_llm":false}'
```

`use_llm` is optional. When enabled and a provider is configured, the model may triage or summarize
the registered evidence; it never replaces the runner's skill execution or writes profile SQL.
When the request is opened with a session, the response also returns `session_log: "logs/ask.jsonl"`;
each completed ask is appended there as a handoff record without changing the session artifact
manifest.

The NVTX tree viewer loads bounded slices from `GET /api/tree`. `depth` controls levels,
`limit` caps children per node, and `max_nodes` caps the complete response. A response with
`"truncated": true` is intentionally partial; nodes may also carry `has_more` and
`children_total` so clients do not mistake the visible children for the complete profile.

The optional MCP transport exposes the same read-only handoff as `get_session`; it returns the
SessionStore projection rather than a second MCP-specific session schema.

Two lower-level entry points remain available when you want them: `nsys-ai evidence build <profile>`
runs the same analyzers and writes findings JSON to stdout or `-o` (add `--session` to publish, and
`--analyzers` to pick a subset), and `nsys-ai skill list` / `nsys-ai skill run <name> <profile>` run a
single analysis on its own.

## 5. Propose a change

A proposal names one finding, states what to change, and — this is the part that matters — records
how the change will be verified.

```console
$ nsys-ai propose --session --profile run-before/profile.sqlite \
    --finding-id profile_kernel_hotspot \
    --runspec run-before/runspec.json
Proposal published to session nsys2_sha256_<...>
```

The `--runspec` argument is the `runspec.json` that `nsys-ai profile` wrote in step 2. It is how the
tool knows the "after" run can be made comparable to the "before" run — same argv, same environment
policy, same trace options.

**Without it, the proposal abstains:**

```console
$ nsys-ai propose --session --profile run-before/profile.sqlite --finding-id profile_kernel_hotspot
Proposal published to session nsys2_sha256_<...>
Abstained: verification RunSpec is required
```

That is the intended behaviour, not a failure. An abstained proposal is recorded honestly as "I
cannot stand behind this without a way to check it" rather than presented as advice. It also cannot
carry the loop forward: publishing an after-profile requires a proposal that did not abstain.

If you captured your baseline with plain `nsys` rather than `nsys-ai profile`, you will not have a
`runspec.json`. Re-capture with `nsys-ai profile` to get one.

## 6. Change something, capture again, compare

Make your change, then capture the candidate the same way:

```console
$ nsys-ai profile -o run-after -- ./my-workload --steps 20
```

Then compare:

```console
$ nsys-ai diff run-before/profile.sqlite run-after/profile.sqlite --no-ai
Per-GPU Overview
GPU |  Before Total |   After Total |          Δ |    Overlap % (B→A)
----+---------------+---------------+------------+-------------------
  0 |      391.57us |        1.20ms |  +805.27us |       0.0% →  0.0%

Top regressions (GPU 0)
    Δ Time  |  Count Change | Kernel
------------+---------------+-------------------------------
 +805.27us  |        20->60 | axpy
```

`--no-ai` keeps the comparison purely deterministic. `diff` reports a verdict and a comparability
score alongside the numbers; a low comparability score means the two captures differ enough that the
delta should not be read as a result. See [Reading a diff](./user/reading-a-diff.md) for the verdict
definitions, metric bases, and the recovery path for an inconclusive comparison.

Useful variants:

- `--session` publishes the diff into the same session directory.
- `diagnose run-after/profile.sqlite --against run-before/profile.sqlite`
  publishes candidate-anchored regression Findings that can be sent directly
  to `propose`; see [the diff findings contract](./dev/diff-findings.md).
- `--gpu N` restricts to one device; the default compares all of them.
- `--against baseline:<name>` diffs against a stored baseline instead of a path — see
  `nsys-ai baseline --help`.
- `nsys-ai diff-web <before> <after>` opens the same comparison in a browser.

## 7. Record the decision

A comparison you do not act on is a comparison you will re-run next month. Record the outcome with
the reason:

```console
$ nsys-ai diff run-before/profile.sqlite run-after/profile.sqlite --no-ai --session \
    --reject --reason "3x the launches for 2x the time - not the win we wanted"
Diff published to session nsys2_sha256_<...>
Decision 'rejected' recorded in session nsys2_sha256_<...>
```

This writes a `decision` block into the session's `diff.json` with the status, the reason, a
timestamp, and the decider taken from your git `user.email`, and moves the session to its final
phase. Use `--accept` for the other outcome. Either flag requires `--reason`, and a session accepts
exactly one decision.

Drop `--session` and the same flags write a `diff.json` in the working directory instead — or wherever
`--decision-out PATH` names, which is what a CI job wants so the checkout it is testing stays clean.
This is the right
choice for a one-off comparison you are not tracking as a session.

## What the session directory holds

Using `--session` throughout leaves a single directory describing the whole investigation:

```text
.nsys-ai/sessions/<profile-content-id>/
  session.json     phase, both profile references, and a sha256 for every artifact
  findings.json    what was measured
  proposal.json    what to change, and how it will be verified
  runspec.json     how to reproduce the run
  diff.json        before/after comparison, verdict, and your decision
```

`session.json` records a hash of each artifact, so a file edited by hand is detectable rather than
silently trusted.

Only the short publication step takes the session writer lock. Analysis runs before that lock is
acquired; concurrent writers receive a conflict instead of overwriting another transport's
publication. Artifact files are written atomically and the store's recovery journal restores a
coherent snapshot after an interrupted publication.

## Things worth knowing

- **A proposal needs a RunSpec to stand behind itself.** If you captured the baseline with plain
  `nsys`, there is no `runspec.json` and `propose` will abstain. Re-capture with `nsys-ai profile`.
- **The decision records your git `user.email`.** If you keep separate identities for separate work,
  check `git config user.email` in the directory you run from.
- **`nsys-ai help` is a starting point, not the full command list.** `nsys-ai --help` is complete.

## Where to go next

| Topic | Page |
|-------|------|
| Driving the same loop in a browser | [Guided loop setup](guided-loop-setup.md) |
| What `doctor` checks, and why | [doctor](doctor.md) |
| Annotating your code so iteration analysis works | [NVTX annotations](03-nvtx-annotations.md) |
| Capturing a representative window of a long run | [Focused profiling](08-focused-profiling.md) |
| Profiling PyTorch specifically | [Python / PyTorch](06-python-pytorch.md) |
| Instruction-level drill-down on a hot kernel | [CUTracer](cutracer-instruction-analysis.md) |
