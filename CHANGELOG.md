# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-08-22

### Removed

- `nsys-ai perfetto` and `open --viewer perfetto`. The command served the trace
  from a local port with `Access-Control-Allow-Origin: *` so that
  `ui.perfetto.dev` — a page this project does not run — could fetch it, which
  meant the feature needed an internet connection on machines that often have
  none, and the wide-open CORS header existed only to serve it. It could not
  take part in the loop either: it received a flat list of kernels and NVTX
  spans, read-only. **`nsys-ai export` is unchanged** — Chrome Trace Event JSON
  is a format rather than a service, and the exported file can still be opened
  in Perfetto by anyone who wants to. `open --viewer` now defaults to `web`.

### Added

- `nsys-ai optimize <profile> --repo <path> -- <command>` carries one baseline
  profile to a recorded decision in a single command: diagnose, propose, capture
  the verification run from the proposal's own RunSpec, diff, decide. It writes
  no new analysis — every step already existed and this composes them. It stops
  before re-profiling when the proposal abstains, exits non-zero when the loop
  does not complete, and resumes from the session store rather than from memory,
  so an interrupted run does not repeat the capture it already made.
- `nsys-ai diagnose <profile>` and `nsys-ai review` are thin verbs over the
  evidence pack and the canonical diff. `diagnose` publishes findings to a
  session and prints the next command with its arguments filled in; `review`
  resumes a session and reports where its decision stands, or compares a pair
  without owning a session. Its pair report is rendered by the same canonical
  diff path as `diff --no-ai`; the report on stdout is identical, while
  `review` adds two next-step hints on stderr for an interactive shell.
- `nsys-ai diff --session --accept/--reject --reason TEXT` records the decision
  on the session's own `diff.json`, so a session that has findings, a proposal
  and a diff can reach a decision without leaving the command line.
- CLI, TUI, Web and MCP now use the same analysis runner and session handoff
  artifacts. A question can enter through different surfaces without creating a
  second analysis implementation or a second session database.
- `.nsys-rep` inputs use the parquetdir ingest path by default, while `.sqlite`
  remains supported as the compatibility path. `doctor` reports the selected
  ingest and cache health before analysis.
- `nsys-ai chat --session` and the CLI/Web/TUI ask paths persist grounded
  handoff records so a later surface can resume from the same session.

- `nsys-ai warm <profile>` builds the Parquet cache and the NVTX kernel map in
  one step, so the stack sweep is paid deliberately up front rather than by
  whoever issues the first NVTX-attribution query. It reports how long each half
  took, tells "warmed" apart from "already warm", and exits non-zero with the
  reason when the cache cannot be written.
- `nsys-ai baseline` keeps a local store of named profile snapshots (`tag`,
  `list`, `show`) so `diff --against baseline:<name>` resolves a stable name to a
  known snapshot instead of a fragile path. The store lives in
  `.nsys-ai-baselines/` and can be relocated with `NSYS_AI_BASELINE_ROOT` for
  cross-job CI use.
- `nsys-ai doctor` diagnoses the environment (Python, nsys CLI, Parquet cache,
  AI provider, CUTracer toolchain including the nvdisasm/framework CUDA match)
  and, when given a profile, its health (duration, GPUs, GPU model, NVTX and
  NCCL presence, profiler overhead). It stays a fast triage — single-digit
  seconds even on multi-gigabyte, multi-GPU profiles — and defers the slow NCCL
  eager/inductor call-mode split to `--deep`. Text and `--format json` output, a
  `--strict` flag, and a non-zero exit on failures so it can gate CI. The same
  report is reusable in-process by the web GUI and TUI, and the analysis skill
  uses it as a preflight.
- Before/after drill-down in the diff agent: compare a kernel's launch
  configuration (grid/block/registers/shared memory) and memory profile (peak
  VRAM and allocation/free deltas), and locate each top kernel regression to a
  specific GPU, stream, and time window.
- `nsys-ai diff --exit-on-regression` exits non-zero on a likely-regression
  verdict, and on an inconclusive one, so a diff can gate CI. A comparison that
  could not be made has not shown the absence of a regression — a capture with no
  GPU kernel activity is the clearest case.
- `nsys-ai diff --format json` emits a versioned envelope: a top-level
  `verdict`, a `comparability_confidence` score, step-time `category_attribution`
  (compute / communication / launch overhead / idle), and a content-derived
  `profile_id` per side. Launch overhead is the exposed kernel-dispatch latency
  — the GPU-idle time spent waiting on a launch call — carved out of idle, so
  the buckets still sum to step time and the verdict is unchanged.
- Structured v0.1 findings — category, severity, confidence, and a located time
  window where possible — for `overlap_breakdown`, `nccl_breakdown`,
  `top_kernels`, `profile_health_manifest`, and `kernel_instances`, so the agent
  and GUI consume them without per-skill parsing.
- New analysis skills: `code_attribution_candidates` maps a selected time window
  back to likely source/config regions; `nccl_compile_context_breakdown`
  classifies NCCL kernels as eager vs. compiled to point at the right fix; and
  `nccl_payload_breakdown` decodes NVTX payloads into NCCL message sizes, peer
  ranks, and communicator IDs.
- A labeled-profile evaluation harness under `tests/eval/` (expected and
  forbidden findings) to keep skill outputs honest as they change.
- Memoized skill execution keyed by the profile identity and every
  answer-affecting parameter, so repeated evidence requests can reuse results
  without confusing a changed window or GPU selection with the old answer.
- Candidate-anchored regression Findings from `diagnose --against`, giving the
  proposal workflow a stable finding ID and evidence window to act on.

### Changed

- Diff aggregates across all GPUs by default. With no device specified it now
  sums every device (kernels, overlap, memory copies, and stream counts)
  instead of silently scoping to GPU 0, matching the documented `--gpu` default.
- CUTracer is pinned to a reproducible upstream revision
  (`facebookexperimental`, `v0.2.1`).
- `nvtx_kernel_map` documents temporal-containment semantics: `nvtx_text` is the
  leaf NVTX label, and ancestor-path containment is temporal, not lexical, so
  matching on a path substring can pick up kernels whose enclosing scope merely
  happened to still be open.
- Idle findings calibrate their severity against the aggregate profile share,
  while retaining conservative warning severity when the aggregate denominator
  is unavailable.

### Scope boundaries

- The persisted pair-level `DiffIndex` memo remains deferred. The canonical diff
  and session handoff are shipped; promotion of pair reuse requires a measured
  checkpoint on real before/after captures.
- Remote verification providers and a separate diagnostics artifact remain
  follow-up work. The 0.3.0 session is the local, inspectable handoff boundary.

### Fixed

- A capture that recorded no GPU kernels is no longer reported as a large
  improvement. Comparability drops to zero, the report states which side is
  empty, and `--gate` exits non-zero: a comparison that could not be made has
  not shown the absence of a regression. The LLM narrative is refused before the
  model is consulted, so it cannot narrate the vanished kernels as a win beneath
  the refusal. `--gate` / `--exit-on-regression` now also exit non-zero on any
  inconclusive verdict, which will fail CI jobs that previously passed silently
  on incomparable pairs.
- `arithmetic_intensity` refuses an MFU above 100% instead of classifying it as
  healthy. Achieved throughput above the peak it is measured against is
  arithmetically impossible, so it abstains and names which input to check; a
  non-positive FLOP count or peak abstains for the same reason.
- `doctor`'s RunSpec check looks at the profile instead of reporting "no" for
  every one, and its hint names a flag that exists.
- The getting-started screen lists the optimization loop and shows forms that
  run: five commands were advertised without a required `--trim`.

- A missing profile path fails immediately with `ProfileNotFoundError` and a
  non-zero exit, instead of creating an empty database and cache directory and
  then reporting a misleading schema error.
- The bare `nsys-ai <profile>` shortcut again opens the web timeline.
- Empty or out-of-range windows and ambiguous multi-GPU root-cause ratios now
  abstain with an explicit reason instead of producing a healthy-looking
  finding or silently dropping the affected evidence.
- Web ask binds before background NVTX warmup, and bounded tree slices report
  truncation metadata instead of returning an unbounded fan-out.
- `profile_health_manifest` profiler-overhead reporting no longer emits
  impossible values, scopes overhead to the analyzed window, and uses
  wall-clock NCCL time for the communication-dominance trigger.
- Multi-GPU single-node profiles are detected as distributed; overlap analysis
  reports which devices are present and warns when no device is specified.
- Parquet cache builds are serialized so concurrent runs against the same cache
  no longer crash.
- Iteration timing separates real training iterations from sub-iteration NVTX
  noise: a loose marker that matched hundreds of short op ranges used to
  contaminate the median so every real iteration looked thousands of percent
  slow. `iteration_timing` and `iteration_detail` now judge variance against the
  real iterations only.
- `overlap_breakdown` only flags a same-stream compute/NCCL collision when a
  meaningful fraction of both actually share a stream, instead of firing on a
  single stray kernel.
- `profile_health_manifest` top kernels now also carry `kernel_name` /
  `invocations`, matching the field names the rest of the codebase uses.

### Performance

- Served responses are compressed when the client accepts it, cutting a cold
  `timeline-web` load from 2,184,087 to 272,501 bytes (8.0x) and the Perfetto
  trace from 1,022,424 to 100,320. Negotiated, never assumed: a client that does
  not advertise gzip gets the plain body.

- SQL sweep-line overlap analysis (about 3x faster on the analysis call), NVTX
  layer breakdown reduced from ~43s to ~1.4s, and automatic trimming of long
  profiles to a representative window.
- The `nvtx_kernel_map` build sweeps one capture thread at a time and streams
  both sides of each, so what it holds in Python tracks the batch rather than
  the profile. On a 3.5 GB capture, peak memory for the build fell from 7.50 GB
  to 3.55 GB — and from 5.28 GB to 1.39 GB inside the sweep itself — producing
  the same map.

### Docs

- Migration guide for users upgrading from 0.2.3: [migrating to 0.3.0](docs/user/migrating-to-0.3.0.md),
  including input, session, trim, CLI, and automation changes.
- Guide for running CUTracer on Modal, plus a real `--trace-size-limit-mb` flag
  and a loud warning when SASS resolution fails.
- Evidence-schema reference updated to v0.1, and the performance-budget guidance
  rewritten for NVTX-heavy profiles.

## [0.2.3] — 2026-05-12 and earlier

See `git log v0.2.3` for changes prior to the introduction of this
changelog.

[Unreleased]: https://github.com/GindaChen/nsys-ai/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/GindaChen/nsys-ai/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/GindaChen/nsys-ai/releases/tag/v0.2.3
