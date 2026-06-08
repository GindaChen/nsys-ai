# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Before/after drill-down in the diff agent: compare a kernel's launch
  configuration (grid/block/registers/shared memory) and memory profile (peak
  VRAM and allocation/free deltas), and locate each top kernel regression to a
  specific GPU, stream, and time window.
- `nsys-ai diff --exit-on-regression` exits non-zero on a likely-regression
  verdict, so a diff can gate CI.
- `nsys-ai diff --format json` emits a versioned envelope: a top-level
  `verdict`, a `comparability_confidence` score, step-time `category_attribution`
  (compute / communication / idle), and a content-derived `profile_id` per side.
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

### Fixed

- A missing profile path fails immediately with `ProfileNotFoundError` and a
  non-zero exit, instead of creating an empty database and cache directory and
  then reporting a misleading schema error.
- The bare `nsys-ai <profile>` shortcut again opens the web timeline.
- `profile_health_manifest` profiler-overhead reporting no longer emits
  impossible values, scopes overhead to the analyzed window, and uses
  wall-clock NCCL time for the communication-dominance trigger.
- Multi-GPU single-node profiles are detected as distributed; overlap analysis
  reports which devices are present and warns when no device is specified.
- Parquet cache builds are serialized so concurrent runs against the same cache
  no longer crash.

### Performance

- SQL sweep-line overlap analysis (about 3x faster on the analysis call), NVTX
  layer breakdown reduced from ~43s to ~1.4s, and automatic trimming of long
  profiles to a representative window.

### Docs

- Guide for running CUTracer on Modal, plus a real `--trace-size-limit-mb` flag
  and a loud warning when SASS resolution fails.
- Evidence-schema reference updated to v0.1, and the performance-budget guidance
  rewritten for NVTX-heavy profiles.

## [0.2.3] — 2026-05-12 and earlier

See `git log v0.2.3` for changes prior to the introduction of this
changelog.

[Unreleased]: https://github.com/GindaChen/nsys-ai/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/GindaChen/nsys-ai/releases/tag/v0.2.3
