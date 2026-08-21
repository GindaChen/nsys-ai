# nsys-ai documentation

Two audiences, kept apart. **User guide** is for operating the tool; **developer guide** is for
changing it. The **Nsight Systems reference** section holds excerpts of NVIDIA's own documentation,
kept as plain markdown so the agent can query them.

New here? Start with the [user guide](./user-guide.md), which walks a capture end to end.

## User guide

Operating the tool.

| Page | Topic |
|------|-------|
| [user-guide.md](./user-guide.md) | nsys-ai end to end: capture, diagnose, propose, diff, decide |
| [user/profile-inputs.md](./user/profile-inputs.md) | The three inputs it accepts, and what conversion happens |
| [user/known-limits.md](./user/known-limits.md) | Trust boundaries, denominators, unsupported evidence, and current Web limits |
| [user/migrating-to-0.3.0.md](./user/migrating-to-0.3.0.md) | Upgrade actions for users coming from 0.2.3 |
| [user/time-windows.md](./user/time-windows.md) | Narrowing analysis with `--trim` and `--iteration` |
| [user/reading-a-diff.md](./user/reading-a-diff.md) | What a diff verdict means, how to recover from low comparability, and which metrics to trust |
| [user/troubleshooting.md](./user/troubleshooting.md) | Symptoms, what they mean, and what to change |
| [user/environment-variables.md](./user/environment-variables.md) | Every variable, its default, and when to change it |
| [guided-loop-setup.md](./guided-loop-setup.md) | The same loop driven from the timeline web UI |
| [doctor.md](./doctor.md) | Environment and profile health checks |
| [ci-diff-gate.md](./ci-diff-gate.md) | Capture, compare, and gate performance regressions in CI |
| [cutracer-instruction-analysis.md](./cutracer-instruction-analysis.md) | Instruction-level drill-down (`cutracer` + `cutracer_analysis`) |
| [cutracer-modal.md](./cutracer-modal.md) | Running CUTracer on Modal (serverless GPU, no local GPU needed) |

## Developer guide

Changing the tool. These describe contracts rather than usage, and assume you are reading the source
alongside them.

| Page | Topic |
|------|-------|
| [dev/ingest-policy.md](./dev/ingest-policy.md) | How a profile path becomes an open backend, and why call sites must not bypass it |
| [dev/skill-contract.md](./dev/skill-contract.md) | How to add a portable, deterministic, JSON-safe analysis skill |
| [dev/surface-adapters.md](./dev/surface-adapters.md) | How CLI, TUI, Web, MCP, and chat delegate shared policy and analysis |
| [support-matrix.md](./support-matrix.md) | Committed export schemas verified by CI |

## Nsight Systems reference

Excerpts of NVIDIA's documentation, for looking up a flag, a table, or an API without leaving the
repository.

| Page | Topic | Source |
|------|-------|--------|
| [01-cli-reference.md](./01-cli-reference.md) | CLI commands, flags, and example command lines | User Guide |
| [02-sqlite-schema.md](./02-sqlite-schema.md) | SQLite export schema and common queries | Exporter Docs (v2022.4) |
| [03-nvtx-annotations.md](./03-nvtx-annotations.md) | NVTX API for instrumenting applications | User Guide |
| [04-cuda-trace.md](./04-cuda-trace.md) | CUDA trace types and GPU memory analysis | User Guide |
| [05-nccl-tracing.md](./05-nccl-tracing.md) | NCCL collective communication tracing | User Guide |
| [06-python-pytorch.md](./06-python-pytorch.md) | Python and PyTorch profiling support | User Guide |
| [07-container-profiling.md](./07-container-profiling.md) | Docker/container profiling setup | User Guide |
| [08-focused-profiling.md](./08-focused-profiling.md) | Limiting scope with cudaProfilerApi and NVTX capture ranges | User Guide |
| [09-performance-questions-mfu.html](./09-performance-questions-mfu.html) | Curated performance questions and MFU calculation playbook | Project Guide |

### Version awareness

Nsight Systems evolves across releases: the export schema, CLI flags, and available trace types all
change between versions. Files in this section are annotated with the source version where
applicable. What to watch for:

- Table names and columns change between major versions
- New trace types are added (advanced NCCL tracing in 2025.6.1, for example)
- CLI flags are deprecated or renamed
- New `--pytorch` options appear in recent versions

Check `nsys --version` on the target system and cross-reference the versioned documentation at
`https://docs.nvidia.com/nsight-systems/<VERSION>/`. For what this project has actually tested
against, see the [support matrix](./support-matrix.md).

**Source URLs**

- User Guide: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- SQLite Exporter (2022.4): https://docs.nvidia.com/nsight-systems/2022.4/nsys-exporter/examples.html
- Latest Exporter: https://docs.nvidia.com/nsight-systems/nsys-exporter/index.html

## How the agent uses this

1. **Context seeding** — relevant files are loaded as context when the agent starts a profiling task
2. **Query answering** — files are searched for specific CLI flags, SQL queries, or API patterns
3. **Version tracking** — compared against the actual nsys version to detect capability differences
4. **Workflow templates** — example commands and SQL queries copied directly into profiling scripts
