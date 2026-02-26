# Nsight Systems — Agent Knowledge Base

> **Purpose:** Machine-readable documentation for the nsys-agent. Each file covers one topic from the official NVIDIA documentation, saved as plain markdown for easy querying, embedding, and retrieval.

## ⚠️ Version Awareness

Nsight Systems evolves across releases. The SQLite export schema, CLI flags, and available trace types change between versions. Files in this knowledge base are annotated with the source version where applicable.

**Key version differences to watch for:**
- SQLite table names and columns may change between major versions
- New trace types are added (e.g., advanced NCCL tracing in 2025.6.1)
- CLI flags may be deprecated or renamed
- New `--pytorch` options added in recent versions

Always check `nsys --version` on the target system and cross-reference with the versioned docs at `https://docs.nvidia.com/nsight-systems/<VERSION>/`.

## File Index

| File | Topic | Source |
|------|-------|--------|
| [01-cli-reference.md](./01-cli-reference.md) | CLI commands, flags, and example command lines | User Guide |
| [02-sqlite-schema.md](./02-sqlite-schema.md) | SQLite export schema and common queries | Exporter Docs (v2022.4) |
| [03-nvtx-annotations.md](./03-nvtx-annotations.md) | NVTX API for instrumenting applications | User Guide |
| [04-cuda-trace.md](./04-cuda-trace.md) | CUDA trace types and GPU memory analysis | User Guide |
| [05-nccl-tracing.md](./05-nccl-tracing.md) | NCCL collective communication tracing | User Guide |
| [06-python-pytorch.md](./06-python-pytorch.md) | Python and PyTorch profiling support | User Guide |
| [07-container-profiling.md](./07-container-profiling.md) | Docker/container profiling setup | User Guide |
| [08-focused-profiling.md](./08-focused-profiling.md) | Limiting profile scope with cudaProfilerApi and NVTX capture ranges | User Guide |

## How This Knowledge Base Is Used

1. **Agent context seeding** — Load relevant files as context when the agent starts a profiling task
2. **Query answering** — Search files for specific CLI flags, SQL queries, or API patterns
3. **Version tracking** — Compare against actual nsys version to detect capability differences
4. **Workflow templates** — Copy example commands and SQL queries directly into profiling scripts

## Source URLs

All content is derived from official NVIDIA documentation:
- **User Guide:** https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- **SQLite Exporter (2022.4):** https://docs.nvidia.com/nsight-systems/2022.4/nsys-exporter/examples.html
- **Latest Exporter:** https://docs.nvidia.com/nsight-systems/nsys-exporter/index.html
