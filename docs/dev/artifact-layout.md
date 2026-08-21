# Artifact layout

This document defines where `nsys-ai` writes files and why the locations are
not all governed by the same root. It is intentionally explicit so a CI job
can keep its checkout clean without moving profile-keyed caches to a shared
namespace.

## The two storage classes

`nsys-ai` writes two kinds of local state:

| Class | Identity | Default location | Relocatable by |
|---|---|---|---|
| Invocation artifacts | the command/session being executed | `.nsys-ai/` plus the historical standalone `diff.json` | `NSYS_AI_ARTIFACT_ROOT` |
| Input-keyed caches | the immutable profile content/path | `<profile>.nsys-cache/` | no shared root; see below |

The distinction is a correctness boundary. A cache beside `perf.sqlite` is
invalidated and reused according to that profile; moving two unrelated
`perf.sqlite` files into one directory would create a collision and change
that identity model.

## Default layout

With no environment override, existing invocations keep their historical
locations:

```text
checkout/
├── .nsys-ai/
│   ├── locks/                 # session writer/state locks
│   ├── profiles/<timestamp>/  # default `nsys-ai profile` capture
│   └── sessions/<id>/         # session.json and findings/proposal/diff artifacts
└── diff.json                  # default standalone accepted/rejected decision
```

Explicit destinations continue to win:

- `--session /path/to/handoff` uses that directory as the handoff location;
- `-o PATH` keeps the existing command-specific output path;
- `--decision-out PATH` keeps the standalone decision at `PATH`.

## Relocating invocation artifacts

Set `NSYS_AI_ARTIFACT_ROOT` to an absolute or working-directory-relative path:

```bash
export NSYS_AI_ARTIFACT_ROOT="$PWD/.nsys-ai-run"
nsys-ai diagnose profile.sqlite --session training-001
nsys-ai profile -- python train.py
nsys-ai diff before.sqlite after.sqlite --accept --reason "verified"
```

The resulting layout is:

```text
.nsys-ai-run/
├── locks/
├── profiles/<timestamp>/
├── sessions/training-001/
└── decisions/diff.json
```

The variable is read at command/runtime resolution time, not import time.
Relative values are resolved against the command's working directory. This is
important for embedding callers and tests that pass a synthetic `cwd`.

The precedence for the outputs covered by this policy is:

1. an explicit command destination (`--session DIR`, `-o`, or
   `--decision-out`);
2. `NSYS_AI_ARTIFACT_ROOT`;
3. the unchanged built-in defaults above.

Project configuration can be layered above the environment in a future config
issue; this resolver deliberately keeps that policy in one place.

## Cache policy

Parquet caches remain beside their source input:

```text
profile.sqlite
profile.nsys-cache/
├── *.parquet
└── .cache_version
```

`nsys-ai doctor profile.sqlite` reports the actual registered cache path when
one is open and the profile-support section states the policy even without a
profile. `NSYS_AI_ARTIFACT_ROOT` does not move this directory. For a read-only
input, use `--no-cache` where the command supports it or set
`NSYS_AI_CACHE_MODE=direct`; disabling the cache is different from relocating
it.

## CI pattern

Use a job-local directory and retain it as an artifact when debugging:

```bash
export NSYS_AI_ARTIFACT_ROOT="$RUNNER_TEMP/nsys-ai-artifacts"
nsys-ai diagnose "$PROFILE" --session ci-diagnose
nsys-ai diff "$BEFORE" "$AFTER" --accept --reason "CI baseline"
test -f "$NSYS_AI_ARTIFACT_ROOT/decisions/diff.json"
```

The repository under test receives no default session, capture, lock, or
decision files. The input profile's cache remains next to that input by design.

## Implementation seam

`src/nsys_ai/artifact_root.py` owns resolution. `SessionStore` and the
transport-neutral `session_cli` facade use its session resolver; the profile
wrapper uses its profile resolver; the diff CLI uses its decision resolver.
This keeps CLI, TUI, Web, MCP, and embedding callers on the same session
policy without introducing another storage database or server.
