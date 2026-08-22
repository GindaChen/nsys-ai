# Architecture and release-maintainer guide

This page is the implementation map for nsys-ai 0.3.0. It describes the
contracts that let CLI, TUI, Web, and MCP share one analysis path, then records
the release boundary so a future change does not accidentally reopen the
architecture migration.

The short version is:

> **Many entrances, one brain, one session for handoff.**

This is a contract document, not a tour of every Python module. For input
selection, start with [ingest policy](ingest-policy.md). For adding a skill,
read the [skill contract](skill-contract.md). For surface-specific wiring, use
[surface adapters](surface-adapters.md).

## 1. Design goals

The 0.3.0 architecture has four operational goals:

1. **One analysis implementation.** A question asked in the CLI, browser,
   TUI, or MCP must execute the same registered skills and the same grounding
   policy.
2. **Evidence before language.** SQL and deterministic heuristics produce the
   evidence. An LLM may select skills or summarize rows, but it does not invent
   profile SQL or replace a missing result.
3. **A directory is the handoff.** A session contains inspectable artifacts,
   not an opaque second database. Another process can resume it without parsing
   the previous process's stdout.
4. **Costs follow changed inputs.** Input-derived state is cached and keyed by
   its source identity and answer-affecting parameters. Work that is not
   invalidated should not be repeated merely because a different surface asked
   the question.

Non-goals for this release are a remote execution service, a second
orchestrator package, and a new database hidden behind the session directory.

## 2. Runtime layers

The stack has five layers and two analysis engines. Dependencies flow downward;
the transport layers do not implement analysis policy.

| Layer | Responsibility | 0.3.0 boundary |
| --- | --- | --- |
| **L5 Transport** | CLI, TUI, Web, MCP input/output | Four verbs and session handoff; no surface-local SQL |
| **L4 Agent core** | Shared runner and lifecycle | `agent/runner.py` owns the loop; legacy CLI loop is a thin adapter |
| **L3 Engines** | Single-profile and pair analysis | `AnalysisKernel` runs packs/skills; `DiffIndex` is the pair-memo extension point |
| **L2 Session** | Handoff artifacts and state | `findings`, `proposal`, `runspec`, `diff`, and indexes are inspectable files |
| **L1 Ingest** | Turn profile inputs into queryable data | `.nsys-rep` defaults to Parquet; `.sqlite` remains a compatibility path |

The two engines have different invalidation domains:

- **AnalysisKernel** is invalidated by the profile identity, selected skill,
  and every parameter that can change the answer (GPU, trim window, iteration,
  schema mode, and relevant feature flags).
- **DiffIndex** is invalidated by the ordered before/after identities and pair
  options. Its persisted pair memo is deliberately deferred from the 0.3.0
  release gate until a real-profile checkpoint proves the invalidation and
  reconciliation costs.

The current cache and memo rules are documented in
[artifact layout](artifact-layout.md) and
[skill memo identity](skill-memo-fingerprint.md).

## 3. The transport contract

Every transport should be explainable as one of four verbs:

| Verb | Input | Engine | Primary output |
| --- | --- | --- | --- |
| `diagnose` | One profile | AnalysisKernel + default pack | `findings.json` |
| `ask` | Question plus profile/session | Triage + AnalysisKernel | Evidence-first text or SSE |
| `diff` | Ordered before/after profiles | DiffIndex/canonical diff | `diff.json` or rendered report |
| `optimize_step` | Baseline plus RunSpec | Diagnose → propose → re-profile → diff | Complete session |

The command names are product verbs, not four independent implementations.
The Web and TUI may expose buttons and panels, but the action they invoke must
map back to one of these verbs. MCP exposes the same handoff projection over
stdio; it must not introduce an MCP-only session schema.

### Session location

`--session <dir>` is the portable form. A caller may pass a directory below a
shared writable root, and a later caller may open that directory from a
different working directory. A bare session value remains a compatibility
shortcut for the local `.nsys-ai/sessions` root.

The session directory is not a replacement for the profile cache. The profile
cache is input-derived query state; the session is an invocation and decision
handoff. Keep those ownership boundaries separate:

```text
profile.nsys-rep
└── profile.parquetdir/       input-derived query state

session/run-001/
├── session.json              state, profile refs, artifact manifest
├── findings.json             ranked evidence
├── proposal.json             proposed change and verification
├── runspec.json              reproducible capture specification
├── diff.json                 pair result and optional decision
├── indices/                  derived indexes when present
└── logs/                     append-only transport records
```

See [artifact layout](artifact-layout.md) for the compatibility and atomic
write rules. A new artifact must be versioned, hashed in the session manifest,
and written atomically before a transport reports success.

## 4. One runner, one lifecycle

All four verbs use the same lifecycle. The planner chooses work; the engines do
the work; the emitter translates the result into a surface response.

```text
Load → ORIENT → PLAN → INVESTIGATE → GROUND → SYNTHESIZE → EMIT
```

### Load

Resolve the profile or session location, apply the ingest policy, and load the
existing artifacts. If a session already contains a valid result, resume from
that result instead of rebuilding it.

### ORIENT

Read profile metadata, health, available schema/features, and session state.
This is where an absent NVTX table, an empty kernel side, or an out-of-range
trim becomes an explicit limitation rather than a healthy-looking empty answer.

### PLAN

`diagnose` uses the fixed default pack. `ask` chooses at most four registered
skills through keyword matching or optional LLM triage. `diff` ensures the pair
inputs are resolved. `optimize_step` follows the recorded proposal RunSpec.

### INVESTIGATE

Execute `run_skill` through the registry. Grounding SQL belongs to the skill
definition and its declared schema requirements. Chat code may request a skill,
but must not assemble ad-hoc profile SQL.

### GROUND

The runner accepts an answer as grounded only when it has usable registry
evidence. Abstention is preserved with its reason; an empty result means a
check ran and found nothing, while an abstention means the check could not
make a defensible claim. See [skill contract](skill-contract.md).

### SYNTHESIZE

The deterministic renderer formats findings and diffs. An LLM is optional and
limited to planning/triage and language synthesis over already-returned rows.
It cannot turn missing evidence into a diagnosis or silently change a metric's
denominator.

### EMIT

Batch CLI commands write files and print a concise next action. Web/TUI use the
same runner and emit `text`, `action`, and `done` SSE events where streaming is
needed. MCP returns the same session projection and JSON-safe evidence.

## 5. Dataflow by verb

### Diagnose: one profile

```text
profile
  → ingest policy
  → run_pack(DIAGNOSE_DEFAULT)
  → Finding[] + limitations
  → findings.json
  → CLI text / Web overlay / MCP JSON
```

The default pack is deterministic and does not require an LLM. A finding must
retain its stable ID, category, severity, confidence, evidence, and time/GPU
location when those values are available.

### Ask: a question over evidence

```text
question
  → root_cause_matcher
  → ≤4 registered skills
  → run_skill × N
  → require registry evidence
  → evidence-first answer
  → stdout or SSE
```

If no profile is connected, profile-backed tools are removed from the tool set
and the prompt must not require them. If a skill abstains, the answer should
name that limitation rather than treating it as a clean profile.

### Diff: an ordered pair

```text
before.ref ──┐
             ├─ resolve identities → canonical diff → diff.json
after.ref  ──┘                         ↓
                              Web Decide / CLI --accept|--reject
```

The before/after order is part of the input. A self-diff is valid but should be
reported as inconclusive when it cannot establish a change. A missing or empty
side is not an improvement. Candidate-anchored findings from
`diagnose --against` are described in [diff findings](diff-findings.md).

### Optimize step: the composed path

```text
diagnose baseline
  → propose one finding with a RunSpec
  → capture candidate from that RunSpec
  → diff baseline/candidate
  → record accept/reject and reason
```

The session is the recovery boundary. If the process stops after capture, the
next invocation reads the proposal and existing references rather than
capturing a second candidate by accident. The composition must not duplicate
the implementation of diagnose, propose, or diff.

## 6. Extension boundaries

### Add an analysis skill when

The question can be answered from profile evidence and should be reusable by
more than one surface. Define the SQL, schema requirements, parameters,
formatter, abstention behavior, and tests in the skill contract. Register it;
do not call it by importing a private module from chat or Web.

### Add a transport adapter when

The user needs a different interaction model, not a different analysis. The
adapter should translate input into a verb, call the runner, and translate
events/artifacts into its protocol. It should not add a fifth HTTP server or a
second session database.

### Add an artifact when

The state is needed for a later process to inspect or resume. Define its schema,
version, producer, consumer, hash/atomic-write behavior, and migration story.
Do not put transient UI state into the session unless it affects the analysis
or the handoff.

### Keep LLMs at the edge

Use an LLM to triage a natural-language question or summarize grounded rows.
Keep skill selection bounded, validate every selected name through the
registry, cap serialized evidence, and preserve abstention rows. A prompt is
not a database API.

## 7. What shipped in 0.3.0

The required architecture migration is complete:

- canonical skill packs and registry execution;
- evidence-first chat tools backed by `run_skill`;
- the shared agent runner and thin CLI/Web/TUI adapters;
- Parquet-first `.nsys-rep` ingest with `.sqlite` compatibility;
- session handoff for findings, proposals, RunSpecs, diffs, and decisions;
- four-verb transport semantics across CLI, TUI, Web, and read-only MCP;
- bounded Web NVTX tree slices and first-class health/abstention reporting;
- deterministic diff findings and calibrated idle severity;
- the `profile`, `doctor`, `diagnose`, `propose`, `review`, and `optimize`
  command surfaces.

The following are intentionally outside the 0.3.0 completion claim:

- persisted pair-level `DiffIndex` memoization, pending a real delta-cost and
  invalidation checkpoint;
- remote verification providers and remote session storage;
- a separate diagnostics artifact beyond the current findings/session records.

These are follow-up roadmap items, not reasons to keep the shipped architecture
in an unreleaseable state. The live roadmap records the promotion criteria.

## 8. Release-maintainer checklist

For the reusable release procedure, including candidate branches, artifact
verification, publishing, patch backports, and post-release handoff, see the
[release process](./release.md). The checklist below remains the short
architecture-specific gate for this release boundary.

Run this sequence from a clean checkout of the release candidate. Do not place
credentials or tokens in logs, release notes, or issue comments.

### Candidate checks

```console
$ python -m nsys_ai --help
$ python -m build
$ python -m twine check dist/*
$ python -m pip install --force-reinstall dist/nsys_ai-*.whl
$ nsys-ai --help
$ python -c 'from nsys_ai.mcp_server import main; print(main.__module__)'
```

Inspect the wheel before installing it: it must contain the templates, prompts,
agent skill context, and documentation metadata required by the runtime. Verify
the two console scripts from `pyproject.toml`; the MCP entry point starts a
stdio server and intentionally has no standalone `--help` mode. Do not assume
a source checkout proves the wheel is correct.

### Four-verb smoke test

Use a small committed fixture for the fast gate and a real `.nsys-rep` or
`.sqlite` capture for the release checkpoint when available:

```console
$ nsys-ai doctor PROFILE --format json
$ nsys-ai diagnose PROFILE --session SESSION_DIR --format json
$ nsys-ai ask --session SESSION_DIR "what is the main bottleneck?"
$ nsys-ai diff BEFORE AFTER --session SESSION_DIR --no-ai --format json
$ nsys-ai review --session SESSION_DIR
```

Confirm that the session contains the expected artifacts and that a deliberate
inconclusive or abstention result is visible as such. For the full test layers,
see [testing](testing.md). For user-facing migration notes, see
[migrating to 0.3.0](../user/migrating-to-0.3.0.md).

### Publish and verify

1. Confirm `pyproject.toml`, `CHANGELOG.md`, and the release tag agree on the
   version and date.
2. Build the wheel and source distribution from the intended commit.
3. Create the Git tag and GitHub Release only after the candidate checks pass.
4. Wait for the publish workflow, then install the released version in a clean
   environment and repeat `--help`, `doctor`, and one deterministic diff.
5. Record the PyPI/GitHub links and the known limitations in the release issue.

The release is complete only when the artifact installed from the package index
passes the same smoke test as the source checkout.
