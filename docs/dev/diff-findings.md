# Diff findings and the candidate handoff boundary

This document defines how a deterministic profile diff becomes actionable
evidence for the next optimization loop. It is an implementation contract for
the CLI, SessionStore, proposal generator, and future Web/TUI consumers.

The short version is:

```text
baseline + candidate
        │
        └── diff_profiles()
              │
              └── findings_from_diff()
                    │
                    └── candidate-seeded session
```

The diff is a pair computation. A Finding is a claim about one profile. The
candidate is therefore the next session's `before_profile`, even though the
Finding carries lineage back to the baseline that exposed the regression.

## Scope

This contract covers:

- per-kernel regression Findings from `ProfileDiffSummary.top_regressions`;
- the `diagnose --against REF` CLI path;
- deterministic headroom and suggested actions;
- proposal and SessionStore handoff semantics.

It does not turn the overall step-time verdict, overlap delta, or category
delta into a Finding. Those values remain diff-level measurements. A Proposal
must point to a concrete candidate trace selection, so a whole-run Finding
would be both hard to verify and unsafe to optimize.

## User workflow

The candidate profile is the positional argument. `--against` supplies the
baseline:

```console
$ nsys-ai diagnose run-after/profile.sqlite \
    --against run-before/profile.sqlite \
    --format json \
    --session artifacts/diagnose-regression
```

The baseline may also be a named snapshot:

```console
$ nsys-ai baseline tag main run-before/profile.sqlite \
    --reason "last known good"
$ nsys-ai diagnose run-after/profile.sqlite \
    --against baseline:main
```

The resulting `findings.json` contains the ordinary default-pack findings
plus any per-kernel regression Findings. Every diff Finding is anchored to the
candidate profile. A user can select its id for the existing proposal flow:

```console
$ nsys-ai propose --session artifacts/diagnose-regression \
    --finding-id finding_diff_regression_<id> \
    --runspec runspec.json
```

If `--session` is omitted, normal diagnose derives a session id from the
candidate profile id. `diagnose --against` derives one from the content-derived
`diff_id` instead. This prevents a baseline comparison from silently replacing
a normal diagnosis session for the same candidate.

## Why the candidate owns the Finding

Session publication has a deliberately strict invariant:

```text
session.state.before_profile.profile_id
    == findings.json.profile_id
    == finding.selection.profile_id
```

For an ordinary diagnosis, all three values identify the opened profile. For a
baseline comparison, the opened profile is the candidate, so the invariant
still holds. The baseline is retained in `finding.diff_lineage` rather than
being made the Finding's profile identity.

This gives the next loop a coherent state transition:

```text
baseline ──compare──> candidate
                         │
                         ├── findings.json (claims about candidate)
                         ├── proposal.json (candidate selection)
                         └── reprofile candidate ──> after profile
```

Publishing the Finding into a session owned by the baseline would violate the
provenance check and would make the subsequent `propose → reprofile → diff`
transition ambiguous. The implementation intentionally does not weaken that
check.

## Finding projection

`findings_from_diff(summary)` is a pure function. It accepts an in-memory
`ProfileDiffSummary` and performs no profile reads, cache writes, or session
writes. For each item in `top_regressions` with a positive `delta_ns`, it
creates one `Finding` with these fields:

| Finding field | Source or rule |
|---|---|
| `type` | `region` |
| `label` | `Regression: <kernel name>` |
| `selection` | The diff's candidate-side `TraceSelection` |
| `selection.profile_id` | `summary.after.profile_id` |
| `selection.source` | `diff` |
| `diff_lineage.diff_id` | `summary.diff_id` |
| `diff_lineage.role` | `regression` |
| `diff_lineage.rank` | Zero-based rank in `top_regressions` |
| `diff_lineage.baseline_profile_id` | `summary.before.profile_id` |
| `headroom_ms` | `kernel.delta_ns / 1e6` |
| `headroom_basis` | `baseline_delta` |
| `severity` | `warning` |
| `category` | `compute` |

The existing diff engine already computes a slowest-instance selection on the
after profile. That selection is reused, including its time bounds and GPU
filter. The pure projector has a deterministic fallback for hand-built
summaries whose kernel row lacks a selection; it never opens a profile to
invent one.

The final list is passed through `rank_findings`, so the largest positive
baseline delta is first. Finding ids are content-derived from the diff id,
kernel key, and rank. They are stable for the same comparison and distinct for
different ranked rows.

## Headroom semantics

`headroom_ms` is not a promise of realized speedup. It is the observed increase
in aggregate GPU time relative to the selected baseline:

```text
headroom_ms = candidate_total_ns - baseline_total_ns
               -----------------------------------
                             1,000,000
```

The explicit basis `baseline_delta` is copied into `ExpectedImpact` when a
Proposal is generated. This makes the proposal honest: it carries a measured
opportunity, not a model prediction or a claim that all overlapping GPU time
can be removed.

Aggregate kernel time can overlap across streams and devices. The Finding's
false-positive notes preserve that caveat for CLI, Web, and MCP consumers.
The selection is the verification target; it is not a source-code location.

## Suggested-action decision table

Actions are deterministic projections of the count and average-time deltas.
They are deliberately phrased as investigation steps, not automatic fixes.

| Classification | Count delta | Average delta | Action focus |
|---|---:|---:|---|
| `new` | any | any | Identify what introduced the kernel; it is absent from baseline |
| existing | `> 0` | `<= 0` | Call count rose; inspect loop and batching structure |
| existing | `0` | `> 0` | Same count, slower calls; compare shapes, dtypes, and occupancy |
| existing | non-zero | non-zero | Separate call-frequency cost from per-call cost |
| existing | `0` | `<= 0` | Inspect the workload path and aggregate attribution |

The second row is important. A regression caused by more launches should not
be narrated as “the kernel got slower.” The decomposition tells the next
investigation which dimension to measure.

## Session and proposal lifecycle

`diagnose --against` performs the expensive analysis before acquiring a
SessionStore writer lease:

1. Resolve and open the candidate.
2. Run the normal `DIAGNOSE_DEFAULT` evidence pack.
3. Resolve the baseline path or `baseline:<name>` snapshot.
4. Run the deterministic baseline → candidate diff.
5. Project candidate Findings and rank the combined report.
6. Derive the session id from `diff_id` unless the caller supplied one.
7. Publish one candidate-owned `findings.json` artifact.

The proposal path is unchanged. A Finding with an id, candidate selection,
suggested action, and a verification `RunSpec` produces a non-abstained
Proposal. The normal SessionStore transition then remains:

```text
diagnose → propose → reprofile → diff → decide
```

No diff Finding is written to `diff.json`. That artifact has an exact diff
schema and remains the representation of the pair comparison. The Finding is
an additive member of `findings.json`, where the evidence and proposal
contracts already live.

## Compatibility and failure behavior

- No schema version bump is needed. `Finding` already has optional selection,
  lineage, headroom, and suggested-action fields.
- A pair with no positive top regression returns no diff Findings. It does not
  fabricate a healthy whole-run Finding.
- A low-comparability pair may still expose per-kernel deltas, but the Finding
  carries the diff's confidence and the baseline-relative caveat. Callers must
  continue to respect the diff verdict and confidence before treating the
  regression as causal.
- An invalid profile or missing named baseline fails the diagnose command
  before publication. No partial session artifact is created by this path.
- An explicit `--session` remains authoritative. Only the default session id
  changes for `--against`.

## Test contract

The implementation is covered at four levels:

| Layer | Assertion |
|---|---|
| Pure projection | Reversed fixtures produce 15 ranked positive headrooms, stable ids, lineage, and candidate selections |
| Action semantics | Count-only regressions mention call frequency, not kernel slowdown |
| Proposal | A projected Finding plus `RunSpec(argv=("true",))` is non-abstained and uses `baseline_delta` |
| Session | Candidate-owned findings accept a RunSpec and reach `propose` |
| CLI | `diagnose --against` writes a diff-derived session id and candidate provenance |
| No-regression path | Forward fixture pair on GPU 0 returns `[]` |

When changing this contract, run the focused suite first:

```console
$ pytest -q tests/test_diff_findings.py tests/test_session_cli.py \
    tests/test_diagnose_review.py tests/test_gpu_selection.py
```

Then run the full repository suite before merging. The fixture pair is small
enough for deterministic tests; large captures should be reserved for the
checkpoint E2E, not embedded in unit tests.
