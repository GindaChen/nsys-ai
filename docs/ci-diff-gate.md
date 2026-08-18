# CI performance-regression gating

Use `nsys-ai diff --gate` when a profile comparison should be a merge gate. The
command is deterministic by default in CI: it computes the diff, writes the
same JSON verdict a reviewer can inspect, and returns a non-zero status when a
regression cannot be accepted.

The workflow has four inputs:

1. a known-good baseline profile;
2. a candidate profile captured by the current job;
3. a threshold passed to the existing `diff --gate` engine;
4. the generated `diff.json` uploaded as a CI artifact.

The baseline and candidate must use the same workload, capture window, and
comparison scope. A different workload or a failed profiling step is not an
improvement: an inconclusive comparison fails a configured gate.

## Capture or reuse profiles

If the job already produces an Nsight Systems export, reuse its `.sqlite` path.
If `nsys-ai` owns the capture, the profile command can produce the candidate:

```bash
mkdir -p artifacts
nsys-ai profile -o artifacts/current -- python train.py --steps 20
# The capture step in this project publishes artifacts/current/profile.sqlite.
```

The baseline normally comes from a protected artifact produced by the default
branch, or from a profile checked into a separate performance-data store. CI
jobs are isolated, so pass the baseline file or `.nsys-ai-baselines/` store
between jobs explicitly; a path in the producer job does not exist in the
consumer job automatically.

For a one-job workflow, tag the baseline and compare it by a stable name:

```bash
export NSYS_AI_BASELINE_ROOT="$PWD/artifacts/baselines"
nsys-ai baseline tag main artifacts/baseline.sqlite \
  --reason "known-good main ${GITHUB_SHA:-local}"
nsys-ai diff --against baseline:main artifacts/current.sqlite \
  --format json --no-ai --gate 3 \
  --output artifacts/diff.json
```

`--against baseline:main` resolves the snapshot from the local baseline store;
it does not contact Git or invent a second comparison path. `NSYS_AI_BASELINE_ROOT`
must point to the same store when tagging and comparing happen in different
working directories.

## GitHub Actions

The repository publishes a composite action that wraps the same command. The
steps that capture or download profiles are project-specific; once they place
the two files at the paths below, this is the complete gate and artifact part
of a workflow:

```yaml
- uses: actions/checkout@v4

- name: Install nsys-ai
  run: python -m pip install nsys-ai

# Project-specific steps go here. They must create:
#   artifacts/baseline.sqlite
#   artifacts/current.sqlite

- name: Compare performance
  id: perf-gate
  uses: GindaChen/nsys-ai/.github/actions/diff-gate@main
  with:
    install: "false"
    before: artifacts/baseline.sqlite
    after: artifacts/current.sqlite
    gate: "3"
    output: artifacts/diff.json

- name: Upload performance diff
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: nsys-ai-diff-${{ github.run_id }}
    path: artifacts/diff.json
    if-no-files-found: error
```

The action always adds `--format json --no-ai --exit-on-regression` and reads
the `verdict` from the JSON file written by the CLI. Do not add those flags to
`extra-args`. The action's `verdict` and `diff-json` outputs are convenience
references to that same file, not a second schema.

For a named baseline, tag the snapshot before the action and use `against`:

```yaml
- name: Tag baseline
  run: |
    nsys-ai baseline tag main artifacts/baseline.sqlite \
      --reason "main performance baseline ${{ github.sha }}"

- name: Compare performance
  uses: GindaChen/nsys-ai/.github/actions/diff-gate@main
  with:
    install: "false"
    against: baseline:main
    after: artifacts/current.sqlite
    gate: "3"
    output: artifacts/diff.json
```

Pin both the action and package to the release used by the project once the
desired release tag exists. The examples use `@main` and an unversioned package
so they are runnable before the 0.3.0 release; replace both with the same
published version when adopting them for a stable project workflow.

## Ordinary CI

Any CI system can invoke the same CLI directly. The repository's thin wrapper
is useful when the job wants JSON, deterministic output, and gate exit codes
without repeating those flags:

```bash
scripts/nsys-ai-diff-gate \
  artifacts/baseline.sqlite artifacts/current.sqlite \
  --gate 3 --output artifacts/diff.json
```

The wrapper only forwards arguments to `nsys-ai diff`; it does not calculate a
threshold or interpret a verdict. In another checkout, use the direct command
instead:

```bash
nsys-ai diff artifacts/baseline.sqlite artifacts/current.sqlite \
  --format json --no-ai --gate 3 --output artifacts/diff.json
```

## Exit codes and the JSON verdict

With `--gate`, `--exit-on-regression`, or `--gate-sol`:

| Exit code | Meaning | CI action |
|---:|---|---|
| `0` | The comparison completed and no configured gate failed. | Allow the job to continue. |
| `1` | `regression_likely`, `inconclusive`, or a speed-of-light gate failed. | Block the merge and inspect `diff.json` and the CI log. |
| `2` | Invalid command/configuration, such as a malformed threshold or unknown baseline. | Fix the workflow; this is not a performance verdict. |

The top-level JSON `verdict` distinguishes a measured regression from an
inconclusive comparison. `comparability_confidence` and `warnings` explain why
the result may not be safe to compare. Always upload the JSON with
`if: always()` so a failed gate leaves evidence for review.

Without `--gate` or `--exit-on-regression`, `diff` still writes the report but
does not block the job based on its verdict. `--gate PCT` is the supported
relative threshold syntax; for example, `--gate 3` means a step-time increase
above 3 percent is a regression. Absolute speed-of-light checks use the
separate `--gate-sol REGION:PCT` contract. It is an independent gate and
requires `--theoretical-flops FLOPS`; an unmeasurable or below-target region
exits `1`. The current diff JSON schema does not add a `gate-sol` result to
`verdict`, so keep the command log as evidence for that check in addition to
uploading `diff.json`.

## Accepting a known regression

Do not silently raise the threshold or delete the artifact. A maintainer can
record an explicit decision with a reason in a separate JSON record:

```bash
nsys-ai diff artifacts/baseline.sqlite artifacts/current.sqlite \
  --format json --no-ai \
  --output artifacts/diff.json \
  --decision-out artifacts/decision.json \
  --accept \
  --reason "Expected +2.8% from the documented tokenizer change; follow-up #123"
```

The decision record includes the status, reason, timestamp, and decider. It is
an audit trail, not a new pass/fail engine. A command that still includes
`--gate` remains a gate and exits non-zero when the measured regression exceeds
the threshold; make an override explicit in the workflow or adjust the
versioned project policy in a reviewed change.

Use `--reject --reason "..."` when the result should be recorded as rejected.
Both decision modes require a non-empty reason, and a session accepts only one
decision.

## Troubleshooting

- **Unknown baseline:** verify that the producer exported the baseline store or
  use the direct `before` path.
- **Inconclusive verdict:** inspect `warnings` and verify that both captures
  contain comparable workload activity and the intended GPU scope.
- **`diff.json` missing after failure:** ensure the command reached the diff
  engine, the output directory is writable, and the upload step uses
  `if: always()`.
- **Unexpectedly slow CI:** cache the profile's Parquet cache using the profile
  content hash and the cache version, or run the explicit `warm` command before
  the gate. The cache changes performance, not gate semantics.
