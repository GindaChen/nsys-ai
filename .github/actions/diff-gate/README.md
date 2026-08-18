# nsys-ai diff gate

This composite action is a thin CI wrapper around the canonical
`nsys-ai diff --gate` command. It always writes a machine-readable JSON report,
disables optional AI narration, and enables the existing non-zero exit contract
for regressions and inconclusive comparisons. Threshold evaluation is not
implemented by the action.

## GitHub Actions

The caller must make a baseline and candidate profile available first. A
baseline can be a direct path or a named local baseline store reference:

```yaml
- uses: GindaChen/nsys-ai/.github/actions/diff-gate@main
  id: perf-gate
  with:
    before: artifacts/baseline.sqlite
    after: artifacts/current.sqlite
    gate: "3"
    output: artifacts/diff.json

- uses: actions/upload-artifact@v4
  if: always()
  with:
    name: nsys-ai-diff
    path: artifacts/diff.json
```

For a named baseline, tag it before the action and replace `before` with
`against`:

```yaml
- run: python -m pip install nsys-ai
- run: nsys-ai baseline tag main artifacts/baseline.sqlite --reason "CI baseline"
- uses: GindaChen/nsys-ai/.github/actions/diff-gate@main
  with:
    against: baseline:main
    after: artifacts/current.sqlite
    gate: "3"
    output: artifacts/diff.json
```

Pin the action and package to a release tag when adopting it in a project. The
`verdict` and `diff-json` step outputs are read from the same JSON file that
the CLI writes; they do not introduce a second verdict format.

`extra-args` accepts additional `diff` arguments one per line, for example:

```yaml
    extra-args: |
      --gpu
      0
```

Do not put `--format`, `--output`, `--no-ai`, or
`--exit-on-regression` in `extra-args`; the action owns those CI contract
flags.

## Ordinary CI

After installing the package, use the repository wrapper when a shell-level
entry point is more convenient:

```bash
scripts/nsys-ai-diff-gate \
  artifacts/baseline.sqlite artifacts/current.sqlite \
  --gate 3 --output artifacts/diff.json
```

The wrapper only forwards arguments to `nsys-ai diff`; `--gate` remains the
single source of threshold and exit-code semantics.
