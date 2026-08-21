# Testing nsys-ai

This page describes how to choose a test layer and how to measure coverage
without losing the tests that launch `nsys-ai` as a child process.

## The test boundary

The repository has two kinds of Python process in a normal test run:

```text
pytest process
├── unit and API tests      (measured by pytest-cov)
└── python -m nsys_ai ...   (measured by sitecustomize)
```

That second branch is intentional. The CLI tests exercise the same public
entry point a user runs, including argument parsing, profile resolution, exit
codes, JSON serialization, and session files. Measuring only the parent would
make those tests pass while reporting the CLI handlers as uncovered.

## Fast layers

Run the smallest useful layer first:

```bash
python -m nsys_ai --help
python -m pytest tests/test_docs_index.py -q
```

For a code boundary, use a focused group:

```bash
# Skills and evidence
python -m pytest tests/test_skills.py tests/test_abstention.py -q

# CLI and shared loop
python -m pytest tests/test_cli.py tests/test_loop_api.py tests/test_loop_state.py -q

# Web and timeline transport
python -m pytest tests/test_web_foundation.py tests/test_timeline_web_data.py -q
```

The complete correctness run remains:

```bash
python -m pytest tests/ -v --tb=short -rs
```

Run it from the repository root. Several CLI tests deliberately change their
child's working directory to a temporary path; the root working directory is
what makes relative fixture and package paths meaningful.

## Coverage command

Install the development extra once in the active environment:

```bash
python -m pip install -e '.[dev]'
```

Then use the checked-in wrapper:

```bash
bash scripts/coverage.sh
```

It accepts the same test selectors as pytest:

```bash
bash scripts/coverage.sh tests/test_cli.py
bash scripts/coverage.sh tests/test_cli.py::test_doctor_json
```

The wrapper does four things:

1. points `COVERAGE_FILE` at one checkout-local data file;
2. starts `pytest --cov=src/nsys_ai` with parallel data enabled;
3. combines `.coverage.*` files emitted by child interpreters; and
4. prints one report with missing lines.

The equivalent direct command is useful when the inline pytest-cov report is
preferred:

```bash
COVERAGE_FILE="$PWD/.coverage" \
  python -m pytest tests/ --cov=src/nsys_ai --cov-report=term-missing
```

`tests/conftest.py` detects the `pytest-cov` plugin and exports
`COVERAGE_PROCESS_START`, `COVERAGE_SOURCE`, `COVERAGE_FILE`, and the checkout's
`src/` directory to children. A normal `pytest` or `nsys-ai` invocation does
not enable coverage.

## Why `sitecustomize.py` exists

Coverage cannot attach to a fresh `python -m nsys_ai` interpreter merely
because its parent is being measured. Python imports the top-level
`sitecustomize` module during startup, before `nsys_ai` is imported. The
repository's [sitecustomize.py](../../src/sitecustomize.py) calls
`coverage.process_startup()` when `COVERAGE_PROCESS_START` is present.

The hook has one important detail: CLI tests often run with `cwd=tmp_path`.
The normal coverage configuration names `src/nsys_ai` relative to the
checkout, so the test hook also passes an absolute `COVERAGE_SOURCE`. Without
that override, the child starts coverage successfully but silently records no
application lines from a temporary working directory.

`parallel = true` is required because multiple Python processes cannot safely
write one SQLite coverage data file at the same time. Each process writes a
sidecar, and `coverage combine` produces the reportable file.

## Timing-sensitive tests

`test_completion_before_deadline_wins_over_coarse_polling` tests a process
completion/deadline race. Line tracing changes scheduling in both the pytest
process and its fake `nsys` child, so the test is marked `no_cover` and removes
the startup variables for that invocation. This is an explicit measurement
boundary, not a skipped correctness test: the timing assertion still runs in
the normal suite, just without instrumentation overhead.

Do not add `no_cover` to a test merely because it is slow. Use it only when
coverage itself changes the behavior under test, and record the reason beside
the marker.

## Reading the number

The subprocess-aware number is the useful baseline for module-level gaps. It
is expected to be higher than the old parent-only number for CLI-heavy modules
such as `cli/handlers.py`; no test files or assertions need to be added just to
make that percentage look larger.

Coverage is a map of executed lines, not a quality score. It does not prove
that a SQL result is correct, that a profile schema is complete, or that an
end-to-end surface agrees with another surface. Keep the full suite,
contract tests, real-profile checks, and skip-reason guard as separate signals.

There is deliberately no coverage threshold in this change. First establish a
trustworthy baseline, then decide whether a gate is appropriate. A threshold
introduced before subprocess data is combined would gate the measurement
harness rather than the product.

## Troubleshooting

### `cli/handlers.py` is still implausibly low

Check all of the following:

```bash
python -m pip show coverage pytest-cov
python -c 'import sys; print(sys.executable)'
pwd
```

Run from the repository root and use `bash scripts/coverage.sh`. If you use a
custom data path, make it absolute:

```bash
COVERAGE_FILE="$PWD/.coverage-local" bash scripts/coverage.sh tests/test_cli.py
```

### `No data collected`

That warning can be legitimate for a test that only exercises code outside the
configured source. It is not legitimate for a CLI test. Confirm that
`COVERAGE_SOURCE` points to this checkout's `src/nsys_ai` and that `PYTHONPATH`
contains this checkout's `src/`; both are set automatically by the test
configuration when `--cov` is present.

### Sidecars remain after a failed run

They are disposable coverage data, ignored by Git. Re-run the wrapper; it
starts with `coverage erase` and uses a fresh report.

## Related contracts

- [Contributing](../../CONTRIBUTING.md) — setup, test layers, and PR checks
- [Skill contract](./skill-contract.md) — deterministic analysis behavior
- [Surface adapters](./surface-adapters.md) — CLI/Web/TUI/MCP delegation
- [Known limits](../user/known-limits.md) — what runtime analysis can and cannot prove
