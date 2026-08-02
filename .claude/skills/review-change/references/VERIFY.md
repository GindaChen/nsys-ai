# Verification — what to run, and what CI actually asserts

## The rule

Identify the command that would prove the claim, run it fresh and completely, read the
exit code and the failure count, then decide whether the output supports the claim. Only
then say it. Skipping a step is not a faster verification; it is not a verification.

Reporting a suite as green from memory, from a partial run, or from a subagent's summary
is the specific failure this exists to prevent.

## Always

```bash
ruff check src/ tests/
bandit -r src/ -c pyproject.toml
python -m nsys_ai --help
NSYS_TEST_PROFILE=tests/fixtures/h100_2gpu_1s.sqlite pytest tests/ -v --tb=short -rs
pytest tests/test_ci_coverage.py -v --tb=short
```

`NSYS_TEST_PROFILE` matters: without it the integration tests skip, and a green run
that skipped them is a weaker claim than it looks. `-rs` prints every skip and its
reason — read them.

## By what changed

| Changed | Also run |
|---|---|
| A builtin skill | `python -m nsys_ai skill run <name> tests/fixtures/mock.sqlite` and the same with `--format json` piped through `json.load` |
| A CLI subcommand | `pytest tests/test_cli.py -v`, plus the command by hand |
| `skills/` (the plugin) or CLI surface | `bash scripts/smoke_test.sh` |
| Dependencies / `pyproject.toml` | `pip-audit .` |
| Abstention or schema resolution | `pytest tests/test_abstention.py tests/test_determinism.py -v` |
| Diff / verdict logic | `python -m nsys_ai diff tests/fixtures/healthy_1pct.sqlite tests/fixtures/healthy_1pct.sqlite --format json` |
| Agent / chat | `pip install -e '.[all,dev]'` first; note that provider-backed tests run nightly, not per-PR |
| `site/` | open `site/index.html` and confirm it renders |

## What CI asserts

Three workflows run on every pull request; a fourth is path-gated.

**`lint`** (`ci.yml`) — `ruff check src/ tests/` on 3.12.

**`test`** (`ci.yml`) — matrix 3.10 / 3.11 / 3.12. Each runs:
- `pip install -e '.[tui,dev]'`
- `python -m nsys_ai --help`
- `pytest tests/ -v --tb=short -rs` with `NSYS_TEST_PROFILE=tests/fixtures/h100_2gpu_1s.sqlite`
- `pytest tests/test_ci_coverage.py` — the "no unexpected skips" gate
- a CLI smoke block that asserts, among others:
  - `skill run nvtx_kernel_map mock.sqlite` prints `not applicable` and exits 0 —
    a traceback here fails the build,
  - `skill run top_kernels --format json` and `analyze --format json` parse as JSON,
  - `agent ask "why is this slow?"` contains `## Verify` (the deterministic
    no-API-key answer contract),
  - `diff healthy_1pct healthy_1pct --format json` parses.

**`security`** (`security.yml`) — `bandit -r src/ -c pyproject.toml` and `pip-audit .`,
against the core install only.

**`smoke`** (`plugin-smoke.yml`) — path-gated: only runs when the PR touches
`skills/**`, `scripts/smoke_test.sh`, `scripts/build_fixture.py`, `.claude-plugin/**`,
or `tests/fixtures/mock.sqlite`. If your change touches any of those, run
`bash scripts/smoke_test.sh` locally rather than discovering it on the PR. If it does
not, the job's absence is correct, not a missing check.

**`credentialed`** (`ci.yml`) — provider-backed tests. Never on `pull_request` (forks
cannot read secrets); runs nightly and on push to `main`, with `continue-on-error` so a
provider outage does not redden `main`. Do not expect it on your PR.

## Reporting

Say what ran, what it produced, and what is left:

> `ruff` clean. `bandit` clean. `pytest tests/` — 812 passed, 137 skipped, 0 failed;
> skips all pre-registered. `test_ci_coverage` passed. `smoke_test.sh` not run — this
> change does not touch `skills/` or the CLI surface.

If a check failed, report the failure and its output. If one was skipped, name it and
say why. A claim without the run behind it is the thing this whole file is against.
