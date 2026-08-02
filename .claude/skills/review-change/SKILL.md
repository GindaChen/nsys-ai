---
name: review-change
description: >
  Use after writing or changing code in this nsys-ai repo and before claiming it works
  or opening a PR — when asked to "review", "check", "verify", "is this done", "run the
  tests", or when finishing a task from the develop skill. Also use when responding to
  review comments on an open PR. Do not use for planning or writing the change itself
  (use the develop skill), or for the push/PR mechanics (use the contribute skill).
---

# Reviewing a change in nsys-ai

Two things happen here, in order: a review pass that looks for defects, and a
verification gate that produces evidence the change works. Neither substitutes for the
other — a clean review of untested code proves nothing, and a green test run does not
catch a violated contract that has no test yet.

## Step 1 — Establish the diff

```bash
git status --short
git diff main...HEAD --stat
git log main..HEAD --oneline
```

If the change is uncommitted, review the working diff. State explicitly which range is
under review; a review of the wrong range is worse than none.

## Step 2 — Review pass

Dispatch a subagent to do the reading rather than reading the diff yourself. The
reviewer gets the diff and the requirements in its own context and returns only
findings; your context stays free for fixing them. Give it:

- the base and head SHAs (or "uncommitted working diff"),
- the plan or issue the change was meant to satisfy,
- `references/CHECKLIST.md` as the criteria.

The general-purpose `/code-review` command covers ordinary defect classes well and can
run alongside. What it will not know is `references/CHECKLIST.md` — the contracts that
are specific to this repo and have each already shipped as a bug once. Those are the
ones worth spending the review on.

Rank findings by severity:

| Severity | Meaning | Action |
|---|---|---|
| Blocking | breaks behavior, violates a contract in the checklist, fails a CI gate | fix before anything else |
| Important | correctness risk, missing test, contract not covered | fix before opening the PR |
| Minor | naming, comment, style | fix if cheap, else note it in the PR |

Disagreeing is allowed and sometimes correct. Say why, technically — "this would break
the SQLite fallback path, which has no cache and therefore no `tc_eligible`" — rather
than accepting a change that makes the code worse. If it turns out the reviewer was
right, say so plainly and fix it.

## Step 3 — Verification gate

**Do not claim anything passes without having just run it and read the output.**

The failure this prevents is specific and common: reporting a green suite from memory,
from a partial run, or from a subagent's summary. Run it, read the exit code and the
failure count, then report.

```bash
ruff check src/ tests/
bandit -r src/ -c pyproject.toml
python -m nsys_ai --help
NSYS_TEST_PROFILE=tests/fixtures/h100_2gpu_1s.sqlite pytest tests/ -v --tb=short -rs
pytest tests/test_ci_coverage.py -v --tb=short
```

Add, when the change touches them:

```bash
pip-audit .                       # dependency change
bash scripts/smoke_test.sh        # skills/ or CLI surface change
pytest tests/test_cli.py -v       # new or changed CLI subcommand
```

`references/VERIFY.md` has the full matrix of what to run for which kind of change,
plus what each CI job actually asserts.

## Step 4 — Report

State what was run, what it produced, and what remains. Concretely:

> `ruff` clean. `bandit` clean. `pytest tests/` — 812 passed, 137 skipped, 0 failed.
> `test_ci_coverage` passed (no new unregistered skips). Review found 1 Blocking
> (abstention row indexed in `_format` before the guard) — fixed and re-ran. 1 Minor
> (threshold constant lacks a rationale comment) — left, noted in the PR.

If something failed, say that, with the output. If a step was skipped, say which and
why. "Should work now", "I'm confident", and "the subagent said it passed" are not
verification — they are the absence of it.

Only once this reports clean does the **contribute** skill open the PR.

## References

| File | Read when |
|------|-----------|
| `references/CHECKLIST.md` | Step 2 — the repo-specific contracts to review against |
| `references/VERIFY.md` | Step 3 — which commands for which change, and what CI asserts |
