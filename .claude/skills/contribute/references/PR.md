# Pull requests

## Title

`<type>: <what changed>` — `feat`, `fix`, `docs`, `test`, `ci`, `refactor`, `chore`.

Describe the code. No roadmap numbers, no phase markers (`P0`, `PR-A`, `Stage C2`), no
`§5.1` or `E#.#` handles — those are internal planning handles and mean nothing to a
reader of the git log. If a GitHub issue exists, link it in the body, not the title.

Good: `fix(skills): abstain when the memcpy table is absent`
Bad: `P1 / PR-B: implement #043 per roadmap §5.2`

## Body

Write it to a file and pass `--body-file`. Never `--body "..."` — shell interpolation
there is how environment values end up in a public PR.

Follow `.github/PULL_REQUEST_TEMPLATE.md`. Empty sections may be deleted; Summary and
Test plan may not.

```markdown
## Summary
- <what this changes, and why — 1–3 bullets>

## Changes
- `src/nsys_ai/skills/builtins/foo.py` — <what>
- `tests/test_foo.py` — <what>

## Backward compatibility
Yes. <Or: No — name the break and the migration path.>

## Test plan
- [x] Tests added or updated for the new behavior
- [x] `python -m nsys_ai --help` — CLI loads without error
- [x] `pytest tests/ -v --tb=short` passes locally
- [x] `ruff check src/ tests/` clean
- [ ] Manually exercised the changed CLI / GUI path — n/a, no user-facing surface change

## Out of scope
- <what this deliberately does not do, and the follow-up issue if there is one>

## Linked issues
Closes #<N>
```

A ticked box is a claim that the command was run and passed. Tick only what the review
skill actually ran; leave the rest unticked with a reason. An unticked box with an
honest note reads better than a ticked one that turns out to be false.

Say what is out of scope. A reviewer who cannot tell whether an omission is an oversight
or a decision has to ask, which costs a round trip.

## Commands

```bash
# create
gh pr create --repo GindaChen/nsys-ai \
  --head rich7420:<branch> --base main \
  --title "<type>: <description>" --body-file /tmp/pr_body.md

# update after pushing more commits
gh pr edit <PR> --repo GindaChen/nsys-ai --body-file /tmp/pr_body.md

# CI
gh pr checks <PR> --repo GindaChen/nsys-ai --watch

# review comments
gh pr view <PR> --repo GindaChen/nsys-ai --comments

# merge
gh pr merge <PR> --repo GindaChen/nsys-ai --squash --delete-branch
```

`--head rich7420:<branch>` is required — `origin` is the fork, and without it `gh`
looks for the branch on `GindaChen/nsys-ai`, where it does not exist.

## Required checks

`lint`, `test` (3.10 / 3.11 / 3.12), and `security` run on every PR.

`smoke` (`plugin-smoke.yml`) is path-gated — it only runs when the PR touches
`skills/**`, `scripts/smoke_test.sh`, `scripts/build_fixture.py`, `.claude-plugin/**`,
or `tests/fixtures/mock.sqlite`.

`credentialed` is excluded from `pull_request` on purpose — forks cannot read secrets,
so running it would fail for every external contributor rather than skip.

For the last two, absence on a PR is correct, not a missing check.

## Copilot review

A Copilot bot posts inline comments automatically. Triage by severity like any review:
blocking first, then trivial, then the ones needing thought. Reply in the comment
thread rather than as a new top-level comment, and push back with a technical reason
when it is wrong — it does not know this repo's contracts and will sometimes suggest
changes that break the SQLite fallback path or the abstention contract.

## Issue labels

```bash
gh issue edit <N> --repo GindaChen/nsys-ai --remove-label agent-ready       --add-label agent-in-progress
gh issue edit <N> --repo GindaChen/nsys-ai --remove-label agent-in-progress --add-label agent-review
```

`agent-ready → agent-in-progress → agent-review → merged`, with `agent-blocked` as the
side exit when work stalls on missing information. Priority is `P0-critical` through
`P3-low`; pillars are `pillar/ai` and `pillar/ui`. `project-sync.yml` moves the project
board off these labels, so a stale label leaves the board wrong.

## Release

Only when asked. A `v*` tag triggers `workflow.yml` ("Publish to PyPI"), which builds
and uploads using the `PYPI_TOKEN` secret. A published version cannot be unpublished, so
the tag push is the point of no return — confirm the version and that `main` is green
before pushing it.

1. Bump `version` in `pyproject.toml`
2. `git commit -m "chore: bump to vX.Y.Z"`
3. `git tag vX.Y.Z && git push upstream main --tags`

Note that `AGENTS.md` and `CLAUDE.md` both still describe this as `publish.yml` using a
trusted publisher. Neither is true — the file is `workflow.yml` and it authenticates
with a token. Worth a docs fix in a separate PR.
