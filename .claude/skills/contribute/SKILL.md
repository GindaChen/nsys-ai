---
name: contribute
description: >
  Use when getting a reviewed nsys-ai change into the upstream repo — committing,
  pushing, opening or updating a pull request, checking CI, updating issue labels, or
  merging. Also use when the user says "push", "open a PR", "ship it", or asks about
  git identity, branch naming, or commit message conventions here. Do not use for
  writing the change (use the develop skill) or for reviewing and testing it (use the
  review-change skill) — both come first.
---

# Getting a change into nsys-ai

This is a fork setup, so several defaults are wrong here:

```
origin    https://github.com/rich7420/nsys-tui.git      <- your fork; push here
upstream  https://github.com/GindaChen/nsys-ai.git      <- the real repo; PR here
```

Do not open the PR without `--repo GindaChen/nsys-ai --head rich7420:<branch>`, and do
not push to `upstream`.

## Step 0 — Pre-flight

Before writing a line of the PR, check nobody is already doing this:

```bash
gh pr list --repo GindaChen/nsys-ai --state open --search "<keywords>"

# open PRs touching a path you are about to touch
gh pr list --repo GindaChen/nsys-ai --state open --json number,title,files \
  --jq '.[] | select(any(.files[]; .path | test("<path>"))) | "\(.number)  \(.title)"'
```

Use `any(.files[]; …)` rather than `.files[].path | test(…)` — the latter emits the PR
once per matching file, so a PR touching six files looks like six PRs.

If an open PR touches the same files, coordinate rather than opening a conflicting one.
Prefer a small focused PR to a broad refactor of shared files.

## Step 1 — Identity gate

**Check before every push. Not once per repo — every push.**

```bash
git fetch upstream --quiet
git config --get user.email    # must be 101171023+rich7420@users.noreply.github.com

n=$(git rev-list --count upstream/main..HEAD)
bad=$(git log upstream/main..HEAD --format='%an <%ae>' | sort -u \
      | grep -vFx 'rich7420 <101171023+rich7420@users.noreply.github.com>')
if [ -n "$bad" ]; then printf 'FAIL — %s commit(s) checked, wrong identity:\n%s\n' "$n" "$bad"
else printf 'OK — all %s commit(s) carry the correct identity\n' "$n"; fi
```

It prints the commit count deliberately. A bare `git log … | sort -u` prints nothing on
an empty range, which is indistinguishable from "everything passed" — the exact failure
this repo already has a rule against elsewhere. `OK — all 0 commit(s)` tells you the
check ran and had nothing to look at.

This is not ceremonial. The global git config on this machine is
`rc910420@gmail.com`, and it wins anywhere the repo-local config is missing — a fresh
clone, a new worktree, a container. Eight commits already carry that address. Once
pushed it is public and permanent.

`references/IDENTITY.md` has the fix for each case, and the `.mailmap` situation.

## Step 2 — Commit

```bash
git commit -m "fix(skills): abstain when the memcpy table is absent"
```

- Describe what the code now does. Nothing else.
- **No** roadmap numbers, `#001`-style milestone handles, `§5.1`/`E#.#` refs, or phase
  markers (`P0`, `PR-A`, `Stage C2`). Those live in the PR body if anywhere.
- **No `Co-Authored-By` trailer.** Not for Claude, not for anyone.
- GitHub issue numbers are only ever real GitHub issue numbers — roadmap item numbers
  are not GitHub numbers, and using one as the other links the wrong issue.

## Step 3 — Branch and push

```bash
git checkout -b feat/issue-<N>-<short-description>   # or fix/…, docs/…, ci/…, test/…
git push -u origin feat/issue-<N>-<short-description>
```

Never push to `main` directly, and never to `upstream`.

## Step 4 — Open the PR

Write the body to a file — never inline with `--body "..."`, which invites shell
interpolation and leaks anything in the environment:

```bash
gh pr create --repo GindaChen/nsys-ai \
  --head rich7420:feat/issue-<N>-<short-description> \
  --base main \
  --title "<type>: <description>" \
  --body-file /tmp/pr_body.md
```

Fill out `.github/PULL_REQUEST_TEMPLATE.md` — Summary, Changes, Backward compatibility,
Test plan, Out of scope, Linked issues. The test plan checkboxes are claims: only tick
what the review-change skill actually ran. `references/PR.md` has the body template and
the label workflow.

## Step 5 — CI and review

```bash
gh pr checks <PR> --repo GindaChen/nsys-ai --watch
```

`lint`, `test` (3.10/3.11/3.12) and `security` run on every PR and must be green.
`smoke` only runs when the PR touches `skills/**`, `.claude-plugin/**` or the smoke
scripts, and `credentialed` never runs on pull requests — forks cannot read secrets. For
those two, absence is expected rather than a failure.

A Copilot bot leaves inline comments automatically. Treat them as review input: triage
by severity, fix or push back with a technical reason, reply in the thread. Hand the
actual fixing back to the **review-change** skill.

## Step 6 — Merge

```bash
gh pr merge <PR> --repo GindaChen/nsys-ai --squash --delete-branch
```

Only when green and approved. Confirm the base branch is `main` before merging — merging
into the wrong base is expensive to undo.

## Issue labels

If the work came from a GitHub issue, move it along the state machine:

```bash
# on claiming
gh issue edit <N> --repo GindaChen/nsys-ai --remove-label agent-ready --add-label agent-in-progress
# after opening the PR
gh issue edit <N> --repo GindaChen/nsys-ai --remove-label agent-in-progress --add-label agent-review
```

`agent-ready → agent-in-progress → agent-review → merged`, with `agent-blocked` as the
side exit.

## Never

- Put env values, API keys, tokens, or raw command output into a commit message, PR
  title, PR body, or comment. If it happens: sanitize the GitHub surface, rotate the
  credential, then open a support ticket for cache and notification redaction.
- Commit `.claude/worktrees/` (85 MB of live worktrees) or `.claude/settings.local.json`
  (machine-local permissions). `.claude/skills/` is meant to be committed.
- Commit a profile `.sqlite` that is not a deliberate, size-checked fixture.

## References

| File | Read when |
|------|-----------|
| `references/IDENTITY.md` | Step 1 — fixing a wrong author, and the `.mailmap` gap |
| `references/PR.md` | Step 4 — PR body template, titles, labels, release flow |
