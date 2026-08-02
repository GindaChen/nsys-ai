# Git identity

## The one correct identity

```
rich7420 <101171023+rich7420@users.noreply.github.com>
```

Nothing else. Not the gmail address, not a display name variant.

## Why this needs a check every push

The repo-local config is correct today:

```
local   user.name  = rich7420
local   user.email = 101171023+rich7420@users.noreply.github.com
```

The global config on this machine is not:

```
global  user.email = rc910420@gmail.com
```

Local overrides global — inside this checkout. Anywhere the local config is absent, the
global one applies: a fresh clone, a `git worktree add` in a new location, a container,
a CI runner, a machine set up later. That is how these eight commits happened:

```
f666c64 release: v0.2.3
00bb99b lint errors
2d5360d update and improve
a6f1228 fix ci errors
f52cb1b feat: migrate TUI tree/timeline to Textual with tests
100fcb8 test: add trajectory-based agent regression suite
164b97d feat ai: refine Brain & Navigator UX across web + TUI
1f70643 Make nsys-ai schema/version aware for Nsight 2026 SQLite exports
```

A private address in a public commit cannot be recalled. The check is two commands and
runs before every push.

## The check

```bash
git fetch upstream --quiet

# 1. is the environment right, right now?
git config --get user.email

# 2. does every commit about to be pushed carry the right identity?
n=$(git rev-list --count upstream/main..HEAD)
bad=$(git log upstream/main..HEAD --format='%an <%ae>' | sort -u \
      | grep -vFx 'rich7420 <101171023+rich7420@users.noreply.github.com>')
if [ -n "$bad" ]; then printf 'FAIL — %s commit(s) checked, wrong identity:\n%s\n' "$n" "$bad"
else printf 'OK — all %s commit(s) carry the correct identity\n' "$n"; fi
```

The first catches the environment being wrong now. The second catches commits already
made with a wrong identity — including ones cherry-picked or rebased in from elsewhere.
The second is the one that actually protects the push.

Three details that are not incidental:

- **`git fetch upstream` first.** A stale `upstream/main` makes the range wrong, and a
  wrong range checks the wrong commits while still reporting confidently.
- **It prints a verdict with a count, not a list.** A bare
  `git log … | sort -u` prints nothing when the range is empty, and nothing looks
  exactly like "all clean". `OK — all 0 commit(s)` says the check ran and found nothing
  to check, which is a different statement.
- **`grep -vFx`** — fixed-string, whole-line. A substring match would pass
  `rich7420 <101171023+rich7420@users.noreply.github.com.evil>`.

Verified against `f666c64`, one of the eight below: the check reports
`FAIL — 1 commit(s) checked, wrong identity: rich7420 <rc910420@gmail.com>`.

## Fixes

**Config is wrong, nothing committed yet:**

```bash
git config --local user.name  "rich7420"
git config --local user.email "101171023+rich7420@users.noreply.github.com"
```

Set it locally, not globally — changing the global config affects the user's other
repos, which is not yours to decide.

**The last commit is wrong, not yet pushed:**

```bash
git commit --amend --reset-author --no-edit
```

**Several unpushed commits are wrong:**

```bash
git rebase -i --exec 'git commit --amend --reset-author --no-edit' upstream/main
```

Rewriting history is safe only while the commits are unpushed. If they are already on
`origin`, say so and ask before force-pushing — someone may have branched from them.

**Already on `upstream/main`:** it stays. Do not rewrite shared history. Record it and
move on; `.mailmap` below limits the damage to attribution, though not to the exposure.

## The name drift, and `.mailmap`

One email, two names:

```
rich7420        <101171023+rich7420@users.noreply.github.com>   # committed locally
KUAN-HAO HUANG  <101171023+rich7420@users.noreply.github.com>   # merged via the GitHub UI
```

Merging through the web UI stamps the GitHub display name, so the same person appears
twice in `git shortlog -sn`, and merge commits read as two contributors. The repo has
no `.mailmap` yet. The standard fix — the one the kernel and git itself use — is to add
one at the repo root:

```
rich7420 <101171023+rich7420@users.noreply.github.com> KUAN-HAO HUANG <101171023+rich7420@users.noreply.github.com>
rich7420 <101171023+rich7420@users.noreply.github.com> rich7420 <rc910420@gmail.com>
```

It rewrites nothing; `git shortlog`, `git log --use-mailmap`, and GitHub's contributor
list read it and collapse the aliases. This is a standalone change — propose it as its
own small PR rather than smuggling it into a feature branch.
