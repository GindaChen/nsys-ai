# Release process

This is the repeatable release checklist for `nsys-ai`. It is deliberately
separate from the [architecture guide](./architecture.md): the architecture
guide explains what 0.3.0 shipped, while this page explains what a maintainer
does every time a version is published.

The release process has one rule above all others:

> A release is not complete when a tag exists. It is complete when the exact
> artifact installed from PyPI passes the same smoke test as the source
> checkout, and the evidence is recorded in the release tracking issue.

The process follows the useful parts of [Apache DataFusion's release
management guide](https://datafusion.apache.org/contributor-guide/release_management.html):
one tracking issue, an explicit candidate/stabilization boundary, targeted
patch backports, and a post-publish verification step. It is adapted to this
repository's current tooling and its single `main` development branch.

## Release types

Decide the release type before changing the version. The type controls the
scope that is allowed into the candidate.

| Type | Example | Allowed scope | Branch policy |
|------|---------|---------------|---------------|
| Feature release | `0.4.0` | New capabilities, architecture work, user-visible behavior, docs, and fixes that belong to the new feature set | Start from `main`; create a temporary `release/X.Y` branch only when stabilization needs to continue while `main` moves on |
| Patch release | `0.3.1` | Targeted correctness, security, packaging, documentation, or release-blocking fixes | Create or reuse `release/0.3`; backport only the selected fixes from `main` |
| Release candidate | `0.4.0rc1` | Candidate validation only; no unrelated feature work | Optional; use when the change surface or artifact risk justifies an installable public candidate |

`nsys-ai` is currently published as an alpha package. Keep that status honest
in `pyproject.toml`, release notes, and user-facing documentation. Do not
reuse a published version or tag. If a published artifact is wrong, fix the
problem and publish the next patch version.

## Ownership and source of truth

Create one GitHub issue for each release, for example:

```text
release: publish 0.4.0
```

The issue is the release's handoff record. It should contain:

- the release manager and target date;
- the release type and the exact scope;
- links to every included issue and PR;
- blockers and deferred follow-ups;
- candidate test results and the commit tested;
- the published GitHub Release and PyPI links;
- post-publish smoke-test output summarized without credentials.

Use the repository's release label for the version and the normal priority
label for the tracking issue. Close the issue only after the package-index
verification in [Phase 5](#phase-5-verify-the-published-artifact) is complete.
Remove or close stale release labels and update the roadmap during
[Phase 6](#phase-6-close-out-and-hand-off), so the next release starts with a
trustworthy queue.

Never put API keys, PyPI tokens, provider credentials, or secret-bearing
environment dumps in the issue, PR, release notes, or command transcript.

### Remote convention

The commands below assume the normal contributor checkout: `origin` is the
fork and `upstream` is the canonical `GindaChen/nsys-ai` repository. Release
branches and tags must be pushed to `upstream`; pushing a tag only to a fork
does not trigger the canonical publish workflow. Before starting, verify the
mapping with `git remote -v`. If the canonical repository is named `origin` in
your checkout, substitute `origin` for `upstream` consistently. The `gh`
commands below always target the canonical repository explicitly with
`-R GindaChen/nsys-ai`.

## The six release phases

Every release passes through these gates:

| Gate | Question | Evidence |
|------|----------|----------|
| 1. Scope | Do we know exactly what is in and out? | Release issue, linked PRs, updated roadmap |
| 2. Candidate | Does the source checkout pass the product and packaging checks? | CI run, local test output, real-profile checkpoint |
| 3. Artifact | Does the wheel/sdist contain the files and metadata users need? | Build directory, `twine check`, package inspection |
| 4. Publish | Did the intended commit become the intended tag, GitHub Release, and PyPI version? | Tag, release URL, workflow run, PyPI URL |
| 5. Verify | Does a clean environment run the package downloaded from PyPI? | Fresh-venv smoke output and artifact hashes |
| 6. Handoff | Can the next maintainer tell what happened and what remains? | Release issue comment, changelog, roadmap, follow-up issues |

Do not skip a gate because the change is "only documentation". Documentation
and site changes have their own broken-link, Pages, and rendered-asset failure
modes.

### What is automated today

Keep this boundary visible when improving the release workflow:

| Check | Current owner |
|-------|---------------|
| Python CI matrix, lint, and security checks | GitHub Actions on PRs |
| Wheel/sdist build and PyPI upload after a `v*` tag | `.github/workflows/workflow.yml` |
| GitHub Pages deployment after site changes | `.github/workflows/pages.yml` |
| Release scope, changelog, and release notes | Maintainer and reviewers |
| `twine check`, wheel-content inspection, and SHA256 recording | Maintainer during candidate verification |
| Fresh PyPI installation and real-profile smoke test | Maintainer after publication |
| Release issue, roadmap, labels, and follow-up handoff | Maintainer |

The manual rows are intentional release gates, not evidence that the package
publish job passed. If a manual check becomes a reliable, deterministic
automation, add the automation and keep the corresponding evidence in the
release issue.

## Phase 1: scope and freeze

### 1.1 Create the tracking issue

Copy the [release issue template](#release-issue-template) into a new issue.
Record the target version before opening the release PR. Link all included
work to this issue; do not rely on a search for PR titles after publishing.

### 1.2 Audit the current state

From a clean checkout:

```bash
git fetch upstream --tags
git status --short --branch
git log --oneline --decorate -20 upstream/main
gh issue list -R GindaChen/nsys-ai --state open --label "release/0.4.0"
gh pr list -R GindaChen/nsys-ai --state open --base main
```

Replace `0.4.0` with the target version. Confirm that:

- all release-blocking issues are closed or explicitly accepted;
- every included PR is merged into the candidate base;
- no open PR silently changes the release scope;
- the roadmap describes the same completion boundary as the release issue;
- the migration guide, README, site, and changelog do not promise work that is
  not in the candidate.

Use a release branch when stabilization needs to happen in parallel with new
development:

```bash
git switch main
git pull --ff-only upstream main
git switch -c release/0.4
git push -u upstream release/0.4
```

For a small release that is intentionally cut from `main`, record the exact
commit in the issue and do not create a branch merely for ceremony.

### 1.3 Freeze the candidate

Once the candidate commit is chosen:

1. stop adding unrelated features;
2. accept only release-blocking fixes, documentation corrections, or targeted
   packaging/security fixes;
3. require the normal review and CI checks for every candidate change;
4. update `CHANGELOG.md`, migration notes, and the roadmap from the final
   included set;
5. write down the candidate commit SHA in the release issue.

If a candidate needs a new feature to pass, it is not a release fix. Move that
work to the next feature release unless the release scope is explicitly
reopened and re-reviewed.

## Phase 2: candidate verification

Run the fast checks first, then the full suite, then the real-profile
checkpoint. Use a fresh virtual environment for packaging checks; an editable
checkout can hide missing package data.

### 2.1 Source and metadata checks

```bash
git status --short
python -m pip install -e '.[dev]'
python -m nsys_ai --help
python -m ruff check src tests
python -m pytest tests/ -v --tb=short -rs
```

If a test is skipped or xfailed, record the reason and whether it is expected
for this release. Do not turn a known environmental limitation into an
unexplained green check.

Check that the following agree on the same version and release date:

- `pyproject.toml`;
- `CHANGELOG.md`;
- the release issue;
- the candidate branch/commit;
- the eventual tag and GitHub Release title.

For a release that changes Python or runtime support, also check the CI matrix
in `.github/workflows/ci.yml` and the `requires-python` declaration. The
release should not claim support that CI never exercises.

### 2.2 Packaging checks

Build both distribution formats from the candidate commit:

```bash
rm -rf dist build
python -m pip install build twine
python -m build
python -m twine check dist/*
sha256sum dist/*
ls -lh dist/
```

There must be one wheel and one source distribution for the intended version.
Inspect the wheel before installing it:

```bash
python -m zipfile -l dist/nsys_ai-*.whl
```

At minimum, confirm that the wheel contains the runtime package data declared
in `pyproject.toml`:

- HTML, CSS, and JavaScript templates;
- agent prompts and persona data;
- agent skill context;
- built-in skill data;
- the two console entry points: `nsys-ai` and `nsys-ai-mcp`.

The source tree is not a packaging test. A missing template or prompt can be
invisible in an editable checkout and fail only for a user who installs the
wheel.

### 2.3 Four-verb product smoke test

Use a small committed fixture for the fast gate. Use a real `.nsys-rep` or
`.sqlite` capture for the release checkpoint when one is available. The exact
capture, Nsight Systems version, trim window, and runtime should be recorded in
the issue; do not claim a real-profile result from a fixture.

```bash
nsys-ai doctor PROFILE --format json
nsys-ai diagnose PROFILE --session SESSION_DIR --format json
nsys-ai ask --session SESSION_DIR "what is the main bottleneck?"
nsys-ai diff BEFORE AFTER --session SESSION_DIR --no-ai --format json
nsys-ai review --session SESSION_DIR
```

Check the result, not only the exit code:

- `doctor` reports the input conversion/cache/schema state honestly;
- `diagnose` writes findings to the requested session;
- `ask` uses grounded evidence or clearly reports abstention/unavailability;
- `diff` preserves comparability and does not silently reverse metric meaning;
- `review` reads the same session handoff and exposes the decision state;
- the session directory contains only the expected artifacts and no secret
  values.

For a release that changes the optimization workflow, add the guided path:

```bash
nsys-ai propose PROFILE --session SESSION_DIR --format json
nsys-ai optimize PROFILE --session SESSION_DIR --format json
```

### 2.4 Surface and documentation checks

When the release changes a transport, viewer, or documentation page, run the
smallest relevant contract checks as well as the full suite:

```bash
python -m pytest tests/test_docs_index.py tests/test_documentation_contracts.py -q
python -m pytest tests/test_web_foundation.py tests/test_timeline_web_data.py -q
```

For site changes, verify the Pages workflow and open the deployed page. For
viewer changes, exercise the documented URL with a real session and confirm
that a missing or partial backend is represented as a visible state rather
than an empty success page.

For release notes and user docs, check every command against the candidate
artifact or a clean checkout. A command that only works from the repository
root is not automatically a valid installed-package example.

## Phase 3: build the release candidate

Run these commands after the candidate checks pass and before creating a
public tag:

```bash
git status --short --branch
git diff upstream/main...HEAD --stat
python -c 'from importlib.metadata import version; print(version("nsys-ai"))'
python -m build
python -m twine check dist/*
sha256sum dist/*
```

If a release candidate is useful, publish it under a pre-release version and
run the same clean-install checks below. Do not silently treat an RC as the
final release: create a new final artifact and tag when the candidate is
accepted.

Before tagging, the release manager should sign off in the issue on:

- scope and version;
- full-suite and CI status;
- packaging contents and `twine check`;
- fixture smoke test;
- real-profile checkpoint or an explicit, documented reason why it is not
  available;
- known limitations and follow-up issues.

## Phase 4: publish

### 4.1 Create the tag from the tested commit

Do not tag a dirty checkout. The tag must point at the exact commit whose
artifacts were checked.

```bash
git status --porcelain
git rev-parse HEAD
git tag -a v0.4.0 -m "Release v0.4.0"
git push upstream v0.4.0
```

Replace the version in every command. Never move an existing published tag.

The tag triggers `.github/workflows/workflow.yml`, which builds the wheel and
source distribution and publishes them to PyPI. Monitor the workflow rather
than assuming a successful tag push means a successful publication:

```bash
gh run list -R GindaChen/nsys-ai --workflow workflow.yml --limit 5
gh run watch RUN_ID -R GindaChen/nsys-ai --exit-status
```

### 4.2 Create the GitHub Release

After the tag is visible, create the GitHub Release with the same version and
the categorized notes from `CHANGELOG.md`:

```bash
gh release create v0.4.0 \
  -R GindaChen/nsys-ai \
  --title "nsys-ai v0.4.0" \
  --notes-file /path/to/release-notes.md
```

The notes should lead with user-visible outcomes, then include compatibility,
installation, known limitations, and links to the migration guide. Do not
enumerate PRs as the release story; link issues or PRs only where they help a
user understand a behavior or a limitation.

If the project later adds signed GitHub assets, attach the exact wheel, sdist,
and checksum/signature files produced from the tagged commit. Until then,
record the `sha256sum` output in the release issue and use the hashes shown by
the package index as an independent comparison point.

## Phase 5: verify the published artifact

This phase must use a new environment and the package index, not an editable
checkout and not a local `dist/` directory.

```bash
RELEASE_VENV="$(mktemp -d)/venv"
python -m venv "$RELEASE_VENV"
"$RELEASE_VENV/bin/python" -m pip install --upgrade pip
"$RELEASE_VENV/bin/python" -m pip install 'nsys-ai==0.4.0'
"$RELEASE_VENV/bin/python" -m pip show nsys-ai
"$RELEASE_VENV/bin/nsys-ai" --help
"$RELEASE_VENV/bin/python" -c 'import nsys_ai; print(nsys_ai.__file__)'
"$RELEASE_VENV/bin/python" -c 'from nsys_ai.mcp_server import main; print(main.__module__)'
```

Repeat the smallest meaningful product smoke test against the installed
package:

```bash
"$RELEASE_VENV/bin/nsys-ai" doctor PROFILE --format json
"$RELEASE_VENV/bin/nsys-ai" diagnose PROFILE --session SESSION_DIR --format json
"$RELEASE_VENV/bin/nsys-ai" diff BEFORE AFTER --session SESSION_DIR --no-ai --format json
```

For a real-profile release checkpoint, compare the installed-package result
with the candidate result on the same input, trim window, and command flags.
Investigate any changed finding count, verdict, schema warning, or session
artifact before closing the release issue.

Record these values in the issue:

- installed version and Python version;
- PyPI URL and GitHub Release URL;
- wheel/sdist filenames and SHA256 hashes;
- smoke-test commands and concise results;
- known limitations that remain after publication.

The [DataFusion download page](https://datafusion.apache.org/download) is a
useful model for making artifact verification visible to users. This project
currently exposes PyPI hashes rather than a maintained signing-key workflow;
do not document signatures until the repository actually publishes and
rotates the required keys.

## Phase 6: close out and hand off

Complete the release only after the package-index smoke test passes.

1. Update the release issue with the final tag, workflow run, release URL,
   PyPI URL, hashes, and verification results.
2. Update `CHANGELOG.md`, `docs/README.md`, the migration guide, and the
   website if the published version changed their claims.
3. Update `docs/roadmap.md` so shipped work, deferred work, and the next
   release boundary agree.
4. Close release-scoped issues that are actually shipped; leave follow-ups
   open when the release only established a boundary.
5. Remove stale `release/X.Y.Z` labels and create explicit follow-up issues
   for known limitations, automation gaps, or deferred architecture work.
6. If a release branch exists, document its maintenance owner and the rule for
   accepting patch fixes. Merge the final release-only documentation changes
   back to `main`.
7. Announce the release using the GitHub Release and PyPI links. Keep the
   announcement shorter than the release issue; the issue is the audit trail.

## Patch releases and backports

Patch maintenance follows the same principle used by mature Apache projects:
develop and review the fix on `main` first, then backport the exact reviewed
commit into the release branch.

```bash
git switch main
git pull --ff-only upstream main
# merge and verify the fix on main first

git switch release/0.3
git pull --ff-only upstream release/0.3
git cherry-pick -x FIX_COMMIT_SHA
python -m pytest tests/ -v --tb=short -rs
git push upstream release/0.3
```

The `-x` record makes the origin of the backport visible. Resolve conflicts by
preserving the release branch's public behavior and rerun the relevant real
profile check. A patch release must not accumulate unrelated feature work.

For each backport, record:

- the original issue and PR;
- the source commit on `main`;
- the release-branch commit;
- why the fix is safe for the patch series;
- the test evidence and affected package surfaces.

## Failure and recovery

### CI fails before tagging

Fix the candidate branch, rerun the failed and affected checks, and update the
candidate SHA in the release issue. Do not publish a tag that has not passed
the release gate.

### The tag was pushed but PyPI publishing failed

Inspect the workflow logs without printing secrets. If the version was not
published, fix the workflow or repository configuration and rerun the tag
workflow according to GitHub's permissions. If the version was already
published, never overwrite it; publish the corrected artifact as the next
patch version.

### The package was published but the clean smoke test fails

Mark the release issue as blocked, document the exact failure, and determine
whether the failure is in the package, the release metadata, or the test
environment. Never ask users to install an editable checkout as a workaround
for a broken published artifact. A bad published package requires a new
version, followed by the complete verification phase again.

### The website or docs are wrong after publication

Fix the source on `main`, confirm the Pages workflow, and update the release
issue. If the mistake changes installation, compatibility, or a safety
limitation, publish a patch release note as well; do not silently edit history
to make the original release appear to have said something it did not say.

## Release issue template

Copy this into a new issue and replace the placeholders:

```markdown
# Release X.Y.Z

Release manager: @handle
Target date: YYYY-MM-DD
Type: feature | patch | release candidate
Candidate commit: `<sha>`

## Scope

- [ ] Link every included issue/PR
- [ ] Confirm deferred work and known limitations
- [ ] Update roadmap and migration notes

## Candidate gate

- [ ] `pyproject.toml`, `CHANGELOG.md`, and target version agree
- [ ] Full CI matrix is green
- [ ] `python -m pytest tests/ -v --tb=short -rs`
- [ ] `python -m build`
- [ ] `python -m twine check dist/*`
- [ ] Wheel package-data inspection passed
- [ ] Fixture doctor/diagnose/ask/diff/review smoke passed
- [ ] Real-profile checkpoint passed, or limitation recorded
- [ ] Docs/site/link checks passed

## Publish evidence

- [ ] Tag: `vX.Y.Z`
- [ ] GitHub Release: <url>
- [ ] Publish workflow run: <url>
- [ ] PyPI: <url>
- [ ] Wheel/sdist SHA256 recorded

## Post-publish verification

- [ ] Fresh venv installed `nsys-ai==X.Y.Z` from the package index
- [ ] Installed `--help` and import smoke passed
- [ ] Installed doctor/diagnose/diff smoke passed
- [ ] Real-profile result compared with the candidate, when available
- [ ] Limitations and follow-up issues recorded
- [ ] Roadmap and release labels cleaned up
```

## Related references

- [Architecture and release-maintainer guide](./architecture.md) — what the
  current architecture promises and the short 0.3.0 boundary checklist
- [Testing nsys-ai](./testing.md) — test layers, subprocess coverage, and
  troubleshooting
- [Contributing](../../CONTRIBUTING.md) — development setup and PR checks
- [User migration guide](../user/migrating-to-0.3.0.md) — an example of a
  version-specific migration document
- [Apache DataFusion release management](https://datafusion.apache.org/contributor-guide/release_management.html)
- [Apache DataFusion maintainer release README](https://github.com/apache/datafusion/blob/main/dev/release/README.md)
- [Apache DataFusion contributor guide](https://github.com/apache/datafusion/blob/main/docs/source/contributor-guide/index.md)
- [Apache DataFusion artifact verification](https://datafusion.apache.org/download)

For 0.3.0, this checklist was executed manually. The release evidence is in
[issue #534](https://github.com/GindaChen/nsys-ai/issues/534) and the
[0.3.0 GitHub Release](https://github.com/GindaChen/nsys-ai/releases/tag/v0.3.0).
Future automation opportunities should be opened as separate issues rather
than weakening this manual gate.
