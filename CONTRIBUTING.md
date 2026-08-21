# Contributing to nsys-ai

This guide is for people changing the repository, not for people analysing a
profile. If you are looking for the latter, start with the [user guide](docs/user-guide.md).

The shortest successful contribution is:

1. create an isolated Python environment;
2. install the development dependencies;
3. run a small test and the CLI smoke test;
4. make one scoped change;
5. run the checks that cover that change and the full suite when practical;
6. open a pull request that records what changed and what was verified.

The project plan is recorded in the [roadmap](ROADMAP.md), the [roadmap
discussion](https://github.com/GindaChen/nsys-ai/issues/271), and the GitHub
issue list. An issue is the source of truth for a change's acceptance criteria;
this page explains the repository workflow around it.

## Before you clone

### Required tools

- Git
- Python 3.10 or newer
- a virtual-environment tool (venv, uv, or an equivalent)

CUDA, an NVIDIA GPU, and an Nsight Systems installation are not required for
the normal unit and integration suite. The committed fixtures are SQLite
exports, and CI uses them. An Nsight Systems installation and nvcc are only
needed for the optional real-capture checks or for capturing a new profile.

### Optional tools

Install these only for the part of the repository you are changing:

| Task | Extra | Additional requirement |
| --- | --- | --- |
| LLM-backed agent or chat | agent or chat | a provider key only for provider-backed runs |
| MCP server | mcp | no GPU or Nsight installation |
| CUTracer instruction analysis | cutracer | matching CUDA toolkit and nvdisasm |
| all optional features | all | the requirements of the selected workflows |

The core package already includes DuckDB, PyArrow, Rich, and Textual. You do
not need to install the optional extras to work on skills, the CLI, the web
server, or the committed-fixture tests.

## Set up a checkout

~~~bash
git clone https://github.com/GindaChen/nsys-ai.git
cd nsys-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
~~~

Verify that the checkout and the installed entry point agree:

~~~bash
python -m nsys_ai --help
nsys-ai --help
~~~

If you need the agent or chat path, add an extra after the development install:

~~~bash
python -m pip install -e '.[agent]'
# or: python -m pip install -e '.[chat]'
~~~

Keep provider keys in your shell or local secret manager. Do not put them in
source files, test fixtures, issue comments, PR descriptions, or command
output copied into a review.

## Repository map

The repository has a deliberate boundary between profile policy, analysis,
session handoff, and presentation:

| Area | Location | Responsibility |
| --- | --- | --- |
| package | src/nsys_ai/ | runtime code and installed assets |
| CLI | src/nsys_ai/cli/, src/nsys_ai/__main__.py | argument parsing and command adapters |
| profile policy | src/nsys_ai/profile.py, src/nsys_ai/connection.py | resolve .nsys-rep, .parquetdir, and .sqlite inputs |
| skills | src/nsys_ai/skills/ | deterministic, registered analysis units |
| agent core | src/nsys_ai/agent/ | shared evidence runner and answer shaping |
| sessions | src/nsys_ai/session_store.py, src/nsys_ai/session_cli.py | durable handoff artifacts |
| transports | src/nsys_ai/web.py, src/nsys_ai/mcp_server.py, TUI modules | publish or render the shared result |
| tests | tests/ | unit, contract, CLI, UI, and profile-backed checks |
| documentation | docs/ | user and developer contracts |
| examples | examples/ | reproducible workflows and optional large-profile benchmarks |
| automation | .github/workflows/ | CI, security, pages, and project synchronization |

Read [the ingest policy](docs/dev/ingest-policy.md), [the skill
contract](docs/dev/skill-contract.md), and [the surface adapter
contract](docs/dev/surface-adapters.md) before changing a corresponding
boundary. A transport should call the shared policy or runner; it should not
silently create a second SQL path.

## Test in layers

Use the smallest layer that gives useful feedback, then widen it before the
pull request. The commands below are intentionally the same commands used by
the repository's CI where possible.

### Layer 1: import and CLI smoke

Run this after changing packaging, imports, parsers, or documentation:

~~~bash
python -m nsys_ai --help
python -m pytest tests/test_docs_index.py -q
~~~

### Layer 2: focused contract tests

Choose tests by the boundary you changed:

~~~bash
# skills and evidence contracts
python -m pytest tests/test_skills.py tests/test_abstention.py tests/test_determinism.py -q

# CLI and shared loop
python -m pytest tests/test_cli.py tests/test_loop_api.py tests/test_loop_state.py -q

# web and timeline transport
python -m pytest tests/test_web_foundation.py tests/test_timeline_web_data.py -q
~~~

A new skill should normally have a focused test in tests/test_skills.py or a
nearby skill-specific file. A transport change should test the route and the
serialized response, not only a source-code string. A documentation-only
change should keep the two indexes valid with test_docs_index.py.

### Layer 3: full local suite

Before opening a PR, run:

~~~bash
python -m pytest tests/ -v --tb=short -rs
~~~

The -rs report is part of the result. A skipped test is acceptable only when
its reason is expected and documented. The suite also checks that committed
fixture binaries remain byte-identical; if a test needs to write indexes, use
the profile_copy fixture from tests/conftest.py rather than opening a
committed fixture directly.

The CI coverage guard can be run separately when you change test collection or
skip conditions:

~~~bash
python -m pytest tests/test_ci_coverage.py -v --tb=short
~~~

To measure the full suite, including CLI subprocesses, use the repository's
[subprocess-aware coverage guide](docs/dev/testing.md) and wrapper:

~~~bash
bash scripts/coverage.sh
~~~

The wrapper combines child-process data before reporting. A plain
`pytest --cov` run from the repository root also gets the child startup hook;
do not run it from another working directory because CLI tests intentionally
use temporary child directories.

To keep local or CI runs from writing session and capture artifacts into the
checkout, set `NSYS_AI_ARTIFACT_ROOT`; see the [artifact layout guide](docs/dev/artifact-layout.md)
for the precedence rules and the separate input-keyed cache policy.

### Layer 4: lint and security

These are required CI checks, not optional polish:

~~~bash
ruff check src/ tests/
bandit -r src/ -c pyproject.toml
pip-audit .
~~~

pip-audit describes the installed dependency set. If it reports a package
that is unrelated to your change, record the output in the PR and explain why
the change does not alter that dependency. Do not suppress a finding merely
to make CI green.

### Layer 5: real profiles and optional providers

The committed fixture is enough for most changes. For profile-backed tests,
point the suite at a profile explicitly:

~~~bash
NSYS_TEST_PROFILE=/path/to/profile.sqlite \
  python -m pytest tests/test_integration.py -v --tb=short
~~~

.nsys-rep, .sqlite, and supported Parquet inputs may be used where the test
accepts them. Add NSYS_TEST_GPU and NSYS_TEST_TRIM only when automatic
metadata selection is not appropriate:

~~~bash
NSYS_TEST_PROFILE=/path/to/profile.nsys-rep \
NSYS_TEST_GPU=0 \
NSYS_TEST_TRIM='39 42' \
  python -m pytest tests/test_integration.py -v --tb=short
~~~

The real CUDA capture security test is opt-out by environment when nsys and
nvcc are unavailable. To run it on a machine with both tools:

~~~bash
NSYS_REAL_CAPTURE=1 python -m pytest tests/test_real_capture_security.py -v
~~~

Provider-backed tests are intentionally not part of pull-request CI because
they can spend money and depend on external availability. Run them only when
you have explicitly configured the required key and understand the cost.

## Add a skill

A skill is a deterministic analysis unit. The LLM may select a registered
skill or summarize its rows, but it must not invent profile SQL. The canonical
execution path is skills.registry.run_skill.

### 1. Define the smallest contract

Create a module under src/nsys_ai/skills/builtins/ and export SKILL. The
registry discovers it automatically:

~~~python
from ..base import Skill, SkillParam


SKILL = Skill(
    name="kernel_device_counts",
    title="Kernel Counts by Device",
    description="Count recorded GPU kernels on each device.",
    category="kernels",
    sql="""
        SELECT k.deviceId AS device_id, COUNT(*) AS invocations
        FROM {kernel_table} k
        WHERE 1 = 1 {trim_clause}
        GROUP BY k.deviceId
        ORDER BY invocations DESC, device_id ASC
        LIMIT {limit}
    """,
    params=[SkillParam("limit", "Maximum rows", "int", False, 10)],
)
~~~

Use the shared placeholders instead of literal activity table names. Declare
parameters with SkillParam; trim_start_ns and trim_end_ns are shared runtime
arguments and do not need to be repeated in every skill. Use execute_fn for
interval math or multi-query analysis, and declare required_tables when the
function cannot answer without a table.

### 2. Define failure and ordering behaviour

The caller must be able to tell these states apart:

- []: the skill ran and found no result;
- abstain("reason"): the skill could not run or the input was ambiguous;
- ordinary rows: measured evidence.

Use the shared abstention helper and the _abstained key. Do not return an
untyped error row for an expected missing-table condition. Return JSON-safe
values and a deterministic total order, including tie-breakers after the main
metric. Preserve trim semantics in every query.

### 3. Add tests

At minimum, cover:

1. discovery through list_skills();
2. a normal result on minimal_nsys_conn or a copied fixture;
3. the required-table or missing-data path;
4. parameter defaults and required parameters;
5. deterministic ordering and JSON serialization;
6. trim boundaries when the skill accepts trim arguments.

Run the skill through the public registry in a contract test:

~~~python
from nsys_ai.skills.registry import run_skill

rows = run_skill("kernel_device_counts", conn, raw=True, limit=5)
assert all(isinstance(row, dict) for row in rows)
~~~

Do not call a builtin's private _execute() from a transport test. The private
function can be tested for narrow algorithmic details, but the public contract
belongs at the registry boundary.

## Change a CLI, Web, TUI, or MCP surface

Treat each surface as an adapter:

~~~text
Load → resolve profile → validate trim → shared runner/registry
     → typed rows → bounded serialization → session handoff → render
~~~

Before adding a helper, search for the canonical implementation. Profile
resolution belongs in the profile/connection policy; skill selection and
evidence belong in the runner; session artifacts belong in SessionStore;
HTTP, SSE, terminal, and MCP code own only their response format and lifecycle.

For a new command or endpoint, add:

- parser or route validation tests;
- one success response test;
- one typed failure or abstention test;
- a payload-size or row-limit assertion when the response can contain profile data;
- a session handoff assertion if the operation reads or writes a session.

Never make an LLM-generated SQL query the only grounding path. Never let a
large skill result bypass an existing serializer cap. If the feature changes
the public command set, update the relevant user guide and its index entry.

## Fixture and profile policy

Committed fixtures are test inputs, not scratch databases. Opening one through
a write-capable connection may add _nsysai_* indexes and change its binary
contents. Use profile_copy or copy into tmp_path before a test that can write.
If your working tree already contains fixture changes, restore or save them
before running the suite so the guard can detect new writes.

Large captures are useful evidence but are not automatically suitable for CI.
Record the path, size, command, and trim window in a PR when a real capture is
necessary. A test that only passes with a private multi-gigabyte file is not a
good first issue until the repository has a reproducible fixture or an explicit
skip contract.

## Documentation changes

Keep the two audiences separate:

- docs/user/ explains how to operate the released tool;
- docs/dev/ explains contracts for contributors;
- docs/README.md is the repository documentation index;
- site/index.html is the published project-page index.

When adding a page under docs/user/ or docs/dev/, add it to both indexes and
run:

~~~bash
python -m pytest tests/test_docs_index.py -q
~~~

Document observable behaviour with a command, output shape, and version or
backend caveat where relevant. If a number or verdict is quoted, reproduce it
from the current command rather than copying an old issue comment. Keep
limitations explicit: a reader should be able to distinguish "no result" from
"could not run".

## Choose an issue

Start with the highest-priority issue that is unblocked and scoped to the
available evidence:

~~~bash
gh issue list -R GindaChen/nsys-ai --state open --label "agent-ready"
gh issue list -R GindaChen/nsys-ai --state open --label "good first issue"
~~~

The current good-first set is deliberately small and verifiable on repository
data:

| Issue | Why it is a good first contribution |
| --- | --- |
| [#282](https://github.com/GindaChen/nsys-ai/issues/282) | Add an evaluation fixture whose idle gaps clear the reporting threshold but remain an immaterial share of the run. |
| [#497](https://github.com/GindaChen/nsys-ai/issues/497) | Write the user-facing diff-verdict page; the acceptance criteria name the fields and commands. |
| [#499](https://github.com/GindaChen/nsys-ai/issues/499) | Write the 0.3.0 migration note against the released command surface. |

Labels change as work is claimed or completed, so use the live issue list as
the final authority. Do not assign a capture-scale performance issue to a
newcomer unless the issue includes the capture or a fixture that reproduces
the claim. If an issue depends on another branch or release, state that
dependency before claiming it.

## Claim, branch, and commit

Claim an issue before editing it so another contributor does not duplicate the
work:

~~~bash
gh issue edit <ISSUE> -R GindaChen/nsys-ai \
  --remove-label "agent-ready" \
  --add-label "agent-in-progress"
~~~

Use a branch that identifies the issue and describes the change:

~~~bash
git fetch upstream main
git switch --create docs/issue-<ISSUE>-<short-description> upstream/main
~~~

Examples include feat/issue-224-diff-findings,
fix/issue-284-idle-severity, and docs/issue-497-reading-a-diff. Keep one
issue's acceptance criteria in one PR unless the issue explicitly calls for a
sequence.

Commits should say what changed, for example:

~~~text
docs: explain diff verdicts and comparability
fix: abstain when idle severity lacks a run denominator
~~~

Do not include API keys, tokens, private profile contents, or machine-specific
paths in commits or PR text.

## Open a pull request

Before pushing, inspect the diff and verify the exact checks relevant to it:

~~~bash
git diff --check
git status --short
git log --oneline upstream/main..HEAD
~~~

Push the branch and create the PR with a file-backed body. A file avoids shell
interpolation and makes it easier to review long acceptance-criteria tables:

~~~bash
git push -u origin <branch>
gh pr create -R GindaChen/nsys-ai \
  --head <your-fork>:<branch> \
  --title "<type>: <short description>" \
  --body-file /path/to/pr-body.md
~~~

The PR body should include:

- Closes #<issue> when the PR satisfies the issue;
- a concise change summary;
- files or contracts changed;
- local commands and results;
- real-profile or optional-provider results, including skips;
- known limitations and follow-up issues.

After opening the PR, move the issue to review:

~~~bash
gh issue edit <ISSUE> -R GindaChen/nsys-ai \
  --remove-label "agent-in-progress" \
  --add-label "agent-review"
~~~

## Review and merge

Review the patch as a user of the changed surface, not only as its author.
Check that:

- the public path uses the canonical profile, skill, runner, or session seam;
- abstention and empty-result states remain distinguishable;
- output is bounded and JSON-safe;
- tests would fail if the implementation were removed, rather than merely
  matching source text;
- docs and examples describe the merged behaviour;
- no committed fixture or unrelated user change is included.

CI must be green for lint, security, and Python 3.10, 3.11, and 3.12. A
credentialed check may be skipped on pull requests by design. Do not merge a
red check by assuming it is environmental; inspect the job log, reproduce it
or record the reason, and fix or explicitly resolve the issue.

Once review is complete and required checks are green, a maintainer can merge:

~~~bash
gh pr checks <PR> -R GindaChen/nsys-ai
gh pr merge <PR> -R GindaChen/nsys-ai --squash --delete-branch
~~~

The issue should be closed by the merge and its workflow label removed. If the
work is blocked, use agent-blocked and leave a comment that names the exact
missing input or dependency.

## Checklist

Before requesting review:

- [ ] The issue is claimed and the branch contains only its scoped change.
- [ ] python -m nsys_ai --help succeeds.
- [ ] Focused tests for the changed boundary pass.
- [ ] python -m pytest tests/ -v --tb=short -rs was run, or the PR explains why not.
- [ ] ruff check src/ tests/ passes.
- [ ] bandit -r src/ -c pyproject.toml passes for runtime changes.
- [ ] pip-audit . was checked for dependency changes.
- [ ] Documentation indexes and examples are updated when needed.
- [ ] Committed fixtures are unchanged.
- [ ] The PR body records commands, results, skips, and follow-ups.
- [ ] No secret, private capture data, or machine-specific credential appears in the patch or PR.
