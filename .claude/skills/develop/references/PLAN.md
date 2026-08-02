# Plan format

A plan exists so a reviewer can accept or reject each piece independently, and so a
later session can resume without re-reading the whole conversation. Both fail if the
plan is prose.

## Task template

```markdown
### Task N — <one line: what changes, not how>

**Files**
- create: src/nsys_ai/skills/builtins/foo.py
- modify: src/nsys_ai/cli/parsers.py, src/nsys_ai/cli/handlers.py
- test:   tests/test_foo.py, tests/test_cli.py

**Interface**
- produces: `SKILL = Skill(name="foo", category="kernels", ...)` returning rows with
  keys `kernel_name: str, total_ms: float, device: int`
- consumed by: `registry.get("foo")`, `nsys-ai skill run foo`

**Steps**
- [ ] <the actual change, concretely>
- [ ] <…>

**Verification**
```bash
python -m nsys_ai skill run foo tests/fixtures/mock.sqlite --format json
```
expected: valid JSON, non-empty rows, exit 0
```

## Rules

**Every step contains the content, not a pointer to it.** No `TBD`, no "handle the
edge cases", no "same as Task 2". If a step cannot be written concretely, Phase 1
research is not finished — go back rather than deferring the unknown into the build.

**Split by responsibility, not by layer.** "Add the skill and its test" is one
reviewable task. "Write all the SQL" then "write all the tests" is two tasks that
must both land to mean anything, and neither can be judged alone.

**Smallest correct change.** Do not bundle a refactor into a feature. If the feature
genuinely needs a refactor first, make it Task 1 with its own verification, and say in
the PR that it is separable. Bundled refactors are the most common review rejection in
this repo.

**Every task ends with a command.** A task whose verification is "looks right" is not
verifiable, and the review-change skill will send it back.

## Sizing

Aim for tasks a reviewer can hold in their head — roughly one module plus its test.
A task that touches eight files across four subsystems is a plan that has not been
decomposed yet.

## When the plan breaks

It will. Say so explicitly:

> Task 3 assumed `NVTX_EVENTS` has a `text` column. This export uses `textId` →
> `StringIds`. Revising Task 3 to use the `{nvtx_text_expr}` placeholder; Task 4 is
> unaffected.

Quietly building something other than the agreed plan is worse than the original
mis-plan, because the review is then against the wrong specification.
