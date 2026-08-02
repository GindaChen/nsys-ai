---
name: develop
description: >
  Use when about to change code in this nsys-ai repo — implementing a GitHub issue,
  adding a CLI subcommand or builtin analysis skill, fixing a bug, or picking up a
  ROADMAP item. Also use when handed an issue number, or asked to "implement", "add",
  "fix", or "make X work". Do not use for reviewing work that is already written
  (use the review-change skill), for opening a pull request (use the contribute skill), or
  for analyzing a .sqlite GPU profile (that is the nsys-ai plugin skill under skills/).
---

# Developing a change in nsys-ai

Four phases, each with a gate. The gates exist because the expensive failures in this
repo are not typos — they are changes built on a wrong reading of the profile schema,
or a second implementation of something `src/nsys_ai/` already does. Both are cheap to
avoid before writing code and expensive to unwind afterwards.

Work through the phases in order. Announce which phase you are in.

## Phase 1 — Research

**Gate: do not write implementation code until the research note exists.**

Produce a short research note (in the conversation, or `docs/notes/<topic>.md` if the
work spans sessions) covering all four:

1. **The problem, restated.** If there is an issue: `gh issue view <N> -R GindaChen/nsys-ai --comments`.
   Restate what breaks and for whom. If the issue is vague, say so and ask — a vague
   issue implemented confidently is the most expensive outcome here.
2. **What the repo already does.** Search before designing. Half of "new" work in this
   repo already exists under a different name — see `references/REPO_MAP.md` for where
   to look and what the existing seams are.
3. **What you need to know that you do not.** Nsight Systems changes its SQLite schema
   between versions; NCCL/HTA accounting has conventions that are easy to get backwards.
   Look it up rather than infer. `references/RESEARCH.md` lists the sources that have
   actually paid off, and the facts already established (so you do not re-derive them).
4. **How other projects solved it.** ROADMAP.md already cites NAV, nsys_recipes,
   nsys_easy and the OpenHackathons bootcamp; HTA and PyTorch Profiler are the other
   standing references. Read the actual implementation, not the README. Say what you
   are taking and what you are deliberately not taking.

Ending Phase 1 with "this is straightforward" and nothing written means the phase did
not happen.

## Phase 2 — Plan

**Gate: the plan names every file it will touch, before any of them is opened for edit.**

Write the plan as a task list. Each task carries:

- **Files** — exact paths, split into create / modify / test.
- **Interface** — the signature or JSON shape it produces, and who consumes it.
- **Verification** — the command that proves the task landed, and the expected result.

Rules that keep plans honest, from `references/PLAN.md`:

- No `TBD`, no "handle edge cases", no "similar to task 2". If you cannot write the
  step concretely, the research phase is not finished.
- Split by responsibility, not by layer. A task that touches one module and its test is
  reviewable; a task called "add the SQL for everything" is not.
- Prefer the smallest change that is actually correct. This repo has a documented
  preference for minimizing diff surface — a refactor bundled into a feature PR makes
  both harder to review and is the most common review rejection here.

State the plan and get agreement before building anything non-trivial.

## Phase 3 — Build

Follow the plan. When reality contradicts the plan — and it will — stop, say what
changed, and revise the plan rather than quietly diverging.

Constraints that apply to every change in this repo:

- **Match the surrounding code.** No formatter is configured. `ruff check src/ tests/`
  must stay clean (`E,W,F,I,UP`; line length 100 but `E501` is not enforced).
- **Never `assert` for a safety guard in `src/`.** Bandit B101 fails the `security` CI
  job. Use `if not ...: raise ValueError(...)`.
- **Never hardcode Nsight table names.** Use the `{kernel_table}` / `{nvtx_table}` /
  `{memcpy_table}` placeholders — `Skill.execute` resolves them, because these tables
  are `_V2`/`_V3` suffixed on newer Nsight exports.
- **A skill that cannot run must abstain, not return `[]` and not raise.** Call
  `abstain(reason)` from `skills/base.py`. Empty list means "ran, found nothing";
  those are different answers and consumers depend on the difference. Formatters must
  not special-case abstention — `Skill.format_rows` handles it centrally.
- **New CLI subcommand → smoke test in `tests/test_cli.py`.** Not optional; CI checks it.
- **A new skipped test must be registered in `tests/test_ci_coverage.py`.** An
  unregistered skip fails the build, deliberately — a skip nobody reads is a test that
  does not exist.
- **Stay off new runtime dependencies.** SQL analysis and the web server are stdlib
  (`sqlite3`, `http.server`, `string.Template`) on purpose. No Flask, no Jinja2.
  AI features live behind the `[agent]` / `[chat]` extras and must degrade cleanly when
  those are absent.

Adding a builtin analysis skill has its own shape — read `references/BUILTIN_SKILL.md`
before starting one.

## Phase 4 — Hand off

Do not claim the work is done here. Invoke the **review-change** skill: it runs the
repo-specific review pass and the verification gate. Only after that reports clean does
the **contribute** skill open the PR.

## References

| File | Read when |
|------|-----------|
| `references/REPO_MAP.md` | Phase 1 — where things already live, and the seams to extend |
| `references/RESEARCH.md` | Phase 1 — external sources, and settled facts not to re-derive |
| `references/PLAN.md` | Phase 2 — plan format and the task template |
| `references/BUILTIN_SKILL.md` | Phase 3 — adding a skill to `src/nsys_ai/skills/builtins/` |
