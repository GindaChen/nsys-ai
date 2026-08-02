# Review checklist — nsys-ai

These are the contracts specific to this repo. Each has shipped as a bug at least once,
which is why it is written down rather than left to judgement. A general-purpose review
will not catch them.

## Blocking — CI fails, or behavior is wrong

- [ ] **No `assert` used as a safety guard in `src/`.** Bandit B101 fails the `security`
      job. Use `if not ...: raise ValueError(...)`. (B110 and B608 are skipped in
      `pyproject.toml`; B101 is not.)
- [ ] **`ruff check src/ tests/` clean.** Rules `E,W,F,I,UP`. `E501` is not enforced,
      so long lines are fine; unsorted imports (`I`) and outdated syntax (`UP`) are not.
- [ ] **No hardcoded Nsight table names.** `CUPTI_ACTIVITY_KIND_KERNEL` and friends are
      `_V2`/`_V3` suffixed on newer exports. SQL templates use `{kernel_table}` etc.;
      `execute_fn` resolves via `wrap_connection(conn).resolve_activity_tables()`.
      A hardcoded name works on the fixture and fails on the user's profile.
- [ ] **NVTX text handled both ways.** `{nvtx_text_expr}` / `{nvtx_text_join}`, not a
      bare `n.text` and not a bare `textId` join.
- [ ] **Abstention contract.** A skill that cannot run returns `abstain(reason)` — not
      `[]`, not an exception. `[]` means "ran, found nothing" and callers act on the
      difference. Rendering is centralized in `Skill.format_rows`; a `format_fn` must
      not re-handle abstention, and must not index data columns that an abstention row
      does not have.
- [ ] **Overlap accounting.** `overlap_ms` counts as compute (HTA convention). Exposed
      communication is `exposed_comm_ms`. There is no `communication_ms` — a reference
      to it is either a typo or an inverted metric.
- [ ] **New CLI subcommand has a smoke test in `tests/test_cli.py`.**
- [ ] **New skipped test is registered in `tests/test_ci_coverage.py`.** An
      unregistered skip fails the build on purpose: a skip nobody reads is a test that
      does not exist.
- [ ] **No new runtime dependency.** SQL analysis and the web server are stdlib
      (`sqlite3`, `http.server`, `string.Template`) deliberately — no Flask, no Jinja2.
      AI features stay behind the `[agent]` / `[chat]` extras and must degrade cleanly
      when those are not installed.
- [ ] **No secrets in any output.** Never inline PR text with `--body "..."`; write a
      file and use `--body-file`. Never echo env values, keys, or tokens into commit
      messages, PR bodies, comments, or logs.

## Important — correctness risk

- [ ] **Deterministic ordering.** `ORDER BY` has a total tiebreaker. Non-determinism
      breaks the golden-loop tests, which is why `tests/test_determinism.py` and
      `test_determinism_outside_skills.py` exist.
- [ ] **Silence over a weak claim.** When a signal is unavailable on the current path
      (`tc_eligible` is `None` on the pure-SQLite fallback), emit nothing rather than
      guessing. An unactionable finding costs more than an absent one.
- [ ] **Findings carry doubt.** A new `Finding` has `explanation`, `suggested_actions`,
      `false_positive_notes`, and a `confidence` that varies with sample size. Magic
      thresholds are module constants with a comment saying why that number.
- [ ] **Params match the skill.** `region_mfu` takes `device_id`; most others take
      `device`. Passing the wrong one fails silently — it does not scope, and does not
      error.
- [ ] **Fallback paths are tested.** If the change adds a "try the cache, else raw
      SQLite" branch, both branches need coverage. The fallback is the one users on old
      exports actually hit.
- [ ] **Tests assert behavior, not implementation.** A test that would still pass with
      the function body deleted is not coverage.

## Important — scope

- [ ] **Minimal diff.** A refactor bundled into a feature is the most common rejection
      here. If the refactor was genuinely needed first, it should be a separate commit
      with its own justification, and the PR should say so.
- [ ] **No dead code left behind.** Superseded helper, unused import, commented-out
      block — delete it. Git has the history.
- [ ] **Reuses existing seams.** A second implementation of formatting, schema
      resolution, or overlap classification is a defect even if it works. The seam list
      is in `.claude/skills/develop/references/REPO_MAP.md`.

## Minor — but check

- [ ] **Docs match the code as it is now.** If the change alters behavior described in
      `README.md`, `CLAUDE.md`, `AGENTS.md`, or `docs/`, update it in the same PR.
- [ ] **New docs are reader-first**: no emoji, no enumeration of PR numbers, and they
      follow the conventions of comparable projects rather than inventing a format.
- [ ] **Comments say why, not what.** The code already says what.

## When reviewing PR feedback rather than your own diff

Order the work: blocking breakage first, then trivial fixes (typo, import), then the
ones needing thought. Test each fix separately so a later one does not mask an earlier
regression. Reply in the inline comment thread, not as a new top-level PR comment.
State what changed — no thanks, no preamble.
