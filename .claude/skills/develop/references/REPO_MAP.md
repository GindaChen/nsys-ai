# Where things already live

Read this in Phase 1, before designing anything. The recurring failure mode in this
repo is building a second implementation of something that already exists under a
different name.

## Search first

```bash
rg -n "<concept>" src/nsys_ai/          # the concept, not the filename you imagined
python -m nsys_ai skill list            # 37 builtin analysis skills — is yours there?
python -m nsys_ai --help                # ~30 subcommands — is the command there?
gh issue list -R GindaChen/nsys-ai --search "<keywords>" --state all
gh pr list -R GindaChen/nsys-ai --search "<keywords>" --state all
```

The last two matter as much as the code search: a closed issue often records *why* the
obvious approach was rejected, and an open PR may already be touching your files.

## The seams

| You want to add | Extend here | Not here |
|---|---|---|
| A profile analysis | `src/nsys_ai/skills/builtins/<name>.py` (see `BUILTIN_SKILL.md`) | A new top-level module |
| A CLI subcommand | `cli/parsers.py` (argparse) + `cli/handlers.py` (`_cmd_*`) | `__main__.py` — it only delegates to `cli/app.py:main` |
| A profile-level query helper | `profile.py` (`Profile`, `NsightSchema`) | Inline `sqlite3` in a new module |
| Schema/table-name resolution | `connection.py` (`wrap_connection`, `resolve_activity_tables`) | Hardcoded `CUPTI_ACTIVITY_KIND_*` strings |
| Before/after comparison | `diff.py` / `diff_tools.py` / `diff_render.py` / `diff_web.py` | A parallel comparison path |
| A web page or endpoint | `web.py` (stdlib `http.server`) + `templates/` (`string.Template`) | Flask, Jinja2, or any new server dep |
| A TUI view | `tree/` or `timeline/` (Textual) | A curses or Rich-only reimplementation |
| Agent behavior | `agent/persona.py` (prompt) / `agent/loop.py` (orchestration) | Prompt text scattered into call sites |
| LLM tool plumbing | `chat.py`, `chat_tools.py`, `chat_config.py`, `tools_profile.py` | A new provider client |
| Shared duration/number formatting | `formatting.py` (`fmt_dur`, `fmt_ns`, `fmt_relative`) | A local helper |

## Contracts you must not break

- **Abstention** — `skills/base.py`: `abstain()`, `is_abstention()`, `requires_nvtx()`.
  `[]` means "ran, nothing found"; `[{"_abstained": True, ...}]` means "could not run".
  Rendering of abstention is centralized in `Skill.format_rows` — do not re-handle it
  in a `format_fn`. Tested by `tests/test_abstention.py`.
- **Versioned tables** — SQL templates use `{kernel_table}`, `{runtime_table}`,
  `{nvtx_table}`, `{memcpy_table}`, `{memset_table}`, `{sync_table}`. `Skill.execute`
  substitutes them from `resolve_activity_tables()`.
- **NVTX text** — `{nvtx_text_expr}` / `{nvtx_text_join}` cover both the legacy `text`
  column and the modern `textId → StringIds` schema. Do not pick one.
- **Overlap accounting** — `overlap_ms` counts as *compute* (HTA convention). Exposed
  communication is `exposed_comm_ms`; there is no `communication_ms` field. Getting this
  backwards silently inverts every diff verdict.
- **Determinism** — `tests/test_determinism.py` and `test_determinism_outside_skills.py`
  exist because non-deterministic output broke golden-loop tests. Sort before emitting.
- **Trim + device params** — many skills take `trim_start_ns`/`trim_end_ns` and `device`;
  `region_mfu` uses `device_id`, not `device`. Check the skill's `params` before assuming.

## Tests

`tests/` is large and named by subject — `test_<module>.py` or `test_<skill>.py`. Two
are load-bearing beyond their own subject:

- `tests/test_cli.py` — every CLI subcommand needs a smoke test here.
- `tests/test_ci_coverage.py` — the registry of accepted skips. A new skip that is not
  registered fails CI on purpose.

Integration tests need `NSYS_TEST_PROFILE=tests/fixtures/h100_2gpu_1s.sqlite`; without
it they skip (which `test_ci_coverage.py` knows about).
