# Adding a builtin analysis skill

A builtin skill lives in `src/nsys_ai/skills/builtins/<name>.py` and is auto-discovered:
`registry._load_builtins()` imports every module in that package and registers its
module-level `SKILL` (or each item of a `SKILLS` list). There is no registration list
to edit — but there is also nothing to catch a skill that silently fails to import.

Read `top_kernels.py` as the reference implementation; it exercises every part of the
contract.

## Before writing one

Check it does not already exist. There are 37, and the names are not always what you
would guess:

```bash
python -m nsys_ai skill list
rg -n "tags=" src/nsys_ai/skills/builtins/ | rg "<your concept>"
```

## Anatomy

```python
"""<One line: what this analyzes.>"""

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    """Return list[dict]. One dict per row, stable key set."""


def _format(rows):
    """Return the human-readable string. Do NOT handle abstention here."""


def _to_findings(rows, *, context=None):
    """Optional. Return list[Finding] for the evidence/timeline pipeline."""


SKILL = Skill(
    name="my_skill",            # snake_case, must match the CLI argument
    title="Human Readable Title",
    description="What it analyzes and when it is useful.",
    category="kernels",         # kernels | memory | nvtx | communication | system | utility
    execute_fn=_execute,        # or sql="SELECT ..." for the SQL path
    params=[SkillParam("limit", "Max rows", "int", False, 15)],
    format_fn=_format,
    to_findings_fn=_to_findings,
    tags=["...", "..."],        # drives skill discovery — be generous and specific
)
```

Two execution paths: set `sql=` for a template query (`Skill.execute` substitutes
params and table names), or `execute_fn=` for Python. Use `execute_fn` when the analysis
needs branching, a fallback path, or post-processing — the SQL path cannot do those.

## The five contracts

**1. Abstain, do not return `[]` and do not raise.**

`[]` means "ran, found nothing". If the profile lacks a table you need, that is a
different answer and callers depend on the difference — `EvidenceBuilder` catches
exceptions and logs, so a raising skill just vanishes from the output with no trace.

```python
from ..base import abstain, requires_nvtx

def _execute(conn, **kwargs):
    if (a := requires_nvtx(conn, needs="Layer attribution")) is not None:
        return a
    ...
    return abstain("This profile has no CUPTI_ACTIVITY_KIND_MEMCPY table, so "
                   "transfer sizes are unavailable. Re-capture with --trace=cuda.")
```

The reason is user-facing. Say what is missing, what that prevents, and what to do
about it. `Skill.format_rows` renders it — never write abstention handling into a
`format_fn`, and never index data columns before checking.

**2. Never hardcode Nsight table names.**

They are `_V2`/`_V3` suffixed on newer exports. In SQL templates use `{kernel_table}`,
`{runtime_table}`, `{nvtx_table}`, `{memcpy_table}`, `{memset_table}`, `{sync_table}`,
`{sync_type_table}`. In `execute_fn` resolve them:

```python
from nsys_ai.connection import wrap_connection
tables = wrap_connection(conn).resolve_activity_tables()
kernel_table = tables.get("kernel", "CUPTI_ACTIVITY_KIND_KERNEL")
```

For NVTX text, use `{nvtx_text_expr}` / `{nvtx_text_join}` — they cover both the legacy
`text` column and the `textId → StringIds` schema. Picking one breaks the other.

**3. Deterministic output.**

`ORDER BY` must be total — add a tiebreaker (`ORDER BY total_ms DESC, name ASC`). Two
test files exist solely because non-determinism broke golden-loop comparisons.

**4. Silence over a weak claim.**

If a signal is unavailable on the current path — `tc_eligible` is `None` on the pure
SQLite fallback, for example — emit nothing rather than guessing. A finding the user
cannot act on costs more than an absent one.

**5. Findings carry their own doubt.**

If you write `to_findings_fn`, each `Finding` needs `explanation`, `suggested_actions`,
`false_positive_notes`, and a `confidence` that actually varies with sample size.
Centralize thresholds as module constants with a comment saying why that number.
Use `type="highlight"` with `start_ns=0` when the span is the kernel's whole invocation
envelope — `type="region"` claims a localization you do not have.

## Wiring and tests

- Nothing to register — auto-discovery handles it. Confirm with `nsys-ai skill list`.
- If you also add a CLI subcommand: `cli/parsers.py` + `cli/handlers.py`, and a smoke
  test in `tests/test_cli.py`.
- Add `tests/test_<name>.py`. Cover: the happy path, the abstention path, and the
  fallback path if you wrote one.
- If a test must skip (needs a real profile, needs an API key), register the skip in
  `tests/test_ci_coverage.py` — an unregistered skip fails CI deliberately.

## Verify

```bash
python -m nsys_ai skill list | grep <name>
python -m nsys_ai skill run <name> tests/fixtures/mock.sqlite
python -m nsys_ai skill run <name> tests/fixtures/mock.sqlite --format json \
  | python -c "import json,sys; json.load(sys.stdin)"
pytest tests/test_<name>.py tests/test_abstention.py -v
ruff check src/ tests/
```

A skill that cannot run against `mock.sqlite` must print its abstention reason and
exit 0 — CI asserts exactly this shape for `nvtx_kernel_map`. A traceback fails the build.
