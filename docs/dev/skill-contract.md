# Skill contract

A skill is the smallest reusable unit of profile analysis: a registered name, a query or Python
executor, declared parameters, and an optional formatter. Skills do not ask an LLM to write SQL. The
registry selects the skill; `Skill.execute()` resolves the profile-specific schema and runs it; the
caller decides whether rows become findings, text, JSON, or session evidence.

This page is for someone adding the next skill. It records the parts that are enforced centrally and
the parts a skill author must implement. It is not a copy of the skill catalog: use `nsys-ai skill
list` and `nsys-ai skill info <name>` for the inventory.

## The execution path

Use the registry as the only public dispatch path:

```python
from nsys_ai.skills.registry import run_skill

rows = run_skill(
    "top_kernels",
    conn,
    raw=True,
    device=0,
    trim_start_ns=start_ns,
    trim_end_ns=end_ns,
)
```

`run_skill(skill_name, conn, raw=False, **kwargs)` does three things:

1. `get_skill()` validates the name and raises `SkillNotFoundError` with the available names.
2. `Skill.execute()` applies defaults, validates required parameters, resolves activity tables,
   checks declared requirements, runs the SQL or `execute_fn`, and returns `list[dict]`.
3. `raw=False` formats those rows for a human; `raw=True` returns the structured rows for callers
   such as chat, MCP, evidence, and the ask runner.

Do not call a builtin's private `_execute()` function from a transport. Do not add a second skill
registry to a CLI command or a chat tool. If the shared path lacks a capability, extend
`skills/base.py` or `skills/registry.py` and test that contract once.

## Smallest useful SQL skill

Put a module under `src/nsys_ai/skills/builtins/` and export a `SKILL` constant. The registry walks
that package and discovers it automatically; no registry edit is needed.

```python
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
```

The example deliberately uses the shared placeholders. A Python-level skill should use
`execute_fn=...` when it needs interval math, multiple queries, or a result that cannot be expressed
as one template. It must still return rows with the same contract and declare any activity tables it
cannot operate without in `required_tables`.

The optional `format_fn(rows)` is a presentation adapter only. It must not query the profile again,
change the meaning of the rows, or assume an abstention row has normal metric columns.

## Parameters

`SkillParam` is the declaration of the public parameter surface:

```python
SkillParam(
    name="limit",
    description="Maximum number of rows",
    type="int",
    required=False,
    default=10,
)
```

`Skill.execute()` applies a declared default and raises `SkillParameterError` for a missing required
parameter. It preserves caller-supplied values. `trim_start_ns` and `trim_end_ns` are shared runtime
arguments and therefore do not need to be repeated in every skill's `params` list.

Unknown keyword arguments are currently preserved for compatibility and may be ignored by a skill.
That makes parameter spelling a correctness issue rather than a reliable validation error. In
particular, `region_mfu` takes `device_id`, not `device`; passing `device=1` does not select GPU 1.
Use the exact name shown by `skill info`, and add a test for every new selector.

The following table is the complete list of parameters other than the ordinary `device` selector
for the 38 built-in skills in this release. A name in the Required column has no default.

| Skill | Parameters other than `device` | Required |
|---|---|---|
| `arithmetic_intensity` | `theoretical_flops`, `bytes_moved`, `peak_tflops`, `hbm_bw_gbps` | `theoretical_flops` |
| `code_attribution_candidates` | `start_ns`, `end_ns`, `limit`, `min_overlap_pct` | `start_ns`, `end_ns` |
| `cpu_gpu_pipeline` | `limit` | — |
| `cutracer_analysis` | `trace_dir` | `trace_dir` |
| `gpu_idle_gaps` | `min_gap_ns`, `limit` | — |
| `host_sync_parent_ranges` | `limit`, `patterns` | — |
| `iteration_detail` | `iteration`, `marker` | `iteration` |
| `iteration_timing` | `marker` | — |
| `kernel_instances` | `name`, `limit` | — |
| `kernel_launch_overhead` | `limit`, `min_launches` | — |
| `kernel_launch_pattern` | `limit` | — |
| `memory_bandwidth` | `limit` | — |
| `nccl_anomaly` | `threshold`, `limit` | — |
| `nvtx_kernel_map` | `limit` | — |
| `nvtx_layer_breakdown` | `limit`, `depth`, `auto_depth`, `report` | — |
| `region_mfu` | `name`, `theoretical_flops`, `source`, `peak_tflops`, `num_gpus`, `occurrence_index`, `device_id`, `match_mode` | `name`, `theoretical_flops` |
| `speedup_estimator` | `iteration_ms`, `compute_ms`, `nccl_ms`, `idle_ms`, `overlap_pct`, `tp_degree`, `model_params_b`, `gpu_memory_gb` | `iteration_ms` |
| `stream_concurrency` | `limit` | — |
| `tensor_core_usage` | `limit` | — |
| `theoretical_flops` | `operation`, `hidden_dim`, `seq_len`, `num_layers`, `ffn_dim`, `batch_size`, `multiplier`, `M`, `N`, `K` | `operation` |
| `top_kernels` | `limit` | — |

Skills not listed here either have no declared parameters or only use `device`; all skills can also
receive the shared trim arguments. The table is generated from the registry's `SkillParam` values in
the current release. If a parameter is renamed, update this page and the skill's tests in the same
change.

## Schema portability

### Activity tables are placeholders

Nsight Systems versions can suffix activity tables (`..._V2`, `..._V3`) or expose equivalent views in
the Parquet backend. A SQL skill must not hardcode the canonical table name. `Skill.execute()` resolves
these placeholders through `ConnectionAdapter.resolve_activity_tables()`:

| Template placeholder | Resolver key | Canonical table |
|---|---|---|
| `{kernel_table}` | `kernel` | `CUPTI_ACTIVITY_KIND_KERNEL` |
| `{runtime_table}` | `runtime` | `CUPTI_ACTIVITY_KIND_RUNTIME` |
| `{nvtx_table}` | `nvtx` | `NVTX_EVENTS` |
| `{memcpy_table}` | `memcpy` | `CUPTI_ACTIVITY_KIND_MEMCPY` |
| `{memset_table}` | `memset` | `CUPTI_ACTIVITY_KIND_MEMSET` |
| `{sync_table}` | `sync` | `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION` |
| `{sync_type_table}` | `sync_type` | `ENUM_CUPTI_SYNC_TYPE` |

If a SQL template uses a table that is absent, `Skill.execute()` returns `abstain(...)` rather than
letting a missing-table error become a false diagnosis. For a Python executor, declare the resolver
keys in `required_tables` when the whole skill cannot answer without them:

```python
SKILL = Skill(
    # ...
    execute_fn=_execute,
    required_tables=("kernel", "runtime"),
)
```

Do not use `get("kernel", "CUPTI_ACTIVITY_KIND_KERNEL")` as a substitute for the contract. The
fallback is only safe after the resolver has established that the profile really has that table.
Some skills intentionally return partial results when an optional activity is absent; document that
choice and test it rather than declaring a requirement that is not real.

### NVTX text has two schema shapes

For SQL that displays NVTX text, use `{nvtx_text_expr}` and `{nvtx_text_join}`:

```sql
SELECT {nvtx_text_expr} AS range_name
FROM {nvtx_table} n
{nvtx_text_join}
```

`Skill.execute()` selects the legacy `n.text` expression or the modern `textId`/`StringIds` join
based on the profile. A bare `n.text`, a bare `textId`, or a hand-written exact `NVTX_EVENTS` lookup
is not portable across the supported exports.

## Time windows

The runner passes `trim_start_ns` and `trim_end_ns` in nanoseconds. Every query that reports a time,
count, or ratio must honour both when they are present. A trim is a closed activity selection in the
existing skill contract: use the same start/end predicate as the surrounding skill and do not mix
trimmed rows with whole-profile denominators.

For a SQL template, include `{trim_clause}` at the point where the kernel alias is `k`; the base class
injects the correct predicate and injects an empty string for a whole-profile run. For an
`execute_fn`, read both kwargs and apply the equivalent filter in every sub-query. A helper query that
quietly omits the window is enough to make the final answer wrong.

Test both cases:

```python
assert len(skill.execute(conn)) == whole_profile_count
trimmed = skill.execute(conn, trim_start_ns=300, trim_end_ns=400)
assert trimmed == expected_window_rows
```

Also test an empty or out-of-range window. A skill that cannot determine whether the window is healthy
must abstain with a reason; it must not return a clean-looking zero row.

## Abstention is not an empty result

Use the shared helper when the skill cannot run:

```python
from ..base import abstain

return abstain(
    "This profile has no NVTX_EVENTS table, so region attribution is unavailable.",
    missing_tables=["NVTX_EVENTS"],
)
```

The three states are different:

| Return value | Meaning | Downstream action |
|---|---|---|
| `[]` | The skill ran and found nothing to report | May be rendered as clean/no results |
| `abstain(reason)` | The skill could not answer | Show the reason; do not use it as evidence |
| normal rows | The skill ran and produced data | Format or convert to findings |

The marker is `_abstained: true`, with a non-empty `reason`. Use `is_abstention(rows)` or
`is_abstention_row(row)` instead of checking the marker string in a consumer. `Skill.format_rows()` is
the one shared renderer for a whole-skill abstention and prints the reason without invoking a
formatter that expects metric columns. `EvidenceBuilder` filters abstentions before calling a
`to_findings_fn`; an unavailable skill therefore cannot mint a finding from its error row.

A composite skill may retain valid rows from checks that ran and add an explicit incomplete/abstention
row for a check that could not compute. In that case, use `is_abstention_row()` per row and keep the
incomplete row out of evidence. Do not collapse the partial result into `[]` or silently discard the
valid rows.

## Determinism and serialization

Structured rows cross the CLI, Web, MCP, chat, session, and test boundaries. A skill's `execute()`
result must be a `list[dict]` whose values are JSON-serialisable. Run the same query through both the
SQLite compatibility path and the Parquet/DuckDB path when the skill supports both. Avoid connection
objects, cursors, `Path` instances, NumPy scalars, and NaN/Infinity in rows.

Every ordered query needs a total order. If `total_ms` ties, add a stable secondary key such as the
kernel name; otherwise SQLite and DuckDB are free to return equal rows in different orders. This is
why a query like the example uses `ORDER BY invocations DESC, device_id ASC`, and why
`tests/test_determinism.py` exists.

The registry's `raw=True` result is the contract consumed by structured callers. The default result is
formatted text, so `len(run_skill(...))` without `raw=True` is a character count, not a row count.

## Checklist for a new skill

- [ ] Add one builtin module with a `SKILL` constant; do not edit the registry.
- [ ] Give it a stable name, useful title, description, category, and either SQL or `execute_fn`.
- [ ] Declare every user-facing parameter with `SkillParam`, including required values and defaults.
- [ ] Use activity-table and NVTX text placeholders instead of canonical table literals.
- [ ] Use `{trim_clause}` or apply `trim_start_ns`/`trim_end_ns` in every executor query.
- [ ] Declare `required_tables` for execute functions that cannot run without an activity table.
- [ ] Return `abstain(reason)` when the skill cannot answer; reserve `[]` for a clean empty result.
- [ ] Return JSON-serialisable rows and a deterministic total order.
- [ ] Add fixture tests for normal output, missing activity, trim scope, parameter behaviour, and
      serialization. Add a versioned-table or Parquet test when the skill uses a resolved table.
- [ ] Run `nsys-ai skill list`, the focused skill tests, and the complete suite before opening the PR.

The contract is enforced primarily in `src/nsys_ai/skills/base.py` and
`src/nsys_ai/skills/registry.py`; the tests named above are executable examples of the intended
behaviour.
