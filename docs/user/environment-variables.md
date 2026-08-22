# Environment variables

Every setting below has a working default. This page exists for the cases where the default is wrong
for your machine — a shared cache location in CI, a memory-constrained box, reproducing a result from
an older release.

Variables are read at the point of use, not at startup, so exporting one for a shell session or
prefixing a single command both work:

```bash
NSYS_AI_INGEST=sqlite nsys-ai diagnose perf.nsys-rep
```

Unless noted, an unset variable and an empty one mean the same thing, and an unrecognised value is
ignored rather than fatal — some warn, most are silent.

## Storage and ingest

| Variable | Default | Description |
|---|---|---|
| `NSYS_AI_ARTIFACT_ROOT` | `.nsys-ai` (relative to the working directory) | Root for invocation-owned artifacts: derived sessions, profile capture outputs, locks, and default decision files. Relative values resolve against the command working directory. It does not move input-keyed Parquet/SQLite caches; an explicit `--session` or `--decision-out` still wins. |
| `NSYS_AI_INGEST` | `auto` | Which storage a profile is read from. `auto` converts `.nsys-rep` to a parquetdir and reads `.sqlite` in place; `parquetdir` refuses `.sqlite` inputs; `sqlite` converts `.nsys-rep` to `.sqlite`, refuses parquetdir inputs, and forces the cache mode to `direct` — see the note below. See [What to hand nsys-ai](./profile-inputs.md). |
| `NSYS_AI_CACHE_MODE` | `auto` | Whether the SQLite path builds a Parquet query cache. `direct` queries the `.sqlite` in place with no cache; `parquet` forces the cache even when it is judged unaffordable; `auto` decides on free disk and profile size. Ignored once the parquetdir backend is in use, and overridden entirely by `NSYS_AI_INGEST=sqlite`. An unrecognised value warns and falls back to `auto`. |
| `NSYS_AI_BASELINE_ROOT` | `.nsys-ai-baselines` (relative to the working directory) | Where the local baseline store lives. Point CI jobs at one absolute path so `baseline tag` and `diff --against baseline:X` agree regardless of which directory the job ran from. An explicit `--root` still wins. |

## Performance and memory

| Variable | Default | Description |
|---|---|---|
| `NSYS_AI_DUCKDB_THREADS` | DuckDB's own default (one per core) | `SET threads` for the query engine. Lower it on a shared machine. Non-numeric or non-positive values are ignored silently. |
| `NSYS_AI_DUCKDB_TEMP_DIRECTORY` | DuckDB's own default | Spill directory for large `GROUP BY` and join operations. Set it when the default temp filesystem is small or slow; the directory is created if missing. Failures to create or set it are ignored, and the query proceeds unspilled. |
| `NSYS_AI_DEFER_NVTX_KERNEL_MAP` | deferred | Set to `0`, `false`, `no` or `off` to build the NVTX-to-kernel map during cache construction instead of on first use. Deferring makes `warm` finish sooner; building eagerly makes the first `nvtx_kernel_map` query fast. |
| `NSYS_AI_ALWAYS_BUILD_NVTX_KERNEL_MAP` | unset | Set to `1`, `true`, `yes` or `on` to force the eager build. Overrides the two variables above, and is the setting that makes "cache ready" mean every artifact is present. |
| `NSYS_AI_DEFER_NVTX_KERNEL_MAP_MB` | unset | Make the choice size-dependent: defer only when the profile is at least this many megabytes, build eagerly below it. An unparseable value logs a warning and defers. |

## Skills and analysis

| Variable | Default | Description |
|---|---|---|
| `NSYS_AI_CUSTOM_SKILLS_DIR` | unset | Directory of additional skill definitions to load alongside the built-ins. Equivalent to `--skills-dir`, which takes precedence. |
| `NSYS_AI_SKILLS_DIR` | the packaged `agent_skills` directory | Where prompt and skill text files are loaded from. Intended for testing and for packaging layouts that relocate package data; changing it in normal use will make skills fail to load. |
| `NSYS_AI_ROOT_CAUSES_DIR` | unset | Directory of root-cause definitions for `root-cause`. Equivalent to `--root-causes-dir`, which takes precedence. |
| `NSYS_AI_MANIFEST_AUTO_TRIM` | enabled | Set to `0`, `false`, `no` or `off` to stop the profile health manifest from auto-selecting an iteration window. Disable it when you want the manifest to describe the whole capture. |

## AI features

These affect `ask`, `chat`, `agent` and the LLM-backed parts of `diagnose`. They do nothing in a
core install without the `[agent]` or `[chat]` extras.

| Variable | Default | Description |
|---|---|---|
| `NSYS_AI_MODEL` | the first model with credentials present | Preferred model ID. It is honoured only when the matching provider API key is also set; otherwise the first available model is used instead, silently. |
| `NSYS_AI_DB_AGENT` | disabled | Set to a truthy value to give the agent the `query_profile_db` tool, letting it write SQL against the profile rather than only calling skills. |

## Output

| Variable | Default | Description |
|---|---|---|
| `NSYS_AI_AGENT` | unset | Set to exactly `1` to switch CLI output to a machine-readable form intended for external AI agents. Any other value, including `true`, leaves human-readable output in place. |

## Notes on a few of these

**`NSYS_AI_INGEST=sqlite` also turns off the query cache.** This is the surprising part. Selecting
the SQLite policy sets the cache mode to `direct` for the whole run, so queries go against the
`.sqlite` in place and no Parquet cache is built — regardless of what `NSYS_AI_CACHE_MODE` says.
A `.sqlite` read under the default `auto` policy *does* get a cache when one is affordable, so the
same file is faster to query without this variable than with it.

Use it when you need the older behaviour reproduced exactly — comparing against a number produced
before the parquetdir backend became the default, or producing a `.sqlite` to open in another tool.
It is a compatibility path, not a deprecated one. Do not leave it set in a shell profile.

**The three NVTX-map variables interact.** `NSYS_AI_ALWAYS_BUILD_NVTX_KERNEL_MAP` is checked first
and wins outright. Then `NSYS_AI_DEFER_NVTX_KERNEL_MAP=0` forces the eager build. Only if neither
applies is `NSYS_AI_DEFER_NVTX_KERNEL_MAP_MB` consulted, and with none of the three set the map is
deferred. Setting the size threshold alone is the usual choice: it keeps small profiles eager, where
the sweep costs a fraction of a second, and defers the large ones where it does not.

**Ignored-value behaviour differs.** `NSYS_AI_CACHE_MODE` and `NSYS_AI_DEFER_NVTX_KERNEL_MAP_MB`
warn on values they cannot use; `NSYS_AI_DUCKDB_THREADS` and `NSYS_AI_DUCKDB_TEMP_DIRECTORY` fail
silently and keep the default. If a tuning variable appears to have had no effect, check the spelling
before concluding it does not work.

**`NSYS_AI_INGEST` and `NSYS_AI_CACHE_MODE` are different layers.** The first chooses which storage
is read; the second chooses whether the SQLite path builds a query cache. Setting the second has no
effect once the first has selected a parquetdir. See [Ingest policy](../dev/ingest-policy.md) for
where each is consumed.

**`NSYS_AI_ARTIFACT_ROOT` and `NSYS_AI_BASELINE_ROOT` relocate different stores.** The artifact root
holds invocation handoffs and default outputs; the baseline root holds named profile snapshots. Set
both explicitly in CI when the job needs portable sessions and a shared baseline, or set only the
one whose default location is unsuitable. A direct `--session` path is already an explicit handoff
location and does not need the artifact-root default.

## Related

- [What to hand nsys-ai](./profile-inputs.md) — the inputs each command accepts
- [Troubleshooting](./troubleshooting.md) — symptoms and what to change
- [Ingest policy](../dev/ingest-policy.md) — how storage selection is implemented
