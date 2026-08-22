# Running deterministic analysis skills

A skill is one registered, deterministic analysis unit. It reads profile
evidence through the canonical ingest path and returns rows or an explicit
abstention. Skills do not need an LLM, an API key, or a model subscription, so
they are the lowest-cost way to answer a focused question and the easiest path
to put in a script.

The registry is the source of truth. It changes as skills are added; the
0.3.0 release-candidate checkout used for this page returned 38 entries. Do not
copy that number into automation or maintain a second catalog.

## Discover the registry

List the human-readable catalog:

```console
$ nsys-ai skill list
Name                            Category         Description
--------------------------------------------------------------------------------
gpu_idle_gaps                   kernels          Finds idle gaps between kernels
top_kernels                     kernels          Lists the heaviest GPU kernels
...
```

For scripting, `--format json` returns a JSON array. Count or filter it with
your JSON tool:

```console
$ nsys-ai skill list --format json | jq 'length'
38
$ nsys-ai skill list --format json | jq -r '.[] | [.name, .category] | @tsv'
```

The text catalog marks skills that require parameters with `*`. Inspect one
skill before running it:

```console
$ nsys-ai skill info top_kernels
{
  "name": "top_kernels",
  "parameters": {
    "limit": {"type": "int", "default": 15, "required": false}
  }
}
```

`skill info` is the authoritative parameter spelling. Do not infer a
parameter name from a different skill.

## Run one skill

The basic form is:

```console
$ nsys-ai skill run top_kernels PROFILE.sqlite
```

Use JSON when another program will consume the result, and cap the number of
rows when the result will be passed into a prompt or log:

```console
$ nsys-ai skill run top_kernels PROFILE.sqlite \
    --format json --max-rows 15
```

The JSON result is an array of rows. If the cap removed rows, the command adds
a final metadata row with `_truncated`, `_total_rows`, and `_shown_rows`.
`--trim START END` uses seconds on the profile's capture clock; `--iteration N`
can derive a window from an iteration marker. Use `--no-cache` when a one-shot
SQLite query should avoid creating a Parquet cache, accepting slower repeated
queries.

## Parameters are validated

Pass parameters with repeated `--param` / `-p` flags:

```console
$ nsys-ai skill run region_mfu PROFILE.sqlite \
    -p name=Attention \
    -p theoretical_flops=1e12 \
    -p peak_tflops=20 \
    -p source=nvtx \
    -p device_id=0
```

`region_mfu` uses `device_id`; `iteration_timing` uses `device`. The registry
rejects an unknown parameter and prints the valid names, so this is an error:

```console
$ nsys-ai skill run region_mfu PROFILE.sqlite -p device=0
Error: unknown parameter 'device' for skill 'region_mfu'.
```

Required values are checked before profile work starts. For example,
`region_mfu` requires both `name` and `theoretical_flops`; provide
`peak_tflops` when the profile does not contain a recognizable GPU model.

## Match the skill to the question

| Question | Start with | Useful follow-up |
|---|---|---|
| Which kernels dominate GPU time? | `top_kernels` | `kernel_instances`, `kernel_launch_overhead` |
| Why is a stream idle? | `gpu_idle_gaps` | `cpu_gpu_pipeline`, `sync_cost_analysis` |
| Is communication competing with compute? | `overlap_breakdown` | `nccl_breakdown`, `nccl_anomaly` |
| Where is data movement happening? | `memory_transfers` | `h2d_distribution`, `memory_bandwidth` |
| Are iterations repeating and stable? | `iteration_timing` | `iteration_detail` |
| Is the capture healthy enough to trust? | `profile_health_manifest` | `schema_inspect` |
| What likely root cause matches the evidence? | `root_cause_matcher` | the detection skill named in the result |

This table is a starting map, not a replacement for the registry. A skill may
abstain when its required table, marker, or denominator is absent.

## Read abstention honestly

An empty array means the skill ran and found no rows. An object/row carrying
`_abstained: true` means it could not make a defensible claim and includes a
`reason`. Do not convert the latter into “healthy”.

For example, running `iteration_timing` against a profile with no NVTX table
returns an abstention explaining that iteration detection needs NVTX
annotations. Re-capture with `--trace=cuda,nvtx` or choose a skill whose inputs
the profile actually contains.

## When to use a higher-level command

Use `diagnose` when you want the default pack, ranked findings, limitations,
and a session artifact. Use `ask` or `agent ask` when you have a natural
language question and want several registered skills selected for you. Use
`chat` for an interactive terminal conversation; it needs a real terminal and
the optional chat/agent dependencies. In every case, the underlying evidence
should remain visible in the session or JSON output.

## Use the agent command family

The `agent` commands are the CLI-facing analysis family. They use the same
registered skills and evidence-first runner, but expose different levels of
automation:

```console
# Full deterministic report; --trim uses seconds.
$ nsys-ai agent analyze PROFILE.sqlite
$ nsys-ai agent analyze PROFILE.sqlite --trim 10 20 --evidence -o findings.json

# Question-driven evidence, from a profile or a session handoff.
$ nsys-ai agent ask PROFILE.sqlite "why is the GPU idle?"
$ nsys-ai agent ask --session /tmp/nsys-run-001 "what should I verify next?"

# Print the onboarding guide for an external agent or automation.
$ nsys-ai agent-guide > agent-guide.txt
```

`agent analyze` runs the fixed analysis pack and can emit a findings JSON file;
it does not need an LLM. `agent ask` selects up to four registered skills and
returns an evidence-first answer. Without model credentials it uses the
deterministic keyword selector and local synthesis, so the command remains
usable in a core installation. The profile form and the session form were
both tested against the same fixture and produced the same evidence shape.

For optional LLM triage or synthesis, install the extra and configure a
provider credential in the environment:

```console
$ pip install 'nsys-ai[agent]'
$ export ANTHROPIC_API_KEY=...  # or the key for another supported provider
```

`NSYS_AI_MODEL` can select a model when its matching provider key is present;
see [environment variables](environment-variables.md). Never put provider
keys in a session directory, findings file, or issue report. `agent-guide` is
static output for external agents and requires neither a profile nor a key.

For authoring a new skill, switch to the developer
[skill contract](../dev/skill-contract.md); this page is intentionally about
operating the existing registry.
