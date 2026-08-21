# Known limits and trust boundaries

nsys-ai is deliberately conservative when it can prove that a number is not supported by the
capture. It cannot make every number meaningful, however. This page records the boundaries that are
easy to miss when a command returns valid JSON and a polished sentence.

Use it when deciding whether to act on a finding, not only after a command fails. For symptoms and
recovery steps, see [Troubleshooting](./troubleshooting.md). For the mechanics of `--trim`, see
[Time windows](./time-windows.md). For the semantics of a before/after verdict, see [Reading a
diff](./reading-a-diff.md).

## The trust model in one minute

Every result has two separate questions:

1. **Did the analyzer run?** A missing table, empty window, unsupported field, or ambiguous
   denominator may make it abstain.
2. **What does the number measure?** A duration, a union of intervals, a sum across streams, and a
   percentage of wall time are not interchangeable even when they are all printed in milliseconds
   or percent.

Read the result in this order:

| Signal | Meaning | Safe interpretation |
|---|---|---|
| `_abstained: true` | The analyzer could not make a trustworthy claim | Read the reason; do not rank it as a finding or as proof of health |
| `skipped` / unavailable | The required activity or field was not present | The capture cannot answer that question; re-capture only if the question matters |
| `[]` | The analyzer ran and found no matching rows | This is not the same as an abstention, but check the skill's summary and scope |
| `confidence` | Evidence quality used by the analyzer | A ranking signal, not a probability that the diagnosis is true |
| `severity: info` | Informational output | Not automatically healthy; an info row can explain an abstention |

The CLI normally catches invalid profile windows before a skill starts. The lower-level registry and
transport APIs accept already-normalized parameters, so their caller still has to preserve the
same scope and abstention contract. That distinction is the most important limitation below.

## 1. A valid-looking empty window is not a healthy profile

### What is bounded today

For most profile-backed CLI commands, a `--trim` window that selects no part of the capture fails
with `TRIM_OUT_OF_RANGE`. This prevents the common mistake of reading an empty query as "there were
no events". The capture clock is not guaranteed to start at zero; ask `info` for the actual range.

```console
$ nsys-ai info PROFILE.sqlite
Profile: PROFILE.sqlite
  Time: 60.112s - 60.411s

$ nsys-ai skill run top_kernels PROFILE.sqlite --trim 0 1
Error [TRIM_OUT_OF_RANGE]: --trim 0.000 1.000 selects no part of this profile...
```

The exact range and error text depend on the profile. A partially overlapping window is clipped and
is valid; only a window with no selected activity is rejected by the CLI guard.

### The remaining boundary: direct skill/API calls

The preflight guard belongs to the command adapter. A caller that invokes the skill registry with
nanosecond bounds can still ask an individual skill to inspect an empty interval. The skills do not
all have the same empty-window policy:

```text
gpu_idle_gaps       returns a summary row with zero gaps whose note says the GPU is well-utilized
root_cause_matcher  returns {_abstained: true, reason: ...}
```

That difference matters to Web, chat, plugins, and tests that call `run_skill(raw=True)` directly.
Do not build a new transport by forwarding trim values to a skill and assuming every skill will
reject an impossible window. Validate the profile range at the transport boundary, or preserve the
skill's abstention row all the way to the consumer.

The command-level fix for the original root-cause false positive is tracked in [#493](https://github.com/GindaChen/nsys-ai/issues/493)
and the follow-up is in [#504](https://github.com/GindaChen/nsys-ai/pull/504). Those changes are in
0.3.0; the direct-API distinction remains a contract for new callers.

### How to protect an automation

1. Use `nsys-ai info` to get the capture-clock range.
2. Prefer `--iteration N` for a paired logical step; each profile resolves its own clock.
3. Treat `TRIM_OUT_OF_RANGE` as an input error, not as an empty result.
4. If calling the registry, check `is_abstention_row()` / `_abstained` before accepting rows as
   evidence.

## 2. Percentages have denominators, and some denominators overlap

A percentage is only meaningful together with its numerator and denominator. nsys-ai reports several
different kinds of percentage:

| Example | Numerator | Denominator | Main caveat |
|---|---|---|---|
| GPU idle percentage | idle interval union or gap estimate | selected device/window span | Per-stream gap totals can exceed device-level idle; use the labelled device value when available |
| overlap percentage | time in the classified overlap | selected span or classified category | Streams and GPUs can overlap; this is not a serial wall-clock sum |
| synchronization ratio | CPU synchronization intervals | runtime wall span | Host threads can block concurrently, so the summed duration can exceed the wall span |
| MFU / efficiency | achieved work rate | supplied or detected hardware ceiling | A missing GPU model or wrong world-size/FLOP input makes the denominator unavailable or misleading |

### The >100% synchronization case

On a real four-GPU capture, the root-cause output includes this shape:

```text
149767 sync calls total 2176889.6ms, which is 271.6% of the
801474.5ms runtime wall span. Concurrent host-thread intervals overlap...
```

That is not a 271.6% utilization measurement and it is not evidence that the GPU ran for more than
the capture. It is a sum of host-side synchronization durations divided by one wall-clock span.
Several host threads can be blocked at the same time. In 0.3.0 this case is emitted as an
`_abstained` informational row with an explicit reason, so it must not be promoted to a normal
warning or used as a ranked diagnosis.

If you need a ranked synchronization cost, attribute intervals per host thread first, or supply a
thread-aware denominator. Until then, retain the row as a limitation record and investigate the
underlying synchronization calls separately.

### The GPU idle case

`gpu_idle_gaps` exposes both the sum of gaps observed per stream and a device-level idle estimate
when the sweep can compute it. A stream can be idle while another stream keeps the device busy, so
the per-stream sum is not recoverable device time. Do not add per-stream percentages across streams.

## 3. Device selection can change the population being analysed

Multi-GPU profiles have two related but different choices:

- the device requested by the caller; and
- a device with activity in the selected window.

The root-cause matcher can fall back to an active device when the requested device has no kernels in
the selected window. This avoids turning an idle rank into a false all-clear, but it also means that
"GPU 0" in a script is not enough to establish that every reported row describes GPU 0. Some
root-cause rows carry `analysed_device`; generic skill rows do not all carry an equivalent field.

Check the population before comparing per-device numbers:

```console
$ nsys-ai info PROFILE.sqlite
```

Then pass `--gpu N` (or the skill's `device=N` parameter) deliberately and keep the selected device
in the session or JSON record. If a requested device is not present in the profile at all, the
command should report the available devices rather than silently inventing one.

For a pair comparison, the default `diff` scope is all GPUs. A single-GPU comparison is a different
question and must say so with `--gpu N`; see [Reading a diff](./reading-a-diff.md#comparability-confidence).

## 4. Parquet and SQLite are compatible storage paths, not identical evidence sources

The supported inputs are `.nsys-rep`, `.parquetdir`, and `.sqlite`. The first two use the Parquet
analysis backend; SQLite remains a compatibility path. This is a storage distinction, but it can
affect which derived fields are available and how much work the first query performs.

### Tensor Core status is derived, not an instruction trace

The canonical `kernels` view exposes `is_tc_eligible` and `uses_tc` on both the cache and direct
SQLite-compatible paths in 0.3.0. On a direct path, those values are inferred from kernel names and
available metadata; they are not a disassembly of the executed SASS. A `tc_active` or
`tc_eligible_inactive` result is a useful routing clue, not proof of the exact Tensor Core
instruction mix.

If the field is missing or `null` in an older cache or a custom export, the `top_kernels` analyzer
must suppress the Tensor Core finding. Treating unknown as false would turn an unavailable field into
a claim that Tensor Cores were not used.

### Schema and trace coverage still limit every backend

Parquet conversion does not create activity that Nsight did not record. If the capture has no kernel
table, no NVTX, no runtime calls, or no NCCL payloads, the corresponding skills remain unavailable
or have lower fidelity. `doctor` and `diagnose` report these limitations; they are not a reason to
re-capture unless that analysis is needed.

```console
$ nsys-ai doctor PROFILE.sqlite --format json
$ nsys-ai diagnose PROFILE.sqlite --format json
```

For export-version differences, compare the reported schema against the [support
matrix](../support-matrix.md). A profile can be a valid SQLite file and still lack the table or
column required by one skill.

## 5. Large profiles are supported, but cold derived state is not free

The 0.3.0 Web path binds before its background NVTX warmup, so a large capture should produce a port
and a shell while the tree is building. That fixes the earlier "no port for 15 minutes" failure
([#490](https://github.com/GindaChen/nsys-ai/issues/490)); it does not make conversion, Parquet
scans, or NVTX attribution free.

On the real profile used for the release checks:

```text
perf.nsys-rep       605 MB
perf.parquetdir     717 MB, 48 Parquet files
```

The first conversion and first NVTX-to-kernel map build are derived-state costs. They can consume
more disk and memory than a small fixture suggests. Run `warm` deliberately before a demo, batch, or
Web session:

```console
$ nsys-ai warm PROFILE
```

If the machine is shared, set `NSYS_AI_DUCKDB_THREADS` and
`NSYS_AI_DUCKDB_TEMP_DIRECTORY`; if the profile directory cannot be written, use a pre-built
`parquetdir` or choose the direct SQLite compatibility path. See [profile inputs](./profile-inputs.md)
and [environment variables](./environment-variables.md#performance-and-memory).

### A bounded tree is not a complete tree

`/api/tree` now caps children per node and total serialized nodes (`limit` defaults to 256 and is
capped at 2000; `max_nodes` defaults to 10000 and is capped at 20000). A response can contain
`truncated: true`, `has_more: true`, and `children_total` while remaining a successful HTTP
response. This is intentional: a successful response means "a bounded slice is ready", not "every
descendant was returned". The bound fixes the earlier 72 MB depth-one response ([#506](https://github.com/GindaChen/nsys-ai/issues/506)); it does not provide cursor pagination.

The viewer marks partial data as partial. API consumers must do the same and must not conclude that a
missing child does not exist. Very high-fan-out children remain unreachable through the current
maximum page size; use the timeline view or a narrower time window for exploration.

## 6. A diff verdict is not causality or a performance proof

`diff` answers whether the captured before/after evidence is comparable and what changed in the
observed metrics. It does not prove that a source-code change caused the difference, and it cannot
measure what was not recorded in either profile.

Before trusting a verdict, check:

- `comparability_confidence` and the reasons for any mismatch;
- whether both sides represent the same workload, warmup state, GPU set, and iteration;
- whether the comparison is whole-profile or a deliberate `--trim` / `--iteration` window;
- whether the reported metric is a wall-clock step, GPU kernel time, communication time, or a
  derived category;
- whether the result is `inconclusive` rather than `improvement_likely`.

An empty or incomparable after profile is not evidence of a speedup. For CI, use
`--exit-on-regression` or `--gate` and keep the JSON envelope as an artifact:

```console
$ nsys-ai diff BEFORE.sqlite AFTER.sqlite --format json --no-ai \
    --exit-on-regression -o diff.json
```

For the interpretation rules and recovery paths, use [Reading a diff](./reading-a-diff.md), not the
colour of a single row in the Web view.

## 7. The tool cannot answer questions outside the trace

Some important performance questions require data that Nsight Systems does not contain by default:

| Question | What nsys-ai can say | What it cannot prove |
|---|---|---|
| Is power or energy the bottleneck? | Correlate observed activity and idle intervals | Board power, energy, thermal throttling, or clock residency without those telemetry sources |
| Which source line caused a kernel hotspot? | Suggest code/config candidates when tracing metadata and CUTracer are available | A source-line attribution from kernel names alone |
| Did the code change cause the regression? | Compare captured evidence and record a decision | Causality; environment, input, and run-to-run variance remain external facts |
| Why is an iteration slow? | Locate recorded CPU/GPU/NVTX activity and identify gaps | Work that was never traced, hidden host scheduling, or an uninstrumented service outside the capture |
| Is an MFU number correct? | Compute against a supplied/detected hardware ceiling when inputs are valid | A trustworthy ceiling when GPU model, world size, FLOP count, or units are missing |

The correct response to missing evidence is to say what is unavailable, not to fill the field with a
zero. Keep `doctor.json`, `findings.json`, and the relevant profile metadata with a report so a later
reader can tell an unmeasured value from a measured zero.

## Recently fixed boundaries in 0.3.0

These were real failure modes during the release work, but they are not current limits on the
released mainline. They are listed so old issue comments and local results are not mistaken for
current behavior:

- CLI `--trim` now rejects a window outside the capture instead of handing an empty result to every
  skill.
- `root_cause_matcher` abstains on an empty window and on an ambiguous multi-GPU synchronization
  denominator; it no longer publishes those cases as a normal warning or a healthy profile.
- Web starts listening before NVTX tree warmup and reports HTTP `202` while the tree is building.
- `/api/tree` returns bounded slices with truncation metadata instead of an unbounded fan-out.
- Direct SQLite-compatible kernel views preserve Tensor Core fields; unknown values are still not
  treated as false.

If one of these symptoms reappears on current 0.3.0, attach `nsys-ai doctor PROFILE --format json`,
the command line, the input kind, and the profile's schema version to a new issue. Do not work around
it by deleting evidence or treating an empty result as healthy.

## Verification record

The page was checked against the released source and these concrete artifacts:

| Boundary | Verification | Result |
|---|---|---|
| Capture clock | `info` and an out-of-range `skill run` on `mfu_2gpu_before.sqlite` | profile range `60.112s–60.411s`; CLI returns `TRIM_OUT_OF_RANGE` |
| Direct skill contract | `run_skill(raw=True)` with `trim_start_ns=0`, `trim_end_ns=1e9` | `gpu_idle_gaps` emits its legacy-looking summary; `root_cause_matcher` emits `_abstained` |
| Device fallback | `root_cause_matcher` with `device=0` on a window containing only device 1 activity | result carries `analysed_device=1`, so requested and analysed device must not be conflated |
| Multi-GPU denominator | `root_cause_matcher` on `perf.parquetdir` | 271.6% synchronization ratio is preserved as an abstention with its wall-span denominator |
| Input size | `ls`, `du`, and Parquet file count on the release E2E profile | 605 MB `.nsys-rep`, 717 MB `.parquetdir`, 48 Parquet files |
| Tree bound | live release verification recorded in [#508](https://github.com/GindaChen/nsys-ai/pull/508) | bounded response, `truncated=true`, `children_total` disclosed |
| Basic health | `doctor --format json` on the committed two-GPU fixture | health report includes sections and explicit warnings/skips |

The verification record is evidence for the documented boundary, not a claim that every profile has
the same size, schema, device layout, or runtime. Re-run it with your own profile before treating a
number as a workload-specific fact.

## Related pages and retirement issues

- [Profile inputs](./profile-inputs.md) — ingest and cache selection
- [Time windows](./time-windows.md) — capture-clock and iteration semantics
- [Environment variables](./environment-variables.md) — cache, DuckDB, and baseline controls
- [Troubleshooting](./troubleshooting.md) — symptom-first recovery
- [Reading a diff](./reading-a-diff.md) — verdict and comparability semantics
- [#493](https://github.com/GindaChen/nsys-ai/issues/493) — root-cause trim correctness, fixed in 0.3.0
- [#490](https://github.com/GindaChen/nsys-ai/issues/490) — Web bind-before-warmup, fixed in 0.3.0
- [#506](https://github.com/GindaChen/nsys-ai/issues/506) — bounded NVTX tree slices, fixed in 0.3.0
