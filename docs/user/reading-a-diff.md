# Reading a diff

A diff is a measured comparison between two profile captures. It answers a
narrow question:

> Does the measured step-time change clear the comparability and noise
> thresholds strongly enough to call it a likely regression or improvement?

It does not prove causality, statistical significance, or that every kernel
moved in the same direction. Read the top-level verdict first, then use the
component axes and trace selections to decide what to investigate.

This page explains the JSON and terminal report from the current diff command.
For CI exit codes and artifact upload, see [the CI diff gate](../ci-diff-gate.md).
For choosing an iteration or a capture-clock window, see [time
windows](./time-windows.md).

## Start with a comparable pair

The two profiles should represent the same workload, capture scope, GPU set,
and logical iteration. A good baseline/candidate command is:

~~~bash
nsys-ai diff before/profile.sqlite after/profile.sqlite \
  --format json --no-ai --output artifacts/diff.json
~~~

The same command accepts .nsys-rep and .parquetdir inputs. A .nsys-rep
is converted through the normal ingest policy; the resulting cache does not
change the meaning of the comparison.

For two captures of one workload, prefer an aligned iteration:

~~~bash
nsys-ai diff tests/fixtures/mfu_2gpu_before.sqlite \
  tests/fixtures/mfu_2gpu_after.sqlite \
  --iteration 0 --format json --no-ai
~~~

--iteration locates the selected iteration separately on each side. The
captures do not need to share an absolute clock. Use --trim START END only
when a shared capture-clock window is the thing you intend to compare. See
[Selecting an iteration instead of seconds](./time-windows.md#selecting-an-iteration-instead-of-seconds).

## A real result

The committed 0.3.0 fixtures produce this deterministic terminal excerpt:

~~~text
$ nsys-ai diff tests/fixtures/mfu_2gpu_before.sqlite \
    tests/fixtures/mfu_2gpu_after.sqlite --format terminal --no-ai

Profile Diff (All GPUs)
Verdict: neutral  (comparability 0.89)
Total GPU: 535.20ms → 488.84ms  (Δ -46.37ms)

Communication/NCCL Summary
Total (exposed comm): 25.400ms → 33.740ms  (Δ +8.340ms)

Top regressions (kernels)
   +8.34ms  | 66->62 | ncclDevKernel_SendRecv

Top improvements (kernels)
  -31.56ms  | 68->60 | flash_fwd_kernel
~~~

The full terminal report also contains category attribution, idle-gap
entries, NVTX range changes, and per-GPU sections. The important reading is
that the pair's GPU work changed substantially while its measured step time
changed by only 0.01%. The verdict is therefore neutral. A faster sum of GPU
kernel time is not the same claim as a faster end-to-end step.

The JSON form carries the step-time field used by the verdict:

~~~json
"step_time": {
  "before_ms": 598.52,
  "after_ms": 598.46,
  "delta_ms": -0.06,
  "delta_pct": -0.01
}
~~~

The same pair compared by logical iteration 0 produces:

~~~text
verdict: improvement_likely
comparability_confidence: 0.894
step_time: 482.87ms → 340.77ms
delta: -142.10ms (-29.43%)
~~~

That is a different question and a different answer: the iteration windows
are aligned by marker rather than by the whole profile span.

## Verdicts

The verdict is computed from step-time delta, comparability, and the configured
regression threshold. The default threshold is 5 percent.

| Verdict | Meaning | Does not mean |
| --- | --- | --- |
| regression_likely | Step time increased by at least the threshold and comparability is sufficient. | Every kernel regressed, or a root cause was proven. |
| improvement_likely | Step time decreased by at least the threshold and comparability is sufficient. | The change is causal, permanent, or safe to ship without checking the new bottleneck. |
| neutral | The pair is comparable, but the measured change is below the decision boundary or is the same-capture/noise case. | Every component is unchanged. A neutral result can contain large offsetting improvements and regressions. |
| inconclusive | The pair does not provide enough evidence for a before/after judgement, or step time could not be computed. | The workload is healthy or unchanged. |

There is also a 2 percent single-run noise floor. With one capture per side,
the tool will not call a change inside that floor a likely regression or
improvement, even if a caller asks for a tighter threshold. Re-run the pair
when a small change matters.

A self-diff is neutral with a warning. It is a useful check of the pipeline,
not evidence that a change had no effect.

### neutral is not a health verdict

In the fixture example, compute fell by 54.70 ms, exposed communication rose
by 8.34 ms, launch overhead rose by 4.95 ms, and idle rose by 41.35 ms. These
components nearly cancel in step time. The neutral verdict says the aggregate
step-time decision boundary was not crossed; it does not say the new profile
is healthy.

Inspect category_attribution, top_regressions, and top_improvements
before accepting a neutral result.

## Comparability confidence

comparability_confidence is a score from 0 to 1. The verdict requires at
least 0.50. The score is a product of independent sanity factors, so one
zero-valued factor can make the comparison inconclusive.

The main factors are:

- export schema compatibility;
- GPU count and selected device compatibility;
- whether either side has no GPU kernel activity;
- the ratio of kernel row counts;
- overlap between kernel identities;
- whether overlap analysis is available.

A different Nsight product build is warned about but remains comparable when
the export schema is the same. A different export schema, missing GPU, missing
activity, or almost entirely different workload can reduce confidence to zero.

### Low-confidence example

Comparing the small mock fixture with the two-GPU fixture produces a refusal:

~~~text
Verdict: inconclusive  (comparability 0.00)

No comparison was made
Comparability scored 0.00, under the 0.50 minimum — these two captures
cannot be read as a before and an after.

Warnings:
- GPU count differs: one side has one device and the other has two.
- Nsight export schema differs: 3.24.14 versus 3.25.0.
- Kernel row counts differ a lot: 2 versus 4010.
~~~

The JSON still contains profile metadata and diagnostic warnings, but the
terminal renderer withholds per-kernel and per-range deltas. Do not replace
this result with a prose claim such as "the candidate is faster."

### Fixes for common causes

| Cause | What to do |
| --- | --- |
| Different absolute clocks | Use --iteration N, not the same seconds on both profiles. |
| Warm-up on one side | Capture the same logical iterations and select the same iteration. |
| Different iteration count | Record the same number of representative steps. |
| Different GPU set | Capture and compare the same devices, or pass --gpu N for a deliberate single-device comparison. |
| Different Nsight export schema | Re-export or recapture with a compatible Nsight version, then inspect the support matrix. |
| One side has almost no kernels | Treat it as a failed or incomplete capture, not as an improvement. |
| Different workload or model shape | Reproduce the same inputs, batch, sequence length, and configuration before comparing. |

The warnings array is the first place to look. It explains the observed
reason; comparability_confidence tells you whether the reason is strong
enough to withhold a verdict.

## Step time versus GPU time

The two numbers are intentionally different:

- step time is the comparison's wall-clock category total. It is the
  denominator used for the verdict;
- total GPU time is the sum of recorded GPU kernel activity. Kernels on
  different streams or devices can overlap, so it is not a wall-clock duration;
- category attribution decomposes the step-time total into compute,
  communication, launch overhead, and idle.

The default comparison aggregates all GPUs. Use --gpu 0 when the question is
about one device. A per-GPU section can show a different story from the
node-wide verdict.

The useful invariant is that category attribution reconstructs the step-time
basis:

~~~text
compute + communication + launch_overhead + idle = step time
~~~

This is a time accounting convention, not a proof that a category caused the
step-time movement.

## Read the component axes

### Compute, communication, and overlap

The overlap object has these measurements for each side and a delta:

| Field | Interpretation |
| --- | --- |
| compute_only_ms | GPU compute time not overlapping exposed NCCL communication. |
| nccl_only_ms | Exposed communication time not overlapped with compute. |
| overlap_ms | Time where compute and NCCL overlap. The HTA convention counts this as compute in category attribution. |
| idle_ms | GPU-visible idle span used by the overlap accounting. |
| launch_overhead_ms | Exposed dispatch latency carved out of idle. |
| total_ms | The overlap analysis wall-clock total. |
| overlap_pct | Overlap as a percentage of communication/overlap time, not a percentage of all GPU time. |

Do not add compute_only_ms, nccl_only_ms, and overlap_ms as though they
were three disjoint wall-clock intervals. overlap_ms is deliberately shared
by the underlying activities. Use category_attribution for the step-time
accounting.

The communication axis calls its total basis exposed comm. In prose this is
often called exposed_comm_ms; the JSON represents it as
communication_summary.before_ms, after_ms, and delta_ms, with
total_basis: "exposed comm". There is no communication_ms field. Do not
invent one or substitute total NCCL activity for exposed communication.

### Idle

idle_summary lists changed gaps and points to a trace selection with GPU,
stream, start, and end timestamps. Its total_basis is wall-clock idle.
A positive item is a larger gap in the after profile; a negative item is a
gap that shrank or disappeared.

Idle entries are evidence to inspect, not root-cause proof. A gap can be
caused by host scheduling, synchronization, a changed kernel sequence, or
capture boundaries. Open the selection in the viewer and correlate it with
the host/runtime activity before proposing a fix.

### Category attribution

Each item has:

- category;
- before_ms and after_ms;
- delta_ms, where positive means more time after the change;
- delta_pct, relative to the before value, or null when the before value is
  zero.

Use the category row for the verdict's denominator. Use the axis summary and
trace selections for investigation.

## Kernels and NVTX ranges

### Kernel rows

top_regressions and top_improvements are ranked by total kernel-time
delta, not by step-time delta. A row includes:

- key: the stable kernel identity used for matching;
- name: the short display name;
- demangled: the longer symbol used when available;
- before_total_ns and after_total_ns;
- delta_ns, where positive is more GPU time after;
- before_count and after_count;
- per-side shares and classification;
- selection, when the tool can locate an after or before time span;
- diff_lineage, including baseline profile id, diff id, rank, and role.

The short name is not always unique. Use key when joining rows or opening
a selection; several template instantiations may share a display name.

A positive kernel delta can come from more calls, slower calls, or both. The
diff reports counts and totals so you can distinguish those hypotheses. It
does not replace a source-level or kernel-configuration investigation.

### NVTX rows

nvtx_regressions and nvtx_improvements compare annotated range text,
counts, and total range wall time. They are a useful way to see which logical
regions appeared, disappeared, or changed, but NVTX range wall time can
include nested work and is not interchangeable with summed GPU kernel time.

### Trace selections

A selection is the handoff from a summary row to a viewer or later analysis.
It identifies:

- the profile id and side;
- GPU ids;
- a start and end timestamp in nanoseconds;
- a stable selection id;
- the source axis.

A selection is evidence location, not a claim that the selected span is the
cause. Use it to correlate the row with NVTX, CUDA runtime, and host activity.

## What the JSON envelope means

The JSON report has a stable top-level envelope. The following fields are
present in a normal report; optional axis values can be null when the relevant
analysis is unavailable.

| Field | Meaning |
| --- | --- |
| schema_version | Diff payload schema version. |
| producer | Component that produced the report. |
| producer_version | nsys-ai version that produced it. |
| diff_id | Content-derived identity of the before/after pair and comparison parameters. |
| verdict | One of the four verdict strings above. |
| comparability_confidence | Unrounded confidence truncated at JSON serialization; the gate uses the unrounded value. |
| warnings | Sanity warnings and reasons not to over-read the result. |
| before | Path, profile id, GPU selection, Nsight schema/product version, and total GPU nanoseconds for the baseline. |
| after | The same metadata for the candidate. |
| step_time | Before/after wall-clock category total, delta milliseconds, and delta percentage. |
| category_attribution | Compute, communication, launch-overhead, and idle rows used to reconstruct step time. |
| overlap | Before/after/delta overlap accounting. |
| communication_summary | Exposed-communication axis, per-entry changes, and trace selections. |
| idle_summary | Wall-clock idle axis, per-entry changes, and trace selections. |
| top_regressions | Ranked kernel rows with positive total GPU-time deltas. |
| top_improvements | Ranked kernel rows with negative total GPU-time deltas. |
| nvtx_regressions | New or slower NVTX ranges. |
| nvtx_improvements | Removed or faster NVTX ranges. |
| decision | Null unless --accept or --reject records an explicit human decision; it is not the computed verdict. |

The overlap, communication_summary, and idle_summary fields use different
time bases. Always read their total_basis before comparing their totals.

## When the verdict is inconclusive

Use this sequence:

1. read every entry in warnings;
2. check before and after for profile id, schema, GPU scope, and product version;
3. run nsys-ai info on both profiles;
4. compare the same logical iteration with --iteration;
5. if the comparison is still not valid, recapture rather than narrate the delta.

For a normal pair, save the JSON report and attach it to the review or CI
artifact:

~~~bash
nsys-ai diff baseline.sqlite candidate.sqlite \
  --format json --no-ai --output artifacts/diff.json
~~~

For a gate, use the documented wrapper or add --gate PCT and
--exit-on-regression. A gate failure is not a reason to hide the artifact.
Upload diff.json, inspect the warnings, and record an explicit decision if a
known regression is intentional.

## What not to conclude

Do not conclude any of the following from a diff alone:

- improvement_likely means the optimization caused the improvement;
- a top kernel regression is the root cause;
- a larger idle axis identifies which host operation blocked the GPU;
- a neutral verdict means all components are healthy;
- a low-confidence result means the candidate is faster or slower;
- a large percentage on a tiny before value is a large end-to-end impact;
- overlap_ms can be added to all other component totals without accounting
  for shared time.

A diff is a reproducible measurement and an investigation handoff. The next
step is to open the largest changed selection, check the corresponding
category, and verify the proposed change with another aligned capture.
