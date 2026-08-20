# Time windows

Most analysis is about part of a capture, not all of it. A profile that ran for four minutes
contains warm-up, a few hundred steady-state iterations, and shutdown, and averaging them together
hides the thing you are looking for.

There are two ways to narrow the window. Pick the one that matches what you know:

| You know | Use | Available on |
|---|---|---|
| The seconds you care about | `--trim START_S END_S` | 31 commands — nearly all of them |
| Which training iteration you care about | `--iteration N` | `diff` and `skill run` |

The two flags interact differently on each of the commands that take both, so see
[combining the two](#combining-the-two) before passing them together.

## The capture clock does not start at zero

This is the one thing that surprises people, and it costs an afternoon when it goes unnoticed.

`--trim` is read on the **capture clock**: the timestamps Nsight Systems recorded, which are relative
to session start, not to the first event. A profile whose first kernel lands 155 seconds in has a
window that begins at 155 seconds. `--trim 0 1` on that profile does not mean "the first second of
GPU work" — it means a window that ends 154 seconds before anything happened.

Ask the profile for its window before trimming:

```
$ nsys-ai info perf.sqlite
Profile: perf.sqlite
  Nsight version (heuristic): 2026.2.1.210
  GPUs: [0, 1]
  Kernels: 3442  |  NVTX: 16782
  Time: 155.360s - 156.321s
```

So the trim window for the middle third of that capture is:

```bash
nsys-ai skill run top_kernels perf.sqlite --trim 155.68 156.00
```

Units are seconds, accepted as floats, and converted to nanoseconds internally. Both bounds are
required; there is no open-ended form.

## An impossible window is usually an error, not an empty result

A window that selects nothing is almost always a mistake about the clock rather than a real request
for zero rows, so most commands refuse it up front and name both windows:

```
$ nsys-ai diff before.sqlite after.sqlite --trim 0 1
Error [TRIM_OUT_OF_RANGE]: before profile: --trim 0.000 1.000 selects no part of this
profile, whose window is 60.112 s to 60.411 s on the capture clock. Use a window inside
that range, or omit --trim.
```

Three shapes are rejected: a window entirely before the capture, one entirely after it, and an empty
or inverted one such as `--trim 156 156`, which sits inside the capture and still selects nothing. On
a two-profile command the message says which side failed, because `--trim` applies to both and two
captures rarely share a clock — the example above is a real pair whose windows are 60.1 s and 72.9 s.

A window that *partly* overlaps is accepted and clipped. That is a legitimate request.

### Commands that do not check

Six commands skip this check. Given a window that selects nothing they produce an empty result
rather than an error:

```
$ nsys-ai skill run top_kernels perf.sqlite --trim 0 1
(No kernels found)
```

| Command | What you get instead of an error |
|---|---|
| `skill run` | `(No kernels found)` |
| `agent analyze` | A report header, then `(No kernels found)` |
| `cutracer plan`, `cutracer analyze`, `cutracer run` | `(No kernels found — nothing to instrument)` |
| `diff-web` | A served page with an empty comparison on it |

That output is indistinguishable from a capture that genuinely has no kernels, which is the whole
reason the check exists elsewhere. `diff-web` is the most confusing of the six, because the empty
result is a rendered page rather than a line of text.

Until this is unified, treat an unexpected empty result from any of the six as a trim-window question
first: re-run the same window through `diagnose`, which does check, and it will tell you whether the
window or the capture is the problem.

## Selecting an iteration instead of seconds

When the interesting unit is "one training step" rather than "these seconds", let the tool find the
boundaries:

```bash
nsys-ai skill run top_kernels perf.sqlite --iteration 0
```

Iteration boundaries come from NVTX markers when the capture has them, and fall back to a kernel-gap
heuristic when it does not. The marker name defaults to `sample_0`; `--marker` changes it.

Prefer this over hand-computed seconds when it applies. It survives re-capturing the same workload,
where the absolute clock does not — and on `diff` it is the only way to compare like with like, since
two captures of the same workload do not share a clock.

```bash
nsys-ai diff before.sqlite after.sqlite --iteration 0
```

Each side gets its own window here: iteration 0 is located separately in each profile, so the
comparison is between the same logical step rather than the same wall-clock seconds.

If the index does not exist, you are told the range that does:

```
Error: iteration 99 out of range (0..2)
```

### Combining the two

The two commands that accept both flags treat the combination differently. This is worth knowing
before you assume either behaviour:

| Command | `--trim` together with `--iteration` |
|---|---|
| `skill run` | Refused — `Error: --iteration and --trim cannot be used together` |
| `diff` | Composed — `--trim` limits the span iteration boundaries are searched for, then the chosen iteration's own bounds become the window |

So on `diff` the pair is a narrowing search ("find iteration 0 within this span"), while on
`skill run` it is an ambiguous request and is rejected.

## Which commands accept `--trim`

Effectively all of them: 31 commands take it, so assume it is available rather than checking.

Seven commands read a profile and do *not*, each for a reason worth knowing:

| Command | Why there is no window |
|---|---|
| `info`, `doctor` | They describe the capture as a whole — a window would defeat the purpose |
| `warm` | It builds the cache for the whole capture, so later windowed queries are fast |
| `baseline tag` | A baseline snapshots the entire profile; trim at comparison time instead |
| `ask`, `chat`, `agent ask` | Conversational, and the window is something you say in the question |

For `ask` and `chat` the window goes in the question itself — "in the second half of the run",
"during iteration 3". Understand what that is and is not: `--trim` filters the data before any
analysis runs, whereas a window described in words is a request the model may or may not apply to
every query it issues. When the number has to be right, get it from a command that takes `--trim`.

## Related

- [What to hand nsys-ai](./profile-inputs.md) — the inputs each command accepts
- [Troubleshooting](./troubleshooting.md) — an empty result that is not about trimming
- [Environment variables](./environment-variables.md) — `NSYS_AI_MANIFEST_AUTO_TRIM` and the rest
