# What to hand nsys-ai

Give it the `.nsys-rep` your capture produced. Everything else is the tool's problem.

```bash
nsys-ai doctor perf.nsys-rep
nsys-ai diagnose perf.nsys-rep
nsys-ai timeline-web perf.nsys-rep
```

You do not need to export anything first, and you do not need to know which storage format a
particular command prefers.

## The three inputs

| You have | What happens |
|---|---|
| `perf.nsys-rep` | Converted once to `perf.parquetdir` beside it, then read from there |
| `perf.parquetdir` | Read directly |
| `perf.sqlite` | Read directly, with a query cache built beside it |

A `.nsys-rep` is the native Nsight Systems report and what `nsys` writes by default. A `parquetdir` is
a directory of Parquet files that `nsys export` produces; it is what nsys-ai converts to, because
analysis queries run against it directly. A `.sqlite` is the older export format — still fully
supported, and the right input when it is what you already have.

## The first run on a `.nsys-rep` is slower

Converting is the one expensive step, and it happens once:

```
$ nsys-ai diagnose perf.nsys-rep
[nsys-ai] Building analysis cache for a 634MB profile (~6s, once per profile) ...
```

The conversion writes `perf.parquetdir` next to `perf.nsys-rep`. Every later command on the same
capture reuses it and starts immediately. Delete the directory and the next command rebuilds it.

Conversion runs `nsys export`, so Nsight Systems has to be on your `PATH`. If it is not, the error says
so and prints the command to run by hand.

Two things cause a rebuild rather than a reuse:

- **The capture is newer than the directory beside it.** Overwrite `perf.nsys-rep` and the stale
  `perf.parquetdir` is ignored. Modification time is the test, so copying a capture around can trigger
  a rebuild even when the contents did not change.
- **The directory is not a complete export.** A conversion interrupted halfway leaves something that
  fails inspection; it is replaced rather than half-read.

## Your Nsight version has to be new enough

A `.nsys-rep` written by a newer Nsight Systems than the one on your machine cannot be converted:

```
Exportation error: Report was created in Nsight Systems version (2026.3.1),
newer than your current version (2026.2.1).
```

That is `nsys export` refusing, not nsys-ai. Upgrading Nsight Systems fixes it. Older captures are
fine — the export schema is versioned, and the versions covered by tests are listed in the
[support matrix](../support-matrix.md).

If you are unsure, `nsys-ai doctor <profile>` reports the schema version it found and whether the
capture is readable before you spend time on anything else.

## Which commands take which input

All of them accept any of the three. Most need nothing but the path:

```bash
nsys-ai doctor perf.nsys-rep
nsys-ai diagnose perf.nsys-rep
nsys-ai ask perf.nsys-rep "why is the GPU idle?"
nsys-ai summary perf.nsys-rep
nsys-ai skill run top_kernels perf.nsys-rep
nsys-ai web perf.nsys-rep
nsys-ai timeline-web perf.nsys-rep
nsys-ai evidence build perf.nsys-rep
nsys-ai diff before.nsys-rep after.nsys-rep
nsys-ai review before.nsys-rep after.nsys-rep
```

Two still ask for more than a path:

- `nsys-ai report` requires `--gpu` and `--trim`.
- `nsys-ai analyze` requires `--trim` for its text output; `--format json` runs on the whole capture.

## Choosing the storage yourself

You will not normally need this. When you do, `NSYS_AI_INGEST` overrides the default everywhere:

| Value | Effect |
|---|---|
| `auto` | The default described above |
| `parquetdir` | Only read a parquetdir; refuse a `.sqlite` input |
| `sqlite` | Convert a `.nsys-rep` to `.sqlite` instead, and refuse a parquetdir input |

`NSYS_AI_INGEST=sqlite` is the useful one. It reproduces the older behaviour, which is worth having
when comparing against a result produced before parquetdir became the default, or when you want a
`.sqlite` you can open in another tool.

Asking for something impossible fails immediately and says why:

```
$ NSYS_AI_INGEST=sqlite nsys-ai diagnose perf.parquetdir
Error [EXPORT_ERROR]: SQLite ingest policy cannot open a parquetdir; use NSYS_AI_INGEST=parquetdir.
```

## A trim window is on the capture clock

`--trim START END` is in seconds measured on the capture's own clock, which does not start at zero. A
profile whose first GPU work is at 34.8 s has nothing in `--trim 0 5`, and the commands say so rather
than returning an empty result:

```
Error [TRIM_OUT_OF_RANGE]: --trim 0.000 5.000 selects no part of this profile, whose
window is 34.808 s to 812.508 s on the capture clock. Use a window inside that range,
or omit --trim.
```

`nsys-ai doctor` prints the capture's duration, and the error above prints the real window, so you do
not have to guess twice.

## When something looks wrong

Run `nsys-ai doctor <profile>` first. It checks that the capture opens, and reports its export schema
version, GPU count, NVTX and NCCL event counts, and profiler overhead — marking which of those is a
problem rather than leaving you to compare numbers.

A capture that opens but produces very little is usually a device-selection question rather than an
ingest one. `nsys-ai diagnose` analyses the first GPU present in the capture; if you name one that is
not there, it lists the ones that are:

```
$ nsys-ai diagnose perf.nsys-rep --gpu 0
Error [USAGE_ERROR]: GPU device 0 is not present in the profile; available devices: 1, 2, 3, 4, 5, 6, 7
```
