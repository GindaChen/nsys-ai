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

Every command accepts all three. A few need more than a path; see
[which commands need extra flags](./troubleshooting.md#some-commands-need-more-than-a-path).

## The first run is slower

Converting is the one expensive step, and it happens once:

```
$ nsys-ai diagnose perf.nsys-rep
[nsys-ai] Building analysis cache for a 634MB profile (~6s, once per profile) ...
```

The conversion writes `perf.parquetdir` next to `perf.nsys-rep`. Every later command on the same
capture reuses it and starts immediately. Delete the directory and the next command rebuilds it.

Conversion runs `nsys export`, so Nsight Systems has to be on your `PATH`. If it is not, the error says
so and prints the command to run by hand.

## What causes a rebuild

Two things, and neither is content-based:

- **The capture is newer than the directory beside it.** Overwrite `perf.nsys-rep` and the stale
  `perf.parquetdir` is ignored. Modification time is the test, so copying a capture around can trigger
  a rebuild even when the contents did not change.
- **The directory is not a complete export.** A conversion interrupted halfway leaves something that
  fails inspection; it is replaced rather than half-read.

## Choosing the storage yourself

You will not normally need this. When you do, `NSYS_AI_INGEST` overrides the default everywhere:

| Value | Effect |
|---|---|
| `auto` | The default described above |
| `parquetdir` | Only read a parquetdir; refuse a `.sqlite` input |
| `sqlite` | Convert a `.nsys-rep` to `.sqlite` instead, and refuse a parquetdir input |

`NSYS_AI_INGEST=sqlite` is the useful one. It reproduces the older behaviour, which is worth having
when comparing against a result produced before parquetdir became the default, or when you want a
`.sqlite` you can open in another tool. It also disables the query cache for the whole run, so do not
leave it set — see [Environment variables](./environment-variables.md#notes-on-a-few-of-these).

Asking for something impossible fails immediately and says why:

```
$ NSYS_AI_INGEST=sqlite nsys-ai diagnose perf.parquetdir
Error [EXPORT_ERROR]: SQLite ingest policy cannot open a parquetdir; use NSYS_AI_INGEST=parquetdir.
```

## Where to go next

- [Time windows](./time-windows.md) — how `--trim` is measured
- [Troubleshooting](./troubleshooting.md) — a capture that will not open, or opens and produces nothing
- [Environment variables](./environment-variables.md) — the full list
- [Ingest policy](../dev/ingest-policy.md) — the same rules from the maintainer's side
