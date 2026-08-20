# Troubleshooting

Organised by what you saw, not by which subsystem produced it. Each entry gives the message or
behaviour, what it actually means, and what to change.

Two general notes before the list. Errors carry a bracketed code — `Error [TRIM_OUT_OF_RANGE]: …` —
which is stable across releases and worth searching for. And `nsys-ai doctor <profile>` answers most
"will this capture work" questions in one command; run it before working through anything below.

## The profile will not open

### `Error [EXPORT_TOOL_MISSING]` — conversion requires 'nsys' on PATH

```
Error [EXPORT_TOOL_MISSING]: Profile is .nsys-rep; conversion requires 'nsys'
(NVIDIA Nsight Systems) on PATH. Install Nsight Systems or export manually:
nsys export --type parquetdir --include-blobs=true -o <out.parquetdir>
--force-overwrite=true <file.nsys-rep>
```

A `.nsys-rep` has to be converted before it can be queried, and the converter is Nsight Systems
itself. Either install it — `apt-get install nsight-systems-cli`, or the CUDA toolkit installer —
or run the printed command on a machine that has it and copy the result across.

The printed command is the exact one that would have run, including the flags. Do not simplify it:
`--include-blobs=true` is what preserves NVTX payloads, and without it NVTX-based analysis comes back
empty later, with no error at the point of failure.

### `file is not a database`

A file that is not a SQLite database was handed to something that expected one — most often a
`.nsys-rep` passed to a tool that does not convert. Every nsys-ai command converts for you, so if
this comes from nsys-ai on a current release it is a bug worth reporting with the command line.

### `SQLite ingest policy cannot open a parquetdir`

```
Error [EXPORT_ERROR]: SQLite ingest policy cannot open a parquetdir;
use NSYS_AI_INGEST=parquetdir.
```

`NSYS_AI_INGEST` is set to `sqlite` — possibly in a shell profile or CI job you have forgotten
about — and the input is a parquetdir. Unset it, or set it to `auto`.

The symmetric case has its own message:

```
Error [EXPORT_ERROR]: Parquetdir ingest requires a parquetdir directory or a
.nsys-rep input.
```

Here `NSYS_AI_INGEST=parquetdir` met a `.sqlite`. There is nothing to convert from — a `.sqlite`
export cannot be turned back into a parquetdir — so the fix is `auto`, which reads it in place. See
[Environment variables](./environment-variables.md).

## The command runs but produces nothing

### An empty result after `--trim`

Almost always the capture clock. `--trim` is measured on Nsight's clock, which does not start at
zero, so a window that looks reasonable can sit entirely before the first event. Run `nsys-ai info`
to see the real window. [Time windows](./time-windows.md) covers this in full.

Six commands do not perform this check and return an empty result instead of an error: `skill run`,
`agent analyze`, `diff-web`, and the three `cutracer` sub-commands. If one of them comes back empty,
re-run the same window through `diagnose`, which does check. See
[Time windows](./time-windows.md#commands-that-do-not-check).

### `This profile has no CUPTI_ACTIVITY_KIND_KERNEL table`

```
This profile has no CUPTI_ACTIVITY_KIND_KERNEL table, so 'top_kernels' cannot run.
Either the capture did not trace that activity kind, or the export names it something
this version does not recognise as a variant.
```

Two genuinely different causes, and the message says so because it cannot tell them apart:

- **The capture did not trace CUDA.** Check the `nsys profile` command that produced it — without
  `--trace=cuda` (or a default that includes it) there are no kernel rows to find. Confirm with
  `nsys-ai info`, which reports a kernel count of 0.
- **The export uses a table name this version does not know.** Newer Nsight releases version their
  table names. `nsys-ai doctor` reports the export schema version; compare it against
  [the support matrix](../support-matrix.md).

A third cause used to exist — the profile was read from the wrong storage, so a parquetdir holding
millions of kernel rows sat unread beside the capture. That path is fixed, but if you see this
message alongside a populated `.parquetdir`, report it; the symptom describes the capture and the
cause is elsewhere.

### The results describe a GPU you did not ask for

`root-cause` substitutes a different device when the one you selected has no kernels on it. Multi-GPU
captures often leave device 0 idle while devices 1-7 do the work, and returning nothing there would
be less useful than answering about a device that ran something.

The substitution is silent — nothing in the output says it happened. If a per-device number looks
wrong, check the per-GPU kernel counts in `nsys-ai info` and pass `--gpu` explicitly.

## The command is slow

### The first command on a capture takes seconds before printing anything

```
[nsys-ai] Building analysis cache for a 634MB profile (~6s, once per profile) ...
```

Expected. The capture is being converted once into a form queries can run against, and the result is
written beside it. Every later command on the same capture starts immediately. See
[What to hand nsys-ai](./profile-inputs.md).

If it happens *every* time, the conversion output is not being kept: check that the directory holding
the capture is writable, and that the capture's modification time is not being updated by whatever
copies it into place.

### `Not building an analysis cache … queries will be several times slower`

```
Not building an analysis cache (<reason>); querying the SQLite export directly.
Queries will be several times slower.
```

Free disk or profile size failed the affordability check, so the query cache was skipped. The run is
correct, just slow. Free space, move the profile to a larger filesystem, or force the cache with
`NSYS_AI_CACHE_MODE=parquet` if you know the space is there.

This is a warning rather than an error precisely so it is visible — a silently degraded run is worse
than a slow one you know about.

### Queries spill or run out of memory on a large capture

Set `NSYS_AI_DUCKDB_TEMP_DIRECTORY` to a filesystem with room, and lower `NSYS_AI_DUCKDB_THREADS` if
the machine is shared. Both are in [Environment variables](./environment-variables.md).

## Some commands need more than a path

Fourteen commands will not run on a profile path alone. This is by design, but the argparse message
is terse, so here is what each one wants:

| Command | Also needs |
|---|---|
| `diff`, `diff-web`, `review` | A second profile — `<before> <after>` (`diff` also accepts `--against`; `review` also accepts `--session`) |
| `export` | `--trim` |
| `report` | `--gpu` and `--trim` |
| `optimize` | `--repo` |
| `propose` | `--finding-id` |
| `ask` | A question, as well as the profile |
| `agent`, `baseline`, `cutracer`, `evidence`, `root-cause`, `skill` | A sub-command first — these are command groups, so it is `skill run <name> <profile>`, not `skill <profile>` |

`chat` additionally requires a real terminal, and refuses with `Error [NOT_A_TERMINAL]` when stdin is
a pipe. Run it interactively rather than from a script.

## Getting more detail

`nsys-ai doctor <profile>` is the first thing to run and the right thing to attach to a bug report.
It checks the environment and the capture together, and every failed check carries the command that
fixes it:

```
$ nsys-ai doctor perf.sqlite
Profile: perf.sqlite
  id: nsys2:sha256:51a2d0fc015ee251f4bb4272ef65586f518d1bf0403761a0f37c3e8a462e0b0c

Profile support
  SQLite analysis                [OK  ]  stdlib sqlite3
  .nsys-rep conversion           [OK  ]  /usr/local/bin/nsys
  Parquet cache                  [OK  ]  duckdb + pyarrow

Profile health
  Schema compatibility           [OK  ]  export schema 3.25.0
  Duration                       [OK  ]  1.0s
  GPUs                           [OK  ]  2
  GPU model                      [WARN]  unknown
                                    -> GPU model missing from CUPTI TARGET_INFO;
                                       MFU / efficiency cannot be computed.
```

Four sections: **System** (Python, nsys-ai, platform), **Profile support** (whether each backend is
usable at all), **Optional features** (AI providers, CUTracer), and **Profile health** (the capture's
own schema version, duration, and GPUs).

`[WARN]` is worth reading rather than skipping. The example above explains why an MFU number will be
missing later, at the point where the cause is visible rather than at the point where the number is
absent.

`doctor` also accepts `--format json`. So do `diff`, `skill run`, `skill list`, `evidence build` and
`cutracer analyze` — six commands in total, not all of them, so check `--help` before assuming it is
available. The JSON output usually carries fields the human-readable rendering omits.

## Related

- [What to hand nsys-ai](./profile-inputs.md) — the three inputs and how they are chosen
- [Time windows](./time-windows.md) — `--trim`, `--iteration`, and the capture clock
- [Environment variables](./environment-variables.md) — every tuning knob and its default
- [Support matrix](../support-matrix.md) — export schema versions covered by tests
