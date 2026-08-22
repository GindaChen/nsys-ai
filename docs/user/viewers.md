# Choosing a Web viewer

nsys-ai has three browser surfaces with similar names but different jobs. They
all run locally and accept the same profile inputs; choose the surface that
matches the question rather than starting every server.

| Command | What it shows | Use it when |
|---|---|---|
| `nsys-ai web PROFILE` | NVTX tree viewer with lazy tree requests | You want to browse nested ranges and the kernels attributed to them |
| `nsys-ai timeline-web PROFILE` | Progressive multi-GPU horizontal timeline | You want streams, kernels, NVTX lanes, search, and a session-backed workflow |
| `nsys-ai diff-web BEFORE AFTER` | Before/after comparison shell with two timeline views | You want to inspect a change and its canonical diff side by side |

The commands are separate transports over the same profile and diff contracts.
The older `nsys-ai perfetto` server is not one of the choices; use
`nsys-ai export` when you need a Chrome Trace Event JSON file instead.

## NVTX tree: `web`

Use this for hierarchy-first investigation:

```console
$ nsys-ai web PROFILE.sqlite --port 8142
```

The server prints the URL and builds the NVTX tree in the background. The root
page is the interactive tree viewer; `GET /api/tree` returns a bounded tree
slice. `depth`, `limit`, and `max_nodes` bound the response, and a truncated
response carries `has_more` / `children_total` metadata rather than pretending
that a high-fan-out node is complete.

`web` selects one GPU when `--gpu` is omitted. Use this surface when the useful
question is “which ranges contain this work?” rather than “how do all streams
overlap over time?”.

## Horizontal timeline: `timeline-web`

Use this for a progressive, multi-GPU timeline:

```console
$ nsys-ai timeline-web PROFILE.sqlite --port 8144 --no-browser
```

The initial HTML shell is served while the NVTX data is prepared in the
background. The client reads profile metadata from `/api/meta` and requests
time tiles from `/api/data?start_s=START&end_s=END`. The query values are
seconds on the Nsight capture clock, not indexes and not nanoseconds.

This surface owns the browser loop UI and can open a session handoff:

```console
$ nsys-ai timeline-web PROFILE.sqlite --session /tmp/nsys-ai/run-001
```

Use `--findings FINDINGS.json` or `--auto-analyze` when you want evidence
overlays. Use the [guided loop setup](../guided-loop-setup.md) for the full
diagnose → propose → re-profile → diff → decide path.

## Pair comparison: `diff-web`

Use this after you have two captures:

```console
$ nsys-ai diff-web BEFORE.sqlite AFTER.sqlite --port 8145 --no-browser
```

The shell exposes the canonical pair metadata at `/api/diff/meta` and the
deterministic summary at `/api/diff/summary`. The two embedded timeline views
are read-only inspection surfaces; record a decision through the session-aware
CLI or the session loop rather than treating a browser colour as the decision.

`--gpu` restricts both sides to one device. Without it, the comparison remains
all-GPU, matching `nsys-ai diff`.

## Shared operational rules

- `--no-browser` is useful on a remote host or in an automated smoke test; the
  command still prints the local URL.
- A requested port may be replaced with a free local port when it is already
  occupied. Use the URL printed by the process, not a hard-coded port.
- Use `nsys-ai doctor PROFILE` before a large `.nsys-rep` analysis. It reports
  whether conversion, cache, and schema checks are ready.
- For a trace file rather than an interactive service, run:

  ```console
  $ nsys-ai export PROFILE.sqlite --trim START_S END_S -o trace-export/
  ```

See [profile inputs](profile-inputs.md), [time windows](time-windows.md), and
[known limits](known-limits.md) before interpreting a partial or incomparable
view.
