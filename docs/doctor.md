# nsys-ai doctor

`nsys-ai doctor` checks whether your environment is ready to analyze profiles
and, when you give it a profile, whether that profile has enough data for a
useful diagnosis. Run it first when something is not working, or attach its
JSON output to a bug report.

```bash
nsys-ai doctor                      # environment checks only
nsys-ai doctor profile.sqlite       # environment + profile health
nsys-ai doctor profile.nsys-rep     # same; reports if conversion is possible
nsys-ai doctor profile.sqlite --format json
```

## What it checks

Doctor covers the **analysis** side — reading and analyzing a profile, and the
optional features layered on top. It does not check whether a host can *capture*
a profile; for that, use NVIDIA's own `nsys status -e`.

**System** — Python version, nsys-ai version, platform string.

**Profile support**

- SQLite analysis (always available; stdlib).
- `.nsys-rep` conversion — needs the `nsys` CLI on `PATH`. Missing it is only a
  warning: `.sqlite` profiles still work, but `.nsys-rep` files cannot be
  converted.
- Parquet cache — `duckdb` + `pyarrow`, which accelerate large profiles.

**Optional features**

- GUI / chat TUI — the `textual` package.
- AI provider — `litellm` plus a configured API key (`ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`, or `GEMINI_API_KEY`; `NSYS_AI_MODEL` to pick a model). Doctor
  reports separately whether `litellm` is installed and whether a key is set, so
  you know which half is missing.
- CUTracer — reported in two parts, following the "installed vs compatible"
  split:
  - **CUTracer**: the `cutracer` package and a built `cutracer.so`.
  - **nvdisasm ↔ framework CUDA**: the CUDA major version of `nvdisasm` must
    match the framework's CUDA build. A mismatch silently drops a hot kernel's
    SASS during CUTracer runs (see [cutracer-modal.md](cutracer-modal.md)). This
    check is best-effort: it is skipped when `nvdisasm` is absent or the
    framework CUDA version cannot be determined (install `torch` on the host, or
    run on the training host, to enable it).

**Profile health** (only when a profile is given)

- Duration, GPU count (`COUNT(DISTINCT deviceId)`), GPU model (from CUPTI
  `TARGET_INFO`; "unknown" blocks MFU), and whether the run is multi-GPU.
- NVTX events — absence is a warning, since layer attribution and code tracing
  depend on annotations.
- NCCL events (a fast presence count). The eager / inductor-captured /
  temporal-only call-mode split — which decides whether a fix belongs in user
  code or in `torch._inductor.config` — is a slow check (it joins the NVTX map),
  so it runs only under `--deep`.
- Profiler overhead — a warning above 10% and a failure above 20%, since high
  overhead distorts timings. An impossible figure (over 100%) is reported as
  unreliable rather than as a failure.
- RunSpec presence — reported as not-yet-available until profile capture records
  a RunSpec.

## Status meanings

| Token  | Meaning |
| ------ | ------- |
| `OK`   | Working / present |
| `WARN` | Usable, but degraded or distorted results likely |
| `FAIL` | Blocking — analysis cannot proceed or the data is unsound |
| `----` | Optional feature not configured |
| `SKIP` | Check could not run (not enough information) |

Hints for `WARN`, `FAIL`, and not-configured checks are shown by default. Pass
`-v` / `--verbose` to also show hints for skipped checks.

## Exit codes

- `0` — no failures (warnings do not change the exit code).
- `1` — at least one `FAIL` check.

Use `--strict` to also exit non-zero on warnings, which is useful in CI.

## Fast by default

Doctor is meant to be a fast triage, so it avoids expensive analysis. The NCCL
call-mode split joins the NVTX map and can take minutes on an NVTX-heavy
profile; it runs only when you pass `--deep`. The default run stays in the
single-digit-seconds range even on multi-gigabyte, multi-GPU profiles.

## JSON output

`--format json` prints a versioned envelope with a `summary` count. It is the
canonical thing to attach to a bug report, since it captures versions, platform,
and profile health in one place.

```json
{
  "schema_version": "0.1",
  "producer": "nsys-ai",
  "producer_version": "0.2.3",
  "profile_path": "profile.sqlite",
  "profile_id": "nsys1:sha256:...",
  "sections": [
    {"name": "System", "checks": [{"name": "Python", "status": "ok", "detail": "3.12.3", "hint": null, "sub": false}]}
  ],
  "summary": {"ok": 12, "warn": 1, "fail": 0, "not_configured": 2, "skipped": 0}
}
```
