# Mode 1 — Auto Triage

Reference for `/analyze` Mode 1 (auto triage). **Read `PRINCIPLES.md` first** — in particular
§4 (guards), §5 (evidence), §7 (fail template), §10 (acceptance checklist).

---

## 1. Precondition gate

§4.1 rows 1–4 (see PRINCIPLES.md). If any fail, render via §7 template and abort.

---

## 2. Stages

| # | Question | Condition |
|---|----------|-----------|
| 1 | Profile path | Only if not supplied in invocation |
| 2 | "Is this training or inference?" | Only if framework ambiguous — see §2.1 |
| 3 | "Detected N iterations. Analyze iterations 2 to N-1? (skips JIT warmup + teardown)" | Only if `iteration_timing` reports ≥3 iterations |

### 2.1 Framework ambiguous definition

Ambiguous = `fingerprint.framework == "Generic CUDA"` AND no top-10 kernel name matches any
of: `ncclDevKernel*`, `flash_*`, `ampere_*gemm*`, `volta_*gemm*`, `cutlass_*`, `sm80_*`,
`sm90_*`, `void at::*`. Otherwise skip Stage 2.

User's answer overrides `fingerprint.framework` for the session only. "inference" →
ms/token framing; "training" → look for `_bwd`/`wgrad`/`dgrad` kernels and DataLoader
signals. No file writes.

### 2.2 Device 0 auto-retry (silent — NOT a user stage)

After Stage 1 runs `profile_health_manifest`, if the response has:
- `overlap.error == "no kernels found"` AND
- `overlap.available_devices` is non-empty

…then plugin picks the first numeric key (lowest device id) and re-runs manifest with
`-p device=N`. Tell user once: `"Detected active GPU: device N"`. No user question.

---

## 3. Skills

### 3.1 Primary

```bash
nsys-ai skill run profile_health_manifest <profile> --format json
```

Then follow `suspected_bottleneck` routing (§4 below).

### 3.2 Fallback

If `suspected_bottleneck` is empty/"None" AND `root_causes[]` is empty:

```bash
nsys-ai skill run root_cause_matcher <profile> --format json
```

If returns patterns, re-route via §4 keyword logic. If still empty, offer menu 2/3/4/6.

### 3.3 Device propagation

If §2.2 auto-retry selected device N, every subsequent skill in the drill-down that accepts
device must receive `-p device=N`. See PRINCIPLES.md §6 for the 11 / 12 / 10 partition.

---

## 4. Signals — routing from `suspected_bottleneck`

| Keyword match (case-insensitive) | Drill into | Skills invoked |
|----------------------------------|------------|----------------|
| `nccl` / `comm` | Mode 2 subroutine | `overlap_breakdown`, `nccl_breakdown`, `kernel_overlap_matrix` |
| `sync` / `idle` / `bubble` | Mode 6 subroutine | `gpu_idle_gaps`, `sync_cost_analysis`, `kernel_launch_overhead`, `cpu_gpu_pipeline` (+ `pipeline_bubble_metrics` if `nccl.collectives > 0`) |
| `hotspot` / `kernel` | Mode 3 subroutine | `top_kernels`, `kernel_launch_pattern`, `tensor_core_usage` |
| `h2d` / `transfer` | Mode 4 subroutine | `memory_bandwidth`, `memory_transfers`, `h2d_distribution` |

### 4.1 CLI-field schema contract (source of truth for every mode ref)

Pinned against `profile_health_manifest`. Do NOT restate this table in other mode refs —
they reference `M1_AUTO.md §4.1`.

| Field | Type | Meaning |
|-------|------|---------|
| `gpu` | str | GPU name (not `gpu_name`) |
| `profile_span_ms` | float | Total span in ms (not seconds) |
| `fingerprint.framework` | str | "vLLM" / "Megatron-LM" / "Generic CUDA" / … |
| `fingerprint.distributed` | bool | **Unreliable alone** — also check `nccl.collectives` |
| `nccl.collectives` | int | >0 means multi-GPU workload |
| `top_kernels[]` | list | `{name, total_ms, count}` (no `device` field — filter via manifest's `-p device=N` if needed) |
| `top_kernels[].name` | str | First `ncclDevKernel*` ⇒ multi-GPU |
| `overlap.overlap_pct` | float | <30% with NCCL ⇒ comm-bound (Mode 2) |
| `overlap.error` | str | "no kernels found" → §2.2 auto-retry trigger |
| `overlap.available_devices` | dict | `{N: kernel_count}` — §2.2 retry target |
| `idle.idle_pct` | float | >15% → Mode 6 |
| `sync.sync_density_pct` | float | >20% → Mode 6 |
| `root_causes[].pattern` | str | Label (e.g. "NCCL Serialization") |
| `root_causes[].severity` | str | `critical` / `warning` / `info` |
| `suspected_bottleneck` | str | **Free-form** — keyword-match per §4 routing |
| `data_quality.overhead_pct` | float | >1% → surface note (not a block) |

---

## 5. Cross-mode exits (placeholders for B/C stages)

After Mode 1 delivery, suggest specialist mode only if a second critical finding exists.
**Cap 2 chains per session** (UX invariant 7).

> **Mode 2 — Comms (NCCL / overlap)**: coming Stage B1. See `M2_COMMS.md` once landed.
> **Mode 3 — Compute (kernels / MFU)**: coming Stage B2. See `M3_COMPUTE.md`.
> **Mode 4 — Memory (H2D / bandwidth)**: coming Stage B2. See `M4_MEMORY.md`.
> **Mode 5 — NVTX / code mapping**: coming Stage B2. See `M5_NVTX.md`.
> **Mode 6 — Idle / sync**: coming Stage B1. See `M6_IDLE.md`.
> **Mode 7 — CUTracer (SASS)**: coming Stage C1. See `M7_CUTRACER.md`.
> **Mode 8 — Diff**: coming Stage C2. See `M8_DIFF.md`.
> **Mode 9 — Variance**: coming Stage C2. See `M9_VARIANCE.md`.

During Stage A (only Mode 1 lives), after auto-triage completes, emit the 3-part summary
directly — no chain prompt until Stage B1 lands.

---

## 6. Delivery

Follow `PRINCIPLES.md` §5 (universal evidence + timeline step). Then the 3-part summary
framed for Mode 1.

### 6.1 Evidence CLI (inline here; other mode refs just point at PRINCIPLES.md §5)

```bash
nsys-ai evidence build <profile> --format json -o /tmp/findings.json
nsys-ai timeline-web <profile> --findings /tmp/findings.json
```

Then print to user the URL that `timeline-web` emitted (look for `http://127.0.0.1:PORT`
in its stdout; `src/nsys_ai/web.py:674` hard-codes `127.0.0.1`, not `localhost`):

> "Timeline ready at http://127.0.0.1:PORT — open in browser to see findings overlay."

**WSL2**: browser does NOT auto-open. Always print the URL. Use `localhost` interchangeably
with `127.0.0.1` — both reach the same server.

**Fail-soft**: if `evidence build` produces empty `{"findings": []}`, still run `timeline-web`.
Never block delivery on evidence failure.

### 6.2 3-part summary (Mode 1 framing)

1. **Root cause** — one sentence citing the dominant bottleneck + quantified impact:
   > "Your NCCL AllReduce is serialized with compute (overlap = 18%). This wastes
   > approximately 3.2 s of every 8.4 s training step."

2. **Specific fix** — code example (ideally file:line if source present):
   ```python
   model = DDP(model, bucket_cap_mb=256)
   ```

3. **Expected gain** — from `speedup_estimator` if NVTX present; **omit** the line otherwise:
   > "speedup_estimator: this fix → ≈ 1.4× faster per step."

### 6.3 Inference framing

`fingerprint.framework ∈ {vLLM, SGLang, TensorRT}` OR no `_bwd`/`wgrad`/`dgrad` in top-10
→ reframe metrics as ms/token and label "inference workload".

### 6.4 Optional text report

On explicit user request only: see PRINCIPLES.md §5.4 for the `nsys-ai report` CLI.

### 6.5 Required output checklist

See PRINCIPLES.md §10 — run that checklist before ending the turn.
