# /nsys:validate — Validate Agent Output

**CRITICAL: Read `PRINCIPLES.md` first before executing any step.**

Validate that the agent's analysis meets all quality standards.
Produces a scored report like CLI-Anything's 52-item validator.

---

## Usage

```
/nsys:validate [claim]
```

- `claim` (optional): a specific statement to validate, e.g.
  - "MFU is 45%"
  - "The regression is in the flash attention kernel"
  - "NCCL overlap is 38%"
  - Omit → validate the most recent analysis output

---

## What This Command Does

Re-derive the key facts independently, then run all checks below.
Produce a scored report. **Do not skip any category.**

---

## Validation Checklist (40 checks)

### Category 1: Correctness of Values (10 checks)

- [ ] C1.1 MFU is between 0% and 100% (inclusive)
- [ ] C1.2 `theoretical_flops` came from a theoretical FLOPs calculator, not from manual arithmetic
- [ ] C1.3 `peak_tflops` came from GPU info query or is explicitly provided by user
- [ ] C1.4 `operation` parameter matches the exact scope of the measured region/kernel
- [ ] C1.5 `num_layers` used in FLOPs calculation matches the actual model layer count
- [ ] C1.6 `multiplier` matches the training mode (1=fwd, 3=fwd+bwd, 4=fwd+bwd+ckpt)
- [ ] C1.7 Time values were converted from ns (÷1e6 for ms, ÷1e9 for s)
- [ ] C1.8 `mfu_pct_wall` is reported as the primary MFU (not `mfu_pct_kernel_union`)
- [ ] C1.9 Step time is from a single representative iteration, not the full profile span
- [ ] C1.10 If using whole-profile span, this was stated and flagged as approximate

### Category 2: Name Discovery (5 checks)

- [ ] C2.1 NVTX region name was discovered by querying `NVTX_EVENTS`, not guessed
- [ ] C2.2 Kernel name was discovered by querying `StringIds`, not guessed
- [ ] C2.3 NVTX regions were searched before running region diff in diff mode
- [ ] C2.4 The discovered name substring actually matches the intended target
- [ ] C2.5 `source="nvtx"` vs `source="kernel"` was set correctly for the target type

### Category 3: Diff Analysis (8 checks)

- [ ] C3.1 Iteration boundaries were detected first in diff mode
- [ ] C3.2 Iteration index 0 was skipped (JIT warmup)
- [ ] C3.3 `is_aligned=false` was flagged and global diff used instead
- [ ] C3.4 JIT_Compilation_Warning was checked and handled
- [ ] C3.5 Hardware_Warning (thermal throttle) was noted if present
- [ ] C3.6 Root cause statement includes: cause, evidence field+delta value, recommendation
- [ ] C3.7 NCCL regression was followed up with GPU imbalance stats
- [ ] C3.8 Profile with ≤1 iteration was flagged as too short (not silently used)

### Category 4: SQL Quality (7 checks)

- [ ] C4.1 No `SELECT *` was used
- [ ] C4.2 All time values divided by correct factor (1e6 for ms, 1e9 for s)
- [ ] C4.3 `k.[end]` used (not `k.end` — reserved word)
- [ ] C4.4 Kernel names accessed via `JOIN StringIds s ON k.shortName=s.id`
- [ ] C4.5 NVTX text accessed via `COALESCE(n.text, s.value)` pattern
- [ ] C4.6 All queries included `LIMIT` clause
- [ ] C4.7 NCCL kernel filter used `LOWER(s.value) LIKE '%nccl%'`

### Category 5: Output Quality (5 checks)

- [ ] C5.1 GPU name and peak TFLOPS stated in output
- [ ] C5.2 All numeric values include units (ms, s, %, TFLOPS)
- [ ] C5.3 MFU result states whether forward-only or forward+backward
- [ ] C5.4 MFU result compared to reference range (< 20% / 30-45% / 45-60% / > 60%)
- [ ] C5.5 No hedging on a wrong result (e.g. MFU > 100% must be corrected, not reported)

### Category 6: Tool Usage (5 checks)

- [ ] C6.1 Only single-profile analysis tools (e.g. `compute_theoretical_flops`, `compute_region_mfu`, `navigate_to_kernel`, `fit_nvtx_range`) used when a single profile is loaded
- [ ] C6.2 Only diff-mode tools (e.g. `get_iteration_diff`, `get_top_nvtx_diffs`) used when comparing two profiles
- [ ] C6.3 `theoretical_flops` passed directly to next analysis step (not re-typed)
- [ ] C6.4 `navigate_to_kernel` / `fit_nvtx_range` used correctly when navigation was needed
- [ ] C6.5 No tool was called with parameters that weren't derived from profile data

---

## Validation Report Format

```
nsys-ai Agent Output Validation Report
=======================================
Claim:    <what was validated>
Profile:  <profile loaded>

Category 1: Correctness of Values    (X/10 passed)
Category 2: Name Discovery           (X/5  passed)
Category 3: Diff Analysis            (X/8  passed)  [N/A if single profile]
Category 4: SQL Quality              (X/7  passed)
Category 5: Output Quality           (X/5  passed)
Category 6: Tool Usage               (X/5  passed)
-----------------------------------------------
Overall: X/40 checks passed

Status: PASS ✓  (all checks pass)
   — or —
Status: FAIL ✗  (N checks failed)
Failed checks: C1.1 MFU = 143% (must be ≤ 100%), C2.1 name was guessed, ...
```

---

## Re-derivation Steps

For each claim, re-run independently:

| Claim type | Re-run |
|-----------|--------|
| MFU value | `compute_theoretical_flops` → `compute_region_mfu` with same params |
| Diff delta | Iteration diff (iteration_index=N) → read same field |
| NCCL claim | Query `CUPTI_ACTIVITY_KIND_KERNEL WHERE LIKE '%nccl%'` fresh |
| Root cause | `get_top_nvtx_diffs` + `get_iteration_diff` signals |

If re-derived value differs from claim by > 5% → flag as INCONSISTENT.
