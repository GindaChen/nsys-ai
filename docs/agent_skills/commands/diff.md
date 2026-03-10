# /nsys:diff — Profile Comparison

**CRITICAL: Read `PRINCIPLES.md` first before executing any step.**

Slash command for: comparing two Nsight Systems profiles (before → after).
Use when the user wants to find what changed between two runs.

---

## Usage

```
/nsys:diff [focus]
```

- `focus` (optional): what to investigate
  - Omit → full regression analysis
  - "mfu" → MFU comparison (Workflow 5 in `skills/diff.md`)
  - "nccl" → communication diff → routes to `skills/distributed.md`
  - "region <name>" → zoom to a specific NVTX region diff

---

## What This Command Does

### Phase 0: Load Check
- Verify TWO profiles are loaded (before + after)
- If only one loaded: ask user to load the second profile
- Run `get_gpu_peak_tflops()` to confirm the profile and record GPU

### Phase 1: Alignment Check
- Run `get_iteration_boundaries()`
- Evaluate: `is_aligned`, `iteration_count_before`, `iteration_count_after`
- If `iteration_count ≤ 1`: stop and tell user profile is too short (need ≥ 3 iterations)
- If `is_aligned=false`: warn user; use `get_global_diff` instead of per-iteration analysis

### Phase 2: Hotspot Radar
- Run `get_top_nvtx_diffs(limit=20)`
- Classify the dominant pattern (see `skills/diff.md` Step 2 classification table)
- State the top 3 regressing regions

### Phase 3: Iteration Diff
- Run `get_iteration_diff(iteration_index=<stable N ≥ 1>)`
- Read all key signals from the table in `skills/diff.md` Step 3
- If NCCL spiked → load `skills/distributed.md` and run Step 4

### Phase 4: Drill Down (if warranted)
- Use `search_nvtx_regions` → `get_region_diff` for the top regressor
- Use `get_gpu_imbalance_stats` if NCCL is implicated

### Phase 5: Root Cause Statement (REQUIRED)
Output a structured conclusion:
```
Root cause: <X>
Evidence:   <field_name>.delta = <value> ms (+N%)
Trigger:    <what the user likely changed>
Fix:        <specific actionable recommendation>
```
Never omit this. See PRINCIPLES.md Rule 6.

---

## Output Requirements

- Wall clock delta in both ms and %
- Top 3 regressing regions by absolute delta
- Root cause statement (required, see Phase 5 format)
- Optional: MFU before/after if user requests efficiency metrics

---

## Success Criteria

Run the Acceptance Checklist from `PRINCIPLES.md` before delivering.
