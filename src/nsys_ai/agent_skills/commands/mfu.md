# /nsys:mfu — MFU / Efficiency Analysis

**CRITICAL: Read `PRINCIPLES.md` first before executing any step.**

Slash command for: computing GPU efficiency (MFU) for a specific region or whole step.

---

## Usage

```
/nsys:mfu [target]
```

- `target` (optional):
  - Omit or "step" → whole training step MFU (Workflow 2 in `skills/mfu.md`)
  - "flash" / "attention" → flash attention MFU (Workflow 1)
  - "forward" / "fwd" → forward pass NVTX region
  - "backward" / "bwd" → backward pass NVTX region
  - `"<region name>"` → specific NVTX region by substring

---

## What This Command Does

### Phase 0: Load Check
- Verify a profile is loaded
- Call `get_gpu_peak_tflops()` → record `{gpu_name, peak_tflops}`
- If GPU unknown: note error; will ask user for peak_tflops when needed

### Phase 1: Discover Target
- Check NVTX annotation presence (query `COUNT(*) FROM NVTX_EVENTS`)
- If NVTX present and target is a region: query NVTX names
- If targeting a kernel directly: query `StringIds` for kernel substring
- Load `skills/mfu.md` for the detailed workflow

### Phase 2: Resolve Model Architecture
**Do NOT ask user first.** Follow this priority order:
1. User already stated the model → use lookup table in `skills/mfu.md`
2. Infer from kernel timing (see `skills/mfu.md` Step 3, option B)
3. Ask user as last resort → end message → wait for response

### Phase 3: Compute FLOPs and MFU
- Select `operation` from the FLOPs mapping table in `skills/mfu.md`
- Call `compute_theoretical_flops(...)` → capture `theoretical_flops`
- Call `compute_region_mfu(...)` or `compute_mfu(...)` depending on target
- Run sanity check (see `skills/mfu.md` Step 6/7)

### Phase 4: Contextualise
- State GPU name, peak TFLOPS, achieved TFLOPS, MFU %
- Compare to reference ranges (see `skills/mfu.md` Workflow 2 Step 6)
- State whether MFU is forward-only or forward+backward

---

## Output Format

```
GPU:     <name>, peak <X> TFLOPS BF16
Region:  <name> (NVTX / kernel)
FLOPs:   <X>e12 theoretical (per step / per region)
Time:    <X> ms wall (median of N occurrences)
MFU:     <X>% wall  (<Y>% kernel-union upper bound)
Context: <within range / below expected / excellent>
```

---

## Success Criteria

- MFU between 0% and 100% (if > 100%, stop and explain error)
- `theoretical_flops` from a theoretical FLOPs calculator, not estimated
- NVTX/kernel name from query, not guessed
- Step time from single representative iteration (not full profile span)
- Run the Acceptance Checklist from `PRINCIPLES.md` before delivering
