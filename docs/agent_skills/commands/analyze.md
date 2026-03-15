# /nsys:analyze — Single Profile Analysis

**CRITICAL: Read `PRINCIPLES.md` first before executing any step.**

Slash command for: analyzing a single Nsight Systems profile.
Use this when no second profile is loaded for comparison.

---

## Usage

```
/nsys:analyze [question]
```

- `question` (optional): what the user wants to know
  - Omit → runs triage and lets user decide
  - "mfu" / "efficiency" → runs MFU workflow
  - "bottleneck" → runs triage
  - "nccl" / "distributed" → runs NCCL workflow
  - "variance" / "spiky" → runs variance workflow

---

## What This Command Does

### Phase 0: Load Check
- Verify a profile is loaded; if not, ask user to provide a `.sqlite` path
- Run `get_gpu_peak_tflops()` to confirm profile connection and record GPU name

### Phase 1: Triage (always runs)
- Load and execute `skills/triage.md` **Workflow 0**
- Classify the bottleneck (attention-bound / GEMM-bound / communication-bound / CPU-bound)
- Check NVTX presence and NCCL activity

### Phase 2: Route to Skill
Based on `question` argument and triage result:

| Condition | Load skill |
|-----------|-----------|
| User asks for MFU, efficiency | `skills/mfu.md` |
| User asks about NCCL, multi-GPU | `skills/distributed.md` |
| Iteration variance detected | `skills/variance.md` |
| No specific question | Give triage summary → ask user to choose |

### Phase 3: Execute Skill Workflow
Follow the loaded skill's workflow exactly. Do not skip steps.

### Phase 4: Deliver Result
- State the primary bottleneck and its % of GPU time
- Report any computed metrics (MFU %, achieved TFLOPS)
- Suggest the next investigation step

### Phase 5: Visual Evidence

After delivering your text conclusion, produce visual evidence so humans can verify your claims against the actual timeline.

1. **Identify key time ranges** — for each conclusion, note the `start_ns` and `end_ns` from the data you queried (e.g. from `skill run gpu_idle_gaps --format json`)
2. **Write findings JSON** — see [`evidence_schema.md`](evidence_schema.md) for the format:
   ```bash
   cat > /tmp/findings.json << 'EOF'
   {
     "title": "Your analysis title",
     "findings": [
       {
         "type": "region",
         "label": "Your conclusion label",
         "start_ns": 89000000000,
         "end_ns": 110000000000,
         "severity": "critical",
         "note": "Evidence supporting this conclusion"
       }
     ]
   }
   EOF
   ```
3. **Open the viewer**:
   ```bash
   nsys-ai timeline-web profile.sqlite --findings /tmp/findings.json
   ```

> Only include findings that directly support your conclusions. Each finding
> should highlight the specific timeline range that proves your claim.

---

## Output Requirements

- **Always state**: GPU name, peak TFLOPS, profile span
- **Always state**: primary bottleneck kernel and % of GPU time
- **If MFU computed**: state whether it is forward-only or forward+backward
- **If MFU > 100%**: stop and explain the error before reporting
- **Never output a number without units** (ms, s, %, TFLOPS)

---

## Success Criteria

Run the Acceptance Checklist from `PRINCIPLES.md` before delivering.
