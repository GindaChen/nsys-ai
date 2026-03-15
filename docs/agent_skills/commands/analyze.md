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

**Complete agent loop** — do not skip the reasoning step:

1. **COLLECT** — query skills for raw data:
   ```bash
   nsys-ai skill run gpu_idle_gaps profile.sqlite --format json > /tmp/gaps.json
   nsys-ai skill run nccl_breakdown profile.sqlite --format json > /tmp/nccl.json
   ```

2. **REASON** — analyze the collected data (this is your AI reasoning, not a command):
   - Cross-reference multiple skill outputs
   - Identify root causes and causal relationships
   - Draw conclusions with specific evidence
   - Example: "21s GPU idle gap is caused by serialized NCCL AllReduce — the gap starts exactly where AllReduce ends"

3. **WRITE** — encode your **conclusions** (not raw data) as findings:
   ```bash
   cat > /tmp/findings.json << 'EOF'
   {
     "title": "Your analysis title",
     "findings": [
       {
         "type": "region",
         "label": "PP Bubble: 21s idle caused by serialized NCCL",
         "start_ns": 89000000000,
         "end_ns": 110000000000,
         "severity": "critical",
         "note": "GPU idle for 21s after AllReduce — NCCL not overlapping with compute"
       }
     ]
   }
   EOF
   ```
   > **Key**: the `label` must state your **conclusion**, not just "idle gap".
   > The `note` must explain **why** this time range matters.
   > See [`evidence_schema.md`](evidence_schema.md) for the full schema.

4. **VIEW** — open timeline with evidence overlay:
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
