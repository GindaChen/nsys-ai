# Program: Profile-Guided ML Optimization

## Goal

Improve `val_bpb` on a transformer language model by making **informed** code
changes. You have access to GPU profiling tools that show you **why** the model
is slow — not just whether a change helped.

## File Structure

```
train.py              — Model + training loop (the file you edit)
experiments.jsonl     — Experiment log (append one JSON line per experiment)
profiles/             — nsys profile outputs (.sqlite files)
skills/               — Custom SQL skills you create during the experiment
```

## Core Loop

For each experiment:

1. **Edit** `train.py` — make ONE focused change
2. **Train** — `uv run train.py`
3. **Evaluate** — record `val_bpb`
4. **Decide** —
   - If `val_bpb` improved → **keep** the change
   - If `val_bpb` worsened → **revert** and try something else
5. **Profile** (optional) — run nsys profiling to understand GPU behavior

> **When to profile**: Profile when you need to understand the *system*
> bottleneck (GPU idle, NCCL overhead, kernel launch cost). Don't profile
> when the change is purely about convergence (optimizer, learning rate,
> regularization).

---

## Profiling Tools

### Step 1: Profile a training run

```bash
scripts/nsys_profile_wrapper.sh profiles/exp_N uv run train.py
```

This creates `profiles/exp_N.sqlite` — a queryable GPU profile database.

### Step 2: Query the profile

Use `nsys-ai skill run` to analyze specific aspects of GPU behavior.
Add `--format json` for machine-readable output.

| Skill | What it tells you | When to use |
|-------|-------------------|-------------|
| `top_kernels` | Heaviest kernels by total GPU time | "What's eating GPU time?" |
| `gpu_idle_gaps` | Pipeline bubbles and stalls between kernels | "Why is GPU underutilized?" |
| `nccl_breakdown` | NCCL collective communication time | "Is multi-GPU comm a bottleneck?" |
| `memory_transfers` | Host↔Device data movement | "Am I data-transfer bound?" |
| `kernel_launch_overhead` | Small kernel dispatch latency | "Too many tiny kernel launches?" |
| `thread_utilization` | CPU thread activity during GPU work | "Is CPU the bottleneck?" |
| `nvtx_kernel_map` | NVTX annotation → kernel mapping | "Which code line causes this kernel?" |
| `schema_inspect` | Available tables in the profile DB | "What data can I query?" |

**Usage:**

```bash
nsys-ai skill run top_kernels profiles/exp_N.sqlite --format json
nsys-ai skill run gpu_idle_gaps profiles/exp_N.sqlite --format json
nsys-ai skill run nccl_breakdown profiles/exp_N.sqlite --format json
```

### Step 3: Diagnose and act

Read the profiling data, form a hypothesis about the bottleneck, and make
an informed edit. Record your reasoning in the experiment log.

---

## Worked Examples

### Example A: Kernel launch overhead → CUDA Graphs

```
Experiment 12:
  Change: Added torch.compile() to MLP layers
  val_bpb: 3.821 → 3.819 (marginal improvement, kept)
  Profiled: YES

  top_kernels output:
    volta_sgemm_128x64_nn     1823 calls  892.4ms total  (38% of GPU time)
    elementwise_kernel         5200 calls   32.1ms total

  gpu_idle_gaps output:
    GPU idle: 22% of wall time
    Average gap: 450µs
    Max gap: 12ms

  Diagnosis: Not a single slow kernel — the problem is 5200 tiny elementwise
  launches creating 450µs gaps between them. Kernel launch overhead, not
  kernel speed.

  Action: Enable CUDA Graphs to batch kernel launches.
```

```
Experiment 13:
  Change: Wrapped forward pass in CUDA Graph capture
  val_bpb: 3.819 → 3.811 (good improvement, kept)
  Profiled: YES

  gpu_idle_gaps output:
    GPU idle: 4% of wall time (was 22%)
    Average gap: 45µs (was 450µs)

  Result: 10× reduction in idle gaps. val_bpb improved from eliminating
  pipeline stalls.
```

### Example B: NCCL blocking compute → gradient bucketing

```
Experiment 20:
  Profiled: YES

  nccl_breakdown output:
    AllReduce: 35% of step time
    Total NCCL: 38% of step time

  top_kernels output:
    flash_fwd_kernel      512 calls   420ms   (fast individually)
    nccl_allreduce_ring   256 calls   680ms   (huge)

  Diagnosis: AllReduce is serialized with compute — not overlapping.
  Each gradient sync blocks until complete before next compute starts.

  Action: Enable gradient bucketing + overlap_comm_compute=True.

  Result: Step time decreased 18%, val_bpb improved 0.012.
```

### Example C: When NOT to profile

```
Experiment 7:
  Change: Switched optimizer from Adam to AdamW with weight decay=0.01
  val_bpb: 3.845 → 3.842 (improved, kept)
  Profiled: NO

  Reasoning: This is a convergence change (optimizer behavior), not a
  systems change. GPU utilization patterns are identical — only the
  gradient updates change mathematically. No profiling needed.
```

---

## Writing New Skills

If existing skills don't answer your question, create a new SQL skill:

```bash
# 1. Write a .md file following this format:
cat > skills/kernel_overlap.md << 'EOF'
# kernel_overlap
## Description
Find pairs of kernels overlapping on different streams.
## Category
kernels
## SQL
```sql
SELECT s1.value AS kernel_a, s2.value AS kernel_b,
       ROUND((MIN(k1.[end], k2.[end]) - MAX(k1.start, k2.start)) / 1e6, 2) AS overlap_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k1
JOIN CUPTI_ACTIVITY_KIND_KERNEL k2
  ON k1.start < k2.[end] AND k2.start < k1.[end]
  AND k1.streamId != k2.streamId AND k1.rowid < k2.rowid
JOIN StringIds s1 ON k1.demangledName = s1.id
JOIN StringIds s2 ON k2.demangledName = s2.id
ORDER BY overlap_ms DESC LIMIT 20
```
EOF

# 2. Add it to nsys-ai
nsys-ai skill add skills/kernel_overlap.md --skills-dir ./skills/

# 3. Use it
nsys-ai skill run kernel_overlap profiles/exp_N.sqlite --format json

# 4. If it's not useful, remove it
nsys-ai skill remove kernel_overlap --skills-dir ./skills/
```

Skills that produce useful insights should be kept. Skills that don't should
be reverted — same keep/revert pattern as code changes.

---

## Experiment Log Format

Append one JSON line per experiment to `experiments.jsonl`:

```json
{
  "id": 1,
  "change": "Enable CUDA Graphs for forward pass",
  "val_bpb": 3.811,
  "prev_val_bpb": 3.819,
  "profiled": true,
  "diagnosis": "5200 elementwise launches → 450µs avg idle gap → CUDA Graphs",
  "kept": true,
  "wall_s": 142,
  "skills_used": ["top_kernels", "gpu_idle_gaps"],
  "new_skills_created": []
}
```

---

## Decision Guide

```
Is val_bpb improving?
├── YES → Keep change. Profile only if curious about mechanism.
└── NO → Revert change.
    ├── Do you understand why?
    │   ├── YES → Try a different approach.
    │   └── NO → Profile to understand the bottleneck.
    │       ├── GPU idle > 15%? → Look at kernel launch overhead, CUDA Graphs
    │       ├── NCCL > 30% of step? → Look at overlap, bucketing
    │       ├── Single kernel > 40%? → Look at kernel optimization
    │       └── Memory transfer > 10%? → Look at data pipeline, pinned memory
    └── Already profiled? → Read the data more carefully before trying more.
```
