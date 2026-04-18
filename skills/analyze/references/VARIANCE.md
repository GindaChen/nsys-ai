# Skill: Iteration Variance

Read this when some training steps are much slower than others,
or the user says "the profile looks choppy" / "some iterations spike".

---

## Workflow 7: Variance Diagnosis

```
Step 1  Get per-iteration timing (NVTX iteration markers preferred for best accuracy):
        Prefer the CLI skill which handles schema differences automatically:
          nsys-ai skill run iteration_timing <profile> --format json

        This works best with NVTX iteration markers. If markers are missing or do not match,
        `iteration_timing` falls back to a kernel-gap heuristic — the inferred iteration
        boundaries are approximate and may be less reliable for some workloads.

        If using raw SQL, first check which columns NVTX_EVENTS has:
          PRAGMA table_info(NVTX_EVENTS)
        Then pick the appropriate query:

        [NVTX_EVENTS has textId column]
        SELECT COALESCE(n.text, s.value) AS name,
               ([end]-start)/1e6 AS duration_ms,
               start/1e9 AS start_s
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
        WHERE COALESCE(n.text, s.value) LIKE '%sample%'
           OR COALESCE(n.text, s.value) LIKE '%step%'
           OR COALESCE(n.text, s.value) LIKE '%iter%'
        ORDER BY start LIMIT 50

        [NVTX_EVENTS has only text column]
        SELECT text AS name, ([end]-start)/1e6 AS duration_ms, start/1e9 AS start_s
        FROM NVTX_EVENTS
        WHERE text LIKE '%sample%' OR text LIKE '%step%' OR text LIKE '%iter%'
        ORDER BY start LIMIT 50

        Compute statistics (branch on the same schema check as above):

        [NVTX_EVENTS has textId column]
        SELECT MIN([end]-start)/1e6 AS min_ms,
               MAX([end]-start)/1e6 AS max_ms,
               AVG([end]-start)/1e6 AS avg_ms,
               COUNT(*) AS n_iters
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
        WHERE COALESCE(n.text, s.value) LIKE '%sample%'
           OR COALESCE(n.text, s.value) LIKE '%step%'
           OR COALESCE(n.text, s.value) LIKE '%iter%'

        [NVTX_EVENTS has only text column]
        SELECT MIN([end]-start)/1e6 AS min_ms,
               MAX([end]-start)/1e6 AS max_ms,
               AVG([end]-start)/1e6 AS avg_ms,
               COUNT(*) AS n_iters
        FROM NVTX_EVENTS
        WHERE text LIKE '%sample%' OR text LIKE '%step%' OR text LIKE '%iter%'

        [If max_ms < 1.5 × avg_ms] → variance is normal; no problem to investigate

Step 2  Classify the variance pattern:
        ┌────────────────────────────────────────┬──────────────────────────────────────────┐
        │ Pattern                                │ Likely cause                             │
        ├────────────────────────────────────────┼──────────────────────────────────────────┤
        │ Every Nth iteration (e.g. every 4th)   │ Gradient accumulation / eval step        │
        │ First 2–3 iterations slow, rest OK     │ JIT compilation / CUDA cache warmup      │
        │ Random spikes, no fixed period         │ DataLoader I/O (disk latency variance)   │
        │ Slow iterations align with NCCL spikes │ NCCL retry / network instability         │
        │ Duration increases gradually over time │ GPU thermal throttle                     │
        │ Alternating fast/slow pairs            │ Double buffering / prefetch pipeline      │
        └────────────────────────────────────────┴──────────────────────────────────────────┘

Step 3  Zoom into one slow iteration to confirm:
        [CLI — preferred for Claude Code plugin]
        nsys-ai skill run iteration_detail <profile> --format json -p iteration=<slow iter index>
        or: nsys-ai skill run top_kernels <profile> --format json --iteration <slow iter index>

        [Interactive UI/tooling only — not available from plugin today]
        `fit_nvtx_range`(nvtx_name="<iteration name>", occurrence_index=<slow iter index>)
        (This is a navigate/UI tool; use the CLI alternatives above in the Claude Code plugin.)

        Query kernels in that window:
        SELECT s.value AS name, COUNT(*) AS cnt, SUM([end]-start)/1e6 AS ms
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
        WHERE k.start >= <slow_iter_start_ns> AND k.[end] <= <slow_iter_end_ns>
        GROUP BY k.shortName ORDER BY ms DESC LIMIT 20

        Compare to a normal iteration. Look for: extra kernels, longer kernel durations,
        large gaps (idle time), or NCCL spikes.

Step 4  Report:
        - What fraction of iterations are outliers (e.g. 3 out of 12 = 25%)
        - Is it expected (eval loop, gradient accumulation) or unexpected (I/O, GC)?
        - Specific recommendation:
          DataLoader I/O → prefetch_factor, num_workers, pin_memory
          JIT → skip first N steps in benchmark; profile after warmup
          Thermal → use nvidia-smi to check GPU clocks; reduce power limit
          NCCL retry → check network stability; NCCL_DEBUG=INFO logs
```

---

## CLI-Based Drill-Down

After Step 1, use the CLI skills for faster iteration:

```bash
# Step 1: Get per-iteration timing
nsys-ai skill run iteration_timing profile.sqlite --format json

# Step 3: Drill into a slow iteration (e.g. iteration 3)
nsys-ai skill run iteration_detail profile.sqlite --format json -p iteration=3

# Alternative: Use --iteration to auto-trim any skill to that iteration
nsys-ai skill run top_kernels profile.sqlite --format json --iteration 3
nsys-ai skill run gpu_idle_gaps profile.sqlite --format json --iteration 3
nsys-ai skill run kernel_instances profile.sqlite --format json --iteration 3

# Get exact ns timestamps for evidence overlay
nsys-ai skill run kernel_instances profile.sqlite --format json --iteration 3 -p name=flash
```
