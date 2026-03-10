# Skill: Profile Triage

Read this when the user opens a profile with no specific question,
or asks "what's the bottleneck?" / "why is it slow?".
**Read `PRINCIPLES.md` first** for rules, error handling, and tool definitions.

---

## Workflow 0: Triage

```
Step 1  get_gpu_peak_tflops()
        → Record gpu_name and peak_tflops. Note error if GPU unknown.

Step 2  Basic profile facts:
        SELECT (MAX([end]) - MIN(start)) / 1e9 AS span_s, COUNT(*) AS n_kernels
        FROM CUPTI_ACTIVITY_KIND_KERNEL

Step 3  Top kernels by GPU time:
        SELECT s.value AS name, COUNT(*) AS cnt,
               SUM(k.[end]-k.start)/1e6 AS total_ms,
               ROUND(100.0*SUM(k.[end]-k.start) /
                 (SELECT MAX([end])-MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL), 1) AS pct_of_span
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
        GROUP BY k.shortName ORDER BY total_ms DESC LIMIT 15

Step 4  Check NVTX and NCCL presence (two separate queries):
        SELECT COUNT(*) AS nvtx_count FROM NVTX_EVENTS
        SELECT COUNT(*) AS nccl_count,
               SUM(k.[end]-k.start)/1e6 AS nccl_ms
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
        WHERE LOWER(s.value) LIKE '%nccl%'

Step 5  Classify the bottleneck from top kernel names:
        ┌────────────────────────────┬──────────────────────────────────────────────┐
        │ Top kernel pattern         │ Diagnosis                                    │
        ├────────────────────────────┼──────────────────────────────────────────────┤
        │ flash_fwd / flash_bwd      │ Attention-bound (normal for seq_len > 4096)  │
        │ sm80_xmma / ampere_sgemm   │ GEMM-bound (projection layers dominate)      │
        │ nccl* > 20% of span        │ Communication-bound → read skills/distributed│
        │ elementwise* / layer_norm  │ Memory bandwidth-bound (small batch/model)   │
        │ All kernels tiny, many gaps│ CPU-bound / launch overhead (not GPU-bound)  │
        └────────────────────────────┴──────────────────────────────────────────────┘

Step 6  [Exploration extension] Check iteration regularity if NVTX present:
        SELECT MIN([end]-start)/1e6 AS min_ms,
               MAX([end]-start)/1e6 AS max_ms,
               AVG([end]-start)/1e6 AS avg_ms,
               COUNT(*) AS n_iters
        FROM NVTX_EVENTS
        WHERE COALESCE(text, '') LIKE '%sample%' OR COALESCE(text, '') LIKE '%step%'
        [If max_ms > 2 × avg_ms] → read skills/variance.md

Step 7  Give the user a concise summary, then ask what to investigate:
        "GPU: <name>, peak <X> TFLOPS BF16
         Profile span: <X>s, <N> kernels
         Primary bottleneck: <kernel/pattern> = <X>% of GPU time
         NVTX: <present/absent>  |  NCCL: <present: X% / absent>

         What would you like to investigate?
         (a) GPU efficiency / MFU  (b) Specific regression vs another run
         (c) NCCL / distributed    (d) Iteration variance"
```

## Routing Table

After triage, route to:

| Finding | Next step |
|---------|-----------|
| User wants MFU number | `skills/mfu.md` |
| Before + after profiles loaded | `skills/diff.md` |
| nccl_count > 0 and user concerned | `skills/distributed.md` |
| max_iter_ms > 2 × avg | `skills/variance.md` |
| Need to query DB further | `skills/sql.md` |
