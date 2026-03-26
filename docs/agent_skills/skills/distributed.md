# Skill: Distributed Training / NCCL Analysis

Read this when the user asks about multi-GPU slowdowns, NCCL time, or GPU imbalance.
**Read `PRINCIPLES.md` first** for rules, error handling, and tool definitions.

> **PhD-student reality**: They see 40% NCCL time and don't know if it's normal.
> The key is overlap_pct — NCCL is acceptable when hidden behind compute.

---

## Workflow 6: NCCL Efficiency Diagnosis

```
Step 1  Confirm NCCL activity + breakdown by operation:
        SELECT s.value AS op, COUNT(*) AS cnt,
               SUM(k.[end]-k.start)/1e6 AS total_ms,
               ROUND(100.0*SUM(k.[end]-k.start) /
                 (SELECT MAX([end])-MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL), 1) AS pct_of_span
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
        WHERE LOWER(s.value) LIKE '%nccl%'
        GROUP BY op ORDER BY total_ms DESC LIMIT 10

Step 2  Read overlap_pct:
        [Single profile] Use `get_gpu_overlap_stats` — returns per-GPU overlap_pct.
        [Diff context]   Use `get_iteration_diff` — overlap_pct for before/after.

        overlap_pct = fraction of NCCL time that overlaps with compute kernels × 100

        Interpretation (framework-dependent, not hard boundaries):
        ┌──────────────┬───────────────────────────────────────────────────────────────┐
        │ overlap_pct  │ Reading                                                       │
        ├──────────────┼───────────────────────────────────────────────────────────────┤
        │ > 60%        │ NCCL well-hidden under compute; likely not the bottleneck      │
        │ 30–60%       │ Partial overlap; ring allreduce partially pipelining           │
        │ < 30%        │ NCCL serialized with compute; throughput loss is significant   │
        └──────────────┴───────────────────────────────────────────────────────────────┘
        Context: FSDP/ZeRO-3 with prefetch typically achieves 50–70%.
                 DDP without bucketing often achieves < 20%.

Step 3  Classify collective type → infer parallelism strategy:
        ┌─────────────────────────┬────────────────────────────────────────────────────┐
        │ Dominant operation      │ Strategy                                           │
        ├─────────────────────────┼────────────────────────────────────────────────────┤
        │ AllReduce               │ DDP (gradient sync after backward)                 │
        │ ReduceScatter + AllGather│ FSDP / ZeRO-2/3 (sharded parameters)            │
        │ AllGather (forward only) │ Tensor Parallelism or FSDP inference             │
        │ Broadcast               │ Checkpoint sync / parameter broadcast at init     │
        └─────────────────────────┴────────────────────────────────────────────────────┘

Step 4  Per-GPU imbalance diagnosis:
        [Single profile] Use `get_gpu_overlap_stats` — compare compute_only_ms
        across gpu_ids. If max/min ratio > 1.2, report GPU imbalance.
        [Diff context]   Use `get_gpu_imbalance_stats(iteration_index=<N>)`
        → Per-GPU: {compute_ms, nccl_ms, idle_ms} for both profiles

        Diagnose:
        • One GPU has nccl_ms >> other GPUs  → STRAGGLER (slow NIC, load imbalance, link fault)
        • All GPUs: nccl_ms up uniformly     → Cross-node bandwidth congestion
        • All GPUs: idle_ms up, nccl_ms same → GPU launch latency / CPU overhead

Step 5  Give actionable recommendations:
        overlap_pct < 30%, using DDP:
          → "Increase DDP bucket_cap_mb, or migrate to FSDP"
        ReduceScatter+AllGather slow:
          → "Check FSDP prefetch_factor; measure with/without cpu_offload"
        Straggler identified (one GPU):
          → "Check NIC health on node <X>; rebalance dataset sharding"
        Cross-node uniform slowdown:
          → "Likely infiniband / RoCE issue; try NCCL_SOCKET_IFNAME, check switch"
```

---

## NCCL Kernel Name Patterns

nsys captures NCCL ops as CUDA kernels. Common name substrings:

| Name contains | Collective |
|---------------|-----------|
| `ncclAllReduce` | AllReduce |
| `ncclReduceScatter` | ReduceScatter |
| `ncclAllGather` | AllGather |
| `ncclBroadcast` | Broadcast |
| `ncclSendRecv` | P2P send/recv (pipeline parallel) |
| `ncclKernel_AllR` | AllReduce internal kernel |

All are matched by `WHERE LOWER(s.value) LIKE '%nccl%'`.
