# kernel_overlap

## Description

Find pairs of GPU kernels that overlap in time on different streams,
indicating compute-communication overlap. High overlap means good pipeline
efficiency — kernels are running concurrently instead of serially.

## Category

kernels

## SQL

```sql
SELECT s1.value AS kernel_a,
       s2.value AS kernel_b,
       k1.streamId AS stream_a,
       k2.streamId AS stream_b,
       ROUND((MIN(k1.[end], k2.[end]) - MAX(k1.start, k2.start)) / 1e6, 2) AS overlap_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k1
JOIN CUPTI_ACTIVITY_KIND_KERNEL k2
  ON k1.start < k2.[end] AND k2.start < k1.[end]
  AND k1.streamId != k2.streamId
  AND k1.rowid < k2.rowid
JOIN StringIds s1 ON k1.demangledName = s1.id
JOIN StringIds s2 ON k2.demangledName = s2.id
ORDER BY overlap_ms DESC
LIMIT 20
```
