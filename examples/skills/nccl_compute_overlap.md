# nccl_compute_overlap

## Description

Measures what fraction of NCCL communication time has concurrent compute
running on other streams. Low overlap (< 10%) means communication is fully
serialized with compute — a prime optimization target. After enabling
overlap_comm_compute, this metric should jump to > 60%.

## Category

communication

## SQL

```sql
WITH nccl AS (
    SELECT k.start, k.[end], k.streamId
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.shortName = s.id
    WHERE s.value LIKE '%nccl%' OR s.value LIKE '%NCCL%'
),
compute AS (
    SELECT k.start, k.[end], k.streamId
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.shortName = s.id
    WHERE s.value NOT LIKE '%nccl%' AND s.value NOT LIKE '%NCCL%'
),
overlap AS (
    SELECT n.start AS nccl_start, n.[end] AS nccl_end,
           MAX(0, MIN(n.[end], c.[end]) - MAX(n.start, c.start)) AS overlap_ns
    FROM nccl n
    LEFT JOIN compute c
      ON c.start < n.[end] AND c.[end] > n.start
      AND c.streamId != n.streamId
)
SELECT
    COUNT(DISTINCT nccl_start) AS nccl_ops,
    ROUND(SUM(nccl_end - nccl_start) / 1e6, 2) AS total_nccl_ms,
    ROUND(SUM(overlap_ns) / 1e6, 2) AS overlapped_ms,
    ROUND(100.0 * SUM(overlap_ns) / NULLIF(SUM(nccl_end - nccl_start), 0), 1) AS overlap_pct
FROM overlap
```
