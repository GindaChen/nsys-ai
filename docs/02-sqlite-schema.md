# SQLite Export Schema & Common Queries

> **Source:** [Common SQLite examples — Nsight Systems 2022.4](https://docs.nvidia.com/nsight-systems/2022.4/nsys-exporter/examples.html)
>
> ⚠️ **VERSION WARNING:** This schema is from Nsight Systems **2022.4**. Table names, column names, and available data may differ in other versions. Always verify schema with:
> ```sql
> SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;
> ```
> and check column names with `.schema TABLE_NAME` in the sqlite3 CLI.

---

## Export Pipeline

```bash
# Step 1: Profile
nsys profile --trace=cuda,nvtx,osrt -o report <app>

# Step 2: Export to SQLite
nsys export --type sqlite report.nsys-rep -o report.sqlite

# Step 3: Query
sqlite3 report.sqlite
```

## Key Tables

| Table | Description |
|-------|-------------|
| `CUPTI_ACTIVITY_KIND_KERNEL` | GPU kernel executions (name, start, end, grid, block, stream) |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | CUDA Runtime API calls (cudaLaunch, cudaMalloc, etc.) |
| `CUPTI_ACTIVITY_KIND_MEMCPY` | Memory copy operations (H2D, D2H, D2D, P2P) |
| `CUPTI_ACTIVITY_KIND_MEMSET` | Memory set operations |
| `NVTX_EVENTS` | NVTX annotations (push/pop ranges, marks, domains, categories) |
| `StringIds` | String lookup table — kernel names, API names, NVTX text are stored as IDs |
| `OSRT_API` | OS runtime library calls (pthread, mutex, etc.) |
| `SCHED_EVENTS` | CPU thread scheduling events |
| `COMPOSITE_EVENTS` | Aggregated sampling events |
| `SAMPLING_CALLCHAINS` | CPU sampling call stacks |
| `ThreadNames` | Thread name to globalTid mapping |
| `ProcessStreams` | Captured stdout/stderr from profiled processes |
| `PROFILER_OVERHEAD` | Ranges where profiler overhead is significant |
| `CUDA_CALLCHAINS` | CUDA API call stacks (when `--cudabacktrace` is enabled) |
| `OSRT_CALLCHAINS` | OS runtime call stacks |
| `CUDA_GRAPH_EVENTS` | CUDA graph node creation/execution events |
| `FECS_EVENTS` | GPU context switch events |
| `DX12_API` | DirectX 12 API calls (Windows only) |

## The StringIds Pattern

**Critical:** Kernel names, API names, and NVTX text are NOT stored inline. They are stored in the `StringIds` table and referenced by ID.

```sql
-- Resolve a kernel name
SELECT s.value FROM StringIds s
JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.demangledName = s.id;

-- Resolve a CUDA API name
SELECT s.value FROM StringIds s
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.nameId = s.id;
```

## The globalTid Pattern

Process and thread IDs are encoded in a single `globalTid` integer:

```sql
-- Extract PID and TID from globalTid
SELECT globalTid / 0x1000000 % 0x1000000 AS PID,
       globalTid % 0x1000000 AS TID
FROM TABLE_NAME;
```

## Common Queries

### Top GPU Kernels by Total Time

```sql
SELECT s.value AS kernel_name,
       COUNT(*) AS invocations,
       SUM(k.end - k.start) / 1e6 AS total_ms,
       AVG(k.end - k.start) / 1e6 AS avg_ms,
       MIN(k.end - k.start) / 1e6 AS min_ms,
       MAX(k.end - k.start) / 1e6 AS max_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
GROUP BY s.value
ORDER BY total_ms DESC
LIMIT 15;
```

### Memory Transfers Summary

```sql
SELECT copyKind,
       COUNT(*) AS count,
       SUM(bytes) / 1e6 AS total_mb,
       SUM(end - start) / 1e6 AS total_ms
FROM CUPTI_ACTIVITY_KIND_MEMCPY
GROUP BY copyKind;
-- copyKind: 1=H2D, 2=D2H, 8=D2D, 10=P2P
```

### Correlate CUDA Kernel Launches with API Calls

```sql
ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN name TEXT;
ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN kernelName TEXT;

UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET kernelName = (
    SELECT value FROM StringIds
    JOIN CUPTI_ACTIVITY_KIND_KERNEL AS k ON k.shortName = StringIds.id
    AND CUPTI_ACTIVITY_KIND_RUNTIME.correlationId = k.correlationId
);
UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET name = (
    SELECT value FROM StringIds WHERE nameId = StringIds.id
);

-- Top 10 longest CUDA API calls that launched kernels
SELECT name, kernelName, start, end
FROM CUPTI_ACTIVITY_KIND_RUNTIME
WHERE kernelName IS NOT NULL
ORDER BY end - start DESC
LIMIT 10;
```

### Map NVTX Ranges to CUDA Kernels

```sql
ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN nvtxRange TEXT;
CREATE INDEX nvtx_start ON NVTX_EVENTS (start);

UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET nvtxRange = (
    SELECT NVTX_EVENTS.text
    FROM NVTX_EVENTS
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME
    ON NVTX_EVENTS.eventType == 59
    AND NVTX_EVENTS.globalTid == CUPTI_ACTIVITY_KIND_RUNTIME.globalTid
    AND NVTX_EVENTS.start <= CUPTI_ACTIVITY_KIND_RUNTIME.start
    AND NVTX_EVENTS.end >= CUPTI_ACTIVITY_KIND_RUNTIME.end
    WHERE CUPTI_ACTIVITY_KIND_KERNEL.correlationId == CUPTI_ACTIVITY_KIND_RUNTIME.correlationId
    ORDER BY NVTX_EVENTS.start DESC
    LIMIT 1
);

SELECT start, end, StringIds.value as kernelName, nvtxRange
FROM CUPTI_ACTIVITY_KIND_KERNEL
JOIN StringIds ON shortName == id
ORDER BY start LIMIT 10;
```

### Resolve NVTX Category Names

```sql
WITH event AS (
    SELECT * FROM NVTX_EVENTS
    WHERE eventType IN (34, 59, 60)  -- mark, push/pop, start/end
),
category AS (
    SELECT category, domainId, text AS categoryName
    FROM NVTX_EVENTS WHERE eventType == 33  -- new category
)
SELECT start, end, globalTid, eventType, domainId,
       category, categoryName, text
FROM event
JOIN category USING (category, domainId)
ORDER BY start;
```

### Thread Summary (CPU Utilization)

```sql
SELECT globalTid / 0x1000000 % 0x1000000 AS PID,
       globalTid % 0x1000000 AS TID,
       ROUND(100.0 * SUM(cpuCycles) / (
           SELECT SUM(cpuCycles) FROM COMPOSITE_EVENTS
           GROUP BY globalTid / 0x1000000000000 % 0x100
       ), 2) as CPU_utilization,
       (SELECT value FROM StringIds WHERE id = (
           SELECT nameId FROM ThreadNames
           WHERE ThreadNames.globalTid = COMPOSITE_EVENTS.globalTid
       )) as thread_name
FROM COMPOSITE_EVENTS
GROUP BY globalTid
ORDER BY CPU_utilization DESC
LIMIT 10;
```

### Remove Profiler Overhead

```sql
-- Count CUDA API ranges overlapping with profiler overhead
-- Replace "SELECT COUNT(*)" with "DELETE" to remove them
SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME
WHERE rowid IN (
    SELECT cuda.rowid
    FROM PROFILER_OVERHEAD AS overhead
    INNER JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS cuda
    ON (cuda.start BETWEEN overhead.start AND overhead.end)
    OR (cuda.end BETWEEN overhead.start AND overhead.end)
    OR (cuda.start < overhead.start AND cuda.end > overhead.end)
);
```

### CUDA Calls with Backtraces

```sql
ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN name TEXT;
UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET name = (
    SELECT value FROM StringIds WHERE CUPTI_ACTIVITY_KIND_RUNTIME.nameId = StringIds.id
);

ALTER TABLE CUDA_CALLCHAINS ADD COLUMN symbolName TEXT;
UPDATE CUDA_CALLCHAINS SET symbolName = (
    SELECT value FROM StringIds WHERE symbol = StringIds.id
);

SELECT globalTid % 0x1000000 AS TID, start, end, name,
       callchainId, stackDepth, symbolName
FROM CUDA_CALLCHAINS
JOIN CUPTI_ACTIVITY_KIND_RUNTIME ON callchainId == CUDA_CALLCHAINS.id
ORDER BY callchainId, stackDepth LIMIT 20;
```

### Process stdout/stderr

```sql
ALTER TABLE ProcessStreams ADD COLUMN filename TEXT;
UPDATE ProcessStreams SET filename = (
    SELECT value FROM StringIds WHERE ProcessStreams.filenameId = StringIds.id
);

ALTER TABLE ProcessStreams ADD COLUMN content TEXT;
UPDATE ProcessStreams SET content = (
    SELECT value FROM StringIds WHERE ProcessStreams.contentId = StringIds.id
);

SELECT globalPid / 0x1000000 % 0x1000000 AS PID, filename, content
FROM ProcessStreams;
```

## NVTX Event Types

| eventType | Meaning |
|-----------|---------|
| 33 | Category registration |
| 34 | Mark (instant event) |
| 59 | Push/Pop range |
| 60 | Start/End range |

## Memory Copy Kinds

| copyKind | Direction |
|----------|-----------|
| 1 | Host → Device (H2D) |
| 2 | Device → Host (D2H) |
| 8 | Device → Device (D2D) |
| 10 | Peer-to-Peer (P2P) |

## Tips for Programmatic Analysis

1. **Always create indexes** before running complex queries — the exported SQLite has no indexes by default
2. **Use `ALTER TABLE ... ADD COLUMN`** to denormalize string lookups for readability
3. **Timestamps are in nanoseconds** — divide by 1e6 for milliseconds, 1e9 for seconds
4. **The `correlationId`** field links CUDA API calls to their GPU kernel executions
