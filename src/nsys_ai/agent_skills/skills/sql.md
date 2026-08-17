# Skill: SQL Reference

Read this when you need to query the profile database, or for help with a specific query.

---

## Guardrails (enforced by `query_profile_db`)

- **SELECT only** — INSERT / UPDATE / DELETE / DROP all rejected
- **No `SELECT *`** — always name the columns you need
- **LIMIT auto-enforced** — max 50 rows; narrow 1-column queries get up to 100
- **Max output**: 8000 chars; will be truncated with a message asking you to narrow

---

## SQLite Dialect Rules

| Rule | Correct | Wrong |
|------|---------|-------|
| Reserved word `end` | `k.[end]` | `k.end` |
| String concat | `a \|\| b` | `CONCAT(a, b)` |
| Date functions | `strftime(...)` | `DATE_TRUNC`, `EXTRACT` |
| Time unit | nanoseconds | ms or seconds (divide yourself) |
| Rounding | `ROUND(x, 2)` | float arithmetic |

**Time conversions**:
- `ns → ms`: divide by `1e6`
- `ns → s`: divide by `1e9`
- `total GPU time`: `SUM([end]-start)/1e6`
- `wall span`: `(MAX([end]) - MIN(start)) / 1e9`

---

## Key Tables

| Table | Key Columns | Notes |
|-------|-------------|-------|
| `CUPTI_ACTIVITY_KIND_KERNEL` | `start`, `[end]`, `shortName`(int), `correlationId`, `deviceId`, `streamId` | Primary kernel table |
| `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` | Same schema | Older profiles use this instead |
| `NVTX_EVENTS` | `start`, `[end]`, `text` (or `textId`→StringIds), `globalTid` | Use COALESCE when textId present |
| `StringIds` | `id`(int), `value`(string) | Maps integer IDs to human-readable names |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | `correlationId`, `start`, `[end]`, `globalTid` | CPU-side CUDA API; links to kernel via correlationId |
| `TARGET_INFO_GPU` | GPU hardware info | GPU name, compute capability, etc. |
| `CUDA_GPU_MEMORY_USAGE_EVENTS` | Memory alloc/free | Only if `--cuda-memory-usage=true` was used |
| `CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR` | Source file + line | Only if source profiling enabled |

**Join pattern for kernel names** (always required):
```sql
JOIN StringIds s ON k.shortName = s.id
```

**Join pattern for NVTX text** (handle both old/new schema):
```sql
COALESCE(n.text, s.value) AS name
FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
```

---

> [!IMPORTANT]
> **For profiles >100 MB, always add a time-range WHERE clause.**
> Without it, queries scan millions of rows and can take 30–90 minutes.
> Use: `WHERE k.start >= <start_ns> AND k.[end] <= <end_ns>`
> See "Kernels in a specific time window" below.

## Common Queries

### Profile overview
```sql
SELECT (MAX([end]) - MIN(start)) / 1e9 AS span_s,
       COUNT(*) AS n_kernels
FROM CUPTI_ACTIVITY_KIND_KERNEL
```

### Top kernels by total GPU time
```sql
SELECT s.value AS name, COUNT(*) AS cnt,
       SUM(k.[end]-k.start)/1e6 AS total_ms,
       ROUND(100.0*SUM(k.[end]-k.start) /
         (SELECT MAX([end])-MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL), 1) AS pct
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
GROUP BY k.shortName ORDER BY total_ms DESC LIMIT 15
```

### List NVTX annotation names
```sql
SELECT DISTINCT COALESCE(n.text, s.value) AS name
FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
WHERE COALESCE(n.text, s.value) IS NOT NULL ORDER BY name LIMIT 30
```

### NCCL time and breakdown
```sql
SELECT s.value AS op, COUNT(*) AS cnt,
       SUM(k.[end]-k.start)/1e6 AS total_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
WHERE LOWER(s.value) LIKE '%nccl%'
GROUP BY op ORDER BY total_ms DESC LIMIT 10
```

### Per-iteration timing (from NVTX)
```sql
SELECT COALESCE(n.text, s.value) AS name,
       ([end]-start)/1e6 AS duration_ms, start/1e9 AS start_s
FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
WHERE COALESCE(n.text, s.value) LIKE '%sample%'
ORDER BY start LIMIT 20
```

### Kernels in a specific time window
```sql
SELECT s.value AS name, COUNT(*) AS cnt,
       SUM(k.[end]-k.start)/1e6 AS ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
WHERE k.start >= <start_ns> AND k.[end] <= <end_ns>
GROUP BY k.shortName ORDER BY ms DESC LIMIT 20
```

---

## Schema Discovery

> **⚠️ PRAGMA is blocked by `query_profile_db`.** The read-only guard rejects PRAGMA
> statements. Use `DESCRIBE <table>` instead (it is automatically rewritten to
> `PRAGMA table_info('<table_name>')` internally). Or use the `schema_inspect` builtin skill.

```sql
-- All tables in this profile
SHOW TABLES

-- Columns in a table (use DESCRIBE, not PRAGMA)
DESCRIBE CUPTI_ACTIVITY_KIND_KERNEL

-- Check if NVTX uses textId (newer) or text (older)
DESCRIBE NVTX_EVENTS
-- If 'textId' column present → JOIN StringIds; else use n.text directly
```

> **Note**: DuckDB profiles use `SHOW TABLES` and `DESCRIBE` natively.
> The `schema_inspect` skill handles both SQLite and DuckDB backends automatically.
> Example usage: `nsys-ai skill run schema_inspect profile.sqlite` for discovery.
