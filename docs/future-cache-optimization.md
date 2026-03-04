# Timeline Cache — Future Optimization Ideas

> Current: single `.timeline-cache.json` (~13MB for 7 GPUs, ~16s build, 0.06s load)

## Scaling Problems at 64+ GPUs

- **Single JSON file** becomes unwieldy at 64 GPUs (est. ~120MB)
- **Full in-memory load** of 120MB JSON on every startup
- **Re-build entire cache if any source file changes**

## Format Alternatives

| Format | Pros | Cons |
|--------|------|------|
| **JSON** (current) | Simple, debuggable | Slow parse at scale, large |
| **JSONL (per GPU)** | Incremental load, partial cache | Still text-based |
| **MessagePack** | Compact binary, fast parse | Needs dependency |
| **SQLite** | Indexed queries, partial load | Schema overhead |
| **Pickle** | Native Python, fastest load | Not portable, security |
| **Parquet/Arrow** | Columnar, zero-copy mmap | Heavy dependency |

## Architecture Ideas

1. **Per-GPU cache files**: `<profile>.cache/gpu-1.json`, `gpu-2.json`, etc.
   - Load only requested GPUs
   - Parallel load
   - Partial invalidation

2. **Lazy/streaming build**: Build GPU trees in background threads, serve each GPU as it becomes ready
   - Frontend shows "GPU 1 ready" → "GPU 2 ready" incrementally

3. **Pre-built at profile time**: `nsys-ai prebuild <profile>` writes cache, then `timeline-web` always uses it

4. **Memory-mapped binary**: Use Arrow or a flat binary format that can be mmap'd without parsing

5. **Accelerated tree building**: Cython/Rust extension for the `build_nvtx_tree` hot path (currently ~2.3s/GPU in Python)

## For 8× nsys-rep files (multi-node)

- Need a manifest/index that maps node → GPUs → cache files
- Each node's cache is independent, can be built in parallel
- Frontend needs multi-file awareness (currently assumes single profile)
