# NVTX Annotations

> **Source:** [Nsight Systems User Guide — NVTX Trace](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#nvtx-trace)

---

## What is NVTX?

The **NVIDIA Tools Extension Library (NVTX)** lets you manually instrument your application with named ranges and markers. Nsight Systems collects these annotations and displays them on the timeline, providing context for GPU activity.

**Key insight:** NVTX is the primary way to add application-level context to profiles. Without NVTX, you see low-level kernel names; with NVTX, you see "forward pass" → "attention layer" → "matmul kernel".

Nsight Systems supports **NVTX 3.0**.

## Supported Features

### Domains
```c
nvtxDomainCreate()
nvtxDomainDestroy()
nvtxDomainRegisterString()
```
Domains allow separate namespacing of annotations (e.g., "training", "data_loading").

### Push-Pop Ranges (Nested, Same Thread)
```c
nvtxRangePush("label")       // or nvtxRangePushEx()
nvtxRangePop()
nvtxDomainRangePushEx()
nvtxDomainRangePop()
```

### Start-End Ranges (Global, Cross-Thread)
```c
nvtxRangeStart()             // or nvtxRangeStartEx()
nvtxRangeEnd()
nvtxDomainRangeStartEx()
nvtxDomainRangeEnd()
```

### Marks (Instant Events)
```c
nvtxMark("event")            // or nvtxMarkEx()
nvtxDomainMarkEx()
```

### Thread Names
```c
nvtxNameOsThread()
```

### Categories
```c
nvtxNameCategory()
nvtxDomainNameCategory()
```

## Usage in C/C++

```c
#include "nvtx3/nvToolsExt.h"

void train_step() {
    nvtxRangePush("forward");
    forward();
    nvtxRangePop();

    nvtxRangePush("backward");
    backward();
    nvtxRangePop();
}
```

Compile with: `-ldl`

## Usage in Python

### Using the `nvtx` pip package:
```python
import nvtx

@nvtx.annotate("forward_pass", color="blue")
def forward(model, data):
    return model(data)

# Or as context manager:
with nvtx.annotate("training_step"):
    output = model(data)
    loss = criterion(output, target)
```

### Using PyTorch's built-in NVTX:
```python
import torch.cuda.nvtx as nvtx

nvtx.range_push("forward")
output = model(data)
nvtx.range_pop()

nvtx.range_push("backward")
loss.backward()
nvtx.range_pop()
```

## NVTX as Capture Range Triggers

NVTX can be used to start/stop profiling at specific points:

```bash
nsys profile -c nvtx -p MESSAGE@DOMAIN <app>
```

This is the recommended way to profile specific training iterations.

## Best Practices

1. **Use registered strings** — enables more performant matching:
   ```c
   nvtxStringHandle_t handle = nvtxDomainRegisterStringA(domain, "my_range");
   ```

2. **RAII wrappers in C++** — guarantee ranges are closed:
   ```cpp
   class NvtxRange {
       public:
           NvtxRange(const char* name) { nvtxRangePush(name); }
           ~NvtxRange() { nvtxRangePop(); }
   };
   ```

3. **Leave NVTX calls in production** — overhead is negligible when profiler is not attached

4. **Use domains** to separate annotations from different subsystems

5. **Use categories** within domains to group related annotations

## NVTX Event Types in SQLite

| eventType | Meaning |
|-----------|---------|
| 33 | Category registration |
| 34 | Mark (instant) |
| 59 | Push/Pop range |
| 60 | Start/End range |
| 70 | Push/Pop range imported from NVTXT |
| 71 | Start/End range imported from NVTXT |
| 75 | Domain creation |

A profile that exports these lists them in its own `ENUM_NSYS_EVENT_TYPE`
table, which is the authoritative source for the export you are holding.

## What nsys-ai attributes

Mapping a GPU kernel to the annotation it ran under uses **Push/Pop ranges
(eventType 59) only**.

Nsight keeps a push/pop range stack per CPU thread, so a thread's push/pop
ranges are strictly nested by construction. That is what makes kernel
attribution a single ordered sweep: walk one thread's ranges in start order,
keep the open ones on a stack, and the innermost still-open range at a
kernel's launch is that kernel's label.

Start/End ranges carry no such guarantee. NVIDIA documents that they "expose
arbitrary concurrency (not just nesting)" and that "the start of a range can
occur on a different thread than the end" — Nsight records that as one row
whose `globalTid` and `endGlobalTid` differ. Handing those to a nesting stack
would produce confident, wrong parents, so nsys-ai does not attribute them.
The same applies to the NVTXT-imported variants (70 and 71).

On a profile with no push/pop ranges at all, the skills that depend on
attribution — `nvtx_kernel_map`, `nvtx_layer_breakdown`,
`code_attribution_candidates`, `nccl_compile_context_breakdown` — abstain and
say why, rather than returning an empty result that reads as "this profile has
no annotation". An entirely Start/End profile gets a reason naming that; any
other case, including an NVTXT import, gets a reason listing the eventTypes
actually present, read from the profile's own `ENUM_NSYS_EVENT_TYPE`. To get
attribution, annotate with `nvtxRangePush` / `nvtxRangePop`
(`torch.cuda.nvtx.range_push` / `range_pop`, or the `torch.cuda.nvtx.range`
context manager). PyTorch's own annotations already use push/pop.

Region *listing* is unaffected and does see Start/End ranges: the NVTX tree,
`region_mfu`, iteration detection, `host_sync_parent_ranges` and `gc_impact`
all read NVTX rows as plain intervals.

One limitation worth stating: on a profile that mixes both kinds, the
attribution skills run on the push/pop ranges and the Start/End ranges are
silently left out.
