"""
projection.py — Project NVTX CPU annotations onto GPU timelines.

The core algorithm:
1. For each NVTX range on a CPU thread, find CUDA runtime calls within it.
2. Look up correlated GPU kernels (filtered to the target device).
3. The projected span = [min(kernel_starts), max(kernel_ends)].
4. Track nesting depth for proper Perfetto layering.
"""
from typing import Optional


def _compute_depth(stacks: dict, tid: int, start: int, end: int) -> int:
    """Track NVTX nesting depth per thread."""
    if tid not in stacks:
        stacks[tid] = []
    stk = stacks[tid]
    while stk and stk[-1] <= start:
        stk.pop()
    depth = len(stk)
    stk.append(end)
    stk.sort(reverse=True)
    return depth


def project_nvtx(profile, device: int,
                 trim: tuple[int, int],
                 pad: int = int(5e9)) -> list[dict]:
    """
    Project NVTX annotations onto a GPU's timeline.

    Returns a list of dicts with keys:
        name, start, end, depth, projected (bool)

    Args:
        profile: An opened Profile instance.
        device:  Target GPU deviceId.
        trim:    (start_ns, end_ns) output window.
        pad:     Extra ns to look back for NVTX/runtime calls.
    """
    kmap = profile.kernel_map(device)
    if not kmap:
        return []

    threads = profile.gpu_threads(device)
    rt_idx = profile.runtime_index(threads, (trim[0] - pad, trim[1] + int(2e9)))
    nvtx = profile.nvtx_events(threads, (trim[0] - pad, trim[1]))

    results = []
    stacks: dict = {}
    max_depth = 0

    for n in nvtx:
        text = n["text"]
        if not text:
            continue

        tid, cpu_start, cpu_end = n["globalTid"], n["start"], n["end"]
        depth = _compute_depth(stacks, tid, cpu_start, cpu_end)
        max_depth = max(max_depth, depth)

        # Correlate runtime calls -> GPU kernels on this device
        gs, ge = [], []
        for rt in rt_idx.get(tid, []):
            if rt["start"] > cpu_end:
                break
            if rt["start"] >= cpu_start and rt["end"] <= cpu_end:
                k = kmap.get(rt["correlationId"])
                if k:
                    gs.append(k["start"]); ge.append(k["end"])

        if gs:
            ps, pe = min(gs), max(ge)
            if pe >= trim[0] and ps <= trim[1]:
                results.append(dict(name=text, start=ps, end=pe,
                                    depth=depth, projected=True))
        else:
            if cpu_end >= trim[0] and cpu_start <= trim[1]:
                results.append(dict(name=text, start=cpu_start, end=cpu_end,
                                    depth=depth, projected=False))

    return results
