"""
export.py — Generate Perfetto-compatible JSON traces from profiles.

Combines kernel events + projected NVTX into chrome://tracing format.
"""
import json, os
from . import profile as _profile
from .projection import project_nvtx

STREAM_NAMES = {21: "Compute", 56: "NCCL"}


def gpu_trace(prof, gpu: int, trim: tuple[int, int]) -> list[dict]:
    """Build Perfetto trace events for one GPU in a time window."""
    kmap = prof.kernel_map(gpu)
    if not kmap:
        return []

    min_ts = min(k["start"] for k in kmap.values())
    events, streams = [], set()

    # Kernels
    for k in prof.kernels(gpu, trim):
        events.append(dict(name=k["name"], cat="gpu_kernel", ph="X",
            ts=(k["start"] - min_ts) / 1000, dur=(k["end"] - k["start"]) / 1000,
            pid=gpu, tid=k["streamId"], cname="thread_state_runnable"))
        streams.add(k["streamId"])

    # NVTX
    projected = project_nvtx(prof, gpu, trim)
    max_depth = 0
    for p in projected:
        if p["projected"]:
            events.append(dict(name=p["name"], cat="nvtx_projected", ph="X",
                ts=(p["start"] - min_ts) / 1000, dur=(p["end"] - p["start"]) / 1000,
                pid=gpu, tid=10 + p["depth"], cname="good"))
        max_depth = max(max_depth, p["depth"])

    # Metadata
    events.append(dict(name="process_name", ph="M", pid=gpu,
                       args=dict(name=f"GPU {gpu}")))
    for d in range(max_depth + 1):
        events += [dict(name="thread_name", ph="M", pid=gpu, tid=10+d,
                        args=dict(name=f"NVTX Lvl {d}")),
                   dict(name="thread_sort_index", ph="M", pid=gpu, tid=10+d,
                        args=dict(sort_index=10+d))]
    for s in sorted(streams):
        label = STREAM_NAMES.get(s, "Aux")
        events += [dict(name="thread_name", ph="M", pid=gpu, tid=s,
                        args=dict(name=f"Stream {s} ({label})")),
                   dict(name="thread_sort_index", ph="M", pid=gpu, tid=s,
                        args=dict(sort_index=50+s))]
    return events


def write_json(events: list[dict], path: str):
    """Write trace events to a Perfetto JSON file."""
    with open(path, "w") as f:
        json.dump(dict(traceEvents=events, displayTimeUnit="ms"), f)


def all_gpu_traces(db_path: str, trim: tuple[int, int], out_dir: str):
    """Generate per-GPU JSON traces for all active GPUs."""
    prof = _profile.open(db_path)
    os.makedirs(out_dir, exist_ok=True)

    for gpu in prof.meta.devices:
        events = gpu_trace(prof, gpu, trim)
        if not events:
            print(f"  GPU {gpu}: no kernels, skipped")
            continue
        out = os.path.join(out_dir, f"trace_gpu{gpu}.json")
        write_json(events, out)
        nk = sum(1 for e in events if e.get("cat") == "gpu_kernel")
        nn = sum(1 for e in events if e.get("cat") == "nvtx_projected")
        print(f"  GPU {gpu}: {nk} kernels, {nn} NVTX → {out} ({os.path.getsize(out)//1024} KB)")

    prof.close()
