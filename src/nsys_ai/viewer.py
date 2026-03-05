"""
viewer.py - Generate interactive HTML visualizations for Nsight profiles.

Uses string.Template with HTML template files for clean separation between
Python logic and HTML/CSS/JS presentation.
"""
import json
import os
from string import Template

from .tree import build_nvtx_tree, to_json

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _load_template(name: str) -> Template:
    """Load an HTML template from the templates directory."""
    path = os.path.join(_TEMPLATE_DIR, name)
    with open(path, encoding="utf-8") as f:
        return Template(f.read())


def generate_html(prof, device: int, trim: tuple[int, int]) -> str:
    """Generate a standalone HTML page showing the NVTX stack trace."""
    roots = build_nvtx_tree(prof, device, trim)
    tree_json = to_json(roots)

    gpu_info = prof.meta.gpu_info.get(device)
    gpu_label = f"GPU {device}"
    if gpu_info:
        gpu_label += (f" - {gpu_info.name} ({gpu_info.pci_bus}), "
                      f"{gpu_info.sm_count} SMs, {gpu_info.memory_bytes/1e9:.0f}GB")

    # Stable id for this profile view (device + time window) for profile-bound chat history
    trim_sec = (trim[0] / 1e9, trim[1] / 1e9)
    profile_id = f"{device}_{trim_sec[0]:.1f}_{trim_sec[1]:.1f}"

    tmpl = _load_template("nvtx_tree.html")
    db_agent_flag = os.environ.get("NSYS_AI_DB_AGENT", "").strip().lower()
    db_agent_enabled = bool(db_agent_flag) and db_agent_flag not in ("0", "false", "no", "off")
    return tmpl.safe_substitute(
        DATA=json.dumps(tree_json),
        GPU_LABEL=gpu_label,
        TRIM_LABEL=f"{trim[0]/1e9:.1f}s - {trim[1]/1e9:.1f}s",
        PROFILE_ID=profile_id,
        PROFILE_PATH=prof.path,
        DB_AGENT_ENABLED="1" if db_agent_enabled else "",
    )


def write_html(prof, device: int, trim: tuple[int, int], path: str):
    """Generate and write the HTML viewer to a file."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(generate_html(prof, device, trim))


def _collect_nvtx_annotations(
    nodes: list[dict],
    spans: list[dict],
    kernel_paths: dict[tuple, str],
    current_thread: str = "",
) -> None:
    """Collect flat NVTX spans and kernel-path annotations from a tree JSON."""
    for node in nodes:
        thread_name = node.get("thread_name") or current_thread
        ntype = node.get("type")
        if ntype == "nvtx":
            path = node.get("path", "")
            depth = max(len(path.split(" > ")) - 1, 0) if path else 0
            spans.append({
                "name": node.get("name", ""),
                "start": node.get("start_ns", 0),
                "end": node.get("end_ns", 0),
                "depth": depth,
                "path": path,
                "dur": node.get("duration_ms", 0),
                "thread": thread_name or "(unnamed)",
            })
        elif ntype == "kernel":
            key = (
                node.get("start_ns"),
                node.get("end_ns"),
                node.get("stream"),
                node.get("name"),
            )
            kernel_paths[key] = node.get("path", "")

        children = node.get("children") or []
        if children:
            _collect_nvtx_annotations(children, spans, kernel_paths, thread_name)


def build_timeline_gpu_data(
    prof,
    device,
    trim: tuple[int, int],
    *,
    include_kernels: bool = True,
    include_nvtx: bool = True,
) -> list[dict]:
    """Build per-GPU timeline payload with kernel rows plus optional NVTX annotations."""
    from collections.abc import Sequence

    from .nvtx_tree import build_nvtx_tree as build_nvtx_tree_all_threads
    from .nvtx_tree import to_json as nvtx_to_json

    devices: list[int] = list(device) if isinstance(device, Sequence) else [device]
    gpu_entries: list[dict] = []

    for dev in devices:
        kernels = []
        if include_kernels:
            # 1) Kernel-first: authoritative source for timeline rows.
            sql = f"""
                SELECT k.start AS start_ns, k.[end] AS end_ns, k.streamId AS stream,
                       s.value AS name
                FROM {prof.schema.kernel_table} k
                JOIN StringIds s ON k.shortName = s.id
                WHERE k.deviceId = ? AND k.[end] >= ? AND k.start <= ?
                ORDER BY k.start
            """
            with prof._lock:
                rows = prof.conn.execute(sql, (dev, trim[0], trim[1])).fetchall()

            for r in rows:
                start_ns = int(r["start_ns"])
                end_ns = int(r["end_ns"])
                kernels.append({
                    "type": "kernel",
                    "name": r["name"],
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "duration_ms": round((end_ns - start_ns) / 1e6, 3),
                    "stream": r["stream"],
                    "path": "",
                })

        # 2) NVTX-only annotations + kernel->path labels.
        #    NVTX is advisory metadata; missing mapping must not drop kernels.
        nvtx_spans: list[dict] = []
        kernel_paths: dict[tuple, str] = {}
        if include_nvtx:
            try:
                roots = build_nvtx_tree_all_threads(prof, dev, trim)
                tree_json = nvtx_to_json(roots)
            except Exception:
                tree_json = []
            _collect_nvtx_annotations(tree_json, nvtx_spans, kernel_paths)

        if include_kernels:
            for k in kernels:
                key = (k["start_ns"], k["end_ns"], k["stream"], k["name"])
                k["path"] = kernel_paths.get(key, k["name"])

        gpu_entries.append({"id": dev, "kernels": kernels, "nvtx_spans": nvtx_spans})

    return gpu_entries


def generate_timeline_data_json(prof, devices, trim: tuple[int, int]) -> str:
    """Return JSON string of per-GPU kernel/NVTX data for a time window.

    Called by the ``/api/data`` endpoint for on-demand tile loading.
    """
    gpu_entries = build_timeline_gpu_data(prof, devices, trim)
    return json.dumps({"gpus": gpu_entries})


def generate_timeline_html(prof, device, trim: tuple[int, int] | None = None) -> str:
    """Generate a standalone HTML page with the horizontal timeline viewer.

    *device* may be a single int or a list of ints.
    When *trim* is None, HTML is generated in progressive mode: ``$DATA``
    is ``null`` and the template fetches data via ``/api/data`` on demand.
    """
    from collections.abc import Sequence
    devices: list[int] = list(device) if isinstance(device, Sequence) else [device]

    # Build GPU info list (for dropdown) and compact label
    gpu_details = []
    gpu_type = "GPU"
    for dev in devices:
        gpu_info = prof.meta.gpu_info.get(dev)
        detail = {"id": dev, "name": "Unknown", "pci": "", "sms": 0, "mem_gb": 0}
        if gpu_info:
            detail["name"] = gpu_info.name
            detail["pci"] = gpu_info.pci_bus
            detail["sms"] = gpu_info.sm_count
            detail["mem_gb"] = round(gpu_info.memory_bytes / 1e9)
            gpu_type = gpu_info.name
        gpu_details.append(detail)
    gpu_info_json = json.dumps(gpu_details)
    gpu_label = f"{len(devices)}× {gpu_type}" if len(devices) > 1 else gpu_type

    if trim is not None:
        # Full data baked into HTML (kernel-first payload).
        gpu_entries = build_timeline_gpu_data(prof, devices, trim)
        data_json = json.dumps({"gpus": gpu_entries})
        trim_label = f"{trim[0]/1e9:.1f}s - {trim[1]/1e9:.1f}s"
        progressive = ""
    else:
        # Progressive mode: no data baked in
        data_json = "null"
        trim_label = "Progressive"
        progressive = "1"

    tmpl = _load_template("timeline.html")
    return tmpl.safe_substitute(
        DATA=data_json,
        GPU_LABEL=gpu_label,
        GPU_INFO_JSON=gpu_info_json,
        TRIM_LABEL=trim_label,
        PROGRESSIVE=progressive,
    )


def write_timeline_html(prof, device: int, trim: tuple[int, int], path: str):
    """Generate and write the timeline HTML viewer to a file."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(generate_timeline_html(prof, device, trim))
