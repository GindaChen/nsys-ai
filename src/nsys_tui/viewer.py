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


def generate_timeline_data_json(prof, devices, trim: tuple[int, int]) -> str:
    """Return JSON string of per-GPU kernel/NVTX data for a time window.

    Called by the ``/api/data`` endpoint for on-demand tile loading.
    """
    from typing import Sequence
    if not isinstance(devices, Sequence):
        devices = [devices]

    gpu_entries = []
    for dev in devices:
        roots = build_nvtx_tree(prof, dev, trim)
        tree_json = to_json(roots)
        gpu_entries.append({"id": dev, "data": tree_json})

    return json.dumps({"gpus": gpu_entries})


def generate_timeline_html(prof, device, trim: tuple[int, int] | None = None) -> str:
    """Generate a standalone HTML page with the horizontal timeline viewer.

    *device* may be a single int or a list of ints.
    When *trim* is None, HTML is generated in progressive mode: ``$DATA``
    is ``null`` and the template fetches data via ``/api/data`` on demand.
    """
    from typing import Sequence
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
        # Full data baked into HTML
        gpu_entries = []
        for dev in devices:
            roots = build_nvtx_tree(prof, dev, trim)
            tree_json = to_json(roots)
            gpu_entries.append({"id": dev, "data": tree_json})

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
