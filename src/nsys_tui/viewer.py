"""
viewer.py — Generate interactive HTML visualizations for Nsight profiles.

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
        gpu_label += (f" — {gpu_info.name} ({gpu_info.pci_bus}), "
                      f"{gpu_info.sm_count} SMs, {gpu_info.memory_bytes/1e9:.0f}GB")

    tmpl = _load_template("nvtx_tree.html")
    return tmpl.safe_substitute(
        DATA=json.dumps(tree_json),
        GPU_LABEL=gpu_label,
        TRIM_LABEL=f"{trim[0]/1e9:.1f}s – {trim[1]/1e9:.1f}s",
    )


def write_html(prof, device: int, trim: tuple[int, int], path: str):
    """Generate and write the HTML viewer to a file."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(generate_html(prof, device, trim))


def generate_timeline_html(prof, device: int, trim: tuple[int, int]) -> str:
    """Generate a standalone HTML page with the horizontal timeline viewer."""
    roots = build_nvtx_tree(prof, device, trim)
    tree_json = to_json(roots)

    gpu_info = prof.meta.gpu_info.get(device)
    gpu_label = f"GPU {device}"
    if gpu_info:
        gpu_label += (f" — {gpu_info.name} ({gpu_info.pci_bus}), "
                      f"{gpu_info.sm_count} SMs, {gpu_info.memory_bytes/1e9:.0f}GB")

    tmpl = _load_template("timeline.html")
    return tmpl.safe_substitute(
        DATA=json.dumps(tree_json),
        GPU_LABEL=gpu_label,
        TRIM_LABEL=f"{trim[0]/1e9:.1f}s – {trim[1]/1e9:.1f}s",
    )


def write_timeline_html(prof, device: int, trim: tuple[int, int], path: str):
    """Generate and write the timeline HTML viewer to a file."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(generate_timeline_html(prof, device, trim))
