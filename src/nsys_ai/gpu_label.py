"""Consistent human-readable labels for GPU metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def format_gpu_label(device: int, info: Any = None) -> str:
    """Format a GPU label without inventing values for missing metadata."""
    label = f"GPU {device}"
    if info is None:
        return label

    if isinstance(info, Mapping):
        get = info.get
        name = str(get("name", "") or "").strip()
        pci_bus = str(get("pci_bus", "") or "").strip()
        sm_count = get("sm_count", 0) or 0
        memory_gb = get("memory_gb", 0) or 0
    else:
        def get(key, default=None):
            return getattr(info, key, default)

        name = str(get("name", "") or "").strip()
        pci_bus = str(get("pci_bus", "") or "").strip()
        sm_count = get("sm_count", 0) or 0
        memory_gb = (get("memory_bytes", 0) or 0) / 1e9

    if name or pci_bus:
        detail = name
        if pci_bus:
            detail = f"{detail} ({pci_bus})" if detail else f"({pci_bus})"
        label += f" - {detail}"
    if sm_count:
        label += f", {sm_count} SMs"
    if memory_gb:
        memory_text = f"{memory_gb:.1f}".rstrip("0").rstrip(".")
        label += f", {memory_text}GB"
    return label


def format_gpu_narrative_label(device: int, info: Any = None) -> str:
    """Format the short GPU identity used at the start of prose sentences."""
    label = f"GPU {device}"
    if isinstance(info, Mapping):
        name = str(info.get("name", "") or "").strip()
    else:
        name = str(getattr(info, "name", "") or "").strip() if info is not None else ""
    return f"{label} ({name})" if name else label
