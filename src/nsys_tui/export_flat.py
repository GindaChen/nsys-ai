"""
export_flat.py — Export profile data as flat CSV/JSON for custom analysis.

Replaces Nsight's complex query interface with simple, script-friendly
formats that users can load in pandas, Excel, or custom tools.
"""
import csv
import json
import io
from typing import Optional
from .profile import Profile
from .tree import build_nvtx_tree


def _kernel_rows(prof: Profile, device: int,
                 trim: Optional[tuple[int, int]] = None,
                 include_nvtx_path: bool = True) -> list[dict]:
    """Build flat rows with optional NVTX path context."""
    kernels = prof.kernels(device, trim)

    # Build NVTX path index if requested
    nvtx_paths: dict[int, str] = {}  # kernel_start -> path
    if include_nvtx_path and trim:
        roots = build_nvtx_tree(prof, device, trim)
        _collect_paths(roots, [], nvtx_paths)

    rows = []
    for k in kernels:
        rows.append(dict(
            name=k["name"],
            start_ns=k["start"],
            end_ns=k["end"],
            duration_ms=round((k["end"] - k["start"]) / 1e6, 3),
            duration_us=round((k["end"] - k["start"]) / 1e3, 1),
            stream=k["streamId"],
            device=device,
            nvtx_path=nvtx_paths.get(k["start"], ""),
        ))
    return rows


def _collect_paths(nodes: list, path: list[str],
                   result: dict[int, str]):
    """Recursively collect kernel -> NVTX path mapping."""
    for node in nodes:
        current_path = path + [node["name"]] if node["type"] == "nvtx" else path
        if node["type"] == "kernel":
            result[node["start"]] = " > ".join(current_path)
        if node.get("children"):
            _collect_paths(node["children"], current_path, result)


def to_csv(prof: Profile, device: int,
           trim: Optional[tuple[int, int]] = None,
           output: Optional[str] = None) -> str:
    """
    Export kernel data as CSV.

    Columns: name, start_ns, end_ns, duration_ms, duration_us, stream, device, nvtx_path

    Args:
        prof: Opened profile
        device: GPU device ID
        trim: Optional (start_ns, end_ns) window
        output: Optional file path to write to. If None, returns the CSV string.

    Returns: CSV string (also writes to file if output specified)
    """
    rows = _kernel_rows(prof, device, trim)
    if not rows:
        return ""

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    content = buf.getvalue()

    if output:
        with open(output, "w") as f:
            f.write(content)

    return content


def to_json_flat(prof: Profile, device: int,
                 trim: Optional[tuple[int, int]] = None,
                 output: Optional[str] = None) -> list[dict]:
    """
    Export kernel data as a flat JSON list.

    Each item: {name, start_ns, end_ns, duration_ms, duration_us, stream, device, nvtx_path}

    Returns: list of dicts (also writes to file if output specified)
    """
    rows = _kernel_rows(prof, device, trim)

    if output:
        with open(output, "w") as f:
            json.dump(rows, f, indent=2)

    return rows


def to_summary_json(prof: Profile, device: int,
                    trim: Optional[tuple[int, int]] = None,
                    output: Optional[str] = None) -> dict:
    """
    Export a structured summary JSON combining hardware, kernels, and timing.

    Reuses gpu_summary for the core data, adds export metadata.
    """
    from .summary import gpu_summary

    summary = gpu_summary(prof, device, trim)
    summary["export"] = {
        "profile_path": prof.path,
        "format_version": 1,
    }

    if output:
        with open(output, "w") as f:
            json.dump(summary, f, indent=2)

    return summary


def format_preview(rows: list[dict], max_rows: int = 20) -> str:
    """Preview exported data as a formatted text table."""
    if not rows:
        return "No data to export."

    lines = [f"Export preview ({len(rows)} rows, showing first {min(len(rows), max_rows)}):", ""]

    # Header
    lines.append(f"  {'Name':<40s} {'Duration':>10s} {'Stream':>7s} {'NVTX Path'}")
    lines.append(f"  {'─' * 40} {'─' * 10} {'─' * 7} {'─' * 30}")

    for row in rows[:max_rows]:
        name = row["name"][:38] + ".." if len(row["name"]) > 40 else row["name"]
        lines.append(
            f"  {name:<40s} {row['duration_ms']:>8.3f}ms "
            f"  {row['stream']:>5d} {row['nvtx_path'][:50]}"
        )

    if len(rows) > max_rows:
        lines.append(f"  ... and {len(rows) - max_rows} more rows")

    return "\n".join(lines)
