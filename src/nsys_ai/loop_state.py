"""loop_state.py — H100 preset discovery and shared ranking helpers.

SessionStore is the only loop source of truth. This module keeps labeling and
preset utilities that are not session state (CLI ``--h100-preset``, UI profile
labels via ``profile_display_name``, and finding ranking for evidence surfaces).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .annotation import headroom_sort_prefix

SEVERITY_WEIGHT = {"critical": 4, "warning": 3, "info": 2}


def _normalize_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank findings so the workflow surfaces the biggest opportunity first.

    When any finding carries a ``headroom_ms`` (recoverable time), those lead —
    largest headroom first — so a small idle gap with large upside outranks a
    dramatic-looking finding with little room to improve; the rest fall back to
    the legacy severity heuristic. When no finding carries a headroom the legacy
    heuristic order is preserved exactly.
    """
    has_headroom = any(isinstance(f.get("headroom_ms"), (int, float)) for f in findings)
    ranked: list[tuple[int, dict[str, Any], int, float | None]] = []
    for idx, f in enumerate(findings):
        severity = str(f.get("severity") or "").lower()
        kind = str(f.get("type") or "").lower()
        score = SEVERITY_WEIGHT.get(severity, 1) * 1000
        if "overlap" in kind or "nccl" in kind:
            score += 250
        if "idle" in kind:
            score += 100
        confidence = f.get("confidence")
        if isinstance(confidence, (int, float)):
            score += int(confidence * 100)
        hv = f.get("headroom_ms")
        headroom = float(hv) if isinstance(hv, (int, float)) else None
        ranked.append((idx, f, score, headroom))

    if has_headroom:
        # Opportunity-first, then the legacy heuristic score, then stable by idx.
        ranked.sort(key=lambda t: (*headroom_sort_prefix(t[3]), -t[2], t[0]))
    else:
        ranked.sort(key=lambda t: t[2] - t[0], reverse=True)
    return [f for _, f, *_ in ranked]


def normalize_profile_path(path: str, *, label: str = "profile") -> str:
    """Expand and validate a profile path for loop/diff operations.

    Symlinks are not followed so Hugging Face snapshot paths keep human-readable
    names (e.g. ``profiles/perf_h100_sp1.sqlite``) instead of blob hashes.
    """
    raw = (path or "").strip()
    if not raw:
        raise ValueError(f"{label} path is required")
    p = Path(raw).expanduser()
    if p.is_symlink():
        resolved = str(p if p.is_absolute() else p.absolute())
    else:
        resolved = str(p.resolve(strict=False))
    candidate = Path(resolved)
    if not candidate.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    if not candidate.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    return resolved


def same_profile_path(left: str, right: str) -> bool:
    """True when two paths refer to the same on-disk profile file."""
    if not left or not right:
        return False
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except OSError:
        return left == right


H100_PRESET_DATASET = "rich7421/fastvideo-wan-h100-sp1-nsys"
H100_PRESET_CACHE = (
    "~/.cache/huggingface/hub/datasets--rich7421--fastvideo-wan-h100-sp1-nsys/snapshots"
)
H100_BEFORE_FILE = "perf_h100_sp1.sqlite"
H100_AFTER_FILE = "perf_h100_sp1_fa3.sqlite"
_BLOB_STEM = re.compile(r"^[a-f0-9]{32,64}$", re.IGNORECASE)


def h100_preset_download_hint() -> str:
    """CLI help text when --h100-preset profiles are missing locally."""
    return (
        f"Download the FA2/FA3 replay pair from Hugging Face (~340 MB):\n"
        f"  hf download {H100_PRESET_DATASET} --repo-type dataset \\\n"
        f"    profiles/perf_h100_sp1.sqlite profiles/perf_h100_sp1_fa3.sqlite\n"
        f"(requires the `hf` CLI: pip install -U huggingface_hub)\n"
        f"Expected cache layout: {H100_PRESET_CACHE}/<rev>/profiles/*.sqlite\n"
        f"Or pass paths explicitly:\n"
        f"  nsys-ai loop /path/to/perf_h100_sp1.sqlite "
        f"--after /path/to/perf_h100_sp1_fa3.sqlite"
    )


def detect_h100_replay_preset() -> dict[str, str] | None:
    """Return before/after paths if the known H100 dataset is present locally."""
    base = Path(
        "~/.cache/huggingface/hub/datasets--rich7421--fastvideo-wan-h100-sp1-nsys/snapshots"
    ).expanduser()
    if not base.exists():
        return None
    snapshots = [p for p in base.iterdir() if p.is_dir()]
    if not snapshots:
        return None
    # Name breaks the tie: co-extracted snapshots share an mtime, and this
    # choice decides which pair of profiles gets diffed.
    snapshots.sort(key=lambda p: (-p.stat().st_mtime, p.name))
    for snap in snapshots:
        before = snap / "profiles" / H100_BEFORE_FILE
        after = snap / "profiles" / H100_AFTER_FILE
        if before.exists() and after.exists():
            return {
                "before_path": normalize_profile_path(str(before), label="before"),
                "after_path": normalize_profile_path(str(after), label="after"),
            }
    return None


def profile_display_name(path: str, preset: dict[str, str] | None = None) -> str:
    """Human-readable profile filename for loop UI labels."""
    raw = (path or "").strip()
    if not raw:
        return "—"
    preset = preset or detect_h100_replay_preset()
    if preset:
        if same_profile_path(raw, preset["before_path"]):
            return H100_BEFORE_FILE
        if same_profile_path(raw, preset["after_path"]):
            return H100_AFTER_FILE
    name = Path(raw).expanduser().name
    if name.endswith(".sqlite") and not _BLOB_STEM.match(Path(name).stem):
        return name
    return name or "—"
