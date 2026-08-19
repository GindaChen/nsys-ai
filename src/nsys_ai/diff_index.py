"""Session-scoped memoization for before/after diff work.

The index is deliberately a derived cache, not a SessionStore artifact.  A
broken or stale file under ``indices/`` is treated as a cache miss and rebuilt;
``diff.json`` remains the auditable source of truth.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .annotation import TraceSelection
from .artifact_io import atomic_write_json
from .diff import (
    CategoryDelta,
    DiffAxisEntry,
    DiffAxisSummary,
    KernelAgg,
    KernelDiff,
    NvtxAgg,
    NvtxDiff,
    ProfileDiffSummary,
    ProfileSummary,
    build_profile_summary,
    diff_profiles,
)
from .fingerprint import get_profile_id, profile_id_is_capture_derived
from .profile import Profile

INDEX_SCHEMA_VERSION = "diff-index-v1"
PAIR_SCHEMA_VERSION = "diff-pair-v1"


def _json_read(path: Path) -> dict[str, Any] | None:
    """Read one optional memo, returning a miss for every corruption shape."""
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _path_signature(path: str) -> list[int | str]:
    """Add a cheap source guard for identities that are not capture-derived."""
    try:
        stat = os.stat(path, follow_symlinks=False)
    except OSError:
        return [os.path.abspath(path), "missing"]
    return [
        os.path.abspath(path),
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    ]


def _profile_identity(profile: Profile) -> dict[str, Any]:
    connection = profile.conn
    profile_id = get_profile_id(connection, fallback_path=profile.path)
    capture_derived = profile_id_is_capture_derived(connection)
    identity: dict[str, Any] = {
        "profile_id": profile_id,
        "capture_derived": capture_derived,
    }
    if not capture_derived:
        identity["source_signature"] = _path_signature(profile.path)
    return identity


def _parameters(
    *, gpu: int | None, trim: tuple[int, int] | None, nvtx_limit: int | None
) -> dict[str, Any]:
    return {
        "gpu": gpu,
        "trim": list(trim) if trim is not None else None,
        "nvtx_limit": nvtx_limit,
    }


def _key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _profile_summary_to_dict(summary: ProfileSummary) -> dict[str, Any]:
    return asdict(summary)


def _profile_summary_from_dict(payload: dict[str, Any]) -> ProfileSummary:
    kernels = [KernelAgg(**row) for row in payload["kernels"]]
    nvtx = [NvtxAgg(**row) for row in payload["nvtx"]]
    return ProfileSummary(
        path=str(payload["path"]),
        gpu=payload.get("gpu"),
        schema_version=payload.get("schema_version"),
        total_gpu_ns=int(payload["total_gpu_ns"]),
        kernel_rows=int(payload["kernel_rows"]),
        kernels=kernels,
        nvtx=nvtx,
        overlap=dict(payload["overlap"]),
        profile_id=str(payload.get("profile_id") or ""),
        product_version=payload.get("product_version"),
        devices=tuple(int(device) for device in payload.get("devices") or ()),
        profile_id_is_capture_derived=bool(
            payload.get("profile_id_is_capture_derived", False)
        ),
        device_kernel_ns={
            int(device): int(total)
            for device, total in (payload.get("device_kernel_ns") or {}).items()
        },
    )


def _selection(payload: dict[str, Any] | None) -> TraceSelection | None:
    return TraceSelection.from_dict(payload) if payload is not None else None


def _kernel_diff(payload: dict[str, Any]) -> KernelDiff:
    value = dict(payload)
    value["selection"] = _selection(value.get("selection"))
    return KernelDiff(**value)


def _axis_summary(payload: dict[str, Any] | None) -> DiffAxisSummary | None:
    if payload is None:
        return None
    entries = []
    for raw in payload.get("entries") or []:
        value = dict(raw)
        value["selection"] = _selection(value.get("selection"))
        entries.append(DiffAxisEntry(**value))
    value = dict(payload)
    value["entries"] = entries
    return DiffAxisSummary(**value)


def _diff_summary_to_dict(summary: ProfileDiffSummary) -> dict[str, Any]:
    return asdict(summary)


def _diff_summary_from_dict(payload: dict[str, Any]) -> ProfileDiffSummary:
    categories = [CategoryDelta(**row) for row in payload.get("category_attribution") or []]
    return ProfileDiffSummary(
        before=_profile_summary_from_dict(payload["before"]),
        after=_profile_summary_from_dict(payload["after"]),
        warnings=[str(warning) for warning in payload.get("warnings") or []],
        kernel_diffs=[_kernel_diff(row) for row in payload.get("kernel_diffs") or []],
        nvtx_diffs=[NvtxDiff(**row) for row in payload.get("nvtx_diffs") or []],
        overlap_before=dict(payload.get("overlap_before") or {}),
        overlap_after=dict(payload.get("overlap_after") or {}),
        overlap_delta=dict(payload.get("overlap_delta") or {}),
        top_regressions=[_kernel_diff(row) for row in payload.get("top_regressions") or []],
        top_improvements=[_kernel_diff(row) for row in payload.get("top_improvements") or []],
        verdict=str(payload.get("verdict") or "neutral"),
        comparability_confidence=float(payload.get("comparability_confidence", 1.0)),
        category_attribution=categories,
        communication_summary=_axis_summary(payload.get("communication_summary")),
        idle_summary=_axis_summary(payload.get("idle_summary")),
        step_time_delta_ms=payload.get("step_time_delta_ms"),
        step_time_delta_pct=payload.get("step_time_delta_pct"),
        diff_id=str(payload.get("diff_id") or ""),
    )


class DiffIndex:
    """Persist side summaries and one reconciled pair summary for a session."""

    def __init__(self, session_directory: str | os.PathLike[str]):
        self.session_directory = Path(session_directory).expanduser().resolve(strict=False)
        self.indices_directory = self.session_directory / "indices"

    def _side_path(self, role: str) -> Path:
        if role not in {"before", "after"}:
            raise ValueError("diff index role must be before or after")
        return self.indices_directory / f"{role}.json"

    def _side_summary(
        self,
        role: str,
        profile: Profile,
        *,
        gpu: int | None,
        trim: tuple[int, int] | None,
        nvtx_limit: int | None,
    ) -> tuple[ProfileSummary, str]:
        identity = _profile_identity(profile)
        parameters = _parameters(gpu=gpu, trim=trim, nvtx_limit=nvtx_limit)
        key_payload = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "role": role,
            "identity": identity,
            "parameters": parameters,
        }
        key = _key(key_payload)
        cached = _json_read(self._side_path(role))
        if cached is not None and cached.get("key") == key:
            try:
                summary = _profile_summary_from_dict(cached["summary"])
                return replace(summary, path=profile.path), key
            except (KeyError, TypeError, ValueError):
                pass

        summary = build_profile_summary(
            profile, gpu, trim, nvtx_limit=nvtx_limit
        )
        atomic_write_json(
            self._side_path(role),
            {
                "schema_version": INDEX_SCHEMA_VERSION,
                "key": key,
                "role": role,
                "identity": identity,
                "parameters": parameters,
                "summary": _profile_summary_to_dict(summary),
            },
        )
        return summary, key

    def reconcile(
        self,
        before: Profile,
        after: Profile,
        *,
        gpu: int | None,
        trim: tuple[int, int] | None = None,
        limit: int = 15,
        sort: str = "delta",
        nvtx_limit: int | None = 200,
        regression_pct: float = 5.0,
    ) -> ProfileDiffSummary:
        """Load/rebuild side indexes and reconcile a pair summary.

        The pair memo is optional derived state. Any read/validation failure
        falls through to the canonical ``diff_profiles`` computation.
        """
        before_summary, before_key = self._side_summary(
            "before", before, gpu=gpu, trim=trim, nvtx_limit=nvtx_limit
        )
        after_summary, after_key = self._side_summary(
            "after", after, gpu=gpu, trim=trim, nvtx_limit=nvtx_limit
        )
        pair_payload = {
            "schema_version": PAIR_SCHEMA_VERSION,
            "before_key": before_key,
            "after_key": after_key,
            "parameters": {
                **_parameters(gpu=gpu, trim=trim, nvtx_limit=nvtx_limit),
                "limit": limit,
                "sort": sort,
                "regression_pct": regression_pct,
            },
        }
        pair_key = _key(pair_payload)
        cached = _json_read(self.indices_directory / "pair_summary.json")
        if cached is not None and cached.get("key") == pair_key:
            try:
                summary = _diff_summary_from_dict(cached["summary"])
                return replace(
                    summary,
                    before=replace(summary.before, path=before.path),
                    after=replace(summary.after, path=after.path),
                )
            except (KeyError, TypeError, ValueError):
                pass

        summary = diff_profiles(
            before,
            after,
            gpu=gpu,
            trim=trim,
            limit=limit,
            sort=sort,
            nvtx_limit=nvtx_limit,
            regression_pct=regression_pct,
            before_summary=before_summary,
            after_summary=after_summary,
        )
        atomic_write_json(
            self.indices_directory / "pair_summary.json",
            {
                **pair_payload,
                "key": pair_key,
                "summary": _diff_summary_to_dict(summary),
            },
        )
        return summary


__all__ = ["DiffIndex", "INDEX_SCHEMA_VERSION", "PAIR_SCHEMA_VERSION"]
