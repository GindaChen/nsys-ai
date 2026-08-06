"""
diff_decision.py - Persist auditable user decisions for profile diffs.
"""

from __future__ import annotations

import copy
import os
import subprocess  # nosec B404
from datetime import datetime, timezone
from pathlib import Path

from .artifact_io import atomic_write_json
from .diff import MIN_COMPARABILITY_CONFIDENCE, ProfileDiffSummary
from .diff_render import to_diff_dict


def resolve_decider_identity(cwd: str | os.PathLike[str] | None = None) -> str:
    """Resolve the decider identity from git config, falling back to USER."""
    try:
        result = subprocess.run(  # nosec B603 B607
            ["git", "config", "user.email"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        result = None

    if result is not None and result.returncode == 0:
        email = result.stdout.strip()
        if email:
            return email

    for name in ("USER", "USERNAME"):
        identity = os.environ.get(name, "").strip()
        if identity:
            return identity
    return "unknown"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_diff_decision_record_from_diff_dict(
    diff_dict: dict,
    *,
    decision: str,
    reason: str,
    decider: str | None = None,
    decided_at: str | None = None,
) -> tuple[dict, list[str]]:
    """Build the byte-stable diff.json payload from a ``to_diff_dict`` mapping.

    Shared by the CLI (which renders the mapping from a fresh ``ProfileDiffSummary``)
    and the web guided loop (which already holds the diff payload in loop state), so
    both surfaces emit byte-identical decision records for the same diff. The input
    mapping is copied, never mutated.
    """
    normalized_decision = decision.strip().lower()
    if normalized_decision not in {"accepted", "rejected"}:
        raise ValueError("decision must be 'accepted' or 'rejected'")

    normalized_reason = reason.strip()
    if not normalized_reason:
        raise ValueError("--reason is required with --accept/--reject")

    resolved_decider = decider if decider is not None else resolve_decider_identity()
    if not isinstance(resolved_decider, str) or not resolved_decider.strip():
        raise ValueError("decider must be a non-empty string when provided")
    resolved_decided_at = decided_at if decided_at is not None else _utc_now_iso()
    if not isinstance(resolved_decided_at, str) or not resolved_decided_at.strip():
        raise ValueError("decided_at must be a non-empty string when provided")

    before_id = (diff_dict.get("before") or {}).get("profile_id")
    after_id = (diff_dict.get("after") or {}).get("profile_id")
    if not before_id or not after_id:
        raise ValueError("cannot write diff decision without before and after profile_id")

    payload = copy.deepcopy(diff_dict)
    warnings = list(payload.get("warnings") or [])
    advisory_warnings: list[str] = []

    confidence = payload.get("comparability_confidence")
    if isinstance(confidence, (int, float)) and confidence < MIN_COMPARABILITY_CONFIDENCE:
        warning = (
            "comparability_confidence "
            f"{confidence:.3f} is below "
            f"{MIN_COMPARABILITY_CONFIDENCE:.3f}; stamping verdict as inconclusive"
        )
        if warning not in warnings:
            warnings.append(warning)
        advisory_warnings.append(warning)
        payload["verdict"] = "inconclusive"

    payload["warnings"] = warnings
    payload["decision"] = {
        "status": normalized_decision,
        "reason": normalized_reason,
        "decider": resolved_decider,
        "decided_at": resolved_decided_at,
    }
    return payload, advisory_warnings


def build_diff_decision_record(
    summary: ProfileDiffSummary,
    *,
    decision: str,
    reason: str,
    decider: str | None = None,
    decided_at: str | None = None,
) -> tuple[dict, list[str]]:
    """Build the byte-stable diff.json payload and return advisory warnings."""
    return build_diff_decision_record_from_diff_dict(
        to_diff_dict(summary),
        decision=decision,
        reason=reason,
        decider=decider,
        decided_at=decided_at,
    )


def _write_decision_payload(payload: dict, path: str | os.PathLike[str]) -> Path:
    """Serialize a decision payload deterministically and publish it atomically."""
    return atomic_write_json(path, payload)


def write_diff_json_from_diff_dict(
    diff_dict: dict, *, path: str | os.PathLike[str] = "diff.json"
) -> tuple[Path, dict]:
    """Publish a canonical diff payload without changing its shape."""
    payload = copy.deepcopy(diff_dict)
    return _write_decision_payload(payload, path), payload


def write_diff_decision_json(
    summary: ProfileDiffSummary,
    *,
    decision: str,
    reason: str,
    path: str | os.PathLike[str] = "diff.json",
    decider: str | None = None,
    decided_at: str | None = None,
) -> tuple[Path, dict, list[str]]:
    """Write a diff decision record using the shared deterministic JSON encoding."""
    payload, warnings = build_diff_decision_record(
        summary,
        decision=decision,
        reason=reason,
        decider=decider,
        decided_at=decided_at,
    )
    return _write_decision_payload(payload, path), payload, warnings


def write_diff_decision_json_from_diff_dict(
    diff_dict: dict,
    *,
    decision: str,
    reason: str,
    path: str | os.PathLike[str] = "diff.json",
    decider: str | None = None,
    decided_at: str | None = None,
) -> tuple[Path, dict, list[str]]:
    """Write a decision record from a stored diff payload (see the dict builder)."""
    payload, warnings = build_diff_decision_record_from_diff_dict(
        diff_dict,
        decision=decision,
        reason=reason,
        decider=decider,
        decided_at=decided_at,
    )
    return _write_decision_payload(payload, path), payload, warnings
