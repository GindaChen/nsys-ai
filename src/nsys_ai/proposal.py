"""Deterministic proposal artifacts derived from evidence findings.

Proposal v0 records a change hypothesis and an exact ``(profile_id,
finding_id)`` pointer. It never infers source code, patches, or realized gain.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .annotation import Finding, TraceSelection
from .exceptions import NsysAiError
from .runspec import (
    RunSpec,
    RunSpecError,
    validate_persisted_secret_strings,
    validate_secret_boundaries,
)

PROPOSAL_SCHEMA_VERSION = "0.1"
_PROPOSAL_ID_PREFIX = "proposal1:sha256:"
_BASE_LIMITATIONS = (
    "Proposal v0 is a change hypothesis, not a source patch.",
    "The trace does not attribute an exact source file or line.",
    "Realized performance gain requires reprofiling and diffing.",
)
_TRACE_TARGET_FIELDS = {
    "id",
    "profile_id",
    "source",
    "start_ns",
    "end_ns",
    "gpu_ids",
    "rank_ids",
    "stream_ids",
    "nvtx_path",
    "event_ids",
    "label",
}


class ProposalError(NsysAiError):
    """A Proposal payload or construction argument is invalid."""

    error_code = "PROPOSAL_INVALID"


class UnsupportedProposalVersionError(ProposalError):
    """A Proposal artifact uses a schema this installation cannot read."""

    error_code = "PROPOSAL_VERSION_UNSUPPORTED"


def _canonical_json_bytes(value: Any, label: str) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProposalError(f"{label} must contain JSON-compatible values") from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _snapshot_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProposalError(f"{label} must be an object")
    canonical = _canonical_json_bytes(dict(value), label)
    return _freeze_json(json.loads(canonical))


def _require_string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProposalError(f"{label} must be a string")
    if not allow_empty and not value:
        raise ProposalError(f"{label} must not be empty")
    if "\x00" in value:
        raise ProposalError(f"{label} must not contain NUL bytes")
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ProposalError(f"{label} must be an array of strings")
    return tuple(
        _require_string(item, f"{label}[{index}]") for index, item in enumerate(value)
    )


def _canonical_id(value: Any, label: str, *, allow_empty: bool = False) -> str:
    value = _require_string(value, label, allow_empty=allow_empty)
    if value and any(character.isspace() for character in value):
        raise ProposalError(f"{label} must not contain whitespace")
    return value


def _trace_target(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProposalError("trace_target must be an object")
    unknown = sorted(set(value) - _TRACE_TARGET_FIELDS)
    if unknown:
        raise ProposalError(f"trace_target has unknown field(s): {', '.join(unknown)}")
    missing = sorted({"id", "profile_id", "source"} - set(value))
    if missing:
        raise ProposalError(f"trace_target is missing field(s): {', '.join(missing)}")
    try:
        selection = TraceSelection.from_dict(dict(value))
        _canonical_id(selection.id, "trace_target.id")
        _canonical_id(selection.profile_id, "trace_target.profile_id", allow_empty=True)
        _require_string(selection.source, "trace_target.source")
        for name in ("start_ns", "end_ns"):
            number = getattr(selection, name)
            if number is not None and (
                isinstance(number, bool) or not isinstance(number, int)
            ):
                raise ProposalError(f"trace_target.{name} must be an integer or null")
        if (
            selection.start_ns is not None
            and selection.end_ns is not None
            and selection.end_ns < selection.start_ns
        ):
            raise ProposalError("trace_target.end_ns must not precede start_ns")
        for name in ("gpu_ids", "rank_ids", "stream_ids"):
            numbers = getattr(selection, name)
            if numbers is None:
                continue
            if not isinstance(numbers, (list, tuple)):
                raise ProposalError(f"trace_target.{name} must be an array or null")
            if any(isinstance(item, bool) or not isinstance(item, int) for item in numbers):
                raise ProposalError(f"trace_target.{name} must contain only integers")
        if selection.nvtx_path is not None:
            _string_tuple(selection.nvtx_path, "trace_target.nvtx_path")
        if selection.event_ids is not None:
            _string_tuple(selection.event_ids, "trace_target.event_ids")
        if selection.label is not None:
            _require_string(selection.label, "trace_target.label", allow_empty=True)
    except ProposalError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProposalError(f"trace_target is invalid: {exc}") from exc
    return _snapshot_mapping(selection.to_dict(), "trace_target")


@dataclass(frozen=True)
class ExpectedImpact:
    """Measured opportunity copied from a Finding, not a predicted gain."""

    headroom_ms: float
    headroom_basis: str

    def __post_init__(self) -> None:
        value = self.headroom_ms
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ProposalError("expected_impact.headroom_ms must be finite and non-negative")
        object.__setattr__(self, "headroom_ms", float(value))
        _require_string(self.headroom_basis, "expected_impact.headroom_basis")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "measured_headroom",
            "headroom_ms": self.headroom_ms,
            "headroom_basis": self.headroom_basis,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ExpectedImpact:
        if not isinstance(payload, Mapping):
            raise ProposalError("expected_impact must be an object or null")
        expected = {"kind", "headroom_ms", "headroom_basis"}
        if set(payload) != expected:
            raise ProposalError("expected_impact requires only kind, headroom_ms, headroom_basis")
        if payload.get("kind") != "measured_headroom":
            raise ProposalError("expected_impact.kind must be 'measured_headroom'")
        return cls(payload["headroom_ms"], payload["headroom_basis"])


@dataclass(frozen=True)
class Proposal:
    """A versioned, round-trippable proposal or explicit abstention."""

    proposal_id: str
    source_finding_id: str
    source_profile_id: str
    summary: str
    suggested_actions: tuple[str, ...]
    trace_target: Mapping[str, Any] | None
    expected_impact: ExpectedImpact | None
    confidence: float | None
    verification: RunSpec | None
    limitations: tuple[str, ...]
    abstained: bool
    abstention_reason: str | None

    def __post_init__(self) -> None:
        _require_string(self.proposal_id, "proposal_id")
        _canonical_id(self.source_finding_id, "source_finding_id", allow_empty=True)
        _canonical_id(self.source_profile_id, "source_profile_id", allow_empty=True)
        _require_string(self.summary, "summary", allow_empty=True)
        object.__setattr__(
            self,
            "suggested_actions",
            _string_tuple(self.suggested_actions, "suggested_actions"),
        )
        if self.trace_target is not None:
            object.__setattr__(self, "trace_target", _trace_target(self.trace_target))
            if self.trace_target["profile_id"] != self.source_profile_id:
                raise ProposalError("trace_target.profile_id must equal source_profile_id")
        if self.expected_impact is not None and not isinstance(
            self.expected_impact, ExpectedImpact
        ):
            raise ProposalError("expected_impact must be an ExpectedImpact or null")
        if self.confidence is not None:
            confidence = self.confidence
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                raise ProposalError("confidence must be between 0 and 1 or null")
            object.__setattr__(self, "confidence", float(confidence))
        if self.verification is not None and not isinstance(self.verification, RunSpec):
            raise ProposalError("verification must be a RunSpec or null")
        object.__setattr__(self, "limitations", _string_tuple(self.limitations, "limitations"))
        if not isinstance(self.abstained, bool):
            raise ProposalError("abstained must be a boolean")
        if self.abstained:
            _require_string(self.abstention_reason, "abstention_reason")
        elif self.abstention_reason is not None:
            raise ProposalError("abstention_reason must be null when abstained is false")
        if not self.abstained and (
            not self.source_finding_id
            or not self.source_profile_id
            or not self.summary
            or not self.suggested_actions
            or self.trace_target is None
            or self.verification is None
        ):
            raise ProposalError("non-abstained Proposal is missing a required proposal input")
        expected_id = _proposal_id(self._identity_payload())
        if self.proposal_id != expected_id:
            raise ProposalError(
                f"proposal_id does not match artifact content; expected {expected_id}"
            )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PROPOSAL_SCHEMA_VERSION,
            "source_finding_id": self.source_finding_id,
            "source_profile_id": self.source_profile_id,
            "summary": self.summary,
            "suggested_actions": list(self.suggested_actions),
            "trace_target": _thaw_json(self.trace_target),
            "expected_impact": (
                self.expected_impact.to_dict() if self.expected_impact is not None else None
            ),
            "confidence": self.confidence,
            "verification": (
                {"kind": "runspec", "runspec": self.verification.to_dict()}
                if self.verification is not None
                else None
            ),
            "limitations": list(self.limitations),
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"proposal_id": self.proposal_id, **self._identity_payload()}

    def canonical_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict(), "Proposal")

    @classmethod
    def from_dict(cls, payload: Any) -> Proposal:
        if not isinstance(payload, Mapping):
            raise ProposalError("Proposal must be an object")
        allowed = {
            "schema_version",
            "proposal_id",
            "source_finding_id",
            "source_profile_id",
            "summary",
            "suggested_actions",
            "trace_target",
            "expected_impact",
            "confidence",
            "verification",
            "limitations",
            "abstained",
            "abstention_reason",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ProposalError(f"Proposal has unknown field(s): {', '.join(unknown)}")
        missing = sorted(allowed - set(payload))
        if missing:
            raise ProposalError(f"Proposal is missing field(s): {', '.join(missing)}")
        version = payload["schema_version"]
        if version != PROPOSAL_SCHEMA_VERSION:
            raise UnsupportedProposalVersionError(
                f"unsupported Proposal schema_version {version!r}; "
                f"expected {PROPOSAL_SCHEMA_VERSION!r}"
            )
        verification_payload = payload["verification"]
        verification = None
        if verification_payload is not None:
            if not isinstance(verification_payload, Mapping):
                raise ProposalError("verification must be an object or null")
            if set(verification_payload) != {"kind", "runspec"}:
                raise ProposalError("verification requires only kind and runspec")
            if verification_payload.get("kind") != "runspec":
                raise ProposalError("verification.kind must be 'runspec'")
            try:
                verification = RunSpec.from_dict(verification_payload["runspec"])
            except RunSpecError as exc:
                raise ProposalError(f"invalid verification RunSpec: {exc}") from exc
        impact_payload = payload["expected_impact"]
        impact = ExpectedImpact.from_dict(impact_payload) if impact_payload is not None else None
        return cls(
            proposal_id=payload["proposal_id"],
            source_finding_id=payload["source_finding_id"],
            source_profile_id=payload["source_profile_id"],
            summary=payload["summary"],
            suggested_actions=payload["suggested_actions"],
            trace_target=payload["trace_target"],
            expected_impact=impact,
            confidence=payload["confidence"],
            verification=verification,
            limitations=payload["limitations"],
            abstained=payload["abstained"],
            abstention_reason=payload["abstention_reason"],
        )

    @classmethod
    def from_json_bytes(cls, payload: bytes | str) -> Proposal:
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProposalError(f"invalid Proposal JSON: {exc}") from exc
        return cls.from_dict(data)


def _proposal_id(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json_bytes(dict(payload), "Proposal")).hexdigest()
    return _PROPOSAL_ID_PREFIX + digest


def _clean_strings(values: Any, label: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ProposalError(f"{label} must be an array of strings or null")
    cleaned: list[str] = []
    for index, item in enumerate(values):
        _require_string(item, f"{label}[{index}]", allow_empty=True)
        if item.strip():
            cleaned.append(item.strip())
    return tuple(cleaned)


def _derive_fields(finding: Finding, verification: RunSpec | None) -> dict[str, Any]:
    selection = finding.selection
    finding_id = "" if finding.id is None else _canonical_id(
        finding.id, "source finding id", allow_empty=True
    )
    profile_id = ""
    target = None
    if selection is not None:
        if not isinstance(selection, TraceSelection):
            raise ProposalError("source finding selection must be a TraceSelection")
        _canonical_id(selection.id, "source finding selection.id")
        profile_id = _canonical_id(
            selection.profile_id,
            "source finding selection.profile_id",
            allow_empty=True,
        )
        target = _thaw_json(_trace_target(selection.to_dict()))
    actions = _clean_strings(finding.suggested_actions, "source finding suggested_actions")
    if finding.explanation is not None:
        _require_string(finding.explanation, "source finding explanation", allow_empty=True)
    _require_string(finding.label, "source finding label", allow_empty=True)
    summary = (finding.explanation or "").strip() or finding.label.strip()
    reasons: list[str] = []
    if not finding_id:
        reasons.append("source finding has no id")
    if selection is None:
        reasons.append("source finding has no trace selection")
    elif not profile_id:
        reasons.append("source finding selection has no profile id")
    if not actions:
        reasons.append("source finding has no suggested action")
    if verification is None:
        reasons.append("verification RunSpec is required")
    if not summary:
        reasons.append("source finding has no summary")
    limitations = [
        *_BASE_LIMITATIONS,
        *_clean_strings(
            finding.false_positive_notes, "source finding false_positive_notes"
        ),
    ]
    if verification is not None:
        limitations.extend(verification.compatibility_limitations())
    impact = None
    if finding.headroom_ms is not None and (
        isinstance(finding.headroom_ms, bool)
        or not isinstance(finding.headroom_ms, (int, float))
        or not math.isfinite(finding.headroom_ms)
        or finding.headroom_ms < 0
    ):
        raise ProposalError("source finding headroom_ms must be finite and non-negative")
    if finding.headroom_basis is not None:
        _require_string(
            finding.headroom_basis,
            "source finding headroom_basis",
            allow_empty=True,
        )
    if finding.headroom_ms is not None and finding.headroom_basis:
        impact = ExpectedImpact(finding.headroom_ms, finding.headroom_basis)
    elif finding.headroom_ms is not None:
        limitations.append("Source headroom has no basis and was not copied.")
    return {
        "source_finding_id": finding_id,
        "source_profile_id": profile_id,
        "summary": summary,
        "suggested_actions": actions,
        "trace_target": target,
        "expected_impact": impact,
        "confidence": _finding_confidence(finding.confidence),
        "limitations": tuple(limitations),
        "abstained": bool(reasons),
        "abstention_reason": "; ".join(reasons) if reasons else None,
    }


def _identity(fields: Mapping[str, Any], verification: RunSpec | None) -> dict[str, Any]:
    return {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "source_finding_id": fields["source_finding_id"],
        "source_profile_id": fields["source_profile_id"],
        "summary": fields["summary"],
        "suggested_actions": list(fields["suggested_actions"]),
        "trace_target": fields["trace_target"],
        "expected_impact": (
            fields["expected_impact"].to_dict()
            if fields["expected_impact"] is not None
            else None
        ),
        "confidence": fields["confidence"],
        "verification": (
            {"kind": "runspec", "runspec": verification.to_dict()}
            if verification is not None
            else None
        ),
        "limitations": list(fields["limitations"]),
        "abstained": fields["abstained"],
        "abstention_reason": fields["abstention_reason"],
    }


def _secret_scan_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(identity)
    verification = identity.get("verification")
    if verification is not None:
        runspec = dict(verification["runspec"])
        environment = dict(runspec["environment"])
        environment["secrets"] = []
        runspec["environment"] = environment
        payload["verification"] = {"kind": "runspec", "runspec": runspec}
    return payload


def _finding_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ProposalError("source finding confidence must be between 0 and 1 or null")
    return float(value)


def generate_proposal(
    finding: Finding,
    verification: RunSpec | None,
    *,
    resolved_secrets: Mapping[str, str] | None = None,
) -> Proposal:
    """Derive a Proposal without IO or storing the resolved-secret mapping.

    Before ID construction, resolved values are checked against every emitted
    string key and value except the intentionally persisted declaration names.
    Non-string measurements are not converted to text for this check.
    """
    if not isinstance(finding, Finding):
        raise ProposalError("finding must be a Finding")
    if verification is not None and not isinstance(verification, RunSpec):
        raise ProposalError("verification must be a RunSpec or null")
    if verification is None and resolved_secrets is not None:
        raise ProposalError("resolved_secrets requires a verification RunSpec")
    resolved = {} if resolved_secrets is None else resolved_secrets
    if verification is not None:
        validate_secret_boundaries(verification, resolved)
    fields = _derive_fields(finding, verification)
    identity = _identity(fields, verification)
    # Resolved values are never stored. Scan all persisted user-controlled
    # string keys and values without textualizing numeric measurements.
    validate_persisted_secret_strings(_secret_scan_identity(identity), resolved)
    proposal = Proposal(
        proposal_id=_proposal_id(identity),
        source_finding_id=fields["source_finding_id"],
        source_profile_id=fields["source_profile_id"],
        summary=fields["summary"],
        suggested_actions=fields["suggested_actions"],
        trace_target=fields["trace_target"],
        expected_impact=fields["expected_impact"],
        confidence=fields["confidence"],
        verification=verification,
        limitations=fields["limitations"],
        abstained=fields["abstained"],
        abstention_reason=fields["abstention_reason"],
    )
    validate_proposal_against_finding(proposal, finding)
    return proposal


def validate_proposal_against_finding(
    proposal: Proposal, finding: Finding
) -> None:
    """Require ``proposal`` to equal the deterministic projection of ``finding``.

    The Proposal's own verification RunSpec is the verification input. This
    semantic check does not resolve secrets or persist any additional data.
    """
    if not isinstance(proposal, Proposal):
        raise ProposalError("proposal must be a Proposal")
    if not isinstance(finding, Finding):
        raise ProposalError("finding must be a Finding")
    fields = _derive_fields(finding, proposal.verification)
    identity = _identity(fields, proposal.verification)
    expected = {"proposal_id": _proposal_id(identity), **identity}
    if proposal.to_dict() != expected:
        raise ProposalError(
            "Proposal is not deterministically derived from the identified Finding"
        )
