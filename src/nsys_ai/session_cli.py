"""Parameter-explicit SessionStore publishers for CLI and other surfaces.

Handlers unpack argparse Namespace values and call these functions. Callers must
finish diagnose / propose / diff analysis before acquiring a writer lease.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .annotation import EvidenceReport
from .profile_reference import LocalProfileReference
from .proposal import Proposal
from .runspec import RunSpec
from .session_store import (
    SessionExistsError,
    SessionSnapshot,
    SessionState,
    SessionStore,
)


def session_id_from_profile_id(profile_id: str) -> str:
    """Derive a SessionStore-safe id from a content ``profile_id``.

    ``LocalProfileReference.profile_id`` uses ``:`` (for example
    ``nsys2:sha256:...``), but ``SessionStore`` rejects that character. The
    design note says to use the content id as the default session id; this
    substitution is the filesystem-safe form of that id.
    """
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("profile_id must be a non-empty string")
    derived = profile_id.replace(":", "_")
    # SessionState.__post_init__ applies the same id grammar as create/load.
    SessionState(session_id=derived)
    return derived


def resolve_session_id(
    explicit: str | None,
    *,
    before: LocalProfileReference | None = None,
) -> str:
    """Return a caller id, or derive one from the before profile content id."""
    if explicit:
        return explicit
    if before is None:
        raise ValueError(
            "session id is required when no before profile is available to derive from"
        )
    return session_id_from_profile_id(before.profile_id)


def publish_session_findings(
    *,
    session_id: str,
    report: EvidenceReport,
    before_profile: LocalProfileReference,
    root: str | os.PathLike[str] = ".nsys-ai/sessions",
) -> SessionState:
    """Create the session if needed, then publish findings under a short writer lease."""
    if not isinstance(report, EvidenceReport):
        raise TypeError("report must be an EvidenceReport")
    store = SessionStore(root)
    try:
        store.create(session_id, before_profile=before_profile)
    except SessionExistsError:
        pass
    with store.writer(session_id) as writer:
        return writer.publish_findings(report, before_profile=before_profile)


def publish_session_proposal(
    *,
    session_id: str,
    proposal: Proposal,
    runspec: RunSpec | None = None,
    resolved_secrets: Mapping[str, str] | None = None,
    root: str | os.PathLike[str] = ".nsys-ai/sessions",
) -> SessionState:
    """Publish an optional RunSpec then the Proposal under one writer lease."""
    if not isinstance(proposal, Proposal):
        raise TypeError("proposal must be a Proposal")
    store = SessionStore(root)
    with store.writer(session_id) as writer:
        if runspec is not None:
            writer.publish_runspec(runspec, resolved_secrets=resolved_secrets)
        return writer.publish_proposal(proposal)


def publish_session_diff(
    *,
    session_id: str,
    diff: Mapping[str, Any],
    after_profile: LocalProfileReference,
    root: str | os.PathLike[str] = ".nsys-ai/sessions",
) -> SessionState:
    """Publish undecided diff.json via SessionWriter.

    Registers ``after_profile`` when the session is in ``propose`` /
    ``reprofile`` (after a non-abstained proposal). Sessions still in
    ``diagnose`` have not been proposed yet — run ``nsys-ai propose`` first.
    When the session is already in ``diff`` (re-publish), the after profile
    must already match ``after_profile``; ``publish_diff`` re-validates the
    references.
    """
    if not isinstance(diff, Mapping):
        raise TypeError("diff must be a mapping")
    store = SessionStore(root)
    snapshot = store.load(session_id)
    with store.writer(session_id) as writer:
        if snapshot.state.phase in {"propose", "reprofile"}:
            if snapshot.proposal is None or snapshot.proposal.abstained:
                reason = (
                    snapshot.proposal.abstention_reason
                    if snapshot.proposal is not None
                    and snapshot.proposal.abstention_reason
                    else "no non-abstained proposal is present"
                )
                raise ValueError(
                    f"session proposal abstained ({reason}); supply a "
                    "verification RunSpec via nsys-ai propose --session "
                    "... --runspec <path> before publishing a diff with "
                    "--session"
                )
            writer.publish_after_profile(after_profile)
        elif snapshot.state.phase == "diagnose":
            raise ValueError(
                "session is still in diagnose phase; run nsys-ai propose "
                "before publishing a diff with --session"
            )
        elif snapshot.state.after_profile != after_profile:
            raise ValueError(
                "session after profile does not match the after profile being diffed"
            )
        return writer.publish_diff(diff)


def publish_session_decision(
    *,
    session_id: str,
    decision: str,
    reason: str,
    decider: str | None = None,
    root: str | os.PathLike[str] = ".nsys-ai/sessions",
) -> SessionState:
    """Record accept/reject on a session's published diff.

    The store permits one decision per finding. An accepted finding remains a
    terminal session; a rejected finding is recorded in ``decisions.json`` and
    returns the session to ``propose`` so another finding can be evaluated.
    ``publish_decision`` raises if the diff is missing or the finding was already
    decided; both are surfaced to the caller unchanged.
    """
    if not reason or not reason.strip():
        raise ValueError("a decision requires a reason")
    store = SessionStore(root)
    with store.writer(session_id) as writer:
        state, _, _ = writer.publish_decision(decision, reason, decider=decider)
    return state


def session_dir(session_id: str, root: str | os.PathLike[str] = ".nsys-ai/sessions") -> Path:
    """Return the on-disk directory for ``session_id`` under ``root``."""
    return SessionStore(root).root / session_id


def project_loop_state(
    snapshot: SessionSnapshot,
    *,
    session_dir_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Project a SessionSnapshot into the LOOP_STATE shape timeline.js reads.

    Field dispositions follow docs/notes/session-wiring-design.md Deliverable 6.
    DERIVED fields come from artifacts at read time; DROP fields are omitted or
    empty. ``decision_path`` is emitted only when a decision is recorded, either
    in ``diff.json`` or in the session's ``decisions.json`` history.
    """
    from .loop_state import profile_display_name

    if not isinstance(snapshot, SessionSnapshot):
        raise TypeError("snapshot must be a SessionSnapshot")

    before = snapshot.state.before_profile
    after = snapshot.state.after_profile
    before_path = before.path if before is not None else ""
    after_path = after.path if after is not None else ""

    findings = snapshot.findings
    diagnose_ran = findings is not None
    diagnose_findings_count = len(findings.findings) if findings is not None else 0

    proposal_payload: dict[str, Any] | None = None
    expected_impact: dict[str, Any] | str | None = None
    if snapshot.proposal is not None:
        proposal_payload = snapshot.proposal.to_dict()
        impact = snapshot.proposal.expected_impact
        if impact is not None:
            expected_impact = {
                "headroom_ms": impact.headroom_ms,
                "headroom_basis": impact.headroom_basis,
            }

    diff_summary: dict[str, Any] | None = (
        dict(snapshot.diff) if snapshot.diff is not None else None
    )
    decision = None
    decision_reason = ""
    decision_path = ""
    verdict = "neutral"
    comparability_confidence = None
    if diff_summary is not None:
        verdict = str(diff_summary.get("verdict") or "neutral")
        confidence = diff_summary.get("comparability_confidence")
        if isinstance(confidence, (int, float)):
            comparability_confidence = float(confidence)
        recorded = diff_summary.get("decision")
        if recorded is not None:
            status = str(recorded.get("status") or "").strip().lower()
            if status == "accepted":
                decision = "accept"
            elif status == "rejected":
                decision = "reject"
            decision_reason = str(recorded.get("reason") or "")
            # C5: path only when a decision has been recorded, not merely when
            # diff.json exists.
            decision_path = str(Path(session_dir_path) / "diff.json")
    if decision is None and snapshot.decisions:
        recorded = snapshot.decisions[-1]
        status = str(recorded.get("status") or "").strip().lower()
        if status == "accepted":
            decision = "accept"
        elif status == "rejected":
            decision = "reject"
        decision_reason = str(recorded.get("reason") or "")
        decision_path = str(Path(session_dir_path) / "decisions.json")

    return {
        "session_mode": True,
        "session_id": snapshot.state.session_id,
        "before_path": before_path,
        "after_path": after_path,
        "before_label": profile_display_name(before_path),
        "after_label": profile_display_name(after_path),
        "phase": snapshot.state.phase,
        "proposal": proposal_payload,
        "expected_impact": expected_impact if expected_impact is not None else "",
        "decision": decision,
        "decision_reason": decision_reason,
        "decision_path": decision_path,
        "diagnose_ran": diagnose_ran,
        "diagnose_findings_count": diagnose_findings_count,
        "diff_summary": diff_summary,
        "comparability_confidence": comparability_confidence,
        "verdict": verdict,
        "last_error": "",
    }
