"""Parameter-explicit SessionStore publishers for CLI and other surfaces.

Handlers unpack argparse Namespace values and call these functions. Callers must
finish diagnose / propose / diff analysis before acquiring a writer lease.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .annotation import EvidenceReport
from .artifact_root import DEFAULT_SESSION_ROOT
from .artifact_root import session_root as resolve_artifact_session_root
from .profile_reference import LocalProfileReference
from .proposal import Proposal
from .runspec import RunSpec
from .session_store import (
    SessionExistsError,
    SessionSnapshot,
    SessionState,
    SessionStore,
)


def append_ask_log(
    session_id: str | None,
    session_root: str | os.PathLike[str] | None,
    *,
    question: str,
    answer: str,
    selected_skills: list[str],
    evidence: Mapping[str, list[dict]],
    profile_path: str,
    trim_kwargs: Mapping,
) -> str | None:
    """Append one completed ask handoff for any transport."""
    if not session_id:
        return None
    record = {
        "schema_version": "0.1",
        "kind": "ask",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "selected_skills": list(selected_skills),
        "evidence": dict(evidence),
        "profile_path": profile_path,
        "trim": dict(trim_kwargs),
    }
    with SessionStore(session_root or DEFAULT_SESSION_ROOT).writer(session_id) as writer:
        writer.append_log("ask", record)
    return "logs/ask.jsonl"


@dataclass(frozen=True)
class SessionLocation:
    """The canonical handoff location shared by every transport.

    Older callers pass a session id and use ``.nsys-ai/sessions`` relative to
    their working directory.  New callers can pass the session directory
    itself (for example ``/tmp/run-001``), which makes the artifact directory
    portable across CLI, TUI, Web, and plugin processes.
    """

    session_id: str
    root: Path
    explicit: bool = False

    @property
    def directory(self) -> Path:
        return self.root / self.session_id

    def store(self) -> SessionStore:
        return SessionStore(self.root)


def resolve_session_location(
    value: str | os.PathLike[str] | None,
    *,
    root: str | os.PathLike[str] = DEFAULT_SESSION_ROOT,
) -> SessionLocation | None:
    """Resolve an id or an explicit session directory into one location.

    A bare value remains a backwards-compatible id.  Absolute paths, paths
    containing a directory component, and existing directories are treated as
    the handoff directory itself; its basename is the SessionStore id and its
    parent is the store root.  The directory need not exist yet, which lets
    ``diagnose --session /path/to/new-session`` create it atomically.
    """
    if value is None or value == "":
        return None
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw:
        raise ValueError("session must be a non-empty id or directory path")

    candidate = Path(raw).expanduser()
    explicit_directory = (
        candidate.is_absolute()
        or candidate.parent != Path(".")
        or raw.startswith(".")
    )
    if explicit_directory:
        directory = candidate.resolve(strict=False)
        if directory.name in {"", ".", ".."}:
            raise ValueError("session directory must have a session id basename")
        return SessionLocation(directory.name, directory.parent, explicit=True)

    return SessionLocation(
        raw,
        resolve_artifact_session_root(root),
        explicit=False,
    )


def session_location(
    value: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] = DEFAULT_SESSION_ROOT,
) -> SessionLocation:
    """Resolve a required session id/path, raising a friendly ValueError."""
    location = resolve_session_location(value, root=root)
    if location is None:
        raise ValueError("session is required")
    return location


def session_argument(
    session_id: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] = DEFAULT_SESSION_ROOT,
) -> str:
    """Return a re-runnable session argument for a user-facing hint."""
    location = session_location(session_id, root=root)
    default_root = resolve_artifact_session_root()
    if location.root == default_root and not location.explicit:
        return location.session_id
    return shlex.quote(str(location.directory))


def _normalize_target(
    session_id: str | os.PathLike[str],
    root: str | os.PathLike[str],
) -> tuple[str, Path]:
    location = session_location(session_id, root=root)
    return location.session_id, location.root


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


def session_id_from_diff_id(diff_id: str) -> str:
    """Derive a SessionStore-safe id from a content ``diff_id``.

    A diagnose seeded by ``--against`` is a new handoff boundary: its
    findings describe the candidate profile, but the pair comparison is the
    identity that makes the run reproducible. Prefixing the normalized diff
    id also keeps it distinct from a normal profile-derived session.
    """
    if not isinstance(diff_id, str) or not diff_id:
        raise ValueError("diff_id must be a non-empty string")
    derived = f"diff_{diff_id.replace(':', '_')}"
    SessionState(session_id=derived)
    return derived


def resolve_session_id(
    explicit: str | None,
    *,
    before: LocalProfileReference | None = None,
    diff_id: str | None = None,
) -> str:
    """Return a caller id, or derive one from a diff/profile content id.

    ``diff_id`` takes precedence over ``before`` because a diagnose seeded by
    a baseline comparison must not silently replace a normal diagnosis for
    the candidate profile.
    """
    if explicit:
        return explicit
    if diff_id:
        return session_id_from_diff_id(diff_id)
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
    root: str | os.PathLike[str] = DEFAULT_SESSION_ROOT,
) -> SessionState:
    """Create the session if needed, then publish findings under a short writer lease."""
    session_id, root = _normalize_target(session_id, root)
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
    root: str | os.PathLike[str] = DEFAULT_SESSION_ROOT,
) -> SessionState:
    """Publish an optional RunSpec then the Proposal under one writer lease."""
    session_id, root = _normalize_target(session_id, root)
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
    root: str | os.PathLike[str] = DEFAULT_SESSION_ROOT,
) -> SessionState:
    """Publish undecided diff.json via SessionWriter.

    Registers ``after_profile`` when the session is in ``propose`` /
    ``reprofile`` (after a non-abstained proposal). Sessions still in
    ``diagnose`` have not been proposed yet — run ``nsys-ai propose`` first.
    When the session is already in ``diff`` (re-publish), the after profile
    must already match ``after_profile``; ``publish_diff`` re-validates the
    references.
    """
    session_id, root = _normalize_target(session_id, root)
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


def validate_session_diff_after_profile(
    *,
    session_id: str,
    after_profile: LocalProfileReference,
    root: str | os.PathLike[str] = DEFAULT_SESSION_ROOT,
) -> SessionSnapshot:
    """Validate a diff candidate before any expensive or persistent work.

    ``publish_session_diff`` remains the authoritative writer-side guard. CLI
    callers also need the same check before running ``DiffIndex.reconcile``:
    a rejected after profile must not spend analysis time or overwrite a warm
    pair memo. Proposal/reprofile and diagnose phases intentionally retain the
    existing publish semantics; only an already-established after profile is
    immutable here.
    """
    session_id, root = _normalize_target(session_id, root)
    if not isinstance(after_profile, LocalProfileReference):
        raise TypeError("after_profile must be a LocalProfileReference")
    snapshot = SessionStore(root).load(session_id)
    if snapshot.state.phase not in {"propose", "reprofile", "diagnose"} and (
        snapshot.state.after_profile != after_profile
    ):
        raise ValueError("session after profile does not match the after profile being diffed")
    return snapshot


def publish_session_decision(
    *,
    session_id: str,
    decision: str,
    reason: str,
    decider: str | None = None,
    root: str | os.PathLike[str] = DEFAULT_SESSION_ROOT,
) -> SessionState:
    """Record accept/reject on a session's published diff.

    The store permits one decision per finding. An accepted finding remains a
    terminal session; a rejected finding is recorded in ``decisions.json`` and
    returns the session to ``propose`` so another finding can be evaluated.
    ``publish_decision`` raises if the diff is missing or the finding was already
    decided; both are surfaced to the caller unchanged.
    """
    session_id, root = _normalize_target(session_id, root)
    if not reason or not reason.strip():
        raise ValueError("a decision requires a reason")
    store = SessionStore(root)
    with store.writer(session_id) as writer:
        state, _, _ = writer.publish_decision(decision, reason, decider=decider)
    return state


def session_dir(
    session_id: str,
    root: str | os.PathLike[str] = DEFAULT_SESSION_ROOT,
) -> Path:
    """Return the on-disk directory for ``session_id`` under ``root``."""
    location = session_location(session_id, root=root)
    return location.directory


def load_session(
    session: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] = DEFAULT_SESSION_ROOT,
) -> SessionSnapshot:
    """Load a session through the transport-neutral handoff facade."""
    location = session_location(session, root=root)
    return location.store().load(location.session_id)


def session_payload(
    session: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] = DEFAULT_SESSION_ROOT,
) -> dict[str, Any]:
    """Return the JSON-shaped handoff contract for plugin/MCP adapters.

    This is a projection only: SessionStore remains the source of truth and
    callers cannot mutate a session by changing the returned dictionaries.
    """
    location = session_location(session, root=root)
    snapshot = location.store().load(location.session_id)
    return {
        "schema_version": "0.1",
        "session": snapshot.state.to_dict(),
        "findings": snapshot.findings.to_dict() if snapshot.findings else None,
        "proposal": snapshot.proposal.to_dict() if snapshot.proposal else None,
        "runspec": snapshot.runspec.to_dict() if snapshot.runspec else None,
        "diff": dict(snapshot.diff) if snapshot.diff is not None else None,
        "decisions": [dict(decision) for decision in snapshot.decisions],
    }


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
