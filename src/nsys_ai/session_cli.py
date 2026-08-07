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
from .session_store import SessionExistsError, SessionState, SessionStore


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


def session_dir(session_id: str, root: str | os.PathLike[str] = ".nsys-ai/sessions") -> Path:
    """Return the on-disk directory for ``session_id`` under ``root``."""
    return SessionStore(root).root / session_id
