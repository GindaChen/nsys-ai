"""Headless ``nsys-ai diagnose`` — thin front door over EvidenceBuilder + SessionStore.

Handlers unpack argparse Namespace values and call :func:`run_diagnose`. The
analysis itself is the existing default skill pack; this module only composes
publication and user-facing output.
"""

from __future__ import annotations

import os
import sys
from typing import TextIO

from .exceptions import NsysAiError, ProfileError
from .profile import Profile, resolve_profile_path
from .profile_runner import build_local_profile_reference
from .session_cli import (
    publish_session_findings,
    resolve_session_id,
    resolve_session_location,
    session_argument,
    session_dir,
)
from .session_store import SessionStore


class DiagnoseCommandError(NsysAiError):
    """The diagnose front door could not complete."""

    error_code = "DIAGNOSE_COMMAND_FAILED"


def run_diagnose(
    *,
    profile_path: str | os.PathLike[str] | None = None,
    session_id: str | None = None,
    gpu: int = 0,
    trim: tuple[int, int] | None = None,
    web: bool = False,
    port: int = 8144,
    open_browser: bool = True,
    session_root: str | os.PathLike[str] = ".nsys-ai/sessions",
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run the default evidence pack once and publish findings into a session.

    Parameters are explicit (profile path, session id, gpu) so MCP and other
    callers can reuse this path without fabricating an argparse Namespace.

    Defaults match ``evidence build`` for the parameters this verb forwards
    (``gpu`` defaults to 0). It does not forward ``--analyzers``, ``--format``,
    or ``--output``: the default pack always runs, evidence prints to stdout,
    and findings land in the session.

    ``--web`` with only ``session_id`` opens the persisted session without
    re-running analysis.
    """
    if profile_path is None:
        if not web or not session_id:
            raise DiagnoseCommandError(
                "diagnose requires a profile, or --session <id> --web to reopen"
            )
        return _open_existing_session_web(
            session_id=session_id,
            port=port,
            open_browser=open_browser,
            session_root=session_root,
            stderr=stderr,
        )

    if not os.fspath(profile_path):
        raise DiagnoseCommandError("profile path is required")

    raw = os.path.abspath(os.path.expanduser(os.fspath(profile_path)))
    try:
        sqlite_path = resolve_profile_path(raw)
    except (OSError, ProfileError, TypeError, ValueError) as exc:
        raise DiagnoseCommandError(f"could not resolve profile: {exc}") from exc

    from .evidence_builder import EvidenceBuilder

    try:
        with Profile(sqlite_path) as prof:
            builder = EvidenceBuilder(prof, device=gpu, trim=trim)
            # Analysis completes before any session writer lease is taken.
            report = builder.build()
            before = build_local_profile_reference(prof.path)
    except (OSError, ProfileError, TypeError, ValueError) as exc:
        raise DiagnoseCommandError(f"diagnose failed: {exc}") from exc

    location = resolve_session_location(
        None if session_id == "" else session_id,
        root=session_root,
    )
    if location is None:
        resolved_id = resolve_session_id(None, before=before)
    else:
        resolved_id = location.session_id
        session_root = location.root
    try:
        publish_session_findings(
            session_id=resolved_id,
            report=report,
            before_profile=before,
            root=session_root,
        )
    except (TypeError, ValueError, ProfileError) as exc:
        raise DiagnoseCommandError(str(exc)) from exc

    directory = session_dir(resolved_id, root=session_root)
    session_ref = session_argument(resolved_id, root=session_root)
    print(f"Session: {resolved_id}", file=stdout)
    print(f"Findings artifact: {directory / 'findings.json'}", file=stdout)
    print(f"── Evidence Findings ({len(report.findings)}) ──", file=stdout)
    sev_icons = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
    for finding in report.findings:
        icon = sev_icons.get(finding.severity, "⚪")
        dur_ms = (finding.end_ns - finding.start_ns) / 1e6 if finding.end_ns else 0
        print(f"  {icon} [{finding.type}] {finding.label}  ({dur_ms:.1f}ms)", file=stdout)
        if finding.note:
            print(f"      {finding.note}", file=stdout)

    if report.skipped:
        print(f"── Limitations / skipped analyses ({len(report.skipped)}) ──", file=stdout)
        for entry in report.skipped:
            print(
                f"  {entry.analyzer} ({entry.skill}) — skipped: {entry.reason}",
                file=stdout,
            )
    else:
        print("── Limitations / skipped analyses ──", file=stdout)
        print("  (none)", file=stdout)

    top_id = next((f.id for f in report.findings if f.id), None)
    if top_id:
        print(
            "Next: nsys-ai propose --session "
            f"{session_ref} --finding-id {top_id} --runspec <runspec.json>",
            file=stdout,
        )
    else:
        print(
            "Next: no actionable finding id to propose from; "
            f"re-open with nsys-ai diagnose --session {session_ref} --web",
            file=stdout,
        )
    print(
        f"Or open the same session: nsys-ai diagnose --session {session_ref} --web",
        file=stdout,
    )

    if web:
        return _open_session_web(
            before_path=before.path,
            session_id=resolved_id,
            gpu=gpu,
            trim=trim,
            port=port,
            open_browser=open_browser,
            session_root=session_root,
            stderr=stderr,
        )
    return 0


def _open_existing_session_web(
    *,
    session_id: str,
    port: int,
    open_browser: bool,
    session_root: str | os.PathLike[str],
    stderr: TextIO,
) -> int:
    store = SessionStore(session_root)
    try:
        snapshot = store.load(session_id)
    except (OSError, ProfileError, TypeError, ValueError) as exc:
        raise DiagnoseCommandError(str(exc)) from exc
    before = snapshot.state.before_profile
    if before is None:
        raise DiagnoseCommandError(
            f"session {session_id} has no before profile to open in --web"
        )
    if snapshot.findings is None:
        print(
            f"Limitation: session {session_id} has no findings.json yet; "
            "open will show an empty overlay until diagnose runs with a profile.",
            file=stderr,
        )
    return _open_session_web(
        before_path=before.path,
        session_id=session_id,
        gpu=0,
        trim=None,
        port=port,
        open_browser=open_browser,
        session_root=session_root,
        stderr=stderr,
    )


def _open_session_web(
    *,
    before_path: str,
    session_id: str,
    gpu: int,
    trim: tuple[int, int] | None,
    port: int,
    open_browser: bool,
    session_root: str | os.PathLike[str],
    stderr: TextIO,
) -> int:
    """Open timeline-web on an already-published session without re-analysis."""
    from .web import serve_timeline

    try:
        with Profile(before_path) as prof:
            devices = list(prof.meta.devices) if prof.meta.devices else [gpu]
            serve_timeline(
                prof,
                devices,
                trim,
                port=port,
                open_browser=open_browser,
                loop_before=before_path,
                session=session_id,
                session_root=session_root,
            )
    except (OSError, ProfileError, TypeError, ValueError) as exc:
        print(f"Error: could not open session web view: {exc}", file=stderr)
        return 2
    return 0
