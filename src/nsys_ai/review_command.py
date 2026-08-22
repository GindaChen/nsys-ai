"""Headless ``nsys-ai review`` — thin front door over canonical diff + session resume.

Handlers unpack argparse Namespace values and call :func:`run_review`.

``review`` does not create a session. With ``--session`` it resumes an existing
run and presents the decision path. With ``<before> <after>`` and no session it
runs the canonical diff, prints the result to stdout and the next command to
stderr, and leaves ``.nsys-ai/`` untouched. Diff analysis and the all-GPU shape
stay in :func:`nsys_ai.diff.diff_profiles` /
:func:`nsys_ai.diff.diff_profiles_all_gpus`; rendering stays in
:mod:`nsys_ai.diff_render`. ``review <before> <after>`` uses the same canonical
renderer as ``diff <before> <after> --no-ai``; stdout is byte-identical and the
two next-step hints are deliberately written to stderr.
"""

from __future__ import annotations

import os
import sys
from typing import TextIO

from .exceptions import NsysAiError, ProfileError
from .profile import Profile
from .profile import open as open_profile
from .session_cli import (
    project_loop_state,
    resolve_session_location,
    session_argument,
    session_dir,
)
from .session_store import SessionSnapshot, SessionStore


class ReviewCommandError(NsysAiError):
    """The review front door could not complete."""

    error_code = "REVIEW_COMMAND_FAILED"


def run_review(
    *,
    before_path: str | os.PathLike[str] | None = None,
    after_path: str | os.PathLike[str] | None = None,
    session_id: str | os.PathLike[str] | None = None,
    gpu: int | None = None,
    trim: tuple[int, int] | None = None,
    web: bool = False,
    port: int = 8144,
    open_browser: bool = True,
    session_root: str | os.PathLike[str] = ".nsys-ai/sessions",
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Resume a session decision path, or compare a before/after pair without a session.

    Pass either ``before_path`` + ``after_path`` (no session ownership), or
    ``session_id`` to resume. The pair form neither invents nor binds a session,
    and combining it with ``session_id`` is an error, not a silent ignore:
    ``ReviewCommandError`` is raised and the CLI exits 2.

    Parameters forwarded from ``diff`` keep the same defaults: ``gpu`` is
    ``None`` (all GPUs), ``trim`` optional. Not forwarded: ``--against``,
    ``--iteration``, ``--marker``, ``--format``, ``--output``, ``--limit``,
    ``--sort``, ``--no-ai``, ``--chat``, gates, or ``--accept``/``--reject``.
    ``review`` never records a decision itself; it reports the decision path and
    leaves the writing to ``SessionWriter.publish_decision``, reached from
    ``nsys-ai diff --session --accept/--reject``, the browser overlay, and
    ``nsys-ai optimize``.
    Internal diff uses the same limit=15 and sort=delta defaults as ``diff``.

    The pair form prints the terminal report ``diff`` prints, with no LLM call.
    Its stdout equals ``diff <before> <after> --no-ai`` byte for byte, including
    the all-GPU shape (executive summary, per-GPU overview, per-GPU top
    regressions) when ``gpu`` is ``None``. ``review`` also writes two next-step
    hints to ``stderr``; a shell that combines both streams therefore shows
    those hints after the report.
    """
    requested_session = session_id
    if session_id is not None:
        location = resolve_session_location(session_id, root=session_root)
        if location is not None:
            session_id = location.session_id
            session_root = location.root
    has_pair = before_path is not None or after_path is not None
    if session_id is not None and not has_pair:
        return _resume_review(
            session_id=session_id,
            web=web,
            port=port,
            open_browser=open_browser,
            session_root=session_root,
            session_argument_value=requested_session,
            stdout=stdout,
            stderr=stderr,
        )
    if before_path is None or after_path is None:
        raise ReviewCommandError(
            "review requires <before> <after>, or --session <id> to resume"
        )
    if session_id is not None:
        raise ReviewCommandError(
            "review <before> <after> does not create or bind a session; "
            "omit --session for a comparison, or pass --session <id> alone "
            "to resume an existing run"
        )
    if web:
        raise ReviewCommandError(
            "review --web requires --session <id> so the browser opens the "
            "same persisted session; pair form is a headless comparison only"
        )

    return _compare_pair(
        before_path=before_path,
        after_path=after_path,
        gpu=gpu,
        trim=trim,
        stdout=stdout,
        stderr=stderr,
    )


def _compare_pair(
    *,
    before_path: str | os.PathLike[str],
    after_path: str | os.PathLike[str],
    gpu: int | None,
    trim: tuple[int, int] | None,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Print the canonical diff report and handoff hints for a pair.

    The profile paths are opened with the caller's spelling (as ``diff`` does)
    because that spelling is echoed in the report header. Next-step hints go to
    ``stderr`` so redirected stdout stays exactly the canonical diff report.
    """
    from .ai.diff_narrative import offline_diff_narrative
    from .diff import STEP_TIME_REGRESSION_PCT, diff_profiles, diff_profiles_all_gpus
    from .diff_render import format_diff_terminal, format_diff_terminal_multi

    try:
        with (
            open_profile(os.fspath(before_path)) as before,
            open_profile(os.fspath(after_path)) as after,
        ):
            # Defaults match ``nsys-ai diff`` (limit=15, sort=delta, gpu=None).
            if gpu is not None:
                summary = diff_profiles(
                    before,
                    after,
                    gpu=gpu,
                    trim=trim,
                    limit=15,
                    sort="delta",
                    regression_pct=STEP_TIME_REGRESSION_PCT,
                )
                report = format_diff_terminal(
                    summary, narrative=offline_diff_narrative(summary)
                )
            else:
                global_summary, per_gpu = diff_profiles_all_gpus(
                    before,
                    after,
                    trim=trim,
                    limit=15,
                    sort="delta",
                    regression_pct=STEP_TIME_REGRESSION_PCT,
                )
                report = format_diff_terminal_multi(
                    global_summary,
                    per_gpu,
                    narrative=offline_diff_narrative(global_summary),
                )
    except (OSError, ProfileError, TypeError, ValueError) as exc:
        raise ReviewCommandError(f"review diff failed: {exc}") from exc

    print(report, end="", file=stdout)
    before_abs = os.path.abspath(os.path.expanduser(os.fspath(before_path)))
    print(
        f"Next: nsys-ai diagnose {before_abs}  "
        "# start a session when you need a recorded decision path",
        file=stderr,
    )
    print(
        "Or resume an existing run: nsys-ai review --session <id>",
        file=stderr,
    )
    return 0


def _resume_review(
    *,
    session_id: str,
    web: bool,
    port: int,
    open_browser: bool,
    session_root: str | os.PathLike[str],
    session_argument_value: str | os.PathLike[str] | None = None,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    if not session_id:
        raise ReviewCommandError(
            "--session without an id requires an existing session id; "
            "review does not derive or create a session from a profile pair"
        )
    store = SessionStore(session_root)
    try:
        snapshot = store.load(session_id)
    except (OSError, ProfileError, TypeError, ValueError) as exc:
        raise ReviewCommandError(str(exc)) from exc

    _print_decision_path(
        session_id,
        snapshot,
        session_root,
        session_argument_value=session_argument_value,
        stdout=stdout,
    )

    if web:
        before = snapshot.state.before_profile
        if before is None:
            raise ReviewCommandError(
                f"session {session_id} has no before profile to open in --web"
            )
        return _open_session_web(
            before_path=before.path,
            session_id=session_id,
            gpu=None,
            trim=before.trim_ns,
            port=port,
            open_browser=open_browser,
            session_root=session_root,
            stderr=stderr,
        )
    return 0


def _print_decision_path(
    session_id: str,
    snapshot: SessionSnapshot,
    session_root: str | os.PathLike[str],
    *,
    session_argument_value: str | os.PathLike[str] | None = None,
    stdout: TextIO,
) -> None:
    directory = session_dir(session_id, root=session_root)
    session_ref = session_argument(
        session_argument_value if session_argument_value is not None else session_id,
        root=session_root,
    )
    projected = project_loop_state(snapshot, session_dir_path=directory)
    print(f"Session: {session_id}", file=stdout)
    print(f"Phase: {projected.get('phase')}", file=stdout)
    if snapshot.diff is not None:
        print(f"Diff artifact: {directory / 'diff.json'}", file=stdout)
        verdict = (snapshot.diff or {}).get("verdict", "unknown")
        print(f"Verdict: {verdict}", file=stdout)
    else:
        print(f"Diff artifact: (none yet under {directory})", file=stdout)
        if snapshot.decisions:
            print(
                f"Decision history: {directory / 'decisions.json'}", file=stdout
            )

    decision = projected.get("decision")
    if decision:
        print(
            f"Decision path: {decision} — recorded at {projected.get('decision_path')}",
            file=stdout,
        )
        reason = projected.get("decision_reason") or ""
        if reason:
            print(f"Reason: {reason}", file=stdout)
    elif snapshot.diff is not None:
        # Both paths record the same decision contract. Accepted decisions stay
        # on diff.json; rejected findings are appended to decisions.json before
        # the session reopens at propose.
        before_ref = getattr(snapshot.state, "before_profile", None)
        after_ref = getattr(snapshot.state, "after_profile", None)
        before_path = getattr(before_ref, "path", None) or "<before>"
        after_path = getattr(after_ref, "path", None) or "<after>"
        print(
            "Decision path: undecided. Record it with: "
            f"nsys-ai diff {before_path} {after_path} --session {session_ref} "
            "--accept|--reject --reason TEXT",
            file=stdout,
        )
        print(
            f"Or in the browser: nsys-ai review --session {session_ref} --web",
            file=stdout,
        )
    else:
        top_id = None
        if snapshot.findings is not None:
            top_id = next((f.id for f in snapshot.findings.findings if f.id), None)
        if top_id:
            print(
                "Decision path: no diff yet. Continue the loop with: "
                f"nsys-ai propose --session {session_ref} --finding-id {top_id} "
                "--runspec <runspec.json>",
                file=stdout,
            )
        else:
            print(
                "Decision path: no diff yet. Finish propose → reprofile → "
                f"diff --session {session_ref} before a decision can be recorded.",
                file=stdout,
            )


def _open_session_web(
    *,
    before_path: str,
    session_id: str,
    gpu: int | None,
    trim: tuple[int, int] | None,
    port: int,
    open_browser: bool,
    session_root: str | os.PathLike[str],
    stderr: TextIO,
) -> int:
    from .web import serve_timeline

    try:
        with Profile(before_path) as prof:
            if gpu is not None:
                devices: list[int] = [gpu]
            else:
                devices = list(prof.meta.devices) if prof.meta.devices else [0]
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
