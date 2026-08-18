"""Headless ``nsys-ai optimize`` — one front door over the verified loop.

``optimize`` owns no analysis. It composes the existing contracts in order —
:func:`nsys_ai.diagnose_command.run_diagnose`,
:func:`nsys_ai.propose_command.run_propose`,
:class:`nsys_ai.profile_runner.LocalProfileRunner`,
:func:`nsys_ai.diff.diff_profiles` and
:meth:`nsys_ai.session_store.SessionWriter.publish_decision` — and persists
every step through :class:`~nsys_ai.session_store.SessionStore`, so an
interrupted run resumes from the last published artifact instead of redoing it.

Two behaviours are load-bearing:

* It stops before re-profiling whenever the loop cannot honestly continue. The
  common case is an abstaining proposal (no verification RunSpec, or a finding
  with no suggested action); ``optimize`` reports which precondition failed and
  never fabricates a proposal or presents a partial run as a completed loop.
* Only a recorded decision exits 0. Every other outcome exits non-zero so a
  wrapping script can tell a completed loop from a stopped one.

Exit codes
----------
``0``
    The loop reached a recorded accept/reject decision (this run or an earlier
    one).
``1``
    A step ran and failed: the verification capture, or the diff.
``2``
    The loop stopped before a decision because a precondition was not met —
    an abstaining proposal, a finding with no id, a rejected publication.
``130``
    The verification capture was cancelled.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TextIO

from .annotation import EvidenceReport
from .diagnose_command import DiagnoseCommandError, run_diagnose
from .exceptions import NsysAiError, ProfileError
from .profile_command import (
    ProfileCommandError,
    default_output_leaf,
    detect_supported_trace_domains,
    discover_git_provenance,
    normalize_workload,
    resolve_nsys_executable,
    select_trace_domains,
)
from .profile_reference import LocalProfileReference
from .profile_runner import (
    LocalProfileRunner,
    RunProgress,
    RunStatus,
    build_local_profile_reference,
)
from .proposal import Proposal
from .propose_command import ProposeCommandError, run_propose
from .runspec import EnvironmentSpec, NsysTraceOptions, RunSpec, RunSpecError
from .session_cli import (
    project_loop_state,
    publish_session_diff,
    resolve_session_id,
    session_dir,
)
from .session_store import SessionNotFoundError, SessionSnapshot, SessionStore

#: Exit code for "the loop stopped before a decision, and said why".
STOPPED_EXIT_CODE = 2
#: Exit code for "a step ran and failed".
FAILED_EXIT_CODE = 1
#: Exit code for "the verification capture was cancelled".
CANCELLED_EXIT_CODE = 130

# Measured on merged main against a 3.5 GB capture (issue #28 cost note): the
# default pack is ~49 s with a warm Parquet cache and ~157 s cold, nearly all of
# it inside profile_health_manifest. Without this line the first minute of a
# large profile reads as a hang.
_DIAGNOSE_COST_NOTE = (
    "Step 1/5 diagnose: running the default skill pack. On a multi-GB capture "
    "this is ~50s with a warm cache and ~2-3min cold, before anything else in "
    "the loop starts."
)


class OptimizeCommandError(NsysAiError):
    """``optimize`` could not start: its own inputs are unusable."""

    error_code = "OPTIMIZE_COMMAND_FAILED"


def _resume_command(
    *, before_path: str, repo: str, session_id: str, workload: Sequence[str]
) -> str:
    """The single command that picks the loop up where it stopped."""
    return (
        f"nsys-ai optimize {before_path} --repo {repo} "
        f"--session {session_id} -- {' '.join(workload)}"
    )


def _select_finding_id(
    report: EvidenceReport, *, decided_ids: set[str] | frozenset[str] = frozenset()
) -> str | None:
    """Pick the finding to propose from: ranked order, actionable first.

    ``findings`` arrives already ranked by :func:`rank_findings`, so the first
    entry carrying both an id and a suggested action is the highest-upside
    finding that can produce a non-abstaining proposal. Falling back to the
    first identified finding keeps the abstention honest and specific ("no
    suggested action") rather than turning it into "no finding id".
    """
    for finding in report.findings:
        if finding.id and finding.id not in decided_ids and finding.suggested_actions:
            return finding.id
    for finding in report.findings:
        if finding.id and finding.id not in decided_ids:
            return finding.id
    return None


def _decision_for_verdict(verdict: str, step_time_delta_pct: float | None) -> tuple[str, str]:
    """Map a canonical diff verdict onto accept/reject plus a mandatory reason.

    The store accepts only ``accept`` or ``reject``, so a verdict that is not an
    improvement is recorded as a reject whose reason names the verdict verbatim.
    A reader must not be able to mistake "not measurably better" for "worse".
    """
    normalized = (verdict or "neutral").strip().lower()
    if step_time_delta_pct is None:
        measured = "step time was not comparable"
    else:
        measured = f"step time moved {step_time_delta_pct:+.2f}%"
    if normalized == "improvement_likely":
        return "accept", f"diff verdict {normalized}: {measured}"
    return "reject", f"diff verdict {normalized}: {measured}"


def _step_time_delta_pct(diff: Mapping[str, object]) -> float | None:
    step_time = diff.get("step_time")
    if not isinstance(step_time, Mapping):
        return None
    value = step_time.get("delta_pct")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _print_recorded_decision(
    session_id: str,
    snapshot: SessionSnapshot,
    session_root: str | os.PathLike[str],
    *,
    stdout: TextIO,
) -> None:
    directory = session_dir(session_id, root=session_root)
    projected = project_loop_state(snapshot, session_dir_path=directory)
    print(f"Session: {session_id}", file=stdout)
    print(f"Phase: {snapshot.state.phase}", file=stdout)
    print(f"Verdict: {projected.get('verdict')}", file=stdout)
    print(f"Decision: {projected.get('decision')}", file=stdout)
    reason = projected.get("decision_reason") or ""
    if reason:
        print(f"Reason: {reason}", file=stdout)
    artifact = projected.get("decision_path") or directory / "diff.json"
    print(f"Decision artifact: {artifact}", file=stdout)


def _session_quiescent(snapshot: SessionSnapshot) -> bool:
    """Return whether every identified finding has one durable decision."""
    if snapshot.findings is None or not snapshot.findings.findings:
        return False
    finding_ids = {finding.id for finding in snapshot.findings.findings if finding.id}
    if len(finding_ids) != len(snapshot.findings.findings):
        return False
    decided_ids = {
        decision.get("finding_id")
        for decision in snapshot.decisions
        if decision.get("finding_id")
    }
    return finding_ids.issubset(decided_ids)


def _print_verification_plan(
    proposal: Proposal, runspec: RunSpec, *, stdout: TextIO
) -> None:
    """Show the proposal and the exact RunSpec before anything is executed."""
    print("-- Proposal --", file=stdout)
    print(f"Summary: {proposal.summary}", file=stdout)
    for action in proposal.suggested_actions:
        print(f"  action: {action}", file=stdout)
    if proposal.expected_impact is not None:
        print(
            f"Expected impact: {proposal.expected_impact.headroom_ms} ms "
            f"({proposal.expected_impact.headroom_basis})",
            file=stdout,
        )
    print(f"Confidence: {proposal.confidence}", file=stdout)
    print("-- Verification RunSpec (about to execute) --", file=stdout)
    print(
        json.dumps(runspec.to_dict(), indent=2, sort_keys=True, allow_nan=False),
        file=stdout,
    )


def _build_runspec(
    *, repo: Path, workload: Sequence[str], nsys_executable: str, stderr: TextIO
) -> RunSpec:
    """Build the verification RunSpec from ``--repo`` and the workload argv."""
    repository, commit, runspec_cwd = discover_git_provenance(repo)
    if repository is None:
        # Not a Git checkout: still bind the run to the directory the user
        # named, so the recorded RunSpec is reproducible without a commit.
        repository, runspec_cwd = str(repo), "."
    supported = detect_supported_trace_domains(nsys_executable)
    trace = select_trace_domains(
        "auto", supported, warning=lambda message: print(f"Warning: {message}", file=stderr)
    )
    return RunSpec(
        argv=tuple(workload),
        cwd=runspec_cwd,
        repository=repository,
        commit=commit,
        environment=EnvironmentSpec(policy="inherit"),
        trace_options=NsysTraceOptions(trace=trace),
    )


def _resolve_repo(repo: str | os.PathLike[str], working_directory: Path) -> Path:
    path = Path(repo).expanduser()
    if not path.is_absolute():
        path = working_directory / path
    path = path.resolve()
    if not path.is_dir():
        raise OptimizeCommandError(f"--repo is not a directory: {path}")
    return path


def _load_snapshot(store: SessionStore, session_id: str) -> SessionSnapshot | None:
    try:
        return store.load(session_id)
    except SessionNotFoundError:
        return None


def _decision_recorded(snapshot: SessionSnapshot) -> bool:
    return isinstance(snapshot.diff, Mapping) and isinstance(
        snapshot.diff.get("decision"), Mapping
    )


def run_optimize(
    *,
    before_path: str | os.PathLike[str],
    repo: str | os.PathLike[str],
    workload: Sequence[str],
    session_id: str | None = None,
    nsys: str = "nsys",
    gpu: int = 0,
    trim: tuple[int, int] | None = None,
    session_root: str | os.PathLike[str] = ".nsys-ai/sessions",
    cwd: str | os.PathLike[str] | None = None,
    runner_factory: Callable[[Path, str], LocalProfileRunner] = LocalProfileRunner,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    environment: Mapping[str, str] = os.environ,
) -> int:
    """Drive diagnose -> propose -> capture -> diff -> decision over one session.

    Each step is skipped when the session already holds its artifact, so an
    interrupted run resumes rather than repeats. Returns one of the documented
    module-level exit codes; only a recorded decision returns ``0``.
    """
    working_directory = Path(cwd or Path.cwd()).resolve()
    repo_path = _resolve_repo(repo, working_directory)

    raw_before = os.path.abspath(os.path.expanduser(os.fspath(before_path)))
    try:
        from .profile import resolve_profile_path

        before_sqlite = resolve_profile_path(raw_before)
        before_ref = build_local_profile_reference(before_sqlite)
    except (OSError, ProfileError, TypeError, ValueError) as exc:
        raise OptimizeCommandError(f"could not resolve before profile: {exc}") from exc

    try:
        normalized = normalize_workload(workload)
    except ProfileCommandError as exc:
        raise OptimizeCommandError(str(exc)) from exc

    resolved_id = resolve_session_id(session_id or None, before=before_ref)
    resume = _resume_command(
        before_path=raw_before,
        repo=str(repo_path),
        session_id=resolved_id,
        workload=normalized,
    )
    store = SessionStore(session_root)

    def stopped(message: str, *, code: int = STOPPED_EXIT_CODE) -> int:
        print(f"Stopped: {message}", file=stderr)
        print(f"Resume: {resume}", file=stderr)
        return code

    snapshot = _load_snapshot(store, resolved_id)
    if snapshot is not None:
        # Before anything is reused or reported, the inputs this run was given
        # must be the inputs the session was built from. The session records
        # both; neither was compared to anything, so the recorded state won over
        # a contradicting argument silently -- including on the completed-loop
        # path below, which would report a decision reached by verifying a
        # different workload than the caller just named.
        recorded_before = snapshot.state.before_profile
        # Compare identity, not the reference: the session id is derived from
        # profile_id alone, so the same capture reached by another path -- a
        # moved artifact directory, a symlink, a container mount -- resolves to
        # this session and must not be told it is a different profile.
        if (
            recorded_before is not None
            and recorded_before.profile_id != before_ref.profile_id
        ):
            raise OptimizeCommandError(
                f"session {resolved_id} was opened on a different before profile "
                f"({recorded_before.path}); pass --session with a new id"
            )
        recorded_runspec = snapshot.runspec
        # The recorded argv was built from the normalized workload, so the raw
        # tokens are the wrong side of this comparison: normalization inserts
        # the interpreter for the documented `train.py` shorthand, which would
        # make every re-run of a Python workload look like a different one.
        if recorded_runspec is not None and tuple(recorded_runspec.argv) != normalized:
            raise OptimizeCommandError(
                f"session {resolved_id} verified a different workload.\n"
                f"  recorded: {' '.join(recorded_runspec.argv)}\n"
                f"  given:    {' '.join(normalized)}\n"
                "A decision recorded against one workload does not carry to another. "
                "Pass --session with a new id to verify this one."
            )
        if _decision_recorded(snapshot):
            print(
                "Loop already complete; reporting the recorded decision.", file=stdout
            )
            _print_recorded_decision(resolved_id, snapshot, session_root, stdout=stdout)
            return 0
        if _session_quiescent(snapshot):
            print(
                "Loop already complete; reporting the recorded decisions.",
                file=stdout,
            )
            _print_recorded_decision(resolved_id, snapshot, session_root, stdout=stdout)
            return 0

    # Step 1 — diagnose.
    if snapshot is None or snapshot.findings is None:
        print(_DIAGNOSE_COST_NOTE, file=stderr, flush=True)
        try:
            code = run_diagnose(
                profile_path=before_sqlite,
                session_id=resolved_id,
                gpu=gpu,
                trim=trim,
                web=False,
                session_root=session_root,
                stdout=stdout,
                stderr=stderr,
            )
        except DiagnoseCommandError as exc:
            return stopped(f"diagnose failed: {exc}")
        if code:
            return stopped(f"diagnose exited {code}", code=code)
        snapshot = _load_snapshot(store, resolved_id)
    else:
        print(
            f"Step 1/5 diagnose: reusing {len(snapshot.findings.findings)} "
            "findings already published to this session.",
            file=stderr,
        )

    if snapshot is None or snapshot.findings is None:
        return stopped("diagnose published no findings.json")

    # Step 2 — propose, or reuse the proposal this session already carries.
    if snapshot.proposal is None:
        decided_ids = {
            str(decision["finding_id"])
            for decision in snapshot.decisions
            if decision.get("finding_id")
        }
        finding_id = _select_finding_id(snapshot.findings, decided_ids=decided_ids)
        if finding_id is None:
            if snapshot.decisions:
                print(
                    "Loop already complete; no undecided finding remains to propose.",
                    file=stdout,
                )
                _print_recorded_decision(
                    resolved_id, snapshot, session_root, stdout=stdout
                )
                return 0
            return stopped(
                "no finding carries a stable id, so there is nothing to propose "
                "from; re-run diagnose or open the session with "
                f"nsys-ai diagnose --session {resolved_id} --web"
            )
        try:
            runspec = _build_runspec(
                repo=repo_path,
                workload=normalized,
                nsys_executable=resolve_nsys_executable(nsys),
                stderr=stderr,
            )
        except (ProfileCommandError, RunSpecError) as exc:
            return stopped(f"could not build the verification RunSpec: {exc}")

        print(
            f"Step 2/5 propose: deriving a proposal from finding {finding_id}.",
            file=stderr,
            flush=True,
        )
        code = _publish_proposal(
            session_id=resolved_id,
            finding_id=finding_id,
            runspec=runspec,
            session_root=session_root,
            stdout=stdout,
            stderr=stderr,
            environment=environment,
            resume=resume,
        )
        if code:
            return code
        snapshot = _load_snapshot(store, resolved_id)
        if snapshot is None:
            return stopped("session disappeared while publishing the proposal")

    proposal = snapshot.proposal
    if proposal is None:
        return stopped("propose published no proposal.json")

    if proposal.abstained:
        reason = proposal.abstention_reason or "no reason recorded"
        print(f"Abstained before re-profiling: {reason}", file=stdout)
        return stopped(
            "the proposal abstained, so nothing was re-profiled and no verdict "
            f"was reached. Precondition that failed: {reason}"
        )

    runspec = snapshot.runspec
    if runspec is None:
        # publish_proposal already rejects this combination; treat a store that
        # somehow holds one as a stop, never as a licence to invent a RunSpec.
        return stopped(
            "the session holds a non-abstaining proposal with no verification "
            "RunSpec, so there is nothing to execute"
        )

    # Step 3 — execute the recorded RunSpec, and Step 4 register the capture.
    if snapshot.state.after_profile is None:
        _print_verification_plan(proposal, runspec, stdout=stdout)
        outcome = _capture_after_profile(
            runspec=runspec,
            nsys=nsys,
            working_directory=working_directory,
            runner_factory=runner_factory,
            stdout=stdout,
            stderr=stderr,
        )
        if isinstance(outcome, int):
            if outcome == CANCELLED_EXIT_CODE:
                print(f"Resume: {resume}", file=stderr)
                return outcome
            return stopped("the verification capture did not produce a profile", code=outcome)
        try:
            with store.writer(resolved_id) as writer:
                writer.publish_after_profile(outcome)
        except (OSError, ProfileError, TypeError, ValueError) as exc:
            return stopped(f"could not register the after profile: {exc}")
        snapshot = _load_snapshot(store, resolved_id)
        if snapshot is None:
            return stopped("session disappeared while registering the after profile")
    else:
        print(
            "Step 3/5 capture: reusing the after profile already registered "
            f"({snapshot.state.after_profile.path}).",
            file=stderr,
        )

    after_ref = snapshot.state.after_profile
    if after_ref is None:
        return stopped("the session has no after profile to diff against")

    # Step 5 — canonical diff, then the decision it implies.
    if snapshot.diff is None:
        code = _publish_diff(
            session_id=resolved_id,
            before_ref=snapshot.state.before_profile or before_ref,
            after_ref=after_ref,
            trim=trim,
            session_root=session_root,
            stdout=stdout,
            stderr=stderr,
            resume=resume,
        )
        if code:
            return code
        snapshot = _load_snapshot(store, resolved_id)
        if snapshot is None:
            return stopped("session disappeared while publishing the diff")
    else:
        print(
            "Step 4/5 diff: reusing the diff already published "
            f"(verdict={snapshot.diff.get('verdict', 'unknown')}).",
            file=stderr,
        )

    if snapshot.diff is None:
        return stopped("diff published no diff.json")
    if _decision_recorded(snapshot):
        _print_recorded_decision(resolved_id, snapshot, session_root, stdout=stdout)
        return 0

    verdict = str(snapshot.diff.get("verdict") or "neutral")
    decision, reason = _decision_for_verdict(verdict, _step_time_delta_pct(snapshot.diff))
    print(f"Step 5/5 decide: recording {decision} for verdict {verdict}.", file=stderr)
    try:
        with store.writer(resolved_id) as writer:
            _state, _payload, warnings = writer.publish_decision(decision, reason)
    except (OSError, ProfileError, TypeError, ValueError) as exc:
        return stopped(f"could not record the decision: {exc}")
    for warning in warnings:
        print(f"Warning: {warning}", file=stderr)

    snapshot = _load_snapshot(store, resolved_id)
    if snapshot is None:
        return stopped("the decision was not persisted to diff.json")
    if not _decision_recorded(snapshot):
        if snapshot.decisions:
            latest = snapshot.decisions[-1]
            status = str(latest.get("status") or "")
            if status == "rejected":
                print(
                    "Decision recorded for finding "
                    f"{latest.get('finding_id')}; session remains open for another finding.",
                    file=stdout,
                )
                return 0
        return stopped("the decision was not persisted to the session")
    _print_recorded_decision(resolved_id, snapshot, session_root, stdout=stdout)
    return 0


def _publish_proposal(
    *,
    session_id: str,
    finding_id: str,
    runspec: RunSpec,
    session_root: str | os.PathLike[str],
    stdout: TextIO,
    stderr: TextIO,
    environment: Mapping[str, str],
    resume: str,
) -> int:
    """Run ``propose`` against the session, handing it the built RunSpec."""
    handle_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".json", prefix="nsys-ai-optimize-runspec-", delete=False
        ) as handle:
            handle.write(runspec.canonical_json_bytes())
            handle_path = handle.name
        try:
            code = run_propose(
                finding_id=finding_id,
                session_id=session_id,
                runspec_path=handle_path,
                session_root=session_root,
                stdout=stdout,
                stderr=stderr,
                environment=environment,
            )
        except (ProposeCommandError, RunSpecError, ProfileError, ValueError) as exc:
            print(f"Stopped: propose failed: {exc}", file=stderr)
            print(f"Resume: {resume}", file=stderr)
            return STOPPED_EXIT_CODE
        if code:
            print(f"Stopped: propose exited {code}", file=stderr)
            print(f"Resume: {resume}", file=stderr)
            return code or STOPPED_EXIT_CODE
    finally:
        if handle_path is not None:
            try:
                os.unlink(handle_path)
            except OSError:
                pass
    return 0


def _capture_after_profile(
    *,
    runspec: RunSpec,
    nsys: str,
    working_directory: Path,
    runner_factory: Callable[[Path, str], LocalProfileRunner],
    stdout: TextIO,
    stderr: TextIO,
) -> LocalProfileReference | int:
    """Execute the recorded RunSpec; return the capture or a failing exit code."""
    print("Step 3/5 capture: executing the recorded RunSpec.", file=stderr, flush=True)
    try:
        nsys_executable = resolve_nsys_executable(nsys)
    except ProfileCommandError as exc:
        print(f"Stopped: {exc}", file=stderr)
        return STOPPED_EXIT_CODE

    def progress(update: RunProgress) -> None:
        print(f"[{update.stage.value}] {update.elapsed_seconds:.1f}s", file=stderr, flush=True)

    runner = runner_factory(default_output_leaf(working_directory), nsys_executable)
    try:
        result = runner.run(runspec, progress_callback=progress)
    except KeyboardInterrupt:
        print("Verification capture cancelled.", file=stderr)
        return CANCELLED_EXIT_CODE
    except (OSError, ProfileError, RunSpecError) as exc:
        print(f"Verification capture failed: {exc}", file=stderr)
        return FAILED_EXIT_CODE

    if result.status is RunStatus.CANCELLED:
        print("Verification capture cancelled.", file=stderr)
        return CANCELLED_EXIT_CODE
    if result.status is not RunStatus.SUCCEEDED or result.profile is None:
        print(f"Verification capture failed: {result.detail or result.status.value}", file=stderr)
        for label, path in (("stdout", result.stdout_path), ("stderr", result.stderr_path)):
            if path:
                print(f"  {label}: {path}", file=stderr)
        return FAILED_EXIT_CODE

    print(f"After SQLite: {result.sqlite_path}", file=stdout)
    return result.profile


def _publish_diff(
    *,
    session_id: str,
    before_ref: LocalProfileReference,
    after_ref: LocalProfileReference,
    trim: tuple[int, int] | None,
    session_root: str | os.PathLike[str],
    stdout: TextIO,
    stderr: TextIO,
    resume: str,
) -> int:
    """Compute and publish the canonical all-GPU diff for this session."""
    from .diff import STEP_TIME_REGRESSION_PCT, diff_profiles
    from .diff_render import format_diff_terminal, to_diff_dict
    from .profile import Profile

    print("Step 4/5 diff: comparing the before and after captures.", file=stderr, flush=True)
    try:
        with Profile(before_ref.path) as before, Profile(after_ref.path) as after:
            # gpu=None is the global summary diff --session publishes; limit and
            # sort match the diff defaults so the artifact is the same shape.
            summary = diff_profiles(
                before,
                after,
                gpu=None,
                trim=trim,
                limit=15,
                sort="delta",
                regression_pct=STEP_TIME_REGRESSION_PCT,
            )
    except (OSError, ProfileError, TypeError, ValueError) as exc:
        print(f"Stopped: diff failed: {exc}", file=stderr)
        print(f"Resume: {resume}", file=stderr)
        return FAILED_EXIT_CODE

    payload = to_diff_dict(summary)
    # to_diff_dict carries Profile.path, which keeps the caller's spelling; the
    # store compares absolute paths against its own references.
    for side, reference in (("before", before_ref), ("after", after_ref)):
        entry = payload.get(side)
        if isinstance(entry, dict):
            entry["path"] = reference.path
    try:
        publish_session_diff(
            session_id=session_id,
            diff=payload,
            after_profile=after_ref,
            root=session_root,
        )
    except (OSError, ProfileError, TypeError, ValueError) as exc:
        print(f"Stopped: could not publish the diff: {exc}", file=stderr)
        print(f"Resume: {resume}", file=stderr)
        return STOPPED_EXIT_CODE

    print(format_diff_terminal(summary), end="", file=stdout)
    return 0
