"""Synchronous local execution of a :class:`~nsys_ai.runspec.RunSpec`.

The local runner owns capture process lifecycle and artifact production.  It
does not own CLI presentation, session layout, or remote execution.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess  # nosec B404
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol

from .connection import DB_ERRORS
from .exceptions import NsysAiError
from .fingerprint import get_profile_id
from .profile import Profile, resolve_profile_path
from .runspec import RunSpec, RunSpecError, build_nsys_profile_argv, validate_secret_boundaries


class RunStatus(str, Enum):
    """Terminal outcome of a local profile run."""

    LAUNCH_FAILED = "launch_failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    NSYS_FAILED = "nsys_failed"
    APPLICATION_FAILED = "application_failed"
    EXPORT_FAILED = "export_failed"
    INVALID_PROFILE = "invalid_profile"
    SUCCEEDED = "succeeded"


class RunStage(str, Enum):
    """Observable stages emitted to a progress callback."""

    PREPARING = "preparing"
    CAPTURING = "capturing"
    EXPORTING = "exporting"
    VALIDATING = "validating"
    FINISHED = "finished"


@dataclass(frozen=True)
class RunProgress:
    """One immutable progress notification."""

    stage: RunStage
    elapsed_seconds: float


@dataclass(frozen=True)
class LocalProfileReference:
    """Validated identity and schema metadata for a local SQLite export."""

    path: str
    profile_id: str
    schema_version: str | None
    product_version: str | None
    kernel_count: int


@dataclass(frozen=True)
class RunTimings:
    """Wall-clock durations for the stages completed by a run."""

    started_at_utc: str
    capture_seconds: float
    export_seconds: float
    validation_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class RunResult:
    """Artifacts and deterministic terminal classification for a run."""

    status: RunStatus
    nsys_return_code: int | None
    application_return_code: int | None
    timings: RunTimings
    runspec_path: str
    stdout_path: str
    stderr_path: str
    report_path: str | None = None
    sqlite_path: str | None = None
    profile: LocalProfileReference | None = None
    detail: str | None = None


class _CancellationEvent(Protocol):
    def is_set(self) -> bool: ...


Cancellation = Callable[[], bool] | _CancellationEvent
ProgressCallback = Callable[[RunProgress], None]

_POLL_SECONDS = 0.05
_TERMINATION_GRACE_SECONDS = 2.0


class _EmptyProfileError(ValueError):
    """A readable profile has no GPU kernel rows."""


class LocalProfileRunner:
    """Run one local ``nsys profile`` capture into an artifact directory.

    ``nsys`` exposes one process return code for both its own work and the
    wrapped application.  A non-zero return with a non-empty report is therefore
    classified as ``application_failed`` by inference; it is not a separately
    observed application exit code.

    Export uses the existing blocking :func:`resolve_profile_path` API.  The
    cancellation token is consequently observed during capture, not while that
    shared export call is in progress.
    """

    def __init__(self, artifact_dir: str | os.PathLike[str], nsys_executable: str = "nsys"):
        self.artifact_dir = Path(artifact_dir)
        self.nsys_executable = os.fspath(nsys_executable)

    def run(
        self,
        spec: RunSpec,
        progress_callback: ProgressCallback | None = None,
        cancellation: Cancellation | None = None,
    ) -> RunResult:
        """Capture, export, and validate one local profile.

        Invalid inputs and secret-boundary failures raise :class:`RunSpecError`
        before the artifact directory or any file is created.  All failures
        after that boundary are returned as a typed :class:`RunResult`.
        """
        self._validate_inputs(spec, progress_callback, cancellation)
        resolved_secrets = self._resolve_declared_secrets(spec)
        validate_secret_boundaries(spec, resolved_secrets)

        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        capture_seconds = 0.0
        export_seconds = 0.0
        validation_seconds = 0.0

        artifact_dir = self.artifact_dir.absolute()
        runspec_path = artifact_dir / "runspec.json"
        stdout_path = artifact_dir / "stdout.log"
        stderr_path = artifact_dir / "stderr.log"
        report_base = artifact_dir / "profile"
        report_path = report_base.with_suffix(".nsys-rep")

        def progress(stage: RunStage) -> None:
            if progress_callback is not None:
                progress_callback(RunProgress(stage, time.monotonic() - started))

        def result(
            status: RunStatus,
            *,
            return_code: int | None = None,
            report: Path | None = None,
            sqlite: Path | None = None,
            profile: LocalProfileReference | None = None,
            detail: str | None = None,
        ) -> RunResult:
            progress(RunStage.FINISHED)
            return RunResult(
                status=status,
                nsys_return_code=return_code,
                # nsys exposes no separate application exit code.  Keep the
                # unknown value explicit even for the inferred failure status.
                application_return_code=None,
                timings=RunTimings(
                    started_at_utc=started_at,
                    capture_seconds=capture_seconds,
                    export_seconds=export_seconds,
                    validation_seconds=validation_seconds,
                    total_seconds=time.monotonic() - started,
                ),
                runspec_path=str(runspec_path),
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                report_path=str(report) if report is not None else None,
                sqlite_path=str(sqlite) if sqlite is not None else None,
                profile=profile,
                detail=detail,
            )

        if self._cancelled(cancellation):
            return result(RunStatus.CANCELLED)

        progress(RunStage.PREPARING)
        try:
            artifact_dir.mkdir(parents=True, exist_ok=False)
            runspec_path.write_bytes(spec.canonical_json_bytes())
        except OSError as exc:
            return result(RunStatus.LAUNCH_FAILED, detail=f"artifact setup failed: {exc}")

        environment = self._build_environment(spec, resolved_secrets)
        executable = shutil.which(self.nsys_executable, path=environment.get("PATH"))
        if executable is not None:
            executable = os.path.abspath(executable)
        elif os.path.isfile(self.nsys_executable):
            executable = os.path.abspath(self.nsys_executable)
        nsys_argv = build_nsys_profile_argv(
            spec, report_base, nsys_executable=executable or self.nsys_executable
        )
        cwd = self._resolve_cwd(spec)

        progress(RunStage.CAPTURING)
        capture_started = time.monotonic()
        process: subprocess.Popen[bytes] | None = None
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                try:
                    process = subprocess.Popen(  # nosec B603
                        nsys_argv,
                        cwd=cwd,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        shell=False,
                        start_new_session=True,
                    )
                except (OSError, ValueError) as exc:
                    capture_seconds = time.monotonic() - capture_started
                    return result(
                        RunStatus.LAUNCH_FAILED,
                        detail=f"capture process could not be launched: {type(exc).__name__}",
                    )

                terminal_status: RunStatus | None = None
                while process.poll() is None:
                    if self._cancelled(cancellation):
                        terminal_status = RunStatus.CANCELLED
                        break
                    if (
                        spec.timeout_seconds is not None
                        and time.monotonic() - capture_started >= spec.timeout_seconds
                    ):
                        terminal_status = RunStatus.TIMED_OUT
                        break
                    time.sleep(_POLL_SECONDS)

                if terminal_status is not None:
                    self._terminate_process_group(process)
                    capture_seconds = time.monotonic() - capture_started
                    return result(
                        terminal_status,
                        return_code=process.poll(),
                        report=report_path if report_path.exists() else None,
                    )
                return_code = process.wait()
        finally:
            if process is not None and process.poll() is None:
                self._terminate_process_group(process)

        capture_seconds = time.monotonic() - capture_started
        report_exists = report_path.exists()
        report_nonempty = report_exists and report_path.stat().st_size > 0
        if return_code != 0:
            # Nsight gives the runner no independent application return code.
            status = RunStatus.APPLICATION_FAILED if report_nonempty else RunStatus.NSYS_FAILED
            return result(
                status,
                return_code=return_code,
                report=report_path if report_exists else None,
                detail=(
                    "non-zero nsys return with report; application failure inferred"
                    if report_nonempty
                    else "nsys returned non-zero without a usable report"
                ),
            )
        if not report_exists:
            return result(
                RunStatus.NSYS_FAILED,
                return_code=return_code,
                detail="nsys completed without producing a report",
            )
        if not report_nonempty:
            return result(
                RunStatus.INVALID_PROFILE,
                return_code=return_code,
                report=report_path,
                detail="nsys produced an empty report",
            )

        progress(RunStage.EXPORTING)
        export_started = time.monotonic()
        try:
            sqlite_path = Path(
                resolve_profile_path(str(report_path), nsys_executable=executable)
            ).absolute()
        except (NsysAiError, OSError, subprocess.SubprocessError) as exc:
            export_seconds = time.monotonic() - export_started
            return result(
                RunStatus.EXPORT_FAILED,
                return_code=return_code,
                report=report_path,
                detail=f"profile export failed: {type(exc).__name__}",
            )
        export_seconds = time.monotonic() - export_started

        progress(RunStage.VALIDATING)
        validation_started = time.monotonic()
        try:
            with Profile(str(sqlite_path), cache_mode="direct") as profile:
                if profile.meta.kernel_count <= 0:
                    raise _EmptyProfileError("profile contains no GPU kernel activity")
                reference = LocalProfileReference(
                    path=str(sqlite_path),
                    profile_id=get_profile_id(
                        profile.conn, fallback_path=str(sqlite_path.absolute())
                    ),
                    schema_version=profile.schema.schema_version,
                    product_version=profile.schema.version,
                    kernel_count=profile.meta.kernel_count,
                )
        except (NsysAiError, OSError, *DB_ERRORS, _EmptyProfileError) as exc:
            validation_seconds = time.monotonic() - validation_started
            return result(
                RunStatus.INVALID_PROFILE,
                return_code=return_code,
                report=report_path,
                sqlite=sqlite_path,
                detail=f"profile validation failed: {type(exc).__name__}",
            )
        validation_seconds = time.monotonic() - validation_started
        return result(
            RunStatus.SUCCEEDED,
            return_code=return_code,
            report=report_path,
            sqlite=sqlite_path,
            profile=reference,
        )

    @staticmethod
    def _validate_inputs(
        spec: RunSpec,
        progress_callback: ProgressCallback | None,
        cancellation: Cancellation | None,
    ) -> None:
        if not isinstance(spec, RunSpec):
            raise RunSpecError("spec must be a RunSpec")
        if progress_callback is not None and not callable(progress_callback):
            raise RunSpecError("progress_callback must be callable or null")
        if cancellation is not None and not (
            callable(cancellation) or callable(getattr(cancellation, "is_set", None))
        ):
            raise RunSpecError("cancellation must be callable, an event, or null")

    @staticmethod
    def _resolve_declared_secrets(spec: RunSpec) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for name in spec.environment.secrets:
            if name not in os.environ:
                raise RunSpecError(f"declared secret {name} is not set in the runner environment")
            resolved[name] = os.environ[name]
        return resolved

    @staticmethod
    def _build_environment(spec: RunSpec, secrets: Mapping[str, str]) -> dict[str, str]:
        environment = dict(os.environ) if spec.environment.policy == "inherit" else {}
        environment.update(spec.environment.public)
        environment.update(secrets)
        return environment

    @staticmethod
    def _resolve_cwd(spec: RunSpec) -> str:
        if spec.repository is not None:
            return str((Path(spec.repository).expanduser() / spec.cwd).absolute())
        return str(Path(spec.cwd).expanduser().absolute())

    @staticmethod
    def _cancelled(cancellation: Cancellation | None) -> bool:
        if cancellation is None:
            return False
        if callable(cancellation):
            return bool(cancellation())
        return bool(cancellation.is_set())

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
        """Terminate the complete session, escalating after a bounded grace."""
        process_group = process.pid
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return

        deadline = time.monotonic() + _TERMINATION_GRACE_SECONDS
        while time.monotonic() < deadline:
            process.poll()
            try:
                os.killpg(process_group, 0)
            except ProcessLookupError:
                process.wait()
                return
            time.sleep(_POLL_SECONDS)

        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            # The group has received SIGKILL; avoid an unbounded wait if the
            # platform cannot promptly reap the group leader.
            pass
