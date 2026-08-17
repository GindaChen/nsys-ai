"""Synchronous local execution of a :class:`~nsys_ai.runspec.RunSpec`.

The local runner owns capture process lifecycle and artifact production.  It
does not own CLI presentation, session layout, or remote execution.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import sqlite3
import stat
import subprocess  # nosec B404
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Protocol

from .connection import DB_ERRORS
from .exceptions import NsysAiError, ProfileError
from .fingerprint import get_profile_id
from .profile import Profile, resolve_profile_path
from .profile_reference import (
    LocalProfileReference,
    inspect_local_profile_file,
    profile_stat_signature,
    validate_local_profile_reference,
)
from .runspec import (
    RunSpec,
    RunSpecError,
    build_nsys_profile_argv,
    validate_persisted_secret_strings,
    validate_secret_boundaries,
)


class RunStatus(str, Enum):
    """Terminal outcome of a local profile run."""

    ARTIFACT_SETUP_FAILED = "artifact_setup_failed"
    CAPTURE_LAUNCH_FAILED = "capture_launch_failed"
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


_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _open_profile_descriptor(profile_path: Path) -> tuple[int, os.stat_result]:
    try:
        resolved_path, path_metadata = inspect_local_profile_file(profile_path)
    except ValueError as exc:
        raise ProfileError(str(exc)) from None

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(resolved_path, flags)
    except FileNotFoundError:
        raise ProfileError("local profile path does not exist") from None
    except OSError:
        raise ProfileError("local profile path cannot be opened safely") from None
    try:
        descriptor_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_metadata.st_mode):
            raise ProfileError("local profile path is not a regular file")
        if profile_stat_signature(path_metadata) != profile_stat_signature(
            descriptor_metadata
        ):
            raise ProfileError("local profile changed while it was being opened")
        return descriptor, descriptor_metadata
    except Exception:
        os.close(descriptor)
        raise


def _assert_profile_descriptor_unchanged(
    descriptor: int,
    profile_path: Path,
    before: os.stat_result,
) -> None:
    try:
        descriptor_after = os.fstat(descriptor)
        resolved_after, path_after = inspect_local_profile_file(profile_path)
    except (OSError, ValueError):
        raise ProfileError("local profile changed during validation") from None
    expected = profile_stat_signature(before)
    if (
        profile_stat_signature(descriptor_after) != expected
        or profile_stat_signature(path_after) != expected
        or resolved_after != profile_path
        or not stat.S_ISREG(path_after.st_mode)
    ):
        raise ProfileError("local profile changed during validation")


def read_local_profile_under_guard(
    path: str | os.PathLike[str],
    *,
    resolved_secrets: Mapping[str, str] | None = None,
) -> tuple[LocalProfileReference, int]:
    """Validate an existing SQLite export; return its reference and GPU count.

    The source profile is opened in direct, read-only analysis mode. The
    function does not export, copy, cache, or publish the profile or create a
    session. ``resolved_secrets`` lets callers reject a declared secret that
    would otherwise be persisted as part of the absolute local path.

    The device count comes back alongside the reference because it is read here
    and nowhere cheaper: this open goes through a pinned descriptor whose
    identity is asserted unchanged afterwards, and ``profile.meta`` is already
    materialised for ``kernel_count``. A caller that re-opens the path to count
    devices pays for a second full ``GROUP BY deviceId`` scan and gets its
    answer from a file the swap guard no longer covers.
    """
    path_conversion_failed = False
    try:
        raw_path = os.fspath(path)
    except Exception:
        path_conversion_failed = True
        raw_path = None
    if path_conversion_failed:
        raise ProfileError("local profile path must be a path string")
    if not isinstance(raw_path, str):
        raise ProfileError("local profile path must be a path string")
    if not raw_path:
        raise ProfileError("local profile path must not be empty")
    if "\x00" in raw_path:
        raise ProfileError("local profile path must not contain NUL bytes")

    try:
        profile_path = Path(raw_path).expanduser().absolute()
    except (OSError, RuntimeError):
        raise ProfileError("local profile path cannot be normalized") from None
    if profile_path.suffix.lower() != ".sqlite":
        raise ProfileError("local profile path must name a .sqlite file")
    secret_error_detail: str | None = None
    try:
        validate_persisted_secret_strings(
            {"path": str(profile_path)},
            resolved_secrets if resolved_secrets is not None else {},
        )
    except Exception:
        declared_names: list[str] = []
        try:
            for name, value in (resolved_secrets or {}).items():
                if (
                    isinstance(name, str)
                    and _ENVIRONMENT_NAME.fullmatch(name)
                    and isinstance(value, str)
                    and value
                    and value in str(profile_path)
                ):
                    declared_names.append(name)
        except Exception:
            declared_names = []
        detail = (
            " " + ", ".join(sorted(declared_names))
            if declared_names
            else ""
        )
        secret_error_detail = detail
    if secret_error_detail is not None:
        raise ProfileError(
            f"local profile path contains a resolved secret{secret_error_detail}"
        )

    descriptor, before = _open_profile_descriptor(profile_path)
    connection: sqlite3.Connection | None = None
    validation_succeeded = False
    unexpected_error: Exception | None = None
    kernel_count = None
    device_count = 0
    profile_id = None
    schema_version = None
    product_version = None
    try:
        descriptor_uri = f"file:/proc/self/fd/{descriptor}?mode=ro&immutable=1"
        connection = sqlite3.connect(
            descriptor_uri,
            uri=True,
            check_same_thread=False,
        )
        connection.execute("PRAGMA query_only = ON")
        with Profile._from_conn(connection) as profile:
            kernel_count = profile.meta.kernel_count
            device_count = len(profile.meta.devices or ())
            profile_id = get_profile_id(
                profile.conn, fallback_path=str(profile_path)
            )
            schema_version = profile.schema.schema_version
            product_version = profile.schema.version
        validation_succeeded = True
    except (NsysAiError, OSError, *DB_ERRORS):
        validation_succeeded = False
    except Exception as exc:
        unexpected_error = exc
    finally:
        if connection is not None:
            try:
                connection.close()
            except (OSError, *DB_ERRORS):
                validation_succeeded = False
            except Exception as exc:
                unexpected_error = unexpected_error or exc

    changed_error = None
    try:
        _assert_profile_descriptor_unchanged(descriptor, profile_path, before)
    except ProfileError as exc:
        changed_error = exc
    finally:
        os.close(descriptor)
    if changed_error is not None:
        raise changed_error from None
    if unexpected_error is not None:
        raise unexpected_error
    if not validation_succeeded:
        raise ProfileError("local profile is not a valid Nsight SQLite export") from None

    reference = LocalProfileReference(
        path=str(profile_path),
        profile_id=profile_id,
        schema_version=schema_version,
        product_version=product_version,
        kernel_count=kernel_count,
    )
    try:
        validated = validate_local_profile_reference(reference, require_file=True)
    except (TypeError, ValueError) as exc:
        raise ProfileError(str(exc)) from None
    return validated, device_count


def build_local_profile_reference(
    path: str | os.PathLike[str],
    *,
    resolved_secrets: Mapping[str, str] | None = None,
) -> LocalProfileReference:
    """The reference alone, for the callers that do not need the GPU count."""
    reference, _ = read_local_profile_under_guard(
        path, resolved_secrets=resolved_secrets
    )
    return reference


def check_expected_gpu_count(spec: RunSpec, observed: int) -> str | None:
    """Return why ``expected_gpu_count`` is unmet, or None when it holds.

    ``expected_gpu_count`` was recorded into the RunSpec and never read, so a
    single-GPU capture declared as eight was published with the declaration
    intact and every consumer downstream — per-GPU scaling, MFU, the diff's
    topology check — inherited a falsehood the artifact asserted about itself.
    A declaration the capture contradicts fails the capture.

    ``expected_rank_count`` stays unchecked here on purpose: a rank is a
    process in a distributed job, one local capture watches one launch, and
    counting processes in the export would answer a different question under
    the same name.

    The count is required rather than optional: it is read inside the guarded
    window, which either yields a number or raises, and the caller turns that
    raise into ``INVALID_PROFILE``. Accepting None here would restate a
    fail-closed rule that no caller can reach and no test can pin.
    """
    expected = spec.expected_gpu_count
    if expected is None:
        return None
    if observed != expected:
        return (
            f"declared expected_gpu_count={expected} but the capture recorded kernels on "
            f"{observed} GPU(s)"
        )
    return None


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
    runspec_path: str | None
    stdout_path: str | None
    stderr_path: str | None
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


class _CaptureLaunchError(Exception):
    """The nsys capture process could not be started."""

    def __init__(self, cause: Exception):
        self.cause_name = type(cause).__name__
        super().__init__(self.cause_name)


class _ArtifactSetupError(Exception):
    """A private runner-owned artifact file could not be created."""

    def __init__(self, cause: Exception):
        self.cause_name = type(cause).__name__
        super().__init__(self.cause_name)


class LocalProfileRunner:
    """Run one local ``nsys profile`` capture into an artifact directory.

    ``nsys`` exposes one process return code for both its own work and the
    wrapped application.  A non-zero return with a non-empty report is therefore
    classified as ``application_failed`` by inference; it is not a separately
    observed application exit code.

    Export uses the existing blocking :func:`resolve_profile_path` API.  The
    cancellation token is consequently observed during capture, not while that
    shared export call is in progress. Progress callbacks are observers:
    ordinary callback exceptions are isolated from the run, while
    :class:`BaseException` is not intercepted.
    """

    def __init__(self, artifact_dir: str | os.PathLike[str], nsys_executable: str = "nsys"):
        self.artifact_dir = Path(artifact_dir)
        self.nsys_executable = nsys_executable

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
        nsys_executable = self._validate_inputs(spec, progress_callback, cancellation)
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
        sqlite_candidate = report_base.with_suffix(".sqlite")
        created_files: set[Path] = set()

        def progress(stage: RunStage) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(RunProgress(stage, time.monotonic() - started))
                except Exception:
                    # Progress is an observer. Its ordinary failures must not
                    # alter capture lifecycle or discard a completed result.
                    pass

        def created_path(path: Path | None) -> str | None:
            return str(path) if path is not None and path in created_files else None

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
                runspec_path=created_path(runspec_path),
                stdout_path=created_path(stdout_path),
                stderr_path=created_path(stderr_path),
                report_path=created_path(report),
                sqlite_path=created_path(sqlite),
                profile=profile,
                detail=detail,
            )

        if self._cancelled(cancellation):
            return result(RunStatus.CANCELLED)

        progress(RunStage.PREPARING)
        try:
            artifact_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
            artifact_dir.chmod(0o700)
            with self._create_private_file(runspec_path) as runspec_file:
                created_files.add(runspec_path)
                runspec_file.write(spec.canonical_json_bytes())
        except (OSError, _ArtifactSetupError) as exc:
            return result(
                RunStatus.ARTIFACT_SETUP_FAILED,
                detail=f"artifact setup failed: {type(exc).__name__}",
            )

        environment = self._build_environment(spec, resolved_secrets)
        executable = shutil.which(nsys_executable, path=environment.get("PATH"))
        if executable is not None:
            executable = os.path.abspath(executable)
        elif os.path.isfile(nsys_executable):
            executable = os.path.abspath(nsys_executable)
        nsys_argv = build_nsys_profile_argv(
            spec, report_base, nsys_executable=executable or nsys_executable
        )
        cwd = self._resolve_cwd(spec)

        progress(RunStage.CAPTURING)
        if self._cancelled(cancellation):
            return result(RunStatus.CANCELLED)
        capture_started = time.monotonic()
        process: subprocess.Popen[bytes] | None = None
        try:
            with self._create_private_file(stdout_path) as stdout:
                created_files.add(stdout_path)
                with self._create_private_file(stderr_path) as stderr:
                    created_files.add(stderr_path)
                    return_code, stop_status, process = self._capture_process(
                        nsys_argv,
                        cwd,
                        environment,
                        stdout,
                        stderr,
                        spec,
                        cancellation,
                        capture_started,
                    )
                if stop_status is not None:
                    capture_seconds = time.monotonic() - capture_started
                    if report_path.exists():
                        created_files.add(report_path)
                    return result(
                        stop_status,
                        return_code=return_code,
                        report=report_path,
                    )
        except _CaptureLaunchError as exc:
            capture_seconds = time.monotonic() - capture_started
            return result(
                RunStatus.CAPTURE_LAUNCH_FAILED,
                detail=f"capture process could not be launched: {exc.cause_name}",
            )
        except _ArtifactSetupError as exc:
            capture_seconds = time.monotonic() - capture_started
            return result(
                RunStatus.ARTIFACT_SETUP_FAILED,
                detail=f"capture log setup failed: {exc.cause_name}",
            )
        finally:
            if process is not None and process.poll() is None:
                self._terminate_process_group(process)

        capture_seconds = time.monotonic() - capture_started
        report_exists = report_path.exists()
        if report_exists:
            created_files.add(report_path)
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
            created_files.add(sqlite_path)
        except (NsysAiError, OSError, subprocess.SubprocessError) as exc:
            if sqlite_candidate.exists():
                created_files.add(sqlite_candidate)
            export_seconds = time.monotonic() - export_started
            return result(
                RunStatus.EXPORT_FAILED,
                return_code=return_code,
                report=report_path,
                sqlite=sqlite_candidate,
                detail=f"profile export failed: {type(exc).__name__}",
            )
        export_seconds = time.monotonic() - export_started

        progress(RunStage.VALIDATING)
        validation_started = time.monotonic()
        try:
            reference, observed_gpus = read_local_profile_under_guard(
                sqlite_path, resolved_secrets=resolved_secrets
            )
        except (NsysAiError, OSError, *DB_ERRORS) as exc:
            validation_seconds = time.monotonic() - validation_started
            return result(
                RunStatus.INVALID_PROFILE,
                return_code=return_code,
                report=report_path,
                sqlite=sqlite_path,
                detail=f"profile validation failed: {type(exc).__name__}",
            )
        gpu_count_detail = check_expected_gpu_count(spec, observed_gpus)
        if gpu_count_detail is not None:
            validation_seconds = time.monotonic() - validation_started
            return result(
                RunStatus.INVALID_PROFILE,
                return_code=return_code,
                report=report_path,
                sqlite=sqlite_path,
                detail=gpu_count_detail,
            )
        validation_seconds = time.monotonic() - validation_started
        return result(
            RunStatus.SUCCEEDED,
            return_code=return_code,
            report=report_path,
            sqlite=sqlite_path,
            profile=reference,
        )

    def _validate_inputs(
        self,
        spec: RunSpec,
        progress_callback: ProgressCallback | None,
        cancellation: Cancellation | None,
    ) -> str:
        if not isinstance(spec, RunSpec):
            raise RunSpecError("spec must be a RunSpec")
        if progress_callback is not None and not callable(progress_callback):
            raise RunSpecError("progress_callback must be callable or null")
        if cancellation is not None and not (
            callable(cancellation) or callable(getattr(cancellation, "is_set", None))
        ):
            raise RunSpecError("cancellation must be callable, an event, or null")
        try:
            executable = os.fspath(self.nsys_executable)
        except TypeError as exc:
            raise RunSpecError("nsys_executable must be a path string") from exc
        if not isinstance(executable, str):
            raise RunSpecError("nsys_executable must be a path string")
        if not executable:
            raise RunSpecError("nsys_executable must not be empty")
        if "\x00" in executable:
            raise RunSpecError("nsys_executable must not contain NUL bytes")
        return executable

    @staticmethod
    def _create_private_file(path: Path) -> BinaryIO:
        descriptor: int | None = None
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            path.chmod(0o600)
            return os.fdopen(descriptor, "wb")
        except (OSError, ValueError) as exc:
            if descriptor is not None:
                os.close(descriptor)
            raise _ArtifactSetupError(exc) from exc

    @classmethod
    def _capture_process(
        cls,
        argv: list[str],
        cwd: str,
        environment: Mapping[str, str],
        stdout: BinaryIO,
        stderr: BinaryIO,
        spec: RunSpec,
        cancellation: Cancellation | None,
        capture_started: float,
    ) -> tuple[int | None, RunStatus | None, subprocess.Popen[bytes] | None]:
        """Launch and poll capture, resolving exit-versus-stop races explicitly."""

        deadline = (
            capture_started + spec.timeout_seconds
            if spec.timeout_seconds is not None
            else None
        )

        def cancellation_status() -> RunStatus | None:
            if cls._cancelled(cancellation):
                return RunStatus.CANCELLED
            return None

        # This check is intentionally adjacent to Popen. A progress callback
        # may have requested cancellation immediately before this helper.
        stop_status = cancellation_status()
        if stop_status is None and deadline is not None and time.monotonic() >= deadline:
            stop_status = RunStatus.TIMED_OUT
        if stop_status is not None:
            return None, stop_status, None

        try:
            process = subprocess.Popen(  # nosec B603
                argv,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            raise _CaptureLaunchError(exc) from exc
        try:
            while True:
                stop_status = cancellation_status()
                if stop_status is not None:
                    cls._terminate_process_group(process)
                    return process.poll(), stop_status, process

                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    cls._terminate_process_group(process)
                    return process.poll(), RunStatus.TIMED_OUT, process
                wait_seconds = (
                    _POLL_SECONDS
                    if remaining is None
                    else min(_POLL_SECONDS, remaining)
                )
                try:
                    return_code = process.wait(timeout=wait_seconds)
                except subprocess.TimeoutExpired:
                    continue

                # wait() returning proves the leader completed inside the
                # bounded interval, so completion wins over a later clock read.
                # Cancellation is still rechecked because it is independent of
                # the deadline and may have won while wait() was blocked.
                stop_status = cancellation_status()
                cls._terminate_process_group(process)
                return return_code, stop_status, process
        except BaseException:
            # Cancellation callables are control inputs, not observers. Their
            # failures (including KeyboardInterrupt) propagate, but never leave
            # the capture tree running.
            cls._terminate_process_group(process)
            raise

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
