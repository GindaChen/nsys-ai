"""Strict CLI orchestration for deterministic Proposal artifacts."""

from __future__ import annotations

import errno
import json
import os
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .annotation import EvidenceReport, Finding, validate_evidence_report_payload
from .artifact_io import atomic_write_bytes_at
from .exceptions import NsysAiError
from .proposal import Proposal, generate_proposal
from .runspec import RunSpec, RunSpecError, validate_persisted_secret_strings
from .session_store import SessionState


class ProposeCommandError(NsysAiError):
    """The propose command received an invalid input or output target."""

    error_code = "PROPOSE_COMMAND_INVALID"


_SESSION_MANIFEST_MAX_BYTES = 256 * 1024
_SESSION_TOP_LEVEL_SIGNATURE = frozenset(
    {"schema_version", "session_id", "phase", "profiles", "artifacts"}
)
_SESSION_PROFILE_SIGNATURE = frozenset({"before", "after"})
_SESSION_ARTIFACT_SIGNATURE = frozenset(
    {"runspec", "findings", "proposal", "diff"}
)
_SESSION_CONTROL_KEYS = frozenset({"session_id", "phase"})
_SESSION_STRUCTURE_KEYS = frozenset({"profiles", "artifacts"})


@dataclass(frozen=True)
class _InputArtifact:
    payload: Any
    device: int
    inode: int


def _read_json(path: Path, label: str) -> _InputArtifact:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            encoded = stream.read()
    except OSError as exc:
        raise ProposeCommandError(
            f"could not read {label} artifact ({type(exc).__name__})"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        payload = json.loads(encoded)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProposeCommandError(f"invalid {label} JSON") from exc
    return _InputArtifact(payload, metadata.st_dev, metadata.st_ino)


def _redact_message(message: str, resolved_secrets: Mapping[str, str]) -> str:
    values = sorted(
        {value for value in resolved_secrets.values() if value},
        key=len,
        reverse=True,
    )
    for value in values:
        if value:
            message = message.replace(value, "<redacted>")
    return message


def _select_finding(report: EvidenceReport, finding_id: str) -> Finding:
    if not finding_id or any(character.isspace() for character in finding_id):
        raise ProposeCommandError("--finding-id must be a non-empty stable ID without whitespace")
    matches = [finding for finding in report.findings if finding.id == finding_id]
    if not matches:
        raise ProposeCommandError("requested finding ID was not found in the evidence report")
    if len(matches) > 1:
        raise ProposeCommandError("requested finding ID is duplicated in the evidence report")
    return matches[0]


def _resolve_declared_secrets(
    spec: RunSpec, environment: Mapping[str, str]
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for name in spec.environment.secrets:
        if name not in environment:
            raise RunSpecError(f"declared secret {name} is not set in the current environment")
        value = environment[name]
        if not isinstance(value, str):
            raise RunSpecError(f"resolved secret {name} must be a string")
        resolved[name] = value
    return resolved


def _read_bounded(descriptor: int, limit: int) -> bytes:
    chunks = []
    remaining = limit
    while remaining:
        chunk = os.read(descriptor, min(64 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _has_raw_session_signature(encoded: bytes) -> bool:
    markers = {
        name
        for name in _SESSION_TOP_LEVEL_SIGNATURE
        if f'"{name}"'.encode("ascii") in encoded
    }
    if markers == _SESSION_TOP_LEVEL_SIGNATURE:
        return True
    nested_artifacts = all(
        f'"{name}"'.encode("ascii") in encoded
        for name in _SESSION_ARTIFACT_SIGNATURE
    )
    nested_profiles = all(
        f'"{name}"'.encode("ascii") in encoded
        for name in _SESSION_PROFILE_SIGNATURE
    )
    if "artifacts" in markers and nested_artifacts:
        return True
    has_control = bool(_SESSION_CONTROL_KEYS & markers)
    has_structure = bool(_SESSION_STRUCTURE_KEYS & markers)
    return has_control and (
        ("schema_version" in markers and has_structure)
        or nested_artifacts
        or nested_profiles
    )


def _has_session_signature(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    keys = set(payload)
    if _SESSION_TOP_LEVEL_SIGNATURE <= keys:
        return True
    profiles = payload.get("profiles")
    artifacts = payload.get("artifacts")
    nested_signature = (
        isinstance(profiles, Mapping)
        and _SESSION_PROFILE_SIGNATURE <= set(profiles)
        and isinstance(artifacts, Mapping)
        and _SESSION_ARTIFACT_SIGNATURE <= set(artifacts)
    )
    has_control = bool(_SESSION_CONTROL_KEYS & keys)
    has_structure = bool(_SESSION_STRUCTURE_KEYS & keys)
    return has_control and (
        ("schema_version" in keys and has_structure) or nested_signature
    )


def _has_session_logs_directory(directory_fd: int) -> bool:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open("logs", flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return False
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.ENXIO}:
            return False
        raise ProposeCommandError(
            f"could not inspect session logs directory ({type(exc).__name__})"
        ) from exc
    try:
        return stat.S_ISDIR(os.fstat(descriptor).st_mode)
    finally:
        os.close(descriptor)


def _session_marker(directory_fd: int) -> bool:
    has_logs_directory = _has_session_logs_directory(directory_fd)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open("session.json", flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ProposeCommandError(
            f"could not inspect session manifest ({type(exc).__name__})"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ProposeCommandError("session manifest is not a regular file")
        if metadata.st_size > _SESSION_MANIFEST_MAX_BYTES:
            raise ProposeCommandError(
                "session manifest exceeds the safe inspection size limit"
            )
        encoded = _read_bounded(descriptor, _SESSION_MANIFEST_MAX_BYTES + 1)
        if len(encoded) > _SESSION_MANIFEST_MAX_BYTES:
            raise ProposeCommandError(
                "session manifest exceeds the safe inspection size limit"
            )
        try:
            payload = json.loads(encoded)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if not has_logs_directory and not _has_raw_session_signature(encoded):
                return False
            raise ProposeCommandError("session manifest is invalid") from exc
        if not _has_session_signature(payload):
            if has_logs_directory:
                raise ProposeCommandError("session manifest is invalid")
            return False
        try:
            SessionState.from_dict(payload)
        except (NsysAiError, KeyError, TypeError, ValueError) as exc:
            raise ProposeCommandError("session manifest is invalid") from exc
        return True
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _reject_session_ancestor(directory_fd: int) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    current = os.dup(directory_fd)
    try:
        while True:
            if _session_marker(current):
                raise ProposeCommandError(
                    "session artifacts must be published through SessionWriter, "
                    "not nsys-ai propose"
                )
            parent = os.open("..", flags, dir_fd=current)
            current_stat = os.fstat(current)
            parent_stat = os.fstat(parent)
            if (current_stat.st_dev, current_stat.st_ino) == (
                parent_stat.st_dev,
                parent_stat.st_ino,
            ):
                os.close(parent)
                return
            os.close(current)
            current = parent
    finally:
        os.close(current)


def _reject_input_aliases(
    directory_fd: int,
    name: str,
    inputs: tuple[_InputArtifact, ...],
) -> None:
    try:
        output_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=True)
    except FileNotFoundError:
        return
    output_identity = (output_stat.st_dev, output_stat.st_ino)
    if any(output_identity == (item.device, item.inode) for item in inputs):
        raise ProposeCommandError("output artifact must not alias an input artifact")


def _open_output_directory(path: Path) -> int:
    """Open/create an output directory while checking each bound directory."""
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    current = os.open(os.sep, flags)
    try:
        _reject_session_ancestor(current)
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=current)
            except FileNotFoundError:
                try:
                    os.mkdir(component, dir_fd=current)
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = child
            _reject_session_ancestor(current)
        return current
    except BaseException:
        os.close(current)
        raise


def _write_proposal(
    path: Path,
    proposal: Proposal,
    inputs: tuple[_InputArtifact, ...],
) -> Path:
    destination = path.expanduser()
    if not destination.name:
        raise ProposeCommandError("proposal output must name a file")
    try:
        directory_fd = _open_output_directory(destination.parent)
    except ProposeCommandError:
        raise
    except OSError as exc:
        raise ProposeCommandError(
            f"could not write proposal artifact ({type(exc).__name__})"
        ) from exc
    try:
        _reject_session_ancestor(directory_fd)
        _reject_input_aliases(directory_fd, destination.name, inputs)
        atomic_write_bytes_at(
            directory_fd,
            destination.name,
            proposal.canonical_json_bytes(),
        )
    except ProposeCommandError:
        raise
    except (OSError, ValueError) as exc:
        raise ProposeCommandError(
            f"could not write proposal artifact ({type(exc).__name__})"
        ) from exc
    finally:
        os.close(directory_fd)
    return destination


def run_propose_command(
    args: Any,
    *,
    stdout: TextIO = sys.stdout,
    environment: Mapping[str, str] = os.environ,
) -> int:
    """Generate one Proposal from a strictly validated evidence artifact."""
    runspec_input = _read_json(Path(args.runspec), "RunSpec") if args.runspec else None
    if runspec_input is not None:
        try:
            runspec = RunSpec.from_dict(runspec_input.payload)
        except (AttributeError, RunSpecError, TypeError, ValueError) as exc:
            raise ProposeCommandError("invalid RunSpec artifact") from exc
    else:
        runspec = None
    resolved_secrets = (
        _resolve_declared_secrets(runspec, environment) if runspec is not None else None
    )
    findings_input = _read_json(Path(args.findings), "evidence")
    try:
        report = validate_evidence_report_payload(findings_input.payload)
    except (TypeError, ValueError) as exc:
        detail = _redact_message(str(exc), resolved_secrets or {})
        raise ProposeCommandError(f"invalid evidence report: {detail}") from exc
    finding = _select_finding(report, args.finding_id)
    proposal = generate_proposal(
        finding,
        runspec,
        resolved_secrets=resolved_secrets,
    )
    output = Path(args.output)
    if resolved_secrets is not None:
        validate_persisted_secret_strings([str(output)], resolved_secrets)
    inputs = (findings_input,) + ((runspec_input,) if runspec_input is not None else ())
    written = _write_proposal(output, proposal, inputs)
    print(f"Proposal written to {written}", file=stdout)
    if proposal.abstained:
        print(f"Abstained: {proposal.abstention_reason}", file=stdout)
    return 0
