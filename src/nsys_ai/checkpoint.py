"""Versioned manifests for reproducible real-profile checkpoints.

The checkpoint harness deliberately lives outside the product CLI. It records
the provenance and commands needed to prove that a real workload went through
the same analysis contract, without making a GPU capture a per-commit CI
dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_MANIFEST_SCHEMA = "checkpoint-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_KINDS = {"git_commit", "git_tag", "dataset_revision"}
_CAPTURE_FORMATS = {"nsys-rep", "sqlite", "parquetdir"}
_STEP_NAMES = {"doctor", "diagnose", "ask", "diff", "review"}
_PLACEHOLDER_VALUES = {
    "",
    "head",
    "latest",
    "main",
    "master",
    "none",
    "n/a",
    "todo",
    "tbd",
    "unknown",
}
_TEMPLATE_RE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
_ALLOWED_TEMPLATES = {"profile", "repo", "session"}


class CheckpointManifestError(ValueError):
    """The manifest cannot be used as a reproducible checkpoint."""


def _fail(errors: list[str], path: str, message: str) -> None:
    errors.append(f"{path}: {message}")


def _mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        _fail(errors, path, "must be an object")
        return None
    return value


def _required_string(
    value: Any,
    path: str,
    errors: list[str],
    *,
    reject_placeholders: bool = False,
) -> str | None:
    if not isinstance(value, str) or not value.strip():
        _fail(errors, path, "must be a non-empty string")
        return None
    normalized = value.strip()
    if reject_placeholders and normalized.lower() in _PLACEHOLDER_VALUES:
        _fail(errors, path, "must be pinned; placeholder values are not accepted")
        return None
    return normalized


def _relative_path(value: Any, path: str, errors: list[str]) -> str | None:
    text = _required_string(value, path, errors)
    if text is None:
        return None
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        _fail(errors, path, "must be a relative path without '..'")
    return text


def _argv(value: Any, path: str, errors: list[str]) -> list[str] | None:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        _fail(errors, path, "must be a non-empty argv array of strings")
        return None
    for index, item in enumerate(value):
        for name in _TEMPLATE_RE.findall(item):
            if name not in _ALLOWED_TEMPLATES:
                _fail(errors, f"{path}[{index}]", f"unknown template {{{name}}}")
    return list(value)


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    profile_root: str | os.PathLike[str] | None = None,
    profile_override: str | os.PathLike[str] | None = None,
    require_profile: bool = False,
) -> dict[str, Any]:
    """Validate and return a checkpoint manifest as a plain dictionary.

    Structural validation is always performed. ``require_profile`` additionally
    checks that the referenced/overridden profile exists, is non-empty, and has
    the recorded SHA-256. A manifest can therefore be checked in CI without a
    capture, while a real checkpoint run fails closed when the artifact differs.
    """

    if not isinstance(manifest, Mapping):
        raise CheckpointManifestError("manifest must be a JSON object")

    errors: list[str] = []
    schema = _required_string(manifest.get("schema_version"), "schema_version", errors)
    if schema is not None and schema != CHECKPOINT_MANIFEST_SCHEMA:
        _fail(errors, "schema_version", f"must be {CHECKPOINT_MANIFEST_SCHEMA!r}")
    _required_string(manifest.get("checkpoint"), "checkpoint", errors, reject_placeholders=True)

    project = _mapping(manifest.get("project"), "project", errors)
    if project is not None:
        _required_string(project.get("name"), "project.name", errors, reject_placeholders=True)
        _required_string(project.get("repository"), "project.repository", errors)
        revision_kind = _required_string(
            project.get("revision_kind"), "project.revision_kind", errors
        )
        if revision_kind is not None and revision_kind not in _REVISION_KINDS:
            _fail(errors, "project.revision_kind", "must be git_commit, git_tag, or dataset_revision")
        _required_string(
            project.get("revision"), "project.revision", errors, reject_placeholders=True
        )

    workload = _mapping(manifest.get("workload"), "workload", errors)
    if workload is not None:
        _required_string(workload.get("name"), "workload.name", errors, reject_placeholders=True)
        _required_string(workload.get("artifact"), "workload.artifact", errors)
        parameters = workload.get("parameters")
        if not isinstance(parameters, Mapping):
            _fail(errors, "workload.parameters", "must be an object")
        _argv(workload.get("capture_command"), "workload.capture_command", errors)
        signals = workload.get("expected_signals")
        if not isinstance(signals, list) or not signals:
            _fail(errors, "workload.expected_signals", "must contain at least one signal")
        else:
            signal_ids: set[str] = set()
            for index, signal in enumerate(signals):
                item = _mapping(signal, f"workload.expected_signals[{index}]", errors)
                if item is None:
                    continue
                signal_id = _required_string(
                    item.get("id"), f"workload.expected_signals[{index}].id", errors
                )
                if signal_id is not None and signal_id in signal_ids:
                    _fail(errors, f"workload.expected_signals[{index}].id", "must be unique")
                if signal_id is not None:
                    signal_ids.add(signal_id)
                _required_string(
                    item.get("description"),
                    f"workload.expected_signals[{index}].description",
                    errors,
                )
                _required_string(
                    item.get("verification"),
                    f"workload.expected_signals[{index}].verification",
                    errors,
                )

    environment = _mapping(manifest.get("environment"), "environment", errors)
    if environment is not None:
        for key in ("python", "cuda", "driver", "gpu", "nsys"):
            _required_string(
                environment.get(key), f"environment.{key}", errors, reject_placeholders=True
            )

    capture = _mapping(manifest.get("capture"), "capture", errors)
    profile_path: str | None = None
    if capture is not None:
        profile_path = _relative_path(capture.get("profile_path"), "capture.profile_path", errors)
        capture_format = _required_string(capture.get("format"), "capture.format", errors)
        if capture_format is not None and capture_format not in _CAPTURE_FORMATS:
            _fail(errors, "capture.format", "must be nsys-rep, sqlite, or parquetdir")
        digest = _required_string(capture.get("sha256"), "capture.sha256", errors)
        if digest is not None and not _SHA256_RE.fullmatch(digest):
            _fail(errors, "capture.sha256", "must be 64 lowercase hexadecimal characters")
        output_paths = capture.get("output_paths")
        if not isinstance(output_paths, list) or not output_paths:
            _fail(errors, "capture.output_paths", "must contain at least one path")
        else:
            for index, output_path in enumerate(output_paths):
                _relative_path(output_path, f"capture.output_paths[{index}]", errors)
        _required_string(capture.get("captured_at"), "capture.captured_at", errors)

    analysis = _mapping(manifest.get("analysis"), "analysis", errors)
    if analysis is not None:
        _relative_path(analysis.get("session_dir"), "analysis.session_dir", errors)
        steps = analysis.get("steps")
        if not isinstance(steps, list) or not steps:
            _fail(errors, "analysis.steps", "must contain at least one step")
        else:
            names: set[str] = set()
            for index, step in enumerate(steps):
                item = _mapping(step, f"analysis.steps[{index}]", errors)
                if item is None:
                    continue
                name = _required_string(item.get("name"), f"analysis.steps[{index}].name", errors)
                if name is not None:
                    if name not in _STEP_NAMES:
                        _fail(errors, f"analysis.steps[{index}].name", "is not a supported analysis step")
                    if name in names:
                        _fail(errors, f"analysis.steps[{index}].name", "must be unique")
                    names.add(name)
                _argv(item.get("command"), f"analysis.steps[{index}].command", errors)
                expected = item.get("expected_exit_codes", [0])
                if (
                    not isinstance(expected, list)
                    or not expected
                    or not all(isinstance(code, int) for code in expected)
                ):
                    _fail(
                        errors,
                        f"analysis.steps[{index}].expected_exit_codes",
                        "must be a non-empty array of integers",
                    )
            missing_steps = _STEP_NAMES - names
            if missing_steps:
                _fail(errors, "analysis.steps", "missing required steps: " + ", ".join(sorted(missing_steps)))

    if errors:
        raise CheckpointManifestError("invalid checkpoint manifest:\n- " + "\n- ".join(errors))

    result = json.loads(json.dumps(manifest))
    if require_profile:
        if profile_override is not None:
            profile = Path(profile_override)
        else:
            if profile_path is None:
                raise CheckpointManifestError("capture.profile_path is required to verify a profile")
            root = Path(profile_root or ".").resolve()
            profile = root / profile_path
        _verify_profile(profile, str(capture["sha256"]))
    return result


def _verify_profile(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise CheckpointManifestError(f"capture profile does not exist: {path}")
    if path.stat().st_size <= 0:
        raise CheckpointManifestError(f"capture profile is empty: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise CheckpointManifestError(
            f"capture checksum mismatch for {path}: expected {expected_sha256}, got {actual}"
        )


def load_manifest(path: str | os.PathLike[str], **kwargs: Any) -> dict[str, Any]:
    """Load and validate one JSON manifest."""
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointManifestError(f"could not read manifest {manifest_path}: {exc}") from exc
    return validate_manifest(payload, **kwargs)


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Return stable JSON bytes for review, hashing, and change detection."""
    validate_manifest(manifest)
    return (json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def resolve_profile_path(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | os.PathLike[str],
    profile_override: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve the manifest profile or an explicit real-profile override."""
    if profile_override is not None:
        return Path(profile_override).expanduser().resolve()
    return (Path(repo_root).resolve() / str(manifest["capture"]["profile_path"])).resolve()


def expand_command(
    command: Sequence[str],
    *,
    profile: str | os.PathLike[str],
    repo: str | os.PathLike[str],
    session: str | os.PathLike[str],
) -> list[str]:
    """Expand the three safe path placeholders in one argv array."""
    values = {"profile": os.fspath(profile), "repo": os.fspath(repo), "session": os.fspath(session)}
    expanded: list[str] = []
    for argument in command:
        names = _TEMPLATE_RE.findall(argument)
        unknown = [name for name in names if name not in _ALLOWED_TEMPLATES]
        if unknown:
            raise CheckpointManifestError(f"unsupported command template: {{{unknown[0]}}}")
        expanded.append(argument.format_map(values))
    return expanded


@dataclass(frozen=True)
class CheckpointStepResult:
    """Machine-readable result of one analysis step."""

    name: str
    command: tuple[str, ...]
    returncode: int
    expected_exit_codes: tuple[int, ...]
    elapsed_seconds: float

    @property
    def passed(self) -> bool:
        return self.returncode in self.expected_exit_codes

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": list(self.command),
            "returncode": self.returncode,
            "expected_exit_codes": list(self.expected_exit_codes),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "passed": self.passed,
        }


def run_steps(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | os.PathLike[str],
    profile: str | os.PathLike[str],
    session: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    timeout: float = 300.0,
) -> list[CheckpointStepResult]:
    """Run manifest analysis steps without a shell and write per-step logs."""
    validate_manifest(manifest, require_profile=True, profile_override=profile)
    root = Path(repo_root).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results: list[CheckpointStepResult] = []
    for index, raw_step in enumerate(manifest["analysis"]["steps"]):
        step = dict(raw_step)
        command = expand_command(
            step["command"], profile=profile, repo=root, session=session
        )
        expected = tuple(step.get("expected_exit_codes", [0]))
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            returncode = 124
            stdout = _decode_output(exc.stdout)
            stderr = _decode_output(exc.stderr) + f"\ncheckpoint step timed out after {timeout}s\n"
        elapsed = time.monotonic() - started
        name = str(step["name"])
        safe_name = re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-") or f"step-{index}"
        (destination / f"{index:02d}-{safe_name}.stdout").write_text(stdout, encoding="utf-8")
        (destination / f"{index:02d}-{safe_name}.stderr").write_text(stderr, encoding="utf-8")
        results.append(
            CheckpointStepResult(name, tuple(command), returncode, expected, elapsed)
        )
    return results


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
