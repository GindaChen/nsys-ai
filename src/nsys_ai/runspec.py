"""Versioned specification for reproducible profile runs.

``RunSpec`` records the workload and capture configuration without resolving
declared environment secrets or invoking a shell.  Raw argv and public
environment overrides are persisted verbatim and must not contain secrets.
Execution is deliberately owned by the local runner (issue #269); this module
defines its input contract, secret-boundary preflight, and the pure
``nsys profile`` argv construction it will consume.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from .exceptions import NsysAiError

RUNSPEC_SCHEMA_VERSION = "0.2"
RUNSPEC_SCHEMA_VERSIONS = frozenset({"0.1", RUNSPEC_SCHEMA_VERSION})
RUNSPEC_COMPATIBILITY_VERSION = "1"
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TRACE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_ENV_POLICIES = frozenset({"inherit", "clean"})
_CAPTURE_RANGES = frozenset({"none", "cudaProfilerApi", "nvtx", "hotkey"})
_SAMPLE_MODES = frozenset({"none", "process-tree", "system-wide"})
_CPUCTXSW_MODES = frozenset({"none", "process-tree", "system-wide"})
# nsys accepts one autograd mode and/or one functions-trace mode, comma-joined.
_PYTORCH_AUTOGRAD_MODES = frozenset({"autograd-nvtx", "autograd-shapes-nvtx"})
_PYTORCH_FUNCTIONS_MODES = frozenset({"functions-trace", "functions-trace-shapes"})
_PYTORCH_MODES = _PYTORCH_AUTOGRAD_MODES | _PYTORCH_FUNCTIONS_MODES


class RunSpecError(NsysAiError):
    """A RunSpec payload or construction argument is invalid."""

    error_code = "RUNSPEC_INVALID"


class UnsupportedRunSpecVersionError(RunSpecError):
    """A RunSpec artifact uses a schema this installation cannot read."""

    error_code = "RUNSPEC_VERSION_UNSUPPORTED"


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunSpecError(f"{label} must be an object")
    return value


def _reject_unknown_keys(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise RunSpecError(f"{label} has unknown field(s): {', '.join(unknown)}")


def normalize_pytorch_modes(
    value: Any, *, label: str = "trace_options.pytorch"
) -> tuple[str, ...]:
    """Validate the requested nsys PyTorch annotation modes and canonicalize order.

    Returns an empty tuple for "not requested", which is also what an explicit
    ``none`` means, so the two spell the same thing on disk and in the argv.

    Public because the CLI must reach the same verdict *before* it decides whether
    to build a RunSpec at all: ``--dry-run`` exists to show a plan that would run,
    so it has to reject the same values the real capture would. ``label`` names the
    thing the caller's user actually typed, so the CLI can say ``--pytorch`` while a
    persisted spec says ``trace_options.pytorch``.
    """
    if isinstance(value, str):
        raise RunSpecError(f"{label} must be an array, not a comma-joined string")
    if not isinstance(value, (list, tuple)):
        raise RunSpecError(f"{label} must be an array")
    modes = tuple(value)
    for mode in modes:
        if not isinstance(mode, str) or (mode != "none" and mode not in _PYTORCH_MODES):
            raise RunSpecError(f"unsupported {label} mode: {mode!r}")
    if len(set(modes)) != len(modes):
        raise RunSpecError(f"{label} must not contain duplicates")
    if "none" in modes:
        if len(modes) > 1:
            raise RunSpecError(f"{label} cannot combine 'none' with a mode")
        return ()
    selected = set(modes)
    if len(selected & _PYTORCH_AUTOGRAD_MODES) > 1:
        raise RunSpecError(
            f"{label} accepts at most one autograd mode: "
            "'autograd-nvtx' or 'autograd-shapes-nvtx'"
        )
    if len(selected & _PYTORCH_FUNCTIONS_MODES) > 1:
        raise RunSpecError(
            f"{label} accepts at most one functions-trace mode: "
            "'functions-trace' or 'functions-trace-shapes'"
        )
    return tuple(sorted(modes))


def _validate_string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise RunSpecError(f"{label} must be a string")
    if not allow_empty and not value:
        raise RunSpecError(f"{label} must not be empty")
    if "\x00" in value:
        raise RunSpecError(f"{label} must not contain NUL bytes")
    return value


def _validate_optional_count(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RunSpecError(f"{label} must be a positive integer or null")
    return value


@dataclass(frozen=True)
class EnvironmentSpec:
    """Environment inputs without persisted secret values.

    ``inherit`` starts from the runner's environment; ``clean`` starts empty.
    ``public`` values are persisted verbatim and participate in comparability;
    callers must treat them as public. Entries in ``secrets`` are names only:
    the runner resolves their values at execution time and validates the
    boundary before persisting the RunSpec or starting a process.
    """

    policy: str = "inherit"
    public: Mapping[str, str] = field(default_factory=dict)
    secrets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.policy, str):
            raise RunSpecError("environment.policy must be a string")
        if self.policy not in _ENV_POLICIES:
            choices = ", ".join(sorted(_ENV_POLICIES))
            raise RunSpecError(f"environment.policy must be one of: {choices}")
        if not isinstance(self.public, Mapping):
            raise RunSpecError("environment.public must be an object")

        public: dict[str, str] = {}
        for name, value in self.public.items():
            self._validate_name(name)
            public[name] = _validate_string(
                value, f"environment.public[{name!r}]", allow_empty=True
            )

        if not isinstance(self.secrets, (list, tuple)):
            raise RunSpecError("environment.secrets must be an array of names")
        secrets = tuple(self.secrets)
        for name in secrets:
            self._validate_name(name)
        if len(set(secrets)) != len(secrets):
            raise RunSpecError("environment.secrets must not contain duplicates")
        overlap = sorted(set(public) & set(secrets))
        if overlap:
            raise RunSpecError(
                "environment variables cannot be both public and secret: "
                + ", ".join(overlap)
            )

        object.__setattr__(
            self, "public", MappingProxyType(dict(sorted(public.items())))
        )
        object.__setattr__(self, "secrets", tuple(sorted(secrets)))

    @staticmethod
    def _validate_name(name: Any) -> None:
        if not isinstance(name, str) or not _ENV_NAME_RE.fullmatch(name):
            raise RunSpecError(f"invalid environment variable name: {name!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "public": dict(self.public),
            "secrets": list(self.secrets),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EnvironmentSpec:
        data = _require_mapping(payload, "environment")
        _reject_unknown_keys(data, {"policy", "public", "secrets"}, "environment")
        return cls(
            policy=data.get("policy", "inherit"),
            public=data.get("public", {}),
            secrets=data.get("secrets", ()),
        )

    def redacted(self) -> dict[str, str]:
        """Return display-safe explicit variables, marking secrets by name."""
        result = dict(self.public)
        result.update({name: "<redacted>" for name in self.secrets})
        return dict(sorted(result.items()))


@dataclass(frozen=True)
class NsysTraceOptions:
    """Supported v0 capture options for ``nsys profile``."""

    trace: tuple[str, ...] = ("cuda", "nvtx", "nccl")
    sample: str = "none"
    cpuctxsw: str = "none"
    capture_range: str = "none"
    cuda_memory_usage: bool = False
    pytorch: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.trace, (list, tuple)):
            raise RunSpecError("trace_options.trace must be an array")
        trace = tuple(self.trace)
        if not trace:
            raise RunSpecError("trace_options.trace must not be empty")
        for api in trace:
            if not isinstance(api, str) or not _TRACE_TOKEN_RE.fullmatch(api):
                raise RunSpecError(f"invalid nsys trace API: {api!r}")
        if len(set(trace)) != len(trace):
            raise RunSpecError("trace_options.trace must not contain duplicates")
        if "cuda" not in trace:
            raise RunSpecError("trace_options.trace must include cuda")
        if not isinstance(self.sample, str) or self.sample not in _SAMPLE_MODES:
            raise RunSpecError("unsupported trace_options.sample mode")
        if not isinstance(self.cpuctxsw, str) or self.cpuctxsw not in _CPUCTXSW_MODES:
            raise RunSpecError("unsupported trace_options.cpuctxsw mode")
        if (
            not isinstance(self.capture_range, str)
            or self.capture_range not in _CAPTURE_RANGES
        ):
            raise RunSpecError("unsupported trace_options.capture_range")
        if not isinstance(self.cuda_memory_usage, bool):
            raise RunSpecError("trace_options.cuda_memory_usage must be a boolean")
        object.__setattr__(self, "trace", tuple(sorted(trace)))
        object.__setattr__(self, "pytorch", normalize_pytorch_modes(self.pytorch))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "trace": list(self.trace),
            "sample": self.sample,
            "cpuctxsw": self.cpuctxsw,
            "capture_range": self.capture_range,
            "cuda_memory_usage": self.cuda_memory_usage,
            # Reports must not capture the runner's resolved environment.
            "discard_environment": True,
        }
        # Emitted only when requested. This mapping is hashed into
        # ``RunSpec.compatibility_key``, so an unconditional key would change the
        # identity of every spec written before the option existed.
        if self.pytorch:
            payload["pytorch"] = list(self.pytorch)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> NsysTraceOptions:
        data = _require_mapping(payload, "trace_options")
        allowed = {
            "trace",
            "sample",
            "cpuctxsw",
            "capture_range",
            "cuda_memory_usage",
            "pytorch",
            "discard_environment",
        }
        _reject_unknown_keys(data, allowed, "trace_options")
        if data.get("discard_environment", True) is not True:
            raise RunSpecError("trace_options.discard_environment must be true")
        return cls(
            trace=data.get("trace", ("cuda", "nvtx", "nccl")),
            sample=data.get("sample", "none"),
            cpuctxsw=data.get("cpuctxsw", "none"),
            capture_range=data.get("capture_range", "none"),
            cuda_memory_usage=data.get("cuda_memory_usage", False),
            pytorch=data.get("pytorch", ()),
        )


@dataclass(frozen=True)
class RunSpec:
    """Serializable input contract for one local profiling run."""

    argv: tuple[str, ...]
    cwd: str = "."
    repository: str | None = None
    commit: str | None = None
    dirty: bool = False
    worktree_diff_sha256: str | None = None
    environment: EnvironmentSpec = field(default_factory=EnvironmentSpec)
    warmup_steps: int = 0
    profile_steps: int = 1
    seed: int | None = None
    expected_gpu_count: int | None = None
    expected_rank_count: int | None = None
    trace_options: NsysTraceOptions = field(default_factory=NsysTraceOptions)
    timeout_seconds: int | None = None
    runner: str = "local"

    def __post_init__(self) -> None:
        if not isinstance(self.argv, (list, tuple)):
            raise RunSpecError("argv must be an array, never a shell string")
        argv = tuple(self.argv)
        if not argv:
            raise RunSpecError("argv must not be empty")
        for index, arg in enumerate(argv):
            _validate_string(arg, f"argv[{index}]", allow_empty=index > 0)
        object.__setattr__(self, "argv", argv)

        cwd = _validate_string(self.cwd, "cwd")
        posix_cwd = PurePosixPath(cwd)
        if self.repository is not None:
            _validate_string(self.repository, "repository")
            if posix_cwd.is_absolute() or ".." in posix_cwd.parts or "\\" in cwd:
                raise RunSpecError(
                    "cwd must be a repository-relative POSIX path when repository is set"
                )
            object.__setattr__(self, "cwd", str(posix_cwd))
        if self.commit is not None:
            _validate_string(self.commit, "commit")
        if not isinstance(self.dirty, bool):
            raise RunSpecError("dirty must be a boolean")
        if self.dirty and (self.repository is None or self.commit is None):
            raise RunSpecError("dirty requires repository and commit provenance")
        if self.worktree_diff_sha256 is not None:
            if not isinstance(self.worktree_diff_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", self.worktree_diff_sha256
            ):
                raise RunSpecError(
                    "worktree_diff_sha256 must be a SHA-256 hex string or null"
                )
        if self.dirty != (self.worktree_diff_sha256 is not None):
            raise RunSpecError(
                "dirty and worktree_diff_sha256 must either describe a dirty worktree or a clean one"
            )
        if not isinstance(self.environment, EnvironmentSpec):
            raise RunSpecError("environment must be an EnvironmentSpec")
        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int):
            raise RunSpecError("warmup_steps must be a non-negative integer")
        if self.warmup_steps < 0:
            raise RunSpecError("warmup_steps must be a non-negative integer")
        if isinstance(self.profile_steps, bool) or not isinstance(self.profile_steps, int):
            raise RunSpecError("profile_steps must be a positive integer")
        if self.profile_steps <= 0:
            raise RunSpecError("profile_steps must be a positive integer")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0
        ):
            raise RunSpecError("seed must be a non-negative integer or null")
        _validate_optional_count(self.expected_gpu_count, "expected_gpu_count")
        _validate_optional_count(self.expected_rank_count, "expected_rank_count")
        if not isinstance(self.trace_options, NsysTraceOptions):
            raise RunSpecError("trace_options must be an NsysTraceOptions")
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or self.timeout_seconds <= 0
        ):
            raise RunSpecError("timeout_seconds must be a positive integer or null")
        if self.runner != "local":
            raise RunSpecError("runner must be 'local' in RunSpec v0.1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUNSPEC_SCHEMA_VERSION,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "repository": self.repository,
            "commit": self.commit,
            "dirty": self.dirty,
            "worktree_diff_sha256": self.worktree_diff_sha256,
            "environment": self.environment.to_dict(),
            "warmup_steps": self.warmup_steps,
            "profile_steps": self.profile_steps,
            "seed": self.seed,
            "expected_gpu_count": self.expected_gpu_count,
            "expected_rank_count": self.expected_rank_count,
            "trace_options": self.trace_options.to_dict(),
            "timeout_seconds": self.timeout_seconds,
            "runner": self.runner,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunSpec:
        data = _require_mapping(payload, "RunSpec")
        version = data.get("schema_version")
        if version not in RUNSPEC_SCHEMA_VERSIONS:
            raise UnsupportedRunSpecVersionError(
                f"unsupported RunSpec schema_version {version!r}; "
                f"expected one of {', '.join(sorted(RUNSPEC_SCHEMA_VERSIONS))!r}"
            )
        allowed = {
            "schema_version",
            "argv",
            "cwd",
            "repository",
            "commit",
            "dirty",
            "worktree_diff_sha256",
            "environment",
            "warmup_steps",
            "profile_steps",
            "seed",
            "expected_gpu_count",
            "expected_rank_count",
            "trace_options",
            "timeout_seconds",
            "runner",
        }
        _reject_unknown_keys(data, allowed, "RunSpec")
        if version == "0.1" and (
            "dirty" in data or "worktree_diff_sha256" in data
        ):
            raise UnsupportedRunSpecVersionError(
                "RunSpec dirty worktree identity requires schema_version '0.2'"
            )
        if "argv" not in data:
            raise RunSpecError("RunSpec.argv is required")
        return cls(
            argv=data["argv"],
            cwd=data.get("cwd", "."),
            repository=data.get("repository"),
            commit=data.get("commit"),
            dirty=data.get("dirty", False),
            worktree_diff_sha256=data.get("worktree_diff_sha256"),
            environment=EnvironmentSpec.from_dict(data.get("environment", {})),
            warmup_steps=data.get("warmup_steps", 0),
            profile_steps=data.get("profile_steps", 1),
            seed=data.get("seed"),
            expected_gpu_count=data.get("expected_gpu_count"),
            expected_rank_count=data.get("expected_rank_count"),
            trace_options=NsysTraceOptions.from_dict(data.get("trace_options", {})),
            timeout_seconds=data.get("timeout_seconds"),
            runner=data.get("runner", "local"),
        )

    def canonical_json_bytes(self) -> bytes:
        """Return deterministic UTF-8 JSON suitable for ``runspec.json``."""
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, payload: bytes | str) -> RunSpec:
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RunSpecError(f"invalid RunSpec JSON: {exc}") from exc
        return cls.from_dict(data)

    def compatibility_payload(self) -> dict[str, Any]:
        """Return only inputs that must match for a normal-confidence diff."""
        return {
            "compatibility_version": RUNSPEC_COMPATIBILITY_VERSION,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "environment": {
                "policy": self.environment.policy,
                "public": dict(self.environment.public),
                "secret_names": list(self.environment.secrets),
            },
            "warmup_steps": self.warmup_steps,
            "profile_steps": self.profile_steps,
            "seed": self.seed,
            "expected_gpu_count": self.expected_gpu_count,
            "expected_rank_count": self.expected_rank_count,
            "trace_options": self.trace_options.to_dict(),
        }

    def compatibility_limitations(self) -> tuple[str, ...]:
        """Name inputs whose equality the persisted artifact cannot prove."""
        limitations: list[str] = []
        if self.dirty:
            limitations.append("uncommitted_worktree")
        if self.environment.policy == "inherit":
            limitations.append("inherited_environment_unresolved")
        if self.environment.secrets:
            limitations.append("secret_environment_values_unresolved")
        return tuple(limitations)

    def compatibility_key(self) -> str:
        canonical = json.dumps(
            self.compatibility_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "runspec1:sha256:" + hashlib.sha256(canonical).hexdigest()


def build_nsys_profile_argv(
    spec: RunSpec,
    output_path: str | Path,
    *,
    nsys_executable: str = "nsys",
) -> list[str]:
    """Build the exact argv for a later ``subprocess`` call with ``shell=False``."""
    if not isinstance(spec, RunSpec):
        raise RunSpecError("spec must be a RunSpec")
    executable = _validate_string(nsys_executable, "nsys_executable")
    output = _validate_string(str(output_path), "output_path")
    options = spec.trace_options
    argv = [
        executable,
        "profile",
        "-o",
        output,
        "--force-overwrite=true",
        f"--sample={options.sample}",
        f"--cpuctxsw={options.cpuctxsw}",
        f"--trace={','.join(options.trace)}",
        "--discard-environment=true",
    ]
    if options.capture_range != "none":
        argv.append(f"--capture-range={options.capture_range}")
    if options.cuda_memory_usage:
        argv.append("--cuda-memory-usage=true")
    if options.pytorch:
        argv.append(f"--pytorch={','.join(options.pytorch)}")
    argv.extend(spec.argv)
    return argv


def validate_secret_boundaries(
    spec: RunSpec, resolved_secrets: Mapping[str, str]
) -> None:
    """Reject declared secret values placed in any persisted RunSpec string.

    This is an execution-boundary check for the local runner. It can only
    protect values whose environment-variable names were declared in
    :attr:`EnvironmentSpec.secrets`; arbitrary strings cannot be identified as
    secrets. The persisted declaration-name list itself is excluded. Secret
    command-line arguments are unsupported in RunSpec v0.1.
    The caller must run this check before writing ``runspec.json`` or logging
    argv. Error messages name the declaration and location, never its value.
    """
    if not isinstance(spec, RunSpec):
        raise RunSpecError("spec must be a RunSpec")
    values = _require_mapping(resolved_secrets, "resolved_secrets")
    for name in values:
        EnvironmentSpec._validate_name(name)
    declared = set(spec.environment.secrets)
    provided = set(values)
    undeclared = sorted(provided - declared)
    if undeclared:
        raise RunSpecError(
            "resolved_secrets contains undeclared name(s): " + ", ".join(undeclared)
        )
    missing = sorted(declared - provided)
    if missing:
        raise RunSpecError(
            "resolved_secrets is missing declared name(s): " + ", ".join(missing)
        )

    def persisted_strings(value: Any, path: str = ""):
        if isinstance(value, Mapping):
            if path == "environment.public":
                for index, (public_name, public_value) in enumerate(value.items()):
                    yield f"environment.public.key[{index}]", public_name
                    yield from persisted_strings(
                        public_value, f"environment.public.value[{index}]"
                    )
                return
            for key, nested in value.items():
                if path == "environment" and key == "secrets":
                    # These persisted strings are declarations, not values.
                    continue
                nested_path = f"{path}.{key}" if path else str(key)
                yield from persisted_strings(nested, nested_path)
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                yield from persisted_strings(nested, f"{path}[{index}]")
        elif isinstance(value, str):
            yield path, value

    persisted = tuple(persisted_strings(spec.to_dict()))
    for secret_name in sorted(declared):
        secret_value = values[secret_name]
        if not isinstance(secret_value, str):
            raise RunSpecError(
                f"resolved secret {secret_name} must be a string"
            )
        if not secret_value:
            continue
        for location, persisted_value in persisted:
            if secret_value in persisted_value:
                suffix = (
                    "; secret command-line arguments are unsupported"
                    if location.startswith("argv[")
                    else ""
                )
                raise RunSpecError(
                    f"declared secret {secret_name} appears in {location}{suffix}"
                )


def validate_persisted_secret_strings(
    payload: Any, resolved_secrets: Mapping[str, str]
) -> None:
    """Reject resolved secrets in persisted string keys and string values.

    Diagnostics identify only the validated secret declaration. They never
    include the secret value or keys from the user-controlled payload.
    Callers must remove secret-name declaration fields before invoking this
    scanner because declaration names are intentionally persistable. Numeric,
    boolean, and null measurements are not converted to text: a secret such as
    ``"1"`` does not make an unrelated integer measurement unsafe.
    """
    values = _require_mapping(resolved_secrets, "resolved_secrets")
    for name in values:
        EnvironmentSpec._validate_name(name)

    def strings(value: Any):
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if not isinstance(key, str):
                    raise RunSpecError("persisted payload mapping keys must be strings")
                yield key
                yield from strings(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                yield from strings(nested)
        elif isinstance(value, str):
            yield value

    persisted = tuple(strings(payload))
    for secret_name in sorted(values):
        secret_value = values[secret_name]
        if not isinstance(secret_value, str):
            raise RunSpecError(f"resolved secret {secret_name} must be a string")
        if secret_value and any(secret_value in value for value in persisted):
            raise RunSpecError(
                f"declared secret {secret_name} appears in a persisted string"
            )
