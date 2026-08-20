"""Public ``nsys-ai profile`` command orchestration.

This module translates CLI inputs into the versioned :class:`RunSpec` and
delegates capture lifecycle to :class:`LocalProfileRunner`.  It deliberately
does not own runner internals or durable session layout.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess  # nosec B404
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .exceptions import NsysAiError
from .profile_runner import LocalProfileRunner, RunProgress, RunStatus
from .runspec import EnvironmentSpec, NsysTraceOptions, RunSpec, RunSpecError

_PROBE_TIMEOUT_SECONDS = 10
_GIT_TIMEOUT_SECONDS = 3
_TRACE_HEADER_RE = re.compile(r"^\s*(?:-t,\s*)?--trace=")
_OPTION_HEADER_RE = re.compile(r"^\s*(?:-[A-Za-z],\s*)?--[A-Za-z0-9-]+(?:=|<)")
_QUOTED_VALUE_RE = re.compile(r"'([^']+)'")


class ProfileCommandError(NsysAiError):
    """The public profiling command could not build or execute its plan."""

    error_code = "PROFILE_COMMAND_FAILED"


def normalize_workload(tokens: Sequence[str]) -> tuple[str, ...]:
    """Remove one CLI delimiter and apply only the documented Python shorthand."""
    workload = list(tokens)
    if workload and workload[0] == "--":
        workload.pop(0)
    if not workload:
        raise ProfileCommandError("a workload command is required")
    if workload[0].endswith(".py"):
        workload.insert(0, sys.executable)
    return tuple(workload)


def resolve_nsys_executable(selected: str) -> str:
    """Resolve the selected Nsight Systems executable exactly once."""
    if not isinstance(selected, str) or not selected or "\x00" in selected:
        raise ProfileCommandError("--nsys must name an executable")
    resolved = shutil.which(selected)
    if resolved is None:
        raise ProfileCommandError(f"Nsight Systems executable not found: {selected}")
    return str(Path(resolved).resolve())


def parse_supported_trace_domains(help_text: str) -> tuple[str, ...]:
    """Parse only the ``-t, --trace=`` option block from nsys help output."""
    lines = help_text.splitlines()
    start = next((i for i, line in enumerate(lines) if _TRACE_HEADER_RE.match(line)), None)
    if start is None:
        raise ProfileCommandError("nsys trace capability output has no --trace option block")

    block: list[str] = []
    for line in lines[start + 1 :]:
        if _OPTION_HEADER_RE.match(line):
            break
        block.append(line)
    block_text = "\n".join(block)
    if "Possible values are" not in block_text or "Select the API" not in block_text:
        raise ProfileCommandError("nsys --trace option block has no domain list")
    possible = block_text.split("Possible values are", 1)[1].split("Select the API", 1)[0]
    values = tuple(dict.fromkeys(_QUOTED_VALUE_RE.findall(possible)))
    domains = tuple(value for value in values if value != "none")
    if not domains or "cuda" not in domains:
        raise ProfileCommandError(
            "nsys trace capability output is unparseable or does not advertise CUDA"
        )
    return domains


def detect_supported_trace_domains(nsys_executable: str) -> tuple[str, ...]:
    """Run the bounded trace capability probe for one resolved executable."""
    try:
        completed = subprocess.run(  # nosec B603
            [nsys_executable, "profile", "--help=trace"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ProfileCommandError(
            f"nsys trace capability probe timed out after {_PROBE_TIMEOUT_SECONDS}s"
        ) from exc
    except OSError as exc:
        raise ProfileCommandError(
            f"nsys trace capability probe could not start: {type(exc).__name__}"
        ) from exc
    if completed.returncode != 0:
        raise ProfileCommandError(
            f"nsys trace capability probe failed with exit code {completed.returncode}"
        )
    return parse_supported_trace_domains(completed.stdout + "\n" + completed.stderr)


def select_trace_domains(
    requested: str,
    supported: Sequence[str],
    *,
    warning: Callable[[str], None] | None = None,
) -> tuple[str, ...]:
    """Resolve ``auto`` or validate an explicit comma-separated trace selection."""
    supported_set = set(supported)
    if "cuda" not in supported_set:
        raise ProfileCommandError("the selected nsys does not advertise CUDA tracing")
    if requested == "auto":
        selected = ["cuda"]
        for optional in ("nvtx", "nccl"):
            if optional in supported_set:
                selected.append(optional)
            elif warning is not None:
                warning(f"nsys does not support optional {optional} tracing; omitting it")
        return tuple(selected)

    selected = requested.split(",")
    if not selected or any(not item for item in selected):
        raise ProfileCommandError("--trace must be 'auto' or a comma-separated domain list")
    duplicates = sorted({item for item in selected if selected.count(item) > 1})
    if duplicates:
        raise ProfileCommandError("--trace contains duplicate domain(s): " + ", ".join(duplicates))
    unsupported = [item for item in selected if item not in supported_set]
    if unsupported:
        raise ProfileCommandError(
            "unsupported trace domain(s): " + ", ".join(unsupported)
        )
    if "cuda" not in selected:
        raise ProfileCommandError("--trace must include cuda")
    return tuple(selected)


def parse_environment(
    public_entries: Sequence[str], secret_names: Sequence[str]
) -> EnvironmentSpec:
    """Validate repeatable public and secret environment declarations."""
    public: dict[str, str] = {}
    public_indexes: dict[str, int] = {}
    for index, entry in enumerate(public_entries):
        field = f"--env entry {index + 1}"
        if "=" not in entry:
            raise RunSpecError(f"{field} must use NAME=VALUE")
        name, value = entry.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise RunSpecError(f"{field} has an invalid variable name")
        if "\x00" in value:
            raise RunSpecError(f"{field} value must not contain NUL bytes")
        if name in public:
            raise RunSpecError(
                f"{field} duplicates --env entry {public_indexes[name] + 1}"
            )
        public[name] = value
        public_indexes[name] = index

    secret_indexes: dict[str, int] = {}
    for index, name in enumerate(secret_names):
        field = f"--secret-env entry {index + 1}"
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise RunSpecError(f"{field} has an invalid variable name")
        if name in secret_indexes:
            raise RunSpecError(
                f"{field} duplicates --secret-env entry {secret_indexes[name] + 1}"
            )
        if name in public_indexes:
            raise RunSpecError(
                f"{field} overlaps --env entry {public_indexes[name] + 1}"
            )
        secret_indexes[name] = index
    return EnvironmentSpec(public=public, secrets=tuple(secret_names))


def discover_git_provenance(
    cwd: Path,
) -> tuple[str | None, str | None, str, bool, str | None]:
    """Return repository identity, worktree identity, and RunSpec cwd.

    The diff itself is never persisted. Only its SHA-256 digest is recorded,
    so a dirty checkout is distinguishable without creating a secret-bearing
    patch artifact.
    """
    absolute_cwd = cwd.resolve()
    git_executable = shutil.which("git")
    if git_executable is None:
        return None, None, str(absolute_cwd), False, None
    git_executable = str(Path(git_executable).resolve())

    def git(*args: str, working_directory: Path) -> str | None:
        try:
            completed = subprocess.run(  # nosec B603
                [git_executable, *args],
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=_GIT_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
        value = completed.stdout.strip()
        return value or None

    def git_bytes(*args: str, working_directory: Path) -> bytes | None:
        try:
            completed = subprocess.run(  # nosec B603
                [git_executable, *args],
                cwd=working_directory,
                capture_output=True,
                timeout=_GIT_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout

    root_text = git("rev-parse", "--show-toplevel", working_directory=absolute_cwd)
    if root_text is None:
        return None, None, str(absolute_cwd), False, None
    root = Path(root_text).resolve()
    try:
        relative = absolute_cwd.relative_to(root).as_posix() or "."
    except ValueError:
        return None, None, str(absolute_cwd), False, None
    commit = git("rev-parse", "--verify", "HEAD", working_directory=root)
    if commit is None or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        return None, None, str(absolute_cwd), False, None

    status = git("status", "--porcelain", working_directory=root)
    dirty = bool(status)
    if not dirty:
        return str(root), commit.lower(), relative, False, None

    diff = git_bytes("diff", "HEAD", "--binary", working_directory=root)
    untracked = git_bytes(
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        working_directory=root,
    )
    if diff is None or untracked is None:
        return str(root), commit.lower(), relative, True, hashlib.sha256(
            b"worktree identity unavailable"
        ).hexdigest()

    digest = hashlib.sha256()

    def add_record(label: bytes, payload: bytes) -> None:
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    add_record(b"git-diff-head\0", diff)
    for encoded_path in untracked.split(b"\0"):
        if not encoded_path:
            continue
        path = root / os.fsdecode(encoded_path)
        try:
            metadata = path.lstat()
            if stat.S_ISREG(metadata.st_mode):
                content_digest = hashlib.sha256()
                with path.open("rb") as file_handle:
                    while chunk := file_handle.read(1 << 20):
                        content_digest.update(chunk)
                payload = (
                    b"regular\0"
                    + stat.S_IMODE(metadata.st_mode).to_bytes(4, "big")
                    + content_digest.digest()
                )
            elif stat.S_ISLNK(metadata.st_mode):
                payload = b"symlink\0" + os.fsencode(os.readlink(path))
            else:
                payload = b"special\0" + stat.S_IMODE(metadata.st_mode).to_bytes(
                    4, "big"
                )
        except OSError as exc:
            payload = b"unreadable\0" + type(exc).__name__.encode("ascii")
        add_record(b"untracked\0" + encoded_path + b"\0", payload)

    diff_sha256 = digest.hexdigest()
    return str(root), commit.lower(), relative, True, diff_sha256


def default_output_leaf(cwd: Path) -> Path:
    """Choose a fresh, sortable local artifact leaf without creating it."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return cwd / ".nsys-ai" / "profiles" / timestamp


def _output_leaf(selected: str | None, cwd: Path) -> Path:
    output = default_output_leaf(cwd) if selected is None else Path(selected).expanduser()
    if not output.is_absolute():
        output = cwd / output
    output = output.resolve()
    if output.exists():
        raise ProfileCommandError(f"output leaf already exists: {output}")
    return output


def _dry_run_payload(
    *,
    nsys_executable: str,
    output: Path,
    workload: Sequence[str],
    environment: EnvironmentSpec,
    trace: Sequence[str],
) -> dict[str, Any]:
    return {
        "mode": "dry-run",
        "nsys_executable": nsys_executable,
        "output": str(output),
        "workload_token_count": len(workload),
        "public_environment_names": sorted(environment.public),
        "secret_environment": {
            name: "unresolved" for name in environment.secrets
        },
        "environment_policy": environment.policy,
        "trace": list(trace),
    }


def run_profile_command(
    args: Any,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    cwd: Path | None = None,
    runner_factory: Callable[[Path, str], LocalProfileRunner] = LocalProfileRunner,
) -> int:
    """Execute the public command and return its documented process exit code."""
    working_directory = (cwd or Path.cwd()).resolve()
    workload = normalize_workload(args.workload)
    environment = parse_environment(args.public_env, args.secret_env)
    environment = EnvironmentSpec(
        policy=args.env_policy,
        public=environment.public,
        secrets=environment.secrets,
    )
    output = _output_leaf(args.output, working_directory)
    nsys_executable = resolve_nsys_executable(args.nsys)
    supported = detect_supported_trace_domains(nsys_executable)
    trace = select_trace_domains(
        args.trace,
        supported,
        warning=lambda message: print(f"Warning: {message}", file=stderr),
    )

    if args.dry_run:
        print(
            json.dumps(
                _dry_run_payload(
                    nsys_executable=nsys_executable,
                    output=output,
                    workload=workload,
                    environment=environment,
                    trace=trace,
                ),
                sort_keys=True,
            ),
            file=stdout,
        )
        return 0

    repository, commit, runspec_cwd, dirty, worktree_diff_sha256 = (
        discover_git_provenance(working_directory)
    )
    spec = RunSpec(
        argv=workload,
        cwd=runspec_cwd,
        repository=repository,
        commit=commit,
        dirty=dirty,
        worktree_diff_sha256=worktree_diff_sha256,
        environment=environment,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
        seed=args.seed,
        expected_gpu_count=args.expected_gpu_count,
        expected_rank_count=args.expected_rank_count,
        trace_options=NsysTraceOptions(
            trace=trace,
            sample=args.sample,
            cpuctxsw=args.cpuctxsw,
            capture_range=args.capture_range,
            cuda_memory_usage=args.cuda_memory_usage,
        ),
        timeout_seconds=args.timeout,
    )
    runner = runner_factory(output, nsys_executable)

    def progress(update: RunProgress) -> None:
        print(
            f"[{update.stage.value}] {update.elapsed_seconds:.1f}s",
            file=stderr,
            flush=True,
        )

    try:
        result = runner.run(spec, progress_callback=progress)
    except KeyboardInterrupt:
        print("Profile cancelled.", file=stderr)
        return 130

    if result.status is RunStatus.CANCELLED:
        print("Profile cancelled.", file=stderr)
        return 130
    if result.status is not RunStatus.SUCCEEDED:
        print(f"Profile failed: {result.status.value}", file=stderr)
        if result.detail:
            print(result.detail, file=stderr)
        if result.stdout_path:
            print(f"stdout: {result.stdout_path}", file=stderr)
        if result.stderr_path:
            print(f"stderr: {result.stderr_path}", file=stderr)
        return 1

    if result.profile is None:
        raise ProfileCommandError("runner returned success without profile metadata")
    print(f"Report: {result.report_path}", file=stdout)
    print(f"SQLite: {result.sqlite_path}", file=stdout)
    print(f"RunSpec: {result.runspec_path}", file=stdout)
    print(f"Profile ID: {result.profile.profile_id}", file=stdout)
    print(f"Export schema: {result.profile.schema_version or 'unknown'}", file=stdout)
    print(f"Nsight version: {result.profile.product_version or 'unknown'}", file=stdout)
    print(f"Kernels: {result.profile.kernel_count}", file=stdout)
    return 0
