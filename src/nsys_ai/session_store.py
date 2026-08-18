"""Versioned local storage for the diagnose-to-verdict session loop.

The store owns persistence only. Expensive diagnose, profile, and diff work is
performed by callers before entering one of the short publication methods.
Each JSON file is published untorn. A durable rollback journal protects the
artifact-plus-manifest boundary: a restart after either file was replaced
restores the prior coherent snapshot before returning it. This is recovery,
not a claim that the operating system replaces multiple files atomically. A
completed commit or rollback first renames the active journal to an inactive
tombstone; deleting tombstones is best-effort and never gates snapshot loading.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Literal

try:
    import fcntl
except ImportError:  # pragma: no cover - nsys-ai profiling is supported on Linux
    fcntl = None  # type: ignore[assignment]

from .annotation import (
    PRODUCER,
    DiffLineage,
    EvidenceReport,
    validate_evidence_report_payload,
    validate_trace_selection_payload,
)
from .annotation import (
    SCHEMA_VERSION as EVIDENCE_SCHEMA_VERSION,
)
from .artifact_io import atomic_write_bytes, atomic_write_json, fsync_directory
from .diff_decision import (
    build_diff_decision_record_from_diff_dict,
    write_diff_json_from_diff_dict,
)
from .exceptions import NsysAiError
from .profile_reference import LocalProfileReference, validate_local_profile_reference
from .proposal import (
    PROPOSAL_SCHEMA_VERSION,
    Proposal,
    ProposalError,
    UnsupportedProposalVersionError,
    validate_proposal_against_finding,
)
from .runspec import (
    RUNSPEC_SCHEMA_VERSION,
    RunSpec,
    RunSpecError,
    validate_secret_boundaries,
)

SESSION_SCHEMA_VERSION = "0.1"
DIFF_SCHEMA_VERSION = EVIDENCE_SCHEMA_VERSION
DECISIONS_SCHEMA_VERSION = "0.1"

SessionPhase = Literal["diagnose", "propose", "reprofile", "diff", "accept"]
_PHASES = frozenset({"diagnose", "propose", "reprofile", "diff", "accept"})
_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ARTIFACT_PATHS = {
    "runspec": "runspec.json",
    "findings": "findings.json",
    "proposal": "proposal.json",
    "diff": "diff.json",
    "decisions": "decisions.json",
}
_TRANSACTION_DIR = ".transaction"
_TRANSACTION_STAGING_PREFIX = ".transaction.stage."
_TRANSACTION_CLEANUP_PREFIX = ".transaction.cleanup."
_DIFF_FIELDS = {
    "schema_version",
    "decision",
    "producer",
    "producer_version",
    "diff_id",
    "verdict",
    "comparability_confidence",
    "step_time",
    "category_attribution",
    "communication_summary",
    "idle_summary",
    "before",
    "after",
    "warnings",
    "top_regressions",
    "top_improvements",
    "nvtx_regressions",
    "nvtx_improvements",
    "overlap",
}
_DECISION_FIELDS = {"finding_id", "status", "reason", "decider", "decided_at"}
_UNSET = object()
_DIFF_SIDE_FIELDS = {
    "path",
    "profile_id",
    "gpu",
    "schema_version",
    "product_version",
    "total_gpu_ns",
}
_DIFF_STEP_FIELDS = {"before_ms", "after_ms", "delta_ms", "delta_pct"}
_DIFF_CATEGORY_FIELDS = {"category", *_DIFF_STEP_FIELDS}
_DIFF_AXIS_FIELDS = {
    "axis",
    "title",
    "total_basis",
    *_DIFF_STEP_FIELDS,
    "entries",
}
_DIFF_AXIS_ENTRY_FIELDS = {
    "key",
    "label",
    *_DIFF_STEP_FIELDS,
    "before_count",
    "after_count",
    "classification",
    "selection",
    "metadata",
}
_DIFF_KERNEL_FIELDS = {
    "key",
    "name",
    "demangled",
    "before_total_ns",
    "after_total_ns",
    "delta_ns",
    "before_count",
    "after_count",
    "classification",
    "before_share",
    "after_share",
    "delta_share",
    "selection",
    "diff_lineage",
}
_DIFF_LINEAGE_FIELDS = {
    "diff_id",
    "role",
    "rank",
    "baseline_profile_id",
}
_DIFF_NVTX_FIELDS = {
    "text",
    "before_total_ns",
    "after_total_ns",
    "delta_ns",
    "before_count",
    "after_count",
    "classification",
}
_OVERLAP_REQUIRED_FIELDS = {
    "compute_only_ms",
    "nccl_only_ms",
    "overlap_ms",
    "idle_ms",
    "total_ms",
    "overlap_pct",
    "compute_kernels",
    "nccl_kernels",
}
_OVERLAP_OPTIONAL_FIELDS = {
    "launch_overhead_ms",
    "span_start_ns",
    "span_end_ns",
}
_OVERLAP_DELTA_FIELDS = {
    "compute_only_ms",
    "nccl_only_ms",
    "overlap_ms",
    "idle_ms",
    "total_ms",
    "overlap_pct",
}


class SessionError(NsysAiError):
    """Base class for session persistence failures."""

    error_code = "SESSION_ERROR"


class SessionNotFoundError(SessionError):
    error_code = "SESSION_NOT_FOUND"


class SessionExistsError(SessionError):
    error_code = "SESSION_EXISTS"


class SessionConflictError(SessionError):
    error_code = "SESSION_WRITER_CONFLICT"


class SessionCorruptError(SessionError):
    error_code = "SESSION_CORRUPT"


class UnsupportedSessionVersionError(SessionError):
    error_code = "SESSION_VERSION_UNSUPPORTED"


@dataclass(frozen=True)
class ArtifactReference:
    path: str
    schema_version: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class SessionState:
    session_id: str
    phase: SessionPhase = "diagnose"
    before_profile: LocalProfileReference | None = None
    after_profile: LocalProfileReference | None = None
    artifacts: Mapping[str, ArtifactReference] = dataclass_field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        _validate_session_id(self.session_id)
        if self.phase not in _PHASES:
            raise SessionCorruptError(f"invalid session phase: {self.phase!r}")
        refs = dict(self.artifacts)
        unknown = sorted(set(refs) - set(_ARTIFACT_PATHS))
        if unknown:
            raise SessionCorruptError(
                "session has unknown artifact reference(s): " + ", ".join(unknown)
            )
        object.__setattr__(self, "artifacts", MappingProxyType(refs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_id": self.session_id,
            "phase": self.phase,
            "profiles": {
                "before": _profile_reference_to_dict(self.before_profile),
                "after": _profile_reference_to_dict(self.after_profile),
            },
            "artifacts": {
                name: self.artifacts[name].to_dict() if name in self.artifacts else None
                for name in _ARTIFACT_PATHS
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SessionState:
        _require_exact_keys(
            payload,
            {"schema_version", "session_id", "phase", "profiles", "artifacts"},
            "session.json",
        )
        version = payload.get("schema_version")
        if version != SESSION_SCHEMA_VERSION:
            raise UnsupportedSessionVersionError(
                f"unsupported session schema_version {version!r}; "
                f"expected {SESSION_SCHEMA_VERSION!r}"
            )
        profiles = _require_mapping(payload.get("profiles"), "session profiles")
        _require_exact_keys(profiles, {"before", "after"}, "session profiles")
        artifacts_payload = _require_mapping(payload.get("artifacts"), "session artifacts")
        artifact_keys = set(artifacts_payload)
        current_artifact_keys = set(_ARTIFACT_PATHS)
        legacy_artifact_keys = current_artifact_keys - {"decisions"}
        if artifact_keys not in {
            frozenset(current_artifact_keys),
            frozenset(legacy_artifact_keys),
        }:
            _require_exact_keys(artifacts_payload, current_artifact_keys, "session artifacts")
        artifacts: dict[str, ArtifactReference] = {}
        for name, value in artifacts_payload.items():
            if value is None:
                continue
            artifact = _require_mapping(value, f"session artifact {name}")
            _require_exact_keys(
                artifact, {"path", "schema_version", "sha256"}, f"artifact {name}"
            )
            expected_path = _ARTIFACT_PATHS[name]
            if artifact.get("path") != expected_path:
                raise SessionCorruptError(
                    f"artifact {name} path must be {expected_path!r}"
                )
            version_value = artifact.get("schema_version")
            if not isinstance(version_value, str) or not version_value:
                raise SessionCorruptError(
                    f"artifact {name} schema_version must be a non-empty string"
                )
            digest = artifact.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise SessionCorruptError(
                    f"artifact {name} sha256 must be 64 lowercase hex characters"
                )
            artifacts[name] = ArtifactReference(expected_path, version_value, digest)
        session_id = payload.get("session_id")
        phase = payload.get("phase")
        if not isinstance(session_id, str):
            raise SessionCorruptError("session_id must be a string")
        if not isinstance(phase, str):
            raise SessionCorruptError("session phase must be a string")
        return cls(
            session_id=session_id,
            phase=phase,  # type: ignore[arg-type]
            before_profile=_profile_reference_from_dict(profiles.get("before")),
            after_profile=_profile_reference_from_dict(profiles.get("after")),
            artifacts=artifacts,
        )


@dataclass(frozen=True)
class SessionSnapshot:
    state: SessionState
    runspec: RunSpec | None
    findings: EvidenceReport | None
    proposal: Proposal | None
    diff: Mapping[str, Any] | None
    decisions: tuple[Mapping[str, Any], ...] = ()


class SessionStore:
    """Read and create sessions rooted at ``.nsys-ai/sessions``."""

    def __init__(self, root: str | os.PathLike[str] = ".nsys-ai/sessions"):
        self.root = Path(root).expanduser().resolve(strict=False)
        self.lock_root = self.root.parent / "locks"

    def create(
        self,
        session_id: str,
        *,
        before_profile: LocalProfileReference | None = None,
    ) -> SessionState:
        """Atomically create the exact v0 session directory layout."""
        _validate_session_id(session_id)
        if before_profile is not None:
            validate_local_profile_reference(before_profile, require_file=True)
        self._ensure_roots()
        with self._lock(session_id, "writer", exclusive=True, blocking=False):
            destination = self._session_dir(session_id)
            if destination.exists():
                raise SessionExistsError(f"session already exists: {session_id}")
            temporary = Path(tempfile.mkdtemp(prefix=f".{session_id}.", dir=self.root))
            try:
                (temporary / "logs").mkdir(mode=0o700)
                state = SessionState(session_id=session_id, before_profile=before_profile)
                atomic_write_json(temporary / "session.json", state.to_dict())
                os.replace(temporary, destination)
                fsync_directory(self.root)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
            return state

    def load(self, session_id: str) -> SessionSnapshot:
        """Reload a coherent snapshot, rolling back an interrupted publication."""
        _validate_session_id(session_id)
        self._require_session(session_id)
        with self._lock(session_id, "state", exclusive=True, blocking=True):
            return self._load_unlocked(session_id)

    def writer(self, session_id: str) -> SessionWriter:
        """Claim the session's sole writer without waiting behind another process."""
        _validate_session_id(session_id)
        self._require_session(session_id)
        handle = self._acquire_lock(session_id, "writer", exclusive=True, blocking=False)
        try:
            self.load(session_id)
        except BaseException:
            handle.close()
            raise
        return SessionWriter(self, session_id, handle)

    def _load_unlocked(self, session_id: str) -> SessionSnapshot:
        self._recover_transaction(session_id)
        session_dir = self._session_dir(session_id)
        state_payload = _read_json(session_dir / "session.json", "session.json")
        state = SessionState.from_dict(_require_mapping(state_payload, "session.json"))
        if state.session_id != session_id:
            raise SessionCorruptError(
                f"session directory {session_id!r} contains id {state.session_id!r}"
            )

        runspec = None
        findings = None
        proposal = None
        diff = None
        decisions: tuple[Mapping[str, Any], ...] = ()
        if "runspec" in state.artifacts:
            self._check_artifact_version(state, "runspec", RUNSPEC_SCHEMA_VERSION)
            self._verify_artifact(state, "runspec")
            try:
                runspec = RunSpec.from_dict(
                    _require_mapping(
                        _read_json(session_dir / "runspec.json", "runspec.json"),
                        "runspec.json",
                    )
                )
            except RunSpecError as exc:
                raise SessionCorruptError(f"invalid runspec.json: {exc}") from exc
        if "findings" in state.artifacts:
            self._check_artifact_version(state, "findings", EVIDENCE_SCHEMA_VERSION)
            self._verify_artifact(state, "findings")
            payload = _require_mapping(
                _read_json(session_dir / "findings.json", "findings.json"),
                "findings.json",
            )
            if payload.get("schema_version") != state.artifacts["findings"].schema_version:
                raise UnsupportedSessionVersionError(
                    "findings.json schema_version does not match session.json: "
                    f"{payload.get('schema_version')!r}"
                )
            findings = _rehydrate_evidence_payload(
                dict(payload), error_type=SessionCorruptError
            )
            _validate_findings_provenance(
                state.before_profile, findings, error_type=SessionCorruptError
            )
        if "proposal" in state.artifacts:
            self._check_artifact_version(state, "proposal", PROPOSAL_SCHEMA_VERSION)
            self._verify_artifact(state, "proposal")
            try:
                proposal = Proposal.from_dict(
                    _require_mapping(
                        _read_json(session_dir / "proposal.json", "proposal.json"),
                        "proposal.json",
                    )
                )
            except UnsupportedProposalVersionError as exc:
                raise UnsupportedSessionVersionError(str(exc)) from exc
            except ProposalError as exc:
                raise SessionCorruptError(f"invalid proposal.json: {exc}") from exc
            _validate_proposal_pointer(state, findings, proposal, error_type=SessionCorruptError)
        if "diff" in state.artifacts:
            self._check_artifact_version(state, "diff", DIFF_SCHEMA_VERSION)
            self._verify_artifact(state, "diff")
            diff_payload = _read_json(session_dir / "diff.json", "diff.json")
            parsed_diff = dict(_require_mapping(diff_payload, "diff.json"))
            if parsed_diff.get("schema_version") != DIFF_SCHEMA_VERSION:
                raise UnsupportedSessionVersionError(
                    "unsupported diff.json schema_version "
                    f"{parsed_diff.get('schema_version')!r}"
                )
            try:
                _validate_diff_payload(parsed_diff)
            except (TypeError, ValueError) as exc:
                raise SessionCorruptError(f"invalid diff.json: {exc}") from exc
            diff = MappingProxyType(parsed_diff)
            _validate_diff_references(state, diff, error_type=SessionCorruptError)
        if "decisions" in state.artifacts:
            self._check_artifact_version(
                state, "decisions", DECISIONS_SCHEMA_VERSION
            )
            self._verify_artifact(state, "decisions")
            decisions_payload = _read_json(
                session_dir / "decisions.json", "decisions.json"
            )
            decisions = _validate_decisions_payload(
                decisions_payload, error_type=SessionCorruptError
            )
        snapshot = SessionSnapshot(state, runspec, findings, proposal, diff, decisions)
        _validate_snapshot_invariants(snapshot)
        return snapshot

    def _begin_transaction(
        self, session_id: str, artifact_name: str | tuple[str, ...]
    ) -> None:
        session_dir = self._session_dir(session_id)
        journal = session_dir / _TRANSACTION_DIR
        if journal.exists():
            raise SessionCorruptError("session has an unrecovered publication journal")
        names = (artifact_name,) if isinstance(artifact_name, str) else artifact_name
        if not names or any(name not in _ARTIFACT_PATHS for name in names):
            raise SessionCorruptError("transaction references an unknown artifact")
        staging = Path(
            tempfile.mkdtemp(prefix=_TRANSACTION_STAGING_PREFIX, dir=session_dir)
        )
        try:
            atomic_write_bytes(
                staging / "session.json", _read_bytes(session_dir / "session.json", "session.json")
            )
            if len(names) == 1:
                name = names[0]
                artifact_path = session_dir / _ARTIFACT_PATHS[name]
                artifact_existed = artifact_path.is_file()
                if artifact_existed:
                    atomic_write_bytes(
                        staging / "artifact.json",
                        _read_bytes(artifact_path, artifact_path.name),
                    )
                metadata: Mapping[str, Any] = {
                    "artifact": name,
                    "artifact_existed": artifact_existed,
                }
            else:
                entries = []
                for name in names:
                    artifact_path = session_dir / _ARTIFACT_PATHS[name]
                    artifact_existed = artifact_path.is_file()
                    if artifact_existed:
                        atomic_write_bytes(
                            staging / f"artifact-{name}.json",
                            _read_bytes(artifact_path, artifact_path.name),
                        )
                    entries.append(
                        {"name": name, "artifact_existed": artifact_existed}
                    )
                metadata = {"artifacts": entries}
            atomic_write_json(staging / "transaction.json", metadata)
            fsync_directory(staging)
            os.replace(staging, journal)
            fsync_directory(session_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _finish_transaction(self, session_id: str) -> None:
        session_dir = self._session_dir(session_id)
        tombstone = self._deactivate_transaction(session_dir)
        self._remove_inactive_transaction(session_dir, tombstone)

    @staticmethod
    def _deactivate_transaction(session_dir: Path) -> Path:
        journal = session_dir / _TRANSACTION_DIR
        tombstone = session_dir / (
            _TRANSACTION_CLEANUP_PREFIX + uuid.uuid4().hex
        )
        os.replace(journal, tombstone)
        fsync_directory(session_dir)
        return tombstone

    @staticmethod
    def _remove_inactive_transaction(session_dir: Path, path: Path) -> None:
        try:
            shutil.rmtree(path)
            fsync_directory(session_dir)
        except OSError:
            pass

    def _recover_transaction(self, session_id: str) -> None:
        session_dir = self._session_dir(session_id)
        journal = session_dir / _TRANSACTION_DIR
        if journal.is_dir():
            metadata = _require_mapping(
                _read_json(journal / "transaction.json", "transaction journal"),
                "transaction journal",
            )
            if "artifact" in metadata:
                _require_exact_keys(
                    metadata,
                    {"artifact", "artifact_existed"},
                    "transaction journal",
                )
                entries = [
                    {
                        "name": metadata.get("artifact"),
                        "artifact_existed": metadata.get("artifact_existed"),
                        "backup": "artifact.json",
                    }
                ]
            else:
                _require_exact_keys(metadata, {"artifacts"}, "transaction journal")
                raw_entries = metadata.get("artifacts")
                if not isinstance(raw_entries, list):
                    raise SessionCorruptError("transaction journal metadata is invalid")
                entries = [
                    {
                        "name": entry.get("name") if isinstance(entry, Mapping) else None,
                        "artifact_existed": (
                            entry.get("artifact_existed")
                            if isinstance(entry, Mapping)
                            else None
                        ),
                        "backup": (
                            f"artifact-{entry.get('name')}.json"
                            if isinstance(entry, Mapping)
                            else ""
                        ),
                    }
                    for entry in raw_entries
                ]
            for entry in entries:
                artifact_name = entry["name"]
                artifact_existed = entry["artifact_existed"]
                backup = entry["backup"]
                if artifact_name not in _ARTIFACT_PATHS or not isinstance(
                    artifact_existed, bool
                ):
                    raise SessionCorruptError("transaction journal metadata is invalid")
                artifact_path = session_dir / _ARTIFACT_PATHS[artifact_name]
                if artifact_existed:
                    atomic_write_bytes(
                        artifact_path,
                        _read_bytes(journal / backup, "journal artifact backup"),
                    )
                else:
                    artifact_path.unlink(missing_ok=True)
            fsync_directory(session_dir)
            atomic_write_bytes(
                session_dir / "session.json",
                _read_bytes(journal / "session.json", "journal manifest backup"),
            )
            tombstone = self._deactivate_transaction(session_dir)
            self._remove_inactive_transaction(session_dir, tombstone)
        for pattern in (
            f"{_TRANSACTION_STAGING_PREFIX}*",
            f"{_TRANSACTION_CLEANUP_PREFIX}*",
        ):
            for inactive in session_dir.glob(pattern):
                if inactive.is_dir():
                    self._remove_inactive_transaction(session_dir, inactive)

    def _verify_artifact(self, state: SessionState, name: str) -> None:
        path = self._session_dir(state.session_id) / _ARTIFACT_PATHS[name]
        actual = _sha256_file(path, f"{name}.json")
        if actual != state.artifacts[name].sha256:
            raise SessionCorruptError(
                f"{name}.json does not match session.json; "
                "a prior publication may have been interrupted"
            )

    @staticmethod
    def _check_artifact_version(
        state: SessionState, name: str, expected: str
    ) -> None:
        actual = state.artifacts[name].schema_version
        if actual != expected:
            raise UnsupportedSessionVersionError(
                f"unsupported {name} artifact schema_version {actual!r}; expected {expected!r}"
            )

    def _require_session(self, session_id: str) -> None:
        if not self._session_dir(session_id).is_dir():
            raise SessionNotFoundError(f"session not found: {session_id}")

    def _session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    def _ensure_roots(self) -> None:
        for directory in (self.root, self.lock_root):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            directory.chmod(0o700)

    @contextmanager
    def _lock(
        self, session_id: str, kind: str, *, exclusive: bool, blocking: bool
    ) -> Iterator[BinaryIO]:
        handle = self._acquire_lock(
            session_id, kind, exclusive=exclusive, blocking=blocking
        )
        try:
            yield handle
        finally:
            handle.close()

    def _acquire_lock(
        self, session_id: str, kind: str, *, exclusive: bool, blocking: bool
    ) -> BinaryIO:
        if fcntl is None:
            raise SessionError("cross-process session locking requires POSIX fcntl")
        self._ensure_roots()
        path = self.lock_root / f"{session_id}.{kind}.lock"
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        os.fchmod(descriptor, 0o600)
        handle = os.fdopen(descriptor, "a+b", buffering=0)
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if not blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(handle.fileno(), operation)
        except BlockingIOError as exc:
            handle.close()
            raise SessionConflictError(
                f"session {session_id!r} already has an active writer"
            ) from exc
        return handle


class SessionWriter:
    """Exclusive writer lease whose publication methods hold only a short state lock."""

    def __init__(self, store: SessionStore, session_id: str, handle: BinaryIO):
        self.store = store
        self.session_id = session_id
        self._handle: BinaryIO | None = handle

    def __enter__(self) -> SessionWriter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def publish_runspec(
        self,
        runspec: RunSpec,
        *,
        resolved_secrets: Mapping[str, str] | None = None,
    ) -> SessionState:
        self._ensure_open()
        if not isinstance(runspec, RunSpec):
            raise TypeError("runspec must be a RunSpec")
        validate_secret_boundaries(
            runspec, resolved_secrets if resolved_secrets is not None else {}
        )

        def validate_snapshot(snapshot: SessionSnapshot) -> None:
            if snapshot.state.phase != "diagnose" and snapshot.runspec != runspec:
                raise ValueError("runspec.json cannot change after proposal phase")
            if (
                snapshot.proposal is not None
                and snapshot.proposal.verification != runspec
            ):
                raise ValueError(
                    "runspec.json must match the Proposal verification RunSpec"
                )

        return self._publish(
            "runspec",
            RUNSPEC_SCHEMA_VERSION,
            lambda path: atomic_write_bytes(path, runspec.canonical_json_bytes()),
            snapshot_validator=validate_snapshot,
        )

    def publish_findings(
        self,
        findings: EvidenceReport,
        *,
        before_profile: LocalProfileReference | None = None,
    ) -> SessionState:
        self._ensure_open()
        if not isinstance(findings, EvidenceReport):
            raise TypeError("findings must be an EvidenceReport")
        if before_profile is not None:
            validate_local_profile_reference(before_profile, require_file=True)

        try:
            payload = findings.to_dict()
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("findings cannot be serialized as EvidenceReport") from exc
        _require_json_serializable(payload, "findings")
        canonical_findings = _rehydrate_evidence_payload(payload, error_type=ValueError)

        def validate_snapshot(snapshot: SessionSnapshot) -> None:
            _require_phase(snapshot, {"diagnose"}, "publish findings")
            state = self._replace_state(
                snapshot.state,
                before_profile=(
                    before_profile
                    if before_profile is not None
                    else snapshot.state.before_profile
                ),
            )
            _validate_findings_provenance(
                state.before_profile, canonical_findings, error_type=ValueError
            )
            if snapshot.proposal is not None:
                _validate_proposal_pointer(
                    state,
                    canonical_findings,
                    snapshot.proposal,
                    error_type=ValueError,
                )
            if snapshot.diff is not None:
                _validate_diff_references(
                    state, snapshot.diff, error_type=ValueError
                )

        return self._publish(
            "findings",
            EVIDENCE_SCHEMA_VERSION,
            lambda path: atomic_write_json(path, payload),
            phase="diagnose",
            before_profile=(
                before_profile if before_profile is not None else _UNSET
            ),
            snapshot_validator=validate_snapshot,
        )

    def publish_after_profile(self, profile: LocalProfileReference) -> SessionState:
        self._ensure_open()
        validate_local_profile_reference(profile, require_file=True)
        return self._update_state(phase="reprofile", after_profile=profile)

    def publish_proposal(self, proposal: Proposal) -> SessionState:
        self._ensure_open()
        if not isinstance(proposal, Proposal):
            raise TypeError("proposal must be a Proposal")

        def validate_snapshot(snapshot: SessionSnapshot) -> None:
            _require_phase(
                snapshot, {"diagnose", "propose"}, "publish a proposal"
            )
            if proposal.verification != snapshot.runspec:
                raise ValueError(
                    "Proposal verification RunSpec must match runspec.json"
                )
            _validate_proposal_pointer(
                snapshot.state,
                snapshot.findings,
                proposal,
                error_type=ValueError,
            )
            if any(
                decision.get("finding_id") == proposal.source_finding_id
                for decision in snapshot.decisions
            ):
                raise ValueError(
                    "finding already has a recorded session decision; choose another finding"
                )

        return self._publish(
            "proposal",
            PROPOSAL_SCHEMA_VERSION,
            lambda path: atomic_write_bytes(path, proposal.canonical_json_bytes()),
            phase="propose",
            snapshot_validator=validate_snapshot,
        )

    def publish_diff(self, diff: Mapping[str, Any]) -> SessionState:
        self._ensure_open()
        if not isinstance(diff, Mapping):
            raise TypeError("diff must be a mapping")
        payload = dict(diff)
        _validate_diff_payload(payload, require_undecided=True)

        def validate_snapshot(snapshot: SessionSnapshot) -> None:
            _require_phase(snapshot, {"reprofile", "diff"}, "publish a diff")
            _validate_diff_references(
                snapshot.state, payload, error_type=ValueError
            )

        return self._publish(
            "diff",
            DIFF_SCHEMA_VERSION,
            lambda path: write_diff_json_from_diff_dict(payload, path=path)[0],
            phase="diff",
            snapshot_validator=validate_snapshot,
        )

    def publish_decision(
        self,
        decision: str,
        reason: str,
        *,
        decider: str | None = None,
        decided_at: str | None = None,
    ) -> tuple[SessionState, Mapping[str, Any], tuple[str, ...]]:
        self._ensure_open()
        with self.store._lock(
            self.session_id, "state", exclusive=True, blocking=True
        ):
            snapshot = self.store._load_unlocked(self.session_id)
            _require_phase(snapshot, {"diff"}, "publish a decision")
            if snapshot.diff is None:
                raise ValueError("publish a diff before recording a decision")
            if _validate_diff_decision(snapshot.diff):
                raise ValueError("diff already has a decision")
            status = decision.strip().lower()
            if status in {"accept", "accepted"}:
                status = "accepted"
            elif status in {"reject", "rejected"}:
                status = "rejected"
            else:
                raise ValueError("decision must be accept or reject")
            candidate, warnings = build_diff_decision_record_from_diff_dict(
                dict(snapshot.diff),
                decision=status,
                reason=reason,
                decider=decider,
                decided_at=decided_at,
            )
            _validate_diff_payload(candidate)
            finding_id = snapshot.proposal.source_finding_id if snapshot.proposal else ""
            if not finding_id:
                raise ValueError(
                    "the published proposal must identify a finding before recording a decision"
                )
            if any(
                record.get("finding_id") == finding_id for record in snapshot.decisions
            ):
                raise ValueError(
                    "finding already has a recorded session decision; choose another finding"
                )
            decision_record = {
                "finding_id": finding_id,
                "status": status,
                "reason": candidate["decision"]["reason"],
                "decider": candidate["decision"]["decider"],
                "decided_at": candidate["decision"]["decided_at"],
            }
            decisions_payload = _decisions_payload(
                (*snapshot.decisions, MappingProxyType(decision_record))
            )
            session_dir = self.store._session_dir(self.session_id)
            if status == "rejected":
                self.store._begin_transaction(
                    self.session_id, ("decisions", "proposal", "diff")
                )
                atomic_write_json(session_dir / "decisions.json", decisions_payload)
                (session_dir / "proposal.json").unlink(missing_ok=True)
                (session_dir / "diff.json").unlink(missing_ok=True)
            else:
                self.store._begin_transaction(self.session_id, ("decisions", "diff"))
                atomic_write_json(session_dir / "decisions.json", decisions_payload)
                write_diff_json_from_diff_dict(
                    candidate,
                    path=session_dir / "diff.json",
                )
            artifacts = dict(snapshot.state.artifacts)
            artifacts["decisions"] = ArtifactReference(
                _ARTIFACT_PATHS["decisions"],
                DECISIONS_SCHEMA_VERSION,
                _sha256_file(session_dir / "decisions.json", "decisions.json"),
            )
            if status == "rejected":
                artifacts.pop("proposal", None)
                artifacts.pop("diff", None)
                state = self._replace_state(
                    snapshot.state,
                    phase="propose",
                    after_profile=None,
                    artifacts=artifacts,
                )
            else:
                artifacts["diff"] = ArtifactReference(
                    _ARTIFACT_PATHS["diff"],
                    DIFF_SCHEMA_VERSION,
                    _sha256_file(session_dir / "diff.json", "diff.json"),
                )
                state = self._replace_state(
                    snapshot.state, phase="accept", artifacts=artifacts
                )
            atomic_write_json(
                session_dir / "session.json",
                state.to_dict(),
            )
            self.store._finish_transaction(self.session_id)
            return state, MappingProxyType(candidate), tuple(warnings)

    def _publish(
        self,
        name: str,
        schema_version: str,
        publisher,
        *,
        phase: SessionPhase | None = None,
        before_profile: LocalProfileReference | None | object = _UNSET,
        snapshot_validator: Callable[[SessionSnapshot], None] | None = None,
    ) -> SessionState:
        self._ensure_open()
        with self.store._lock(
            self.session_id, "state", exclusive=True, blocking=True
        ):
            snapshot = self.store._load_unlocked(self.session_id)
            state = snapshot.state
            if snapshot_validator is not None:
                snapshot_validator(snapshot)
            artifact_path = self.store._session_dir(self.session_id) / _ARTIFACT_PATHS[name]
            self.store._begin_transaction(self.session_id, name)
            publisher(artifact_path)
            artifacts = dict(state.artifacts)
            artifacts[name] = ArtifactReference(
                _ARTIFACT_PATHS[name],
                schema_version,
                _sha256_file(artifact_path, f"{name}.json"),
            )
            state = self._replace_state(
                state,
                phase=phase,
                before_profile=before_profile,
                artifacts=artifacts,
            )
            atomic_write_json(
                self.store._session_dir(self.session_id) / "session.json",
                state.to_dict(),
            )
            self.store._finish_transaction(self.session_id)
            return state

    def _update_state(
        self,
        *,
        phase: SessionPhase,
        after_profile: LocalProfileReference,
    ) -> SessionState:
        self._ensure_open()
        with self.store._lock(
            self.session_id, "state", exclusive=True, blocking=True
        ):
            snapshot = self.store._load_unlocked(self.session_id)
            _require_phase(
                snapshot,
                {"propose", "reprofile"},
                "publish an after profile",
            )
            if snapshot.proposal is None or snapshot.proposal.abstained:
                raise ValueError(
                    "after profile publication requires a non-abstained proposal"
                )
            state = snapshot.state
            state = self._replace_state(
                state, phase=phase, after_profile=after_profile
            )
            if snapshot.diff is not None:
                _validate_diff_references(
                    state, snapshot.diff, error_type=ValueError
                )
            atomic_write_json(
                self.store._session_dir(self.session_id) / "session.json",
                state.to_dict(),
            )
            return state

    @staticmethod
    def _replace_state(
        state: SessionState,
        *,
        phase: SessionPhase | None = None,
        before_profile: LocalProfileReference | None | object = _UNSET,
        after_profile: LocalProfileReference | None | object = _UNSET,
        artifacts: Mapping[str, ArtifactReference] | None | object = _UNSET,
    ) -> SessionState:
        return SessionState(
            session_id=state.session_id,
            phase=phase or state.phase,
            before_profile=(
                state.before_profile if before_profile is _UNSET else before_profile
            ),
            after_profile=(
                state.after_profile if after_profile is _UNSET else after_profile
            ),
            artifacts=(state.artifacts if artifacts is _UNSET else artifacts),
        )

    def _ensure_open(self) -> None:
        if self._handle is None:
            raise SessionError("session writer is closed")


def _validate_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
        raise ValueError(
            "session_id must be 1-128 letters, digits, dots, underscores, or hyphens"
        )


def _normalized_local_path(path: str) -> str:
    """Normalize local spelling without dereferencing a user-facing symlink path."""
    return os.path.abspath(os.path.expanduser(path))


def _rehydrate_evidence_payload(
    payload: Mapping[str, Any], *, error_type: type[Exception]
) -> EvidenceReport:
    try:
        rehydrated = validate_evidence_report_payload(payload)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise error_type(
            f"invalid findings.json: {exc}"
        ) from exc
    return rehydrated


def _validate_findings_provenance(
    before: LocalProfileReference | None,
    findings: EvidenceReport,
    *,
    error_type: type[Exception],
) -> None:
    if before is None:
        return
    if not findings.profile_id:
        raise error_type("findings profile_id is required with a before profile")
    if findings.profile_id != before.profile_id:
        raise error_type("findings profile_id does not match the before profile")
    if not findings.profile_path:
        raise error_type("findings profile_path is required with a before profile")
    if _normalized_local_path(findings.profile_path) != _normalized_local_path(
        before.path
    ):
        raise error_type("findings profile_path does not match the before profile")
    for index, finding in enumerate(findings.findings):
        selection = finding.selection
        if selection is not None and selection.profile_id != before.profile_id:
            raise error_type(
                f"findings[{index}] selection profile_id does not match "
                "the before profile"
            )


def _validate_proposal_pointer(
    state: SessionState,
    findings: EvidenceReport | None,
    proposal: Proposal,
    *,
    error_type: type[Exception],
) -> None:
    before = state.before_profile
    if before is None:
        raise error_type("proposal publication requires a before profile reference")
    if not proposal.source_finding_id:
        raise error_type("proposal source_finding_id must be non-empty")
    if findings is None:
        raise error_type("proposal publication requires findings.json")
    matches = [
        finding for finding in findings.findings if finding.id == proposal.source_finding_id
    ]
    if len(matches) != 1:
        raise error_type(
            "proposal source_finding_id must identify exactly one session finding"
        )
    missing_selection_abstention = proposal.abstained and matches[0].selection is None
    if missing_selection_abstention:
        if proposal.source_profile_id:
            raise error_type(
                "proposal without a source selection must have an empty source_profile_id"
            )
    elif proposal.source_profile_id != before.profile_id:
        raise error_type("proposal source_profile_id does not match the before profile")
    try:
        validate_proposal_against_finding(proposal, matches[0])
    except ProposalError as exc:
        raise error_type(str(exc)) from exc


def _validate_diff_references(
    state: SessionState,
    diff: Mapping[str, Any],
    *,
    error_type: type[Exception],
) -> None:
    if state.before_profile is None or state.after_profile is None:
        raise error_type(
            "diff publication requires before and after session profile references"
        )
    for side, expected in (
        ("before", state.before_profile),
        ("after", state.after_profile),
    ):
        reference = diff[side]
        comparisons = (
            ("path", _normalized_local_path(reference["path"]), _normalized_local_path(expected.path)),
            ("profile_id", reference["profile_id"], expected.profile_id),
            ("schema_version", reference["schema_version"], expected.schema_version),
            ("product_version", reference["product_version"], expected.product_version),
        )
        for field, actual, wanted in comparisons:
            if actual != wanted:
                raise error_type(
                    f"diff {side} {field} does not match the session reference"
                )


def _validate_snapshot_invariants(snapshot: SessionSnapshot) -> None:
    state = snapshot.state
    phase = state.phase
    proposal = snapshot.proposal
    diff = snapshot.diff

    if proposal is not None and proposal.verification != snapshot.runspec:
        raise SessionCorruptError(
            "proposal verification RunSpec does not match runspec.json"
        )

    if phase == "diagnose":
        if proposal is not None or state.after_profile is not None or diff is not None:
            raise SessionCorruptError(
                "diagnose phase cannot retain proposal, after profile, or diff artifacts"
            )
        return

    if proposal is None:
        if phase == "propose" and snapshot.decisions:
            if state.after_profile is not None or diff is not None:
                raise SessionCorruptError(
                    "propose phase without a proposal cannot retain an after profile or diff artifact"
                )
            return
        raise SessionCorruptError(f"{phase} phase requires proposal.json")
    if phase == "propose":
        if state.after_profile is not None or diff is not None:
            raise SessionCorruptError(
                "propose phase cannot retain an after profile or diff artifact"
            )
        return

    if proposal.abstained:
        raise SessionCorruptError(f"{phase} phase requires a non-abstained proposal")
    if state.after_profile is None:
        raise SessionCorruptError(f"{phase} phase requires an after profile reference")
    if phase == "reprofile":
        if diff is not None:
            raise SessionCorruptError("reprofile phase cannot retain a diff artifact")
        return

    if diff is None:
        raise SessionCorruptError(f"{phase} phase requires diff.json")
    decided = _validate_diff_decision(diff)
    if phase == "diff" and decided:
        raise SessionCorruptError("diff phase requires an undecided diff.json")
    if phase == "accept" and not decided:
        raise SessionCorruptError("accept phase requires a decided diff.json")


def _require_phase(
    snapshot: SessionSnapshot,
    allowed: set[SessionPhase],
    action: str,
) -> None:
    if snapshot.state.phase not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(
            f"cannot {action} during {snapshot.state.phase} phase; expected {choices}"
        )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SessionCorruptError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise SessionCorruptError(
            f"{label} fields do not match schema; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _profile_reference_to_dict(
    reference: LocalProfileReference | None,
) -> dict[str, Any] | None:
    if reference is None:
        return None
    validate_local_profile_reference(reference, require_file=False)
    return {
        "kind": "local",
        "path": reference.path,
        "storage_kind": reference.storage_kind,
        "resolved_path": reference.resolved_path or reference.path,
        "profile_id": reference.profile_id,
        "export_schema_version": reference.schema_version,
        "product_version": reference.product_version,
        "kernel_count": reference.kernel_count,
    }


def _profile_reference_from_dict(value: Any) -> LocalProfileReference | None:
    if value is None:
        return None
    payload = _require_mapping(value, "profile reference")
    legacy_keys = {
        "kind",
        "path",
        "profile_id",
        "export_schema_version",
        "product_version",
        "kernel_count",
    }
    current_keys = legacy_keys | {"storage_kind", "resolved_path"}
    actual_keys = set(payload)
    if actual_keys == legacy_keys:
        storage_kind = "sqlite"
        resolved_path = payload.get("path")
    elif actual_keys == current_keys:
        storage_kind = payload.get("storage_kind")
        resolved_path = payload.get("resolved_path")
    else:
        _require_exact_keys(payload, current_keys, "profile reference")
    if payload.get("kind") != "local":
        raise SessionCorruptError("profile reference kind must be 'local'")
    try:
        reference = LocalProfileReference(
            path=payload["path"],
            storage_kind=storage_kind,
            resolved_path=resolved_path,
            profile_id=payload["profile_id"],
            schema_version=payload["export_schema_version"],
            product_version=payload["product_version"],
            kernel_count=payload["kernel_count"],
        )
        validate_local_profile_reference(reference, require_file=False)
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionCorruptError(f"invalid local profile reference: {exc}") from exc
    return reference
def _diff_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _diff_exact_fields(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields do not match canonical shape; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _diff_string(value: Any, label: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{label} must be {qualifier}")


def _diff_integer(value: Any, label: str, *, non_negative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if non_negative and value < 0:
        raise ValueError(f"{label} must be non-negative")


def _diff_number(value: Any, label: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        suffix = " or null" if nullable else ""
        raise ValueError(f"{label} must be a finite number{suffix}")


def _validate_diff_measurements(value: Mapping[str, Any], label: str) -> None:
    for field in ("before_ms", "after_ms", "delta_ms"):
        _diff_number(value[field], f"{label}.{field}")
    _diff_number(value["delta_pct"], f"{label}.delta_pct", nullable=True)


def _validate_diff_selection(
    value: Any,
    *,
    label: str,
    expected_profile_id: str,
    expected_source: str | None = None,
) -> None:
    if value is None:
        return
    try:
        selection = validate_trace_selection_payload(value, label=label)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if selection.profile_id != expected_profile_id:
        raise ValueError(f"{label}.profile_id does not match its selected diff side")
    if expected_source is not None and selection.source != expected_source:
        raise ValueError(f"{label}.source must be {expected_source!r}")


def _validate_diff_categories(payload: Mapping[str, Any]) -> None:
    step_time = payload["step_time"]
    if step_time is not None:
        step = _diff_mapping(step_time, "diff step_time")
        _diff_exact_fields(step, _DIFF_STEP_FIELDS, "diff step_time")
        _validate_diff_measurements(step, "diff step_time")

    categories = payload["category_attribution"]
    allowed = {"compute", "communication", "launch_overhead", "idle"}
    seen: set[str] = set()
    for index, item in enumerate(categories):
        label = f"diff category_attribution[{index}]"
        category = _diff_mapping(item, label)
        _diff_exact_fields(category, _DIFF_CATEGORY_FIELDS, label)
        if category["category"] not in allowed or category["category"] in seen:
            raise ValueError(f"{label}.category is invalid or duplicated")
        seen.add(category["category"])
        _validate_diff_measurements(category, label)


def _validate_diff_axis(
    value: Any,
    *,
    field_name: str,
    axis_name: str,
    total_basis: str,
    before_profile_id: str,
    after_profile_id: str,
) -> None:
    if value is None:
        return
    label = f"diff {field_name}"
    axis = _diff_mapping(value, label)
    _diff_exact_fields(axis, _DIFF_AXIS_FIELDS, label)
    if axis["axis"] != axis_name or axis["total_basis"] != total_basis:
        raise ValueError(f"{label} axis metadata is inconsistent")
    _diff_string(axis["title"], f"{label}.title")
    _validate_diff_measurements(axis, label)
    entries = axis["entries"]
    if not isinstance(entries, list):
        raise ValueError(f"{label}.entries must be an array")
    classifications = {
        "regression",
        "improvement",
        "new",
        "removed",
        "neutral",
        "grown",
        "shrunk",
    }
    for index, item in enumerate(entries):
        entry_label = f"{label}.entries[{index}]"
        entry = _diff_mapping(item, entry_label)
        _diff_exact_fields(entry, _DIFF_AXIS_ENTRY_FIELDS, entry_label)
        for field in ("key", "label"):
            _diff_string(entry[field], f"{entry_label}.{field}", allow_empty=True)
        _validate_diff_measurements(entry, entry_label)
        for field in ("before_count", "after_count"):
            _diff_integer(entry[field], f"{entry_label}.{field}", non_negative=True)
        if entry["classification"] not in classifications:
            raise ValueError(f"{entry_label}.classification is invalid")
        metadata = _diff_mapping(entry["metadata"], f"{entry_label}.metadata")
        if axis_name == "communication":
            _diff_exact_fields(
                metadata, {"selection_side"}, f"{entry_label}.metadata"
            )
        else:
            _diff_exact_fields(
                metadata,
                {
                    "device_id",
                    "stream_id",
                    "before_kernel",
                    "after_kernel",
                    "selection_side",
                },
                f"{entry_label}.metadata",
            )
            for field in ("device_id", "stream_id"):
                _diff_integer(
                    metadata[field],
                    f"{entry_label}.metadata.{field}",
                    non_negative=True,
                )
            for field in ("before_kernel", "after_kernel"):
                _diff_string(
                    metadata[field],
                    f"{entry_label}.metadata.{field}",
                    allow_empty=True,
                )
        selection_side = metadata["selection_side"]
        if selection_side not in {"before", "after"}:
            raise ValueError(f"{entry_label}.metadata.selection_side is invalid")
        expected_profile_id = (
            before_profile_id if selection_side == "before" else after_profile_id
        )
        _validate_diff_selection(
            entry["selection"],
            label=f"{entry_label}.selection",
            expected_profile_id=expected_profile_id,
            expected_source=f"diff:{axis_name}_summary",
        )


def _validate_directional_diff_entry(
    entry: Mapping[str, Any], *, field_name: str, label: str
) -> None:
    before = entry["before_total_ns"]
    after = entry["after_total_ns"]
    delta = entry["delta_ns"]
    if delta != after - before:
        raise ValueError(f"{label}.delta_ns must equal after_total_ns - before_total_ns")
    if field_name.endswith("regressions"):
        if delta <= 0:
            raise ValueError(f"{label}.delta_ns must be positive")
        expected_classification = "new" if before == 0 else "regression"
    else:
        if delta >= 0:
            raise ValueError(f"{label}.delta_ns must be negative")
        expected_classification = "removed" if after == 0 else "improvement"
    if entry["classification"] != expected_classification:
        raise ValueError(
            f"{label}.classification must be {expected_classification!r} "
            "for its totals and list"
        )


def _validate_diff_lineage(
    value: Any,
    *,
    label: str,
    diff_id: str,
    role: str,
    rank: int,
    baseline_profile_id: str,
) -> None:
    lineage_payload = _diff_mapping(value, label)
    _diff_exact_fields(lineage_payload, _DIFF_LINEAGE_FIELDS, label)
    for field in ("diff_id", "baseline_profile_id"):
        _diff_string(lineage_payload[field], f"{label}.{field}")
    if lineage_payload["role"] not in {"regression", "improvement", "stable"}:
        raise ValueError(f"{label}.role is invalid")
    _diff_integer(lineage_payload["rank"], f"{label}.rank", non_negative=True)

    lineage = DiffLineage.from_dict(dict(lineage_payload))
    expected = {
        "diff_id": diff_id,
        "role": role,
        "rank": rank,
        "baseline_profile_id": baseline_profile_id,
    }
    if lineage.to_dict() != expected:
        raise ValueError(f"{label} does not match its parent diff entry")


def _validate_diff_kernel_entries(payload: Mapping[str, Any]) -> None:
    after_profile_id = payload["after"]["profile_id"]
    baseline_profile_id = payload["before"]["profile_id"]
    seen_keys: set[str] = set()
    for field_name in ("top_regressions", "top_improvements"):
        expected_role = (
            "regression" if field_name == "top_regressions" else "improvement"
        )
        for index, item in enumerate(payload[field_name]):
            label = f"diff {field_name}[{index}]"
            entry = _diff_mapping(item, label)
            _diff_exact_fields(entry, _DIFF_KERNEL_FIELDS, label)
            for field in ("key", "name", "demangled"):
                _diff_string(entry[field], f"{label}.{field}", allow_empty=True)
            for field in ("before_total_ns", "after_total_ns", "before_count", "after_count"):
                _diff_integer(entry[field], f"{label}.{field}", non_negative=True)
            _diff_integer(entry["delta_ns"], f"{label}.delta_ns")
            for field in ("before_share", "after_share", "delta_share"):
                _diff_number(entry[field], f"{label}.{field}")
            if entry["key"] in seen_keys:
                raise ValueError(f"{label}.key is duplicated across top kernel lists")
            seen_keys.add(entry["key"])
            _validate_directional_diff_entry(
                entry, field_name=field_name, label=label
            )
            if entry["delta_share"] != entry["after_share"] - entry["before_share"]:
                raise ValueError(
                    f"{label}.delta_share must equal after_share - before_share"
                )
            _validate_diff_selection(
                entry["selection"],
                label=f"{label}.selection",
                expected_profile_id=after_profile_id,
                expected_source="diff",
            )
            _validate_diff_lineage(
                entry["diff_lineage"],
                label=f"{label}.diff_lineage",
                diff_id=payload["diff_id"],
                role=expected_role,
                rank=index,
                baseline_profile_id=baseline_profile_id,
            )


def _validate_diff_nvtx_entries(payload: Mapping[str, Any]) -> None:
    seen_text: set[str] = set()
    for field_name in ("nvtx_regressions", "nvtx_improvements"):
        for index, item in enumerate(payload[field_name]):
            label = f"diff {field_name}[{index}]"
            entry = _diff_mapping(item, label)
            _diff_exact_fields(entry, _DIFF_NVTX_FIELDS, label)
            _diff_string(entry["text"], f"{label}.text", allow_empty=True)
            for field in ("before_total_ns", "after_total_ns", "before_count", "after_count"):
                _diff_integer(entry[field], f"{label}.{field}", non_negative=True)
            _diff_integer(entry["delta_ns"], f"{label}.delta_ns")
            if entry["text"] in seen_text:
                raise ValueError(f"{label}.text is duplicated across NVTX lists")
            seen_text.add(entry["text"])
            _validate_directional_diff_entry(
                entry, field_name=field_name, label=label
            )


def _validate_overlap_side(value: Any, label: str) -> None:
    overlap = _diff_mapping(value, label)
    if "error" in overlap:
        allowed = {
            "error",
            "requested_device",
            "requested_trim_ns",
            "available_devices",
            "hint",
        }
        unknown = set(overlap) - allowed
        if unknown or "requested_device" not in overlap:
            raise ValueError(f"{label} diagnostic fields are invalid")
        _diff_string(overlap["error"], f"{label}.error")
        device = overlap["requested_device"]
        if device is not None:
            _diff_integer(device, f"{label}.requested_device", non_negative=True)
        if "requested_trim_ns" in overlap:
            trim = overlap["requested_trim_ns"]
            if not isinstance(trim, list) or len(trim) != 2:
                raise ValueError(f"{label}.requested_trim_ns must contain two integers")
            for index, item in enumerate(trim):
                _diff_integer(item, f"{label}.requested_trim_ns[{index}]")
        if "available_devices" in overlap:
            devices = overlap["available_devices"]
            if not isinstance(devices, Mapping):
                raise ValueError(f"{label}.available_devices must be an object")
            for device_id, count in devices.items():
                if not str(device_id).isdigit():
                    raise ValueError(f"{label}.available_devices keys must be GPU ids")
                _diff_integer(count, f"{label}.available_devices[{device_id}]", non_negative=True)
        if "hint" in overlap:
            _diff_string(overlap["hint"], f"{label}.hint")
        return

    actual = set(overlap)
    allowed = _OVERLAP_REQUIRED_FIELDS | _OVERLAP_OPTIONAL_FIELDS
    if not _OVERLAP_REQUIRED_FIELDS <= actual or actual - allowed:
        raise ValueError(f"{label} fields do not match canonical overlap shape")
    for field in (
        "compute_only_ms",
        "nccl_only_ms",
        "overlap_ms",
        "idle_ms",
        "total_ms",
        "overlap_pct",
        "launch_overhead_ms",
    ):
        if field in overlap:
            _diff_number(overlap[field], f"{label}.{field}")
    for field in ("compute_kernels", "nccl_kernels", "span_start_ns", "span_end_ns"):
        if field in overlap:
            _diff_integer(overlap[field], f"{label}.{field}", non_negative=True)


def _validate_diff_overlap(value: Any) -> None:
    overlap = _diff_mapping(value, "diff overlap")
    _diff_exact_fields(overlap, {"before", "after", "delta"}, "diff overlap")
    _validate_overlap_side(overlap["before"], "diff overlap.before")
    _validate_overlap_side(overlap["after"], "diff overlap.after")
    delta = _diff_mapping(overlap["delta"], "diff overlap.delta")
    unknown = set(delta) - _OVERLAP_DELTA_FIELDS
    if unknown:
        raise ValueError(f"diff overlap.delta has unknown fields: {sorted(unknown)}")
    for field, value in delta.items():
        _diff_number(value, f"diff overlap.delta.{field}")


def _validate_diff_payload(
    payload: Mapping[str, Any], *, require_undecided: bool = False
) -> None:
    actual_fields = set(payload)
    if actual_fields != _DIFF_FIELDS:
        raise ValueError(
            "diff fields do not match canonical to_diff_dict shape; "
            f"missing={sorted(_DIFF_FIELDS - actual_fields)}, "
            f"unknown={sorted(actual_fields - _DIFF_FIELDS)}"
        )
    _require_json_serializable(payload, "diff")
    if payload.get("schema_version") != DIFF_SCHEMA_VERSION:
        raise ValueError(f"diff schema_version must be {DIFF_SCHEMA_VERSION!r}")
    if payload.get("producer") != PRODUCER:
        raise ValueError(f"diff producer must be {PRODUCER!r}")
    for field in ("producer_version", "diff_id"):
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"diff {field} must be a non-empty string")
    if payload.get("verdict") not in {
        "improvement_likely",
        "regression_likely",
        "neutral",
        "inconclusive",
    }:
        raise ValueError("diff verdict is not canonical")
    confidence = payload.get("comparability_confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("diff comparability_confidence must be between 0 and 1")

    for field in (
        "category_attribution",
        "warnings",
        "top_regressions",
        "top_improvements",
        "nvtx_regressions",
        "nvtx_improvements",
    ):
        if not isinstance(payload.get(field), list):
            raise ValueError(f"diff {field} must be an array")
    if any(not isinstance(warning, str) for warning in payload["warnings"]):
        raise ValueError("diff warnings must contain only strings")
    for field in ("step_time", "communication_summary", "idle_summary"):
        value = payload.get(field)
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"diff {field} must be an object or null")
    for side in ("before", "after"):
        reference = payload.get(side)
        if not isinstance(reference, Mapping):
            raise ValueError(f"diff {side} must be an object")
        reference_fields = set(reference)
        if reference_fields != _DIFF_SIDE_FIELDS:
            raise ValueError(
                f"diff {side} fields do not match canonical shape; "
                f"missing={sorted(_DIFF_SIDE_FIELDS - reference_fields)}, "
                f"unknown={sorted(reference_fields - _DIFF_SIDE_FIELDS)}"
            )
        for field in ("path", "profile_id"):
            value = reference.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"diff {side} {field} must be a non-empty string")
        if not Path(reference["path"]).is_absolute():
            raise ValueError(f"diff {side} path must be absolute")
        for field in ("schema_version", "product_version"):
            value = reference.get(field)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"diff {side} {field} must be a string or null")
        gpu = reference.get("gpu")
        if gpu is not None and (isinstance(gpu, bool) or not isinstance(gpu, int)):
            raise ValueError(f"diff {side} gpu must be an integer or null")
        total_gpu_ns = reference.get("total_gpu_ns")
        if (
            isinstance(total_gpu_ns, bool)
            or not isinstance(total_gpu_ns, int)
            or total_gpu_ns < 0
        ):
            raise ValueError(f"diff {side} total_gpu_ns must be non-negative")

    _validate_diff_categories(payload)
    _validate_diff_axis(
        payload["communication_summary"],
        field_name="communication_summary",
        axis_name="communication",
        total_basis="exposed comm",
        before_profile_id=payload["before"]["profile_id"],
        after_profile_id=payload["after"]["profile_id"],
    )
    _validate_diff_axis(
        payload["idle_summary"],
        field_name="idle_summary",
        axis_name="idle",
        total_basis="wall-clock idle",
        before_profile_id=payload["before"]["profile_id"],
        after_profile_id=payload["after"]["profile_id"],
    )
    _validate_diff_kernel_entries(payload)
    _validate_diff_nvtx_entries(payload)
    _validate_diff_overlap(payload["overlap"])

    decided = _validate_diff_decision(payload)
    if require_undecided and decided:
        raise ValueError(
            "publish_diff requires decision to be null; use publish_decision"
        )


def _validate_diff_decision(diff: Mapping[str, Any]) -> bool:
    decision = diff.get("decision")
    if decision is None:
        return False
    if not isinstance(decision, Mapping):
        raise ValueError("diff decision must be an object or null")
    expected = {"status", "reason", "decider", "decided_at"}
    if set(decision) != expected:
        raise ValueError(
            "diff decision requires only status, reason, decider, and decided_at"
        )
    if decision.get("status") not in {"accepted", "rejected"}:
        raise ValueError("diff decision status must be accepted or rejected")
    for field in ("reason", "decider", "decided_at"):
        value = decision.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"diff decision {field} must be a non-empty string")
    return True


def _decisions_payload(
    decisions: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": DECISIONS_SCHEMA_VERSION,
        "decisions": [dict(decision) for decision in decisions],
    }


def _validate_decisions_payload(
    payload: Any, *, error_type: type[Exception]
) -> tuple[Mapping[str, Any], ...]:
    try:
        if not isinstance(payload, Mapping):
            raise ValueError("decisions.json must be an object")
        _require_exact_keys(payload, {"schema_version", "decisions"}, "decisions.json")
        if payload.get("schema_version") != DECISIONS_SCHEMA_VERSION:
            raise UnsupportedSessionVersionError(
                "unsupported decisions.json schema_version "
                f"{payload.get('schema_version')!r}"
            )
        values = payload.get("decisions")
        if not isinstance(values, list):
            raise ValueError("decisions.json decisions must be an array")
        parsed: list[Mapping[str, Any]] = []
        finding_ids: set[str] = set()
        for index, value in enumerate(values):
            if not isinstance(value, Mapping):
                raise ValueError(f"decision {index} must be an object")
            if set(value) != _DECISION_FIELDS:
                raise ValueError(
                    f"decision {index} fields do not match schema"
                )
            finding_id = value.get("finding_id")
            if not isinstance(finding_id, str) or not finding_id.strip():
                raise ValueError(f"decision {index} finding_id must be non-empty")
            if finding_id in finding_ids:
                raise ValueError(f"finding {finding_id!r} has more than one decision")
            finding_ids.add(finding_id)
            if value.get("status") not in {"accepted", "rejected"}:
                raise ValueError(f"decision {index} status is invalid")
            for field_name in ("reason", "decider", "decided_at"):
                field_value = value.get(field_name)
                if not isinstance(field_value, str) or not field_value.strip():
                    raise ValueError(
                        f"decision {index} {field_name} must be non-empty"
                    )
            parsed.append(MappingProxyType(dict(value)))
        return tuple(parsed)
    except (SessionCorruptError, UnsupportedSessionVersionError):
        raise
    except (TypeError, ValueError) as exc:
        raise error_type(str(exc)) from exc


def _require_json_serializable(value: Any, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be JSON serializable: {exc}") from exc


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(
            _read_bytes(path, label),
            parse_constant=_reject_non_finite_json,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise SessionCorruptError(f"invalid {label}: {exc}") from exc


def _reject_non_finite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise SessionCorruptError(f"missing {label}") from exc


def _sha256_file(path: Path, label: str) -> str:
    return hashlib.sha256(_read_bytes(path, label)).hexdigest()
