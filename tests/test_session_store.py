import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from nsys_ai.annotation import SCHEMA_VERSION as DIFF_SCHEMA_VERSION
from nsys_ai.annotation import EvidenceReport, EvidenceRow, Finding, TraceSelection
from nsys_ai.profile_runner import LocalProfileReference
from nsys_ai.proposal import PROPOSAL_SCHEMA_VERSION, Proposal, generate_proposal
from nsys_ai.runspec import EnvironmentSpec, RunSpec, RunSpecError
from nsys_ai.session_store import (
    SESSION_SCHEMA_VERSION,
    SessionConflictError,
    SessionCorruptError,
    SessionError,
    SessionStore,
    UnsupportedSessionVersionError,
)


def _profile_reference(path: Path, profile_id: str) -> LocalProfileReference:
    path.write_bytes(b"profile bytes remain outside the session")
    return LocalProfileReference(
        path=str(path.absolute()),
        profile_id=profile_id,
        schema_version="3.25.0",
        product_version="2026.2.1.106",
        kernel_count=7,
    )


def _diff(before: LocalProfileReference, after: LocalProfileReference) -> dict:
    return {
        "schema_version": DIFF_SCHEMA_VERSION,
        "decision": None,
        "producer": "nsys-ai",
        "producer_version": "test",
        "diff_id": "diff-test",
        "verdict": "improvement_likely",
        "comparability_confidence": 0.9,
        "step_time": None,
        "category_attribution": [],
        "communication_summary": None,
        "idle_summary": None,
        "warnings": [],
        "before": {
            "path": before.path,
            "profile_id": before.profile_id,
            "gpu": 0,
            "schema_version": before.schema_version,
            "product_version": before.product_version,
            "total_gpu_ns": 10,
        },
        "after": {
            "path": after.path,
            "profile_id": after.profile_id,
            "gpu": 0,
            "schema_version": after.schema_version,
            "product_version": after.product_version,
            "total_gpu_ns": 9,
        },
        "top_regressions": [],
        "top_improvements": [],
        "nvtx_regressions": [],
        "nvtx_improvements": [],
        "overlap": {
            "before": {
                "compute_only_ms": 1.0,
                "nccl_only_ms": 0.0,
                "overlap_ms": 0.0,
                "idle_ms": 0.0,
                "total_ms": 1.0,
                "overlap_pct": 0.0,
                "compute_kernels": 1,
                "nccl_kernels": 0,
            },
            "after": {
                "compute_only_ms": 0.9,
                "nccl_only_ms": 0.0,
                "overlap_ms": 0.0,
                "idle_ms": 0.0,
                "total_ms": 0.9,
                "overlap_pct": 0.0,
                "compute_kernels": 1,
                "nccl_kernels": 0,
            },
            "delta": {"compute_only_ms": -0.1},
        },
    }


def _nested_diff(before: LocalProfileReference, after: LocalProfileReference) -> dict:
    payload = _diff(before, after)
    payload["step_time"] = {
        "before_ms": 1.0,
        "after_ms": 0.9,
        "delta_ms": -0.1,
        "delta_pct": -10.0,
    }
    payload["category_attribution"] = [
        {
            "category": "compute",
            "before_ms": 1.0,
            "after_ms": 0.9,
            "delta_ms": -0.1,
            "delta_pct": -10.0,
        }
    ]
    selection = TraceSelection(
        id="selection-diff",
        profile_id=after.profile_id,
        source="diff:communication_summary",
        start_ns=1,
        end_ns=2,
        gpu_ids=[0],
    ).to_dict()
    payload["communication_summary"] = {
        "axis": "communication",
        "title": "Communication/NCCL Summary",
        "total_basis": "exposed comm",
        "before_ms": 1.0,
        "after_ms": 0.9,
        "delta_ms": -0.1,
        "delta_pct": -10.0,
        "entries": [
            {
                "key": "allreduce",
                "label": "NCCL allreduce",
                "before_ms": 1.0,
                "after_ms": 0.9,
                "delta_ms": -0.1,
                "delta_pct": -10.0,
                "before_count": 1,
                "after_count": 1,
                "classification": "improvement",
                "selection": selection,
                "metadata": {"selection_side": "after"},
            }
        ],
    }
    kernel_selection = TraceSelection(
        id="selection-kernel",
        profile_id=after.profile_id,
        source="diff",
        gpu_ids=[0],
    ).to_dict()
    payload["top_regressions"] = [
        {
            "key": "kernel",
            "name": "kernel",
            "demangled": "kernel",
            "before_total_ns": 1,
            "after_total_ns": 2,
            "delta_ns": 1,
            "before_count": 1,
            "after_count": 1,
            "classification": "regression",
            "before_share": 0.5,
            "after_share": 0.6,
            "delta_share": 0.6 - 0.5,
            "selection": kernel_selection,
        }
    ]
    payload["nvtx_regressions"] = [
        {
            "text": "step",
            "before_total_ns": 1,
            "after_total_ns": 2,
            "delta_ns": 1,
            "before_count": 1,
            "after_count": 1,
            "classification": "regression",
        }
    ]
    return payload


def _finding(
    profile: LocalProfileReference,
    finding_id: str | None = "f1",
    *,
    action: str = "Test the change",
) -> Finding:
    return Finding(
        type="region",
        label="Idle region",
        start_ns=10,
        end_ns=20,
        id=finding_id,
        explanation="Reduce the measured idle region.",
        suggested_actions=[action],
        selection=TraceSelection(
            id=f"selection-{finding_id or 'missing'}",
            profile_id=profile.profile_id,
            source="skill:test",
            start_ns=10,
            end_ns=20,
        ),
    )


def _proposal_id_for_payload(payload: dict) -> str:
    identity = dict(payload)
    identity.pop("proposal_id", None)
    canonical = json.dumps(
        identity,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "proposal1:sha256:" + hashlib.sha256(canonical).hexdigest()


def _reprofiled_session(tmp_path: Path, session_id: str):
    before = _profile_reference(
        tmp_path / f"{session_id}-before.sqlite", f"nsys1:{session_id}:before"
    )
    after = _profile_reference(
        tmp_path / f"{session_id}-after.sqlite", f"nsys1:{session_id}:after"
    )
    finding = _finding(before)
    spec = RunSpec(argv=("true",))
    report = EvidenceReport(
        "diagnosis",
        profile_path=before.path,
        profile_id=before.profile_id,
        findings=[finding],
    )
    proposal = generate_proposal(finding, spec)
    store = SessionStore(tmp_path / "sessions")
    store.create(session_id, before_profile=before)
    with store.writer(session_id) as writer:
        writer.publish_runspec(spec)
        writer.publish_findings(report)
        writer.publish_proposal(proposal)
        writer.publish_after_profile(after)
    return store, before, after, finding, spec, proposal


def test_complete_session_round_trip_uses_exact_layout_and_local_references(tmp_path):
    sessions = tmp_path / ".nsys-ai" / "sessions"
    before = _profile_reference(tmp_path / "before.sqlite", "nsys1:before")
    after = _profile_reference(tmp_path / "after.sqlite", "nsys1:after")
    store = SessionStore(sessions)

    state = store.create("run-001", before_profile=before)
    assert state.phase == "diagnose"
    session_dir = sessions / "run-001"
    assert {item.name for item in session_dir.iterdir()} == {"session.json", "logs"}
    assert (tmp_path / ".nsys-ai" / "locks").is_dir()
    assert stat.S_IMODE(sessions.stat().st_mode) == 0o700
    assert stat.S_IMODE(session_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((session_dir / "logs").stat().st_mode) == 0o700

    runspec = RunSpec(argv=("python3", "train.py"), cwd=str(tmp_path))
    finding = _finding(before)
    report = EvidenceReport(
        "diagnosis",
        profile_path=before.path,
        profile_id=before.profile_id,
        findings=[finding],
    )
    proposal = generate_proposal(finding, runspec)
    with store.writer("run-001") as writer:
        writer.publish_runspec(runspec)
        writer.publish_findings(report)
        writer.publish_proposal(proposal)
        writer.publish_after_profile(after)
        writer.publish_diff(_diff(before, after))
        state, decided, warnings = writer.publish_decision(
            "accept",
            "measured improvement",
            decider="test@example.com",
            decided_at="2026-08-06T00:00:00Z",
        )

    assert state.phase == "accept"
    assert decided["decision"] == {
        "status": "accepted",
        "reason": "measured improvement",
        "decider": "test@example.com",
        "decided_at": "2026-08-06T00:00:00Z",
    }
    assert warnings == ()
    assert {item.name for item in session_dir.iterdir()} == {
        "session.json",
        "runspec.json",
        "findings.json",
        "proposal.json",
        "diff.json",
        "logs",
    }
    assert not list(session_dir.rglob("*.sqlite"))
    for artifact in session_dir.glob("*.json"):
        assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
    for lock in (tmp_path / ".nsys-ai" / "locks").glob("*.lock"):
        assert stat.S_IMODE(lock.stat().st_mode) == 0o600

    restarted = SessionStore(sessions).load("run-001")
    assert restarted.state.phase == "accept"
    assert restarted.state.before_profile == before
    assert restarted.state.after_profile == after
    assert restarted.runspec == runspec
    assert restarted.findings.to_dict() == report.to_dict()
    assert restarted.proposal == proposal
    assert restarted.diff["decision"]["status"] == "accepted"

    manifest = json.loads((session_dir / "session.json").read_text())
    assert manifest["schema_version"] == SESSION_SCHEMA_VERSION
    assert manifest["profiles"]["before"] == {
        "kind": "local",
        "path": before.path,
        "profile_id": before.profile_id,
        "export_schema_version": before.schema_version,
        "product_version": before.product_version,
        "kernel_count": before.kernel_count,
    }
    assert manifest["artifacts"]["findings"]["schema_version"] == DIFF_SCHEMA_VERSION
    assert manifest["artifacts"]["proposal"]["schema_version"] == (
        PROPOSAL_SCHEMA_VERSION
    )
    assert json.loads((session_dir / "diff.json").read_text())["schema_version"] == (
        DIFF_SCHEMA_VERSION
    )


def test_active_writer_conflict_is_observed_from_a_second_process(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("conflict")
    script = """
import sys
from nsys_ai.session_store import SessionStore
store = SessionStore(sys.argv[1])
with store.writer("conflict"):
    print("LOCKED", flush=True)
    sys.stdin.readline()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(store.root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_subprocess_environment(),
    )
    stderr = ""
    try:
        assert process.stdout.readline().strip() == "LOCKED"
        with pytest.raises(SessionConflictError, match="active writer") as conflict:
            store.writer("conflict")
        assert conflict.value.error_code == "SESSION_WRITER_CONFLICT"
    finally:
        _stdout, stderr = process.communicate("\n", timeout=10)
    assert process.returncode == 0, stderr


def test_symlink_alias_cannot_bypass_the_single_writer_lock(tmp_path):
    real_root = tmp_path / "real" / "sessions"
    store = SessionStore(real_root)
    store.create("alias-conflict")
    alias_root = tmp_path / "alias-sessions"
    alias_root.symlink_to(real_root, target_is_directory=True)
    alias_store = SessionStore(alias_root)

    assert alias_store.root == store.root
    with store.writer("alias-conflict"):
        with pytest.raises(SessionConflictError, match="active writer"):
            alias_store.writer("alias-conflict")


def test_new_process_reloads_persisted_phase_and_profile_reference(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    before = _profile_reference(tmp_path / "before.sqlite", "nsys1:before")
    after = _profile_reference(tmp_path / "after.sqlite", "nsys1:after")
    finding = _finding(before)
    spec = RunSpec(argv=("true",))
    store.create("restart", before_profile=before)
    with store.writer("restart") as writer:
        writer.publish_runspec(spec)
        writer.publish_findings(
            EvidenceReport(
                "diagnosis",
                profile_path=before.path,
                profile_id=before.profile_id,
                findings=[finding],
            )
        )
        writer.publish_proposal(generate_proposal(finding, spec))
        writer.publish_after_profile(after)

    script = """
import json, sys
from nsys_ai.session_store import SessionStore
snapshot = SessionStore(sys.argv[1]).load("restart")
print(json.dumps({"phase": snapshot.state.phase, "profile_id": snapshot.state.after_profile.profile_id}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(store.root)],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_environment(),
    )
    assert json.loads(result.stdout) == {
        "phase": "reprofile",
        "profile_id": "nsys1:after",
    }


def test_failed_atomic_replace_preserves_previous_artifact(monkeypatch, tmp_path):
    import nsys_ai.artifact_io as artifact_io

    store = SessionStore(tmp_path / "sessions")
    store.create("atomic")
    with store.writer("atomic") as writer:
        writer.publish_findings(EvidenceReport("old"))
        findings_path = store.root / "atomic" / "findings.json"
        before = findings_path.read_bytes()
        real_replace = artifact_io.os.replace

        def fail_findings_replace(source, destination):
            if Path(destination) == findings_path:
                raise OSError("simulated publication failure")
            return real_replace(source, destination)

        monkeypatch.setattr(artifact_io.os, "replace", fail_findings_replace)
        with pytest.raises(OSError, match="publication failure"):
            writer.publish_findings(EvidenceReport("new"))

    assert findings_path.read_bytes() == before
    monkeypatch.undo()
    assert store.load("atomic").findings.title == "old"
    assert not list(findings_path.parent.glob(".findings.json.*.tmp"))


def test_manifest_failure_recovers_the_previous_snapshot(monkeypatch, tmp_path):
    import nsys_ai.session_store as session_store

    store = SessionStore(tmp_path / "sessions")
    store.create("interrupted")
    with store.writer("interrupted") as writer:
        writer.publish_findings(EvidenceReport("old"))
        manifest_path = store.root / "interrupted" / "session.json"
        findings_path = store.root / "interrupted" / "findings.json"
        manifest_before = json.loads(manifest_path.read_text())
        real_atomic_write_json = session_store.atomic_write_json

        def fail_manifest(path, payload):
            if Path(path) == manifest_path:
                raise OSError("simulated manifest failure")
            return real_atomic_write_json(path, payload)

        monkeypatch.setattr(session_store, "atomic_write_json", fail_manifest)
        with pytest.raises(OSError, match="manifest failure"):
            writer.publish_findings(EvidenceReport("new"))

    assert json.loads(findings_path.read_text())["title"] == "new"
    assert json.loads(manifest_path.read_text()) == manifest_before
    assert manifest_before["phase"] == "diagnose"
    recovered = SessionStore(store.root).load("interrupted")
    assert recovered.findings.title == "old"
    assert json.loads(findings_path.read_text())["title"] == "old"
    assert not (store.root / "interrupted" / ".transaction").exists()


def test_interruption_after_manifest_publication_recovers_in_a_new_process(
    monkeypatch, tmp_path
):
    store = SessionStore(tmp_path / "sessions")
    store.create("manifest-published")
    with store.writer("manifest-published") as writer:
        writer.publish_findings(EvidenceReport("old"))

        def interrupt_cleanup(_session_id):
            raise OSError("simulated crash after manifest publication")

        monkeypatch.setattr(store, "_finish_transaction", interrupt_cleanup)
        with pytest.raises(OSError, match="after manifest publication"):
            writer.publish_findings(EvidenceReport("new"))

    script = """
import json, sys
from nsys_ai.session_store import SessionStore
snapshot = SessionStore(sys.argv[1]).load("manifest-published")
print(json.dumps({"phase": snapshot.state.phase, "title": snapshot.findings.title}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(store.root)],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_environment(),
    )
    assert json.loads(result.stdout) == {"phase": "diagnose", "title": "old"}
    assert not (store.root / "manifest-published" / ".transaction").exists()


def test_committed_tombstone_cleanup_failure_does_not_affect_restart(
    monkeypatch, tmp_path
):
    import nsys_ai.session_store as session_store

    store = SessionStore(tmp_path / "sessions")
    store.create("commit-tombstone")
    with store.writer("commit-tombstone") as writer:
        writer.publish_findings(EvidenceReport("old"))
        real_rmtree = session_store.shutil.rmtree

        def fail_tombstone_cleanup(path, *args, **kwargs):
            if Path(path).name.startswith(".transaction.cleanup."):
                raise OSError("simulated tombstone cleanup failure")
            return real_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(session_store.shutil, "rmtree", fail_tombstone_cleanup)
        writer.publish_findings(EvidenceReport("new"))

    session_dir = store.root / "commit-tombstone"
    assert not (session_dir / ".transaction").exists()
    tombstone = next(session_dir.glob(".transaction.cleanup.*"))
    (tombstone / "transaction.json").unlink()
    monkeypatch.undo()

    script = """
import json, sys
from nsys_ai.session_store import SessionStore
snapshot = SessionStore(sys.argv[1]).load("commit-tombstone")
print(json.dumps({"phase": snapshot.state.phase, "title": snapshot.findings.title}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(store.root)],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_environment(),
    )
    assert json.loads(result.stdout) == {"phase": "diagnose", "title": "new"}
    assert not list(session_dir.glob(".transaction.cleanup.*"))


def test_rollback_tombstone_cleanup_failure_does_not_affect_restart(
    monkeypatch, tmp_path
):
    import nsys_ai.session_store as session_store

    store = SessionStore(tmp_path / "sessions")
    store.create("rollback-tombstone")
    with store.writer("rollback-tombstone") as writer:
        writer.publish_findings(EvidenceReport("old"))
        manifest_path = store.root / "rollback-tombstone" / "session.json"
        real_atomic_write_json = session_store.atomic_write_json

        def fail_manifest(path, payload):
            if Path(path) == manifest_path:
                raise OSError("simulated manifest interruption")
            return real_atomic_write_json(path, payload)

        monkeypatch.setattr(session_store, "atomic_write_json", fail_manifest)
        with pytest.raises(OSError, match="manifest interruption"):
            writer.publish_findings(EvidenceReport("new"))
    monkeypatch.undo()

    real_rmtree = session_store.shutil.rmtree

    def fail_tombstone_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(".transaction.cleanup."):
            raise OSError("simulated tombstone cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(session_store.shutil, "rmtree", fail_tombstone_cleanup)
    recovered = SessionStore(store.root).load("rollback-tombstone")
    assert recovered.findings.title == "old"
    session_dir = store.root / "rollback-tombstone"
    assert not (session_dir / ".transaction").exists()
    tombstone = next(session_dir.glob(".transaction.cleanup.*"))
    (tombstone / "artifact.json").unlink()
    monkeypatch.undo()

    script = """
import json, sys
from nsys_ai.session_store import SessionStore
snapshot = SessionStore(sys.argv[1]).load("rollback-tombstone")
print(json.dumps({"phase": snapshot.state.phase, "title": snapshot.findings.title}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(store.root)],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_environment(),
    )
    assert json.loads(result.stdout) == {"phase": "diagnose", "title": "old"}
    assert not list(session_dir.glob(".transaction.cleanup.*"))


def test_findings_unknown_schema_is_rejected_before_canonical_rehydration(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("future-findings")
    with store.writer("future-findings") as writer:
        writer.publish_findings(EvidenceReport("future compatible"))

    session_dir = store.root / "future-findings"
    findings_path = session_dir / "findings.json"
    payload = json.loads(findings_path.read_text())
    payload["schema_version"] = "99"
    findings_path.write_text(json.dumps(payload))
    digest = hashlib.sha256(findings_path.read_bytes()).hexdigest()
    manifest_path = session_dir / "session.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["findings"]["schema_version"] = "99"
    manifest["artifacts"]["findings"]["sha256"] = digest
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(UnsupportedSessionVersionError, match="unsupported findings"):
        store.load("future-findings")


def test_findings_payload_version_must_match_manifest(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("mismatch")
    with store.writer("mismatch") as writer:
        writer.publish_findings(EvidenceReport("mismatch"))

    session_dir = store.root / "mismatch"
    findings_path = session_dir / "findings.json"
    payload = json.loads(findings_path.read_text())
    payload["schema_version"] = "0.2"
    findings_path.write_text(json.dumps(payload))
    manifest_path = session_dir / "session.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["findings"]["sha256"] = hashlib.sha256(
        findings_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(UnsupportedSessionVersionError, match="does not match"):
        store.load("mismatch")


def test_findings_producer_version_drift_remains_compatible(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("producer-drift")
    with store.writer("producer-drift") as writer:
        writer.publish_findings(EvidenceReport("diagnosis"))

    session_dir = store.root / "producer-drift"
    findings_path = session_dir / "findings.json"
    payload = json.loads(findings_path.read_text())
    payload["producer_version"] = "99.7.3-compatible-writer"
    findings_path.write_text(json.dumps(payload))
    manifest_path = session_dir / "session.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["findings"]["sha256"] = hashlib.sha256(
        findings_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    snapshot = store.load("producer-drift")
    assert snapshot.findings.title == "diagnosis"
    assert json.loads(findings_path.read_text())["producer_version"] == (
        "99.7.3-compatible-writer"
    )


def test_findings_and_diff_must_match_session_profile_identities(tmp_path):
    before = _profile_reference(tmp_path / "before.sqlite", "nsys1:before")
    after = _profile_reference(tmp_path / "after.sqlite", "nsys1:after")
    store = SessionStore(tmp_path / "sessions")
    store.create("identity", before_profile=before)
    finding = _finding(before)
    spec = RunSpec(argv=("true",))
    with store.writer("identity") as writer:
        with pytest.raises(ValueError, match="profile_id is required"):
            writer.publish_findings(EvidenceReport("missing provenance"))
        with pytest.raises(ValueError, match="findings profile_id"):
            writer.publish_findings(
                EvidenceReport("wrong", profile_id="nsys1:other")
            )
        writer.publish_runspec(spec)
        writer.publish_findings(
            EvidenceReport(
                "diagnosis",
                profile_path=before.path,
                profile_id=before.profile_id,
                findings=[finding],
            )
        )
        writer.publish_proposal(generate_proposal(finding, spec))
        writer.publish_after_profile(after)
        mismatched = _diff(before, after)
        mismatched["after"]["profile_id"] = "nsys1:other"
        with pytest.raises(ValueError, match="diff after profile_id"):
            writer.publish_diff(mismatched)

    snapshot = store.load("identity")
    assert snapshot.findings.findings[0].id == finding.id
    assert snapshot.diff is None


def test_malformed_typed_findings_are_rejected_before_publication(tmp_path):
    before = _profile_reference(tmp_path / "before.sqlite", "nsys1:before")
    malformed = [
        Finding(type=None, label="missing type", start_ns=1),
        Finding(
            type="region",
            label="missing selection id",
            start_ns=1,
            selection=TraceSelection(
                id=None,
                profile_id=before.profile_id,
                source="skill:test",
            ),
        ),
        Finding(
            type="region",
            label="missing evidence id",
            start_ns=1,
            evidence=[EvidenceRow(id=None, source_skill="test")],
        ),
    ]
    store = SessionStore(tmp_path / "sessions")
    store.create("malformed-findings", before_profile=before)
    session_dir = store.root / "malformed-findings"
    manifest_before = (session_dir / "session.json").read_bytes()

    with store.writer("malformed-findings") as writer:
        for finding in malformed:
            report = EvidenceReport(
                "diagnosis",
                profile_path=before.path,
                profile_id=before.profile_id,
                findings=[finding],
            )
            with pytest.raises(ValueError, match="fields do not match schema"):
                writer.publish_findings(report)
            assert not (session_dir / "findings.json").exists()
            assert (session_dir / "session.json").read_bytes() == manifest_before

    assert store.load("malformed-findings").findings is None


def test_malformed_findings_payload_is_rejected_on_restart(tmp_path):
    before = _profile_reference(tmp_path / "before.sqlite", "nsys1:before")
    report = EvidenceReport(
        "diagnosis",
        profile_path=before.path,
        profile_id=before.profile_id,
        findings=[_finding(before)],
    )
    store = SessionStore(tmp_path / "sessions")
    store.create("malformed-findings-load", before_profile=before)
    with store.writer("malformed-findings-load") as writer:
        writer.publish_findings(report)

    session_dir = store.root / "malformed-findings-load"
    findings_path = session_dir / "findings.json"
    payload = json.loads(findings_path.read_text())
    payload["findings"][0].pop("type")
    findings_path.write_text(json.dumps(payload))
    manifest_path = session_dir / "session.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["findings"]["sha256"] = hashlib.sha256(
        findings_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SessionCorruptError, match="fields do not match schema"):
        store.load("malformed-findings-load")


def test_wrong_nested_evidence_scalar_and_container_types_are_rejected(tmp_path):
    before = _profile_reference(tmp_path / "before.sqlite", "nsys1:before")
    malformed = [
        Finding(type=123, label="wrong scalar", start_ns=1),
        Finding(
            type="region",
            label="wrong selection container",
            start_ns=1,
            selection=TraceSelection(
                id="selection",
                profile_id=before.profile_id,
                source="skill:test",
                gpu_ids="0",
            ),
        ),
        Finding(
            type="region",
            label="wrong evidence container",
            start_ns=1,
            evidence=[EvidenceRow(id="row", source_skill="test", units=[])],
        ),
    ]
    store = SessionStore(tmp_path / "sessions")
    store.create("typed-findings", before_profile=before)
    session_dir = store.root / "typed-findings"
    manifest_before = (session_dir / "session.json").read_bytes()

    with store.writer("typed-findings") as writer:
        for finding in malformed:
            with pytest.raises(ValueError):
                writer.publish_findings(
                    EvidenceReport(
                        "diagnosis",
                        profile_path=before.path,
                        profile_id=before.profile_id,
                        findings=[finding],
                    )
                )
            assert not (session_dir / "findings.json").exists()
            assert (session_dir / "session.json").read_bytes() == manifest_before


def test_nested_finding_selection_must_match_before_profile_on_publish_and_load(
    tmp_path,
):
    before = _profile_reference(tmp_path / "before.sqlite", "nsys1:before")
    finding = _finding(before)
    report = EvidenceReport(
        "diagnosis",
        profile_path=before.path,
        profile_id=before.profile_id,
        findings=[finding],
    )
    store = SessionStore(tmp_path / "sessions")
    store.create("selection-profile", before_profile=before)
    session_dir = store.root / "selection-profile"
    manifest_before = (session_dir / "session.json").read_bytes()

    finding.selection.profile_id = "nsys1:other"
    with store.writer("selection-profile") as writer:
        with pytest.raises(ValueError, match="selection profile_id"):
            writer.publish_findings(report)
    assert not (session_dir / "findings.json").exists()
    assert (session_dir / "session.json").read_bytes() == manifest_before

    finding.selection.profile_id = before.profile_id
    with store.writer("selection-profile") as writer:
        writer.publish_findings(report)
    findings_path = session_dir / "findings.json"
    payload = json.loads(findings_path.read_text())
    payload["findings"][0]["selection"]["profile_id"] = "nsys1:other"
    findings_path.write_text(json.dumps(payload))
    manifest_path = session_dir / "session.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["findings"]["sha256"] = hashlib.sha256(
        findings_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SessionCorruptError, match="selection profile_id"):
        store.load("selection-profile")


def test_diff_requires_reprofile_phase(tmp_path):
    before = _profile_reference(tmp_path / "before.sqlite", "nsys1:before")
    after = _profile_reference(tmp_path / "after.sqlite", "nsys1:after")
    store = SessionStore(tmp_path / "sessions")
    store.create("missing-refs")
    with store.writer("missing-refs") as writer:
        with pytest.raises(ValueError, match="expected diff, reprofile"):
            writer.publish_diff(_diff(before, after))
    assert store.load("missing-refs").diff is None


@pytest.mark.parametrize("shape", ["minimal", "missing", "unknown"])
def test_diff_publication_requires_exact_canonical_top_level_shape(tmp_path, shape):
    store, before, after, _finding_value, _spec, _proposal = _reprofiled_session(
        tmp_path, "diff-shape"
    )
    payload = _diff(before, after)
    if shape == "minimal":
        payload = {
            "schema_version": DIFF_SCHEMA_VERSION,
            "before": {"profile_id": before.profile_id},
            "after": {"profile_id": after.profile_id},
            "decision": None,
        }
    elif shape == "missing":
        payload.pop("top_regressions")
    else:
        payload["invented"] = True

    session_dir = store.root / "diff-shape"
    manifest_before = (session_dir / "session.json").read_bytes()
    with store.writer("diff-shape") as writer:
        with pytest.raises(ValueError, match="canonical to_diff_dict shape"):
            writer.publish_diff(payload)

    assert not (session_dir / "diff.json").exists()
    assert (session_dir / "session.json").read_bytes() == manifest_before
    assert store.load("diff-shape").state.phase == "reprofile"


@pytest.mark.parametrize("shape", ["missing", "unknown"])
def test_diff_canonical_top_level_shape_is_revalidated_on_restart(tmp_path, shape):
    store, before, after, _finding_value, _spec, _proposal = _reprofiled_session(
        tmp_path, "diff-shape-load"
    )
    with store.writer("diff-shape-load") as writer:
        writer.publish_diff(_diff(before, after))

    session_dir = store.root / "diff-shape-load"
    diff_path = session_dir / "diff.json"
    payload = json.loads(diff_path.read_text())
    if shape == "missing":
        payload.pop("top_regressions")
    else:
        payload["invented"] = True
    diff_path.write_text(json.dumps(payload))
    manifest_path = session_dir / "session.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["diff"]["sha256"] = hashlib.sha256(
        diff_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SessionCorruptError, match="canonical to_diff_dict shape"):
        store.load("diff-shape-load")


@pytest.mark.parametrize(
    "mutation",
    [
        "category_string",
        "category_unknown",
        "axis_selection_side",
        "kernel_selection_profile",
        "nvtx_count_string",
        "overlap_metric_string",
        "kernel_neutral",
        "kernel_delta",
        "kernel_wrong_list",
        "nvtx_delta",
        "nvtx_neutral",
        "nvtx_duplicate",
    ],
)
def test_diff_nested_schema_rejects_malformed_or_invented_data(tmp_path, mutation):
    store, before, after, _finding_value, _spec, _proposal = _reprofiled_session(
        tmp_path, "nested-diff"
    )
    payload = _nested_diff(before, after)
    if mutation == "category_string":
        payload["category_attribution"] = "compute"
    elif mutation == "category_unknown":
        payload["category_attribution"][0]["invented"] = True
    elif mutation == "axis_selection_side":
        payload["communication_summary"]["entries"][0]["metadata"][
            "selection_side"
        ] = "before"
    elif mutation == "kernel_selection_profile":
        payload["top_regressions"][0]["selection"]["profile_id"] = before.profile_id
    elif mutation == "nvtx_count_string":
        payload["nvtx_regressions"][0]["before_count"] = "1"
    elif mutation == "overlap_metric_string":
        payload["overlap"]["before"]["compute_only_ms"] = "1.0"
    elif mutation == "kernel_neutral":
        payload["top_regressions"][0]["classification"] = "neutral"
    elif mutation == "kernel_delta":
        payload["top_regressions"][0]["delta_ns"] = 2
    elif mutation == "kernel_wrong_list":
        payload["top_improvements"] = payload.pop("top_regressions")
        payload["top_regressions"] = []
    elif mutation == "nvtx_delta":
        payload["nvtx_regressions"][0]["delta_ns"] = 2
    elif mutation == "nvtx_neutral":
        payload["nvtx_regressions"][0]["classification"] = "neutral"
    else:
        payload["nvtx_improvements"] = [
            dict(payload["nvtx_regressions"][0], delta_ns=-1, classification="removed")
        ]

    session_dir = store.root / "nested-diff"
    manifest_before = (session_dir / "session.json").read_bytes()
    with store.writer("nested-diff") as writer:
        with pytest.raises(ValueError):
            writer.publish_diff(payload)

    assert not (session_dir / "diff.json").exists()
    assert (session_dir / "session.json").read_bytes() == manifest_before
    assert store.load("nested-diff").state.phase == "reprofile"


def test_diff_nested_selection_side_binding_is_revalidated_on_restart(tmp_path):
    store, before, after, _finding_value, _spec, _proposal = _reprofiled_session(
        tmp_path, "nested-diff-load"
    )
    with store.writer("nested-diff-load") as writer:
        writer.publish_diff(_nested_diff(before, after))

    session_dir = store.root / "nested-diff-load"
    diff_path = session_dir / "diff.json"
    payload = json.loads(diff_path.read_text())
    payload["communication_summary"]["entries"][0]["metadata"][
        "selection_side"
    ] = "before"
    diff_path.write_text(json.dumps(payload))
    manifest_path = session_dir / "session.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["diff"]["sha256"] = hashlib.sha256(
        diff_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SessionCorruptError, match="selected diff side"):
        store.load("nested-diff-load")


@pytest.mark.parametrize("contradiction", ["kernel_delta", "nvtx_neutral"])
def test_directional_diff_invariants_are_revalidated_on_restart(
    tmp_path, contradiction
):
    store, before, after, _finding_value, _spec, _proposal = _reprofiled_session(
        tmp_path, "directional-diff-load"
    )
    with store.writer("directional-diff-load") as writer:
        writer.publish_diff(_nested_diff(before, after))

    session_dir = store.root / "directional-diff-load"
    diff_path = session_dir / "diff.json"
    payload = json.loads(diff_path.read_text())
    if contradiction == "kernel_delta":
        payload["top_regressions"][0]["delta_ns"] = 2
    else:
        payload["nvtx_regressions"][0]["classification"] = "neutral"
    diff_path.write_text(json.dumps(payload))
    manifest_path = session_dir / "session.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["diff"]["sha256"] = hashlib.sha256(
        diff_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SessionCorruptError):
        store.load("directional-diff-load")


@pytest.mark.parametrize(
    ("side", "field", "wrong_value"),
    [
        ("before", "path", "/tmp/not-the-before-profile.sqlite"),
        ("after", "profile_id", "nsys1:not-after"),
        ("before", "schema_version", "9.9"),
        ("after", "product_version", "2099.1"),
    ],
)
def test_diff_side_metadata_must_match_session_references_before_publication(
    tmp_path, side, field, wrong_value
):
    store, before, after, _finding_value, _spec, _proposal = _reprofiled_session(
        tmp_path, "diff-reference"
    )
    payload = _diff(before, after)
    payload[side][field] = wrong_value
    session_dir = store.root / "diff-reference"
    manifest_before = (session_dir / "session.json").read_bytes()

    with store.writer("diff-reference") as writer:
        with pytest.raises(ValueError, match=f"diff {side} {field}"):
            writer.publish_diff(payload)

    assert not (session_dir / "diff.json").exists()
    assert (session_dir / "session.json").read_bytes() == manifest_before
    assert store.load("diff-reference").state.phase == "reprofile"


@pytest.mark.parametrize(
    ("side", "field", "wrong_value"),
    [
        ("before", "path", "/tmp/not-the-before-profile.sqlite"),
        ("after", "profile_id", "nsys1:not-after"),
        ("before", "schema_version", "9.9"),
        ("after", "product_version", "2099.1"),
    ],
)
def test_diff_side_metadata_is_revalidated_on_restart(
    tmp_path, side, field, wrong_value
):
    store, before, after, _finding_value, _spec, _proposal = _reprofiled_session(
        tmp_path, "diff-load-reference"
    )
    with store.writer("diff-load-reference") as writer:
        writer.publish_diff(_diff(before, after))

    session_dir = store.root / "diff-load-reference"
    diff_path = session_dir / "diff.json"
    payload = json.loads(diff_path.read_text())
    payload[side][field] = wrong_value
    diff_path.write_text(json.dumps(payload))
    manifest_path = session_dir / "session.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["diff"]["sha256"] = hashlib.sha256(
        diff_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SessionCorruptError, match=f"diff {side} {field}"):
        store.load("diff-load-reference")


def test_diff_paths_must_be_absolute_and_restart_is_cwd_independent(
    monkeypatch, tmp_path
):
    store, before, after, _finding_value, _spec, _proposal = _reprofiled_session(
        tmp_path, "absolute-diff-paths"
    )
    relative = _diff(before, after)
    relative["before"]["path"] = os.path.relpath(before.path, Path.cwd())
    with store.writer("absolute-diff-paths") as writer:
        with pytest.raises(ValueError, match="diff before path must be absolute"):
            writer.publish_diff(relative)
        writer.publish_diff(_diff(before, after))

    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    snapshot = SessionStore(store.root).load("absolute-diff-paths")
    assert snapshot.diff["before"]["path"] == before.path
    assert snapshot.diff["after"]["path"] == after.path


def test_diff_overlap_diagnostics_round_trip_numeric_device_keys(tmp_path):
    store, before, after, _finding_value, _spec, _proposal = _reprofiled_session(
        tmp_path, "overlap-diagnostic"
    )
    payload = _diff(before, after)
    diagnostic = {
        "error": "no kernels found",
        "requested_device": 7,
        "available_devices": {0: 3},
        "hint": "Try device 0",
    }
    payload["overlap"] = {
        "before": diagnostic,
        "after": diagnostic,
        "delta": {},
    }
    with store.writer("overlap-diagnostic") as writer:
        writer.publish_diff(payload)

    snapshot = store.load("overlap-diagnostic")
    assert snapshot.diff["overlap"]["before"]["available_devices"] == {"0": 3}


def test_runspec_secret_preflight_happens_before_publication(tmp_path):
    secret = "sentinel-secret-value"
    environment = EnvironmentSpec(secrets=("RUNNER_SECRET",))
    unsafe = RunSpec(argv=("python3", "train.py", f"--token={secret}"), environment=environment)
    safe = RunSpec(argv=("python3", "train.py"), environment=environment)
    store = SessionStore(tmp_path / "sessions")
    store.create("secret-boundary")

    with store.writer("secret-boundary") as writer:
        with pytest.raises(RunSpecError) as rejected:
            writer.publish_runspec(
                unsafe, resolved_secrets={"RUNNER_SECRET": secret}
            )
        assert secret not in str(rejected.value)
        assert store.load("secret-boundary").runspec is None
        writer.publish_runspec(safe, resolved_secrets={"RUNNER_SECRET": secret})

    runspec_bytes = (store.root / "secret-boundary" / "runspec.json").read_bytes()
    assert secret.encode() not in runspec_bytes
    assert store.load("secret-boundary").runspec == safe


def test_proposal_future_version_is_rejected_by_manifest_contract(tmp_path):
    before = _profile_reference(tmp_path / "before.sqlite", "nsys1:before")
    store = SessionStore(tmp_path / "sessions")
    store.create("proposal", before_profile=before)
    finding = _finding(before)
    report = EvidenceReport(
        "diagnosis",
        profile_path=before.path,
        profile_id=before.profile_id,
        findings=[finding],
    )
    spec = RunSpec(argv=("true",))
    with store.writer("proposal") as writer:
        writer.publish_runspec(spec)
        writer.publish_findings(report)
        writer.publish_proposal(generate_proposal(finding, spec))

    session_dir = store.root / "proposal"
    proposal_path = session_dir / "proposal.json"
    payload = json.loads(proposal_path.read_text())
    payload["schema_version"] = "99"
    proposal_path.write_text(json.dumps(payload))
    manifest_path = session_dir / "session.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["proposal"]["schema_version"] = "99"
    manifest["artifacts"]["proposal"]["sha256"] = hashlib.sha256(
        proposal_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(UnsupportedSessionVersionError, match="unsupported proposal"):
        store.load("proposal")


def test_invented_proposal_projection_is_rejected_on_publish_and_reload(tmp_path):
    before = _profile_reference(tmp_path / "before.sqlite", "nsys1:before")
    finding = _finding(before)
    report = EvidenceReport(
        "diagnosis",
        profile_path=before.path,
        profile_id=before.profile_id,
        findings=[finding],
    )
    valid = generate_proposal(finding, RunSpec(argv=("true",)))
    invented_payload = valid.to_dict()
    invented_payload["summary"] = "Invented summary not present in the finding"
    invented_payload["proposal_id"] = _proposal_id_for_payload(invented_payload)
    invented = Proposal.from_dict(invented_payload)
    store = SessionStore(tmp_path / "sessions")
    store.create("semantic", before_profile=before)

    with store.writer("semantic") as writer:
        writer.publish_runspec(valid.verification)
        writer.publish_findings(report)
        with pytest.raises(ValueError, match="not deterministically derived"):
            writer.publish_proposal(invented)
        writer.publish_proposal(valid)

    session_dir = store.root / "semantic"
    proposal_path = session_dir / "proposal.json"
    proposal_path.write_bytes(invented.canonical_json_bytes())
    manifest_path = session_dir / "session.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["proposal"]["sha256"] = hashlib.sha256(
        proposal_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SessionCorruptError, match="not deterministically derived"):
        store.load("semantic")


def test_missing_selection_abstention_round_trips_without_weakening_identity(tmp_path):
    before = _profile_reference(tmp_path / "before.sqlite", "nsys1:before")
    finding = _finding(before, "missing-selection")
    finding.selection = None
    report = EvidenceReport(
        "diagnosis",
        profile_path=before.path,
        profile_id=before.profile_id,
        findings=[finding],
    )
    spec = RunSpec(argv=("true",))
    proposal = generate_proposal(finding, spec)
    assert proposal.abstained
    assert proposal.source_profile_id == ""

    invented_payload = proposal.to_dict()
    invented_payload["summary"] = "Invented abstention summary"
    invented_payload["proposal_id"] = _proposal_id_for_payload(invented_payload)
    invented = Proposal.from_dict(invented_payload)

    store = SessionStore(tmp_path / "sessions")
    store.create("missing-selection", before_profile=before)
    with store.writer("missing-selection") as writer:
        writer.publish_runspec(spec)
        writer.publish_findings(report)
        with pytest.raises(ValueError, match="not deterministically derived"):
            writer.publish_proposal(invented)
        writer.publish_proposal(proposal)
        with pytest.raises(ValueError, match="non-abstained proposal"):
            writer.publish_after_profile(
                _profile_reference(tmp_path / "after.sqlite", "nsys1:after")
            )

    snapshot = store.load("missing-selection")
    assert snapshot.state.phase == "propose"
    assert snapshot.state.before_profile == before
    assert snapshot.findings == report
    assert snapshot.proposal == proposal


def test_proposal_requires_session_profile_and_exactly_one_finding(tmp_path):
    before = _profile_reference(tmp_path / "before.sqlite", "nsys1:before")
    other = _profile_reference(tmp_path / "other.sqlite", "nsys1:other")
    known = _finding(before, "known")
    unknown = _finding(before, "unknown")
    wrong_profile = _finding(other, "known")
    spec = RunSpec(argv=("true",))
    store = SessionStore(tmp_path / "sessions")
    store.create("proposal-pointer", before_profile=before)

    with store.writer("proposal-pointer") as writer:
        writer.publish_runspec(spec)
        with pytest.raises(ValueError, match="requires findings.json"):
            writer.publish_proposal(generate_proposal(known, spec))
        writer.publish_findings(
            EvidenceReport(
                "diagnosis",
                profile_path=before.path,
                profile_id=before.profile_id,
                findings=[known],
            )
        )
        with pytest.raises(ValueError, match="exactly one session finding"):
            writer.publish_proposal(generate_proposal(unknown, spec))
        with pytest.raises(ValueError, match="does not match the before profile"):
            writer.publish_proposal(generate_proposal(wrong_profile, spec))

    assert store.load("proposal-pointer").proposal is None


def test_interrupted_proposal_manifest_update_recovers_previous_proposal(
    monkeypatch, tmp_path
):
    import nsys_ai.session_store as session_store

    before = _profile_reference(tmp_path / "before.sqlite", "nsys1:before")
    first_finding = _finding(before, "f1", action="First action")
    second_finding = _finding(before, "f2", action="Second action")
    report = EvidenceReport(
        "diagnosis",
        profile_path=before.path,
        profile_id=before.profile_id,
        findings=[first_finding, second_finding],
    )
    spec = RunSpec(argv=("true",))
    first = generate_proposal(first_finding, spec)
    second = generate_proposal(second_finding, spec)
    store = SessionStore(tmp_path / "sessions")
    store.create("proposal-interrupted", before_profile=before)

    with store.writer("proposal-interrupted") as writer:
        writer.publish_runspec(spec)
        writer.publish_findings(report)
        writer.publish_proposal(first)
        session_dir = store.root / "proposal-interrupted"
        manifest_path = session_dir / "session.json"
        manifest_before = json.loads(manifest_path.read_text())
        real_atomic_write_json = session_store.atomic_write_json

        def fail_manifest(path, payload):
            if Path(path) == manifest_path:
                raise OSError("simulated proposal manifest failure")
            return real_atomic_write_json(path, payload)

        monkeypatch.setattr(session_store, "atomic_write_json", fail_manifest)
        with pytest.raises(OSError, match="proposal manifest failure"):
            writer.publish_proposal(second)

    proposal_path = store.root / "proposal-interrupted" / "proposal.json"
    assert Proposal.from_json_bytes(proposal_path.read_bytes()) == second
    assert json.loads(manifest_path.read_text()) == manifest_before
    assert manifest_before["phase"] == "propose"
    recovered = SessionStore(store.root).load("proposal-interrupted")
    assert recovered.proposal == first
    assert Proposal.from_json_bytes(proposal_path.read_bytes()) == first


def test_dependent_overwrites_cannot_break_proposal_or_diff_references(tmp_path):
    before = _profile_reference(tmp_path / "before.sqlite", "nsys1:before")
    after = _profile_reference(tmp_path / "after.sqlite", "nsys1:after")
    replacement_after = _profile_reference(
        tmp_path / "replacement.sqlite", "nsys1:replacement"
    )
    finding = _finding(before)
    report = EvidenceReport(
        "diagnosis",
        profile_path=before.path,
        profile_id=before.profile_id,
        findings=[finding],
    )
    proposal = generate_proposal(finding, RunSpec(argv=("true",)))
    store = SessionStore(tmp_path / "sessions")
    store.create("dependencies", before_profile=before)

    with store.writer("dependencies") as writer:
        writer.publish_runspec(proposal.verification)
        writer.publish_findings(report)
        writer.publish_proposal(proposal)
        writer.publish_after_profile(after)
        writer.publish_diff(_diff(before, after))
        with pytest.raises(ValueError, match="cannot publish findings during diff"):
            writer.publish_findings(
                EvidenceReport(
                    "replacement",
                    profile_path=before.path,
                    profile_id=before.profile_id,
                    findings=[],
                )
            )
        with pytest.raises(ValueError, match="cannot publish an after profile during diff"):
            writer.publish_after_profile(replacement_after)
        with pytest.raises(ValueError, match="cannot publish a proposal during diff"):
            writer.publish_proposal(proposal)

    snapshot = store.load("dependencies")
    assert snapshot.findings.to_dict() == report.to_dict()
    assert snapshot.proposal == proposal
    assert snapshot.state.after_profile == after
    assert snapshot.diff["after"]["profile_id"] == after.profile_id


@pytest.mark.parametrize("phase", ["propose", "reprofile", "diff", "accept"])
def test_manifest_cannot_claim_a_phase_without_required_artifacts(tmp_path, phase):
    store = SessionStore(tmp_path / "sessions")
    store.create(f"phase-{phase}")
    manifest_path = store.root / f"phase-{phase}" / "session.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["phase"] = phase
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SessionCorruptError, match=f"{phase} phase requires proposal"):
        store.load(f"phase-{phase}")


def test_manifest_phase_and_diff_decision_must_agree(tmp_path):
    store, before, after, _finding_value, _spec, _proposal = _reprofiled_session(
        tmp_path, "phase-decision"
    )
    with store.writer("phase-decision") as writer:
        writer.publish_diff(_diff(before, after))

    manifest_path = store.root / "phase-decision" / "session.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["phase"] = "accept"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SessionCorruptError, match="accept phase requires a decided"):
        store.load("phase-decision")

    manifest["phase"] = "diff"
    manifest_path.write_text(json.dumps(manifest))
    with store.writer("phase-decision") as writer:
        writer.publish_decision(
            "accept",
            "verified",
            decider="test@example.com",
            decided_at="2026-08-06T00:00:00Z",
        )
    with store.writer("phase-decision") as writer:
        with pytest.raises(ValueError, match="during accept phase"):
            writer.publish_decision("reject", "cannot overwrite")
    decided_manifest = json.loads(manifest_path.read_text())
    decided_manifest["phase"] = "diff"
    manifest_path.write_text(json.dumps(decided_manifest))
    with pytest.raises(SessionCorruptError, match="diff phase requires an undecided"):
        store.load("phase-decision")


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"decider": "   "}, "decider must be a non-empty string"),
        ({"decided_at": "\t"}, "decided_at must be a non-empty string"),
    ],
)
def test_blank_decision_metadata_cannot_corrupt_an_undecided_diff(
    tmp_path, metadata, message
):
    store, before, after, _finding_value, _spec, _proposal = _reprofiled_session(
        tmp_path, "blank-decision-metadata"
    )
    with store.writer("blank-decision-metadata") as writer:
        writer.publish_diff(_diff(before, after))

    session_dir = store.root / "blank-decision-metadata"
    diff_before = (session_dir / "diff.json").read_bytes()
    manifest_before = (session_dir / "session.json").read_bytes()
    with store.writer("blank-decision-metadata") as writer:
        with pytest.raises(ValueError, match=message):
            writer.publish_decision("accept", "verified", **metadata)

    assert (session_dir / "diff.json").read_bytes() == diff_before
    assert (session_dir / "session.json").read_bytes() == manifest_before
    snapshot = store.load("blank-decision-metadata")
    assert snapshot.state.phase == "diff"
    assert snapshot.diff["decision"] is None


@pytest.mark.parametrize("interruption", ["after_artifact", "after_manifest"])
def test_interrupted_decision_publication_recovers_undecided_snapshot(
    monkeypatch, tmp_path, interruption
):
    import nsys_ai.session_store as session_store

    store, before, after, _finding_value, _spec, _proposal = _reprofiled_session(
        tmp_path, "decision-recovery"
    )
    with store.writer("decision-recovery") as writer:
        writer.publish_diff(_diff(before, after))

    session_dir = store.root / "decision-recovery"
    diff_before = (session_dir / "diff.json").read_bytes()
    manifest_before = (session_dir / "session.json").read_bytes()
    if interruption == "after_artifact":
        manifest_path = session_dir / "session.json"
        real_atomic_write_json = session_store.atomic_write_json

        def fail_manifest(path, payload):
            if Path(path) == manifest_path:
                raise OSError("simulated crash after decision artifact")
            return real_atomic_write_json(path, payload)

        monkeypatch.setattr(session_store, "atomic_write_json", fail_manifest)
    else:

        def interrupt_cleanup(_session_id):
            raise OSError("simulated crash after decision manifest")

        monkeypatch.setattr(store, "_finish_transaction", interrupt_cleanup)

    with store.writer("decision-recovery") as writer:
        with pytest.raises(OSError, match="simulated crash"):
            writer.publish_decision(
                "accept",
                "verified",
                decider="test@example.com",
                decided_at="2026-08-06T00:00:00Z",
            )

    recovered = SessionStore(store.root).load("decision-recovery")
    assert recovered.state.phase == "diff"
    assert recovered.diff["decision"] is None
    assert (session_dir / "diff.json").read_bytes() == diff_before
    assert (session_dir / "session.json").read_bytes() == manifest_before
    assert not (session_dir / ".transaction").exists()


def test_manifest_cannot_regress_while_retaining_downstream_artifacts(tmp_path):
    store, before, after, _finding_value, _spec, _proposal = _reprofiled_session(
        tmp_path, "phase-regression"
    )
    manifest_path = store.root / "phase-regression" / "session.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["phase"] = "diagnose"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SessionCorruptError, match="diagnose phase cannot retain"):
        store.load("phase-regression")

    manifest["phase"] = "reprofile"
    manifest_path.write_text(json.dumps(manifest))
    with store.writer("phase-regression") as writer:
        writer.publish_diff(_diff(before, after))
    manifest = json.loads(manifest_path.read_text())
    manifest["phase"] = "reprofile"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SessionCorruptError, match="reprofile phase cannot retain"):
        store.load("phase-regression")


def test_publish_diff_rejects_prepopulated_decision(tmp_path):
    store, before, after, _finding_value, _spec, _proposal = _reprofiled_session(
        tmp_path, "decided-input"
    )
    payload = _diff(before, after)
    payload["decision"] = {
        "status": "accepted",
        "reason": "bypass",
        "decider": "test@example.com",
        "decided_at": "2026-08-06T00:00:00Z",
    }
    with store.writer("decided-input") as writer:
        with pytest.raises(ValueError, match="requires decision to be null"):
            writer.publish_diff(payload)
    assert store.load("decided-input").state.phase == "reprofile"


@pytest.mark.parametrize(
    "decision",
    [
        {
            "status": "maybe",
            "reason": "reason",
            "decider": "test@example.com",
            "decided_at": "2026-08-06T00:00:00Z",
        },
        {
            "status": "accepted",
            "reason": "   ",
            "decider": "test@example.com",
            "decided_at": "2026-08-06T00:00:00Z",
        },
        {
            "status": "rejected",
            "reason": "reason",
            "decider": "test@example.com",
            "decided_at": "2026-08-06T00:00:00Z",
            "extra": "not canonical",
        },
    ],
)
def test_load_rejects_noncanonical_diff_decisions(tmp_path, decision):
    store, before, after, _finding_value, _spec, _proposal = _reprofiled_session(
        tmp_path, "bad-decision"
    )
    with store.writer("bad-decision") as writer:
        writer.publish_diff(_diff(before, after))
    session_dir = store.root / "bad-decision"
    diff_path = session_dir / "diff.json"
    payload = json.loads(diff_path.read_text())
    payload["decision"] = decision
    diff_path.write_text(json.dumps(payload))
    manifest_path = session_dir / "session.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["phase"] = "accept"
    manifest["artifacts"]["diff"]["sha256"] = hashlib.sha256(
        diff_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SessionCorruptError, match="invalid diff.json"):
        store.load("bad-decision")


def test_runspec_and_proposal_verification_cannot_diverge(tmp_path):
    before = _profile_reference(tmp_path / "before.sqlite", "nsys1:before")
    finding = _finding(before)
    first_spec = RunSpec(argv=("true", "--mode=first"))
    second_spec = RunSpec(argv=("true", "--mode=second"))
    report = EvidenceReport(
        "diagnosis",
        profile_path=before.path,
        profile_id=before.profile_id,
        findings=[finding],
    )
    store = SessionStore(tmp_path / "sessions")
    store.create("runspec-consistency", before_profile=before)
    with store.writer("runspec-consistency") as writer:
        writer.publish_runspec(first_spec)
        writer.publish_findings(report)
        with pytest.raises(ValueError, match="must match runspec.json"):
            writer.publish_proposal(generate_proposal(finding, second_spec))
        first_proposal = generate_proposal(finding, first_spec)
        writer.publish_proposal(first_proposal)
        with pytest.raises(ValueError, match="cannot change after proposal"):
            writer.publish_runspec(second_spec)
        writer.publish_runspec(first_spec)

    session_dir = store.root / "runspec-consistency"
    runspec_path = session_dir / "runspec.json"
    runspec_path.write_bytes(second_spec.canonical_json_bytes())
    manifest_path = session_dir / "session.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["runspec"]["sha256"] = hashlib.sha256(
        runspec_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SessionCorruptError, match="does not match runspec.json"):
        store.load("runspec-consistency")


def test_null_verification_abstention_requires_null_session_runspec(tmp_path):
    before = _profile_reference(tmp_path / "before.sqlite", "nsys1:before")
    finding = _finding(before)
    report = EvidenceReport(
        "diagnosis",
        profile_path=before.path,
        profile_id=before.profile_id,
        findings=[finding],
    )
    proposal = generate_proposal(finding, None)
    assert proposal.abstained
    store = SessionStore(tmp_path / "sessions")
    store.create("null-verification", before_profile=before)
    with store.writer("null-verification") as writer:
        writer.publish_findings(report)
        writer.publish_proposal(proposal)
        with pytest.raises(ValueError, match="cannot change after proposal"):
            writer.publish_runspec(RunSpec(argv=("true",)))
        with pytest.raises(ValueError, match="non-abstained proposal"):
            writer.publish_after_profile(
                _profile_reference(tmp_path / "after.sqlite", "nsys1:after")
            )

    snapshot = store.load("null-verification")
    assert snapshot.runspec is None
    assert snapshot.proposal == proposal
    assert snapshot.state.phase == "propose"


def test_non_finite_json_values_are_rejected_before_publication(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("non-finite")
    finding = Finding(
        type="region",
        label="bad value",
        start_ns=1,
        provenance={"measurement": float("nan")},
    )
    report = EvidenceReport("non-finite", findings=[finding])
    with store.writer("non-finite") as writer:
        with pytest.raises(TypeError, match="JSON serializable"):
            writer.publish_findings(report)

    session_dir = store.root / "non-finite"
    assert not (session_dir / "findings.json").exists()
    assert store.load("non-finite").state.phase == "diagnose"

    reprofiled, before, after, _finding_value, _spec, _proposal = _reprofiled_session(
        tmp_path, "non-finite-diff"
    )
    payload = _diff(before, after)
    payload["comparability_confidence"] = float("inf")
    with reprofiled.writer("non-finite-diff") as writer:
        with pytest.raises(TypeError, match="JSON serializable"):
            writer.publish_diff(payload)
    assert not (reprofiled.root / "non-finite-diff" / "diff.json").exists()


def test_closed_writer_rejects_every_publication_method(tmp_path):
    before = _profile_reference(tmp_path / "before.sqlite", "nsys1:before")
    after = _profile_reference(tmp_path / "after.sqlite", "nsys1:after")
    store = SessionStore(tmp_path / "sessions")
    store.create("closed", before_profile=before)
    writer = store.writer("closed")
    writer.close()

    actions = (
        lambda: writer.publish_runspec(RunSpec(argv=("true",))),
        lambda: writer.publish_findings(EvidenceReport("report")),
        lambda: writer.publish_after_profile(after),
        lambda: writer.publish_proposal(
            generate_proposal(_finding(before), RunSpec(argv=("true",)))
        ),
        lambda: writer.publish_diff(_diff(before, after)),
        lambda: writer.publish_decision("accept", "reason"),
    )
    for action in actions:
        with pytest.raises(SessionError, match="writer is closed"):
            action()


def test_malformed_or_unsupported_json_is_rejected_as_corrupt(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create("broken")
    manifest_path = store.root / "broken" / "session.json"
    manifest_path.write_text('{"schema_version":')
    with pytest.raises(SessionCorruptError, match="invalid session.json"):
        store.load("broken")

    store.create("future")
    future_path = store.root / "future" / "session.json"
    future = json.loads(future_path.read_text())
    future["schema_version"] = "99"
    future_path.write_text(json.dumps(future))
    with pytest.raises(UnsupportedSessionVersionError, match="unsupported session"):
        store.load("future")


def test_profile_reference_rejects_remote_kind_and_is_not_revalidated_on_restart(tmp_path):
    profile = _profile_reference(tmp_path / "profile.sqlite", "nsys1:profile")
    store = SessionStore(tmp_path / "sessions")
    store.create("local", before_profile=profile)
    Path(profile.path).unlink()
    assert store.load("local").state.before_profile == profile

    manifest_path = store.root / "local" / "session.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["profiles"]["before"]["kind"] = "remote"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SessionCorruptError, match="kind must be 'local'"):
        store.load("local")


def _subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1] / "src")
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not current else f"{source}{os.pathsep}{current}"
    return environment
