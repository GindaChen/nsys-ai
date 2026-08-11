import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import nsys_ai.propose_command as propose_command
from nsys_ai.annotation import EvidenceReport, Finding, TraceSelection
from nsys_ai.proposal import Proposal
from nsys_ai.runspec import EnvironmentSpec, RunSpec
from nsys_ai.session_store import SessionStore

ROOT = Path(__file__).resolve().parents[1]


def _finding(finding_id: str, *, explanation: str | None = None) -> Finding:
    return Finding(
        type="region",
        label="Exposed communication",
        start_ns=100,
        end_ns=200,
        id=finding_id,
        confidence=0.84,
        explanation=explanation or "Communication is exposed in the backward pass.",
        suggested_actions=["Test moving the all-reduce earlier."],
        false_positive_notes=["Confirm the workload is representative."],
        headroom_ms=12.5,
        headroom_basis="capture_total",
        selection=TraceSelection(
            id=f"selection-{finding_id}",
            profile_id="nsys2:sha256:" + "1" * 64,
            source="skill:overlap_breakdown",
            start_ns=100,
            end_ns=200,
            gpu_ids=[0],
            nvtx_path=["iteration", "backward"],
        ),
    )


def _write_evidence(path: Path, findings: list[Finding]) -> dict:
    payload = EvidenceReport(
        "Auto-Analysis",
        profile_path="/profiles/before.sqlite",
        profile_id="nsys2:sha256:" + "1" * 64,
        findings=findings,
    ).to_dict()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _write_runspec(
    path: Path, *, secrets: tuple[str, ...] = (), argv: tuple[str, ...] = ("python", "train.py")
) -> RunSpec:
    spec = RunSpec(
        argv=argv,
        environment=EnvironmentSpec(policy="clean", secrets=secrets),
    )
    path.write_bytes(spec.canonical_json_bytes())
    return spec


def _run_cli(
    cwd: Path,
    *args: str,
    environment: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT / "src"), env.get("PYTHONPATH", "")) if part
    )
    if environment is not None:
        env.update(environment)
    return subprocess.run(
        [sys.executable, "-m", "nsys_ai", "propose", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_cli_generates_round_trippable_proposal_selected_only_by_id(tmp_path):
    findings = tmp_path / "findings.json"
    runspec = tmp_path / "runspec.json"
    output = tmp_path / "artifacts" / "proposal.json"
    _write_evidence(findings, [_finding("other"), _finding("target")])
    expected_spec = _write_runspec(runspec)

    result = _run_cli(
        tmp_path,
        str(findings),
        "--finding-id",
        "target",
        "--runspec",
        str(runspec),
        "-o",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    proposal = Proposal.from_json_bytes(output.read_bytes())
    assert proposal.source_finding_id == "target"
    assert proposal.verification == expected_spec
    assert not proposal.abstained
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert str(output) in result.stdout
    assert result.stderr == ""


def test_missing_runspec_writes_formal_abstention_and_exits_zero(tmp_path):
    findings = tmp_path / "findings.json"
    _write_evidence(findings, [_finding("target")])

    result = _run_cli(tmp_path, str(findings), "--finding-id", "target")

    assert result.returncode == 0, result.stderr
    proposal = Proposal.from_json_bytes((tmp_path / "proposal.json").read_bytes())
    assert proposal.abstained
    assert proposal.verification is None
    assert "verification RunSpec is required" in proposal.abstention_reason
    assert "Abstained:" in result.stdout


@pytest.mark.parametrize(
    ("findings", "finding_id", "message"),
    [
        ([_finding("known")], "unknown", "not found"),
        ([_finding("duplicate"), _finding("duplicate")], "duplicate", "duplicated"),
    ],
)
def test_unknown_and_duplicate_finding_ids_are_clear_input_errors(
    tmp_path, findings, finding_id, message
):
    source = tmp_path / "findings.json"
    _write_evidence(source, findings)

    result = _run_cli(tmp_path, str(source), "--finding-id", finding_id)

    assert result.returncode == 1
    assert message in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "proposal.json").exists()


@pytest.mark.parametrize("mutation", ["legacy", "unknown", "nested"])
def test_evidence_input_uses_strict_current_artifact_validation(tmp_path, mutation):
    source = tmp_path / "findings.json"
    payload = _write_evidence(source, [_finding("target")])
    if mutation == "legacy":
        del payload["schema_version"]
    elif mutation == "unknown":
        payload["extra"] = True
    else:
        payload["findings"][0]["selection"]["start_ns"] = "later"
    source.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "proposal.json"
    output.write_text("existing", encoding="utf-8")

    result = _run_cli(tmp_path, str(source), "--finding-id", "target")

    assert result.returncode == 1
    assert "invalid evidence report" in result.stderr
    assert output.read_text(encoding="utf-8") == "existing"


def test_success_atomically_overwrites_an_existing_ordinary_artifact(tmp_path):
    source = tmp_path / "findings.json"
    output = tmp_path / "proposal.json"
    _write_evidence(source, [_finding("target")])
    output.write_text("stale", encoding="utf-8")

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "-o",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    assert Proposal.from_json_bytes(output.read_bytes()).source_finding_id == "target"
    assert not list(tmp_path.glob(".proposal.json.*.tmp"))


def test_declared_secret_is_preflight_only_and_never_persisted_or_printed(tmp_path):
    secret = "sentinel-propose-secret"
    source = tmp_path / "findings.json"
    runspec = tmp_path / "runspec.json"
    output = tmp_path / "proposal.json"
    _write_evidence(source, [_finding("target")])
    _write_runspec(runspec, secrets=("PROPOSE_SECRET",))

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "--runspec",
        str(runspec),
        "-o",
        str(output),
        environment={"PROPOSE_SECRET": secret},
    )

    assert result.returncode == 0, result.stderr
    artifact = output.read_text(encoding="utf-8")
    assert secret not in artifact
    assert secret not in result.stdout
    assert secret not in result.stderr
    proposal = Proposal.from_json_bytes(artifact)
    assert proposal.verification.environment.secrets == ("PROPOSE_SECRET",)
    assert "secret_environment_values_unresolved" in proposal.limitations


def test_secret_in_projected_finding_leaves_existing_output_unchanged(tmp_path):
    secret = "sentinel-projected-secret"
    source = tmp_path / "findings.json"
    runspec = tmp_path / "runspec.json"
    output = tmp_path / "proposal.json"
    _write_evidence(source, [_finding("target", explanation=f"Do not persist {secret}")])
    _write_runspec(runspec, secrets=("PROPOSE_SECRET",))
    output.write_text("existing", encoding="utf-8")

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "--runspec",
        str(runspec),
        "-o",
        str(output),
        environment={"PROPOSE_SECRET": secret},
    )

    assert result.returncode == 1
    assert "persisted string" in result.stderr
    assert secret not in result.stderr
    assert secret not in result.stdout
    assert output.read_text(encoding="utf-8") == "existing"


def test_strict_evidence_error_redacts_overlapping_declared_secrets(tmp_path):
    short_secret = "sentinel-overlap"
    long_secret = f"{short_secret}-longer"
    source = tmp_path / "findings.json"
    runspec = tmp_path / "runspec.json"
    payload = _write_evidence(source, [_finding("target")])
    payload[long_secret] = "unknown field"
    source.write_text(json.dumps(payload), encoding="utf-8")
    _write_runspec(runspec, secrets=("A_SHORT_SECRET", "Z_LONG_SECRET"))

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "--runspec",
        str(runspec),
        environment={
            "A_SHORT_SECRET": short_secret,
            "Z_LONG_SECRET": long_secret,
        },
    )

    assert result.returncode == 1
    assert "invalid evidence report" in result.stderr
    assert short_secret not in result.stderr
    assert long_secret not in result.stderr
    assert "-longer" not in result.stderr
    assert not (tmp_path / "proposal.json").exists()


def test_missing_declared_secret_is_an_input_error_with_no_output(tmp_path, monkeypatch):
    source = tmp_path / "findings.json"
    runspec = tmp_path / "runspec.json"
    _write_evidence(source, [_finding("target")])
    _write_runspec(runspec, secrets=("DELIBERATELY_UNSET_PROPOSE_SECRET",))

    monkeypatch.delenv("DELIBERATELY_UNSET_PROPOSE_SECRET", raising=False)
    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "--runspec",
        str(runspec),
    )

    assert result.returncode == 1
    assert "declared secret DELIBERATELY_UNSET_PROPOSE_SECRET is not set" in result.stderr
    assert not (tmp_path / "proposal.json").exists()


def test_malformed_runspec_cannot_reflect_a_declared_secret(tmp_path):
    secret = "sentinel-malformed-runspec-secret"
    source = tmp_path / "findings.json"
    runspec = tmp_path / "runspec.json"
    output = tmp_path / "proposal.json"
    _write_evidence(source, [_finding("target")])
    payload = _write_runspec(runspec, secrets=("PROPOSE_SECRET",)).to_dict()
    payload[secret] = "unknown field"
    runspec.write_text(json.dumps(payload), encoding="utf-8")
    output.write_text("existing", encoding="utf-8")

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "--runspec",
        str(runspec),
        environment={"PROPOSE_SECRET": secret},
    )

    assert result.returncode == 1
    assert "invalid RunSpec artifact" in result.stderr
    assert secret not in result.stderr
    assert secret not in result.stdout
    assert output.read_text(encoding="utf-8") == "existing"


def test_secret_in_output_path_is_rejected_without_echo(tmp_path):
    secret = "sentinel-path-secret"
    source = tmp_path / "findings.json"
    runspec = tmp_path / "runspec.json"
    output = tmp_path / secret / "proposal.json"
    _write_evidence(source, [_finding("target")])
    _write_runspec(runspec, secrets=("PROPOSE_SECRET",))

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "--runspec",
        str(runspec),
        "-o",
        str(output),
        environment={"PROPOSE_SECRET": secret},
    )

    assert result.returncode == 1
    assert "persisted string" in result.stderr
    assert secret not in result.stderr
    assert secret not in result.stdout
    assert not output.exists()


def test_session_named_directory_without_manifest_is_an_ordinary_output(tmp_path):
    source = tmp_path / "findings.json"
    output = tmp_path / ".nsys-ai" / "sessions" / "run-001" / "proposal.json"
    _write_evidence(source, [_finding("target")])

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "-o",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    assert Proposal.from_json_bytes(output.read_bytes()).source_finding_id == "target"


def test_unrelated_session_json_is_an_ordinary_output_ancestor(tmp_path):
    source = tmp_path / "findings.json"
    output_directory = tmp_path / "web-state"
    output_directory.mkdir()
    manifest = output_directory / "session.json"
    ordinary_payload = {
        "schema_version": "web-v1",
        "profiles": {"theme": "dark"},
        "artifacts": {"bundle": "app.js"},
        "route": "/review",
    }
    manifest.write_text(json.dumps(ordinary_payload), encoding="utf-8")
    manifest_before = manifest.read_bytes()
    output = output_directory / "proposal.json"
    _write_evidence(source, [_finding("target")])

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "-o",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    assert Proposal.from_json_bytes(output.read_bytes()).source_finding_id == "target"
    assert manifest.read_bytes() == manifest_before


@pytest.mark.parametrize("manifest_kind", ["empty", "object", "unrelated"])
def test_true_logs_layout_rejects_unrecognized_session_manifest(
    tmp_path, manifest_kind
):
    source = tmp_path / "findings.json"
    session_root = tmp_path / "session-root"
    SessionStore(session_root).create("run-001")
    session = session_root / "run-001"
    manifest = session / "session.json"
    if manifest_kind == "empty":
        encoded = b""
    elif manifest_kind == "object":
        encoded = b"{}"
    else:
        encoded = json.dumps(
            {
                "schema_version": "web-v1",
                "profiles": {"theme": "dark"},
                "artifacts": {"bundle": "app.js"},
                "route": "/review",
            }
        ).encode("utf-8")
    manifest.write_bytes(encoded)
    output = session / "new" / "proposal.json"
    _write_evidence(source, [_finding("target")])

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "-o",
        str(output),
    )

    assert result.returncode == 1
    assert "session manifest is invalid" in result.stderr
    assert manifest.read_bytes() == encoded
    assert not output.parent.exists()


def test_early_truncated_sessionstate_is_rejected_without_logs_evidence(tmp_path):
    source = tmp_path / "findings.json"
    session_root = tmp_path / "source-session-root"
    SessionStore(session_root).create("run-001")
    valid = (session_root / "run-001" / "session.json").read_bytes()
    truncated = valid[: valid.index(b'"profiles"')]
    output_directory = tmp_path / "no-logs-partial-session"
    output_directory.mkdir()
    manifest = output_directory / "session.json"
    manifest.write_bytes(truncated)
    output = output_directory / "proposal.json"
    _write_evidence(source, [_finding("target")])

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "-o",
        str(output),
    )

    assert result.returncode == 1
    assert "session manifest is invalid" in result.stderr
    assert manifest.read_bytes() == truncated
    assert not output.exists()


@pytest.mark.parametrize("missing_key", ["schema_version", "profiles", "artifacts"])
def test_partial_sessionstate_object_is_rejected_without_logs_evidence(
    tmp_path, missing_key
):
    source = tmp_path / "findings.json"
    session_root = tmp_path / "source-session-root"
    SessionStore(session_root).create("run-001")
    payload = json.loads((session_root / "run-001" / "session.json").read_bytes())
    del payload[missing_key]
    output_directory = tmp_path / f"missing-{missing_key}"
    output_directory.mkdir()
    manifest = output_directory / "session.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    manifest_before = manifest.read_bytes()
    output = output_directory / "proposal.json"
    _write_evidence(source, [_finding("target")])

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "-o",
        str(output),
    )

    assert result.returncode == 1
    assert "session manifest is invalid" in result.stderr
    assert manifest.read_bytes() == manifest_before
    assert not output.exists()


@pytest.mark.parametrize("logs_kind", ["symlink", "fifo"])
def test_non_directory_logs_entry_is_not_followed_or_used_as_layout_evidence(
    tmp_path, logs_kind
):
    source = tmp_path / "findings.json"
    output_directory = tmp_path / f"ordinary-{logs_kind}-logs"
    output_directory.mkdir()
    manifest = output_directory / "session.json"
    ordinary_payload = {"schema_version": "web-v1", "route": "/review"}
    manifest.write_text(json.dumps(ordinary_payload), encoding="utf-8")
    logs = output_directory / "logs"
    target = tmp_path / "logs-target"
    if logs_kind == "symlink":
        target.mkdir()
        sentinel = target / "sentinel"
        sentinel.write_text("unchanged", encoding="utf-8")
        logs.symlink_to(target, target_is_directory=True)
    else:
        os.mkfifo(logs)
        sentinel = None
    output = output_directory / "proposal.json"
    _write_evidence(source, [_finding("target")])

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "-o",
        str(output),
        timeout=2.0,
    )

    assert result.returncode == 0, result.stderr
    assert Proposal.from_json_bytes(output.read_bytes()).source_finding_id == "target"
    if sentinel is not None:
        assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_fifo_session_manifest_fails_without_blocking(tmp_path):
    source = tmp_path / "findings.json"
    output_directory = tmp_path / "fifo-ancestor"
    output_directory.mkdir()
    manifest = output_directory / "session.json"
    os.mkfifo(manifest)
    output = output_directory / "proposal.json"
    _write_evidence(source, [_finding("target")])

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "-o",
        str(output),
        timeout=2.0,
    )

    assert result.returncode == 1
    assert "session manifest is not a regular file" in result.stderr
    assert not output.exists()


def test_symlink_session_manifest_is_not_followed(tmp_path):
    source = tmp_path / "findings.json"
    state_root = tmp_path / "state-source"
    SessionStore(state_root).create("run-001")
    target = state_root / "run-001" / "session.json"
    target_before = target.read_bytes()
    output_directory = tmp_path / "symlink-manifest-ancestor"
    output_directory.mkdir()
    (output_directory / "session.json").symlink_to(target)
    output = output_directory / "proposal.json"
    _write_evidence(source, [_finding("target")])

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "-o",
        str(output),
    )

    assert result.returncode == 1
    assert "could not inspect session manifest" in result.stderr
    assert target.read_bytes() == target_before
    assert not output.exists()


def test_oversized_recognizable_manifest_is_rejected_before_read(
    tmp_path, monkeypatch
):
    source = tmp_path / "findings.json"
    output_directory = tmp_path / "oversized-session"
    output_directory.mkdir()
    manifest = output_directory / "session.json"
    prefix = (
        b'{"artifacts":{},"phase":"diagnose","profiles":{},'
        b'"schema_version":"0.1","session_id":"oversized","padding":"'
    )
    manifest.write_bytes(
        prefix + b"x" * (propose_command._SESSION_MANIFEST_MAX_BYTES * 2) + b'"}'
    )
    manifest_before = manifest.stat().st_size
    output = output_directory / "proposal.json"
    _write_evidence(source, [_finding("target")])
    real_read = propose_command.os.read
    requested: list[int] = []
    returned = 0

    def recording_read(descriptor, size):
        nonlocal returned
        requested.append(size)
        chunk = real_read(descriptor, size)
        returned += len(chunk)
        return chunk

    monkeypatch.setattr(propose_command.os, "read", recording_read)
    args = SimpleNamespace(
        findings=str(source),
        finding_id="target",
        runspec=None,
        output=str(output),
    )

    with pytest.raises(propose_command.ProposeCommandError, match="safe inspection size"):
        propose_command.run_propose_command(args, stdout=io.StringIO(), environment={})

    assert not requested
    assert returned == 0
    assert manifest.stat().st_size == manifest_before
    assert not output.exists()


def test_alternate_root_session_output_is_rejected_without_mutation(tmp_path):
    source = tmp_path / "findings.json"
    session_root = tmp_path / "custom-session-root"
    SessionStore(session_root).create("run-001")
    session = session_root / "run-001"
    manifest = session / "session.json"
    manifest_before = manifest.read_bytes()
    output = session / "proposal.json"
    _write_evidence(source, [_finding("target")])

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "-o",
        str(output),
    )

    assert result.returncode == 1
    assert "SessionWriter" in result.stderr
    assert manifest.read_bytes() == manifest_before
    assert not output.exists()


def test_symlink_to_session_directory_is_rejected_without_mutation(tmp_path):
    source = tmp_path / "findings.json"
    session_root = tmp_path / "alternate-session-root"
    SessionStore(session_root).create("run-001")
    session = session_root / "run-001"
    manifest = session / "session.json"
    manifest_before = manifest.read_bytes()
    alias = tmp_path / "ordinary-looking-output"
    alias.symlink_to(session, target_is_directory=True)
    output = alias / "proposal.json"
    _write_evidence(source, [_finding("target")])

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "-o",
        str(output),
    )

    assert result.returncode == 1
    assert "SessionWriter" in result.stderr
    assert manifest.read_bytes() == manifest_before
    assert not (session / "proposal.json").exists()


def test_corrupt_alternate_root_session_is_rejected_before_creating_parent(tmp_path):
    source = tmp_path / "findings.json"
    session_root = tmp_path / "alternate-session-root"
    SessionStore(session_root).create("run-001")
    session = session_root / "run-001"
    manifest = session / "session.json"
    corrupt_manifest = manifest.read_bytes().rstrip()[:-1]
    manifest.write_bytes(corrupt_manifest)
    output = session / "new" / "proposal.json"
    _write_evidence(source, [_finding("target")])

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "-o",
        str(output),
    )

    assert result.returncode == 1
    assert "session manifest is invalid" in result.stderr
    assert str(session) not in result.stderr
    assert "Traceback" not in result.stderr
    assert manifest.read_bytes() == corrupt_manifest
    assert not output.parent.exists()


@pytest.mark.parametrize("mutation", ["corrupt", "unsupported"])
def test_recognizable_invalid_session_manifest_fails_closed(tmp_path, mutation):
    source = tmp_path / "findings.json"
    session_root = tmp_path / "recognizable-invalid-root"
    SessionStore(session_root).create("run-001")
    session = session_root / "run-001"
    manifest = session / "session.json"
    payload = json.loads(manifest.read_bytes())
    if mutation == "corrupt":
        payload["profiles"] = "not-an-object"
    else:
        payload["schema_version"] = "999"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    manifest_before = manifest.read_bytes()
    output = session / "new" / "proposal.json"
    _write_evidence(source, [_finding("target")])

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "-o",
        str(output),
    )

    assert result.returncode == 1
    assert "session manifest is invalid" in result.stderr
    assert manifest.read_bytes() == manifest_before
    assert not output.parent.exists()


@pytest.mark.parametrize("source_name", ["findings", "runspec"])
@pytest.mark.parametrize("alias_kind", ["exact", "relative", "symlink", "hardlink"])
def test_output_alias_of_input_is_rejected_before_write(
    tmp_path, source_name, alias_kind
):
    findings = tmp_path / "findings.json"
    runspec = tmp_path / "runspec.json"
    _write_evidence(findings, [_finding("target")])
    _write_runspec(runspec)
    source = findings if source_name == "findings" else runspec

    if alias_kind == "exact":
        output_arg = str(source)
    elif alias_kind == "relative":
        output_arg = source.name
    else:
        output = tmp_path / f"{source_name}-{alias_kind}.json"
        if alias_kind == "symlink":
            output.symlink_to(source)
        else:
            os.link(source, output)
        output_arg = str(output)

    findings_before = findings.read_bytes()
    runspec_before = runspec.read_bytes()
    result = _run_cli(
        tmp_path,
        str(findings),
        "--finding-id",
        "target",
        "--runspec",
        str(runspec),
        "-o",
        output_arg,
    )

    assert result.returncode == 1
    assert "must not alias an input artifact" in result.stderr
    assert "Traceback" not in result.stderr
    assert findings.read_bytes() == findings_before
    assert runspec.read_bytes() == runspec_before
    assert not list(tmp_path.glob(".*.tmp"))


def test_symlink_retarget_after_validation_writes_only_to_bound_directory(
    tmp_path, monkeypatch
):
    findings = tmp_path / "findings.json"
    _write_evidence(findings, [_finding("target")])
    safe = tmp_path / "ordinary-output"
    safe.mkdir()
    session_root = tmp_path / "alternate-session-root"
    SessionStore(session_root).create("run-001")
    session = session_root / "run-001"
    manifest = session / "session.json"
    manifest_before = manifest.read_bytes()
    alias = tmp_path / "output-alias"
    alias.symlink_to(safe, target_is_directory=True)
    real_atomic_write = propose_command.atomic_write_bytes_at

    def retarget_then_write(directory_fd, name, payload, *, mode=0o600):
        alias.unlink()
        alias.symlink_to(session, target_is_directory=True)
        real_atomic_write(directory_fd, name, payload, mode=mode)

    monkeypatch.setattr(
        propose_command,
        "atomic_write_bytes_at",
        retarget_then_write,
    )
    stdout = io.StringIO()
    args = SimpleNamespace(
        findings=str(findings),
        finding_id="target",
        runspec=None,
        output=str(alias / "proposal.json"),
    )

    result = propose_command.run_propose_command(args, stdout=stdout, environment={})

    assert result == 0
    assert Proposal.from_json_bytes(
        (safe / "proposal.json").read_bytes()
    ).source_finding_id == "target"
    assert not (session / "proposal.json").exists()
    assert manifest.read_bytes() == manifest_before


def test_output_io_failure_is_clean_and_preserves_the_directory(tmp_path):
    source = tmp_path / "findings.json"
    output = tmp_path / "proposal.json"
    _write_evidence(source, [_finding("target")])
    output.mkdir()

    result = _run_cli(
        tmp_path,
        str(source),
        "--finding-id",
        "target",
        "-o",
        str(output),
    )

    assert result.returncode == 1
    assert "could not write proposal artifact" in result.stderr
    assert "Traceback" not in result.stderr
    assert output.is_dir()


# ── the supplied RunSpec against the capture's own ──────────────────────


def _capture_with_runspec(directory: Path, argv: tuple[str, ...] | None) -> Path:
    """A profile path, with the runspec `nsys-ai profile` writes beside it."""
    directory.mkdir(parents=True, exist_ok=True)
    profile = directory / "capture.sqlite"
    profile.write_bytes(b"")
    if argv is not None:
        _write_runspec(directory / "runspec.json", argv=argv)
    return profile


def _evidence_for(path: Path, capture: Path) -> None:
    """An evidence report naming *capture* as the profile it came from.

    `--profile` is session-only, so in file mode the report is the only thing
    that says which capture these findings describe.
    """
    payload = EvidenceReport(
        "Auto-Analysis",
        profile_path=str(capture),
        profile_id="nsys2:sha256:" + "1" * 64,
        findings=[_finding("f1")],
    ).to_dict()
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_a_runspec_unrelated_to_the_capture_is_said_out_loud(tmp_path, capsys):
    """The proposal records the RunSpec as the way the change will be verified.

    A RunSpec from an entirely different program was accepted and written in as
    that verification path, with nothing comparing it to anything. The capture's
    own runspec.json sits beside it and was never read.
    """
    profile = _capture_with_runspec(tmp_path / "capture", ("python", "train.py"))
    findings = tmp_path / "findings.json"
    _evidence_for(findings, profile)
    unrelated = tmp_path / "unrelated.json"
    _write_runspec(unrelated, argv=("./bench_axpy", "--n", "1024"))

    code = propose_command.run_propose(
        finding_id="f1",
        findings_path=str(findings),
        runspec_path=str(unrelated),
        output=str(tmp_path / "proposal.json"),
        stdout=io.StringIO(),
    )

    assert code == 0
    stderr = capsys.readouterr().err
    assert "nothing in common with the capture" in stderr, stderr
    assert "./bench_axpy --n 1024" in stderr
    assert "python train.py" in stderr


def test_a_runspec_that_matches_the_capture_says_nothing(tmp_path, capsys):
    """Advisory means quiet when there is nothing to advise."""
    profile = _capture_with_runspec(tmp_path / "capture", ("python", "train.py"))
    findings = tmp_path / "findings.json"
    _evidence_for(findings, profile)
    same = tmp_path / "same.json"
    _write_runspec(same, argv=("python", "train.py"))

    propose_command.run_propose(
        finding_id="f1",
        findings_path=str(findings),
        runspec_path=str(same),
        output=str(tmp_path / "proposal.json"),
        stdout=io.StringIO(),
    )

    assert "nothing in common" not in capsys.readouterr().err


def test_a_narrower_harness_sharing_a_target_says_nothing(tmp_path, capsys):
    """Verifying with a tighter harness than the capture ran is legitimate."""
    profile = _capture_with_runspec(tmp_path / "capture", ("python", "train.py"))
    findings = tmp_path / "findings.json"
    _evidence_for(findings, profile)
    narrow = tmp_path / "narrow.json"
    _write_runspec(narrow, argv=("pytest", "train.py", "-k", "step"))

    propose_command.run_propose(
        finding_id="f1",
        findings_path=str(findings),
        runspec_path=str(narrow),
        output=str(tmp_path / "proposal.json"),
        stdout=io.StringIO(),
    )

    assert "nothing in common" not in capsys.readouterr().err


def test_no_runspec_beside_the_capture_means_no_claim(tmp_path, capsys):
    """Most captures have no sibling; absence must not read as a mismatch."""
    profile = _capture_with_runspec(tmp_path / "bare", None)
    findings = tmp_path / "findings.json"
    _evidence_for(findings, profile)
    spec = tmp_path / "spec.json"
    _write_runspec(spec, argv=("./bench_axpy",))

    propose_command.run_propose(
        finding_id="f1",
        findings_path=str(findings),
        runspec_path=str(spec),
        output=str(tmp_path / "proposal.json"),
        stdout=io.StringIO(),
    )

    assert "nothing in common" not in capsys.readouterr().err
