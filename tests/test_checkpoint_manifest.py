from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nsys_ai.checkpoint import (
    CHECKPOINT_MANIFEST_SCHEMA,
    CheckpointManifestError,
    canonical_manifest_bytes,
    expand_command,
    load_manifest,
    run_steps,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "examples/checkpoints/b0-contract/manifest.json"


def _manifest(*, profile_path: str = "profile.sqlite", sha256: str | None = None) -> dict:
    return {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
        "checkpoint": "B0-test",
        "project": {
            "name": "test-project",
            "repository": "https://example.test/project",
            "revision_kind": "git_commit",
            "revision": "0123456789abcdef",
        },
        "workload": {
            "name": "small-gpu-workload",
            "artifact": "model@0123456",
            "parameters": {"batch": 1},
            "capture_command": ["nsys", "profile", "workload"],
            "expected_signals": [
                {
                    "id": "kernel-activity",
                    "description": "The workload launches a CUDA kernel.",
                    "verification": "Check the kernel table independently.",
                }
            ],
        },
        "environment": {
            "python": "3.12.1",
            "cuda": "12.8",
            "driver": "580.95",
            "gpu": "H100",
            "nsys": "2026.1",
        },
        "capture": {
            "profile_path": profile_path,
            "format": "sqlite",
            "sha256": sha256 or "a" * 64,
            "output_paths": [profile_path],
            "captured_at": "2026-08-23T00:00:00Z",
        },
        "analysis": {
            "session_dir": ".nsys-ai/checkpoints/test",
            "steps": [
                {
                    "name": name,
                    "command": [sys.executable, "-c", "print('ok')"],
                    "expected_exit_codes": [0],
                }
                for name in ("doctor", "diagnose", "ask", "diff", "review")
            ],
        },
    }


def test_committed_contract_manifest_validates_and_matches_fixture():
    manifest = load_manifest(
        MANIFEST,
        profile_root=ROOT,
        require_profile=True,
    )
    assert manifest["checkpoint"] == "B0-contract"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("project.revision", "main"),
        ("environment.driver", "unknown"),
        ("capture.profile_path", "/tmp/profile.sqlite"),
    ],
)
def test_validation_rejects_unpinned_or_machine_local_metadata(path, value):
    manifest = _manifest()
    target = manifest
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value

    with pytest.raises(CheckpointManifestError, match=path.replace(".", r"\.")):
        validate_manifest(manifest)


def test_validation_rejects_duplicate_expected_signals_and_missing_step():
    manifest = _manifest()
    signals = manifest["workload"]["expected_signals"]
    signals.append(dict(signals[0]))
    manifest["analysis"]["steps"] = manifest["analysis"]["steps"][:-1]

    with pytest.raises(CheckpointManifestError) as error:
        validate_manifest(manifest)
    message = str(error.value)
    assert "must be unique" in message
    assert "missing required steps: review" in message


def test_checksum_validation_fails_closed(tmp_path):
    profile = tmp_path / "profile.sqlite"
    profile.write_bytes(b"profile")
    manifest = _manifest(
        profile_path=profile.name,
        sha256=hashlib.sha256(b"different").hexdigest(),
    )

    with pytest.raises(CheckpointManifestError, match="checksum mismatch"):
        validate_manifest(manifest, profile_root=tmp_path, require_profile=True)


def test_canonical_manifest_bytes_are_stable_and_newline_terminated():
    manifest = _manifest()
    first = canonical_manifest_bytes(manifest)
    reordered = json.loads(json.dumps(manifest, sort_keys=False))
    second = canonical_manifest_bytes(reordered)
    assert first == second
    assert first.endswith(b"\n")
    assert first.index(b'"analysis"') < first.index(b'"capture"')


def test_command_expansion_only_accepts_known_paths():
    expanded = expand_command(
        ["tool", "{profile}", "{repo}", "{session}"],
        profile="/tmp/profile.sqlite",
        repo="/repo",
        session="/tmp/session",
    )
    assert expanded == ["tool", "/tmp/profile.sqlite", "/repo", "/tmp/session"]

    with pytest.raises(CheckpointManifestError, match="unsupported command template"):
        expand_command(
            ["tool", "{shell}"],
            profile="/tmp/profile.sqlite",
            repo="/repo",
            session="/tmp/session",
        )


def test_run_steps_writes_logs_and_reports_expected_exit_codes(tmp_path):
    manifest = _manifest()
    profile = tmp_path / "profile.sqlite"
    profile.write_bytes(b"profile")
    manifest["capture"]["sha256"] = hashlib.sha256(b"profile").hexdigest()
    for step in manifest["analysis"]["steps"]:
        step["command"] = [sys.executable, "-c", "print('step output')"]

    results = run_steps(
        manifest,
        repo_root=tmp_path,
        profile=profile,
        session=tmp_path / "session",
        output_dir=tmp_path / "logs",
    )
    assert len(results) == 5
    assert all(result.passed for result in results)
    assert (tmp_path / "logs/00-doctor.stdout").read_text() == "step output\n"
    assert (tmp_path / "logs/04-review.stderr").read_text() == ""


def test_documented_cli_validate_and_plan(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    validate = subprocess.run(
        [
            sys.executable,
            "scripts/checkpoint.py",
            "validate",
            str(MANIFEST),
            "--repo-root",
            str(ROOT),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr
    assert "valid checkpoint manifest" in validate.stdout

    plan = subprocess.run(
        [
            sys.executable,
            "scripts/checkpoint.py",
            "plan",
            str(MANIFEST),
            "--repo-root",
            str(ROOT),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert plan.returncode == 0, plan.stderr
    assert "doctor:" in plan.stdout
    assert "review:" in plan.stdout
