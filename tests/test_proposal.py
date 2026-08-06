import json

import pytest

from nsys_ai.annotation import Finding, TraceSelection
from nsys_ai.profile_runner import LocalProfileRunner
from nsys_ai.proposal import (
    PROPOSAL_SCHEMA_VERSION,
    Proposal,
    ProposalError,
    UnsupportedProposalVersionError,
    generate_proposal,
)
from nsys_ai.runspec import EnvironmentSpec, RunSpec, RunSpecError


def _finding(**overrides):
    values = {
        "type": "region",
        "label": "Exposed communication",
        "start_ns": 100,
        "end_ns": 200,
        "id": "overlap-exposed-0",
        "confidence": 0.84,
        "explanation": "Communication is exposed in the backward pass.",
        "suggested_actions": ["Test moving the all-reduce earlier."],
        "false_positive_notes": ["Confirm the workload is representative."],
        "headroom_ms": 12.5,
        "headroom_basis": "capture_total",
        "selection": TraceSelection(
            id="selection-0",
            profile_id="nsys1:sha256:profile",
            source="skill:overlap_breakdown",
            start_ns=100,
            end_ns=200,
            gpu_ids=[0],
            nvtx_path=["iteration", "backward"],
        ),
        "provenance": {"skill": "overlap_breakdown", "row": 0},
    }
    values.update(overrides)
    return Finding(**values)


def _runspec():
    return RunSpec(
        argv=("python", "train.py", "--steps", "4"),
        repository="/work/training",
        cwd="jobs",
        environment=EnvironmentSpec(policy="clean", public={"MODE": "profile"}),
        profile_steps=4,
        seed=7,
    )


def test_proposal_round_trip_is_deterministic_and_projected():
    finding = _finding()
    spec = _runspec()

    first = generate_proposal(finding, spec)
    second = generate_proposal(finding, spec)
    canonical = first.canonical_json_bytes()

    assert not first.abstained
    assert first.proposal_id == second.proposal_id
    assert canonical == second.canonical_json_bytes()
    assert Proposal.from_dict(first.to_dict()) == first
    assert Proposal.from_json_bytes(canonical) == first

    payload = json.loads(canonical)
    assert payload["schema_version"] == PROPOSAL_SCHEMA_VERSION
    assert payload["source_finding_id"] == finding.id
    assert payload["source_profile_id"] == finding.selection.profile_id
    assert "source_finding" not in payload
    assert payload["trace_target"] == finding.selection.to_dict()
    assert payload["trace_target"]["nvtx_path"] == ["iteration", "backward"]
    assert payload["summary"] == finding.explanation
    assert payload["suggested_actions"] == finding.suggested_actions
    assert payload["expected_impact"] == {
        "kind": "measured_headroom",
        "headroom_ms": 12.5,
        "headroom_basis": "capture_total",
    }
    assert payload["confidence"] == finding.confidence
    assert payload["verification"] == {"kind": "runspec", "runspec": spec.to_dict()}
    assert "source file" in " ".join(payload["limitations"])
    assert "gain" in " ".join(payload["limitations"])
    assert "file" not in payload["trace_target"]
    assert "line" not in payload["trace_target"]
    assert "patch" not in payload


def test_generated_proposal_copies_projected_mutable_inputs():
    finding = _finding()
    proposal = generate_proposal(finding, _runspec())
    before = proposal.canonical_json_bytes()

    finding.suggested_actions.append("A later mutation")
    finding.selection.nvtx_path.append("later")
    finding.provenance["row"] = 99

    assert proposal.canonical_json_bytes() == before
    with pytest.raises(TypeError):
        proposal.trace_target["label"] = "changed"


@pytest.mark.parametrize(
    ("finding", "spec", "reason"),
    [
        (_finding(id=None), _runspec(), "no id"),
        (_finding(selection=None), _runspec(), "no trace selection"),
        (
            _finding(
                selection=TraceSelection(id="selection-0", profile_id="", source="skill:x")
            ),
            _runspec(),
            "no profile id",
        ),
        (_finding(suggested_actions=[]), _runspec(), "no suggested action"),
        (_finding(), None, "verification RunSpec is required"),
    ],
)
def test_missing_required_input_produces_round_trippable_abstention(finding, spec, reason):
    proposal = generate_proposal(finding, spec)

    assert proposal.abstained
    assert reason in proposal.abstention_reason
    assert Proposal.from_json_bytes(proposal.canonical_json_bytes()) == proposal


def test_missing_selection_abstention_keeps_exact_finding_pointer():
    finding = _finding(selection=None)
    proposal = generate_proposal(finding, _runspec())
    payload = proposal.to_dict()

    assert payload["source_finding_id"] == finding.id
    assert payload["source_profile_id"] == ""
    assert payload["trace_target"] is None


def test_unquantified_impact_and_confidence_stay_null():
    proposal = generate_proposal(
        _finding(headroom_ms=None, headroom_basis=None, confidence=None),
        _runspec(),
    )

    assert not proposal.abstained
    assert proposal.expected_impact is None
    assert proposal.confidence is None
    assert proposal.to_dict()["expected_impact"] is None
    assert proposal.to_dict()["confidence"] is None


def test_runspec_comparability_limitations_are_preserved():
    spec = RunSpec(
        argv=("python", "train.py"),
        environment=EnvironmentSpec(policy="inherit", secrets=("TOKEN",)),
    )
    proposal = generate_proposal(
        _finding(), spec, resolved_secrets={"TOKEN": "not-persisted"}
    )

    assert "inherited_environment_unresolved" in proposal.limitations
    assert "secret_environment_values_unresolved" in proposal.limitations


def test_changed_artifact_content_is_rejected_by_proposal_id():
    payload = generate_proposal(_finding(), _runspec()).to_dict()
    payload["proposal_id"] = "proposal1:sha256:" + "0" * 64

    with pytest.raises(ProposalError, match="proposal_id does not match"):
        Proposal.from_dict(payload)


def test_unsupported_version_and_unknown_fields_are_rejected():
    payload = generate_proposal(_finding(), _runspec()).to_dict()
    payload["schema_version"] = "9.0"
    with pytest.raises(UnsupportedProposalVersionError, match="9.0.*expected '0.1'"):
        Proposal.from_dict(payload)

    payload = generate_proposal(_finding(), _runspec()).to_dict()
    payload["source_patch"] = "not allowed"
    with pytest.raises(ProposalError, match="unknown field.*source_patch"):
        Proposal.from_dict(payload)


def test_invalid_finding_and_verification_types_are_rejected():
    with pytest.raises(ProposalError, match="finding must be"):
        generate_proposal({}, _runspec())
    with pytest.raises(ProposalError, match="verification must be"):
        generate_proposal(_finding(), "python train.py")


@pytest.mark.parametrize("missing", ["expected_impact", "confidence"])
def test_writer_fields_are_required_on_read(missing):
    payload = generate_proposal(_finding(), _runspec()).to_dict()
    del payload[missing]

    with pytest.raises(ProposalError, match=f"missing field.*{missing}"):
        Proposal.from_dict(payload)


def test_runner_rejected_secret_cannot_be_persisted_in_proposal(tmp_path, monkeypatch):
    secret = "sentinel-secret-value"
    monkeypatch.setenv("RUNNER_SECRET", secret)
    spec = RunSpec(
        argv=("python", "train.py", f"--token={secret}"),
        environment=EnvironmentSpec(secrets=("RUNNER_SECRET",)),
    )

    with pytest.raises(RunSpecError) as runner_error:
        LocalProfileRunner(tmp_path / "runner-artifacts").run(spec)
    with pytest.raises(RunSpecError) as proposal_error:
        generate_proposal(
            _finding(), spec, resolved_secrets={"RUNNER_SECRET": secret}
        )

    assert str(proposal_error.value) == str(runner_error.value)
    assert secret not in str(proposal_error.value)
    assert not (tmp_path / "runner-artifacts").exists()


def test_resolved_secret_mapping_is_preflight_only():
    secret = "sentinel-secret-value"
    spec = RunSpec(
        argv=("python", "train.py"),
        environment=EnvironmentSpec(secrets=("RUNNER_SECRET",)),
    )

    payload = generate_proposal(
        _finding(), spec, resolved_secrets={"RUNNER_SECRET": secret}
    ).to_dict()

    environment = payload["verification"]["runspec"]["environment"]
    assert environment == {
        "policy": "inherit",
        "public": {},
        "secrets": ["RUNNER_SECRET"],
    }
    assert "resolved_secrets" not in payload


def test_secret_in_projected_finding_text_is_rejected_without_echo():
    secret = "sentinel-secret-value"
    spec = RunSpec(
        argv=("python", "train.py"),
        environment=EnvironmentSpec(secrets=("RUNNER_SECRET",)),
    )

    with pytest.raises(RunSpecError) as exc_info:
        generate_proposal(
            _finding(explanation=f"Do not persist {secret}"),
            spec,
            resolved_secrets={"RUNNER_SECRET": secret},
        )

    assert "persisted string" in str(exc_info.value)
    assert secret not in str(exc_info.value)


@pytest.mark.parametrize(
    "provenance",
    [
        {"sentinel-secret-value": "safe"},
        {"safe": "sentinel-secret-value"},
    ],
)
def test_non_projected_provenance_is_not_persisted(provenance):
    secret = "sentinel-secret-value"
    spec = RunSpec(
        argv=("python", "train.py"),
        environment=EnvironmentSpec(secrets=("RUNNER_SECRET",)),
    )

    proposal = generate_proposal(
        _finding(provenance=provenance),
        spec,
        resolved_secrets={"RUNNER_SECRET": secret},
    )
    baseline = generate_proposal(
        _finding(provenance={}),
        spec,
        resolved_secrets={"RUNNER_SECRET": secret},
    )

    assert proposal.canonical_json_bytes() == baseline.canonical_json_bytes()
    assert "provenance" not in proposal.to_dict()


@pytest.mark.parametrize("invalid", [[], (), ""])
def test_falsey_non_mapping_resolved_secrets_are_rejected(invalid):
    with pytest.raises(RunSpecError, match="resolved_secrets must be an object"):
        generate_proposal(_finding(), _runspec(), resolved_secrets=invalid)


def test_non_finite_finding_number_is_not_serialized():
    with pytest.raises(ProposalError, match="confidence must be between"):
        generate_proposal(_finding(confidence=float("nan")), _runspec())


def test_malformed_nested_trace_target_error_is_wrapped():
    payload = generate_proposal(_finding(), _runspec()).to_dict()
    payload["trace_target"] = "not-an-object"

    with pytest.raises(ProposalError, match="trace_target must be an object"):
        Proposal.from_dict(payload)


@pytest.mark.parametrize("field", ["suggested_actions", "false_positive_notes"])
def test_string_container_finding_fields_reject_scalar_strings(field):
    with pytest.raises(ProposalError, match=f"source finding {field} must be an array"):
        generate_proposal(_finding(**{field: "abc"}), _runspec())


def test_suggested_actions_reject_non_string_members():
    with pytest.raises(ProposalError, match=r"suggested_actions\[1\] must be a string"):
        generate_proposal(
            _finding(suggested_actions=["valid", 7]),
            _runspec(),
        )


def test_malformed_trace_selection_shape_is_rejected():
    selection = TraceSelection(
        id="selection-0",
        profile_id="nsys1:sha256:profile",
        source="skill:test",
        nvtx_path="iteration",
    )

    with pytest.raises(ProposalError, match="nvtx_path must be an array"):
        generate_proposal(_finding(selection=selection), _runspec())


@pytest.mark.parametrize(
    ("finding_id", "profile_id", "message"),
    [
        (" finding-0", "nsys1:sha256:profile", "source finding id"),
        ("finding-0", "nsys1:sha256: profile", "selection.profile_id"),
    ],
)
def test_provenance_ids_with_whitespace_are_rejected(finding_id, profile_id, message):
    selection = TraceSelection(id="selection-0", profile_id=profile_id, source="skill:test")

    with pytest.raises(ProposalError, match=message):
        generate_proposal(_finding(id=finding_id, selection=selection), _runspec())


@pytest.mark.parametrize(
    "overrides",
    [
        {"type": 7},
        {"start_ns": "later"},
        {"end_ns": False},
        {"provenance": []},
    ],
)
def test_non_projected_finding_fields_do_not_expand_proposal_schema(overrides):
    baseline = generate_proposal(_finding(), _runspec()).canonical_json_bytes()

    assert generate_proposal(
        _finding(**overrides), _runspec()
    ).canonical_json_bytes() == baseline


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"confidence": -1}, "confidence must be between"),
        ({"headroom_basis": 7}, "headroom_basis must be a string"),
        ({"explanation": 7}, "explanation must be a string"),
    ],
)
def test_malformed_projected_finding_fields_are_rejected(overrides, message):
    with pytest.raises(ProposalError, match=message):
        generate_proposal(_finding(**overrides), _runspec())
