import json

import pytest

from nsys_ai.runspec import (
    EnvironmentSpec,
    NsysTraceOptions,
    RunSpec,
    RunSpecError,
    UnsupportedRunSpecVersionError,
    build_nsys_profile_argv,
    validate_persisted_secret_strings,
    validate_secret_boundaries,
)


def _full_spec(**overrides):
    values = {
        "argv": ("python", "train.py", "--model", "wan"),
        "cwd": "training/jobs",
        "repository": "/work/nsys-ai",
        "commit": "abc123",
        "environment": EnvironmentSpec(
            policy="inherit",
            public={"PROFILE_STEPS": "4", "PYTHONUNBUFFERED": "1"},
            secrets=("HF_TOKEN",),
        ),
        "warmup_steps": 2,
        "profile_steps": 4,
        "seed": 1234,
        "expected_gpu_count": 2,
        "expected_rank_count": 2,
        "trace_options": NsysTraceOptions(
            trace=("cuda", "nvtx", "nccl"),
            capture_range="cudaProfilerApi",
            cuda_memory_usage=True,
        ),
        "timeout_seconds": 600,
    }
    values.update(overrides)
    return RunSpec(**values)


def test_full_runspec_round_trip_and_deterministic_json():
    spec = _full_spec()
    canonical = spec.canonical_json_bytes()

    assert RunSpec.from_dict(spec.to_dict()) == spec
    assert RunSpec.from_json_bytes(canonical) == spec
    assert canonical == spec.canonical_json_bytes()
    assert canonical == json.dumps(
        spec.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def test_runspec_copies_mutable_constructor_inputs():
    argv = ["python", "train.py"]
    public = {"MODE": "profile"}
    secrets = ["TOKEN"]
    spec = RunSpec(
        argv=argv,
        environment=EnvironmentSpec(public=public, secrets=secrets),
    )
    before = spec.canonical_json_bytes()

    argv.append("--changed")
    public["MODE"] = "changed"
    secrets.append("OTHER_TOKEN")

    assert spec.canonical_json_bytes() == before


def test_public_environment_is_immutable_after_construction():
    spec = _full_spec()
    before_bytes = spec.canonical_json_bytes()
    before_key = spec.compatibility_key()

    with pytest.raises(TypeError):
        spec.environment.public["PROFILE_STEPS"] = "99"

    assert spec.canonical_json_bytes() == before_bytes
    assert spec.compatibility_key() == before_key


def test_canonical_json_normalizes_environment_order():
    first = _full_spec(
        environment=EnvironmentSpec(
            public={"Z_VAR": "last", "A_VAR": "first"},
            secrets=("Z_SECRET", "A_SECRET"),
        )
    )
    second = _full_spec(
        environment=EnvironmentSpec(
            public={"A_VAR": "first", "Z_VAR": "last"},
            secrets=("A_SECRET", "Z_SECRET"),
        )
    )

    assert first.canonical_json_bytes() == second.canonical_json_bytes()
    assert first.compatibility_key() == second.compatibility_key()


def test_trace_order_and_repo_relative_cwd_are_semantically_canonical():
    first = _full_spec(
        cwd="./training//jobs",
        trace_options=NsysTraceOptions(trace=("nccl", "cuda", "nvtx")),
    )
    second = _full_spec(
        cwd="training/jobs",
        trace_options=NsysTraceOptions(trace=("cuda", "nvtx", "nccl")),
    )

    assert first.cwd == second.cwd == "training/jobs"
    assert first.trace_options.trace == second.trace_options.trace
    assert first.canonical_json_bytes() == second.canonical_json_bytes()
    assert first.compatibility_key() == second.compatibility_key()


def test_unknown_schema_version_has_stated_reason():
    payload = _full_spec().to_dict()
    payload["schema_version"] = "9.0"

    with pytest.raises(UnsupportedRunSpecVersionError, match="9.0.*expected '0.2'"):
        RunSpec.from_dict(payload)


def test_schema_v01_remains_readable_without_new_identity_fields():
    payload = _full_spec().to_dict()
    payload["schema_version"] = "0.1"
    payload.pop("dirty")
    payload.pop("worktree_diff_sha256")

    parsed = RunSpec.from_dict(payload)

    assert parsed.dirty is False
    assert parsed.worktree_diff_sha256 is None


def test_schema_v01_rejects_v02_identity_fields():
    payload = _full_spec(dirty=True, worktree_diff_sha256="a" * 64).to_dict()
    payload["schema_version"] = "0.1"

    with pytest.raises(UnsupportedRunSpecVersionError, match="requires schema_version '0.2'"):
        RunSpec.from_dict(payload)


def test_dirty_worktree_identity_round_trips_and_is_distinct():
    first = _full_spec(
        dirty=True,
        worktree_diff_sha256="a" * 64,
    )
    second = _full_spec(
        dirty=True,
        worktree_diff_sha256="b" * 64,
    )

    assert RunSpec.from_dict(first.to_dict()) == first
    assert first.canonical_json_bytes() != second.canonical_json_bytes()
    assert first.compatibility_key() == second.compatibility_key()
    assert "uncommitted_worktree" in first.compatibility_limitations()


@pytest.mark.parametrize(
    "overrides",
    [
        {"dirty": True},
        {"worktree_diff_sha256": "not-a-sha"},
        {"dirty": False, "worktree_diff_sha256": "a" * 64},
    ],
)
def test_dirty_worktree_identity_requires_a_complete_provenance_pair(overrides):
    with pytest.raises(RunSpecError, match="dirty|worktree_diff_sha256"):
        _full_spec(**overrides)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"argv": "python train.py"}, "never a shell string"),
        ({"argv": ()}, "must not be empty"),
        ({"argv": ("",)}, r"argv\[0\].*must not be empty"),
        ({"cwd": "/absolute"}, "repository-relative"),
        ({"cwd": "../outside"}, "repository-relative"),
        ({"cwd": r"training\jobs"}, "repository-relative"),
        ({"warmup_steps": -1}, "non-negative"),
        ({"profile_steps": 0}, "positive"),
        ({"seed": -1}, "non-negative"),
        ({"expected_gpu_count": 0}, "positive"),
        ({"expected_rank_count": True}, "positive"),
        ({"timeout_seconds": 0}, "positive"),
        ({"runner": "remote"}, "must be 'local'"),
    ],
)
def test_runspec_validation_errors(kwargs, message):
    with pytest.raises(RunSpecError, match=message):
        _full_spec(**kwargs)


def test_nested_payloads_reject_unknown_fields():
    payload = _full_spec().to_dict()
    payload["output_path"] = "/tmp/report"
    with pytest.raises(RunSpecError, match="RunSpec has unknown field.*output_path"):
        RunSpec.from_dict(payload)

    payload = _full_spec().to_dict()
    payload["environment"]["resolved"] = {"HF_TOKEN": "must-not-be-read"}
    with pytest.raises(RunSpecError, match="environment has unknown field.*resolved"):
        RunSpec.from_dict(payload)


def test_malformed_nested_values_raise_runspec_errors():
    payload = _full_spec().to_dict()
    payload["environment"]["policy"] = []
    with pytest.raises(RunSpecError, match="environment.policy"):
        RunSpec.from_dict(payload)

    payload = _full_spec().to_dict()
    payload["trace_options"]["sample"] = []
    with pytest.raises(RunSpecError, match="trace_options.sample"):
        RunSpec.from_dict(payload)

    payload = _full_spec().to_dict()
    payload["trace_options"]["output"] = "/tmp/report"
    with pytest.raises(RunSpecError, match="trace_options has unknown field.*output"):
        RunSpec.from_dict(payload)


def test_environment_persists_names_but_never_resolved_secret_values(monkeypatch):
    secret_value = "private-value-that-must-not-leak"
    monkeypatch.setenv("HF_TOKEN", secret_value)
    environment = EnvironmentSpec(
        public={"VISIBLE_SETTING": "enabled"}, secrets=("HF_TOKEN",)
    )
    spec = _full_spec(environment=environment)

    encoded = spec.canonical_json_bytes().decode("utf-8")
    redacted = json.dumps(environment.redacted(), sort_keys=True)
    profile_argv = build_nsys_profile_argv(spec, "/tmp/report")

    assert secret_value not in encoded
    assert secret_value not in redacted
    assert secret_value not in profile_argv
    assert secret_value not in json.dumps(spec.compatibility_payload())
    assert environment.redacted() == {
        "HF_TOKEN": "<redacted>",
        "VISIBLE_SETTING": "enabled",
    }
    assert spec.to_dict()["environment"]["secrets"] == ["HF_TOKEN"]


@pytest.mark.parametrize(
    ("argv", "public", "location"),
    [
        (("python", "train.py", "--token=private-value"), {}, r"argv\[2\]"),
        (("python", "train.py", "private-value"), {}, r"argv\[2\]"),
        (
            ("python", "train.py"),
            {"TRAINING_ENDPOINT": "https://host/private-value/data"},
            r"environment.public.value\[0\]",
        ),
    ],
)
def test_secret_preflight_rejects_persisted_fields_without_echoing_value(
    argv, public, location
):
    spec = _full_spec(
        argv=argv,
        environment=EnvironmentSpec(public=public, secrets=("HF_TOKEN",)),
    )

    with pytest.raises(RunSpecError, match=location) as exc_info:
        validate_secret_boundaries(spec, {"HF_TOKEN": "private-value"})

    assert "HF_TOKEN" in str(exc_info.value)
    assert "private-value" not in str(exc_info.value)


def test_secret_preflight_accepts_clean_fields_and_empty_values():
    spec = _full_spec()
    validate_secret_boundaries(spec, {"HF_TOKEN": "private-value"})
    validate_secret_boundaries(spec, {"HF_TOKEN": ""})


def test_secret_preflight_requires_exact_declared_names_without_values_in_errors():
    spec = _full_spec()
    with pytest.raises(RunSpecError, match="missing declared name.*HF_TOKEN"):
        validate_secret_boundaries(spec, {})
    with pytest.raises(RunSpecError, match="undeclared name.*OTHER_TOKEN"):
        validate_secret_boundaries(
            spec, {"HF_TOKEN": "private-value", "OTHER_TOKEN": "other-value"}
        )


@pytest.mark.parametrize(
    ("secret_value", "overrides", "location"),
    [
        ("private-commit", {"commit": "rev-private-commit"}, "commit"),
        (
            "private-repository",
            {"repository": "/work/private-repository", "cwd": "."},
            "repository",
        ),
        (
            "private-cwd",
            {"repository": "/work/repo", "cwd": "jobs/private-cwd"},
            "cwd",
        ),
        (
            "process-tree",
            {"trace_options": NsysTraceOptions(sample="process-tree")},
            "trace_options.sample",
        ),
    ],
)
def test_secret_preflight_checks_every_persisted_string_value(
    secret_value, overrides, location
):
    spec = _full_spec(
        environment=EnvironmentSpec(secrets=("HF_TOKEN",)),
        **overrides,
    )

    with pytest.raises(RunSpecError, match=location) as exc_info:
        validate_secret_boundaries(spec, {"HF_TOKEN": secret_value})

    assert secret_value not in str(exc_info.value)


def test_secret_preflight_checks_public_mapping_keys_without_echoing_them():
    secret_value = "PRIVATE_KEY_FRAGMENT"
    public_name = f"PREFIX_{secret_value}_SUFFIX"
    spec = _full_spec(
        environment=EnvironmentSpec(
            public={public_name: "safe-value"}, secrets=("HF_TOKEN",)
        )
    )

    with pytest.raises(RunSpecError, match=r"environment.public.key\[0\]") as exc_info:
        validate_secret_boundaries(spec, {"HF_TOKEN": secret_value})

    message = str(exc_info.value)
    assert secret_value not in message
    assert public_name not in message


def test_persisted_secret_string_scanner_never_echoes_payload_keys():
    secret_value = "private-key-fragment"
    user_key = f"prefix-{secret_value}-suffix"

    with pytest.raises(RunSpecError) as exc_info:
        validate_persisted_secret_strings(
            {user_key: "safe"}, {"HF_TOKEN": secret_value}
        )

    message = str(exc_info.value)
    assert "HF_TOKEN" in message
    assert secret_value not in message
    assert user_key not in message


def test_persisted_secret_string_scanner_ignores_numeric_coincidence():
    validate_persisted_secret_strings(
        {"count": 1, "enabled": True, "missing": None},
        {"HF_TOKEN": "1"},
    )


def test_persisted_secret_string_scanner_rejects_non_string_keys():
    with pytest.raises(RunSpecError, match="mapping keys must be strings"):
        validate_persisted_secret_strings({1: "safe"}, {"HF_TOKEN": "private"})


def test_public_value_error_does_not_echo_its_user_controlled_key():
    secret_value = "private_value"
    public_name = "SENSITIVE_LOOKING_PUBLIC_NAME"
    spec = _full_spec(
        environment=EnvironmentSpec(
            public={public_name: f"prefix-{secret_value}"}, secrets=("HF_TOKEN",)
        )
    )

    with pytest.raises(RunSpecError, match=r"environment.public.value\[0\]") as exc_info:
        validate_secret_boundaries(spec, {"HF_TOKEN": secret_value})

    message = str(exc_info.value)
    assert secret_value not in message
    assert public_name not in message


def test_environment_validation_rejects_ambiguous_or_invalid_names():
    with pytest.raises(RunSpecError, match="both public and secret"):
        EnvironmentSpec(public={"TOKEN": "public"}, secrets=("TOKEN",))
    with pytest.raises(RunSpecError, match="invalid environment variable name"):
        EnvironmentSpec(secrets=("BAD-NAME",))
    with pytest.raises(RunSpecError, match="duplicates"):
        EnvironmentSpec(secrets=("TOKEN", "TOKEN"))


def test_trace_options_enforce_cuda_and_discard_environment():
    with pytest.raises(RunSpecError, match="must include cuda"):
        NsysTraceOptions(trace=("nvtx",))

    payload = NsysTraceOptions().to_dict()
    assert payload["discard_environment"] is True
    payload["discard_environment"] = False
    with pytest.raises(RunSpecError, match="must be true"):
        NsysTraceOptions.from_dict(payload)


def test_compatibility_key_includes_workload_inputs():
    baseline = _full_spec()
    variants = [
        _full_spec(argv=("python", "other.py")),
        _full_spec(cwd="training/other"),
        _full_spec(environment=EnvironmentSpec(public={"PROFILE_STEPS": "8"})),
        _full_spec(warmup_steps=3),
        _full_spec(profile_steps=5),
        _full_spec(seed=99),
        _full_spec(expected_gpu_count=4),
        _full_spec(expected_rank_count=4),
        _full_spec(trace_options=NsysTraceOptions(trace=("cuda", "nvtx"))),
    ]

    for candidate in variants:
        assert candidate.compatibility_key() != baseline.compatibility_key()
    assert baseline.compatibility_key().startswith("runspec1:sha256:")
    assert len(baseline.compatibility_key()) == len("runspec1:sha256:") + 64


def test_compatibility_excludes_provenance_timeout_and_secret_values():
    baseline = _full_spec()
    variants = [
        _full_spec(repository="/another/checkout"),
        _full_spec(commit="new-optimization-commit"),
        _full_spec(timeout_seconds=1200),
    ]

    for candidate in variants:
        assert candidate.compatibility_key() == baseline.compatibility_key()
    payload = json.dumps(baseline.compatibility_payload(), sort_keys=True)
    assert "repository" not in payload
    assert "commit" not in payload
    assert "timeout" not in payload


def test_secret_names_are_comparable_but_values_remain_an_explicit_limitation():
    baseline = _full_spec()
    different_name = _full_spec(
        environment=EnvironmentSpec(
            policy=baseline.environment.policy,
            public=baseline.environment.public,
            secrets=("A_DIFFERENT_TOKEN",),
        )
    )

    assert different_name.compatibility_key() != baseline.compatibility_key()
    assert baseline.compatibility_payload()["environment"]["secret_names"] == [
        "HF_TOKEN"
    ]
    assert baseline.compatibility_limitations() == (
        "inherited_environment_unresolved",
        "secret_environment_values_unresolved",
    )


def test_clean_public_environment_has_no_compatibility_limitations():
    spec = _full_spec(environment=EnvironmentSpec(policy="clean", public={"MODE": "test"}))

    assert spec.compatibility_limitations() == ()


def test_nsys_profile_argv_is_token_preserving_and_has_no_shell_interpolation():
    workload = (
        "python",
        "train script.py",
        "--label",
        "$(touch /tmp/should-not-run); echo unsafe",
    )
    spec = _full_spec(argv=workload)

    argv = build_nsys_profile_argv(
        spec, "/tmp/profile with spaces", nsys_executable="/opt/nsys bin/nsys"
    )

    assert argv[:4] == [
        "/opt/nsys bin/nsys",
        "profile",
        "-o",
        "/tmp/profile with spaces",
    ]
    assert argv[-len(workload) :] == list(workload)
    assert "--trace=cuda,nccl,nvtx" in argv
    assert "--discard-environment=true" in argv
    assert "--capture-range=cudaProfilerApi" in argv
    assert "--cuda-memory-usage=true" in argv
    assert all(isinstance(token, str) for token in argv)


def test_output_path_is_operational_not_persisted_or_comparable():
    spec = _full_spec()

    left = build_nsys_profile_argv(spec, "/tmp/left")
    right = build_nsys_profile_argv(spec, "/tmp/right")

    assert left != right
    assert "/tmp/left" not in spec.canonical_json_bytes().decode("utf-8")
    assert "/tmp/right" not in json.dumps(spec.compatibility_payload())
