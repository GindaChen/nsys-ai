import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from nsys_ai.exceptions import ProfileError
from nsys_ai.fingerprint import PROFILE_ID_VERSION
from nsys_ai.profile_reference import (
    LocalProfileReference,
    validate_local_profile_reference,
)
from nsys_ai.profile_runner import build_local_profile_reference

VALID_PROFILE_ID = f"{PROFILE_ID_VERSION}:sha256:" + "0" * 64


def _digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree_digest(root: Path) -> dict[str, str]:
    snapshot = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            snapshot[relative] = f"symlink:{path.readlink()}"
        elif path.is_dir():
            snapshot[relative] = "directory"
        elif path.is_file():
            snapshot[relative] = f"file:{path.stat().st_size}:{_digest(path)}"
        else:
            snapshot[relative] = "special"
    return snapshot


def test_existing_profile_reference_matches_profile_and_evidence_identity(
    minimal_nsys_db_path,
):
    from nsys_ai.evidence_builder import EvidenceBuilder
    from nsys_ai.profile import Profile

    reference = build_local_profile_reference(minimal_nsys_db_path)
    with Profile(minimal_nsys_db_path, cache_mode="direct") as profile:
        report = EvidenceBuilder(profile).build(only=[])
        assert reference.profile_id == report.profile_id
        assert reference.schema_version == profile.schema.schema_version
        assert reference.product_version == profile.schema.version
        assert reference.kernel_count == profile.meta.kernel_count

    assert reference.path == str(Path(minimal_nsys_db_path).absolute())
    assert reference.profile_id.startswith("nsys2:sha256:")
    assert reference.kernel_count == 5


def test_parquetdir_reference_records_source_and_resolved_storage(tmp_path):
    from test_parquetdir_backend import _create_parquetdir_profile

    cache = Path(_create_parquetdir_profile(tmp_path / "mock.parquetdir"))

    reference = build_local_profile_reference(cache)

    assert reference.storage_kind == "parquetdir"
    assert reference.path == str(cache.absolute())
    assert reference.resolved_path == str(cache.absolute())
    assert reference.profile_id.startswith("nsys2:")
    assert reference.kernel_count > 0


def test_nsys_rep_reference_validation_accepts_parquetdir_resolution(tmp_path):
    source = tmp_path / "profile.nsys-rep"
    resolved = tmp_path / "profile.parquetdir"
    source.write_bytes(b"capture")
    resolved.mkdir()
    (resolved / "kernels.parquet").write_bytes(b"parquet")
    reference = LocalProfileReference(
        path=str(source),
        profile_id=VALID_PROFILE_ID,
        schema_version=None,
        product_version=None,
        kernel_count=1,
        storage_kind="nsys-rep",
        resolved_path=str(resolved),
    )

    assert validate_local_profile_reference(reference, require_file=True) is reference


def test_parquetdir_reference_rejects_symlink_parquet_file(tmp_path):
    parquetdir = tmp_path / "profile.parquetdir"
    parquetdir.mkdir()
    target = tmp_path / "outside.parquet"
    target.write_bytes(b"parquet")
    (parquetdir / "kernels.parquet").symlink_to(target)
    reference = LocalProfileReference(
        path=str(parquetdir),
        profile_id=VALID_PROFILE_ID,
        schema_version=None,
        product_version=None,
        kernel_count=1,
        storage_kind="parquetdir",
        resolved_path=str(parquetdir),
    )

    with pytest.raises(ValueError, match="symlink parquet files"):
        validate_local_profile_reference(reference, require_file=True)


@pytest.mark.parametrize("source_kind", ["rep", "parquetdir", "parquet_file"])
def test_profile_reference_rejects_symlinked_ingest_source_before_profile(
    tmp_path, monkeypatch, source_kind
):
    import nsys_ai.profile_runner as profile_runner

    real_source = tmp_path / "real.nsys-rep"
    real_source.write_bytes(b"capture")
    source = tmp_path / "input.nsys-rep"
    source.symlink_to(real_source)
    if source_kind == "parquetdir":
        real_source = tmp_path / "real.parquetdir"
        real_source.mkdir()
        (real_source / "kernels.parquet").write_bytes(b"parquet")
        source = tmp_path / "input.parquetdir"
        source.symlink_to(real_source, target_is_directory=True)
    elif source_kind == "parquet_file":
        source = tmp_path / "input.parquetdir"
        source.mkdir()
        parquet_target = tmp_path / "kernels.parquet"
        parquet_target.write_bytes(b"parquet")
        (source / "kernels.parquet").symlink_to(parquet_target)

    called = False
    real_profile = profile_runner.Profile

    def unexpected_profile(*args, **kwargs):
        nonlocal called
        called = True
        return real_profile(*args, **kwargs)

    monkeypatch.setattr(profile_runner, "Profile", unexpected_profile)
    with pytest.raises(ProfileError):
        build_local_profile_reference(source)
    assert not called


def test_existing_profile_reference_has_no_local_artifact_or_cache_side_effects(
    minimal_nsys_db_path, tmp_path
):
    profile = Path(minimal_nsys_db_path)
    before_names = {item.relative_to(tmp_path) for item in tmp_path.rglob("*")}
    before_size = profile.stat().st_size
    before_digest = _digest(profile)

    build_local_profile_reference(profile)

    assert profile.stat().st_size == before_size
    assert _digest(profile) == before_digest
    assert {item.relative_to(tmp_path) for item in tmp_path.rglob("*")} == before_names
    assert not (tmp_path / ".nsys-ai").exists()


def test_relative_profile_path_is_normalized_without_copying(
    minimal_nsys_db_path, monkeypatch
):
    profile = Path(minimal_nsys_db_path)
    monkeypatch.chdir(profile.parent)

    reference = build_local_profile_reference(profile.name)

    assert reference.path == str(profile.absolute())
    assert list(profile.parent.glob("*.sqlite")) == [profile]


@pytest.mark.parametrize(
    ("path_value", "message"),
    [
        ("", "must not be empty"),
        (b"profile.sqlite", "must be a path string"),
        ("bad\x00.sqlite", "must not contain NUL bytes"),
    ],
)
def test_profile_reference_rejects_invalid_path_contract(path_value, message):
    with pytest.raises(ProfileError, match=message):
        build_local_profile_reference(path_value)


def test_profile_reference_rejects_missing_directory_empty_and_invalid_files(
    tmp_path,
):
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(ProfileError, match="does not exist"):
        build_local_profile_reference(missing)

    directory = tmp_path / "directory.sqlite"
    directory.mkdir()
    with pytest.raises(ProfileError, match="not a regular file"):
        build_local_profile_reference(directory)

    empty = tmp_path / "empty.sqlite"
    empty.touch()
    with pytest.raises(ProfileError, match="file is empty"):
        build_local_profile_reference(empty)

    invalid = tmp_path / "invalid.sqlite"
    invalid.write_bytes(b"not a sqlite database")
    with pytest.raises(ProfileError, match="not a valid Nsight SQLite export"):
        build_local_profile_reference(invalid)


def test_profile_reference_rejects_symlink_and_special_file(
    minimal_nsys_db_path, tmp_path
):
    symlink = tmp_path / "symlink.sqlite"
    symlink.symlink_to(minimal_nsys_db_path)
    with pytest.raises(ProfileError, match="symbolic link"):
        build_local_profile_reference(symlink)

    real_directory = tmp_path / "real-directory"
    real_directory.mkdir()
    nested_profile = real_directory / "nested.sqlite"
    shutil.copyfile(minimal_nsys_db_path, nested_profile)
    directory_symlink = tmp_path / "directory-symlink"
    directory_symlink.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(ProfileError, match="symbolic link"):
        build_local_profile_reference(directory_symlink / nested_profile.name)

    fifo = tmp_path / "special.sqlite"
    os.mkfifo(fifo)
    with pytest.raises(ProfileError, match="not a regular file"):
        build_local_profile_reference(fifo)


def test_same_inode_same_size_header_mutation_is_rejected(
    minimal_nsys_db_path, monkeypatch
):
    import nsys_ai.profile_runner as profile_runner

    profile = Path(minimal_nsys_db_path)
    real_identity = profile_runner.get_profile_id
    before = profile.stat()

    def mutate_after_identity(*args, **kwargs):
        identity = real_identity(*args, **kwargs)
        descriptor = os.open(profile, os.O_RDWR)
        try:
            os.pwrite(descriptor, b"\x01\x02\x03\x04", 68)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.utime(profile, ns=(before.st_atime_ns, before.st_mtime_ns))
        return identity

    monkeypatch.setattr(profile_runner, "get_profile_id", mutate_after_identity)

    with pytest.raises(ProfileError, match="changed during validation"):
        build_local_profile_reference(profile)

    after = profile.stat()
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    assert after.st_ctime_ns != before.st_ctime_ns


def test_path_replacement_during_validation_is_rejected(
    minimal_nsys_db_path, tmp_path, monkeypatch
):
    import nsys_ai.profile_runner as profile_runner

    profile = Path(minimal_nsys_db_path)
    replacement = tmp_path / "replacement.sqlite"
    shutil.copyfile(profile, replacement)
    real_identity = profile_runner.get_profile_id
    original_inode = profile.stat().st_ino

    def replace_after_identity(*args, **kwargs):
        identity = real_identity(*args, **kwargs)
        os.replace(replacement, profile)
        return identity

    monkeypatch.setattr(profile_runner, "get_profile_id", replace_after_identity)

    with pytest.raises(ProfileError, match="changed during validation"):
        build_local_profile_reference(profile)

    assert profile.stat().st_ino != original_inode


def test_validation_closes_profile_descriptor_on_failure(tmp_path, monkeypatch):
    import nsys_ai.profile_runner as profile_runner

    invalid = tmp_path / "invalid.sqlite"
    invalid.write_bytes(b"not a sqlite database")
    real_open = profile_runner.os.open
    descriptors: list[int] = []

    def recording_open(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(profile_runner.os, "open", recording_open)

    with pytest.raises(ProfileError, match="not a valid Nsight SQLite export"):
        build_local_profile_reference(invalid)

    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_profile_reference_rejects_profile_without_kernel_activity(
    minimal_nsys_db_path,
):
    with sqlite3.connect(minimal_nsys_db_path) as connection:
        connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")

    with pytest.raises(ProfileError, match="no GPU kernel activity"):
        build_local_profile_reference(minimal_nsys_db_path)


@pytest.mark.parametrize(
    "invalid_identity",
    [
        "nsys1:sha256:" + "0" * 64,
        f"{PROFILE_ID_VERSION}:path:" + "0" * 64,
        "opaque-profile-id",
        f"{PROFILE_ID_VERSION}:sha256:" + "A" * 64,
        f"{PROFILE_ID_VERSION}:sha256:" + "0" * 63,
        f"{PROFILE_ID_VERSION}:sha256:" + "0" * 65,
    ],
)
def test_profile_reference_factory_rejects_noncanonical_identity(
    minimal_nsys_db_path, monkeypatch, invalid_identity
):
    import nsys_ai.profile_runner as profile_runner

    monkeypatch.setattr(
        profile_runner,
        "get_profile_id",
        lambda *_args, **_kwargs: invalid_identity,
    )

    with pytest.raises(ProfileError, match="identity is invalid"):
        build_local_profile_reference(minimal_nsys_db_path)


def test_profile_reference_factory_calls_shared_validator(
    minimal_nsys_db_path, monkeypatch
):
    import nsys_ai.profile_runner as profile_runner

    calls = []
    real_validator = profile_runner.validate_local_profile_reference

    def recording_validator(reference, *, require_file):
        calls.append((reference, require_file))
        return real_validator(reference, require_file=require_file)

    monkeypatch.setattr(
        profile_runner,
        "validate_local_profile_reference",
        recording_validator,
    )

    reference = build_local_profile_reference(minimal_nsys_db_path)

    assert calls == [(reference, True)]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("path", "relative.sqlite", "absolute path string"),
        ("path", "/tmp/profile.nsys-rep", "must name a .sqlite"),
        ("path", "/tmp/bad\x00.sqlite", "absolute path string"),
        ("schema_version", "", "schema_version"),
        ("schema_version", 1, "schema_version"),
        ("product_version", "", "product_version"),
        ("kernel_count", 0, "no GPU kernel activity"),
        ("kernel_count", True, "kernel count is invalid"),
    ],
)
def test_shared_profile_reference_validator_rejects_invalid_fields(
    field, value, message
):
    reference = LocalProfileReference(
        path="/tmp/profile.sqlite",
        profile_id=VALID_PROFILE_ID,
        schema_version="3.25.0",
        product_version="2026.2.1.106",
        kernel_count=1,
    )

    with pytest.raises(ValueError, match=message):
        validate_local_profile_reference(
            replace(reference, **{field: value}),
            require_file=False,
        )


def test_shared_file_inspection_detects_path_swap(tmp_path, monkeypatch):
    import nsys_ai.profile_reference as profile_reference

    profile = tmp_path / "profile.sqlite"
    replacement = tmp_path / "replacement.sqlite"
    profile.write_bytes(b"original")
    replacement.write_bytes(b"replaced")
    real_resolve = profile_reference.Path.resolve
    swapped = False

    def swap_then_resolve(path, *args, **kwargs):
        nonlocal swapped
        if path == profile and not swapped:
            swapped = True
            os.replace(replacement, profile)
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(profile_reference.Path, "resolve", swap_then_resolve)

    with pytest.raises(ValueError, match="changed while being inspected"):
        profile_reference.inspect_local_profile_file(profile)


def test_missing_file_inspection_detects_parent_swap(tmp_path, monkeypatch):
    import nsys_ai.profile_reference as profile_reference

    parent = tmp_path / "profile-parent"
    parent.mkdir()
    profile = parent / "missing.sqlite"
    replaced_parent = tmp_path / "replaced-parent"
    real_open_chain = profile_reference._open_profile_parent_chain
    calls = 0

    def swap_after_first_parent_walk(path):
        nonlocal calls
        result = real_open_chain(path)
        calls += 1
        if calls == 1:
            parent.rename(replaced_parent)
            parent.mkdir()
        return result

    monkeypatch.setattr(
        profile_reference,
        "_open_profile_parent_chain",
        swap_after_first_parent_walk,
    )

    with pytest.raises(ValueError, match="changed while being inspected"):
        profile_reference.inspect_local_profile_file(profile, allow_missing=True)


@pytest.mark.parametrize(
    "path_kind",
    [
        "parent_symlink",
        "broken_parent_symlink",
        "special_parent",
        "final_symlink",
        "broken_final_symlink",
        "canonical_mismatch",
        "missing_canonical_mismatch",
        "empty",
        "special",
    ],
)
@pytest.mark.parametrize("require_file", [True, False])
def test_shared_reference_file_contract_rejects_unsafe_existing_path(
    tmp_path, path_kind, require_file
):
    valid = tmp_path / "valid.sqlite"
    valid.write_bytes(b"profile")
    if path_kind == "parent_symlink":
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        target = real_parent / "profile.sqlite"
        target.write_bytes(b"profile")
        alias = tmp_path / "parent-alias"
        alias.symlink_to(real_parent, target_is_directory=True)
        path = alias / target.name
    elif path_kind == "broken_parent_symlink":
        alias = tmp_path / "broken-parent-alias"
        alias.symlink_to(tmp_path / "missing-parent", target_is_directory=True)
        path = alias / "profile.sqlite"
    elif path_kind == "special_parent":
        parent = tmp_path / "special-parent"
        os.mkfifo(parent)
        path = parent / "profile.sqlite"
    elif path_kind == "final_symlink":
        path = tmp_path / "final.sqlite"
        path.symlink_to(valid)
    elif path_kind == "broken_final_symlink":
        path = tmp_path / "broken-final.sqlite"
        path.symlink_to(tmp_path / "missing-target.sqlite")
    elif path_kind == "canonical_mismatch":
        nested = tmp_path / "nested"
        nested.mkdir()
        path = nested / ".." / valid.name
    elif path_kind == "missing_canonical_mismatch":
        nested = tmp_path / "nested"
        nested.mkdir()
        path = nested / ".." / "missing.sqlite"
    elif path_kind == "empty":
        path = tmp_path / "empty.sqlite"
        path.touch()
    else:
        path = tmp_path / "special.sqlite"
        os.mkfifo(path)
    reference = LocalProfileReference(
        path=str(path),
        profile_id=VALID_PROFILE_ID,
        schema_version="3.25.0",
        product_version="2026.2.1.106",
        kernel_count=1,
    )

    with pytest.raises(ValueError) as rejected:
        validate_local_profile_reference(reference, require_file=require_file)

    assert str(path) not in str(rejected.value)


def test_shared_reference_contract_allows_only_a_truly_missing_optional_file(tmp_path):
    reference = LocalProfileReference(
        path=str(tmp_path / "missing.sqlite"),
        profile_id=VALID_PROFILE_ID,
        schema_version="3.25.0",
        product_version="2026.2.1.106",
        kernel_count=1,
    )

    assert (
        validate_local_profile_reference(reference, require_file=False) is reference
    )
    with pytest.raises(ValueError, match="does not exist"):
        validate_local_profile_reference(reference, require_file=True)


def test_secret_bearing_path_is_rejected_without_disclosing_secret(
    minimal_nsys_db_path, tmp_path
):
    secret = "private-profile-token"
    secret_dir = tmp_path / secret
    secret_dir.mkdir()
    profile = secret_dir / "profile.sqlite"
    shutil.copyfile(minimal_nsys_db_path, profile)
    before_digest = _digest(profile)

    with pytest.raises(ProfileError) as rejected:
        build_local_profile_reference(
            profile,
            resolved_secrets={"RUNNER_SECRET": secret},
        )

    assert "RUNNER_SECRET" in str(rejected.value)
    assert secret not in str(rejected.value)
    assert _digest(profile) == before_digest
    assert set(secret_dir.iterdir()) == {profile}


def test_untrusted_pathlike_exception_has_no_secret_chain_or_log(caplog):
    secret = "sentinel-fspath-secret"

    class SecretPath:
        def __fspath__(self):
            raise RuntimeError(secret)

    with pytest.raises(ProfileError) as rejected:
        build_local_profile_reference(SecretPath())

    rendered = "".join(
        traceback.format_exception(rejected.type, rejected.value, rejected.tb)
    )
    assert rejected.value.__cause__ is None
    assert rejected.value.__context__ is None
    assert secret not in rendered
    assert secret not in caplog.text


def test_secret_validator_exception_has_no_secret_chain_or_log(
    minimal_nsys_db_path, monkeypatch, caplog
):
    import nsys_ai.profile_runner as profile_runner

    secret = "sentinel-validator-secret"

    def fail_validation(*_args, **_kwargs):
        try:
            raise RuntimeError(secret)
        except RuntimeError as exc:
            raise profile_runner.RunSpecError(secret) from exc

    monkeypatch.setattr(
        profile_runner,
        "validate_persisted_secret_strings",
        fail_validation,
    )

    with pytest.raises(ProfileError) as rejected:
        build_local_profile_reference(minimal_nsys_db_path)

    rendered = "".join(
        traceback.format_exception(rejected.type, rejected.value, rejected.tb)
    )
    assert rejected.value.__cause__ is None
    assert rejected.value.__context__ is None
    assert secret not in rendered
    assert secret not in caplog.text


def test_hot_rollback_journal_is_never_recovered_in_place(
    minimal_nsys_db_path, tmp_path
):
    profile = Path(minimal_nsys_db_path)
    crash_script = """
import os
import sqlite3
import sys

connection = sqlite3.connect(sys.argv[1])
connection.execute("PRAGMA journal_mode=DELETE")
connection.execute("PRAGMA synchronous=FULL")
connection.execute("PRAGMA cache_size=1")
connection.execute("PRAGMA cache_spill=ON")
connection.execute("BEGIN IMMEDIATE")
for index in range(256):
    connection.execute(
        "INSERT INTO StringIds(id, value) VALUES (?, ?)",
        (1000 + index, "uncommitted-" + ("x" * 4096)),
    )
os._exit(0)
"""
    crashed = subprocess.run(
        [sys.executable, "-c", crash_script, str(profile)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert crashed.returncode == 0, crashed.stderr
    journal = Path(str(profile) + "-journal")
    assert journal.is_file() and journal.stat().st_size > 512
    before = _tree_digest(tmp_path)

    reference = build_local_profile_reference(profile)

    assert reference.kernel_count == 5
    assert _tree_digest(tmp_path) == before


def test_canonical_and_versioned_kernel_identity_is_content_based(
    minimal_nsys_db_path, tmp_path
):
    from nsys_ai.evidence_builder import EvidenceBuilder
    from nsys_ai.profile import Profile

    equivalent_identities = []
    for suffix in ("", "_V2", "_V3"):
        profile_path = tmp_path / f"equivalent{suffix or '-canonical'}.sqlite"
        shutil.copyfile(minimal_nsys_db_path, profile_path)
        if suffix:
            with sqlite3.connect(profile_path) as connection:
                connection.execute(
                    "ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL "
                    f"RENAME TO CUPTI_ACTIVITY_KIND_KERNEL{suffix}"
                )

        reference = build_local_profile_reference(profile_path)
        with Profile(str(profile_path), cache_mode="direct") as profile:
            report = EvidenceBuilder(profile).build(only=[])
        assert reference.profile_id == report.profile_id
        equivalent_identities.append(reference.profile_id)

    assert len(set(equivalent_identities)) == 1

    content_identities = []
    for kernel_rows in (1, 2):
        profile_path = tmp_path / f"kernels-v3-{kernel_rows}.sqlite"
        shutil.copyfile(minimal_nsys_db_path, profile_path)
        with sqlite3.connect(profile_path) as connection:
            connection.execute(
                "ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL "
                "RENAME TO CUPTI_ACTIVITY_KIND_KERNEL_V3"
            )
            connection.execute(
                "DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL_V3 WHERE rowid > ?",
                (kernel_rows,),
            )

        reference = build_local_profile_reference(profile_path)
        with Profile(str(profile_path), cache_mode="direct") as profile:
            report = EvidenceBuilder(profile).build(only=[])
        assert reference.profile_id == report.profile_id
        assert reference.kernel_count == kernel_rows
        content_identities.append(reference.profile_id)

    assert content_identities[0] != content_identities[1]


def test_profile_reference_is_stable_across_process_restart(
    minimal_nsys_db_path, tmp_path
):
    profile = Path(minimal_nsys_db_path)
    script = """
import json
import sys
from dataclasses import asdict
from nsys_ai.profile_runner import build_local_profile_reference
print(json.dumps(asdict(build_local_profile_reference(sys.argv[1])), sort_keys=True))
"""
    environment = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = source
    before_names = {item.relative_to(tmp_path) for item in tmp_path.rglob("*")}
    before_digest = _digest(profile)

    outputs = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", script, str(profile)],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(json.loads(result.stdout))

    assert outputs[0] == outputs[1] == asdict(build_local_profile_reference(profile))
    assert _digest(profile) == before_digest
    assert {item.relative_to(tmp_path) for item in tmp_path.rglob("*")} == before_names
