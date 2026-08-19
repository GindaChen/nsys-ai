"""Persistent side-summary and pair reconciliation tests for #433."""

from __future__ import annotations

import json
from pathlib import Path

from nsys_ai.diff import ProfileDiffSummary, ProfileSummary
from nsys_ai.diff_index import DiffIndex, _path_signature, _selection


class _FakeProfile:
    def __init__(self, path: Path):
        self.path = str(path)
        self.conn = object()


def _summary(profile: _FakeProfile, profile_id: str, gpu: int | None, trim):
    return ProfileSummary(
        path=profile.path,
        gpu=gpu,
        schema_version="3.25.0",
        product_version="2026.2.1",
        total_gpu_ns=10,
        kernel_rows=1,
        kernels=[],
        nvtx=[],
        overlap={},
        profile_id=profile_id,
        profile_id_is_capture_derived=True,
    )


def _pair(before: ProfileSummary, after: ProfileSummary) -> ProfileDiffSummary:
    return ProfileDiffSummary(
        before=before,
        after=after,
        warnings=[],
        kernel_diffs=[],
        nvtx_diffs=[],
        overlap_before={},
        overlap_after={},
        overlap_delta={},
        top_regressions=[],
        top_improvements=[],
        diff_id=f"{before.profile_id}:{after.profile_id}",
    )


def test_reconcile_reuses_before_and_rebuilds_only_changed_after(tmp_path, monkeypatch):
    before = _FakeProfile(tmp_path / "before.sqlite")
    after = _FakeProfile(tmp_path / "after.sqlite")
    ids = {before.path: "before-v1", after.path: "after-v1"}
    builds: list[str] = []
    diffs: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "nsys_ai.diff_index._profile_identity",
        lambda profile: {"profile_id": ids[profile.path], "capture_derived": True},
    )

    def build(profile, gpu, trim, *, nvtx_limit):
        builds.append(profile.path)
        return _summary(profile, ids[profile.path], gpu, trim)

    def diff(before_profile, after_profile, **kwargs):
        diffs.append((kwargs["before_summary"].profile_id, kwargs["after_summary"].profile_id))
        return _pair(kwargs["before_summary"], kwargs["after_summary"])

    monkeypatch.setattr("nsys_ai.diff_index.build_profile_summary", build)
    monkeypatch.setattr("nsys_ai.diff_index.diff_profiles", diff)
    index = DiffIndex(tmp_path / "session")

    index.reconcile(before, after, gpu=0, trim=(0, 100), limit=5, nvtx_limit=10)
    index.reconcile(before, after, gpu=0, trim=(0, 100), limit=5, nvtx_limit=10)

    assert builds == [before.path, after.path]
    assert diffs == [("before-v1", "after-v1")]
    assert (tmp_path / "session" / "indices" / "pair_summary.json").is_file()

    ids[after.path] = "after-v2"
    index.reconcile(before, after, gpu=0, trim=(0, 100), limit=5, nvtx_limit=10)
    assert builds == [before.path, after.path, after.path]
    assert diffs[-1] == ("before-v1", "after-v2")


def test_corrupt_pair_memo_is_a_cache_miss(tmp_path, monkeypatch):
    before = _FakeProfile(tmp_path / "before.sqlite")
    after = _FakeProfile(tmp_path / "after.sqlite")
    monkeypatch.setattr(
        "nsys_ai.diff_index._profile_identity",
        lambda profile: {"profile_id": Path(profile.path).stem, "capture_derived": True},
    )
    monkeypatch.setattr(
        "nsys_ai.diff_index.build_profile_summary",
        lambda profile, gpu, trim, *, nvtx_limit: _summary(
            profile, Path(profile.path).stem, gpu, trim
        ),
    )
    calls = {"diff": 0}

    def diff(before_profile, after_profile, **kwargs):
        calls["diff"] += 1
        return _pair(kwargs["before_summary"], kwargs["after_summary"])

    monkeypatch.setattr("nsys_ai.diff_index.diff_profiles", diff)
    index = DiffIndex(tmp_path / "session")
    index.reconcile(before, after, gpu=None)
    pair_path = tmp_path / "session" / "indices" / "pair_summary.json"
    pair_path.write_text("{not-json", encoding="utf-8")

    index.reconcile(before, after, gpu=None)

    assert calls["diff"] == 2
    assert json.loads(pair_path.read_text(encoding="utf-8"))["schema_version"] == "diff-pair-v1"


def test_diff_parameters_are_part_of_side_and_pair_keys(tmp_path, monkeypatch):
    before = _FakeProfile(tmp_path / "before.sqlite")
    after = _FakeProfile(tmp_path / "after.sqlite")
    monkeypatch.setattr(
        "nsys_ai.diff_index._profile_identity",
        lambda profile: {"profile_id": Path(profile.path).stem, "capture_derived": True},
    )
    builds: list[tuple[int | None, tuple[int, int] | None, int | None]] = []

    def build(profile, gpu, trim, *, nvtx_limit):
        builds.append((gpu, trim, nvtx_limit))
        return _summary(profile, Path(profile.path).stem, gpu, trim)

    monkeypatch.setattr("nsys_ai.diff_index.build_profile_summary", build)
    monkeypatch.setattr(
        "nsys_ai.diff_index.diff_profiles",
        lambda before_profile, after_profile, **kwargs: _pair(
            kwargs["before_summary"], kwargs["after_summary"]
        ),
    )
    index = DiffIndex(tmp_path / "session")
    index.reconcile(before, after, gpu=0, trim=(0, 100), nvtx_limit=10)
    index.reconcile(before, after, gpu=1, trim=(100, 200), nvtx_limit=20)

    assert builds == [(0, (0, 100), 10), (0, (0, 100), 10), (1, (100, 200), 20), (1, (100, 200), 20)]


def test_parquetdir_signature_changes_when_a_file_is_replaced(tmp_path):
    parquetdir = tmp_path / "profile.parquetdir"
    parquetdir.mkdir()
    parquet = parquetdir / "kernels.parquet"
    parquet.write_bytes(b"before")
    before = _path_signature(str(parquetdir))

    parquet.write_bytes(b"after-content")
    parquet.touch()
    after = _path_signature(str(parquetdir))

    assert before != after


def test_malformed_nested_selection_is_a_recoverable_memo_error():
    import pytest

    with pytest.raises(ValueError, match="selection memo"):
        _selection([])  # type: ignore[arg-type]
