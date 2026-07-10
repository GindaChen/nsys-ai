"""Tests for the local baseline snapshot store (nsys_ai.baseline)."""

import json
import sqlite3

import pytest

from nsys_ai import baseline


def _make_profile(path, *, kernels):
    """Create a minimal Nsight-like SQLite export sufficient for Profile()."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE StringIds(id INT PRIMARY KEY, value TEXT)")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
        "start INT, [end] INT, deviceId INT, streamId INT, correlationId INT, "
        "shortName INT, demangledName INT)"
    )
    conn.execute("CREATE TABLE NVTX_EVENTS(text TEXT, globalTid INT, start INT, [end] INT)")
    conn.executemany(
        "INSERT INTO StringIds(id, value) VALUES(?,?)",
        [(1, "kA"), (2, "kA_dem")],
    )
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start, [end], deviceId, streamId, "
        "correlationId, shortName, demangledName) VALUES(?,?,?,?,?,?,?)",
        kernels,
    )
    conn.commit()
    conn.close()


@pytest.fixture
def profile_a(tmp_path):
    p = tmp_path / "before.sqlite"
    _make_profile(p, kernels=[(0, 10_000_000, 0, 7, 1, 1, 2)])
    return p


def test_parse_baseline_ref():
    assert baseline.parse_baseline_ref("baseline:main") == "main"
    assert baseline.parse_baseline_ref("baseline:  v1  ") == "v1"
    assert baseline.parse_baseline_ref("some/path.sqlite") is None
    assert baseline.parse_baseline_ref("baseline:") is None
    assert baseline.parse_baseline_ref(None) is None


def test_tag_round_trip(tmp_path, profile_a):
    root = tmp_path / "store"
    meta = baseline.tag_baseline("v1", str(profile_a), "known good", root=root)

    # meta shape
    assert meta["name"] == "v1"
    assert meta["profile_id"].startswith("nsys1:")
    assert meta["reason"] == "known good"
    assert meta["runspec"] is None
    assert meta["tagger"]
    assert meta["tagged_at"].endswith("Z")
    assert meta["source_path"].endswith("before.sqlite")

    entry = root / "v1"
    snapshot = entry / baseline.SNAPSHOT_FILENAME
    meta_file = entry / baseline.META_FILENAME
    assert snapshot.is_file()
    assert meta_file.is_file()

    # snapshot is a real copy of the source profile
    assert snapshot.read_bytes() == profile_a.read_bytes()

    # meta.json is deterministic: sorted keys, indent 2, trailing newline
    text = meta_file.read_text(encoding="utf-8")
    assert text.endswith("\n")
    on_disk = json.loads(text)
    assert on_disk == meta
    assert text == json.dumps(meta, indent=2, sort_keys=True) + "\n"


def test_tag_records_profile_id_matching_get_profile_id(tmp_path, profile_a):
    from nsys_ai import profile as profile_mod
    from nsys_ai.fingerprint import get_profile_id

    with profile_mod.open(str(profile_a)) as prof:
        expected = get_profile_id(prof.conn, fallback_path=prof.path)

    root = tmp_path / "store"
    meta = baseline.tag_baseline("v1", str(profile_a), "reason", root=root)
    assert meta["profile_id"] == expected


def test_tag_blank_reason_refused(tmp_path, profile_a):
    root = tmp_path / "store"
    with pytest.raises(ValueError):
        baseline.tag_baseline("v1", str(profile_a), "   ", root=root)
    with pytest.raises(ValueError):
        baseline.tag_baseline("v1", str(profile_a), "", root=root)


def test_tag_invalid_name_refused(tmp_path, profile_a):
    root = tmp_path / "store"
    for bad in ["", "  ", "..", ".", "a/b", "../evil", "foo\\bar"]:
        with pytest.raises(ValueError):
            baseline.tag_baseline(bad, str(profile_a), "reason", root=root)


def test_list_and_show(tmp_path, profile_a):
    root = tmp_path / "store"
    assert baseline.list_baselines(root=root) == []

    baseline.tag_baseline("v1", str(profile_a), "first", root=root)
    baseline.tag_baseline("v2", str(profile_a), "second", root=root)

    names = [m["name"] for m in baseline.list_baselines(root=root)]
    assert names == ["v1", "v2"]

    shown = baseline.show_baseline("v1", root=root)
    assert shown["reason"] == "first"


def test_show_unknown_raises(tmp_path):
    root = tmp_path / "store"
    with pytest.raises(ValueError):
        baseline.show_baseline("nope", root=root)


def test_resolve_baseline_ref(tmp_path, profile_a):
    root = tmp_path / "store"
    baseline.tag_baseline("v1", str(profile_a), "reason", root=root)

    # both the prefixed ref and the bare name resolve to the snapshot
    resolved = baseline.resolve_baseline_ref("baseline:v1", root=root)
    assert resolved.endswith(baseline.SNAPSHOT_FILENAME)
    assert baseline.resolve_baseline_ref("v1", root=root) == resolved


def test_resolve_unknown_ref_raises(tmp_path):
    root = tmp_path / "store"
    with pytest.raises(ValueError):
        baseline.resolve_baseline_ref("baseline:missing", root=root)


def test_retag_overwrites(tmp_path, profile_a):
    root = tmp_path / "store"
    baseline.tag_baseline("v1", str(profile_a), "first", root=root)
    meta = baseline.tag_baseline("v1", str(profile_a), "second", root=root)
    assert meta["reason"] == "second"
    assert baseline.show_baseline("v1", root=root)["reason"] == "second"
