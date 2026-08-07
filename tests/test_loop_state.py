"""loop_state helpers: H100 preset discovery and profile path labeling.

DiffLoopState was removed; SessionStore is the only loop source of truth.
Decision/diff coverage lives in test_web_session.py and test_tui_session.py.
"""

import os
import time

from nsys_ai.loop_state import (
    detect_h100_replay_preset,
    normalize_profile_path,
    profile_display_name,
    same_profile_path,
)


def test_detect_h100_replay_preset_picks_newest_complete_snapshot(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    base = (
        home
        / ".cache"
        / "huggingface"
        / "hub"
        / "datasets--rich7421--fastvideo-wan-h100-sp1-nsys"
        / "snapshots"
    )
    old_snap = base / "old_rev"
    new_snap = base / "new_rev"
    for snap in (old_snap, new_snap):
        (snap / "profiles").mkdir(parents=True)
    (old_snap / "profiles" / "perf_h100_sp1.sqlite").write_text("")
    new_before = new_snap / "profiles" / "perf_h100_sp1.sqlite"
    new_after = new_snap / "profiles" / "perf_h100_sp1_fa3.sqlite"
    new_before.write_text("")
    new_after.write_text("")

    os.utime(old_snap, (time.time() - 100, time.time() - 100))
    os.utime(new_snap, (time.time(), time.time()))

    out = detect_h100_replay_preset()
    assert out is not None
    assert "new_rev" in out["before_path"]
    assert "new_rev" in out["after_path"]


def test_detect_h100_replay_preset(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    snap = (
        home
        / ".cache"
        / "huggingface"
        / "hub"
        / "datasets--rich7421--fastvideo-wan-h100-sp1-nsys"
        / "snapshots"
        / "abc123"
        / "profiles"
    )
    snap.mkdir(parents=True)
    before = snap / "perf_h100_sp1.sqlite"
    after = snap / "perf_h100_sp1_fa3.sqlite"
    before.write_text("")
    after.write_text("")

    out = detect_h100_replay_preset()
    assert out is not None
    assert out["before_path"].endswith("perf_h100_sp1.sqlite")
    assert out["after_path"].endswith("perf_h100_sp1_fa3.sqlite")


def test_normalize_profile_path_rejects_missing(tmp_path):
    missing = tmp_path / "missing.sqlite"
    try:
        normalize_profile_path(str(missing), label="before")
    except FileNotFoundError as exc:
        assert "before not found" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_same_profile_path_symlink(tmp_path):
    target = tmp_path / "profile.sqlite"
    target.write_text("x")
    link = tmp_path / "link.sqlite"
    link.symlink_to(target)
    assert same_profile_path(str(target), str(link))


def test_profile_display_name_maps_h100_blob(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    snap = (
        home
        / ".cache"
        / "huggingface"
        / "hub"
        / "datasets--rich7421--fastvideo-wan-h100-sp1-nsys"
        / "snapshots"
        / "abc123"
        / "profiles"
    )
    snap.mkdir(parents=True)
    blob = (
        home
        / ".cache"
        / "huggingface"
        / "hub"
        / "datasets--rich7421--fastvideo-wan-h100-sp1-nsys"
        / "blobs"
        / ("a" * 40)
    )
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"SQLite format 3\x00")
    before = snap / "perf_h100_sp1.sqlite"
    after = snap / "perf_h100_sp1_fa3.sqlite"
    before.symlink_to(blob)
    after.write_bytes(b"SQLite format 3\x00")
    preset = detect_h100_replay_preset()
    assert preset is not None
    assert profile_display_name(str(blob), preset) == "perf_h100_sp1.sqlite"
    assert profile_display_name(str(after), preset) == "perf_h100_sp1_fa3.sqlite"


def test_normalize_profile_path_keeps_symlink_name(tmp_path):
    blob = tmp_path / "blobs" / "803cf28fff228c523caf78689e65d39b8a33f6555cc677bdf00000000"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"SQLite format 3\x00")
    profiles = tmp_path / "snapshots" / "rev1" / "profiles"
    profiles.mkdir(parents=True)
    link = profiles / "perf_h100_sp1.sqlite"
    link.symlink_to(blob)
    out = normalize_profile_path(str(link), label="before")
    assert out.endswith("perf_h100_sp1.sqlite")
    assert "803cf28" not in out
