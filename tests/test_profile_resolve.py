import os
import subprocess
from pathlib import Path

import pytest

from nsys_tui import profile as profile_mod


def test_resolve_non_nsys_rep_passthrough(tmp_path: Path):
    """Non-.nsys-rep paths should be returned unchanged."""
    path = tmp_path / "foo.sqlite"
    path.write_bytes(b"0")
    result = profile_mod.resolve_profile_path(str(path))
    assert result == str(path)


def test_resolve_nsys_rep_missing_nsys(monkeypatch, tmp_path: Path):
    """If nsys is not on PATH, resolve_profile_path should raise a clear error."""
    monkeypatch.setattr(profile_mod.shutil, "which", lambda name: None)
    path = tmp_path / "foo.nsys-rep"
    path.write_bytes(b"0")
    with pytest.raises(RuntimeError) as exc:
        profile_mod.resolve_profile_path(str(path))
    assert "requires 'nsys'" in str(exc.value)


def test_resolve_nsys_rep_success(monkeypatch, tmp_path: Path):
    """Happy path: nsys is found and subprocess.run succeeds, returning .sqlite path."""
    calls = {}

    monkeypatch.setattr(profile_mod.shutil, "which", lambda name: "/opt/nsys")

    def fake_run(args, check, capture_output, text, timeout):
        calls["args"] = args
        # Simulate nsys producing the output file so postcondition passes.
        if "-o" in args:
            o_idx = args.index("-o")
            if o_idx + 1 < len(args):
                Path(args[o_idx + 1]).write_bytes(b"x")
        class Result:
            stdout = ""
            stderr = ""
        return Result()

    monkeypatch.setattr(profile_mod.subprocess, "run", fake_run)

    rep = tmp_path / "foo.nsys-rep"
    rep.write_bytes(b"0")
    out = profile_mod.resolve_profile_path(str(rep))
    assert out.endswith(".sqlite")
    args = calls["args"]
    assert args[0] == "/opt/nsys"
    assert any("overwrite" in a for a in args)
    assert "-o" in args
    o_idx = args.index("-o")
    assert o_idx + 1 < len(args)
    assert args[o_idx + 1] == out
    assert str(rep) in args


def test_resolve_nsys_rep_failure(monkeypatch, tmp_path: Path):
    """If nsys export fails, the stderr/stdout should be surfaced in a RuntimeError."""
    monkeypatch.setattr(profile_mod.shutil, "which", lambda name: "/opt/nsys")

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1, ["nsys"], output="", stderr="nsys error"
        )

    monkeypatch.setattr(profile_mod.subprocess, "run", fake_run)

    rep = tmp_path / "foo.nsys-rep"
    rep.write_bytes(b"0")
    with pytest.raises(RuntimeError) as exc:
        profile_mod.resolve_profile_path(str(rep))
    assert "nsys export failed" in str(exc.value)
    assert "nsys error" in str(exc.value)


def test_resolve_nsys_rep_timeout(monkeypatch, tmp_path: Path):
    """If nsys export hangs past timeout, raise a clear RuntimeError."""
    monkeypatch.setattr(profile_mod.shutil, "which", lambda name: "/opt/nsys")

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["nsys"], timeout=300)

    monkeypatch.setattr(profile_mod.subprocess, "run", fake_run)

    rep = tmp_path / "foo.nsys-rep"
    rep.write_bytes(b"0")
    with pytest.raises(RuntimeError) as exc:
        profile_mod.resolve_profile_path(str(rep))
    assert "timed out after 300 seconds" in str(exc.value)

