import logging
import sqlite3
import subprocess
from pathlib import Path

import pytest

from nsys_ai import profile as profile_mod
from nsys_ai.exceptions import ExportError, ExportTimeoutError, ExportToolMissingError


def test_resolve_non_nsys_rep_passthrough(tmp_path: Path):
    """Non-.nsys-rep paths should be returned unchanged."""
    path = tmp_path / "foo.sqlite"
    path.write_bytes(b"0")
    result = profile_mod.resolve_profile_path(str(path))
    assert result == str(path)


def test_resolve_parquetdir_passthrough_directory(tmp_path: Path):
    path = tmp_path / "foo.parquetdir"
    path.mkdir()
    (path / "NVTX_EVENTS.parquet").write_bytes(b"x")
    result = profile_mod.resolve_profile_path(str(path), backend="parquetdir")
    assert result == str(path)


def test_resolve_nsys_rep_missing_nsys(monkeypatch, tmp_path: Path):
    """If nsys is not on PATH, resolve_profile_path should raise a clear error."""
    monkeypatch.setattr(profile_mod.shutil, "which", lambda name: None)
    path = tmp_path / "foo.nsys-rep"
    path.write_bytes(b"0")
    with pytest.raises(ExportToolMissingError) as exc:
        profile_mod.resolve_profile_path(str(path))
    assert "requires 'nsys'" in str(exc.value)


def test_resolve_nsys_rep_missing_nsys_reuses_blobless_sqlite(
    monkeypatch, tmp_path: Path, caplog
):
    """Up-to-date .sqlite without payload blobs: reuse when nsys is unavailable."""
    monkeypatch.setattr(profile_mod.shutil, "which", lambda name: None)
    rep = tmp_path / "foo.nsys-rep"
    rep.write_bytes(b"0")
    sqlite_path = tmp_path / "foo.sqlite"
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("CREATE TABLE dummy (id INTEGER)")
        conn.commit()

    with caplog.at_level(logging.WARNING, logger="nsys_ai.profile"):
        out = profile_mod.resolve_profile_path(str(rep))
    assert out == str(sqlite_path)
    assert any("without NVTX payload blobs" in r.message for r in caplog.records)


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


def test_resolve_nsys_rep_parquetdir_success(monkeypatch, tmp_path: Path):
    calls = {}
    monkeypatch.setattr(profile_mod.shutil, "which", lambda name: "/opt/nsys")

    def fake_run(args, check, capture_output, text, timeout):
        calls["args"] = args
        if "-o" in args:
            o_idx = args.index("-o")
            if o_idx + 1 < len(args):
                out_dir = Path(args[o_idx + 1])
                out_dir.mkdir()
                (out_dir / "NVTX_EVENTS.parquet").write_bytes(b"x")

        class Result:
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(profile_mod.subprocess, "run", fake_run)

    rep = tmp_path / "foo.nsys-rep"
    rep.write_bytes(b"0")
    out = profile_mod.resolve_profile_path(str(rep), backend="parquetdir")
    assert out.endswith(".parquetdir")
    args = calls["args"]
    assert "--type=parquetdir" in args
    assert "--include-blobs=true" in args


def test_resolve_nsys_rep_failure(monkeypatch, tmp_path: Path):
    """If nsys export fails, the stderr/stdout should be surfaced in a RuntimeError."""
    monkeypatch.setattr(profile_mod.shutil, "which", lambda name: "/opt/nsys")

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["nsys"], output="", stderr="nsys error")

    monkeypatch.setattr(profile_mod.subprocess, "run", fake_run)

    rep = tmp_path / "foo.nsys-rep"
    rep.write_bytes(b"0")
    with pytest.raises(ExportError) as exc:
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
    with pytest.raises(ExportTimeoutError) as exc:
        profile_mod.resolve_profile_path(str(rep))
    assert "timed out after 300 seconds" in str(exc.value)


def test_sqlite_blob_reexport_check_uses_schema_not_row_values(tmp_path: Path):
    sqlite_path = tmp_path / "blob_schema_only.sqlite"
    with sqlite3.connect(sqlite_path) as conn:
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE NVTX_EVENTS (
                start INTEGER,
                end INTEGER,
                binaryData BLOB
            );
            CREATE TABLE NVTX_PAYLOAD_SCHEMAS (
                domainId INTEGER,
                schemaId INTEGER
            );
            INSERT INTO NVTX_EVENTS (start, end, binaryData) VALUES (1, 2, NULL);
        """)
        conn.commit()

    assert profile_mod._sqlite_needs_blob_reexport(str(sqlite_path)) is False
