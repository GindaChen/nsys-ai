"""Crash-safe publication helpers for local JSON artifacts."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any


def atomic_write_bytes(
    path: str | os.PathLike[str], payload: bytes, *, mode: int = 0o600
) -> Path:
    """Publish bytes with flush/fsync followed by an atomic same-directory replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    """Publish deterministic, human-readable JSON without exposing a torn file."""
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return atomic_write_bytes(path, encoded)


def atomic_write_bytes_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    mode: int = 0o600,
) -> None:
    """Atomically publish one file relative to an already-open directory.

    Holding ``directory_fd`` across validation and publication prevents a
    symlinked pathname from being redirected to another directory between the
    two operations. ``name`` is deliberately a single leaf: callers own parent
    traversal and policy checks before invoking this low-level seam.
    """
    if not isinstance(name, str) or not name or name in {".", ".."}:
        raise ValueError("artifact name must be a non-empty file name")
    if os.path.basename(name) != name or os.sep in name or (
        os.altsep is not None and os.altsep in name
    ):
        raise ValueError("artifact name must not contain path separators")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    temporary_name = ""
    for _attempt in range(16):
        candidate = f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            descriptor = os.open(candidate, flags, mode, dir_fd=directory_fd)
        except FileExistsError:
            continue
        temporary_name = candidate
        break
    if descriptor < 0:
        raise FileExistsError("could not allocate a temporary artifact file")

    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(
            temporary_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
