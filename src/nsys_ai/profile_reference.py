"""Strict local profile references shared by runners and session storage."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .fingerprint import PROFILE_ID_VERSION

LOCAL_PROFILE_ID_PATTERN = re.compile(
    rf"{re.escape(PROFILE_ID_VERSION)}:sha256:[0-9a-f]{{64}}"
)
StorageKind = Literal["nsys-rep", "parquetdir", "sqlite"]


@dataclass(frozen=True)
class LocalProfileReference:
    """Validated identity and schema metadata for one local profile input."""

    path: str
    profile_id: str
    schema_version: str | None
    product_version: str | None
    kernel_count: int
    storage_kind: StorageKind = "sqlite"
    resolved_path: str | None = None

    def __post_init__(self) -> None:
        if self.resolved_path is None:
            object.__setattr__(self, "resolved_path", self.path)


def profile_stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
    """Return the fields used to detect a profile path or inode swap."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_profile_parent_chain(
    profile_path: Path,
) -> tuple[int, tuple[tuple[int, ...], ...]]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    signatures: list[tuple[int, ...]] = []
    try:
        descriptor = os.open(profile_path.anchor, flags)
        signatures.append(profile_stat_signature(os.fstat(descriptor)))
        for component in profile_path.parts[1:-1]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            signatures.append(profile_stat_signature(os.fstat(descriptor)))
        return descriptor, tuple(signatures)
    except (OSError, ValueError):
        if descriptor >= 0:
            os.close(descriptor)
        raise ValueError(
            "local profile reference file parent cannot be inspected safely"
        ) from None


def _confirm_local_profile_leaf_is_missing(
    profile_path: Path,
    raw_path: str,
) -> None:
    if not profile_path.is_absolute() or os.path.normpath(raw_path) != raw_path:
        raise ValueError("local profile reference path must be canonical")

    first_descriptor = -1
    second_descriptor = -1
    try:
        first_descriptor, first_signatures = _open_profile_parent_chain(profile_path)
        try:
            os.stat(
                profile_path.name,
                dir_fd=first_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise ValueError(
                "local profile reference file changed while being inspected"
            )

        second_descriptor, second_signatures = _open_profile_parent_chain(profile_path)
        if first_signatures != second_signatures:
            raise ValueError(
                "local profile reference file changed while being inspected"
            )
        try:
            os.stat(
                profile_path.name,
                dir_fd=second_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        raise ValueError("local profile reference file changed while being inspected")
    except ValueError:
        raise
    except OSError:
        raise ValueError(
            "local profile reference file cannot be inspected"
        ) from None
    finally:
        if first_descriptor >= 0:
            os.close(first_descriptor)
        if second_descriptor >= 0:
            os.close(second_descriptor)


def inspect_local_profile_file(
    path: str | Path,
    *,
    allow_missing: bool = False,
) -> tuple[Path, os.stat_result] | None:
    """Inspect one canonical, non-symlinked, non-empty regular profile file.

    ``allow_missing`` permits only a path absent at the initial no-follow stat.
    Once an inode is observed, every later inspection failure is treated as a
    path mutation or unsafe file instead of being downgraded to absence.
    """
    raw_path = os.fspath(path)
    profile_path = Path(raw_path)
    try:
        before = os.stat(profile_path, follow_symlinks=False)
    except FileNotFoundError:
        if allow_missing:
            _confirm_local_profile_leaf_is_missing(profile_path, raw_path)
            return None
        raise ValueError("local profile reference file does not exist") from None
    except (OSError, ValueError):
        raise ValueError("local profile reference file cannot be inspected") from None

    try:
        resolved = profile_path.resolve(strict=True)
        after = os.stat(profile_path, follow_symlinks=False)
    except FileNotFoundError:
        raise ValueError("local profile reference file changed while being inspected") from None
    except (OSError, RuntimeError, ValueError):
        raise ValueError("local profile reference file cannot be inspected") from None

    if profile_stat_signature(before) != profile_stat_signature(after):
        raise ValueError("local profile reference file changed while being inspected")
    if resolved != profile_path or stat.S_ISLNK(after.st_mode):
        raise ValueError(
            "local profile reference path must be canonical and contain no symbolic links"
        )
    if not stat.S_ISREG(after.st_mode):
        raise ValueError("local profile reference path is not a regular file")
    if after.st_size <= 0:
        raise ValueError("local profile reference file is empty")
    return resolved, after


def validate_local_profile_reference(
    reference: LocalProfileReference,
    *,
    require_file: bool,
) -> LocalProfileReference:
    """Validate the single persisted contract for a local profile reference."""
    if not isinstance(reference, LocalProfileReference):
        raise TypeError("profile reference must be a LocalProfileReference")

    path = reference.path
    if (
        not isinstance(path, str)
        or not path
        or "\x00" in path
        or not Path(path).is_absolute()
    ):
        raise ValueError("local profile reference path must be an absolute path string")
    storage_kind = reference.storage_kind
    if storage_kind not in {"nsys-rep", "parquetdir", "sqlite"}:
        raise ValueError("local profile reference storage kind is invalid")
    if storage_kind == "sqlite" and Path(path).suffix.lower() != ".sqlite":
        raise ValueError("sqlite profile reference path must name a .sqlite file")
    if storage_kind == "nsys-rep" and Path(path).suffix.lower() != ".nsys-rep":
        raise ValueError("nsys-rep profile reference path must name a .nsys-rep file")
    if storage_kind == "parquetdir" and not Path(path).is_dir():
        raise ValueError("parquetdir profile reference path must name a directory")

    if storage_kind == "parquetdir":
        inspect_local_parquetdir(path, allow_missing=not require_file)
    else:
        inspect_local_profile_file(path, allow_missing=not require_file)

    resolved_path = reference.resolved_path or path
    if (
        not isinstance(resolved_path, str)
        or not resolved_path
        or "\x00" in resolved_path
        or not Path(resolved_path).is_absolute()
        or os.path.normpath(resolved_path) != resolved_path
    ):
        raise ValueError("local profile resolved_path must be a canonical absolute path")
    if storage_kind == "parquetdir" or resolved_path.lower().endswith(".parquetdir"):
        inspect_local_parquetdir(resolved_path, allow_missing=not require_file)
    else:
        inspect_local_profile_file(resolved_path, allow_missing=not require_file)

    if not isinstance(reference.profile_id, str) or not LOCAL_PROFILE_ID_PATTERN.fullmatch(
        reference.profile_id
    ):
        raise ValueError("local profile identity is invalid")
    for name, value in (
        ("schema_version", reference.schema_version),
        ("product_version", reference.product_version),
    ):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(
                f"local profile reference {name} must be a non-empty string or null"
            )
    if isinstance(reference.kernel_count, bool) or not isinstance(
        reference.kernel_count, int
    ):
        raise ValueError("local profile kernel count is invalid")
    if reference.kernel_count <= 0:
        raise ValueError("local profile contains no GPU kernel activity")
    return reference


def inspect_local_parquetdir(path: str, *, allow_missing: bool) -> Path | None:
    """Validate a canonical parquet directory without following symlinks."""
    parquet_path = Path(path)
    if not parquet_path.is_absolute() or os.path.normpath(path) != path:
        raise ValueError("local profile parquetdir path must be canonical")
    try:
        if not parquet_path.exists():
            if allow_missing:
                return None
            raise ValueError("local profile parquetdir does not exist")
        resolved = parquet_path.resolve(strict=True)
        if resolved != parquet_path or not parquet_path.is_dir():
            raise ValueError("local profile parquetdir must be a canonical directory")
        parquet_files = [
            item for item in parquet_path.iterdir() if item.suffix.lower() == ".parquet"
        ]
        if any(item.is_symlink() for item in parquet_files):
            raise ValueError("local profile parquetdir cannot contain symlink parquet files")
        if not parquet_files:
            raise ValueError("local profile parquetdir contains no parquet files")
        return resolved
    except (OSError, RuntimeError):
        raise ValueError("local profile parquetdir cannot be inspected") from None
