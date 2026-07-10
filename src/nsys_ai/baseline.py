"""
baseline.py - Local store of named profile snapshots.

A persisted diff verdict is a record; a *reproducible* verdict needs a named
snapshot to compare against. This module keeps a local, CI-friendly store of
tagged ``.sqlite`` profiles so a CI invocation like
``diff --against baseline:main`` resolves a stable name to a known snapshot
instead of a fragile file path.

Layout (relative to the current working directory by default)::

    .nsys-ai-baselines/
        <name>/
            snapshot.sqlite   # self-contained copy of the resolved profile
            meta.json         # deterministic identity + provenance record

``meta.json`` carries ``runspec: null`` as a forward-compatible placeholder.
RunSpec recording (issue #25) does not exist yet and is only needed for later
"compatible setup" checks, not for resolving a baseline snapshot, so it is
deliberately deferred here rather than implemented.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from . import profile as _profile
from .diff_decision import _utc_now_iso, resolve_decider_identity
from .fingerprint import get_profile_id

DEFAULT_ROOT = ".nsys-ai-baselines"
"""Directory (relative to CWD) that holds the local baseline store."""

_REF_PREFIX = "baseline:"

# Names become directory names, so keep them to a safe identifier alphabet and
# reject the current/parent directory tokens to avoid any path traversal.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

SNAPSHOT_FILENAME = "snapshot.sqlite"
META_FILENAME = "meta.json"


def _store_root(root: str | os.PathLike[str] | None) -> Path:
    return Path(root) if root is not None else Path(DEFAULT_ROOT)


def _validate_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("baseline name must be a string")
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("baseline name must not be empty")
    if cleaned in (".", ".."):
        raise ValueError(f"invalid baseline name: {name!r}")
    if not _NAME_RE.match(cleaned):
        raise ValueError(
            "baseline name may only contain letters, digits, '.', '_' and '-' "
            f"and must start alphanumerically: {name!r}"
        )
    return cleaned


def parse_baseline_ref(s: str | None) -> str | None:
    """Return the ``<name>`` from a ``baseline:<name>`` token, else None.

    Only the ``baseline:`` prefix form is recognised as a baseline reference;
    any other string (a file path, a bare name) returns None so callers can
    treat it as an ordinary profile path.
    """
    if not isinstance(s, str):
        return None
    if not s.startswith(_REF_PREFIX):
        return None
    name = s[len(_REF_PREFIX) :].strip()
    return name or None


def tag_baseline(
    name: str,
    profile_path: str | os.PathLike[str],
    reason: str,
    root: str | os.PathLike[str] | None = None,
) -> dict:
    """Record a tagged snapshot of *profile_path* under *name*.

    Opens the profile through the existing loader (resolving ``.nsys-rep`` to
    its ``.sqlite`` sidecar), copies that resolved ``.sqlite`` into the store so
    the baseline stays valid even if the original file moves, and writes a
    deterministic ``meta.json``. Returns the metadata mapping.
    """
    clean_name = _validate_name(name)

    clean_reason = reason.strip() if isinstance(reason, str) else ""
    if not clean_reason:
        raise ValueError("a non-empty --reason is required to tag a baseline")

    root_path = _store_root(root)
    entry_dir = root_path / clean_name

    with _profile.open(os.fspath(profile_path)) as prof:
        profile_id = get_profile_id(prof.conn, fallback_path=prof.path)
        source_path = os.path.abspath(prof.path)

        entry_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, entry_dir / SNAPSHOT_FILENAME)

    meta = {
        "name": clean_name,
        "profile_id": profile_id,
        "source_path": source_path,
        "reason": clean_reason,
        "tagged_at": _utc_now_iso(),
        "tagger": resolve_decider_identity(),
        # Forward-compatible placeholder; a future RunSpec recording (#25) can
        # populate this without changing the on-disk layout.
        "runspec": None,
    }
    _write_meta(entry_dir / META_FILENAME, meta)
    return meta


def _write_meta(path: str | os.PathLike[str], meta: dict) -> None:
    text = json.dumps(meta, indent=2, sort_keys=True) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _read_meta(entry_dir: Path) -> dict:
    with open(entry_dir / META_FILENAME, encoding="utf-8") as f:
        return json.load(f)


def list_baselines(root: str | os.PathLike[str] | None = None) -> list[dict]:
    """Return metadata for every tagged baseline, sorted by name."""
    root_path = _store_root(root)
    if not root_path.is_dir():
        return []
    entries: list[dict] = []
    for child in sorted(root_path.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if not (child / META_FILENAME).is_file():
            continue
        try:
            entries.append(_read_meta(child))
        except (OSError, ValueError):
            # Skip corrupt entries rather than failing the whole listing.
            continue
    return entries


def show_baseline(name: str, root: str | os.PathLike[str] | None = None) -> dict:
    """Return the metadata for a single tagged baseline.

    Raises ValueError if no baseline with *name* exists.
    """
    clean_name = _validate_name(name)
    entry_dir = _store_root(root) / clean_name
    if not (entry_dir / META_FILENAME).is_file():
        raise ValueError(f"unknown baseline: {clean_name!r}")
    return _read_meta(entry_dir)


def resolve_baseline_ref(
    ref: str, root: str | os.PathLike[str] | None = None
) -> str:
    """Resolve a ``baseline:<name>`` reference (or bare name) to a snapshot path.

    Returns the filesystem path of the stored ``.sqlite`` snapshot, ready to
    hand to the profile loader. Raises ValueError if the name is unknown.
    """
    name = parse_baseline_ref(ref)
    if name is None:
        name = ref
    clean_name = _validate_name(name)
    entry_dir = _store_root(root) / clean_name
    snapshot = entry_dir / SNAPSHOT_FILENAME
    if not snapshot.is_file():
        raise ValueError(
            f"unknown baseline: {clean_name!r} "
            f"(no snapshot at {snapshot}); tag one with 'nsys-ai baseline tag'"
        )
    return str(snapshot)
