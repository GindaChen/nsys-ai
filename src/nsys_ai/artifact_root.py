"""Resolve the local root for invocation-owned nsys-ai artifacts.

Input-keyed caches deliberately live beside their source profile.  This module
only owns outputs of an invocation: sessions, locks, profile captures, and
default decision records.  Keeping the distinction here prevents a shared
artifact directory from accidentally changing cache identity or invalidation.

The default is intentionally unchanged.  Set ``NSYS_AI_ARTIFACT_ROOT`` for a
CI job or another caller that must keep the working directory clean::

    NSYS_AI_ARTIFACT_ROOT=/tmp/nsys-artifacts nsys-ai diagnose profile.sqlite

Relative roots are resolved against the command's working directory when one
is supplied, which keeps ``run_profile_command(cwd=...)`` deterministic in
tests and embedding callers.
"""

from __future__ import annotations

import os
from pathlib import Path

ARTIFACT_ROOT_ENV_VAR = "NSYS_AI_ARTIFACT_ROOT"
DEFAULT_ARTIFACT_ROOT = ".nsys-ai"
DEFAULT_SESSION_ROOT = f"{DEFAULT_ARTIFACT_ROOT}/sessions"


def _base_directory(cwd: str | os.PathLike[str] | None = None) -> Path:
    return Path.cwd() if cwd is None else Path(cwd).expanduser()


def artifact_root(
    root: str | os.PathLike[str] | None = None,
    *,
    cwd: str | os.PathLike[str] | None = None,
) -> Path:
    """Return an absolute invocation-artifact root.

    An explicit *root* wins over the environment.  ``None`` means read
    ``NSYS_AI_ARTIFACT_ROOT`` and then use the historical ``.nsys-ai`` root.
    The path is not created by this resolver.
    """
    selected = root
    if selected is None:
        selected = os.environ.get(ARTIFACT_ROOT_ENV_VAR) or DEFAULT_ARTIFACT_ROOT
    path = Path(selected).expanduser()
    if not path.is_absolute():
        path = _base_directory(cwd) / path
    return path.resolve(strict=False)


def session_root(
    root: str | os.PathLike[str] | None = None,
    *,
    cwd: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the SessionStore root under the configured artifact root."""
    if root is None or _is_historical_session_root(root):
        return artifact_root(cwd=cwd) / "sessions"
    path = Path(root).expanduser()
    if not path.is_absolute():
        path = _base_directory(cwd) / path
    return path.resolve(strict=False)


def profile_root(*, cwd: str | os.PathLike[str] | None = None) -> Path:
    """Return the default directory for ``nsys-ai profile`` captures."""
    return artifact_root(cwd=cwd) / "profiles"


def default_decision_path(*, cwd: str | os.PathLike[str] | None = None) -> Path:
    """Return the default standalone decision path.

    With no environment override the historical ``./diff.json`` remains the
    default.  Once an artifact root is explicitly configured, the decision is
    placed under ``<root>/decisions`` with the other invocation outputs.
    An explicit ``--decision-out`` is handled by the caller and always wins.
    """
    configured = os.environ.get(ARTIFACT_ROOT_ENV_VAR)
    if configured and configured.strip():
        return artifact_root(cwd=cwd) / "decisions" / "diff.json"
    if cwd is None:
        # Preserve the historical spelling in CLI output and error messages;
        # the process still writes this relative path in its current directory.
        return Path("diff.json")
    return _base_directory(cwd).resolve() / "diff.json"


def _is_historical_session_root(root: str | os.PathLike[str]) -> bool:
    try:
        candidate = Path(root).expanduser()
    except TypeError:
        return False
    return candidate == Path(DEFAULT_SESSION_ROOT)


__all__ = [
    "ARTIFACT_ROOT_ENV_VAR",
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_SESSION_ROOT",
    "artifact_root",
    "default_decision_path",
    "profile_root",
    "session_root",
]
