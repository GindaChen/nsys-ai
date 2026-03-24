"""
exceptions.py — Custom exception hierarchy for nsys-ai.

All exceptions carry a machine-readable ``error_code`` and a ``.to_dict()``
method that produces the same ``{"error": {"code": …, "message": …}}`` JSON
shape used throughout the tool-dispatch and agent layers.  This lets external
AI agents switch on error codes instead of parsing free-text messages.
"""

from __future__ import annotations


class NsysAiError(Exception):
    """Base exception for all nsys-ai errors.

    Attributes:
        error_code: Machine-readable error code (e.g. ``"PROFILE_NOT_FOUND"``).
    """

    error_code: str = "NSYS_AI_ERROR"

    def __init__(self, message: str = "", *, error_code: str | None = None):
        if error_code is not None:
            self.error_code = error_code
        super().__init__(message)

    def to_dict(self) -> dict:
        """Structured error payload for JSON serialization / AI agent consumption."""
        return {"error": {"code": self.error_code, "message": str(self)}}


# ── Profile errors ─────────────────────────────────────────────────────

class ProfileError(NsysAiError):
    """Errors related to opening or querying profile databases."""

    error_code = "PROFILE_ERROR"


class ProfileNotFoundError(ProfileError):
    """Profile file does not exist or cannot be opened."""

    error_code = "PROFILE_NOT_FOUND"


class SchemaError(ProfileError):
    """Profile has an unexpected schema (missing tables, columns, etc.).

    Common cause: the profile was captured without CUDA kernel tracing or
    was exported from a schema layout this version of nsys-ai does not
    yet understand.
    """

    error_code = "SCHEMA_ERROR"


# ── Export errors ──────────────────────────────────────────────────────

class ExportError(NsysAiError):
    """Errors during .nsys-rep → .sqlite conversion (nsys export)."""

    error_code = "EXPORT_ERROR"


class ExportTimeoutError(ExportError):
    """``nsys export`` timed out."""

    error_code = "EXPORT_TIMEOUT"


class ExportToolMissingError(ExportError):
    """``nsys`` command-line tool is not on PATH."""

    error_code = "EXPORT_TOOL_MISSING"


# ── Skill errors ───────────────────────────────────────────────────────

class SkillError(NsysAiError):
    """Errors during skill execution."""

    error_code = "SKILL_ERROR"


class SkillNotFoundError(SkillError, KeyError):
    """Requested skill does not exist.

    Inherits ``KeyError`` for backward compatibility with existing
    ``except KeyError:`` handlers.
    """

    error_code = "SKILL_NOT_FOUND"

    def __init__(self, message: str = "", *, available: list[str] | None = None):
        self.available = available or []
        super().__init__(message, error_code=self.error_code)

    def __str__(self) -> str:
        # Override KeyError's default repr-like quoting
        return str(self.args[0]) if self.args else ""

    def to_dict(self) -> dict:
        d = super().to_dict()
        if self.available:
            d["error"]["available_skills"] = self.available
        return d


class SkillExecutionError(SkillError):
    """A skill failed during SQL execution or Python-level processing."""

    error_code = "SKILL_EXECUTION_ERROR"

    def __init__(self, message: str = "", *, skill_name: str = ""):
        self.skill_name = skill_name
        super().__init__(message, error_code=self.error_code)

    def to_dict(self) -> dict:
        d = super().to_dict()
        if self.skill_name:
            d["error"]["skill"] = self.skill_name
        return d
