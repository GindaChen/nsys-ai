"""Tests for the nsys_ai.exceptions module.

Validates:
- Exception hierarchy and inheritance
- Machine-readable error codes
- to_dict() structured output for AI agent consumption
- Backward compatibility (SkillNotFoundError inherits KeyError)
- from-chaining preservation
"""

import pytest

from nsys_ai.exceptions import (
    ExportError,
    ExportTimeoutError,
    ExportToolMissingError,
    NsysAiError,
    ProfileError,
    ProfileNotFoundError,
    SchemaError,
    SkillError,
    SkillExecutionError,
    SkillNotFoundError,
)


# ── Hierarchy ────────────────────────────────────────────────────────


class TestHierarchy:
    """Exception inheritance should form a clean tree."""

    def test_base_is_exception(self):
        assert issubclass(NsysAiError, Exception)

    @pytest.mark.parametrize(
        "cls",
        [ProfileError, ExportError, SkillError],
    )
    def test_mid_level_inherits_base(self, cls):
        assert issubclass(cls, NsysAiError)

    @pytest.mark.parametrize(
        "cls,parent",
        [
            (ProfileNotFoundError, ProfileError),
            (SchemaError, ProfileError),
            (ExportTimeoutError, ExportError),
            (ExportToolMissingError, ExportError),
            (SkillNotFoundError, SkillError),
            (SkillExecutionError, SkillError),
        ],
    )
    def test_leaf_inherits_parent(self, cls, parent):
        assert issubclass(cls, parent)
        assert issubclass(cls, NsysAiError)


# ── Backward Compatibility ──────────────────────────────────────────


class TestBackwardCompat:
    """SkillNotFoundError must be caught by `except KeyError`."""

    def test_skill_not_found_is_key_error(self):
        assert issubclass(SkillNotFoundError, KeyError)

    def test_caught_as_key_error(self):
        with pytest.raises(KeyError):
            raise SkillNotFoundError("no such skill")

    def test_caught_as_nsys_ai_error(self):
        with pytest.raises(NsysAiError):
            raise SkillNotFoundError("no such skill")


# ── Error Codes ─────────────────────────────────────────────────────


class TestErrorCodes:
    """Each exception should carry a unique, machine-readable error_code."""

    @pytest.mark.parametrize(
        "cls,expected_code",
        [
            (NsysAiError, "NSYS_AI_ERROR"),
            (ProfileError, "PROFILE_ERROR"),
            (ProfileNotFoundError, "PROFILE_NOT_FOUND"),
            (SchemaError, "SCHEMA_ERROR"),
            (ExportError, "EXPORT_ERROR"),
            (ExportTimeoutError, "EXPORT_TIMEOUT"),
            (ExportToolMissingError, "EXPORT_TOOL_MISSING"),
            (SkillError, "SKILL_ERROR"),
            (SkillNotFoundError, "SKILL_NOT_FOUND"),
            (SkillExecutionError, "SKILL_EXECUTION_ERROR"),
        ],
    )
    def test_default_error_code(self, cls, expected_code):
        exc = cls("test message")
        assert exc.error_code == expected_code


# ── to_dict() ───────────────────────────────────────────────────────


class TestToDict:
    """to_dict() should produce a structured JSON-serializable payload."""

    def test_basic_structure(self):
        exc = NsysAiError("something failed")
        d = exc.to_dict()
        assert "error" in d
        assert d["error"]["code"] == "NSYS_AI_ERROR"
        assert d["error"]["message"] == "something failed"

    def test_profile_not_found_dict(self):
        exc = ProfileNotFoundError("/path/to/missing.sqlite")
        d = exc.to_dict()
        assert d["error"]["code"] == "PROFILE_NOT_FOUND"
        assert "/path/to/missing.sqlite" in d["error"]["message"]

    def test_skill_not_found_with_available(self):
        exc = SkillNotFoundError(
            "Unknown skill 'foo'",
            available=["bar", "baz"],
        )
        d = exc.to_dict()
        assert d["error"]["code"] == "SKILL_NOT_FOUND"
        assert d["error"]["available_skills"] == ["bar", "baz"]

    def test_to_dict_is_json_serializable(self):
        import json

        exc = SchemaError("missing table NVTX_EVENTS")
        payload = json.dumps(exc.to_dict())
        parsed = json.loads(payload)
        assert parsed["error"]["code"] == "SCHEMA_ERROR"


# ── Custom error_code override ──────────────────────────────────────


class TestCustomErrorCode:
    """Users can pass a custom error_code to override the default."""

    def test_custom_code(self):
        exc = ProfileError("custom issue", error_code="CUSTOM_PROFILE_ERR")
        assert exc.error_code == "CUSTOM_PROFILE_ERR"
        assert exc.to_dict()["error"]["code"] == "CUSTOM_PROFILE_ERR"


# ── Exception chaining ──────────────────────────────────────────────


class TestChaining:
    """raise ... from e should preserve the original cause."""

    def test_from_chaining(self):
        original = RuntimeError("root cause")
        wrapped = SchemaError("schema broken") 
        wrapped.__cause__ = original
        assert wrapped.__cause__ is original
        assert str(wrapped.__cause__) == "root cause"

    def test_raise_from(self):
        with pytest.raises(SchemaError) as exc_info:
            try:
                raise RuntimeError("db locked")
            except RuntimeError as e:
                raise SchemaError("cannot read schema") from e
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RuntimeError)


# ── SkillNotFoundError extras ───────────────────────────────────────


class TestSkillNotFoundExtras:
    """SkillNotFoundError should carry an available skills list."""

    def test_available_attribute(self):
        exc = SkillNotFoundError("no such", available=["a", "b"])
        assert exc.available == ["a", "b"]

    def test_available_default(self):
        exc = SkillNotFoundError("no such")
        assert exc.available == []

    def test_available_in_to_dict(self):
        exc = SkillNotFoundError("no such", available=["x"])
        d = exc.to_dict()
        assert "available_skills" in d["error"]
