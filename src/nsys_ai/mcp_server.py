"""Optional read-only MCP transport for the canonical analysis contracts.

The module itself has no MCP import at import time.  Installing the ``mcp``
extra enables the ``nsys-ai-mcp`` entry point; importing the core package does
not acquire a new runtime dependency.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .annotation import PRODUCER, SCHEMA_VERSION
from .connection import DB_ERRORS
from .diff import diff_profiles, diff_profiles_all_gpus
from .diff_render import to_diff_dict
from .exceptions import (
    ExportToolMissingError,
    NsysAiError,
    ProfileNotFoundError,
    SkillNotFoundError,
    SkillParameterError,
)
from .fingerprint import get_profile_id
from .profile import Profile

MAX_ROWS = 50
MAX_DIFF_LIMIT = 50
MAX_SKILL_PARAMS = 32
MAX_PARAM_INT = 10**15
MAX_PARAM_FLOAT = 10**18
MAX_PARAM_STRING_CHARS = 4096
# The SQL tool's 8,000-character cap is tuned for an LLM prompt. MCP returns
# the canonical structured diff envelope, so it needs a larger bounded budget
# while retaining an explicit payload guardrail.
MCP_MAX_PAYLOAD_CHARS = 100_000
_TRIM_KEYS = frozenset({"trim_start_ns", "trim_end_ns"})
_READ_ERRORS = DB_ERRORS + (OSError, ValueError)


def _error(code: str, message: str, **detail: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"error": {"code": code, "message": message}}
    payload["error"].update(detail)
    return payload


def _json_safe(value: Any) -> Any:
    """Normalize database values to the JSON values MCP clients receive."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _catalog_entry(skill) -> dict[str, Any]:
    return {
        "name": skill.name,
        "title": skill.title,
        "description": skill.description,
        "category": skill.category,
        "params": [
            {
                "name": param.name,
                "type": param.type,
                "description": getattr(param, "description", ""),
                "required": param.required,
                "default": param.default,
            }
            for param in skill.params
        ],
    }


def _list_skills() -> dict[str, Any]:
    """Return the same skill metadata exposed by ``skill list --format json``."""
    from .skills.registry import all_skills

    return {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "skills": [_catalog_entry(skill) for skill in all_skills()],
    }


def _get_session(session: str) -> dict[str, Any]:
    """Read the canonical SessionStore handoff without mutating it."""
    from .session_cli import session_payload

    try:
        return _json_safe(session_payload(session))
    except NsysAiError as exc:
        return exc.to_dict()
    except (OSError, TypeError, ValueError) as exc:
        return _error("SESSION_READ_ERROR", str(exc))


def _coerce_param(value: Any, param_type: Any) -> Any:
    """Validate the JSON value against the type declared by a SkillParam."""
    type_name = str(param_type).lower()
    if param_type is int or type_name in {"int", "integer"}:
        if isinstance(value, bool):
            raise ValueError("boolean is not a valid integer parameter")
        parsed = int(value)
        if abs(parsed) > MAX_PARAM_INT:
            raise ValueError(f"integer parameter exceeds the {MAX_PARAM_INT} bound")
        return parsed
    if param_type is float or type_name in {"float", "double"}:
        if isinstance(value, bool):
            raise ValueError("boolean is not a valid float parameter")
        parsed = float(value)
        if not math.isfinite(parsed) or abs(parsed) > MAX_PARAM_FLOAT:
            raise ValueError(f"float parameter must be finite and within ±{MAX_PARAM_FLOAT}")
        return parsed
    if param_type is bool or type_name in {"bool", "boolean"}:
        if not isinstance(value, bool):
            raise ValueError("parameter must be a boolean")
        return value
    if not isinstance(value, str):
        raise ValueError("parameter must be a string")
    if len(value) > MAX_PARAM_STRING_CHARS:
        raise ValueError(
            f"string parameter exceeds the {MAX_PARAM_STRING_CHARS}-character bound"
        )
    return value


def _validate_param_bound(name: str, value: Any, param_type: Any) -> Any:
    """Apply the same safety limits to declared defaults and user parameters."""
    parsed = _coerce_param(value, param_type)
    if name == "limit" and parsed > MAX_ROWS:
        raise ValueError(f"limit must not exceed {MAX_ROWS}")
    return parsed


def _resolve_skill_params(skill, params: Mapping[str, Any] | None) -> dict[str, Any]:
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise ValueError("params must be an object")
    if len(params) > MAX_SKILL_PARAMS:
        raise ValueError(f"params must contain at most {MAX_SKILL_PARAMS} entries")
    if any(not isinstance(name, str) for name in params):
        raise ValueError("all parameter names must be strings")

    declared = {param.name: param for param in skill.params}
    allowed = set(declared) | _TRIM_KEYS
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(
            f"unknown parameter(s) for skill '{skill.name}': {', '.join(unknown)}"
        )

    resolved: dict[str, Any] = {}
    for name, value in params.items():
        if name in _TRIM_KEYS:
            if isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
            resolved[name] = _validate_param_bound(name, value, int)
        else:
            resolved[name] = _validate_param_bound(name, value, declared[name].type)

    # Skill.execute applies defaults internally. Copying safe defaults into the
    # call makes the bound apply before SQL/Python execution as well, including
    # a builtin whose declared default limit is larger than MAX_ROWS.
    for param in skill.params:
        if param.name not in resolved and param.default is not None:
            resolved[param.name] = _validate_param_bound(
                param.name, param.default, param.type
            )
    return resolved


def _cap_rows(rows: list[dict], max_rows: int) -> tuple[list[dict], bool]:
    if isinstance(max_rows, bool) or not isinstance(max_rows, int):
        raise ValueError("max_rows must be an integer")
    if max_rows < 0 or max_rows > MAX_ROWS:
        raise ValueError(f"max_rows must be between 0 and {MAX_ROWS}")
    if len(rows) <= max_rows:
        return rows, False
    capped = list(rows[:max_rows])
    capped.append(
        {
            "_truncated": True,
            "_total_rows": len(rows),
            "_shown_rows": max_rows,
        }
    )
    return capped, True


@contextmanager
def _open_readonly(path: str) -> Iterator[tuple[Any, str]]:
    """Yield a read-only connection without exporting or building a sidecar."""
    from .parquet_cache import open_direct_sqlite, open_with_direct_fallback

    source = Path(path)
    if not source.exists():
        raise ProfileNotFoundError(f"profile not found: {path}")
    if source.suffix.lower() == ".nsys-rep":
        sidecar = source.with_suffix(".sqlite")
        if not sidecar.is_file() or sidecar.stat().st_size == 0:
            raise ExportToolMissingError(
                "MCP access to .nsys-rep is read-only and will not export a sidecar; "
                "provide an up-to-date .sqlite export next to the capture"
            )
        if sidecar.stat().st_mtime < source.stat().st_mtime:
            raise ExportToolMissingError(
                "MCP access to .nsys-rep will not refresh a stale sidecar; "
                "export an up-to-date .sqlite file before calling the server"
            )
        resolved = str(sidecar)
    else:
        resolved = str(source)
    conn, _error_detail = open_with_direct_fallback(resolved, open_direct_sqlite)
    if conn is None:
        uri = Path(resolved).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
    try:
        yield conn, resolved
    finally:
        conn.close()


def _run_skill(
    profile: str,
    skill_name: str,
    params: Mapping[str, Any] | None = None,
    max_rows: int = MAX_ROWS,
) -> dict[str, Any]:
    """Run one registered skill and return rows plus canonical Finding evidence."""
    from .evidence_builder import _invoke_to_findings
    from .skills.base import is_abstention
    from .skills.registry import all_skills, get_skill

    try:
        _cap_rows([], max_rows)
    except ValueError as exc:
        return _error("INVALID_PARAMETER", str(exc))

    skill = get_skill(skill_name)
    if skill is None:
        return SkillNotFoundError(
            f"Unknown skill '{skill_name}'",
            available=[item.name for item in all_skills()],
        ).to_dict()

    try:
        skill_params = _resolve_skill_params(skill, params)
    except (TypeError, ValueError, OverflowError) as exc:
        return _error("INVALID_PARAMETER", str(exc), skill=skill.name)

    try:
        with _open_readonly(profile) as (conn, resolved):
            rows = skill.execute(conn, **skill_params)
            profile_id = get_profile_id(conn, fallback_path=resolved)
            findings = (
                _invoke_to_findings(
                    skill.to_findings_fn,
                    rows,
                    {"profile_id": profile_id},
                )
                if skill.to_findings_fn
                else []
            )
            capped_rows, truncated = _cap_rows(rows, max_rows)
            capped_findings, findings_truncated = _cap_rows(
                [finding.to_dict() for finding in findings], max_rows
            )
            payload = {
                "schema_version": SCHEMA_VERSION,
                "producer": PRODUCER,
                "skill": _catalog_entry(skill),
                "profile": {"path": resolved, "profile_id": profile_id},
                "rows": capped_rows,
                "findings": capped_findings,
                "abstained": is_abstention(rows),
                "truncated": truncated or findings_truncated,
            }
            encoded = json.dumps(payload, ensure_ascii=False, default=str)
            if len(encoded) > MCP_MAX_PAYLOAD_CHARS:
                return _error(
                    "PAYLOAD_TOO_LARGE",
                    "skill result exceeds the MCP payload cap; lower max_rows or narrow the skill parameters",
                    max_payload_chars=MCP_MAX_PAYLOAD_CHARS,
                )
            return _json_safe(payload)
    except SkillParameterError as exc:
        return exc.to_dict()
    except NsysAiError as exc:
        return exc.to_dict()
    except _READ_ERRORS as exc:
        return _error("SKILL_EXECUTION_ERROR", str(exc), skill=skill.name)
    except Exception as exc:
        return _error("SKILL_EXECUTION_ERROR", str(exc), skill=skill.name)


def _diff_payload(
    before: str,
    after: str,
    gpu: int | None = None,
    trim_start_ns: int | None = None,
    trim_end_ns: int | None = None,
    limit: int = 15,
    sort: str = "delta",
) -> dict[str, Any]:
    """Return the canonical ``diff.json`` payload without persisting it."""
    if (trim_start_ns is None) != (trim_end_ns is None):
        return _error("INVALID_PARAMETER", "trim_start_ns and trim_end_ns must be provided together")
    if limit < 1 or limit > MAX_DIFF_LIMIT:
        return _error("INVALID_PARAMETER", f"limit must be between 1 and {MAX_DIFF_LIMIT}")
    if sort not in {"delta", "percent", "total"}:
        return _error("INVALID_PARAMETER", "sort must be one of: delta, percent, total")

    trim = None if trim_start_ns is None else (int(trim_start_ns), int(trim_end_ns))
    try:
        with _open_readonly(before) as (before_conn, before_path):
            with _open_readonly(after) as (after_conn, after_path):
                before_prof = Profile._from_conn(before_conn)
                after_prof = Profile._from_conn(after_conn)
                before_prof.path = before_path
                after_prof.path = after_path
                if gpu is None:
                    summary, _per_gpu = diff_profiles_all_gpus(
                        before_prof,
                        after_prof,
                        trim=trim,
                        limit=limit,
                        sort=sort,
                    )
                else:
                    summary = diff_profiles(
                        before_prof,
                        after_prof,
                        gpu=int(gpu),
                        trim=trim,
                        limit=limit,
                        sort=sort,
                    )
                payload = to_diff_dict(summary)
                encoded = json.dumps(payload, ensure_ascii=False, default=str)
                if len(encoded) > MCP_MAX_PAYLOAD_CHARS:
                    return _error(
                        "PAYLOAD_TOO_LARGE",
                        "diff result exceeds the MCP payload cap; lower limit or select one GPU",
                        max_payload_chars=MCP_MAX_PAYLOAD_CHARS,
                    )
                return _json_safe(payload)
    except NsysAiError as exc:
        return exc.to_dict()
    except _READ_ERRORS as exc:
        return _error("DIFF_EXECUTION_ERROR", str(exc))
    except Exception as exc:
        return _error("DIFF_EXECUTION_ERROR", str(exc))


def create_server():
    """Build the optional FastMCP server; importing this module stays dependency-free."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "MCP support is optional; install it with `pip install 'nsys-ai[mcp]'`."
        ) from exc

    server = FastMCP("nsys-ai")

    @server.tool(name="list_skills")
    def list_skills() -> dict[str, Any]:
        """List registered analysis skills and their parameter schemas."""
        return _list_skills()

    @server.tool(name="get_session")
    def get_session(session: str) -> dict[str, Any]:
        """Read a CLI/Web/TUI session's canonical artifacts read-only."""
        return _get_session(session)

    @server.tool(name="run_skill")
    def run_skill(
        profile: str,
        skill_name: str,
        params: dict[str, Any] | None = None,
        max_rows: int = MAX_ROWS,
    ) -> dict[str, Any]:
        """Run one named skill read-only and return rows plus Finding evidence."""
        return _run_skill(profile, skill_name, params, max_rows)

    @server.tool(name="diff_profiles")
    def diff_profiles_tool(
        before: str,
        after: str,
        gpu: int | None = None,
        trim_start_ns: int | None = None,
        trim_end_ns: int | None = None,
        limit: int = 15,
        sort: str = "delta",
    ) -> dict[str, Any]:
        """Compare two profiles read-only and return the canonical diff contract."""
        return _diff_payload(before, after, gpu, trim_start_ns, trim_end_ns, limit, sort)

    return server


def main() -> None:
    """Run the MCP server over stdio."""
    create_server().run(transport="stdio")


__all__ = ["create_server", "main"]
