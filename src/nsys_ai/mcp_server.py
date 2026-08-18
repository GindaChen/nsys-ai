"""Optional read-only MCP transport for the canonical analysis contracts.

The module itself has no MCP import at import time.  Installing the ``mcp``
extra enables the ``nsys-ai-mcp`` entry point; importing the core package does
not acquire a new runtime dependency.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .annotation import PRODUCER, SCHEMA_VERSION
from .diff import diff_profiles, diff_profiles_all_gpus
from .diff_render import to_diff_dict
from .exceptions import SkillExecutionError, SkillNotFoundError, SkillParameterError
from .fingerprint import get_profile_id
from .profile import Profile, resolve_profile_path

MAX_ROWS = 50
MAX_DIFF_LIMIT = 50
# The SQL tool's 8,000-character cap is tuned for an LLM prompt. MCP returns
# the canonical structured diff envelope, so it needs a larger bounded budget
# while retaining an explicit payload guardrail.
MCP_MAX_PAYLOAD_CHARS = 100_000
_TRIM_KEYS = frozenset({"trim_start_ns", "trim_end_ns"})


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


def _coerce_param(value: Any, param_type: Any) -> Any:
    """Validate the JSON value against the type declared by a SkillParam."""
    type_name = str(param_type).lower()
    if param_type is int or type_name in {"int", "integer"}:
        if isinstance(value, bool):
            raise ValueError("boolean is not a valid integer parameter")
        return int(value)
    if param_type is float or type_name in {"float", "double"}:
        if isinstance(value, bool):
            raise ValueError("boolean is not a valid float parameter")
        return float(value)
    if param_type is bool or type_name in {"bool", "boolean"}:
        if not isinstance(value, bool):
            raise ValueError("parameter must be a boolean")
        return value
    if not isinstance(value, str):
        raise ValueError("parameter must be a string")
    return value


def _resolve_skill_params(skill, params: Mapping[str, Any] | None) -> dict[str, Any]:
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ValueError("params must be an object")

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
            resolved[name] = int(value)
        else:
            resolved[name] = _coerce_param(value, declared[name].type)
    return resolved


def _cap_rows(rows: list[dict], max_rows: int) -> tuple[list[dict], bool]:
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
    """Yield a direct-attach or SQLite ``mode=ro`` connection without a cache build."""
    from .parquet_cache import open_direct_sqlite, open_with_direct_fallback

    resolved = resolve_profile_path(path)
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

    skill = get_skill(skill_name)
    if skill is None:
        return SkillNotFoundError(
            f"Unknown skill '{skill_name}'",
            available=[item.name for item in all_skills()],
        ).to_dict()

    try:
        skill_params = _resolve_skill_params(skill, params)
    except (TypeError, ValueError) as exc:
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
    except (SkillExecutionError, sqlite3.Error, OSError, ValueError) as exc:
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
    except (OSError, sqlite3.Error, ValueError) as exc:
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
