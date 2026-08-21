"""
tool_dispatch.py — Centralized tool dispatch for the AI chat agent.

Replaces the large if/elif chains in ``chat.py`` (in particular,
``stream_agent_loop``) with a registry-based dispatcher.

Usage::

    dispatcher = ToolDispatcher(
        conn=conn,
        sqlite_path=sqlite_path,
        query_runner=query_runner,
        finding_counter=finding_counter_fn,
    )
    result = dispatcher.dispatch(tool_name, args_str)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from .ai.backend.profile_db_tool import DEFAULT_MAX_JSON_CHARS

_log = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool dispatch call."""

    content: str = ""
    """JSON-serialized (or plain string) result for the LLM."""

    events: list[dict] = field(default_factory=list)
    """Side-effect events to yield (e.g., ``{type: "system", ...}``)."""

    is_action: bool = False
    """True when the tool produces a navigation/zoom action (not a tool result)."""

    action: dict | None = None
    """Parsed action dict (navigate, zoom, fit_nvtx) when ``is_action`` is True."""

    skip_tool_message: bool = False
    """If True, the caller should NOT append a tool result message."""


def _parse_json_args(args_str: str) -> dict:
    """Parse JSON tool arguments.

    Returns an empty dict if the input is empty/whitespace. Otherwise,
    returns the decoded JSON object, and propagates
    ``json.JSONDecodeError`` if the input is not valid JSON.
    """
    if not args_str or not args_str.strip():
        return {}
    return json.loads(args_str)


def _serialize_skill_result(skill_name: str, rows: list) -> str:
    """Serialize registry rows without allowing one skill to flood the prompt."""
    payload = {"skill": skill_name, "rows": rows}
    encoded = json.dumps(payload, ensure_ascii=False, default=str)
    if len(encoded) <= DEFAULT_MAX_JSON_CHARS:
        return encoded

    capped_rows = []
    for row in rows:
        candidate = {
            "skill": skill_name,
            "rows": capped_rows + [row],
            "_truncated": True,
            "_total_rows": len(rows),
            "_shown_rows": len(capped_rows) + 1,
        }
        if len(json.dumps(candidate, ensure_ascii=False, default=str)) > DEFAULT_MAX_JSON_CHARS:
            break
        capped_rows.append(row)

    return json.dumps(
        {
            "skill": skill_name,
            "rows": capped_rows,
            "_truncated": True,
            "_total_rows": len(rows),
            "_shown_rows": len(capped_rows),
        },
        ensure_ascii=False,
        default=str,
    )


class ToolDispatcher:
    """Registry-based tool dispatcher for the AI agent loop.

    Unifies the tool handling previously scattered across:
    - ``stream_agent_loop`` (~320 lines of if/elif)
    - ``run_agent_loop`` (~50 lines)

    Each tool handler is a method that accepts ``(args: dict)`` and returns
    a ``ToolResult``.  Handlers are registered in ``__init__`` based on
    what resources are available (conn, sqlite_path, query_runner, etc.).
    """

    def __init__(
        self,
        *,
        conn=None,
        sqlite_path: str | None = None,
        query_runner: Callable[[str], str] | None = None,
        finding_counter: Callable[[], int] | None = None,
        finding_sink: Callable[[dict], None] | None = None,
        finding_provenance: dict | None = None,
        max_consecutive_db_errors: int = 3,
        mode: str = "profile",
        diff_context=None,
    ):
        self._conn = conn
        self._sqlite_path = sqlite_path
        self._query_runner = query_runner
        self._finding_counter = finding_counter
        self._finding_sink = finding_sink
        self._finding_provenance = dict(finding_provenance or {})
        self._consecutive_db_errors = 0
        self._max_consecutive_db_errors = max_consecutive_db_errors
        self._mode = mode
        self._diff_context = diff_context

        self._handlers: dict[str, Callable[[dict], ToolResult]] = {}
        if mode == "profile":
            self._register_profile_tools()
        elif mode == "diff":
            self._register_diff_tools()
        else:
            raise ValueError(
                f"Unsupported ToolDispatcher mode: {mode!r}. Expected 'profile' or 'diff'."
            )

    def _register_profile_tools(self) -> None:
        """Register all single-profile tool handlers."""
        self._handlers["get_gpu_peak_tflops"] = self._handle_gpu_peak_tflops
        self._handlers["compute_mfu"] = self._handle_compute_mfu
        self._handlers["compute_theoretical_flops"] = self._handle_theoretical_flops
        self._handlers["compute_region_mfu"] = self._handle_region_mfu
        self._handlers["submit_finding"] = self._handle_submit_finding
        self._handlers["run_skill"] = self._handle_run_skill
        self._handlers["get_gpu_overlap_stats"] = self._handle_gpu_overlap_stats
        self._handlers["get_nccl_breakdown"] = self._handle_nccl_breakdown
        self._handlers["query_profile_db"] = self._handle_query_profile_db

    def _register_diff_tools(self) -> None:
        """Register all diff-profile tool handlers."""
        self._handlers["search_nvtx_regions"] = self._handle_search_nvtx_regions
        self._handlers["get_iteration_boundaries"] = self._handle_get_iteration_boundaries
        self._handlers["explore_nvtx_hierarchy"] = self._handle_explore_nvtx_hierarchy
        self._handlers["get_top_nvtx_diffs"] = self._handle_get_top_nvtx_diffs
        self._handlers["get_iteration_diff"] = self._handle_get_iteration_diff
        self._handlers["get_region_diff"] = self._handle_get_region_diff
        self._handlers["get_source_code_context"] = self._handle_get_source_code_context
        self._handlers["get_global_diff"] = self._handle_get_global_diff
        self._handlers["get_memory_profile_diff"] = self._handle_get_memory_profile_diff
        self._handlers["get_gpu_imbalance_stats"] = self._handle_get_gpu_imbalance_stats
        self._handlers["summarize_nvtx_subtree"] = self._handle_summarize_nvtx_subtree
        self._handlers["get_launch_config_diff"] = self._handle_get_launch_config_diff
        self._handlers["get_gpu_peak_tflops"] = self._handle_gpu_peak_tflops
        self._handlers["compute_mfu"] = self._handle_compute_mfu

    def knows(self, name: str) -> bool:
        """Return True if this dispatcher has a handler for the named tool."""
        return name in self._handlers

    def dispatch(self, name: str, args_str: str) -> ToolResult:
        """Dispatch a tool call by name, returning a ToolResult.

        If the tool name is not registered, returns a ToolResult with
        ``skip_tool_message=True`` and ``content`` explaining the tool
        is not handled.
        """
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(
                content=f"Tool '{name}' is not handled by the dispatcher.",
                skip_tool_message=True,
            )
        try:
            args = _parse_json_args(args_str)
        except Exception as exc:
            _log.exception("Failed to parse JSON args for tool '%s'", name)
            return ToolResult(
                content=json.dumps(
                    {"error": "invalid_tool_arguments", "tool": name, "message": str(exc)}
                ),
                events=[
                    {
                        "type": "system",
                        "content": "Tool argument parsing failed; see logs for details.",
                    }
                ],
            )
        try:
            return handler(args)
        except Exception as exc:
            _log.exception("Unhandled exception in tool handler '%s'", name)
            return ToolResult(
                content=json.dumps(
                    {"error": "tool_execution_error", "tool": name, "message": str(exc)}
                ),
                events=[
                    {"type": "system", "content": "Tool execution failed; see logs for details."}
                ],
            )

    # ── Tool handlers ─────────────────────────────────────────────────

    def _handle_gpu_peak_tflops(self, args: dict) -> ToolResult:
        from .hardware import get_peak_tflops
        from .profile import get_first_gpu_name

        events = [{"type": "system", "content": "Getting GPU peak TFLOPS..."}]
        if self._conn is not None:
            gpu_name = get_first_gpu_name(self._conn)
            result = get_peak_tflops(gpu_name)
        elif getattr(self, "_diff_context", None) is not None:
            devs = self._diff_context.after.meta.devices or [0]
            gpu_name = ""
            if devs:
                gi = self._diff_context.after.meta.gpu_info.get(devs[0])
                if gi:
                    gpu_name = gi.name or ""
            result = get_peak_tflops(gpu_name)
        else:
            result = {"gpu_name": "", "error": "No profile loaded"}
        return ToolResult(content=json.dumps(result), events=events)

    def _handle_compute_mfu(self, args: dict) -> ToolResult:
        from .mfu import compute_mfu_from_args

        events = [{"type": "system", "content": "Running compute_mfu..."}]
        result = compute_mfu_from_args(args)
        return ToolResult(content=json.dumps(result), events=events)

    def _handle_theoretical_flops(self, args: dict) -> ToolResult:
        from .region_mfu import compute_theoretical_flops

        events = [{"type": "system", "content": "Computing theoretical FLOPs..."}]
        op = str(args.get("operation") or "")
        if not op:
            result = {"error": {"code": "MISSING_PARAMETER", "message": "operation is required."}}
        else:
            result = compute_theoretical_flops(
                op,
                hidden_dim=int(args.get("hidden_dim") or 0),
                seq_len=int(args.get("seq_len") or 0),
                num_layers=int(args.get("num_layers") or 1),
                ffn_dim=int(args["ffn_dim"]) if args.get("ffn_dim") is not None else None,
                batch_size=int(args.get("batch_size") or 1),
                multiplier=int(args.get("multiplier") or 1),
                M=int(args.get("M") or 0),
                N=int(args.get("N") or 0),
                K=int(args.get("K") or 0),
            )
        return ToolResult(content=json.dumps(result), events=events)

    def _handle_region_mfu(self, args: dict) -> ToolResult:
        events = [{"type": "system", "content": "Running compute_region_mfu..."}]
        if self._conn is None:
            result = {
                "error": {
                    "code": "PROFILE_NOT_LOADED",
                    "message": "No profile loaded; cannot compute region MFU.",
                }
            }
            return ToolResult(content=json.dumps(result), events=events)

        region_name = args.get("name") or ""
        raw_flops = args.get("theoretical_flops")
        if not region_name:
            result = {
                "error": {
                    "code": "MISSING_PARAMETER",
                    "message": "name is required (NVTX range or kernel name).",
                }
            }
        elif raw_flops is None:
            result = {
                "error": {
                    "code": "MISSING_PARAMETER",
                    "message": "theoretical_flops is required and must be positive. "
                    "Compute it from model architecture using the MFU REFERENCE FORMULAS.",
                }
            }
        else:
            try:
                flops_val = float(raw_flops)
            except (ValueError, TypeError):
                flops_val = -1.0
            if flops_val <= 0:
                result = {
                    "error": {
                        "code": "INVALID_PARAMETER",
                        "message": f"theoretical_flops must be a positive number, got: {raw_flops!r}",
                    }
                }
            else:
                rows = self._run_registered_skill(
                    "region_mfu",
                    {
                        "name": region_name,
                        "theoretical_flops": flops_val,
                        "profile_path": self._sqlite_path,
                        "source": str(args.get("source") or "nvtx"),
                        "peak_tflops": (
                            float(args["peak_tflops"])
                            if "peak_tflops" in args and args["peak_tflops"] is not None
                            else None
                        ),
                        "num_gpus": int(args.get("num_gpus") or 1),
                        "occurrence_index": int(args.get("occurrence_index") or 1),
                        "device_id": (
                            int(args["device_id"])
                            if "device_id" in args and args["device_id"] is not None
                            else None
                        ),
                        "match_mode": str(args.get("match_mode") or "contains"),
                    },
                )
                result = rows[0] if len(rows) == 1 else rows
        return ToolResult(content=json.dumps(result), events=events)

    def _run_registered_skill(self, skill_name: str, args: dict) -> list[dict]:
        """Execute a profile skill through the canonical registry.

        CLI callers keep the registry's formatted-text default. Chat tools need
        the same rows as structured JSON, so the registry exposes that result
        explicitly instead of each tool importing an analysis implementation.
        """
        if self._conn is None:
            return [{"error": "No profile loaded"}]

        from .skills import registry

        return registry.run_skill(skill_name, self._conn, raw=True, **args)

    def _handle_submit_finding(self, args: dict) -> ToolResult:
        explicit_index = args.get("index")
        if isinstance(explicit_index, int):
            finding_index = explicit_index
        elif self._finding_counter is not None:
            finding_index = self._finding_counter()
        else:
            finding_index = 0

        finding_payload = dict(args)
        finding_payload["index"] = finding_index

        if self._finding_sink is not None:
            persisted_payload = dict(args)
            persisted_payload["provenance"] = dict(self._finding_provenance)
            self._finding_sink(persisted_payload)
            finding_payload["provenance"] = dict(self._finding_provenance)

        events = [{"type": "finding", "finding": finding_payload}]
        result = {
            "status": "submitted",
            "index": finding_index,
            "note": (
                "Finding overlaid on timeline. "
                f"Reference as [Finding {finding_index}] in your answer."
            ),
        }
        return ToolResult(content=json.dumps(result), events=events)

    def _handle_run_skill(self, args: dict) -> ToolResult:
        """Run a named registry skill as the formal chat evidence path."""
        events = [{"type": "system", "content": "Running registered analysis skill..."}]
        skill_name = str(args.get("skill_name") or "").strip()
        if not skill_name:
            return ToolResult(
                content=json.dumps(
                    {
                        "error": {
                            "code": "MISSING_PARAMETER",
                            "message": "skill_name is required.",
                        }
                    }
                ),
                events=events,
            )
        params = args.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return ToolResult(
                content=json.dumps(
                    {
                        "error": {
                            "code": "INVALID_PARAMETER",
                            "message": "params must be an object.",
                        }
                    }
                ),
                events=events,
            )

        from .skills.registry import get_skill

        if get_skill(skill_name) is None:
            return ToolResult(
                content=json.dumps(
                    {
                        "error": {
                            "code": "UNKNOWN_SKILL",
                            "message": f"Unknown skill '{skill_name}'.",
                        }
                    }
                ),
                events=events,
            )
        rows = self._run_registered_skill(skill_name, dict(params))
        return ToolResult(
            content=_serialize_skill_result(skill_name, rows),
            events=events,
        )

    def _handle_gpu_overlap_stats(self, args: dict) -> ToolResult:
        events = [{"type": "system", "content": "Computing GPU overlap stats..."}]
        if self._conn is None:
            result = {"error": "No profile loaded"}
            return ToolResult(content=json.dumps(result), events=events)

        try:
            from .profile import Profile
            from .skills.base import is_abstention

            _prof = Profile._from_conn(self._conn)
            _devices = _prof.meta.devices or [0]
            _start_s = args.get("start_s")
            _end_s = args.get("end_s")
            _skill_args = {}
            if _start_s is not None and _end_s is not None:
                _skill_args["trim_start_ns"] = int(float(_start_s) * 1e9)
                _skill_args["trim_end_ns"] = int(float(_end_s) * 1e9)

            _per_gpu = []
            _unavailable = []
            for _dev in _devices:
                _rows = self._run_registered_skill(
                    "overlap_breakdown", {"device": _dev, **_skill_args}
                )
                if _rows and isinstance(_rows[0], dict):
                    _oa = _rows[0]
                    if is_abstention(_rows):
                        _unavailable.extend(_rows)
                    elif "error" not in _oa:
                        _oa = dict(_oa)
                        _oa["gpu_id"] = _dev
                        _gpu_info = _prof.meta.gpu_info.get(_dev)
                        if _gpu_info:
                            _oa["gpu_name"] = _gpu_info.name
                        _per_gpu.append(_oa)
            result = {
                "device_count": len(_devices),
                "per_gpu": _per_gpu,
            }
            if _unavailable:
                result["unavailable"] = _unavailable
        except Exception as _e:
            result = {"error": str(_e)}
        return ToolResult(content=json.dumps(result), events=events)

    def _handle_nccl_breakdown(self, args: dict) -> ToolResult:
        events = [{"type": "system", "content": "Analyzing NCCL collectives..."}]
        if self._conn is None:
            result = {"error": "No profile loaded"}
            return ToolResult(content=json.dumps(result), events=events)

        try:
            from .profile import Profile

            _prof = Profile._from_conn(self._conn)
            _devices = _prof.meta.devices or [0]
            _dev = args.get("device_id", _devices[0])
            _skill_args = {"device": int(_dev)}
            _start_s = args.get("start_s")
            _end_s = args.get("end_s")
            if _start_s is not None and _end_s is not None:
                _skill_args["trim_start_ns"] = int(float(_start_s) * 1e9)
                _skill_args["trim_end_ns"] = int(float(_end_s) * 1e9)
            _rows = self._run_registered_skill("nccl_breakdown", _skill_args)
            result = {
                "device_id": _dev,
                "collectives": _rows,
            }
        except Exception as _e:
            result = {"error": str(_e)}
        return ToolResult(content=json.dumps(result), events=events)

    def _handle_query_profile_db(self, args: dict) -> ToolResult:
        events = [{"type": "system", "content": "Running DB query..."}]
        if self._query_runner is None:
            return ToolResult(
                content="Not executed (no profile loaded).",
                events=events,
            )

        sql = args.get("sql_query", "")
        try:
            result = self._query_runner(sql)
        except Exception as e:
            result = f"Error: {e}"

        if result.startswith("Error:"):
            self._consecutive_db_errors += 1
            if self._consecutive_db_errors >= self._max_consecutive_db_errors:
                result += (
                    "\n[System: Repeated SQL errors. "
                    "Do not infer profile facts. Unless another successful tool result "
                    "supports the answer, state that the question cannot be answered.]"
                )
        else:
            self._consecutive_db_errors = 0

        return ToolResult(content=result, events=events)

    def _handle_search_nvtx_regions(self, args: dict) -> ToolResult:
        from .diff_tools import search_nvtx_regions

        events = [{"type": "system", "content": "Searching NVTX regions..."}]
        if not self._diff_context:
            return ToolResult(content="No diff context.", events=events)
        res = search_nvtx_regions(
            self._diff_context,
            args.get("query", ""),
            args.get("limit", 50),
            args.get("use_glob", False),
        )
        return ToolResult(content=json.dumps(res), events=events)

    def _handle_get_iteration_boundaries(self, args: dict) -> ToolResult:
        from .diff_tools import get_iteration_boundaries

        events = [{"type": "system", "content": "Getting iteration boundaries..."}]
        if not self._diff_context:
            return ToolResult(content="No diff context.", events=events)
        res = get_iteration_boundaries(
            self._diff_context,
            args.get("marker"),
            args.get("target_gpu"),
        )
        return ToolResult(content=json.dumps(res), events=events)

    def _handle_explore_nvtx_hierarchy(self, args: dict) -> ToolResult:
        from .diff_tools import explore_nvtx_hierarchy

        events = [{"type": "system", "content": "Exploring NVTX hierarchy..."}]
        if not self._diff_context:
            return ToolResult(content="No diff context.", events=events)
        res = explore_nvtx_hierarchy(
            self._diff_context,
            args.get("parent_path", ""),
            args.get("depth", 1),
            args.get("target_gpu"),
            args.get("profile_side", "after"),
        )
        return ToolResult(content=json.dumps(res), events=events)

    def _handle_get_top_nvtx_diffs(self, args: dict) -> ToolResult:
        from .diff_tools import get_top_nvtx_diffs

        events = [{"type": "system", "content": "Getting top NVTX diffs..."}]
        if not self._diff_context:
            return ToolResult(content="No diff context.", events=events)
        res = get_top_nvtx_diffs(
            self._diff_context,
            args.get("limit", 20),
            args.get("target_gpu"),
        )
        return ToolResult(content=json.dumps(res), events=events)

    def _handle_get_iteration_diff(self, args: dict) -> ToolResult:
        from .diff_tools import get_iteration_diff

        events = [{"type": "system", "content": "Getting iteration diff..."}]
        if not self._diff_context:
            return ToolResult(content="No diff context.", events=events)
        res = get_iteration_diff(
            self._diff_context,
            int(args["iteration_index"]),
            args.get("marker"),
            args.get("target_gpu"),
        )
        return ToolResult(content=json.dumps(res), events=events)

    def _handle_get_region_diff(self, args: dict) -> ToolResult:
        from .diff_tools import get_region_diff

        events = [{"type": "system", "content": "Getting region diff..."}]
        if not self._diff_context:
            return ToolResult(content="No diff context.", events=events)
        nvtx = args.get("nvtx_exact_match")
        if nvtx is None:
            nvtx = ""
        res = get_region_diff(
            self._diff_context,
            nvtx,
            args.get("iteration_index"),
            args.get("target_gpu"),
        )
        return ToolResult(content=json.dumps(res), events=events)

    def _handle_get_source_code_context(self, args: dict) -> ToolResult:
        from .diff_tools import get_source_code_context

        events = [{"type": "system", "content": "Getting source context..."}]
        if not self._diff_context:
            return ToolResult(content="No diff context.", events=events)
        res = get_source_code_context(self._diff_context, args.get("nvtx_path", ""))
        return ToolResult(content=json.dumps(res), events=events)

    def _handle_get_global_diff(self, args: dict) -> ToolResult:
        from .diff_tools import get_global_diff

        events = [{"type": "system", "content": "Getting global diff..."}]
        if not self._diff_context:
            return ToolResult(content="No diff context.", events=events)
        res = get_global_diff(
            self._diff_context,
            args.get("skip_first_ms", 0),
            args.get("duration_ms"),
            args.get("target_gpu"),
        )
        return ToolResult(content=json.dumps(res), events=events)

    def _handle_get_memory_profile_diff(self, args: dict) -> ToolResult:
        from .diff_tools import get_memory_profile_diff

        events = [{"type": "system", "content": "Getting memory profile diff..."}]
        if not self._diff_context:
            return ToolResult(content="No diff context.", events=events)
        res = get_memory_profile_diff(
            self._diff_context,
            args.get("iteration_index"),
            args.get("target_gpu"),
        )
        return ToolResult(content=json.dumps(res), events=events)

    def _handle_get_gpu_imbalance_stats(self, args: dict) -> ToolResult:
        from .diff_tools import get_gpu_imbalance_stats

        events = [{"type": "system", "content": "Getting GPU imbalance stats..."}]
        if not self._diff_context:
            return ToolResult(content="No diff context.", events=events)
        res = get_gpu_imbalance_stats(
            self._diff_context,
            args.get("iteration_index"),
            args.get("marker"),
        )
        return ToolResult(content=json.dumps(res), events=events)

    def _handle_summarize_nvtx_subtree(self, args: dict) -> ToolResult:
        from .diff_tools import summarize_nvtx_subtree

        events = [{"type": "system", "content": "Summarizing NVTX subtree..."}]
        if not self._diff_context:
            return ToolResult(content="No diff context.", events=events)
        res = summarize_nvtx_subtree(
            self._diff_context,
            args.get("parent_path", ""),
            args.get("iteration_index"),
            args.get("target_gpu"),
            args.get("top_n", 3),
        )
        return ToolResult(content=json.dumps(res), events=events)

    def _handle_get_launch_config_diff(self, args: dict) -> ToolResult:
        from .diff_tools import get_launch_config_diff

        events = [{"type": "system", "content": "Getting launch config diff..."}]
        if not self._diff_context:
            return ToolResult(content="No diff context.", events=events)
        res = get_launch_config_diff(
            self._diff_context,
            args.get("kernel_name", ""),
            args.get("iteration_index"),
            args.get("target_gpu"),
        )
        return ToolResult(content=json.dumps(res), events=events)
