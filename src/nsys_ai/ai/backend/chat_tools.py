"""
chat_tools.py — Tool definitions, system-prompt construction, and action
parsing for the AI chat layer.

This module is the "data / prompt" boundary:
- It knows what tools the LLM can call (OpenAI-style function specs).
- It knows how to build the system prompt from UI context.
- It knows how to parse a tool-call result back into a UI action.

It does NOT make any LLM API calls (those live in chat.py).
"""
from __future__ import annotations

import json

from .profile_db_tool import TOOL_QUERY_PROFILE_DB

# Pure MFU tool: one step_time_s per call. Same tool in single-profile and diff; diff compares by calling twice.
TOOL_COMPUTE_MFU = {
    "type": "function",
    "function": {
        "name": "compute_mfu",
        "description": (
            "Compute MFU (Model FLOPs Utilization) for one step. Pure calculation: step_time_s, model_flops_per_step, peak_tflops. "
            "Get step_time_s from profile (e.g. query_profile_db: (MAX([end])-MIN(start))/1e9) or from get_iteration_diff wall_clock_ms/1000. "
            "User must provide model_flops_per_step (nsys does not store it). peak_tflops from GPU spec (e.g. 989 H100, 312 A100). For diff comparison, call twice with before/after step_time_s."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "step_time_s": {"type": "number", "description": "Step or profile span in seconds (from query_profile_db or summary)."},
                "model_flops_per_step": {"type": "number", "description": "Model FLOPs per step (user must provide; e.g. 6*N_params*tokens for Transformer)."},
                "peak_tflops": {"type": "number", "description": "GPU peak TFLOPS for precision (e.g. 989 for H100 FP16, 312 for A100 FP16)."},
            },
            "required": ["step_time_s", "model_flops_per_step", "peak_tflops"],
        },
    },
}

# Get peak TFLOPS from profile GPU name (BF16/FP16). Call before compute_mfu so you only ask user for model_flops_per_step.
TOOL_GET_GPU_PEAK_TFLOPS = {
    "type": "function",
    "function": {
        "name": "get_gpu_peak_tflops",
        "description": (
            "Get the peak TFLOPS (BF16/FP16 Tensor Core) for the GPU in the current profile. "
            "Call this before compute_mfu so you only need to ask the user for model_flops_per_step. "
            "Returns gpu_name and peak_tflops, or error if GPU unknown."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

def _tools_openai() -> list[dict]:
    """Return the OpenAI-style tool list: navigate, zoom, query_profile_db, get_gpu_peak_tflops, compute_mfu."""
    return [
        {
            "type": "function",
            "function": {
                "name": "navigate_to_kernel",
                "description": (
                    "Navigate the UI to a specific kernel. "
                    "Match by EXACT kernel name from visible_kernels_summary or "
                    "global_top_kernels. The frontend will resolve the exact occurrence."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_name": {
                            "type": "string",
                            "description": "The exact name of the kernel to navigate to.",
                        },
                        "occurrence_index": {
                            "type": "integer",
                            "description": "Which occurrence to jump to (1-based). Default is 1.",
                            "default": 1,
                        },
                        "reason": {
                            "type": "string",
                            "description": "A short, 1-sentence reason why this kernel was chosen.",
                        },
                    },
                    "required": ["target_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "zoom_to_time_range",
                "description": "Zoom the UI to a specific time range in seconds.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_s": {"type": "number", "description": "Start time in seconds"},
                        "end_s":   {"type": "number", "description": "End time in seconds"},
                    },
                    "required": ["start_s", "end_s"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fit_nvtx_range",
                "description": (
                    "Fit an NVTX range to the viewport width. "
                    "Prefer nvtx_name when possible; otherwise use explicit start/end seconds."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nvtx_name": {
                            "type": "string",
                            "description": "NVTX name substring to find and fit.",
                        },
                        "occurrence_index": {
                            "type": "integer",
                            "description": "Which occurrence to fit (1-based). Default is 1.",
                            "default": 1,
                        },
                        "start_s": {"type": "number", "description": "Start time in seconds"},
                        "end_s": {"type": "number", "description": "End time in seconds"},
                    },
                },
            },
        },
        TOOL_QUERY_PROFILE_DB,
        TOOL_GET_GPU_PEAK_TFLOPS,
        TOOL_COMPUTE_MFU,
    ]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_system_prompt(
    ui_context: dict,
    profile_schema: str | None = None,
) -> str:
    """Build the system prompt that instructs the LLM on its role and tools.

    Args:
        ui_context:      JSON-serialisable dict with visible kernel summary,
                         selected stream, time range, etc.
        profile_schema:  Optional schema string from ``get_profile_schema_cached``.
                         When provided the LLM is instructed to use
                         ``query_profile_db`` for whole-profile questions.
    """
    ctx_json = json.dumps(ui_context, separators=(",", ":"))
    schema_block = ""
    if profile_schema:
        schema_block = (
            "\n=== PROFILE DATABASE SCHEMA (for query_profile_db) ===\n"
            f"{profile_schema}\n"
            "NOTE: Write strict SQLite3 SQL only (use strftime() not DATE_TRUNC/EXTRACT; "
            "use || for concatenation not CONCAT()).\n"
            "=====================================================\n\n"
        )
    return (
        "You are an expert GPU performance analyst and UI navigator for an Nsight Systems viewer.\n"
        "Your goal is to explain CUDA/GPU bottlenecks clearly and help users navigate the timeline.\n"
        f"{schema_block}"
        "=== CURRENT UI CONTEXT ===\n"
        f"```json\n{ctx_json}\n```\n"
        "==========================\n\n"
        "INSTRUCTIONS:\n"
        "1. When asked to explain a kernel or bottleneck, use the provided context. "
        "Be concise, professional, and use Markdown for formatting.\n"
        "2. If the user asks to go to, find, or locate a specific kernel or time range, "
        "YOU MUST use the provided tools (`navigate_to_kernel`, `zoom_to_time_range`, "
        "or `fit_nvtx_range`).\n"
        "3. When a PROFILE DATABASE SCHEMA is provided above, you MUST use the "
        "`query_profile_db` tool to answer whole-profile questions (e.g. first kernel, "
        "slowest kernel, counts, total GPU time, total kernel count). Run a SELECT; "
        "the backend returns the result and you answer from it. Never use `SELECT *`; "
        "always select only the columns you need. For total GPU time use SUM(duration_ns)/1e6; "
        "for kernel count use COUNT(*). Kernel names are stored as IDs referencing StringIds: "
        "join with StringIds (e.g. k.shortName = StringIds.id) and use StringIds.value for "
        "human-readable names. IMPORTANT: stats.total_gpu_ms, stats.total_kernel_count, and "
        "global_top_kernels are intentionally OMITTED from ui_context when the DB agent is "
        "enabled. You MUST use `query_profile_db` to answer any whole-profile questions - "
        "do NOT guess or say the data is missing.\n"
        "4. TOOL USE RULES:\n"
        "   - Match kernel names exactly from `visible_kernels_summary` or `global_top_kernels`.\n"
        "   - Do NOT explain what you are about to do before calling a tool. Just call the tool.\n"
        "   - For `navigate_to_kernel`, `zoom_to_time_range`, and `fit_nvtx_range`: execution is immediate on the "
        "client; you do not wait for a result. For `query_profile_db`: the backend runs the "
        "query and returns rows; use them in your answer.\n"
        "   - Do NOT output code blocks or JSON for navigation - use the actual tool call mechanism only.\n"
        "5. If a requested kernel is not in the context, politely say it is not visible or "
        "does not exist.\n"
        "6. For MFU: (1) Call get_gpu_peak_tflops to get peak_tflops from the profile GPU. (2) Use query_profile_db to get step_time_s (e.g. (MAX([end])-MIN(start))/1e9). (3) Ask the user for model_flops_per_step (nsys does not store it). Do NOT call compute_mfu until the user has provided it — after asking, end your response and wait for their reply; only then call compute_mfu with that value. (4) If get_gpu_peak_tflops returns an error, ask the user for peak_tflops as well."
    )


# ---------------------------------------------------------------------------
# Tool-call parsing — converts raw LLM function calls to UI action dicts
# ---------------------------------------------------------------------------

def _parse_tool_call(name: str, arguments: str) -> dict | None:
    """Parse a tool call into a UI action dict, or ``None`` if unrecognised.

    Only ``navigate_to_kernel``, ``zoom_to_time_range``, and ``fit_nvtx_range`` produce UI actions.
    ``query_profile_db`` is handled by the agent loop itself and returns None
    here (it is not a UI action).
    """
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (json.JSONDecodeError, TypeError):
        return None

    if name == "navigate_to_kernel":
        target = args.get("target_name")
        if not target:
            return None
        return {
            "type": "navigate_to_kernel",
            "target_name": target,
            "occurrence_index": args.get("occurrence_index", 1),
            "reason": args.get("reason"),
        }

    if name == "zoom_to_time_range":
        start_s = args.get("start_s")
        end_s = args.get("end_s")
        if start_s is None or end_s is None:
            return None
        return {
            "type": "zoom_to_time_range",
            "start_s": float(start_s),
            "end_s": float(end_s),
        }

    if name == "fit_nvtx_range":
        out = {"type": "fit_nvtx_range"}
        if args.get("nvtx_name"):
            out["nvtx_name"] = str(args.get("nvtx_name"))
            out["occurrence_index"] = int(args.get("occurrence_index", 1))
            return out
        start_s = args.get("start_s")
        end_s = args.get("end_s")
        if start_s is None or end_s is None:
            return None
        out["start_s"] = float(start_s)
        out["end_s"] = float(end_s)
        return out

    return None
