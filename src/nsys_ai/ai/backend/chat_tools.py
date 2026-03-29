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

import functools
import json
import logging
from pathlib import Path

from .profile_db_tool import TOOL_QUERY_PROFILE_DB

logger = logging.getLogger(__name__)

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
                "step_time_s": {
                    "type": "number",
                    "description": "Step or profile span in seconds (from query_profile_db or summary).",
                },
                "model_flops_per_step": {
                    "type": "number",
                    "description": "Model FLOPs per step (user must provide; e.g. 6*N_params*tokens for Transformer).",
                },
                "peak_tflops": {
                    "type": "number",
                    "description": "GPU peak TFLOPS for precision (e.g. 989 for H100 FP16, 312 for A100 FP16).",
                },
            },
            "required": ["step_time_s", "model_flops_per_step", "peak_tflops"],
        },
    },
}

# Region-level MFU tool: compute MFU for a specific NVTX region or kernel.
# The backend injects the current profile_path; the model MUST NOT pass a profile_path argument.
TOOL_COMPUTE_REGION_MFU = {
    "type": "function",
    "function": {
        "name": "compute_region_mfu",
        "description": (
            "Compute MFU (Model FLOPs Utilization) for a named NVTX region or CUDA kernel. "
            "Two modes: (1) source='nvtx' — finds an NVTX range by name, attributes kernels inside it; "
            "(2) source='kernel' — finds CUDA kernels by name directly (use when no custom NVTX labels exist). "
            "BEFORE calling this tool, compute theoretical_flops using the MFU REFERENCE formulas. "
            "Do NOT pass a profile_path argument."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name to match: NVTX range text (source='nvtx') or kernel shortName (source='kernel'). Substring match by default.",
                },
                "theoretical_flops": {
                    "type": "number",
                    "description": "Theoretical model FLOPs for this region, computed from model architecture formulas.",
                },
                "source": {
                    "type": "string",
                    "description": "'nvtx' to match NVTX ranges (default), 'kernel' to match CUDA kernels directly by name.",
                    "enum": ["nvtx", "kernel"],
                    "default": "nvtx",
                },
                "peak_tflops": {
                    "type": "number",
                    "description": "Optional per-GPU peak TFLOPS (BF16/FP16). If omitted, inferred from profile GPU.",
                },
                "num_gpus": {
                    "type": "integer",
                    "description": "Number of GPUs used (world_size). Peak is scaled by this. Default 1.",
                    "default": 1,
                },
                "occurrence_index": {
                    "type": "integer",
                    "description": "Which matching NVTX occurrence to use (1-based, only for source='nvtx'). Default 1.",
                    "default": 1,
                },
                "device_id": {
                    "type": "integer",
                    "description": "Optional CUDA deviceId to restrict to a single GPU.",
                },
                "match_mode": {
                    "type": "string",
                    "description": "'contains' (substring, default) or 'exact'.",
                    "enum": ["contains", "exact"],
                    "default": "contains",
                },
            },
            "required": ["name", "theoretical_flops"],
        },
    },
}

# Theoretical FLOPs calculator — does exact arithmetic so the LLM doesn't have to.
TOOL_COMPUTE_THEORETICAL_FLOPS = {
    "type": "function",
    "function": {
        "name": "compute_theoretical_flops",
        "description": (
            "Compute theoretical FLOPs for transformer operations using EXACT arithmetic. "
            "ALWAYS call this BEFORE compute_region_mfu — do NOT compute FLOPs yourself. "
            "IMPORTANT: set num_layers to the model's layer count (e.g. 32 for LLaMA-7B) "
            "to get total FLOPs across all layers. "
            "Pass the returned theoretical_flops value directly to compute_region_mfu."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": (
                        "Operation type: "
                        "'attention' (QK^T+softmax*V: 4*S²*H), "
                        "'qkv_proj' (Q/K/V linear: 6*S*H²), "
                        "'output_proj' (out linear: 2*S*H²), "
                        "'mlp' (FFN up+down: 4*S*H*ffn), "
                        "'full_layer' (attn+proj+mlp combined), "
                        "'full_model' (same as full_layer), "
                        "'linear' (generic: 2*M*N*K)."
                    ),
                    "enum": [
                        "attention",
                        "qkv_proj",
                        "output_proj",
                        "mlp",
                        "full_layer",
                        "full_model",
                        "linear",
                    ],
                },
                "hidden_dim": {
                    "type": "integer",
                    "description": "Model hidden dimension (H). Required for all operations except 'linear'.",
                },
                "seq_len": {
                    "type": "integer",
                    "description": "Sequence length (S). Required for all operations except 'linear'.",
                },
                "num_layers": {
                    "type": "integer",
                    "description": (
                        "Number of transformer layers (L). MUST match the model's actual layer count "
                        "(e.g. 32 for LLaMA-7B, 80 for LLaMA-70B). Per-layer FLOPs are multiplied by this."
                    ),
                    "default": 1,
                },
                "ffn_dim": {
                    "type": "integer",
                    "description": "FFN intermediate dimension. Defaults to 4*hidden_dim if omitted.",
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Batch size. Default 1.",
                    "default": 1,
                },
                "multiplier": {
                    "type": "integer",
                    "description": "1=forward only, 3=fwd+bwd (no ckpt), 4=fwd+bwd+recompute. Default 1.",
                    "default": 1,
                },
                "M": {"type": "integer", "description": "For 'linear' only: first dimension."},
                "N": {"type": "integer", "description": "For 'linear' only: second dimension."},
                "K": {"type": "integer", "description": "For 'linear' only: third dimension."},
            },
            "required": ["operation"],
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

# Submit a visual finding to overlay on the timeline for human verification.
TOOL_SUBMIT_FINDING = {
    "type": "function",
    "function": {
        "name": "submit_finding",
        "description": (
            "Submit a visual finding to overlay on the GPU timeline. "
            "Call this when you identify a bottleneck, stall, idle gap, or anomaly. "
            "The finding appears as a colored annotation on the timeline and a card "
            "in the Evidence sidebar. Returns the finding number (1-based). "
            "Reference it in your answer as [Finding N] so the user can click to zoom."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["region", "highlight", "marker"],
                    "description": (
                        "region: colored overlay spanning a time range. "
                        "highlight: overlay on a specific stream. "
                        "marker: single point in time."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": "Short label, e.g. 'NCCL Stall' or 'Compute Gap'",
                },
                "start_ns": {
                    "type": "integer",
                    "description": "Start timestamp in nanoseconds (from kernel start column)",
                },
                "end_ns": {
                    "type": "integer",
                    "description": "End timestamp in nanoseconds (omit for marker type)",
                },
                "gpu_id": {"type": "integer", "description": "GPU device ID"},
                "stream": {"type": "integer", "description": "Stream ID (for highlight type)"},
                "severity": {
                    "type": "string",
                    "enum": ["critical", "warning", "info"],
                    "description": "critical=red, warning=yellow, info=blue",
                },
                "note": {
                    "type": "string",
                    "description": "Explanation text shown in sidebar card and tooltip",
                },
            },
            "required": ["type", "label", "start_ns", "severity"],
        },
    },
}

# Per-GPU compute/NCCL overlap breakdown for multi-GPU analysis.
TOOL_GET_GPU_OVERLAP_STATS = {
    "type": "function",
    "function": {
        "name": "get_gpu_overlap_stats",
        "description": (
            "Compute per-GPU breakdown of compute vs NCCL communication overlap. "
            "Returns for EACH GPU: compute_only_ms, nccl_only_ms, overlap_ms, "
            "idle_ms, overlap_pct (fraction of NCCL time hidden behind compute). "
            "Use to diagnose: "
            "(1) GPU imbalance — compare compute_only_ms across GPUs; "
            "(2) NCCL hiding efficiency — overlap_pct >60% means well-hidden; "
            "(3) idle bubbles between kernels. "
            "Optional time range for per-iteration analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_s": {
                    "type": "number",
                    "description": "Optional start time in seconds (omit for full profile).",
                },
                "end_s": {
                    "type": "number",
                    "description": "Optional end time in seconds (omit for full profile).",
                },
            },
            "required": [],
        },
    },
}

# NCCL collective breakdown by type (AllReduce, AllGather, ReduceScatter, etc.).
TOOL_GET_NCCL_BREAKDOWN = {
    "type": "function",
    "function": {
        "name": "get_nccl_breakdown",
        "description": (
            "Break down NCCL operations by CUDA stream and collective type (AllReduce, AllGather, "
            "ReduceScatter, SendRecv, Broadcast). Returns stream_id, count, total_ms, avg_ms, "
            "pct for each (stream, collective) pair. Use stream_id to infer parallelism strategy: "
            "different streams = different parallelism dimensions (TP/PP/DP). "
            "AllReduce=DDP, ReduceScatter+AllGather=FSDP/ZeRO, SendRecv=Pipeline Parallel."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "integer",
                    "description": "GPU device ID (optional, default: first GPU).",
                },
                "start_s": {
                    "type": "number",
                    "description": "Optional start time in seconds.",
                },
                "end_s": {
                    "type": "number",
                    "description": "Optional end time in seconds.",
                },
            },
            "required": [],
        },
    },
}

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------


def _tools_openai() -> list[dict]:
    """Return the OpenAI-style tool list for single-profile chat."""
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
                        "end_s": {"type": "number", "description": "End time in seconds"},
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
        TOOL_COMPUTE_REGION_MFU,
        TOOL_COMPUTE_THEORETICAL_FLOPS,
        TOOL_SUBMIT_FINDING,
        TOOL_GET_GPU_OVERLAP_STATS,
        TOOL_GET_NCCL_BREAKDOWN,
    ]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _load_prompt_files() -> tuple[str | None, str | None]:
    """Load chat_system and MFU reference prompt files once, with caching.

    Returns:
        A tuple ``(chat_system, mfu_ref)`` where each element may be ``None`` if
        the corresponding file could not be read.
    """
    prompts_dir = Path(__file__).resolve().parent.parent.parent / "prompts"
    chat_system: str | None = None
    mfu_ref: str | None = None

    # Load each file independently so a failure in one doesn't drop the other.
    try:
        chat_system = (prompts_dir / "chat_system.txt").read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        logger.warning("chat_system.txt not found in %s: %s", prompts_dir, exc)
    except OSError as exc:
        logger.error("Error reading chat_system.txt from %s: %s", prompts_dir, exc)

    try:
        mfu_ref = (prompts_dir / "mfu_reference.txt").read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        logger.warning("mfu_reference.txt not found in %s: %s", prompts_dir, exc)
    except OSError as exc:
        logger.error("Error reading mfu_reference.txt from %s: %s", prompts_dir, exc)

    return (chat_system, mfu_ref)


def _build_system_prompt(
    ui_context: dict,
    profile_schema: str | None = None,
    skill_docs: str | None = None,
) -> str:
    """Build the system prompt that instructs the LLM on its role and tools.

    Args:
        ui_context:      JSON-serialisable dict with visible kernel summary,
                         selected stream, time range, etc.
        profile_schema:  Optional schema string from ``get_profile_schema_cached``.
                         When provided the LLM is instructed to use
                         ``query_profile_db`` for whole-profile questions.
        skill_docs:      Optional pre-loaded skill content to append at the end.
                         Use ``prompt_loader.load_skill_context(["skills/mfu.md"])``
                         to inject a specific skill for the current session.
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
    chat_system, mfu_ref = _load_prompt_files()

    if chat_system is not None:
        # Use the template from the prompt file when available.
        chat_system = chat_system.replace("{schema_block}", schema_block)
        chat_system = chat_system.replace("{ctx_json}", ctx_json)
        base_prompt = chat_system
    else:
        # Fallback minimal prompt if the prompt files could not be loaded.
        logger.warning("Using fallback system prompt because prompt files could not be loaded.")
        base_prompt = (
            "You are an AI assistant helping users analyze performance profiles and MFU.\n"
            "Respond with clear, concise, and technically accurate guidance.\n"
            f"{schema_block}"
            "=== UI CONTEXT (JSON) ===\n"
            f"{ctx_json}\n"
            "=====================================================\n"
        )

    parts: list[str] = [base_prompt]
    if mfu_ref:
        parts.append(mfu_ref)

    if skill_docs:
        parts.append(
            f"\n=== SESSION SKILL CONTEXT ===\n{skill_docs}\n=== END SESSION SKILL CONTEXT ===\n"
        )

    return "\n".join(parts)


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
