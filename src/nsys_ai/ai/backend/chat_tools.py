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
                        "attention", "qkv_proj", "output_proj",
                        "mlp", "full_layer", "full_model", "linear",
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
            "Break down NCCL operations by collective type (AllReduce, AllGather, "
            "ReduceScatter, SendRecv, Broadcast). Returns count, total_ms, avg_ms, "
            "pct for each collective. Use to infer parallelism strategy: "
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
        "5. NEVER REFUSE to calculate MFU when the user asks. Even if the result is approximate or the time "
        "covers only a single kernel, compute it. Use compute_region_mfu with source='kernel' to get "
        "kernel execution time directly. The user can judge whether the result is meaningful.\n"
        "6. For whole-step MFU: (1) Call get_gpu_peak_tflops to get peak_tflops from the profile GPU. "
        "   (2) Use query_profile_db to get step_time_s (e.g. (MAX([end])-MIN(start))/1e9). "
        "   (3) Ask the user for model_flops_per_step (nsys does not store it). Do NOT call compute_mfu until the user "
        "has provided it — after asking, end your response and wait for their reply; only then call compute_mfu with that value. "
        "If get_gpu_peak_tflops returns an error, ask the user for peak_tflops as well.\n"
        "7. For MFU of a specific NVTX region or kernel: use compute_region_mfu. "
        "   - For NVTX ranges (e.g. 'Forward Pass'): set source='nvtx', name=<nvtx_text>. "
        "   - For kernels (e.g. 'flash_fwd_kernel'): set source='kernel', name=<kernel_name>. "
        "   The tool handles both modes. Provide theoretical_flops and optional peak_tflops / num_gpus.\n"
        "   KERNEL NAME TIPS: The name parameter uses substring matching (LIKE '%%name%%').\n"
        "   - Use SHORT technical names: 'flash' (not 'flash attention kernel'), 'gemm', 'nccl'.\n"
        "   - If KERNEL_NOT_FOUND, retry with a shorter/broader keyword.\n"
        "   - When unsure of exact name, use query_profile_db first to discover kernel names:\n"
        "     SELECT DISTINCT s.value FROM StringIds s JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.shortName=s.id WHERE s.value LIKE '%%flash%%'\n"
        "   IMPORTANT: Use compute_theoretical_flops to compute FLOPs — do NOT compute manually.\n"
        "   Workflow: (1) compute_theoretical_flops → get exact FLOPs, (2) compute_region_mfu with that value.\n"
        "8. AUTONOMY: When a skill workflow is loaded at the end of this prompt, execute ALL steps in sequence "
        "without pausing for user confirmation between steps. Only stop mid-workflow if: (a) a tool returns an "
        "error that needs user action, (b) you need model architecture parameters the user hasn't provided, or "
        "(c) the user explicitly asks you to pause. Do not ask 'shall I proceed?' — just proceed.\n"
        "9. EFFICIENCY: You have a limited tool-call budget per question. Prefer fewer, broader SQL queries "
        "over many narrow ones. When a workflow requires multiple queries (e.g. triage steps 2-4), batch them "
        "into a single tool call round using parallel tool calls. Never run more than 3 separate "
        "query_profile_db calls when you could combine them into one.\n"
        "10. VISUAL EVIDENCE: When you identify a bottleneck, stall, idle gap, or anomaly, "
        "call `submit_finding` to overlay it on the timeline. Get start_ns/end_ns from "
        "query_profile_db (kernel start/[end] columns are in nanoseconds). After submitting, "
        "reference it as [Finding N] (N = the returned index number) in your text so the user "
        "can click it to zoom to the evidence.\n"
        "11. MULTI-GPU ANALYSIS: When asked about GPU imbalance, NCCL overlap, or "
        "communication overhead:\n"
        "   - Call `get_gpu_overlap_stats` to get per-GPU compute/nccl/overlap/idle breakdown.\n"
        "   - Call `get_nccl_breakdown` to identify collective types and infer parallelism strategy.\n"
        "   - Compare compute_only_ms across GPUs to detect imbalance (max/min ratio > 1.2 = imbalanced).\n"
        "   - overlap_pct > 60% = NCCL well-hidden; < 30% = serialized with compute.\n"
        "   - After analysis, call `submit_finding` for any GPU with unusual overlap_pct or idle_ms.\n"
        "\n"
        "=== MFU REFERENCE (for choosing the right operation in compute_theoretical_flops) ===\n"
        "The nsys profile does NOT store model FLOPs — you must calculate them from model architecture.\n"
        "CRITICAL: theoretical_flops must match ONLY the computation the target kernel/region performs.\n\n"
        "## 1. CORE PRINCIPLE — Match FLOPs to the Kernel\n"
        "  If the user asks for the MFU of a SPECIFIC kernel, compute ONLY the FLOPs that kernel does.\n"
        "  Do NOT use the full-model FLOPs for a single kernel's MFU.\n"
        "  Example: flash_fwd_kernel only does attention matmuls (QK^T + softmax*V),\n"
        "    so use 4*S*S*H per layer — NOT the full transformer layer FLOPs.\n\n"
        "## 2. Common Kernel → FLOPs Mapping (per layer, forward only)\n"
        "  Variables: H=hidden_dim, S=seq_len, L=num_layers, ffn=ffn_dim, head_dim=H/num_heads\n"
        "  | Kernel type                     | What it computes        | FLOPs per layer         |\n"
        "  |--------------------------------|-------------------------|-------------------------|\n"
        "  | Attention matmul (flash_fwd)   | QK^T + softmax*V        | 4 * S * S * H           |\n"
        "  | GEMM / linear projection       | Matrix multiply W*x     | 2 * M * N * K           |\n"
        "  | QKV projection                 | Linear proj for Q,K,V   | 6 * S * H * H           |\n"
        "  | Output projection              | Linear proj after attn  | 2 * S * H * H           |\n"
        "  | MLP / FFN                      | Up + down projection    | 4 * S * H * ffn         |\n"
        "  Total for all layers: multiply per-layer by L.\n"
        "  For fwd+bwd: multiply by 3 (no checkpointing) or 4 (with checkpointing).\n\n"
        "## 3. Full Model FLOPs (use for whole-step or NVTX-wrapped regions)\n"
        "  Transformer per-layer FLOPs (forward, batch=1):\n"
        "    flops_per_layer = 8*H*H*S + 4*H*ffn*S + 4*S*S*H  (self-attn + MLP)\n"
        "  Full step:\n"
        "    theoretical_flops = batch_size * flops_per_layer * L * multiplier * grad_accum\n"
        "  Quick estimate: flops_per_step ≈ 6 * N_params * tokens_per_step (fwd+bwd)\n\n"
        "## 4. Multi-GPU\n"
        "  Pass num_gpus=world_size to compute_region_mfu. Peak is scaled automatically.\n\n"
        "## 5. SANITY CHECK (MANDATORY)\n"
        "  After computing MFU, check the result:\n"
        "  - MFU > 100%  → theoretical_flops is TOO HIGH. You likely used full-model FLOPs\n"
        "                   for a single kernel. Recalculate with only that kernel's FLOPs.\n"
        "  - MFU < 0.1%  → theoretical_flops may be TOO LOW, or the kernel barely ran.\n"
        "  - MFU 10-80%  → typical reasonable range for compute-bound kernels.\n"
        "  If MFU > 100%, do NOT report it as-is. Recompute with correct FLOPs and explain.\n"
        "=============================\n"
        + (
            "\n=== SESSION SKILL CONTEXT ===\n"
            + skill_docs
            + "\n=== END SESSION SKILL CONTEXT ===\n"
            if skill_docs
            else ""
        )
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
