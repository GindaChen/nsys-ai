"""Classify GPU kernel names by CUTracer instrumentation value.

Kernels fall into five categories based on name patterns.  Each category
has an associated *instrumentation value* (HIGH / MEDIUM / LOW) that
``build_plan`` uses to prioritise which kernels are worth the CUTracer
overhead (10–100x slower execution).

Categories
----------
custom_gemm
    nvjet, cutlass, or custom GEMM variants.  Instruction mix is unknown
    a priori — HIGH value, CUTracer may reveal suboptimal tile choice or
    missing Tensor Core usage.

unknown
    Unrecognised name pattern.  Worth investigating — HIGH value.

flash_attn
    flash::flash_fwd / flash_bwd.  Already a memory-aware, highly-tuned
    implementation.  MEDIUM value: useful to confirm TC is active, but
    rarely actionable at the instruction level.

vendor_lib
    cuDNN, cublasLt, CUTLASS library kernels.  Closed-source — CUTracer
    can measure, but there is nothing the user can change.  LOW value.

elementwise
    PyTorch elementwise / unary / activation kernels.  Trivially
    memory-bound; the bottleneck is HBM bandwidth, not instruction mix.
    LOW value.

nccl_comms
    NCCL collective kernels (AllGather, AllReduce, ReduceScatter, …).
    Always bandwidth / latency bound by the interconnect.  CUTracer adds
    no actionable insight.  LOW value.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Value levels
# ---------------------------------------------------------------------------

VALUE_HIGH = "HIGH"
VALUE_MEDIUM = "MEDIUM"
VALUE_LOW = "LOW"

# ---------------------------------------------------------------------------
# Classification rules
# (checked in order; first match wins)
# ---------------------------------------------------------------------------

# Each entry: (category, value, reason, compiled_pattern)
_RULES: list[tuple[str, str, str, re.Pattern[str]]] = []


def _rule(category: str, value: str, reason: str, pattern: str) -> None:
    _RULES.append((category, value, reason, re.compile(pattern, re.IGNORECASE)))


# NCCL — always bandwidth/latency bound, no actionable instruction-level data
_rule("nccl_comms",  VALUE_LOW,    "NCCL collective — always bandwidth-bound",
      r"^nccl")

# PyTorch elementwise / copy / fill — trivially memory-bound (HBM bandwidth ceiling)
_rule("elementwise", VALUE_LOW,    "elementwise op — trivially memory-bound",
      r"void at::native::(?:<[^>]*>::)?(?:elementwise|unrolled_elementwise|"
      r"unary_kernel|gpu_kernel|CUDAKernel|vectorized_elementwise|"
      r"fill_kernel|copy_device_to_device|CatArrayBatchedCopy|"
      r"index_elementwise_kernel|reduce_kernel)")

# Vendor libraries — closed source, user cannot change the implementation
_rule("vendor_lib",  VALUE_LOW,    "vendor library kernel — closed source, not actionable",
      r"(?:void cudnn|cublasLt|void cutlass::gemm::kernel::DefaultGemm"
      r"|void cuda::std::)")

# Flash Attention — highly-tuned memory-aware kernel; rarely improvable further
_rule("flash_attn",  VALUE_MEDIUM, "Flash Attention — verify TC usage; rarely actionable",
      r"void flash::flash_(?:fwd|bwd)")

# Custom GEMM — instruction mix unknown, high investigative value
_rule("custom_gemm", VALUE_HIGH,   "custom GEMM — instruction mix unknown, high value",
      r"(?:^nvjet_tst_"
      r"|void cutlass::Kernel"
      r"|void gemm_"
      r"|void xmma_(?:new_)?(?:implicit_)?gemm)")

# Triton kernels — user-authored, CUTracer can guide optimisation
_rule("triton",      VALUE_HIGH,   "Triton kernel — user-authored, instruction mix valuable",
      r"triton_(?:fused|kernel|gemm|attention)")

# Normalisation / activation custom kernels (TE, Apex)
_rule("norm_kernel", VALUE_MEDIUM, "normalisation kernel — CUTracer useful if slow",
      r"(?:rmsnorm|layernorm|fused_bias_act|gelu_kernel)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_kernel(name: str) -> tuple[str, str, str]:
    """Return ``(category, value, reason)`` for a kernel name.

    Parameters
    ----------
    name:
        Full demangled kernel name as stored in the nsys profile.

    Returns
    -------
    category : str
        One of ``nccl_comms``, ``elementwise``, ``vendor_lib``,
        ``flash_attn``, ``custom_gemm``, ``triton``, ``norm_kernel``,
        ``unknown``.
    value : str
        ``HIGH``, ``MEDIUM``, or ``LOW``.
    reason : str
        Human-readable explanation for the classification.

    Examples
    --------
    >>> classify_kernel("ncclDevKernel_AllGather_RING_LL(...)")
    ('nccl_comms', 'LOW', 'NCCL collective — always bandwidth-bound')
    >>> classify_kernel("nvjet_tst_128x320_64x3_2x1_v_bz_coopB_NNN")
    ('custom_gemm', 'HIGH', 'custom GEMM — instruction mix unknown, high value')
    """
    for category, value, reason, pattern in _RULES:
        if pattern.search(name):
            return category, value, reason
    return "unknown", VALUE_HIGH, "unrecognised kernel — worth investigating"


# Priority order used when sorting candidates for plan selection.
_VALUE_ORDER = {VALUE_HIGH: 0, VALUE_MEDIUM: 1, VALUE_LOW: 2}


def instrumentation_priority(value: str) -> int:
    """Return sort key for a value level (lower = higher priority)."""
    return _VALUE_ORDER.get(value, 99)
