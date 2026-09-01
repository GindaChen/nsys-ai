"""The Tensor Core name heuristics, pinned against real kernel names.

``is_tc_eligible`` and ``uses_tc`` are decided by two regexes over the kernel
name, computed once at cache-build time and stored in ``kernels.parquet``. They
had no test, and they went stale: ``nvjet`` is what recent cuBLAS calls its
Hopper GEMMs, it matched neither pattern, and ``tensor_core_usage`` drops
non-eligible kernels in its own ``WHERE`` — so on an H100/H200 capture the
family that dominates the profile was invisible to the Tensor Core check. Had a
GEMM there fallen back to CUDA cores, nothing would have reported it.

A name-based heuristic will go stale again. These cases are the record of what
it is currently expected to know, so the next rename is a failing test rather
than a silent blind spot.
"""

import re

import pytest

from nsys_ai.parquet_cache import _TC_ACTIVE_PATTERN, _TC_ELIGIBLE_PATTERN


def _matches(pattern: str, name: str) -> bool:
    """Apply a pattern the way the DuckDB build does: unquoted, against lower()."""
    return re.search(pattern.strip("'"), name.lower()) is not None


def _stored_eligible(name: str) -> bool:
    """What the build actually writes to ``is_tc_eligible``.

    Not the eligibility regex alone: the build ORs the two patterns, so a name
    the active pattern recognises is eligible whether or not the first pattern
    also matches. Asserting the regex instead of the stored value would forbid
    an active-only marker -- a bare ``s16816`` with no ``gemm`` in the name --
    which production handles correctly today.
    """
    return _matches(_TC_ELIGIBLE_PATTERN, name) or _matches(_TC_ACTIVE_PATTERN, name)


#: (kernel name, eligible, tc-active). Names are real, from vendor libraries.
CASES = [
    # cuBLAS on Hopper. The regression this file exists for.
    ("nvjet_tst_128x128_64x4_1x2_h_bz_coopA", True, True),
    ("nvjet_hsh_256x128_64x4_2x1_v_bz_TNN", True, True),
    ("nvjet_sm90_hss_320x128", True, True),
    # Older cuBLAS/CUTLASS naming, which already worked.
    ("sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn", True, True),
    ("sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x256x8", True, True),
    ("void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm>", True, True),
    ("flash_fwd_splitkv_kernel", True, True),
    # Eligible by shape, but the name carries no evidence of a TC path. These
    # are the ones a fallback would show up as.
    ("volta_sgemm_128x64_nn", True, False),
    ("implicit_convolve_sgemm", True, False),
    # Active-only: the name carries a TC instruction marker but none of the
    # shape words. The build ORs the patterns, so this is stored eligible --
    # asserting the eligibility regex directly would have forbidden it.
    ("some_kernel_with_s16816_marker", True, True),
    # Not GEMM-shaped at all.
    ("vectorized_elementwise_kernel", False, False),
    ("ncclDevKernel_AllReduce_Sum_f32_RING_LL", False, False),
    ("at::native::reduce_kernel", False, False),
]


@pytest.mark.parametrize("name, eligible, active", CASES)
def test_tensor_core_name_heuristics(name, eligible, active):
    """``eligible`` is the stored column, which is the OR of the two patterns."""
    assert _stored_eligible(name) is eligible
    assert _matches(_TC_ACTIVE_PATTERN, name) is active


@pytest.mark.parametrize("name, eligible, active", CASES)
def test_tc_active_implies_stored_eligible(name, eligible, active):
    """A kernel cannot be recorded as using Tensor Cores while ineligible for them.

    This holds by construction — the build ORs the patterns — so it is here to
    catch a future edit that computes the two columns independently, not to
    constrain which pattern a name matches.
    """
    if _matches(_TC_ACTIVE_PATTERN, name):
        assert _stored_eligible(name)


def test_nvjet_is_active_not_merely_eligible():
    """Stated as its own case because the alternative is a tempting mistake.

    Adding ``nvjet`` to the eligibility pattern alone would make every Hopper
    cuBLAS GEMM report 0% Tensor Core usage — reading as a total fallback on a
    healthy profile, which is a louder wrong answer than the silence it replaces.
    These are warpgroup (wgmma) kernels; ``nvjet_sm90_hss_320x128`` measures
    around 77% of an H100's dense FP16 peak, which no CUDA-core path reaches.
    """
    assert _matches(_TC_ACTIVE_PATTERN, "nvjet_sm90_hss_320x128")
