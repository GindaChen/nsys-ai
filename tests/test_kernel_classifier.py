"""Tests for nsys_ai.cutracer.kernel_classifier."""



class TestClassifyKernel:
    def _classify(self, name):
        from nsys_ai.cutracer.kernel_classifier import classify_kernel
        return classify_kernel(name)

    # ── NCCL ────────────────────────────────────────────────────────────────

    def test_nccl_allgather(self):
        cat, val, reason = self._classify(
            "ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<...>)"
        )
        assert cat == "nccl_comms"
        assert val == "LOW"

    def test_nccl_reducescatter(self):
        cat, val, _ = self._classify(
            "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<...>)"
        )
        assert cat == "nccl_comms"
        assert val == "LOW"

    def test_nccl_allreduce(self):
        cat, val, _ = self._classify("ncclDevKernel_AllReduce_Sum_f32_TREE_LL128(...)")
        assert cat == "nccl_comms"
        assert val == "LOW"

    # ── elementwise ─────────────────────────────────────────────────────────

    def test_elementwise_kernel(self):
        cat, val, _ = self._classify(
            "void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl...>"
        )
        assert cat == "elementwise"
        assert val == "LOW"

    def test_unary_kernel(self):
        cat, val, _ = self._classify(
            "void at::native::unary_kernel<float>(at::TensorIteratorBase&)"
        )
        assert cat == "elementwise"
        assert val == "LOW"

    # ── Flash Attention ──────────────────────────────────────────────────────

    def test_flash_fwd(self):
        cat, val, _ = self._classify(
            "void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64>>"
        )
        assert cat == "flash_attn"
        assert val == "MEDIUM"

    def test_flash_bwd(self):
        cat, val, _ = self._classify(
            "void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<...>>"
        )
        assert cat == "flash_attn"
        assert val == "MEDIUM"

    # ── Custom GEMM ──────────────────────────────────────────────────────────

    def test_nvjet(self):
        cat, val, _ = self._classify("nvjet_tst_128x320_64x3_2x1_v_bz_coopB_NNN")
        assert cat == "custom_gemm"
        assert val == "HIGH"

    def test_nvjet_variant(self):
        cat, val, _ = self._classify("nvjet_tst_192x192_64x4_2x1_v_bz_coopB_bias_TNN")
        assert cat == "custom_gemm"
        assert val == "HIGH"

    # ── Vendor libraries ─────────────────────────────────────────────────────

    def test_cudnn(self):
        cat, val, _ = self._classify(
            "void cudnn::detail::implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, false, true>(...)"
        )
        assert cat == "vendor_lib"
        assert val == "LOW"

    # ── Unknown ──────────────────────────────────────────────────────────────

    def test_unknown_kernel(self):
        cat, val, _ = self._classify("my_custom_reduction_kernel")
        assert cat == "unknown"
        assert val == "HIGH"

    def test_unknown_complex_name(self):
        cat, val, _ = self._classify(
            "void custom_attention_forward<(int)32>(float*, float*, int)"
        )
        assert cat == "unknown"
        assert val == "HIGH"


class TestInstrumentationPriority:
    def test_high_before_medium(self):
        from nsys_ai.cutracer.kernel_classifier import (
            VALUE_HIGH,
            VALUE_MEDIUM,
            instrumentation_priority,
        )
        assert instrumentation_priority(VALUE_HIGH) < instrumentation_priority(VALUE_MEDIUM)

    def test_medium_before_low(self):
        from nsys_ai.cutracer.kernel_classifier import (
            VALUE_LOW,
            VALUE_MEDIUM,
            instrumentation_priority,
        )
        assert instrumentation_priority(VALUE_MEDIUM) < instrumentation_priority(VALUE_LOW)


class TestBuildPlanClassification:
    """Verify that build_plan correctly separates HIGH/MEDIUM from LOW kernels."""

    def test_nccl_goes_to_skipped(self, minimal_nsys_conn):
        """NCCL kernels in the profile should land in plan.skipped, not plan.targets."""
        from unittest.mock import patch

        from nsys_ai.cutracer.planner import build_plan

        fake_rows = [
            {"kernel_name": "ncclDevKernel_AllGather_RING_LL(...)", "total_ms": 500.0, "invocations": 100},
            {"kernel_name": "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(...)", "total_ms": 300.0, "invocations": 100},
            {"kernel_name": "nvjet_tst_128x320_64x3_2x1_v_bz_coopB_NNN", "total_ms": 200.0, "invocations": 50},
        ]

        with patch("nsys_ai.skills.builtins.top_kernels.SKILL") as mock_skill:
            mock_skill.execute_fn.return_value = fake_rows
            plan = build_plan(minimal_nsys_conn, "/fake/p.sqlite", top_n=5)

        target_names = [t.name for t in plan.targets]
        skipped_names = [t.name for t in plan.skipped]

        assert "nvjet_tst_128x320_64x3_2x1_v_bz_coopB_NNN" in target_names
        assert "ncclDevKernel_AllGather_RING_LL(...)" in skipped_names
        assert "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(...)" in skipped_names

    def test_high_fills_slots_before_medium(self, minimal_nsys_conn):
        """HIGH-value kernels should be selected before MEDIUM even if MEDIUM has more GPU time."""
        from unittest.mock import patch

        from nsys_ai.cutracer.planner import build_plan

        fake_rows = [
            # MEDIUM value but highest GPU time
            {"kernel_name": "void flash::flash_fwd_kernel<...>", "total_ms": 900.0, "invocations": 500},
            # HIGH value but lower GPU time
            {"kernel_name": "nvjet_tst_128x320_64x3_2x1_v_bz_coopB_NNN", "total_ms": 100.0, "invocations": 50},
        ]

        with patch("nsys_ai.skills.builtins.top_kernels.SKILL") as mock_skill:
            mock_skill.execute_fn.return_value = fake_rows
            plan = build_plan(minimal_nsys_conn, "/fake/p.sqlite", top_n=1)

        # top_n=1: only one slot — HIGH value nvjet should win over MEDIUM flash
        assert len(plan.targets) == 1
        assert "nvjet_tst" in plan.targets[0].name

    def test_kernel_target_has_category(self, minimal_nsys_conn):
        from nsys_ai.cutracer.planner import build_plan

        plan = build_plan(minimal_nsys_conn, "/fake/p.sqlite")
        for t in plan.targets + plan.skipped:
            assert t.category != ""
            assert t.cutracer_value in ("HIGH", "MEDIUM", "LOW")
