"""Tests for nsys_ai.cutracer Phase 0: parser, sass_ops, correlator, report."""

from __future__ import annotations

from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "cutracer"


# ---------------------------------------------------------------------------
# sass_ops
# ---------------------------------------------------------------------------


class TestSassOps:
    def test_classify_compute(self):
        from nsys_ai.cutracer.sass_ops import classify_opcode

        assert classify_opcode("FFMA") == "compute"
        assert classify_opcode("IMAD") == "compute"
        assert classify_opcode("FFMA.FTZ") == "compute"

    def test_classify_tensor(self):
        from nsys_ai.cutracer.sass_ops import classify_opcode

        assert classify_opcode("HMMA") == "tensor"
        assert classify_opcode("IMMA") == "tensor"

    def test_classify_memory(self):
        from nsys_ai.cutracer.sass_ops import classify_opcode

        assert classify_opcode("LDG") == "memory"
        assert classify_opcode("STG") == "memory"
        assert classify_opcode("LDS") == "memory"
        assert classify_opcode("LDGSTS") == "memory"

    def test_classify_sync(self):
        from nsys_ai.cutracer.sass_ops import classify_opcode

        assert classify_opcode("BAR") == "sync"
        assert classify_opcode("MEMBAR") == "sync"

    def test_classify_control(self):
        from nsys_ai.cutracer.sass_ops import classify_opcode

        assert classify_opcode("BRA") == "control"
        assert classify_opcode("RET") == "control"

    def test_classify_unknown(self):
        from nsys_ai.cutracer.sass_ops import classify_opcode

        assert classify_opcode("UNKNOWNOP") == "other"

    def test_stall_score_no_stall(self):
        from nsys_ai.cutracer.sass_ops import stall_score

        # FFMA ideal ~4 cycles; actual 4 → no stall
        assert stall_score("FFMA", total_cycles=4000, total_count=1000) == 0.0

    def test_stall_score_high_ldg(self):
        from nsys_ai.cutracer.sass_ops import stall_score

        # LDG with 200 cycles/instr (ideal 30) → heavy stall
        score = stall_score("LDG", total_cycles=200_000, total_count=1000)
        assert score > 0.5

    def test_stall_score_zero_count(self):
        from nsys_ai.cutracer.sass_ops import stall_score

        assert stall_score("LDG", total_cycles=1000, total_count=0) == 0.0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


class TestParser:
    def test_parse_histogram_csv(self):
        from nsys_ai.cutracer.parser import parse_histogram_csv

        f = FIXTURE_DIR / "kernel_volta_h16gemm_32x32_nt_nn_abc12345_hist.csv"
        hist = parse_histogram_csv(f)

        assert hist is not None
        assert hist.kernel_name == "volta_h16gemm_32x32_nt_nn"
        # 4 warps × several opcodes each
        assert hist.warp_count == 4
        assert "FFMA" in hist.instruction_counts
        assert "LDG" in hist.instruction_counts
        assert "HMMA" in hist.instruction_counts

        # counts should be aggregated across all warps
        assert hist.instruction_counts["FFMA"] > 12000
        assert hist.total_count > 0
        assert hist.total_cycles > 0

    def test_parse_histogram_dir(self):
        from nsys_ai.cutracer.parser import parse_histogram_dir

        histograms = parse_histogram_dir(FIXTURE_DIR)

        assert len(histograms) == 2
        assert "volta_h16gemm_32x32_nt_nn" in histograms
        assert "cutlass_gemm_s8" in histograms

    def test_parse_nonexistent_dir(self):
        from nsys_ai.cutracer.parser import parse_histogram_dir

        result = parse_histogram_dir(Path("/nonexistent/path"))
        assert result == {}

    def test_parse_empty_dir(self, tmp_path):
        from nsys_ai.cutracer.parser import parse_histogram_dir

        result = parse_histogram_dir(tmp_path)
        assert result == {}

    def test_parse_malformed_csv(self, tmp_path):
        from nsys_ai.cutracer.parser import parse_histogram_csv

        bad = tmp_path / "kernel_test_abc_hist.csv"
        bad.write_text("this,is,not,valid\ngarbage,,,\n")
        # Should not raise; returns None or empty histogram
        hist = parse_histogram_csv(bad)
        # Either None or zero-count histogram is acceptable
        assert hist is None or hist.total_count == 0

    def test_cycles_per_instruction(self):
        from nsys_ai.cutracer.parser import parse_histogram_csv

        f = FIXTURE_DIR / "kernel_volta_h16gemm_32x32_nt_nn_abc12345_hist.csv"
        hist = parse_histogram_csv(f)

        ldg_cpi = hist.cycles_per_instruction("LDG")
        ffma_cpi = hist.cycles_per_instruction("FFMA")

        # LDG should have much higher CPI than FFMA (memory latency)
        assert ldg_cpi > ffma_cpi
        assert ldg_cpi > 50  # HBM-level latency in fixture

    def test_parse_histogram_hash_name_filename(self, tmp_path):
        from nsys_ai.cutracer.parser import parse_histogram_csv

        f = tmp_path / "kernel_74397ef380c2e51_volta_sgemm_128x64_nn_hist.csv"
        f.write_text(
            "warp_id,region_id,instruction,count,cycles\n"
            "0,0,FFMA,10,40\n"
            "0,0,LDG,5,150\n"
        )
        hist = parse_histogram_csv(f)

        assert hist is not None
        assert hist.kernel_name == "volta_sgemm_128x64_nn"
        assert hist.instruction_counts["FFMA"] == 10


# ---------------------------------------------------------------------------
# correlator
# ---------------------------------------------------------------------------


class TestCorrelator:
    def test_normalize_strips_templates(self):
        from nsys_ai.cutracer.correlator import normalize_kernel_name

        name = "void volta_h16gemm_32x32_nt_nn<bool, int, 128>(cublas::...)"
        result = normalize_kernel_name(name)
        assert "volta_h16gemm_32x32_nt_nn" in result
        assert "<" not in result
        assert "(" not in result

    def test_normalize_lowercase(self):
        from nsys_ai.cutracer.correlator import normalize_kernel_name

        assert normalize_kernel_name("FFMA_KERNEL").islower()

    def test_match_exact_after_normalize(self):
        from nsys_ai.cutracer.correlator import match_kernels

        nsys_kernels = [
            "void volta_h16gemm_32x32_nt_nn<bool, int>(cublas::gemm_args)"
        ]
        cutracer_kernels = ["volta_h16gemm_32x32_nt_nn"]

        result = match_kernels(nsys_kernels, cutracer_kernels)
        assert result[nsys_kernels[0]] == "volta_h16gemm_32x32_nt_nn"

    def test_match_substring(self):
        from nsys_ai.cutracer.correlator import match_kernels

        nsys_kernels = ["void volta_h16gemm_32x32_nt_nn_some_variant<T>(args)"]
        cutracer_kernels = ["volta_h16gemm"]

        result = match_kernels(nsys_kernels, cutracer_kernels)
        # Should find "volta_h16gemm" as substring match
        assert result[nsys_kernels[0]] == "volta_h16gemm"

    def test_match_no_match_returns_none(self):
        from nsys_ai.cutracer.correlator import match_kernels

        nsys_kernels = ["totally_unrelated_kernel_xyz"]
        cutracer_kernels = ["volta_h16gemm"]

        result = match_kernels(nsys_kernels, cutracer_kernels)
        assert result[nsys_kernels[0]] is None

    def test_match_empty_inputs(self):
        from nsys_ai.cutracer.correlator import match_kernels

        assert match_kernels([], []) == {}
        assert match_kernels(["some_kernel"], []) == {"some_kernel": None}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


class TestReport:
    def _make_histogram(self):
        from nsys_ai.cutracer.parser import parse_histogram_csv

        f = FIXTURE_DIR / "kernel_volta_h16gemm_32x32_nt_nn_abc12345_hist.csv"
        return parse_histogram_csv(f)

    def test_compute_mix_categories(self):
        from nsys_ai.cutracer.report import compute_mix

        hist = self._make_histogram()
        mix = compute_mix(hist)

        assert mix.total_count > 0
        # Should have compute, tensor, memory, sync categories
        assert "compute" in mix.category_pct or "tensor" in mix.category_pct
        assert "memory" in mix.category_pct
        # Sum of all categories ≈ 100%
        total = sum(mix.category_pct.values())
        assert abs(total - 100.0) < 1.0

    def test_tensor_core_detected(self):
        from nsys_ai.cutracer.report import compute_mix

        hist = self._make_histogram()
        mix = compute_mix(hist)
        assert mix.tc_active  # fixture has HMMA

    def test_memory_bottleneck(self):
        from nsys_ai.cutracer.report import compute_mix

        # Fixture has high LDG stall (80 cycles/instr vs ideal 30)
        hist = self._make_histogram()
        mix = compute_mix(hist)
        assert mix.bottleneck == "memory"

    def test_top_stalls_non_empty(self):
        from nsys_ai.cutracer.report import compute_mix

        hist = self._make_histogram()
        mix = compute_mix(hist)
        assert len(mix.top_stalls) > 0
        # LDG should be in top stalls
        stall_opcodes = [op for op, _ in mix.top_stalls]
        assert "LDG" in stall_opcodes

    def test_format_kernel_report(self):
        from nsys_ai.cutracer.report import KernelReport, compute_mix, format_kernel_report

        hist = self._make_histogram()
        mix = compute_mix(hist)
        kr = KernelReport(
            mix=mix,
            nsys_kernel_name="volta_h16gemm_32x32_nt_nn",
            nvtx_path="Layer 12 > Attention Backward",
            total_ms=31.5,
            pct_of_gpu=66.0,
        )
        text = format_kernel_report(kr)

        assert "volta_h16gemm" in text
        assert "Layer 12 > Attention Backward" in text
        assert "31.50 ms" in text
        assert "MEMORY" in text or "memory" in text
        assert "LDG" in text

    def test_to_dict_schema(self):
        from nsys_ai.cutracer.report import KernelReport, compute_mix, to_dict

        hist = self._make_histogram()
        mix = compute_mix(hist)
        kr = KernelReport(mix=mix, total_ms=31.5)
        d = to_dict(kr)

        assert "kernel_name" in d
        assert "bottleneck" in d
        assert "instruction_mix_pct" in d
        assert "tensor_core_active" in d
        assert isinstance(d["top_stalls"], list)


# ---------------------------------------------------------------------------
# cutracer_analysis skill (integration-level, no GPU needed)
# ---------------------------------------------------------------------------


class TestCutracerSkill:
    def test_skill_registered(self):
        from nsys_ai.skills.builtins.cutracer_analysis import SKILL

        assert SKILL.name == "cutracer_analysis"
        assert any(p.name == "trace_dir" for p in SKILL.params)

    def test_skill_missing_trace_dir(self, minimal_nsys_conn):
        from nsys_ai.skills.builtins.cutracer_analysis import SKILL

        result = SKILL.execute_fn(minimal_nsys_conn)
        assert result[0].get("error")

    def test_skill_nonexistent_trace_dir(self, minimal_nsys_conn):
        from nsys_ai.skills.builtins.cutracer_analysis import SKILL

        result = SKILL.execute_fn(minimal_nsys_conn, trace_dir="/nonexistent/path")
        assert result[0].get("error")

    def test_skill_with_fixture(self, minimal_nsys_conn):
        from nsys_ai.skills.builtins.cutracer_analysis import SKILL

        result = SKILL.execute_fn(minimal_nsys_conn, trace_dir=str(FIXTURE_DIR))

        # Should return at least 2 kernels from our 2 fixture files
        assert isinstance(result, list)
        assert len(result) >= 2

        # Each result must have required fields
        for r in result:
            assert "kernel_name" in r
            assert "bottleneck" in r
            assert "instruction_mix_pct" in r

    def test_skill_format(self, minimal_nsys_conn):
        from nsys_ai.skills.builtins.cutracer_analysis import SKILL

        result = SKILL.execute_fn(minimal_nsys_conn, trace_dir=str(FIXTURE_DIR))
        text = SKILL.format_fn(result)

        assert "Bottleneck" in text
        assert "Instruction Mix" in text

    def test_skill_trim_allows_zero_start(self, minimal_nsys_conn, monkeypatch):
        from nsys_ai.skills.builtins.cutracer_analysis import SKILL

        seen = {}

        def _fake_attr(_conn, trim=None):
            seen["trim"] = trim
            return []

        monkeypatch.setattr("nsys_ai.nvtx_attribution.attribute_kernels_to_nvtx", _fake_attr)
        SKILL.execute_fn(
            minimal_nsys_conn,
            trace_dir=str(FIXTURE_DIR),
            trim_start_ns=0,
            trim_end_ns=1_000_000,
        )
        assert seen.get("trim") == (0, 1_000_000)
