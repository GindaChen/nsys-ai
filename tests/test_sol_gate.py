"""Tests for the absolute speed-of-light CI gate (#204).

The behaviour that matters most here is that a misconfigured gate fails loudly:
a CI gate that silently does not run reports green on a real regression, which
is worse than having no gate at all.
"""

import pytest

from nsys_ai.sol_gate import (
    SolGateError,
    SolGateResult,
    evaluate_sol_gates,
    parse_sol_gate,
    resolve_theoretical_flops,
)

# ── spec parsing ─────────────────────────────────────────────────────


class TestParse:
    def test_basic(self):
        spec = parse_sol_gate("attn:60")
        assert spec.region == "attn"
        assert spec.threshold_pct == 60.0

    def test_region_may_contain_colons(self):
        """Kernel names routinely contain '::', so split on the last colon."""
        spec = parse_sol_gate("at::native::gemm:72.5")
        assert spec.region == "at::native::gemm"
        assert spec.threshold_pct == 72.5

    @pytest.mark.parametrize(
        "bad",
        ["attn", "", "attn:", ":60", "attn:abc", "attn:0", "attn:-5", "attn:101"],
    )
    def test_rejects_malformed(self, bad):
        with pytest.raises(SolGateError):
            parse_sol_gate(bad)

    def test_error_message_shows_expected_form(self):
        with pytest.raises(SolGateError, match="REGION:PCT"):
            parse_sol_gate("attn")


# ── the FLOPs resolver seam ──────────────────────────────────────────


class TestResolveFlops:
    def test_cli_value_wins(self):
        assert resolve_theoretical_flops(2.4e13) == 2.4e13

    def test_missing_flops_raises_rather_than_skipping(self):
        """The core contract: never degrade to 'gate passed'."""
        with pytest.raises(SolGateError, match="theoretical-flops"):
            resolve_theoretical_flops(None)

    def test_missing_flops_message_explains_why_it_refuses(self):
        with pytest.raises(SolGateError, match="silently does not run"):
            resolve_theoretical_flops(None)

    def test_rejects_non_positive(self):
        with pytest.raises(SolGateError):
            resolve_theoretical_flops(0)
        with pytest.raises(SolGateError):
            resolve_theoretical_flops(-1.0)


# ── evaluation against region_mfu ────────────────────────────────────


class _FakeSkill:
    """Stands in for region_mfu so the gate logic is tested without a profile."""

    def __init__(self, row):
        self._row = row
        self.calls = []

    def execute(self, conn, **kwargs):
        self.calls.append(kwargs)
        return [self._row]


def _mfu_row(mfu_pct, union_s=0.1):
    return {
        "name": "attn",
        "matched_text": "attn",
        "gpu_kernel_union_s": union_s,
        "mfu_pct_kernel_union": mfu_pct,
        "achieved_tflops_kernel_union": 100.0,
        "peak_tflops": 300.0,
    }


def _patch_skill(monkeypatch, skill):
    monkeypatch.setattr("nsys_ai.skills.registry.get_skill", lambda name: skill)


class TestEvaluate:
    def test_passes_above_threshold(self, monkeypatch):
        _patch_skill(monkeypatch, _FakeSkill(_mfu_row(72.0)))
        (r,) = evaluate_sol_gates(
            None, [parse_sol_gate("attn:60")], theoretical_flops=1e13
        )
        assert r.passed is True
        assert r.mfu_pct == 72.0
        assert r.headroom_ms == pytest.approx(28.0)  # 100ms * (1 - 0.72)

    def test_fails_below_threshold(self, monkeypatch):
        _patch_skill(monkeypatch, _FakeSkill(_mfu_row(41.0)))
        (r,) = evaluate_sol_gates(
            None, [parse_sol_gate("attn:60")], theoretical_flops=1e13
        )
        assert r.passed is False

    def test_exactly_at_threshold_passes(self, monkeypatch):
        """'stay at or above' — the boundary is inclusive."""
        _patch_skill(monkeypatch, _FakeSkill(_mfu_row(60.0)))
        (r,) = evaluate_sol_gates(
            None, [parse_sol_gate("attn:60")], theoretical_flops=1e13
        )
        assert r.passed is True

    def test_unmeasurable_region_raises_rather_than_passing(self, monkeypatch):
        """A region that cannot be measured must fail the gate, not pass by omission."""
        _patch_skill(monkeypatch, _FakeSkill({"error": {"message": "region not found"}}))
        with pytest.raises(SolGateError, match="could not measure"):
            evaluate_sol_gates(None, [parse_sol_gate("nope:60")], theoretical_flops=1e13)

    def test_peak_tflops_is_forwarded_when_given(self, monkeypatch):
        skill = _FakeSkill(_mfu_row(80.0))
        _patch_skill(monkeypatch, skill)
        evaluate_sol_gates(
            None, [parse_sol_gate("attn:60")], theoretical_flops=1e13, peak_tflops=989.0
        )
        assert skill.calls[0]["peak_tflops"] == 989.0
        assert skill.calls[0]["theoretical_flops"] == 1e13

    def test_peak_tflops_omitted_lets_autodetect_run(self, monkeypatch):
        skill = _FakeSkill(_mfu_row(80.0))
        _patch_skill(monkeypatch, skill)
        evaluate_sol_gates(None, [parse_sol_gate("attn:60")], theoretical_flops=1e13)
        assert "peak_tflops" not in skill.calls[0]

    def test_multiple_targets_evaluated_independently(self, monkeypatch):
        rows = {"attn": _mfu_row(80.0), "mlp": _mfu_row(30.0)}

        class _MultiSkill:
            def execute(self, conn, **kwargs):
                return [rows[kwargs["name"]]]

        _patch_skill(monkeypatch, _MultiSkill())
        results = evaluate_sol_gates(
            None,
            [parse_sol_gate("attn:60"), parse_sol_gate("mlp:60")],
            theoretical_flops=1e13,
        )
        assert [r.passed for r in results] == [True, False]


# ── result serialization ─────────────────────────────────────────────


def test_result_to_dict_round_trips():
    d = SolGateResult(
        region="attn", threshold_pct=60.0, mfu_pct=41.234, headroom_ms=12.3456, passed=False
    ).to_dict()
    assert d == {
        "region": "attn",
        "threshold_pct": 60.0,
        "mfu_pct": 41.234,
        "passed": False,
        "headroom_ms": 12.346,
    }


def test_result_to_dict_omits_absent_headroom():
    d = SolGateResult(
        region="attn", threshold_pct=60.0, mfu_pct=100.0, headroom_ms=None, passed=True
    ).to_dict()
    assert "headroom_ms" not in d


# ── CLI wiring ───────────────────────────────────────────────────────


class TestCliWiring:
    """The gate's value is that it fails the process; these pin the exit codes
    so a misconfiguration can never be mistaken for a pass."""

    def _run(self, *extra):
        import subprocess
        import sys

        return subprocess.run(
            [sys.executable, "-m", "nsys_ai", "diff",
             "tests/fixtures/mock.sqlite", "tests/fixtures/mock.sqlite",
             "--format", "json", *extra],
            capture_output=True, text=True,
        )

    def test_missing_flops_exits_2_and_does_not_silently_pass(self):
        r = self._run("--gate-sol", "attn:60")
        assert r.returncode == 2, "a gate that cannot run must not report success"
        assert "theoretical-flops" in r.stderr

    def test_malformed_spec_exits_2(self):
        r = self._run("--gate-sol", "attn", "--theoretical-flops", "1e13")
        assert r.returncode == 2
        assert "REGION:PCT" in r.stderr

    def test_out_of_range_threshold_exits_2(self):
        r = self._run("--gate-sol", "attn:150", "--theoretical-flops", "1e13")
        assert r.returncode == 2

    def test_unmeasurable_region_exits_2_not_0(self):
        r = self._run(
            "--gate-sol", "definitely_not_a_region:60",
            "--theoretical-flops", "1e13", "--peak-tflops", "989",
        )
        assert r.returncode == 2
        assert r.returncode != 0

    def test_no_gate_sol_leaves_diff_unaffected(self):
        assert self._run().returncode == 0


# ── holistic-review regressions ──────────────────────────────────────


class TestMeasurementScope:
    """The gate's verdict must not move for reasons unrelated to the code under
    test, so every parameter that changes *what is measured* is explicit."""

    def test_scope_parameters_are_forwarded_not_defaulted(self, monkeypatch):
        skill = _FakeSkill(_mfu_row(80.0))
        _patch_skill(monkeypatch, skill)
        evaluate_sol_gates(
            None,
            [parse_sol_gate("attn:60")],
            theoretical_flops=1e13,
            device_id=3,
            occurrence_index=7,
            num_gpus=8,
        )
        call = skill.calls[0]
        assert call["device_id"] == 3
        assert call["occurrence_index"] == 7
        assert call["num_gpus"] == 8

    def test_device_id_omitted_when_not_scoped(self, monkeypatch):
        skill = _FakeSkill(_mfu_row(80.0))
        _patch_skill(monkeypatch, skill)
        evaluate_sol_gates(None, [parse_sol_gate("attn:60")], theoretical_flops=1e13)
        assert "device_id" not in skill.calls[0]


class TestImplausibleMfu:
    """MFU above the hardware ceiling means the inputs do not describe the
    region — the one thing this gate must never report as a pass."""

    def test_mfu_above_ceiling_is_an_error_not_a_pass(self, monkeypatch):
        _patch_skill(monkeypatch, _FakeSkill(_mfu_row(3538.0)))
        with pytest.raises(SolGateError, match="implausible MFU"):
            evaluate_sol_gates(None, [parse_sol_gate("attn:60")], theoretical_flops=1e13)

    def test_error_names_the_likely_cause(self, monkeypatch):
        _patch_skill(monkeypatch, _FakeSkill(_mfu_row(150.0)))
        with pytest.raises(SolGateError, match="theoretical-flops"):
            evaluate_sol_gates(None, [parse_sol_gate("attn:60")], theoretical_flops=1e13)

    def test_exactly_at_ceiling_is_still_measurable(self, monkeypatch):
        _patch_skill(monkeypatch, _FakeSkill(_mfu_row(100.0)))
        (r,) = evaluate_sol_gates(
            None, [parse_sol_gate("attn:60")], theoretical_flops=1e13
        )
        assert r.passed is True


class TestNonFiniteInputs:
    """NaN slips past every ordered comparison, so it needs explicit exclusion —
    otherwise a broken measurement is indistinguishable from a regression."""

    def test_nan_and_inf_flops_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(SolGateError, match="finite"):
                resolve_theoretical_flops(bad)

    @pytest.mark.parametrize("bad_mfu", [float("nan"), float("inf"), -5.0])
    def test_non_measurable_mfu_is_an_error_not_a_gate_failure(self, monkeypatch, bad_mfu):
        """Must raise (exit 2, misconfigured) rather than report passed=False
        (exit 1, regressed) — CI has to be able to tell those apart."""
        _patch_skill(monkeypatch, _FakeSkill(_mfu_row(bad_mfu)))
        with pytest.raises(SolGateError, match="non-measurable"):
            evaluate_sol_gates(None, [parse_sol_gate("attn:60")], theoretical_flops=1e13)
