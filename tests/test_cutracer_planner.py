"""Tests for nsys_ai.cutracer.planner — CUTracer plan generation."""

from __future__ import annotations

import sqlite3

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_conn(tmp_path):
    """SQLite connection with a minimal nsys-like schema (kernels + StringIds)."""
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            shortName INTEGER,
            demangledName INTEGER,
            start INTEGER,
            "end" INTEGER,
            deviceId INTEGER DEFAULT 0
        );

        INSERT INTO StringIds VALUES (1, 'flash_bwd_dq_dk_dv_loop_seqk_parallel');
        INSERT INTO StringIds VALUES (2, 'flash_fwd_splitkv_kernel');
        INSERT INTO StringIds VALUES (3, 'ncclDevKernel_SendRecv');

        -- flash_bwd: 10 calls × 60ms each = 600ms
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1, 1, 0,          60000000, 0);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1, 1, 70000000,  130000000, 0);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1, 1, 140000000, 200000000, 0);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1, 1, 210000000, 270000000, 0);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1, 1, 280000000, 340000000, 0);

        -- flash_fwd: 5 calls × 20ms each = 100ms
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (2, 2, 0,         20000000, 0);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (2, 2, 25000000,  45000000, 0);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (2, 2, 50000000,  70000000, 0);

        -- nccl: 2 calls × 5ms each = 10ms
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (3, 3, 0,  5000000, 0);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (3, 3, 10000000, 15000000, 0);
    """)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# build_plan
# ---------------------------------------------------------------------------


class TestBuildPlan:
    def test_returns_plan(self, minimal_conn):
        from nsys_ai.cutracer.planner import build_plan

        plan = build_plan(minimal_conn, profile_path="/fake/profile.sqlite")
        assert plan is not None
        assert plan.profile_path == "/fake/profile.sqlite"

    def test_top_n_respected(self, minimal_conn):
        from nsys_ai.cutracer.planner import build_plan

        plan = build_plan(minimal_conn, profile_path="/fake/p.sqlite", top_n=2)
        assert len(plan.targets) <= 2

    def test_targets_sorted_by_gpu_time(self, minimal_conn):
        from nsys_ai.cutracer.planner import build_plan

        plan = build_plan(minimal_conn, profile_path="/fake/p.sqlite", top_n=5)
        if len(plan.targets) >= 2:
            assert plan.targets[0].total_ms >= plan.targets[1].total_ms

    def test_flash_bwd_is_top(self, minimal_conn):
        from nsys_ai.cutracer.planner import build_plan

        plan = build_plan(minimal_conn, profile_path="/fake/p.sqlite", top_n=5)
        assert len(plan.targets) > 0
        assert "flash_bwd" in plan.targets[0].name.lower()

    def test_pct_sums_to_100(self, minimal_conn):
        from nsys_ai.cutracer.planner import build_plan

        plan = build_plan(minimal_conn, profile_path="/fake/p.sqlite", top_n=10)
        # pct is computed relative to the full candidate pool — targets + skipped = 100%
        all_kernels = plan.targets + plan.skipped
        total = sum(t.pct_of_gpu for t in all_kernels)
        assert abs(total - 100.0) < 1.0

    def test_invocations_positive(self, minimal_conn):
        from nsys_ai.cutracer.planner import build_plan

        plan = build_plan(minimal_conn, profile_path="/fake/p.sqlite")
        for t in plan.targets:
            assert t.invocations > 0

    def test_empty_profile_returns_empty_plan(self, tmp_path):
        from nsys_ai.cutracer.planner import build_plan

        conn = sqlite3.connect(str(tmp_path / "empty.sqlite"))
        conn.row_factory = sqlite3.Row
        plan = build_plan(conn, profile_path="/fake/empty.sqlite")
        assert plan.targets == []
        conn.close()


# ---------------------------------------------------------------------------
# format_plan_script
# ---------------------------------------------------------------------------


class TestFormatPlanScript:
    def _make_plan(self):
        from nsys_ai.cutracer.planner import CutracerPlan, KernelTarget

        return CutracerPlan(
            profile_path="/data/trace.sqlite",
            targets=[
                KernelTarget(
                    name="flash_bwd_dq_dk_dv_loop_seqk_parallel",
                    total_ms=600.0,
                    pct_of_gpu=84.5,
                    invocations=10,
                ),
                KernelTarget(
                    name="flash_fwd_splitkv_kernel",
                    total_ms=100.0,
                    pct_of_gpu=14.1,
                    invocations=5,
                ),
            ],
        )

    def test_returns_string(self):
        from nsys_ai.cutracer.planner import format_plan_script

        script = format_plan_script(self._make_plan())
        assert isinstance(script, str)
        assert len(script) > 0

    def test_shebang_present(self):
        from nsys_ai.cutracer.planner import format_plan_script

        script = format_plan_script(self._make_plan())
        assert script.startswith("#!/usr/bin/env bash")

    def test_contains_profile_path(self):
        from nsys_ai.cutracer.planner import format_plan_script

        script = format_plan_script(self._make_plan())
        assert "/data/trace.sqlite" in script

    def test_contains_kernel_names(self):
        from nsys_ai.cutracer.planner import format_plan_script

        script = format_plan_script(self._make_plan())
        # Normalised names should appear in the filter section
        assert "flash_bwd" in script
        assert "flash_fwd" in script

    def test_contains_output_dir(self):
        from nsys_ai.cutracer.planner import format_plan_script

        script = format_plan_script(self._make_plan(), output_dir="./my_traces")
        assert "./my_traces" in script

    def test_custom_launch_cmd(self):
        from nsys_ai.cutracer.planner import format_plan_script

        script = format_plan_script(self._make_plan(), launch_cmd="torchrun --nproc 8 train.py")
        assert "torchrun" in script

    def test_default_launch_cmd_has_todo(self):
        from nsys_ai.cutracer.planner import format_plan_script

        script = format_plan_script(self._make_plan(), launch_cmd="")
        assert "TODO" in script

    def test_uses_cutracer_trace_cli(self):
        """Generated bash script should use 'cutracer trace' CLI, not LD_PRELOAD."""
        from nsys_ai.cutracer.planner import format_plan_script

        script = format_plan_script(self._make_plan())
        assert "cutracer trace" in script
        assert "LD_PRELOAD" not in script

    def test_contains_cutracer_mode(self):
        from nsys_ai.cutracer.planner import format_plan_script

        script = format_plan_script(self._make_plan(), mode="proton_instr_histogram")
        assert "proton_instr_histogram" in script

    def test_empty_targets_still_produces_valid_script(self):
        from nsys_ai.cutracer.planner import CutracerPlan, format_plan_script

        empty_plan = CutracerPlan(profile_path="/fake/p.sqlite", targets=[])
        script = format_plan_script(empty_plan)
        assert "#!/usr/bin/env bash" in script
        assert "KERNEL_FILTER_CSV" in script

    def test_analyze_command_in_footer(self):
        from nsys_ai.cutracer.planner import format_plan_script

        script = format_plan_script(self._make_plan(), output_dir="./out")
        assert "nsys-ai cutracer analyze" in script


# ---------------------------------------------------------------------------
# format_plan_summary
# ---------------------------------------------------------------------------


class TestFormatPlanSummary:
    def test_summary_shows_kernels(self):
        from nsys_ai.cutracer.planner import CutracerPlan, KernelTarget, format_plan_summary

        plan = CutracerPlan(
            profile_path="/fake/p.sqlite",
            targets=[
                KernelTarget("flash_bwd", 600.0, 85.0, 10),
                KernelTarget("flash_fwd", 100.0, 14.0, 5),
            ],
        )
        summary = format_plan_summary(plan)
        assert "flash_bwd" in summary
        assert "flash_fwd" in summary
        assert "600" in summary

    def test_empty_plan_summary(self):
        from nsys_ai.cutracer.planner import CutracerPlan, format_plan_summary

        summary = format_plan_summary(CutracerPlan(profile_path="/fake/p.sqlite"))
        assert "No kernels" in summary

    def test_summary_shows_next_steps(self):
        from nsys_ai.cutracer.planner import CutracerPlan, KernelTarget, format_plan_summary

        plan = CutracerPlan(
            profile_path="/fake/p.sqlite",
            targets=[KernelTarget("kernel_a", 100.0, 100.0, 1)],
        )
        summary = format_plan_summary(plan)
        assert "nsys-ai cutracer install" in summary
        assert "nsys-ai cutracer analyze" in summary


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestCutracerPlanCLI:
    def test_plan_summary_from_real_fixture(self, minimal_nsys_conn):
        from nsys_ai.cutracer.planner import build_plan, format_plan_summary

        plan = build_plan(minimal_nsys_conn, profile_path="/fake/p.sqlite", top_n=3)
        summary = format_plan_summary(plan)
        # Should produce some output without raising
        assert isinstance(summary, str)

    def test_plan_script_from_real_fixture(self, minimal_nsys_conn):
        from nsys_ai.cutracer.planner import build_plan, format_plan_script

        plan = build_plan(minimal_nsys_conn, profile_path="/fake/p.sqlite", top_n=3)
        script = format_plan_script(plan, output_dir="./out", launch_cmd="python train.py")
        assert "#!/usr/bin/env bash" in script
        assert "python train.py" in script
