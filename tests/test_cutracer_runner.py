"""Tests for nsys_ai.cutracer.runner — local runner and Modal script generation."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------


class TestRunConfig:
    def test_defaults(self, tmp_path):
        from nsys_ai.cutracer.runner import RunConfig

        cfg = RunConfig(launch_cmd="python train.py", output_dir=tmp_path)
        assert cfg.mode == "proton_instr_histogram"
        assert cfg.kernel_filter == []
        assert cfg.so_path is None
        assert cfg.max_iters is None

    def test_kernel_filter_stored(self, tmp_path):
        from nsys_ai.cutracer.runner import RunConfig

        cfg = RunConfig(
            launch_cmd="python train.py",
            output_dir=tmp_path,
            kernel_filter=["flash_bwd", "flash_fwd"],
        )
        assert cfg.kernel_filter == ["flash_bwd", "flash_fwd"]


# ---------------------------------------------------------------------------
# _build_cutracer_cmd
# ---------------------------------------------------------------------------


class TestBuildCutracerCmd:
    def test_uses_cutracer_trace(self, tmp_path):
        from nsys_ai.cutracer.runner import RunConfig, _build_cutracer_cmd

        cfg = RunConfig(launch_cmd="python train.py", output_dir=tmp_path)
        so = tmp_path / "cutracer.so"
        so.write_bytes(b"")
        argv = _build_cutracer_cmd(cfg, so)
        assert argv[0] == "cutracer"
        assert argv[1] == "trace"

    def test_includes_so_path(self, tmp_path):
        from nsys_ai.cutracer.runner import RunConfig, _build_cutracer_cmd

        cfg = RunConfig(launch_cmd="cmd", output_dir=tmp_path)
        so = tmp_path / "cutracer.so"
        so.write_bytes(b"")
        argv = _build_cutracer_cmd(cfg, so)
        assert "--cutracer-so" in argv
        assert str(so) in argv

    def test_no_so_omits_cutracer_so_flag(self, tmp_path):
        from nsys_ai.cutracer.runner import RunConfig, _build_cutracer_cmd

        cfg = RunConfig(launch_cmd="cmd", output_dir=tmp_path)
        argv = _build_cutracer_cmd(cfg, None)
        assert "--cutracer-so" not in argv

    def test_analysis_flag(self, tmp_path):
        from nsys_ai.cutracer.runner import RunConfig, _build_cutracer_cmd

        cfg = RunConfig(launch_cmd="cmd", output_dir=tmp_path, mode="proton_instr_histogram")
        argv = _build_cutracer_cmd(cfg, None)
        assert "--analysis" in argv
        assert "proton_instr_histogram" in argv

    def test_kernel_filters_flag(self, tmp_path):
        from nsys_ai.cutracer.runner import RunConfig, _build_cutracer_cmd

        cfg = RunConfig(
            launch_cmd="cmd", output_dir=tmp_path,
            kernel_filter=["flash_bwd", "flash_fwd"]
        )
        argv = _build_cutracer_cmd(cfg, None)
        assert "--kernel-filters" in argv
        idx = argv.index("--kernel-filters")
        assert argv[idx + 1] == "flash_bwd,flash_fwd"

    def test_no_kernel_filter_omits_flag(self, tmp_path):
        from nsys_ai.cutracer.runner import RunConfig, _build_cutracer_cmd

        cfg = RunConfig(launch_cmd="cmd", output_dir=tmp_path, kernel_filter=[])
        argv = _build_cutracer_cmd(cfg, None)
        assert "--kernel-filters" not in argv

    def test_output_dir_in_argv(self, tmp_path):
        from nsys_ai.cutracer.runner import RunConfig, _build_cutracer_cmd

        cfg = RunConfig(launch_cmd="cmd", output_dir=tmp_path)
        argv = _build_cutracer_cmd(cfg, None)
        assert "--output-dir" in argv

    def test_launch_cmd_after_double_dash(self, tmp_path):
        from nsys_ai.cutracer.runner import RunConfig, _build_cutracer_cmd

        cfg = RunConfig(launch_cmd="python train.py", output_dir=tmp_path)
        argv = _build_cutracer_cmd(cfg, None)
        assert "--" in argv
        # launch_cmd is split into separate tokens so cutracer can exec them correctly
        dash_idx = argv.index("--")
        assert argv[dash_idx + 1] == "python"
        assert argv[dash_idx + 2] == "train.py"


# ---------------------------------------------------------------------------
# run_local
# ---------------------------------------------------------------------------


class TestRunLocal:
    def test_dry_run_returns_output_dir(self, tmp_path):
        from nsys_ai.cutracer.runner import RunConfig, run_local

        so = tmp_path / "cutracer.so"
        so.write_bytes(b"\x7fELF")
        cfg = RunConfig(
            launch_cmd="python train.py",
            output_dir=tmp_path / "out",
            so_path=so,
        )
        result = run_local(cfg, dry_run=True, progress=False)
        assert result == cfg.output_dir

    def test_creates_output_dir(self, tmp_path):
        from nsys_ai.cutracer.runner import RunConfig, run_local

        out = tmp_path / "deep" / "out"
        so = tmp_path / "cutracer.so"
        so.write_bytes(b"\x7fELF")
        cfg = RunConfig(launch_cmd="true", output_dir=out, so_path=so)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            run_local(cfg, progress=False)
        assert out.exists()

    def test_raises_on_nonzero_exit(self, tmp_path):
        from nsys_ai.cutracer.runner import RunConfig, run_local

        so = tmp_path / "cutracer.so"
        so.write_bytes(b"\x7fELF")
        cfg = RunConfig(launch_cmd="false", output_dir=tmp_path / "out", so_path=so)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            with pytest.raises(subprocess.CalledProcessError):
                run_local(cfg, progress=False)

    def test_no_so_runs_in_logger_mode(self, tmp_path):
        """Without a .so, run_local falls back to kernel logger mode (no --analysis)."""
        from nsys_ai.cutracer.runner import RunConfig, run_local

        cfg = RunConfig(
            launch_cmd="python train.py",
            output_dir=tmp_path / "out",
            so_path=tmp_path / "nonexistent.so",
        )
        with patch("nsys_ai.cutracer.installer._find_cutracer_so_path", return_value=None):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                run_local(cfg, progress=False)
                call_argv = mock_run.call_args[0][0]
                # Logger mode: no --analysis flag
                assert "--analysis" not in call_argv

    def test_max_iters_sets_cutracer_env(self, tmp_path):
        """CUTRACER_MAX_ITERS is passed when ``RunConfig.max_iters`` is set."""
        from nsys_ai.cutracer.runner import RunConfig, run_local

        so = tmp_path / "cutracer.so"
        so.write_bytes(b"\x7fELF")
        cfg = RunConfig(
            launch_cmd="true",
            output_dir=tmp_path / "out",
            so_path=so,
            max_iters=42,
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            run_local(cfg, progress=False)
        env = mock_run.call_args.kwargs.get("env") or {}
        assert env.get("CUTRACER_MAX_ITERS") == "42"


# ---------------------------------------------------------------------------
# format_modal_app
# ---------------------------------------------------------------------------


def _make_plan_and_config(tmp_path):
    from nsys_ai.cutracer.planner import CutracerPlan, KernelTarget
    from nsys_ai.cutracer.runner import RunConfig

    plan = CutracerPlan(
        profile_path="/data/trace.sqlite",
        targets=[
            KernelTarget("flash_bwd_dq_dk_dv_loop", 600.0, 85.0, 10),
            KernelTarget("flash_fwd_kernel", 100.0, 14.0, 5),
        ],
    )
    config = RunConfig(
        launch_cmd="python train.py",
        output_dir=tmp_path / "cutracer_out",
        kernel_filter=["flash_bwd_dq_dk_dv_loop", "flash_fwd_kernel"],
        max_iters=5,
    )
    return plan, config


class TestFormatModalApp:
    def test_returns_string(self, tmp_path):
        from nsys_ai.cutracer.runner import format_modal_app

        plan, config = _make_plan_and_config(tmp_path)
        script = format_modal_app(plan, config, profile_path="/data/trace.sqlite")
        assert isinstance(script, str)
        assert len(script) > 100

    def test_shebang(self, tmp_path):
        from nsys_ai.cutracer.runner import format_modal_app

        plan, config = _make_plan_and_config(tmp_path)
        script = format_modal_app(plan, config)
        assert script.startswith("#!/usr/bin/env python3")

    def test_contains_import_modal(self, tmp_path):
        from nsys_ai.cutracer.runner import format_modal_app

        plan, config = _make_plan_and_config(tmp_path)
        script = format_modal_app(plan, config)
        assert "import modal" in script

    def test_contains_kernel_filter(self, tmp_path):
        from nsys_ai.cutracer.runner import format_modal_app

        plan, config = _make_plan_and_config(tmp_path)
        script = format_modal_app(plan, config)
        assert "flash_bwd" in script
        assert "flash_fwd" in script

    def test_contains_launch_cmd(self, tmp_path):
        from nsys_ai.cutracer.runner import format_modal_app

        plan, config = _make_plan_and_config(tmp_path)
        script = format_modal_app(plan, config)
        assert "python train.py" in script

    def test_contains_cuda_injection_path(self, tmp_path):
        from nsys_ai.cutracer.runner import format_modal_app

        plan, config = _make_plan_and_config(tmp_path)
        script = format_modal_app(plan, config)
        # cutracer uses --cutracer-so flag (sets CUDA_INJECTION64_PATH internally)
        assert "--cutracer-so" in script

    def test_contains_volume_name(self, tmp_path):
        from nsys_ai.cutracer.runner import ModalConfig, format_modal_app

        plan, config = _make_plan_and_config(tmp_path)
        modal_cfg = ModalConfig(volume_name="my-special-volume")
        script = format_modal_app(plan, config, modal_cfg)
        assert "my-special-volume" in script

    def test_contains_gpu_type(self, tmp_path):
        from nsys_ai.cutracer.runner import ModalConfig, format_modal_app

        plan, config = _make_plan_and_config(tmp_path)
        modal_cfg = ModalConfig(gpu="A100")
        script = format_modal_app(plan, config, modal_cfg)
        assert "A100" in script

    def test_contains_github_clone(self, tmp_path):
        """Generated Modal script should clone from GitHub, not use nsys-ai cutracer install."""
        from nsys_ai.cutracer.runner import format_modal_app

        plan, config = _make_plan_and_config(tmp_path)
        script = format_modal_app(plan, config)
        assert "git clone" in script
        assert "facebookresearch/CUTracer" in script
        assert "install_third_party.sh" in script
        assert "nsys-ai cutracer install" not in script

    def test_contains_analyze_command(self, tmp_path):
        from nsys_ai.cutracer.runner import format_modal_app

        plan, config = _make_plan_and_config(tmp_path)
        script = format_modal_app(plan, config, profile_path="/data/trace.sqlite")
        assert "nsys-ai cutracer analyze" in script
        assert "/data/trace.sqlite" in script

    def test_contains_max_iters(self, tmp_path):
        from nsys_ai.cutracer.runner import format_modal_app

        plan, config = _make_plan_and_config(tmp_path)
        script = format_modal_app(plan, config)
        assert "CUTRACER_MAX_ITERS" in script
        assert "5" in script  # max_iters=5

    def test_contains_vol_iterdir(self, tmp_path):
        from nsys_ai.cutracer.runner import format_modal_app

        plan, config = _make_plan_and_config(tmp_path)
        script = format_modal_app(plan, config)
        assert "vol.iterdir" in script

    def test_extra_pip_packages(self, tmp_path):
        from nsys_ai.cutracer.runner import ModalConfig, format_modal_app

        plan, config = _make_plan_and_config(tmp_path)
        modal_cfg = ModalConfig(extra_pip=["torch==2.3.0", "transformers"])
        script = format_modal_app(plan, config, modal_cfg)
        assert "torch==2.3.0" in script
        assert "transformers" in script

    def test_empty_plan_targets_still_valid(self, tmp_path):
        from nsys_ai.cutracer.planner import CutracerPlan
        from nsys_ai.cutracer.runner import RunConfig, format_modal_app

        empty_plan = CutracerPlan(profile_path="/data/trace.sqlite", targets=[])
        config = RunConfig(launch_cmd="python train.py", output_dir=tmp_path / "out")
        script = format_modal_app(empty_plan, config)
        assert "import modal" in script

    def test_valid_python_syntax(self, tmp_path):
        """Generated Modal script must be syntactically valid Python."""
        import ast

        from nsys_ai.cutracer.runner import format_modal_app

        plan, config = _make_plan_and_config(tmp_path)
        script = format_modal_app(plan, config, profile_path="/data/trace.sqlite")
        # Should not raise SyntaxError
        ast.parse(script)


# ---------------------------------------------------------------------------
# _cuda_image_for_version
# ---------------------------------------------------------------------------


class TestCudaImageForVersion:
    def test_known_version(self):
        from nsys_ai.cutracer.runner import _cuda_image_for_version

        img = _cuda_image_for_version((12, 4))
        assert "12.4" in img
        assert "cuda" in img

    def test_unknown_version_returns_default(self):
        from nsys_ai.cutracer.runner import _CUDA_IMAGE_DEFAULT, _cuda_image_for_version

        img = _cuda_image_for_version((99, 0))
        assert img == _CUDA_IMAGE_DEFAULT

    def test_none_returns_default(self):
        from nsys_ai.cutracer.runner import _CUDA_IMAGE_DEFAULT, _cuda_image_for_version

        img = _cuda_image_for_version(None)
        assert img == _CUDA_IMAGE_DEFAULT
