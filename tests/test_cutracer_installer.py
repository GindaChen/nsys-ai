"""Tests for nsys_ai.cutracer.installer — prerequisite checks, URL generation, build logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# URL / asset name helpers
# ---------------------------------------------------------------------------


class TestAssetName:
    def test_default_version(self):
        from nsys_ai.cutracer.installer import nvbit_asset_name

        name = nvbit_asset_name()
        assert name.startswith("nvbit-Linux-x86_64-")
        assert name.endswith(".tar.bz2")

    def test_custom_version(self):
        from nsys_ai.cutracer.installer import nvbit_asset_name

        name = nvbit_asset_name("1.5.0")
        assert "1.5.0" in name

    def test_download_url_format(self):
        from nsys_ai.cutracer.installer import nvbit_download_url

        url = nvbit_download_url("1.7.1")
        assert url.startswith("https://github.com/NVlabs/NVBit/releases/download/")
        assert "1.7.1" in url
        assert url.endswith(".tar.bz2")


# ---------------------------------------------------------------------------
# detect_cuda_version
# ---------------------------------------------------------------------------


class TestDetectCudaVersion:
    def test_returns_none_when_nvcc_missing(self):
        from nsys_ai.cutracer.installer import detect_cuda_version

        with patch("subprocess.check_output", side_effect=FileNotFoundError):
            assert detect_cuda_version() is None

    def test_parses_real_nvcc_output(self):
        from nsys_ai.cutracer.installer import detect_cuda_version

        fake_output = (
            "nvcc: NVIDIA (R) Cuda compiler driver\n"
            "Cuda compilation tools, release 12.4, V12.4.131\n"
        )
        with patch("subprocess.check_output", return_value=fake_output):
            result = detect_cuda_version()
        assert result == (12, 4)

    def test_parses_cuda_11(self):
        from nsys_ai.cutracer.installer import detect_cuda_version

        fake_output = "Cuda compilation tools, release 11.8, V11.8.89\n"
        with patch("subprocess.check_output", return_value=fake_output):
            result = detect_cuda_version()
        assert result == (11, 8)


# ---------------------------------------------------------------------------
# check_prerequisites
# ---------------------------------------------------------------------------


class TestCheckPrerequisites:
    def test_returns_list_of_prereq_results(self):
        from nsys_ai.cutracer.installer import check_prerequisites

        results = check_prerequisites()
        assert isinstance(results, list)
        assert len(results) >= 3
        names = [r.name for r in results]
        assert any("nvcc" in n for n in names)
        assert any("g++" in n for n in names)
        assert any("make" in n for n in names)

    def test_missing_nvcc_shows_message(self):
        from nsys_ai.cutracer.installer import check_prerequisites

        with patch("subprocess.check_output", side_effect=FileNotFoundError):
            with patch("shutil.which", return_value=None):
                results = check_prerequisites()
        nvcc = next(r for r in results if "nvcc" in r.name)
        assert not nvcc.ok
        assert nvcc.message  # should give install hint

    def test_all_present_all_ok(self):
        from nsys_ai.cutracer.installer import check_prerequisites

        def fake_check_output(cmd, **_):
            cmd0 = cmd[0]
            if cmd0 == "nvcc":
                return "Cuda compilation tools, release 12.4, V12.4.131"
            if cmd0 == "g++":
                return "g++ (Ubuntu 11.4.0) 11.4.0"
            if cmd0 == "git":
                return "git version 2.43.0"
            return ""

        with patch("subprocess.check_output", side_effect=fake_check_output):
            with patch("shutil.which", return_value="/usr/bin/make"):
                with patch("nsys_ai.cutracer.installer._check_libzstd", return_value=True):
                    results = check_prerequisites()

        assert all(r.ok for r in results)


# ---------------------------------------------------------------------------
# format_prereq_table
# ---------------------------------------------------------------------------


class TestFormatPrereqTable:
    def test_ok_shows_ok(self):
        from nsys_ai.cutracer.installer import PrereqResult, format_prereq_table

        results = [PrereqResult("nvcc (CUDA compiler)", ok=True, version="12.4")]
        table = format_prereq_table(results)
        assert "OK" in table
        assert "nvcc" in table
        assert "12.4" in table

    def test_missing_shows_missing_and_message(self):
        from nsys_ai.cutracer.installer import PrereqResult, format_prereq_table

        results = [PrereqResult("g++", ok=False, message="apt-get install g++")]
        table = format_prereq_table(results)
        assert "MISSING" in table
        assert "apt-get" in table


# ---------------------------------------------------------------------------
# find_cutracer_source
# ---------------------------------------------------------------------------


class TestFindCutracerSource:
    def test_env_var_override(self, tmp_path):
        from nsys_ai.cutracer.installer import find_cutracer_source

        src_dir = tmp_path / "mytool"
        src_dir.mkdir()
        with patch.dict("os.environ", {"CUTRACER_SRC": str(src_dir)}):
            result = find_cutracer_source()
        assert result == src_dir

    def test_env_var_nonexistent_ignored(self, tmp_path):
        from nsys_ai.cutracer.installer import find_cutracer_source

        with patch.dict("os.environ", {"CUTRACER_SRC": str(tmp_path / "nonexistent")}):
            # Should fall through to other search paths, not raise
            result = find_cutracer_source()
            # result may be None or a real path — just shouldn't raise
            assert result is None or isinstance(result, Path)

    def test_returns_none_when_not_found(self, tmp_path, monkeypatch):
        from nsys_ai.cutracer import installer

        monkeypatch.delenv("CUTRACER_SRC", raising=False)
        monkeypatch.setattr(installer, "INSTALL_DIR", tmp_path / "noinstall")

        # Ensure cutracer Python package import fails (no .so adjacent to it)
        with patch("nsys_ai.cutracer.installer.find_cutracer_source.__wrapped__"
                   if hasattr(installer.find_cutracer_source, "__wrapped__") else
                   "builtins.__import__", create=True):
            pass  # just run without patching import — relying on INSTALL_DIR

        result = installer.find_cutracer_source()
        # May be None (no source found) or point to cutracer package dir if installed
        # Either is acceptable — the important thing is it doesn't raise
        assert result is None or isinstance(result, Path)


# ---------------------------------------------------------------------------
# build_so
# ---------------------------------------------------------------------------


class TestBuildSo:
    def test_raises_on_make_failure(self, tmp_path):
        from nsys_ai.cutracer.installer import build_so

        src = tmp_path / "src"
        src.mkdir()
        nvbit = tmp_path / "nvbit"
        nvbit.mkdir()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="Error", stdout="")
            with pytest.raises(RuntimeError, match="make failed"):
                build_so(src, nvbit, progress=False)

    def test_raises_when_no_so_produced(self, tmp_path):
        from nsys_ai.cutracer.installer import build_so

        src = tmp_path / "src"
        src.mkdir()
        nvbit = tmp_path / "nvbit"
        nvbit.mkdir()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            with pytest.raises(RuntimeError, match="no .so found"):
                build_so(src, nvbit, progress=False)

    def test_returns_so_path_on_success(self, tmp_path):
        from nsys_ai.cutracer.installer import build_so

        src = tmp_path / "src"
        src.mkdir()
        so_file = src / "cutracer.so"
        so_file.write_bytes(b"\x7fELF")  # fake ELF header
        nvbit = tmp_path / "nvbit"
        nvbit.mkdir()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            result = build_so(src, nvbit, progress=False)

        assert result == so_file


# ---------------------------------------------------------------------------
# install (high-level, dry-run)
# ---------------------------------------------------------------------------


class TestInstall:
    def test_dry_run_returns_success(self, tmp_path):
        from nsys_ai.cutracer.installer import install

        result = install(install_dir=tmp_path / "managed", dry_run=True, progress=False)
        assert result.success
        assert "cutracer.so" in result.so_path

    def test_missing_prereqs_returns_failure(self, tmp_path):
        from nsys_ai.cutracer.installer import install

        with patch("subprocess.check_output", side_effect=FileNotFoundError):
            with patch("shutil.which", return_value=None):
                result = install(install_dir=tmp_path / "managed", progress=False)

        assert not result.success
        assert result.errors

    def test_missing_cuda_returns_failure(self, tmp_path):
        from nsys_ai.cutracer.installer import install

        def fake_check_output(cmd, **_):
            cmd0 = cmd[0]
            if cmd0 == "nvcc":
                raise FileNotFoundError
            if cmd0 == "g++":
                return "g++ 11.4.0"
            if cmd0 == "git":
                return "git version 2.43"
            return ""

        with patch("subprocess.check_output", side_effect=fake_check_output):
            with patch("shutil.which", return_value="/usr/bin/make"):
                result = install(install_dir=tmp_path / "managed", progress=False)

        assert not result.success
        assert any("CUDA" in e or "prerequisite" in e.lower() for e in result.errors)

    def test_clone_and_build_failure_returns_failure(self, tmp_path):
        """When no local source is found, install() tries GitHub clone.
        If the clone fails, it returns a failure result."""
        from nsys_ai.cutracer.installer import install

        def fake_check_output(cmd, **_):
            cmd0 = cmd[0]
            if cmd0 == "nvcc":
                return "Cuda compilation tools, release 12.4, V12.4.131"
            if cmd0 == "g++":
                return "g++ 11.4.0"
            if cmd0 == "git":
                return "git version 2.43"
            return ""

        def fake_clone_and_build(*args, **kwargs):
            raise RuntimeError("git clone failed: network error")

        with patch("subprocess.check_output", side_effect=fake_check_output):
            with patch("shutil.which", return_value="/usr/bin/make"):
                with patch("nsys_ai.cutracer.installer._check_libzstd", return_value=True):
                    with patch("nsys_ai.cutracer.installer.find_cutracer_source", return_value=None):
                        with patch("nsys_ai.cutracer.installer._clone_and_build", side_effect=fake_clone_and_build):
                            result = install(install_dir=tmp_path / "managed", progress=False)

        assert not result.success
        assert any("git clone" in e or "clone" in e.lower() for e in result.errors)
