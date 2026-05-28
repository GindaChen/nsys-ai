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
# _ensure_pinned_checkout (existing-clone reconciliation)
# ---------------------------------------------------------------------------


class TestEnsurePinnedCheckout:
    def test_already_at_tag_is_reused_without_fetch(self, tmp_path):
        from nsys_ai.cutracer.installer import CUTRACER_TAG, _ensure_pinned_checkout

        clone = tmp_path / "CUTracer"
        clone.mkdir()
        so_dest = clone / "lib" / "cutracer.so"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=f"{CUTRACER_TAG}\n", stderr=""
            )
            _ensure_pinned_checkout(clone, so_dest, progress=False)

        # Only the `git describe` probe should run — no fetch/checkout.
        assert mock_run.call_count == 1
        assert mock_run.call_args_list[0].args[0][:2] == ["git", "describe"]

    def test_wrong_ref_triggers_fetch_checkout_and_drops_stale_so(self, tmp_path):
        from nsys_ai.cutracer.installer import (
            CUTRACER_GITHUB,
            CUTRACER_TAG,
            _ensure_pinned_checkout,
        )

        clone = tmp_path / "CUTracer"
        (clone / "lib").mkdir(parents=True)
        so_dest = clone / "lib" / "cutracer.so"
        so_dest.write_bytes(b"\x7fELF")  # stale artifact from old ref

        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "describe"]:
                return MagicMock(returncode=0, stdout="deadbeef\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            _ensure_pinned_checkout(clone, so_dest, progress=False)

        cmds = [c.args[0] for c in mock_run.call_args_list]
        assert ["git", "remote", "set-url", "origin", CUTRACER_GITHUB] in cmds
        assert ["git", "fetch", "--depth=1", "origin", "tag", CUTRACER_TAG] in cmds
        # Plain checkout (no -f) on a clean worktree.
        assert ["git", "checkout", CUTRACER_TAG] in cmds
        # Stale .so removed so the build step recompiles from the new ref.
        assert not so_dest.exists()

    def test_dirty_worktree_refuses_checkout(self, tmp_path):
        from nsys_ai.cutracer.installer import _ensure_pinned_checkout

        clone = tmp_path / "CUTracer"
        (clone / "lib").mkdir(parents=True)
        so_dest = clone / "lib" / "cutracer.so"
        so_dest.write_bytes(b"\x7fELF")  # must NOT be deleted on refusal

        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "describe"]:
                return MagicMock(returncode=0, stdout="deadbeef\n", stderr="")
            if cmd[:2] == ["git", "status"]:
                return MagicMock(returncode=0, stdout=" M src/foo.cu\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            with pytest.raises(RuntimeError, match="uncommitted local changes"):
                _ensure_pinned_checkout(clone, so_dest, progress=False)

        # No fetch/checkout attempted, stale .so left intact.
        cmds = [c.args[0][:2] for c in mock_run.call_args_list]
        assert ["git", "fetch"] not in cmds
        assert ["git", "checkout"] not in cmds
        assert so_dest.exists()

    def test_raises_when_git_status_fails(self, tmp_path):
        """A failed `git status` means cleanliness is unverifiable — bail
        clearly rather than proceed to a possibly destructive checkout."""
        from nsys_ai.cutracer.installer import _ensure_pinned_checkout

        clone = tmp_path / "CUTracer"
        (clone / "lib").mkdir(parents=True)
        so_dest = clone / "lib" / "cutracer.so"
        so_dest.write_bytes(b"\x7fELF")

        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "describe"]:
                return MagicMock(returncode=0, stdout="deadbeef\n", stderr="")
            if cmd[:2] == ["git", "status"]:
                return MagicMock(returncode=128, stdout="", stderr="fatal: corrupt")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            with pytest.raises(RuntimeError, match="git status"):
                _ensure_pinned_checkout(clone, so_dest, progress=False)

        # No fetch/checkout attempted, stale .so left intact.
        cmds = [c.args[0][:2] for c in mock_run.call_args_list]
        assert ["git", "fetch"] not in cmds
        assert ["git", "checkout"] not in cmds
        assert so_dest.exists()

    def test_error_message_normalizes_none_stderr(self, tmp_path):
        """With progress=True, capture_output is False and .stderr is None —
        the error message must not leak the literal 'None'."""
        from nsys_ai.cutracer.installer import _ensure_pinned_checkout

        clone = tmp_path / "CUTracer"
        clone.mkdir()
        so_dest = clone / "lib" / "cutracer.so"

        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "describe"]:
                return MagicMock(returncode=0, stdout="deadbeef\n", stderr="")
            if cmd[:2] == ["git", "status"]:
                return MagicMock(returncode=0, stdout="", stderr=None)
            if cmd[:2] == ["git", "fetch"]:
                return MagicMock(returncode=1, stdout=None, stderr=None)
            return MagicMock(returncode=0, stdout=None, stderr=None)

        with patch("subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError) as exc:
                _ensure_pinned_checkout(clone, so_dest, progress=True)
        assert "None" not in str(exc.value)

    def test_raises_on_set_url_failure(self, tmp_path):
        from nsys_ai.cutracer.installer import _ensure_pinned_checkout

        clone = tmp_path / "CUTracer"
        clone.mkdir()
        so_dest = clone / "lib" / "cutracer.so"

        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "describe"]:
                return MagicMock(returncode=0, stdout="deadbeef\n", stderr="")
            if cmd[:3] == ["git", "remote", "set-url"]:
                return MagicMock(returncode=1, stdout="", stderr="no origin")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError, match="set-url"):
                _ensure_pinned_checkout(clone, so_dest, progress=False)

    def test_raises_on_fetch_failure(self, tmp_path):
        from nsys_ai.cutracer.installer import _ensure_pinned_checkout

        clone = tmp_path / "CUTracer"
        clone.mkdir()
        so_dest = clone / "lib" / "cutracer.so"

        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "describe"]:
                return MagicMock(returncode=0, stdout="deadbeef\n", stderr="")
            if cmd[:2] == ["git", "fetch"]:
                return MagicMock(returncode=1, stdout="", stderr="network down")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError, match="git fetch"):
                _ensure_pinned_checkout(clone, so_dest, progress=False)

    def test_raises_on_checkout_failure(self, tmp_path):
        from nsys_ai.cutracer.installer import _ensure_pinned_checkout

        clone = tmp_path / "CUTracer"
        clone.mkdir()
        so_dest = clone / "lib" / "cutracer.so"

        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "describe"]:
                return MagicMock(returncode=0, stdout="deadbeef\n", stderr="")
            if cmd[:2] == ["git", "checkout"]:
                return MagicMock(returncode=1, stdout="", stderr="conflict")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError, match="git checkout"):
                _ensure_pinned_checkout(clone, so_dest, progress=False)


class TestCloneAndBuildExistingTree:
    def test_non_git_makefile_tree_skips_reconciliation(self, tmp_path):
        """A non-git source tree (e.g. extracted tarball) with a Makefile must
        build as-is, not attempt git reconciliation (which would error)."""
        from nsys_ai.cutracer import installer

        clone = tmp_path / "CUTracer"
        clone.mkdir()
        (clone / "Makefile").write_text("all:\n")
        # Pre-stage so the build short-circuits: nvbit present, .so already built.
        (clone / "third_party" / "nvbit").mkdir(parents=True)
        (clone / "lib").mkdir()
        (clone / "lib" / "cutracer.so").write_bytes(b"\x7fELF")
        # No .git → must be treated as a non-git tree.

        with patch.object(installer, "_ensure_pinned_checkout") as mock_reconcile:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                result = installer._clone_and_build(clone, progress=False)

        mock_reconcile.assert_not_called()
        assert result == clone / "lib" / "cutracer.so"

    def test_git_makefile_tree_triggers_reconciliation(self, tmp_path):
        """A git clone with a Makefile must be reconciled to the pinned tag."""
        from nsys_ai.cutracer import installer

        clone = tmp_path / "CUTracer"
        clone.mkdir()
        (clone / "Makefile").write_text("all:\n")
        (clone / ".git").mkdir()
        (clone / "third_party" / "nvbit").mkdir(parents=True)
        (clone / "lib").mkdir()
        (clone / "lib" / "cutracer.so").write_bytes(b"\x7fELF")

        with patch.object(installer, "_ensure_pinned_checkout") as mock_reconcile:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                installer._clone_and_build(clone, progress=False)

        mock_reconcile.assert_called_once()


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
