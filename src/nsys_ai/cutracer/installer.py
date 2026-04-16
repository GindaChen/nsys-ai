"""Build and install the CUTracer NVBit .so instrumentation library.

Flow
----
1. Check prerequisites (nvcc, g++, git, make).
2. Detect CUDA version from ``nvcc --version``.
3. Download the matching NVBit release tarball from GitHub.
4. Extract NVBit and locate / extract the CUTracer tool source.
5. Run ``make`` inside the tool directory.
6. Copy the resulting ``*.so`` to the managed install dir.

Managed install directory: ``~/.nsys-ai/cutracer/``
  lib/cutracer.so    ← LD_PRELOAD target
  nvbit/             ← extracted NVBit release
  src/               ← CUTracer tool source
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSTALL_DIR = Path.home() / ".nsys-ai" / "cutracer"
NVBIT_REPO = "https://github.com/NVlabs/NVBit/releases/download"
NVBIT_VERSION = "1.7.1"
CUTRACER_GITHUB = "https://github.com/facebookresearch/CUTracer"

# CUDA major.minor → NVBit release asset name pattern
# NVBit ships separate builds for each CUDA toolkit version.
_NVBIT_ASSET_TMPL = "nvbit-Linux-x86_64-{nvbit_ver}.tar.bz2"

# CUTracer tool directory name inside the NVBit release (if bundled) or
# the name used when building from the cutracer PyPI package source tree.
_CUTRACER_TOOL_DIRS = [
    "tools/proton_instr_histogram",
    "proton_instr_histogram",
    "cutracer",
]


# ---------------------------------------------------------------------------
# Prerequisite checking
# ---------------------------------------------------------------------------


@dataclass
class PrereqResult:
    name: str
    ok: bool
    version: str = ""
    message: str = ""


def check_prerequisites() -> list[PrereqResult]:
    """Return the status of each build prerequisite."""
    results: list[PrereqResult] = []

    # nvcc
    nvcc_ver = _run_version_cmd(["nvcc", "--version"], r"release (\S+),")
    results.append(PrereqResult(
        name="nvcc (CUDA compiler)",
        ok=nvcc_ver is not None,
        version=nvcc_ver or "",
        message="" if nvcc_ver else "Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads",
    ))

    # g++
    gxx_ver = _run_version_cmd(["g++", "--version"], r"g\+\+.*?(\d+\.\d+\.\d+)")
    results.append(PrereqResult(
        name="g++ (C++ compiler)",
        ok=gxx_ver is not None,
        version=gxx_ver or "",
        message="" if gxx_ver else "Install with: apt-get install g++",
    ))

    # make
    make_ok = shutil.which("make") is not None
    results.append(PrereqResult(
        name="make",
        ok=make_ok,
        message="" if make_ok else "Install with: apt-get install build-essential",
    ))

    # git (required for GitHub clone)
    git_ver = _run_version_cmd(["git", "--version"], r"git version (\S+)")
    results.append(PrereqResult(
        name="git",
        ok=git_ver is not None,
        version=git_ver or "",
        message="" if git_ver else "Install with: apt-get install git",
    ))

    # libzstd-dev (required by CUTracer build)
    libzstd_ok = _check_libzstd()
    results.append(PrereqResult(
        name="libzstd-dev (build dep)",
        ok=libzstd_ok,
        message="" if libzstd_ok else "Install with: apt-get install libzstd-dev",
    ))

    return results


def _check_libzstd() -> bool:
    """Return True if zstd.h is available (required by CUTracer's Makefile)."""
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write("#include <zstd.h>\nint main(){return 0;}\n")
            fname = f.name
        r = subprocess.run(
            ["gcc", "-include", "zstd.h", "-fsyntax-only", fname],
            capture_output=True,
        )
        Path(fname).unlink(missing_ok=True)
        return r.returncode == 0
    except Exception:
        # Fallback: check for the header file directly
        for p in ["/usr/include/zstd.h", "/usr/local/include/zstd.h"]:
            if Path(p).exists():
                return True
        return False


def _run_version_cmd(cmd: list[str], pattern: str) -> str | None:
    """Run *cmd* and extract the version string matching *pattern*, or None."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        m = re.search(pattern, out)
        return m.group(1) if m else "?"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


# ---------------------------------------------------------------------------
# CUDA version detection
# ---------------------------------------------------------------------------


def detect_cuda_version() -> tuple[int, int] | None:
    """Return (major, minor) CUDA version from nvcc, or None."""
    raw = _run_version_cmd(["nvcc", "--version"], r"release (\d+)\.(\d+)")
    if not raw:
        return None
    # pattern above captures two groups — redo with two-group match
    try:
        out = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT, text=True)
        m = re.search(r"release (\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# NVBit download helpers
# ---------------------------------------------------------------------------


def nvbit_asset_name(nvbit_version: str = NVBIT_VERSION) -> str:
    return _NVBIT_ASSET_TMPL.format(nvbit_ver=nvbit_version)


def nvbit_download_url(nvbit_version: str = NVBIT_VERSION) -> str:
    asset = nvbit_asset_name(nvbit_version)
    return f"{NVBIT_REPO}/{nvbit_version}/{asset}"


def download_nvbit(
    dest_dir: Path,
    nvbit_version: str = NVBIT_VERSION,
    *,
    progress: bool = True,
) -> Path:
    """Download and extract the NVBit release tarball into *dest_dir*.

    Returns the path to the extracted NVBit root directory.
    """
    url = nvbit_download_url(nvbit_version)
    asset = nvbit_asset_name(nvbit_version)
    tarball = dest_dir / asset

    dest_dir.mkdir(parents=True, exist_ok=True)

    if not tarball.exists():
        if progress:
            print(f"  Downloading NVBit {nvbit_version} …")
            print(f"    {url}")
        try:
            parsed = urlparse(url)
            if parsed.scheme != "https" or (parsed.hostname or "").lower() != "github.com":
                raise RuntimeError(f"Refusing download from non-GitHub HTTPS URL: {url!r}")
            urllib.request.urlretrieve(url, tarball)  # nosec B310
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download NVBit from {url}: {exc}\n"
                "Check your internet connection or download manually."
            ) from exc
    else:
        if progress:
            print(f"  Using cached: {tarball}")

    # Extract
    nvbit_root = dest_dir / f"nvbit-Linux-x86_64-{nvbit_version}"
    if not nvbit_root.exists():
        if progress:
            print("  Extracting NVBit …")
        with tarfile.open(tarball, "r:bz2") as tf:
            # NVBit tarball from NVlabs GitHub release (see nvbit_download_url).
            tf.extractall(dest_dir)  # nosec B202

    return nvbit_root


# ---------------------------------------------------------------------------
# CUTracer source location
# ---------------------------------------------------------------------------


def _find_cutracer_so_path() -> str | None:
    """Return path to a pre-built cutracer.so, or None.

    Search order:
    1. ``CUTRACER_SO`` environment variable.
    2. Managed install dir (``~/.nsys-ai/cutracer/lib/cutracer.so``).
    3. Alongside the installed ``cutracer`` Python package.
    """
    import shutil as _shutil

    env_path = os.environ.get("CUTRACER_SO")
    if env_path and Path(env_path).is_file():
        return env_path

    managed = INSTALL_DIR / "lib" / "cutracer.so"
    if managed.is_file():
        return str(managed)

    try:
        import cutracer as _ct  # type: ignore[import]
        candidate = Path(_ct.__file__).parent / "lib" / "cutracer.so"
        if candidate.is_file():
            return str(candidate)
    except ImportError:
        pass

    ct_bin = _shutil.which("cutracer")
    if ct_bin:
        candidate = Path(ct_bin).parent.parent / "lib" / "cutracer.so"
        if candidate.is_file():
            return str(candidate)

    return None


def find_cutracer_source() -> Path | None:
    """Locate CUTracer tool source directory.

    Search order:
    1. ``CUTRACER_SRC`` environment variable.
    2. Alongside the installed ``cutracer`` Python package (``../src/``).
    3. ``~/.nsys-ai/cutracer/src/`` (previously cloned).
    """
    # 1. Env override
    env_src = os.environ.get("CUTRACER_SRC")
    if env_src and Path(env_src).is_dir():
        return Path(env_src)

    # 2. Adjacent to installed Python package
    try:
        import cutracer as _ct  # type: ignore[import]
        pkg_dir = Path(_ct.__file__).parent
        for rel in _CUTRACER_TOOL_DIRS:
            candidate = pkg_dir.parent / rel
            if candidate.is_dir():
                return candidate
    except ImportError:
        pass

    # 3. Managed src dir
    managed = INSTALL_DIR / "src"
    for rel in _CUTRACER_TOOL_DIRS:
        candidate = managed / rel
        if candidate.is_dir():
            return candidate

    return None


# ---------------------------------------------------------------------------
# GitHub clone + build (primary install path)
# ---------------------------------------------------------------------------


def _clone_and_build(
    clone_dir: Path,
    *,
    progress: bool = True,
) -> Path:
    """Clone facebookresearch/CUTracer from GitHub and build cutracer.so.

    This is the primary install path — the repo's own ``install_third_party.sh``
    handles NVBit and nlohmann/json download, so no separate NVBit step needed.

    Returns the path to the built ``lib/cutracer.so``.
    Raises ``RuntimeError`` on any failure.
    """
    so_dest = clone_dir / "lib" / "cutracer.so"

    # ── Clone ────────────────────────────────────────────────────────────────
    if clone_dir.exists() and (clone_dir / "Makefile").exists():
        if progress:
            print(f"  Using existing clone at: {clone_dir}")
    else:
        if progress:
            print("  Cloning CUTracer from GitHub …")
            print(f"    {CUTRACER_GITHUB}")
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["git", "clone", "--depth=1", CUTRACER_GITHUB, str(clone_dir)],
            capture_output=not progress,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"git clone failed (exit {r.returncode}):\n{getattr(r, 'stderr', '')}"
            )

    # ── install_third_party.sh — downloads NVBit + nlohmann/json ────────────
    third_party_nvbit = clone_dir / "third_party" / "nvbit"
    if not third_party_nvbit.exists():
        if progress:
            print("  Running install_third_party.sh (downloads NVBit …)")
        r = subprocess.run(
            ["bash", "install_third_party.sh"],
            cwd=clone_dir,
            capture_output=not progress,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"install_third_party.sh failed (exit {r.returncode}):\n"
                f"{getattr(r, 'stderr', '')}"
            )
    else:
        if progress:
            print("  third_party/nvbit already present — skipping download")

    # ── make ─────────────────────────────────────────────────────────────────
    if so_dest.exists():
        if progress:
            print(f"  .so already built: {so_dest}")
        return so_dest

    if progress:
        print("  Building cutracer.so (this takes a few minutes) …")
    r = subprocess.run(
        ["make", "-j4"],
        cwd=clone_dir,
        capture_output=not progress,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"make failed (exit {r.returncode}):\n"
            f"{getattr(r, 'stderr', '')}\n{getattr(r, 'stdout', '')}"
        )

    if not so_dest.exists():
        raise RuntimeError(
            f"make succeeded but {so_dest} not found.\n"
            f"stdout:\n{getattr(r, 'stdout', '')}"
        )
    return so_dest


# ---------------------------------------------------------------------------
# Build (legacy: explicit NVBit path)
# ---------------------------------------------------------------------------


def build_so(
    cutracer_src: Path,
    nvbit_root: Path,
    *,
    progress: bool = True,
) -> Path:
    """Compile the CUTracer NVBit tool and return the path to the built .so.

    Raises ``RuntimeError`` on build failure.
    """
    if progress:
        print(f"  Building CUTracer .so from: {cutracer_src}")
        print(f"  Using NVBit at: {nvbit_root}")

    env = {**os.environ, "NVBIT_PATH": str(nvbit_root)}
    result = subprocess.run(
        ["make", "-C", str(cutracer_src), "-j4"],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"make failed (exit {result.returncode}):\n"
            f"{result.stderr}\n{result.stdout}"
        )

    # Locate the built .so
    candidates = sorted(cutracer_src.glob("*.so")) + sorted(cutracer_src.glob("**/*.so"))
    if not candidates:
        raise RuntimeError(
            f"make succeeded but no .so found under {cutracer_src}.\n"
            f"stdout:\n{result.stdout}"
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# High-level install orchestrator
# ---------------------------------------------------------------------------


@dataclass
class InstallResult:
    success: bool
    so_path: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def install(
    *,
    install_dir: Path = INSTALL_DIR,
    nvbit_version: str = NVBIT_VERSION,
    dry_run: bool = False,
    progress: bool = True,
) -> InstallResult:
    """Orchestrate the full CUTracer .so build and install.

    Parameters
    ----------
    install_dir:
        Root directory for all managed artefacts (default: ``~/.nsys-ai/cutracer``).
    nvbit_version:
        NVBit release version to download.
    dry_run:
        Print what would be done without executing.
    progress:
        Print status messages to stdout.
    """
    errors: list[str] = []

    lib_dir = install_dir / "lib"
    so_dest = lib_dir / "cutracer.so"

    if dry_run:
        print("[dry-run] Would install to:", so_dest)
        print("[dry-run] NVBit version:", nvbit_version)
        print("[dry-run] NVBit URL:", nvbit_download_url(nvbit_version))
        return InstallResult(success=True, so_path=str(so_dest))

    # 1. Check prerequisites
    prereqs = check_prerequisites()
    missing = [p for p in prereqs if not p.ok]
    if missing:
        for p in missing:
            errors.append(f"Missing prerequisite: {p.name}  — {p.message}")
        return InstallResult(success=False, errors=errors)

    # 2. Detect CUDA
    cuda_ver = detect_cuda_version()
    if cuda_ver is None:
        errors.append("Could not detect CUDA version from nvcc. Is CUDA toolkit installed?")
        return InstallResult(success=False, errors=errors)
    if progress:
        print(f"  CUDA version: {cuda_ver[0]}.{cuda_ver[1]}")

    # 3. Find or clone CUTracer source and build
    #
    # Primary path:   GitHub clone (facebookresearch/CUTracer) — handles its
    #                 own NVBit download via install_third_party.sh.
    # Fallback path:  pre-existing local source + separate NVBit download.
    cutracer_src = find_cutracer_source()
    clone_dir = install_dir / "CUTracer"

    if cutracer_src is None and not (clone_dir / "Makefile").exists():
        # Neither a local source nor a previous clone — clone from GitHub.
        if progress:
            print("  No local CUTracer source found — cloning from GitHub …")
        try:
            built_so = _clone_and_build(clone_dir, progress=progress)
        except RuntimeError as exc:
            errors.append(str(exc))
            return InstallResult(success=False, errors=errors)

    elif (clone_dir / "Makefile").exists():
        # Previous GitHub clone exists — rebuild if .so is missing.
        try:
            built_so = _clone_and_build(clone_dir, progress=progress)
        except RuntimeError as exc:
            errors.append(str(exc))
            return InstallResult(success=False, errors=errors)

    else:
        # Local source found — use the legacy NVBit-separate build path.
        if cutracer_src is None:
            errors.append("Internal error: expected local CUTracer source path.")
            return InstallResult(success=False, errors=errors)
        nvbit_cache = install_dir / "nvbit"
        try:
            nvbit_root = download_nvbit(nvbit_cache, nvbit_version, progress=progress)
            built_so = build_so(cutracer_src, nvbit_root, progress=progress)
        except RuntimeError as exc:
            errors.append(str(exc))
            return InstallResult(success=False, errors=errors)

    # 6. Copy to managed lib dir
    lib_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_so, so_dest)
    if progress:
        print(f"  Installed: {so_dest}")

    return InstallResult(success=True, so_path=str(so_dest))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_prereq_table(results: list[PrereqResult]) -> str:
    lines = ["Prerequisite check:"]
    for r in results:
        status = "OK" if r.ok else "MISSING"
        ver = f" ({r.version})" if r.version and r.version != "?" else ""
        lines.append(f"  [{status:^7s}]  {r.name}{ver}")
        if not r.ok and r.message:
            lines.append(f"             {r.message}")
    return "\n".join(lines)
