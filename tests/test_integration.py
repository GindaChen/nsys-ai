"""Integration smoke tests: require a real profile (.sqlite, .nsys-rep, or path).

Use NSYS_TEST_PROFILE to point at a profile; CLI accepts .sqlite/.nsys-rep and
converts when needed. Optional NSYS_TEST_GPU and NSYS_TEST_TRIM (e.g. "39 42")
override GPU id and time window; if unset, they are derived from profile metadata.
"""
import os
import subprocess
import sys
import tempfile

import pytest

PROFILE_ENV = "NSYS_TEST_PROFILE"
GPU_ENV = "NSYS_TEST_GPU"
TRIM_ENV = "NSYS_TEST_TRIM"
DEFAULT_PROFILE = "data/nsys-hero/distca-0/baseline.t128k.host-fs-mbz-gpu-899"


def _profile_path():
    path = os.environ.get(PROFILE_ENV, DEFAULT_PROFILE)
    return path if os.path.exists(path) else None


def _test_gpu_trim():
    """Return (gpu, trim_start_s, trim_end_s) from env or profile metadata."""
    path = _profile_path()
    if path is None:
        return None, None, None
    gpu_env = os.environ.get(GPU_ENV)
    trim_env = os.environ.get(TRIM_ENV)
    if gpu_env is not None and trim_env is not None:
        parts = trim_env.split()
        if len(parts) >= 2:
            return int(gpu_env), float(parts[0]), float(parts[1])
    from nsys_tui.profile import open as profile_open
    prof = profile_open(path)
    try:
        if gpu_env is not None:
            gpu = int(gpu_env)
        else:
            devices = getattr(prof.meta, "devices", []) or []
            if not devices:
                pytest.skip("No GPU devices in profile for integration tests")
            gpu = devices[0]
        if trim_env is not None:
            parts = trim_env.split()
            if len(parts) >= 2:
                return gpu, float(parts[0]), float(parts[1])
        start_s = prof.meta.time_range[0] / 1e9
        end_s = prof.meta.time_range[1] / 1e9
        return gpu, start_s, end_s
    finally:
        prof.close()


@pytest.mark.skipif(_profile_path() is None, reason="No test profile")
def test_info():
    path = _profile_path()
    r = subprocess.run(
        [sys.executable, "-m", "nsys_tui", "info", path],
        capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    assert "GPU" in r.stdout or "Kernels" in r.stdout


@pytest.mark.skipif(_profile_path() is None, reason="No test profile")
def test_summary():
    path = _profile_path()
    r = subprocess.run(
        [sys.executable, "-m", "nsys_tui", "summary", path],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0


@pytest.mark.skipif(_profile_path() is None, reason="No test profile")
def test_analyze():
    path = _profile_path()
    gpu, t0, t1 = _test_gpu_trim()
    r = subprocess.run(
        [sys.executable, "-m", "nsys_tui", "analyze", path, "--gpu", str(gpu), "--trim", str(t0), str(t1)],
        capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    assert "Span:" in r.stdout or "Kernels:" in r.stdout


@pytest.mark.skipif(_profile_path() is None, reason="No test profile")
def test_analyze_markdown_output():
    path = _profile_path()
    gpu, t0, t1 = _test_gpu_trim()
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        out = f.name
    try:
        r = subprocess.run(
            [sys.executable, "-m", "nsys_tui", "analyze", path, "--gpu", str(gpu), "--trim", str(t0), str(t1), "-o", out],
            capture_output=True, text=True, timeout=30)
        assert r.returncode == 0
        assert os.path.getsize(out) > 100
    finally:
        os.unlink(out)


@pytest.mark.skipif(_profile_path() is None, reason="No test profile")
def test_export_perfetto_json():
    path = _profile_path()
    gpu, t0, t1 = _test_gpu_trim()
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run(
            [sys.executable, "-m", "nsys_tui", "export", path, "--gpu", str(gpu), "--trim", str(t0), str(t1), "-o", d],
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0
        assert any(f.startswith("trace_gpu") and f.endswith(".json") for f in os.listdir(d))


@pytest.mark.skipif(_profile_path() is None, reason="No test profile")
def test_export_csv():
    path = _profile_path()
    gpu, t0, t1 = _test_gpu_trim()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        out = f.name
    try:
        r = subprocess.run(
            [sys.executable, "-m", "nsys_tui", "export-csv", path, "--gpu", str(gpu), "--trim", str(t0), str(t1), "-o", out],
            capture_output=True, text=True, timeout=30)
        assert r.returncode == 0
        assert os.path.getsize(out) > 50
    finally:
        os.unlink(out)
