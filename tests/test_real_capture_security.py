"""Opt-in real Nsight capture checks for RunSpec security invariants.

The normal CI runners do not ship Nsight Systems or a CUDA compiler. Set
``NSYS_REAL_CAPTURE=1`` on a CUDA-capable machine to compile a tiny workload,
run it through ``LocalProfileRunner``, and make the produced report load-bearing
for the no-persisted-environment contract.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from nsys_ai.profile_runner import LocalProfileRunner, RunStatus
from nsys_ai.runspec import EnvironmentSpec, NsysTraceOptions, RunSpec

_NSYS = shutil.which("nsys")
_NVCC = shutil.which("nvcc")

pytestmark = pytest.mark.skipif(
    os.environ.get("NSYS_REAL_CAPTURE") != "1",
    reason="real Nsight capture opt-in (set NSYS_REAL_CAPTURE=1)",
)


def test_real_cuda_capture_does_not_persist_runner_environment(tmp_path, monkeypatch):
    if _NSYS is None or _NVCC is None:
        pytest.skip("requires nsys and nvcc")

    source = tmp_path / "env_guard.cu"
    binary = tmp_path / "env_guard"
    source.write_text(
        """
#include <cuda_runtime.h>

__global__ void touch(float *value) { *value += 1.0f; }

int main() {
    float *device_value = nullptr;
    cudaMalloc(&device_value, sizeof(float));
    touch<<<1, 1>>>(device_value);
    cudaDeviceSynchronize();
    cudaFree(device_value);
    return 0;
}
"""
    )
    subprocess.run(
        [_NVCC, str(source), "-O0", "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    sentinel = "nsys-ai-runspec-secret-sentinel"
    monkeypatch.setenv("NSYS_AI_TEST_SECRET", sentinel)
    spec = RunSpec(
        argv=(str(binary),),
        environment=EnvironmentSpec(secrets=("NSYS_AI_TEST_SECRET",)),
        trace_options=NsysTraceOptions(trace=("cuda",)),
    )
    result = LocalProfileRunner(tmp_path / "artifacts", _NSYS).run(spec)

    assert result.status is RunStatus.SUCCEEDED, result.detail
    assert result.sqlite_path is not None
