"""
hardware.py — GPU name → peak TFLOPS lookup for MFU.

Nsight Systems profile has GPU name (e.g. from TARGET_INFO_GPU) but not
data precision. We expose BF16/FP16 Tensor Core peak as the default for
training; user can override if using FP8/TF32.
"""

from __future__ import annotations

# Peak TFLOPS for BF16/FP16 Tensor Core (typical training), dense (no sparsity).
# Sources: NVIDIA datasheets (A100, H100, L40). Prefer conservative figures.
GPU_PEAK_TFLOPS: dict[str, float] = {
    "H100 SXM": 989.0,
    "H100 80GB HBM3": 989.0,
    "H100 PCIe": 756.0,
    "H100 NVL": 835.0,
    "A100-SXM4-80GB": 312.0,
    "A100-SXM4-40GB": 312.0,
    "A100-PCIE-80GB": 312.0,
    "A100-PCIE-40GB": 312.0,
    "A100 80GB": 312.0,
    "A100 40GB": 312.0,
    "RTX 4090": 330.0,
    "RTX 3090": 142.0,
    "L40": 362.0,
}


def get_peak_tflops(gpu_name: str) -> dict:
    """
    Look up peak TFLOPS (BF16/FP16 Tensor Core) for a GPU name from nsys profile.

    Returns {"gpu_name": str, "peak_tflops": float} or {"gpu_name": str, "error": str}.
    Uses substring match (e.g. "NVIDIA H100 80GB HBM3" → H100 80GB HBM3).
    """
    if not (gpu_name or "").strip():
        return {"gpu_name": "", "error": "No GPU name provided"}
    name = gpu_name.strip()
    # Try longest key first so "H100 80GB HBM3" wins over "H100"
    for key in sorted(GPU_PEAK_TFLOPS.keys(), key=len, reverse=True):
        if key.replace("-", " ") in name.replace("-", " "):
            return {"gpu_name": name, "peak_tflops": GPU_PEAK_TFLOPS[key]}
    return {"gpu_name": name, "error": "Unknown GPU; provide peak_tflops from your GPU spec"}
