"""
hardware.py — GPU name → peak TFLOPS lookup for MFU.

Nsight Systems profile has GPU name (e.g. from TARGET_INFO_GPU) but not
data precision. We expose BF16/FP16 Tensor Core peak as the default for
training; user can override if using FP8/TF32.

Lookup strategy:
    1. Static table covering all common NVIDIA GPUs (datacenter + consumer).
    2. Fallback: try nvidia-smi for the current host's GPU name.
"""

from __future__ import annotations

import re
import subprocess  # nosec B404 — only used for hardcoded nvidia-smi

# ---------------------------------------------------------------------------
# Peak TFLOPS for BF16/FP16 Tensor Core, DENSE (no sparsity).
# Sources: NVIDIA official datasheets.
# Key convention: canonical short names; we do substring matching.
# ---------------------------------------------------------------------------
GPU_PEAK_TFLOPS: dict[str, float] = {
    # === Blackwell (2024-2025) ===
    "B200": 2250.0,  # GB200/B200 SXM — BF16 dense
    "B100": 1750.0,  # BF16 dense
    "GB200": 2250.0,  # Grace Blackwell Superchip
    "RTX 5090": 838.0,  # GeForce RTX 5090 — BF16/FP16 dense
    "RTX 5080": 453.0,  # GeForce RTX 5080 — BF16/FP16 dense
    "RTX 5070 Ti": 370.0,  # GeForce RTX 5070 Ti — BF16/FP16 dense
    "RTX 5070": 246.0,  # GeForce RTX 5070 — BF16/FP16 dense
    # === Hopper (2022-2023) ===
    "H200": 989.0,  # H200 SXM — BF16 dense (same die as H100 SXM)
    "H100 SXM": 989.0,  # H100 SXM5 — BF16 dense
    "H100 80GB HBM3": 989.0,
    "H100 PCIe": 756.0,  # H100 PCIe — BF16 dense
    "H100 NVL": 835.0,  # H100 NVL — BF16 dense
    "H800": 989.0,  # H800 (China variant, same die)
    # === Ada Lovelace (2022-2023) ===
    "L40S": 362.0,  # L40S — BF16/FP16 dense
    "L40": 181.0,  # L40 — BF16/FP16 dense
    "L4": 121.0,  # L4 — BF16/FP16 dense
    "RTX 6000 Ada": 364.0,  # RTX 6000 Ada Generation
    "RTX 5880 Ada": 305.0,  # RTX 5880 Ada
    "RTX 5000 Ada": 200.0,  # RTX 5000 Ada
    "RTX 4500 Ada": 160.0,  # RTX 4500 Ada
    "RTX 4000 Ada": 102.0,  # RTX 4000 Ada
    "RTX 4090": 330.0,  # GeForce RTX 4090 — BF16/FP16 w/ FP32 accum (dense, effective)
    "RTX 4080 SUPER": 204.0,
    "RTX 4080": 204.0,  # GeForce RTX 4080 — BF16/FP16 dense
    "RTX 4070 Ti SUPER": 184.0,
    "RTX 4070 Ti": 184.0,
    "RTX 4070 SUPER": 175.0,
    "RTX 4070": 147.0,
    # === Ampere (2020-2022) ===
    "A100-SXM4-80GB": 312.0,
    "A100-SXM4-40GB": 312.0,
    "A100-PCIE-80GB": 312.0,
    "A100-PCIE-40GB": 312.0,
    "A100 80GB": 312.0,  # A100 80GB (SXM or PCIe) — BF16 dense
    "A100 40GB": 312.0,
    "A100X": 312.0,
    "A800": 312.0,  # A800 (China variant, same die)
    "A40": 150.0,  # A40 — FP16 dense
    "A30": 165.0,  # A30 — BF16 dense
    "A10G": 125.0,  # A10G (AWS) — BF16 dense
    "A10": 125.0,  # A10 — BF16 dense
    "A16": 16.9,  # A16 (per GPU, 4 GPUs per card)
    "A2": 36.0,
    "RTX A6000": 310.0,  # RTX A6000 Ampere
    "RTX A5500": 256.0,
    "RTX A5000": 222.0,
    "RTX A4500": 185.0,
    "RTX A4000": 153.0,
    "RTX 3090 Ti": 160.0,  # GeForce RTX 3090 Ti
    "RTX 3090": 142.0,  # GeForce RTX 3090
    "RTX 3080 Ti": 136.0,
    "RTX 3080": 119.0,
    "RTX 3070 Ti": 87.0,
    "RTX 3070": 81.0,
    # === Turing (2018-2019) ===
    "T4": 65.0,  # T4 — FP16 dense
    "RTX 2080 Ti": 53.8,  # GeForce RTX 2080 Ti
    # === Volta (2017-2018) ===
    "V100 SXM2": 125.0,  # V100 SXM2 — FP16 dense
    "V100S PCIe": 130.0,  # V100S PCIe
    "V100 PCIe": 112.0,  # V100 PCIe
    "V100": 112.0,  # V100 (PCIe fallback)
}


def get_peak_tflops(gpu_name: str) -> dict:
    """
    Look up peak TFLOPS (BF16/FP16 Tensor Core, dense) for a GPU name.

    Returns {"gpu_name": str, "peak_tflops": float} or {"gpu_name": str, "error": str}.
    Uses substring match (e.g. "NVIDIA L40S" → L40S).
    """
    if not (gpu_name or "").strip():
        return {"gpu_name": "", "error": "No GPU name provided"}
    name = gpu_name.strip()
    # Normalize for matching: remove "NVIDIA" prefix, collapse whitespace/hyphens
    norm = re.sub(r"\s+", " ", name.replace("-", " ").replace("NVIDIA", "").strip())
    # Try longest key first so "H100 80GB HBM3" wins over "H100"
    for key in sorted(GPU_PEAK_TFLOPS.keys(), key=len, reverse=True):
        key_norm = key.replace("-", " ")
        if key_norm in norm:
            return {"gpu_name": name, "peak_tflops": GPU_PEAK_TFLOPS[key]}
    return {"gpu_name": name, "error": f"Unknown GPU '{name}'; provide peak_tflops manually"}


def detect_gpu_from_nvidia_smi() -> dict:
    """
    Attempt to detect the GPU on the current host via nvidia-smi.

    Returns {"gpu_name": str, "peak_tflops": float, "source": "nvidia-smi"}
    or {"error": str}.

    ⚠️  WARNING: This detects the LOCAL machine's GPU, which may NOT match
    the GPU that generated the nsys profile.  Profiles are often collected
    on remote machines (Modal, cloud VMs, clusters) and analysed locally.
    Always prefer GPU info from the profile itself (TARGET_INFO_GPU /
    TARGET_INFO_CUDA_GPU) or explicit user-provided peak_tflops.
    """
    try:
        result = subprocess.run(  # nosec B603 B607 — hardcoded command, no user input
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {"error": f"nvidia-smi failed: {result.stderr.strip()}"}
        # Take the first GPU (line 0)
        gpu_name = result.stdout.strip().split("\n")[0].strip()
        if not gpu_name:
            return {"error": "nvidia-smi returned empty GPU name"}
        lookup = get_peak_tflops(gpu_name)
        lookup["source"] = "nvidia-smi"
        return lookup
    except FileNotFoundError:
        return {"error": "nvidia-smi not found; cannot auto-detect GPU"}
    except subprocess.TimeoutExpired:
        return {"error": "nvidia-smi timed out"}
    except Exception as e:
        return {"error": f"nvidia-smi failed: {e}"}
