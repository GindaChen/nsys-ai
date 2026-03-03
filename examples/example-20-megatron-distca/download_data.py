#!/usr/bin/env python3
"""
Download the Megatron-LM DistCA profile from HuggingFace.

Dataset: https://huggingface.co/datasets/GindaChen/nsys-hero
Profile: distca-0/baseline.t128k.host-fs-mbz-gpu-899.sqlite
         → saved locally as output/megatron_distca.sqlite

Downloads the pre-converted .sqlite file so you can immediately use nsys-ai
without needing nsys or Modal installed.
"""
import os
import subprocess
import sys

HF_REPO = "GindaChen/nsys-hero"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Remote names on HuggingFace
_HF_SQLITE = "distca-0/baseline.t128k.host-fs-mbz-gpu-899.sqlite"
_HF_NSYS = "distca-0/baseline.t128k.host-fs-mbz-gpu-899.nsys-rep"

# Simplified local names
LOCAL_SQLITE = "megatron_distca.sqlite"
LOCAL_NSYS = "megatron_distca.nsys-rep"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sqlite_path = os.path.join(OUTPUT_DIR, LOCAL_SQLITE)
    nsys_path = os.path.join(OUTPUT_DIR, LOCAL_NSYS)

    # Check if already downloaded
    if os.path.exists(sqlite_path):
        size_mb = os.path.getsize(sqlite_path) / 1e6
        print(f"✓ SQLite already exists: {LOCAL_SQLITE} ({size_mb:.1f} MB)")
        print("  Ready to use:")
        print(f"  nsys-ai info output/{LOCAL_SQLITE}")
        return

    # Try to import huggingface_hub, install if missing
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "huggingface_hub"],
            capture_output=True,
        )
        from huggingface_hub import hf_hub_download

    # Try downloading pre-converted .sqlite first
    print(f"↓ Downloading profile from {HF_REPO}...")
    try:
        hf_hub_download(
            repo_id=HF_REPO,
            filename=_HF_SQLITE,
            repo_type="dataset",
            local_dir=OUTPUT_DIR,
            local_dir_use_symlinks=False,
        )
        # Move from HF subdirectory to simplified local name
        downloaded = os.path.join(OUTPUT_DIR, _HF_SQLITE)
        if os.path.exists(downloaded):
            os.rename(downloaded, sqlite_path)
            try:
                os.rmdir(os.path.join(OUTPUT_DIR, "distca-0"))
            except OSError:
                pass

        size_mb = os.path.getsize(sqlite_path) / 1e6
        print(f"✅ Downloaded: {LOCAL_SQLITE} ({size_mb:.1f} MB)")
        print()
        print("Next steps:")
        print(f"  nsys-ai info output/{LOCAL_SQLITE}")
        print(f"  nsys-ai timeline output/{LOCAL_SQLITE} --gpu 4 --trim 39 42")
        return
    except Exception as e:
        print(f"⚠  SQLite not available on HuggingFace: {e}")
        print("   Falling back to .nsys-rep download...")

    # Fallback: download .nsys-rep
    print(f"↓ Downloading .nsys-rep from {HF_REPO}...")
    hf_hub_download(
        repo_id=HF_REPO,
        filename=_HF_NSYS,
        repo_type="dataset",
        local_dir=OUTPUT_DIR,
        local_dir_use_symlinks=False,
    )
    downloaded = os.path.join(OUTPUT_DIR, _HF_NSYS)
    if os.path.exists(downloaded):
        os.rename(downloaded, nsys_path)
        try:
            os.rmdir(os.path.join(OUTPUT_DIR, "distca-0"))
        except OSError:
            pass

    size_mb = os.path.getsize(nsys_path) / 1e6
    print(f"✅ Downloaded: {LOCAL_NSYS} ({size_mb:.1f} MB)")
    print()
    print("To convert to SQLite, run:")
    print(f"  nsys export --type sqlite output/{LOCAL_NSYS}")


if __name__ == "__main__":
    main()
