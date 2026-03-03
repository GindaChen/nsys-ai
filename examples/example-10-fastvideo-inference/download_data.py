#!/usr/bin/env python3
"""
Download the FastVideo inference profile from HuggingFace.

Dataset: https://huggingface.co/datasets/GindaChen/nsys-hero
Profile: fastvideo/fastvideo_inference.sqlite

Downloads the pre-converted .sqlite file so you can immediately use nsys-ai
without needing Modal or GPU access.
"""
import os
import subprocess
import sys

HF_REPO = "GindaChen/nsys-hero"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SQLITE_FILE = "fastvideo_inference.sqlite"
HF_PATH = f"fastvideo/{SQLITE_FILE}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sqlite_path = os.path.join(OUTPUT_DIR, SQLITE_FILE)

    # Check if already downloaded
    if os.path.exists(sqlite_path):
        size_mb = os.path.getsize(sqlite_path) / 1e6
        print(f"✓ SQLite already exists: {SQLITE_FILE} ({size_mb:.1f} MB)")
        print("  Ready to use:")
        print(f"  nsys-ai info {sqlite_path}")
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

    print(f"↓ Downloading {SQLITE_FILE} from {HF_REPO}...")
    try:
        hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_PATH,
            repo_type="dataset",
            local_dir=OUTPUT_DIR,
            local_dir_use_symlinks=False,
        )
        # Move from subdirectory to output root
        downloaded = os.path.join(OUTPUT_DIR, HF_PATH)
        if os.path.exists(downloaded) and downloaded != sqlite_path:
            os.rename(downloaded, sqlite_path)
            try:
                os.rmdir(os.path.join(OUTPUT_DIR, "fastvideo"))
            except OSError:
                pass

        size_mb = os.path.getsize(sqlite_path) / 1e6
        print(f"✅ Downloaded: {SQLITE_FILE} ({size_mb:.1f} MB)")
        print()
        print("Next steps:")
        print(f"  nsys-ai info output/{SQLITE_FILE}")
        print(f"  nsys-ai timeline output/{SQLITE_FILE} --gpu 0")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print()
        print("This profile may not be uploaded to HuggingFace yet.")
        print("You can capture your own profile using Modal:")
        print("  modal run profile_inference.py")


if __name__ == "__main__":
    main()
