"""Execute a CUTracer-instrumented training run — locally or on Modal.

Two backends
------------
Local
    Builds the environment variable set, then calls ``subprocess.run`` with
    ``LD_PRELOAD`` pointing at the CUTracer ``.so``.  Requires a local GPU and
    a pre-built ``.so``.

Modal
    Generates a self-contained Modal app Python file.  The generated app:

    * Uses a CUDA devel image with ``make`` / ``g++`` so ``nsys-ai cutracer
      install`` can be baked into the image layer (cached after first run).
    * Mounts a ``modal.Volume`` for output so CSVs survive the container.
    * Downloads histogram CSVs back to a local directory from the
      ``local_entrypoint``.

    The generated script is human-readable and intentionally easy to
    customise — users are expected to adjust the base image, GPU type, and
    training command.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from nsys_ai.cutracer.planner import CutracerPlan

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Parameters shared by all runner backends."""

    launch_cmd: str
    """The training / inference command to wrap."""

    output_dir: Path
    """Local directory where histogram CSVs will land."""

    kernel_filter: list[str] = field(default_factory=list)
    """Normalised kernel name tokens for ``CUTRACER_KERNEL_FILTER``."""

    so_path: Path | None = None
    """Path to ``cutracer.so``.  ``None`` → auto-detect at runtime."""

    mode: str = "proton_instr_histogram"
    """CUTracer instrumentation mode."""

    max_iters: int | None = None
    """If set, sets ``CUTRACER_MAX_ITERS`` to limit profiling to N iterations."""


# ---------------------------------------------------------------------------
# Local runner
# ---------------------------------------------------------------------------


def _resolve_so(config: RunConfig) -> Path:
    """Return the .so path from config or from the managed install location."""
    if config.so_path and config.so_path.is_file():
        return config.so_path

    from nsys_ai.cutracer.installer import _find_cutracer_so_path

    managed = _find_cutracer_so_path()
    if managed:
        return Path(managed)

    raise FileNotFoundError(
        "cutracer.so not found. Run: nsys-ai cutracer install"
    )


def _build_cutracer_cmd(config: RunConfig, so_path: Path | None) -> list[str]:
    """Return the ``cutracer trace`` argv list for this config."""
    cmd = ["cutracer", "trace"]
    if so_path:
        cmd += ["--cutracer-so", str(so_path)]
    # proton_instr_histogram is an analysis (post-process), not an instrument
    cmd += ["--analysis", config.mode]
    if config.kernel_filter:
        cmd += ["--kernel-filters", ",".join(config.kernel_filter)]
    cmd += ["--output-dir", str(config.output_dir.resolve())]
    cmd += ["--"] + shlex.split(config.launch_cmd)
    return cmd


def run_local(
    config: RunConfig,
    *,
    dry_run: bool = False,
    progress: bool = True,
) -> Path:
    """Run the training command locally with CUTracer instrumentation.

    Returns the ``output_dir`` path containing histogram CSVs.
    Raises ``subprocess.CalledProcessError`` on non-zero exit.
    """
    try:
        so_path = _resolve_so(config)
    except FileNotFoundError:
        so_path = None  # allow logger-only mode (no --instrument, no .so)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    argv = _build_cutracer_cmd(config, so_path)

    if progress:
        print(f"  .so     : {so_path or '(not found — kernel logger mode only)'}")
        print(f"  analysis: {config.mode}")
        filter_str = ",".join(config.kernel_filter) if config.kernel_filter else "(all kernels)"
        print(f"  filter  : {filter_str}")
        print(f"  output  : {config.output_dir}")
        print(f"  cmd     : {config.launch_cmd}")

    if dry_run:
        print("\n[dry-run] Would run:")
        print("  " + " ".join(argv))
        return config.output_dir

    # If no .so, drop --analysis and run as kernel launch logger (no trace files)
    if so_path is None:
        argv = ["cutracer", "trace"]
        if config.kernel_filter:
            argv += ["--kernel-filters", ",".join(config.kernel_filter)]
        argv += ["--"] + shlex.split(config.launch_cmd)
        if progress:
            print("  (running in kernel launch logger mode — no .so available)")

    result = subprocess.run(
        argv,
        shell=False,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, argv[0])

    return config.output_dir


# ---------------------------------------------------------------------------
# Modal script generator
# ---------------------------------------------------------------------------


# CUDA devel images that include nvcc + headers required by NVBit build
_CUDA_IMAGE_MAP = {
    (12, 4): "nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04",
    (12, 3): "nvcr.io/nvidia/cuda:12.3.0-devel-ubuntu22.04",
    (12, 2): "nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04",
    (12, 1): "nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu22.04",
    (12, 0): "nvcr.io/nvidia/cuda:12.0.0-devel-ubuntu22.04",
    (11, 8): "nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04",
}
_CUDA_IMAGE_DEFAULT = "nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04"


def _cuda_image_for_version(cuda_ver: tuple[int, int] | None) -> str:
    if cuda_ver is None:
        return _CUDA_IMAGE_DEFAULT
    return _CUDA_IMAGE_MAP.get(cuda_ver, _CUDA_IMAGE_DEFAULT)


@dataclass
class ModalConfig:
    """Knobs for the generated Modal app."""

    gpu: str = "H100"
    """Modal GPU type string — e.g. ``'H100'``, ``'A100'``, ``'A10G'``."""

    cuda_image: str = _CUDA_IMAGE_DEFAULT
    """Base Docker image (must include nvcc for CUTracer build)."""

    volume_name: str = "cutracer-histograms"
    """Modal Volume name for storing histogram CSVs."""

    timeout: int = 3600
    """Function timeout in seconds."""

    extra_pip: list[str] = field(default_factory=list)
    """Extra pip packages to install in the image (e.g. your training deps)."""

    extra_apt: list[str] = field(default_factory=list)
    """Extra apt packages."""

    nsys_ai_version: str = ""
    """Pin nsys-ai version in the image (empty → latest)."""


def format_modal_app(
    plan: CutracerPlan,
    config: RunConfig,
    modal_cfg: ModalConfig | None = None,
    *,
    profile_path: str = "",
) -> str:
    """Return a complete, runnable Modal Python app as a string.

    The generated script:
    - Bakes ``nsys-ai cutracer install`` into the image (cached layer).
    - Runs the training command with CUTracer ``LD_PRELOAD`` on a GPU.
    - Downloads histogram CSVs from the Modal Volume back to local disk.
    """
    if modal_cfg is None:
        modal_cfg = ModalConfig()

    filter_csv = ",".join(config.kernel_filter) if config.kernel_filter else ""
    output_dir_str = str(config.output_dir)
    # Use repr() for Python string literals inside the generated script;
    # use shlex.quote() only for shell command-line fragments (analyze_cmd comment).
    local_out_repr = repr(output_dir_str)
    local_out_shell = shlex.quote(output_dir_str)
    nsys_ai_pin = f"nsys-ai=={modal_cfg.nsys_ai_version}" if modal_cfg.nsys_ai_version else "nsys-ai"
    extra_pip_lines = ""
    if modal_cfg.extra_pip:
        pkgs = ", ".join(f'"{p}"' for p in modal_cfg.extra_pip)
        extra_pip_lines = f"\n    .pip_install({pkgs})"
    extra_apt_lines = ""
    if modal_cfg.extra_apt:
        pkgs = ", ".join(f'"{p}"' for p in modal_cfg.extra_apt)
        extra_apt_lines = f'\n    .apt_install({pkgs})'

    # Top-kernel table for the comment header
    kernel_comments = ""
    if plan.targets:
        rows = []
        for i, t in enumerate(plan.targets, 1):
            short = t.name if len(t.name) <= 65 else t.name[:62] + "…"
            rows.append(f"#   {i}. {t.total_ms:8.2f} ms  {t.pct_of_gpu:5.1f}%  {short}")
        kernel_comments = "\n".join(rows)
    else:
        kernel_comments = "#   (no kernels found)"

    analyze_cmd = f"nsys-ai cutracer analyze {shlex.quote(profile_path or '<profile.sqlite>')} {local_out_shell}"

    script = f'''\
#!/usr/bin/env python3
# Auto-generated by: nsys-ai cutracer plan --modal
# Profile: {profile_path or "<profile.sqlite>"}
#
# Top kernels targeted for instrumentation:
{kernel_comments}
#
# Usage:
#   1. Adjust the image / GPU / LAUNCH_CMD below for your environment.
#   2. modal run {Path(output_dir_str).stem}_cutracer.py
#   3. {analyze_cmd}
#
# Requirements:  pip install modal nsys-ai
#                modal token new   (first time only)

import os
import modal

# ---------------------------------------------------------------------------
# Image — bakes CUTracer .so build into a cached layer
# ---------------------------------------------------------------------------
# TODO: Replace with your training image or add .pip_install(...) for your deps.
image = (
    modal.Image.from_registry(
        "{modal_cfg.cuda_image}",
        add_python="3.11",
    )
    .apt_install("git", "make", "g++", "curl", "libzstd-dev"){extra_apt_lines}
    .pip_install(
        "{nsys_ai_pin}",
        "cutracer>=0.2.0",
    ){extra_pip_lines}
    .run_commands(
        # Clone and build CUTracer NVBit .so (cached layer after first image build)
        "git clone --depth=1 https://github.com/facebookresearch/CUTracer /opt/CUTracer",
        "cd /opt/CUTracer && ./install_third_party.sh",
        "cd /opt/CUTracer && make",
        "mkdir -p /root/.nsys-ai/cutracer/lib && cp /opt/CUTracer/lib/cutracer.so /root/.nsys-ai/cutracer/lib/",
    )
)

app = modal.App("nsys-ai-cutracer")
vol = modal.Volume.from_name("{modal_cfg.volume_name}", create_if_missing=True)


# ---------------------------------------------------------------------------
# GPU function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="{modal_cfg.gpu}",
    volumes={{"/cutracer_out": vol}},
    timeout={modal_cfg.timeout},
)
def run_with_cutracer(
    launch_cmd: str,
    kernel_filter: str,
    analysis: str = "{config.mode}",
    max_iters: int | None = None,
) -> None:
    import csv
    import re
    import shlex
    import subprocess
    from pathlib import Path

    from nsys_ai.cutracer.installer import INSTALL_DIR
    from nsys_ai.cutracer.parser import sass_resolve_dir

    so_path = str(INSTALL_DIR / "lib" / "cutracer.so")
    has_so = Path(so_path).is_file()

    # Build cutracer trace command — launch_cmd split via shlex to handle quoted args
    argv = ["cutracer", "trace"]
    if has_so:
        argv += ["--cutracer-so", so_path]
    else:
        print(f"WARNING: cutracer.so not found at {{so_path}} — running in kernel logger mode only")
    argv += ["--analysis", analysis]
    if kernel_filter:
        argv += ["--kernel-filters", kernel_filter]
    argv += ["--output-dir", "/cutracer_out"]
    argv += ["--"] + shlex.split(launch_cmd)

    print("==> CUTracer run starting")
    print(f"    so      : {{so_path if has_so else '(not found)'}}")
    print(f"    analysis: {{analysis}}")
    print(f"    filter  : {{kernel_filter or '(all)'}}")
    print(f"    output  : /cutracer_out")
    print(f"    argv    : {{' '.join(argv)}}")

    result = subprocess.run(argv, shell=False)
    if result.returncode != 0:
        vol.commit()
        raise SystemExit(f"Training command exited with code {{result.returncode}}")

    # -----------------------------------------------------------------------
    # Post-process: resolve opcode_id → SASS mnemonic, write _hist.csv files.
    # sass_resolve_dir (from nsys_ai.cutracer.parser, installed above) handles
    # the cutracer sass + SASS disassembly pipeline.  We just write CSVs from
    # the returned KernelHistogram objects so they survive Volume download.
    # -----------------------------------------------------------------------
    out_dir = Path("/cutracer_out")

    # Capture base_name → hash mapping from ndjson filenames before SASS resolution,
    # so we can reconstruct the kernel_<name>_<hash>_hist.csv naming expected by the parser.
    hash_map: dict = {{}}
    for ndf in sorted(out_dir.glob("*.ndjson")):
        base = re.sub(r"_iter\\d+", "", ndf.stem)
        parts = base.split("_", 2)
        if len(parts) >= 3:
            arch = "_".join(base.split("_")[2:])
            hash_map[arch] = parts[1]

    hists = sass_resolve_dir(out_dir)
    if not hists:
        print("WARNING: sass_resolve_dir returned no histograms — check cutracer/nvdisasm output")

    for arch_suffix, hist in hists.items():
        hash_part = hash_map.get(arch_suffix, "0")
        csv_path = out_dir / f"kernel_{{arch_suffix}}_{{hash_part}}_hist.csv"
        with open(csv_path, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["warp_id", "region_id", "instruction", "count", "cycles"])
            for mn, cnt in sorted(hist.instruction_counts.items(), key=lambda x: -x[1]):
                writer.writerow([0, 0, mn, cnt, 0])
        total = sum(hist.instruction_counts.values())
        print(f"==> Wrote {{csv_path.name}} ({{total:,}} instructions, {{len(hist.instruction_counts)}} opcodes)")

    vol.commit()  # flush Volume before container exits


# ---------------------------------------------------------------------------
# Local entrypoint — runs the function, then downloads CSVs
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main() -> None:
    # TODO: Replace with your actual training command (short profiling run, not full training).
    launch_cmd = {repr(config.launch_cmd or "python train.py  # TODO: replace")}
    kernel_filter = {repr(filter_csv)}
    max_iters = {config.max_iters!r}  # set to e.g. 10 to profile only 10 iterations

    print("==> Submitting CUTracer run to Modal ...")
    run_with_cutracer.remote(
        launch_cmd=launch_cmd,
        kernel_filter=kernel_filter,
        max_iters=max_iters,
    )

    # Download histogram CSVs from Modal Volume to local disk
    print(f"\\n==> Downloading histogram CSVs ...")
    local_dir = {local_out_repr}
    os.makedirs(local_dir, exist_ok=True)

    downloaded = 0
    for entry in vol.iterdir("/"):
        fname = os.path.basename(entry.path)
        if entry.path.endswith("_hist.csv"):
            data = b"".join(vol.read_file(entry.path))  # read_file returns a generator
            dest = os.path.join(local_dir, fname)
            with open(dest, "wb") as f:
                f.write(data)
            print(f"    downloaded: {{dest}}")
            downloaded += 1

    if downloaded == 0:
        print("  WARNING: no *_hist.csv files found in volume.")
        print("  Check that the training command ran successfully and")
        print("  CUTracer wrote output to /cutracer_out inside the container.")
    else:
        print(f"\\n==> {{downloaded}} CSV(s) downloaded to: {{local_dir}}")
        print(f"    Analyze with:")
        print(f"      {analyze_cmd}")
'''
    return script
