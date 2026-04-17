"""CUTracer integration for nsys-ai.

Provides offline analysis of CUTracer instruction-level traces,
correlating them with NVTX attribution data from Nsight Systems profiles.

Typical flow:
    1. ``nsys-ai cutracer plan profile.sqlite``  → generates shell script
    2. User runs the script on a GPU machine     → produces trace dir
    3. ``nsys-ai cutracer analyze profile.sqlite <trace_dir>``  → report
"""

from .correlator import match_kernels, normalize_kernel_name
from .installer import InstallResult, check_prerequisites, install
from .kernel_classifier import classify_kernel, instrumentation_priority
from .parser import KernelHistogram, parse_histogram_dir
from .planner import CutracerPlan, KernelTarget, build_plan, format_plan_script
from .report import format_kernel_report, summarize_all
from .runner import ModalConfig, RunConfig, format_modal_app, run_local

__all__ = [
    "KernelHistogram",
    "parse_histogram_dir",
    "normalize_kernel_name",
    "match_kernels",
    "classify_kernel",
    "instrumentation_priority",
    "format_kernel_report",
    "summarize_all",
    "CutracerPlan",
    "KernelTarget",
    "build_plan",
    "format_plan_script",
    "check_prerequisites",
    "install",
    "InstallResult",
    "RunConfig",
    "ModalConfig",
    "run_local",
    "format_modal_app",
]
