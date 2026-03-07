"""
diff.py — Structured before/after comparison for Nsight Systems profiles.

This module computes a stable, structured diff payload that can be rendered
as terminal/markdown/json output and later reused by a web compare UI.
"""

from __future__ import annotations

from dataclasses import dataclass

from .overlap import overlap_analysis
from .profile import Profile


@dataclass(frozen=True)
class KernelAgg:
    key: str
    name: str
    demangled: str
    total_ns: int
    count: int
    avg_ns: float
    min_ns: int
    max_ns: int


@dataclass(frozen=True)
class NvtxAgg:
    text: str
    total_ns: int
    count: int
    avg_ns: float


@dataclass(frozen=True)
class ProfileSummary:
    path: str
    gpu: int | None
    schema_version: str | None
    total_gpu_ns: int
    kernel_rows: int
    kernels: list[KernelAgg]
    nvtx: list[NvtxAgg]
    overlap: dict


@dataclass(frozen=True)
class KernelDiff:
    key: str
    name: str
    demangled: str
    before_total_ns: int
    after_total_ns: int
    delta_ns: int
    before_count: int
    after_count: int
    delta_count: int
    before_avg_ns: float
    after_avg_ns: float
    delta_avg_ns: float
    before_share: float
    after_share: float
    delta_share: float
    classification: str  # regression|improvement|new|removed|neutral


@dataclass(frozen=True)
class NvtxDiff:
    text: str
    before_total_ns: int
    after_total_ns: int
    delta_ns: int
    before_count: int
    after_count: int
    delta_count: int
    classification: str


@dataclass(frozen=True)
class ProfileDiffSummary:
    before: ProfileSummary
    after: ProfileSummary
    warnings: list[str]
    kernel_diffs: list[KernelDiff]
    nvtx_diffs: list[NvtxDiff]
    overlap_before: dict
    overlap_after: dict
    overlap_delta: dict
    top_regressions: list[KernelDiff]
    top_improvements: list[KernelDiff]


def _safe_int(x) -> int:
    try:
        return int(x or 0)
    except Exception:
        return 0


def build_profile_summary(
    prof: Profile,
    gpu: int | None,
    trim: tuple[int, int] | None,
    *,
    nvtx_limit: int | None = None,
) -> ProfileSummary:
    kernel_rows = prof.meta.kernel_count
    if gpu is not None:
        devices = getattr(prof.meta, "devices", [])
        gpu_info = getattr(prof.meta, "gpu_info", None)
        if gpu_info is not None and gpu in gpu_info:
            kernel_rows = gpu_info[gpu].kernel_count
        elif gpu not in devices:
            # Requested GPU not present in profile; treat as zero rows so
            # sanity checks stay consistent with the empty aggregation.
            kernel_rows = 0
    agg = prof.aggregate_kernels(gpu, trim=trim, limit=None)
    kernels: list[KernelAgg] = []
    for r in agg:
        name = str(r.get("name") or "")
        demangled = str(r.get("demangled") or "")
        key = demangled or name
        kernels.append(
            KernelAgg(
                key=key,
                name=name,
                demangled=demangled,
                total_ns=_safe_int(r.get("total_ns")),
                count=_safe_int(r.get("count")),
                avg_ns=float(r.get("avg_ns") or 0.0),
                min_ns=_safe_int(r.get("min_ns")),
                max_ns=_safe_int(r.get("max_ns")),
            )
        )

    total_gpu_ns = sum(k.total_ns for k in kernels) if kernels else 0

    nvtx_rows = prof.aggregate_nvtx_ranges(trim=trim, limit=nvtx_limit)
    nvtx: list[NvtxAgg] = []
    for r in nvtx_rows:
        text = str(r.get("text") or "")
        nvtx.append(
            NvtxAgg(
                text=text,
                total_ns=_safe_int(r.get("total_ns")),
                count=_safe_int(r.get("count")),
                avg_ns=float(r.get("avg_ns") or 0.0),
            )
        )

    if gpu is not None:
        overlap = overlap_analysis(prof, gpu, trim=trim)
    else:
        # Node-wide aggregation: sum up individual GPU overlap stats.
        overlap = {
            "compute_only_ms": 0.0,
            "nccl_only_ms": 0.0,
            "overlap_ms": 0.0,
            "idle_ms": 0.0,
            "total_ms": 0.0,
            "overlap_pct": 0.0,
            "compute_kernels": 0,
            "nccl_kernels": 0,
        }
        devices = prof.meta.devices if prof.meta.devices else []
        for dev in devices:
            dev_stats = overlap_analysis(prof, dev, trim=trim)
            if "error" not in dev_stats:
                overlap["compute_only_ms"] += dev_stats.get("compute_only_ms", 0.0)
                overlap["nccl_only_ms"] += dev_stats.get("nccl_only_ms", 0.0)
                overlap["overlap_ms"] += dev_stats.get("overlap_ms", 0.0)
                overlap["idle_ms"] += dev_stats.get("idle_ms", 0.0)
                overlap["total_ms"] += dev_stats.get("total_ms", 0.0)
                overlap["compute_kernels"] += dev_stats.get("compute_kernels", 0)
                overlap["nccl_kernels"] += dev_stats.get("nccl_kernels", 0)

        # Round logic to avoid float drift, and set a clean combined overlap pct
        # Node-wide idle logic is tricky because overlap might not overlap across GPUs perfectly,
        # but summing them gives a sense of "total wasted throughput".
        if overlap["nccl_only_ms"] + overlap["overlap_ms"] > 0:
            c_nccl = overlap["nccl_only_ms"] + overlap["overlap_ms"]
            overlap["overlap_pct"] = round(100 * overlap["overlap_ms"] / c_nccl, 1)

        for k in ("compute_only_ms", "nccl_only_ms", "overlap_ms", "idle_ms", "total_ms"):
            overlap[k] = round(overlap[k], 2)

    return ProfileSummary(
        path=prof.path,
        gpu=gpu,
        schema_version=prof.schema.version,
        total_gpu_ns=total_gpu_ns,
        kernel_rows=kernel_rows,
        kernels=kernels,
        nvtx=nvtx,
        overlap=overlap,
    )


def collect_sanity_warnings(before: ProfileSummary, after: ProfileSummary) -> list[str]:
    warnings: list[str] = []
    if (
        before.schema_version
        and after.schema_version
        and before.schema_version != after.schema_version
    ):
        warnings.append(
            f"Nsight schema/version differs: before='{before.schema_version}' after='{after.schema_version}'."
        )
    if before.gpu is not None and after.gpu is not None and before.gpu != after.gpu:
        warnings.append("Different GPU IDs selected between before/after (unexpected).")
    if before.kernel_rows and after.kernel_rows:
        # Very rough signal for "not comparable"
        ratio = max(before.kernel_rows, after.kernel_rows) / max(
            1, min(before.kernel_rows, after.kernel_rows)
        )
        if ratio >= 3.0:
            warnings.append(
                f"Kernel row counts differ a lot (before={before.kernel_rows}, after={after.kernel_rows}); compare may be dominated by workload differences."
            )

    b_keys = {k.key for k in before.kernels}
    a_keys = {k.key for k in after.kernels}
    if b_keys and a_keys:
        shared = b_keys.intersection(a_keys)
        # If both profiles have a meaningful amount of kernels but share fewer than 5% of their names
        if len(b_keys) > 5 and len(a_keys) > 5:
            overlap_pct = len(shared) / min(len(b_keys), len(a_keys))
            if overlap_pct < 0.05:
                warnings.append(
                    f"Profiles share almost no common kernels ({len(shared)} shared out of {len(b_keys)} and {len(a_keys)}). Are you comparing unrelated traces?"
                )

    if before.overlap.get("error") or after.overlap.get("error"):
        warnings.append("Overlap analysis unavailable (missing kernels or schema).")
    return warnings


def _classify_delta(delta_ns: int, before_ns: int, after_ns: int) -> str:
    if before_ns <= 0 and after_ns > 0:
        return "new"
    if after_ns <= 0 and before_ns > 0:
        return "removed"
    if delta_ns > 0:
        return "regression"
    if delta_ns < 0:
        return "improvement"
    return "neutral"


def diff_profiles(
    before_prof: Profile,
    after_prof: Profile,
    *,
    gpu: int | None,
    trim: tuple[int, int] | None = None,
    trim_before: tuple[int, int] | None = None,
    trim_after: tuple[int, int] | None = None,
    limit: int = 15,
    sort: str = "delta",
    nvtx_limit: int | None = 200,
) -> ProfileDiffSummary:
    """Compare two profiles. Use trim for same window, or trim_before/trim_after for iteration diff."""
    t_before = trim_before if trim_before is not None else trim
    t_after = trim_after if trim_after is not None else trim
    before = build_profile_summary(before_prof, gpu, t_before, nvtx_limit=nvtx_limit)
    after = build_profile_summary(after_prof, gpu, t_after, nvtx_limit=nvtx_limit)
    warnings = collect_sanity_warnings(before, after)

    before_by_key = {k.key: k for k in before.kernels}
    after_by_key = {k.key: k for k in after.kernels}
    keys = sorted(set(before_by_key) | set(after_by_key))

    kernel_diffs: list[KernelDiff] = []
    for key in keys:
        b = before_by_key.get(key)
        a = after_by_key.get(key)
        b_total = b.total_ns if b else 0
        a_total = a.total_ns if a else 0
        delta = a_total - b_total
        b_cnt = b.count if b else 0
        a_cnt = a.count if a else 0
        cls = _classify_delta(delta, b_total, a_total)
        b_share = (b_total / before.total_gpu_ns) if before.total_gpu_ns else 0.0
        a_share = (a_total / after.total_gpu_ns) if after.total_gpu_ns else 0.0
        kernel_diffs.append(
            KernelDiff(
                key=key,
                name=(a.name if a else (b.name if b else key)),
                demangled=(a.demangled if a else (b.demangled if b else "")),
                before_total_ns=b_total,
                after_total_ns=a_total,
                delta_ns=delta,
                before_count=b_cnt,
                after_count=a_cnt,
                delta_count=a_cnt - b_cnt,
                before_avg_ns=(b.avg_ns if b else 0.0),
                after_avg_ns=(a.avg_ns if a else 0.0),
                delta_avg_ns=(a.avg_ns if a else 0.0) - (b.avg_ns if b else 0.0),
                before_share=b_share,
                after_share=a_share,
                delta_share=a_share - b_share,
                classification=cls,
            )
        )

    # NVTX diff (by text)
    before_nvtx = {n.text: n for n in before.nvtx}
    after_nvtx = {n.text: n for n in after.nvtx}
    nvtx_keys = sorted(set(before_nvtx) | set(after_nvtx))
    nvtx_diffs: list[NvtxDiff] = []
    for text in nvtx_keys:
        b = before_nvtx.get(text)
        a = after_nvtx.get(text)
        b_total = b.total_ns if b else 0
        a_total = a.total_ns if a else 0
        delta = a_total - b_total
        b_cnt = b.count if b else 0
        a_cnt = a.count if a else 0
        cls = _classify_delta(delta, b_total, a_total)
        nvtx_diffs.append(
            NvtxDiff(
                text=text,
                before_total_ns=b_total,
                after_total_ns=a_total,
                delta_ns=delta,
                before_count=b_cnt,
                after_count=a_cnt,
                delta_count=a_cnt - b_cnt,
                classification=cls,
            )
        )

    # Sorting & top lists
    def sort_key(kd: KernelDiff):
        if sort == "percent":
            base = kd.before_total_ns
            return (
                (kd.delta_ns / base)
                if base
                else (float("inf") if kd.delta_ns > 0 else float("-inf"))
            )
        if sort == "total":
            return kd.after_total_ns
        # default: delta
        return kd.delta_ns

    regressions = [k for k in kernel_diffs if k.delta_ns > 0]
    improvements = [k for k in kernel_diffs if k.delta_ns < 0]
    regressions.sort(key=sort_key, reverse=True)
    improvements.sort(key=sort_key)  # most negative first

    overlap_before = before.overlap
    overlap_after = after.overlap
    overlap_delta = {}
    for key in (
        "compute_only_ms",
        "nccl_only_ms",
        "overlap_ms",
        "idle_ms",
        "total_ms",
        "overlap_pct",
    ):
        if key in overlap_before and key in overlap_after:
            try:
                overlap_delta[key] = round(
                    float(overlap_after[key]) - float(overlap_before[key]), 3
                )
            except Exception:
                pass

    return ProfileDiffSummary(
        before=before,
        after=after,
        warnings=warnings,
        kernel_diffs=kernel_diffs,
        nvtx_diffs=nvtx_diffs,
        overlap_before=overlap_before,
        overlap_after=overlap_after,
        overlap_delta=overlap_delta,
        top_regressions=regressions[: max(0, int(limit))],
        top_improvements=improvements[: max(0, int(limit))],
    )
