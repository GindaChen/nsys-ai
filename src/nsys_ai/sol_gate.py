"""Absolute speed-of-light gate for the diff CI check.

The relative gate (``diff --gate PCT``) asks "did this get more than PCT worse
than the baseline". This asks a different and more stable question: "is this
region still within PCT of what the hardware can do". An absolute target does
not drift when the baseline itself is slow, which makes it a better long-lived
CI assertion.

The speed-of-light computation is *not* reimplemented here — it reuses the
``region_mfu`` skill and its ``_sol_headroom_ms`` helper, so there is exactly
one definition of achieved-vs-peak in the codebase.

A note on failing loudly: the model FLOPs needed for MFU cannot be derived from
a trace, so it must be supplied. When it is missing this module raises rather
than skipping the check — a CI gate that silently does not run is worse than no
gate at all, because the pipeline reports green either way.
"""

from dataclasses import dataclass

__all__ = [
    "SolGateSpec",
    "SolGateResult",
    "SolGateError",
    "parse_sol_gate",
    "resolve_theoretical_flops",
    "evaluate_sol_gates",
]


# MFU above the hardware ceiling is not a great result, it is a broken
# measurement — the FLOPs or the peak do not describe the region being measured.
_IMPLAUSIBLE_MFU_PCT = 100.0


class SolGateError(ValueError):
    """A speed-of-light gate could not be parsed or evaluated.

    Raised instead of degrading to "gate passed" so a misconfigured gate fails
    the CI job loudly rather than waving a regression through.
    """


@dataclass(frozen=True)
class SolGateSpec:
    """A single ``REGION:PCT`` gate target."""

    region: str
    threshold_pct: float

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.region}:{self.threshold_pct:g}"


@dataclass(frozen=True)
class SolGateResult:
    """Outcome of evaluating one gate target against a profile."""

    region: str
    threshold_pct: float
    mfu_pct: float
    headroom_ms: float | None
    passed: bool

    def to_dict(self) -> dict:
        d: dict = {
            "region": self.region,
            "threshold_pct": round(self.threshold_pct, 3),
            "mfu_pct": round(self.mfu_pct, 3),
            "passed": self.passed,
        }
        if self.headroom_ms is not None:
            d["headroom_ms"] = round(self.headroom_ms, 3)
        return d


def parse_sol_gate(spec: str) -> SolGateSpec:
    """Parse a ``REGION:PCT`` gate specification.

    The region may itself contain colons (kernel names often do), so the split
    is on the *last* colon and the percentage is what follows it.
    """
    raw = (spec or "").strip()
    if ":" not in raw:
        raise SolGateError(
            f"invalid --gate-sol {spec!r}: expected REGION:PCT, e.g. 'attn:60'"
        )
    region, _, pct_text = raw.rpartition(":")
    region = region.strip()
    if not region:
        raise SolGateError(f"invalid --gate-sol {spec!r}: region name is empty")
    try:
        threshold = float(pct_text)
    except ValueError:
        raise SolGateError(
            f"invalid --gate-sol {spec!r}: {pct_text!r} is not a number"
        ) from None
    if not 0 < threshold <= 100:
        raise SolGateError(
            f"invalid --gate-sol {spec!r}: threshold must be >0 and <=100, got {threshold:g}"
        )
    return SolGateSpec(region=region, threshold_pct=threshold)


def resolve_theoretical_flops(cli_value: float | None) -> float:
    """Resolve the model FLOPs the MFU computation needs.

    Resolution order, highest precedence first:

    1. the explicit ``--theoretical-flops`` value passed on the command line;
    2. *(future)* a recorded RunSpec (#25) or project settings file (#228) —
       this is the seam those should plug into, as an additional ordered layer
       rather than a special case at the call site;
    3. otherwise raise.

    There is deliberately no default. FLOPs is a property of the workload that
    cannot be recovered from a trace, and guessing it would produce a confident
    MFU number with no basis.
    """
    if cli_value is not None:
        if cli_value <= 0:
            raise SolGateError(
                f"--theoretical-flops must be positive, got {cli_value:g}"
            )
        return float(cli_value)
    raise SolGateError(
        "--gate-sol requires --theoretical-flops (model FLOPs per step); it cannot "
        "be derived from a trace. Refusing to skip the gate, because a gate that "
        "silently does not run reports green on a real regression."
    )


def evaluate_sol_gates(
    conn,
    specs: list[SolGateSpec],
    *,
    theoretical_flops: float,
    peak_tflops: float | None = None,
    source: str = "nvtx",
    device_id: int | None = None,
    occurrence_index: int = 1,
    num_gpus: int = 1,
) -> list[SolGateResult]:
    """Evaluate each gate target against ``conn``, reusing the region_mfu skill.

    Every parameter that changes *what is measured* is passed explicitly rather
    than left to a default, because each one silently moves the resulting MFU:
    ``num_gpus`` scales the peak the achieved rate is divided by,
    ``occurrence_index`` selects which instance of an NVTX region is measured
    (the first is usually a warmup/compile iteration), and ``device_id``
    restricts which GPU's kernels count. A gate whose scope is implicit is a
    gate that can flip for reasons unrelated to the code under test.

    Raises :class:`SolGateError` when a region cannot be measured, so an
    unmeasurable target fails the gate instead of passing by omission.
    """
    from .skills.builtins.region_mfu import _sol_headroom_ms
    from .skills.registry import get_skill

    skill = get_skill("region_mfu")
    if skill is None:  # pragma: no cover - registry always has the builtin
        raise SolGateError("region_mfu skill is unavailable; cannot evaluate --gate-sol")

    results: list[SolGateResult] = []
    for spec in specs:
        kwargs = {
            "name": spec.region,
            "source": source,
            "theoretical_flops": theoretical_flops,
            "occurrence_index": occurrence_index,
            "num_gpus": num_gpus,
        }
        if peak_tflops is not None:
            kwargs["peak_tflops"] = peak_tflops
        if device_id is not None:
            kwargs["device_id"] = device_id
        try:
            rows = skill.execute(conn, **kwargs)
        except Exception as exc:  # noqa: BLE001 — see below
            # Any failure to measure is a gate error, not a pass. A missing
            # table or malformed profile would otherwise surface as a raw
            # traceback and an exit code shared with "gate failed", leaving CI
            # unable to distinguish a regression from a broken setup.
            # Collapse the underlying error to one bounded line; engine errors
            # carry multi-line SQL context that would swamp a CI log.
            detail = " ".join(str(exc).split())[:200]
            raise SolGateError(
                f"--gate-sol {spec}: could not measure region {spec.region!r}: "
                f"{type(exc).__name__}: {detail}"
            ) from exc
        row = rows[0] if rows else {}
        if not row or "error" in row:
            err = (row.get("error") or {}) if row else {}
            detail = err.get("message") or "region not found or not measurable"
            raise SolGateError(
                f"--gate-sol {spec}: could not measure region {spec.region!r}: {detail}"
            )
        mfu = row.get("mfu_pct_kernel_union")
        if mfu is None:
            raise SolGateError(
                f"--gate-sol {spec}: region {spec.region!r} produced no MFU value"
            )
        mfu = float(mfu)
        if mfu > _IMPLAUSIBLE_MFU_PCT:
            # Exceeding the hardware ceiling means the inputs do not describe
            # the measured region — almost always step-scoped FLOPs applied to a
            # sub-region, or the wrong peak. Passing here would be the failure
            # mode this module exists to prevent: a green gate that never really
            # ran. (Reported as a gate failure rather than raising, because an
            # FP8/sparse run can legitimately exceed an auto-detected BF16-dense
            # peak, and that is a threshold problem the user must resolve.)
            raise SolGateError(
                f"--gate-sol {spec}: implausible MFU {mfu:.1f}% for region "
                f"{spec.region!r} — above the hardware ceiling. Check that "
                "--theoretical-flops describes this region (not the whole step) "
                "and that --peak-tflops matches the precision actually used."
            )
        results.append(
            SolGateResult(
                region=spec.region,
                threshold_pct=spec.threshold_pct,
                mfu_pct=mfu,
                headroom_ms=_sol_headroom_ms(row),
                passed=mfu >= spec.threshold_pct,
            )
        )
    return results
