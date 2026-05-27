"""
evidence_builder.py — Convert profile analysis into visual Finding overlays.

Each method queries individual kernel instances (not aggregates)
to produce findings with exact nanosecond timestamps for timeline overlay.
"""

import inspect
import logging
import os
from collections.abc import Callable

from .annotation import EvidenceReport, Finding
from .profile import Profile

_log = logging.getLogger(__name__)


def _invoke_to_findings(fn: Callable, rows: list[dict], context: dict) -> list[Finding]:
    """Call ``Skill.to_findings_fn`` with optional v0.1 context.

    Skills upgraded to the v0.1 schema declare a ``context`` keyword
    parameter (or accept ``**kwargs``) to receive the profile-level
    metadata (``profile_id``, etc.) needed to construct
    ``TraceSelection`` / ``EvidenceRow`` objects.

    Legacy skills with the single-argument signature ``(rows)`` are
    invoked unchanged for backward compatibility.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        # Builtins / C extensions may not expose a signature; fall back
        # to the legacy single-argument calling convention.
        return fn(rows)

    accepts_context = "context" in sig.parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_context:
        return fn(rows, context=context)
    return fn(rows)


class EvidenceBuilder:
    """Generates findings from a profile using direct SQL queries.

    Usage::

        with Profile("profile.sqlite") as prof:
            builder = EvidenceBuilder(prof, device=0)
            report = builder.build()
            # report.findings is a list of Finding objects
    """

    def __init__(
        self,
        prof: Profile,
        device: int = 0,
        trim: tuple[int, int] | None = None,
    ):
        self.prof = prof
        self.device = device
        self.trim = trim or tuple(prof.meta.time_range)

    # Map analyzer_name -> (skill_name, params)
    _SKILL_PIPELINE = {
        "slow_iterations": ("iteration_timing", {}),
        "idle_gaps": ("gpu_idle_gaps", {"limit": 5, "min_gap_ns": 1000000}),
        "nccl_stalls": ("kernel_instances", {"name": "nccl", "limit": 3}),
        "kernel_hotspots": ("kernel_instances", {"limit": 3}),
        "overlap_ratio": ("overlap_breakdown", {}),
        "memory_anomalies": ("memory_bandwidth", {"limit": 5}),
        "h2d_spikes": ("h2d_distribution", {}),
        "nccl_breakdown": ("nccl_breakdown", {}),
        # Roll-up characterization of the whole profile (comm-bound,
        # sync-bound, idle-dominant, coverage gaps). Reads only the
        # already-assembled manifest dict, so its findings are
        # independent of the other pipeline entries' to_findings status.
        "profile_health": ("profile_health_manifest", {}),
    }

    def build(self, only: list[str] | None = None) -> EvidenceReport:
        """Run analyzers (via skill pipeline) and return a combined EvidenceReport.

        Args:
            only: If provided, run only the named analyzers.
                  Valid names: slow_iterations, idle_gaps, nccl_stalls,
                  kernel_hotspots, overlap_ratio, memory_anomalies, h2d_spikes.
                  If None, run all analyzers.
        """
        from .fingerprint import get_profile_id
        from .skills.registry import get_skill

        # Coerce to str up-front: ``Profile.path`` is whatever the caller
        # passed (often ``pathlib.Path`` in tests). Both ``get_profile_id``
        # and downstream JSON serialisation need a real ``str``.
        raw_path = getattr(self.prof, "path", None)
        profile_path: str = os.fspath(raw_path) if raw_path is not None else ""
        # ``profile_id`` is a content-derived stable hash (see
        # ``fingerprint.get_profile_id``). It uses ``self.prof.conn``
        # because the META_DATA / TARGET_INFO tables it reads are *not*
        # part of the parquet cache — only the original SQLite (or a
        # direct-attach DuckDB view) carries them. Falls back to a
        # path-derived id when those tables are unreachable
        # (e.g. backend='parquetdir').
        profile_id = get_profile_id(getattr(self.prof, "conn", None), fallback_path=profile_path)

        findings: list[Finding] = []
        # v0.1 context handed to upgraded skills' to_findings_fn for
        # constructing TraceSelection / EvidenceRow with provenance.
        context: dict = {"profile_id": profile_id}
        for analyzer_name, (skill_name, params) in self._SKILL_PIPELINE.items():
            if only is not None and analyzer_name not in only:
                continue

            try:
                skill = get_skill(skill_name)
                if skill is None:
                    _log.debug(
                        "Analyzer %s skipped (skill %s not found)", analyzer_name, skill_name
                    )
                    continue

                # Map runtime parameters into skill args
                kwargs = {**params, "device": self.device}
                if self.trim:
                    kwargs["trim_start_ns"] = self.trim[0]
                    kwargs["trim_end_ns"] = self.trim[1]

                # Use DuckDB if available, fallback to SQLite
                conn = self.prof.db if self.prof.db is not None else self.prof.conn
                rows = skill.execute(conn, **kwargs)
                if skill.to_findings_fn:
                    findings.extend(_invoke_to_findings(skill.to_findings_fn, rows, context))
            except Exception as e:
                _log.error(
                    "Analyzer %s (skill %s) failed: %s", analyzer_name, skill_name, e, exc_info=True
                )

        return EvidenceReport(
            title="Auto-Analysis",
            profile_id=profile_id,
            profile_path=profile_path,
            findings=findings,
        )
