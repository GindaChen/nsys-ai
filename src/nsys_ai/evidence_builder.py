"""
evidence_builder.py — Convert profile analysis into visual Finding overlays.

Each method queries individual kernel instances (not aggregates)
to produce findings with exact nanosecond timestamps for timeline overlay.
"""

import inspect
import logging
import os
from collections.abc import Callable

from .annotation import EvidenceReport, Finding, SkippedAnalysis, rank_findings
from .profile import Profile
from .skill_packs import EVIDENCE_OVERLAY

_log = logging.getLogger(__name__)


def _invoke_to_findings(fn: Callable, rows: list[dict], context: dict) -> list[Finding]:
    """Call ``Skill.to_findings_fn`` with optional v0.1 context.

    Skills upgraded to the v0.1 schema declare a ``context`` keyword
    parameter (or accept ``**kwargs``) to receive the profile-level
    metadata (``profile_id``, etc.) needed to construct
    ``TraceSelection`` / ``EvidenceRow`` objects.

    Legacy skills with the single-argument signature ``(rows)`` are
    invoked unchanged for backward compatibility.

    Abstention rows never reach ``fn``. A skill that could not run has nothing
    to turn into a finding, and filtering here rather than in each
    ``to_findings_fn`` makes that true by construction: the skills that are
    safe today are safe only through unrelated guards (an early return on a
    row count, a length check), which a refactor could remove without anyone
    noticing.
    """
    from .skills.base import is_abstention

    if is_abstention(rows):
        return []

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
    _SKILL_PIPELINE = EVIDENCE_OVERLAY

    def build(self, only: list[str] | None = None) -> EvidenceReport:
        """Run analyzers (via skill pipeline) and return a combined EvidenceReport.

        Args:
            only: If provided, run only the named analyzers.
                  Valid names are the keys in :attr:`_SKILL_PIPELINE`.
                  If None, run all analyzers.
        """
        from .fingerprint import get_profile_id
        from .skills.base import is_abstention
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
        # Analyses that abstained, in pipeline order. An abstention makes no
        # finding by design, so without this the report is indistinguishable
        # from one where every skill ran and the profile came out clean.
        skipped: list[SkippedAnalysis] = []
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
                conn = self.prof.query_conn()
                rows = skill.execute(conn, **kwargs)
                if is_abstention(rows):
                    # One entry per analyzer, not per skill: two pipeline
                    # entries can share a skill (kernel_instances runs as both
                    # nccl_stalls and kernel_hotspots), and each of them is a
                    # separate piece of coverage the report lost. The pipeline
                    # is a dict, so the analyzer name cannot repeat.
                    #
                    # ``reason`` should always be present — ``abstain`` sets
                    # it — but a hand-rolled abstention row could omit it, and
                    # an empty reason renders as a dangling "skipped:" line.
                    reason = str(rows[0].get("reason") or "could not run")
                    skipped.append(
                        SkippedAnalysis(analyzer=analyzer_name, skill=skill_name, reason=reason)
                    )
                    continue
                if skill.to_findings_fn:
                    findings.extend(_invoke_to_findings(skill.to_findings_fn, rows, context))
            except Exception as e:
                _log.error(
                    "Analyzer %s (skill %s) failed: %s", analyzer_name, skill_name, e, exc_info=True
                )

        # Rank by optimization opportunity (headroom) so the biggest
        # recoverable win surfaces first. No-op when no skill emits a headroom.
        return EvidenceReport(
            title="Auto-Analysis",
            profile_id=profile_id,
            profile_path=profile_path,
            findings=rank_findings(findings),
            skipped=skipped,
        )
