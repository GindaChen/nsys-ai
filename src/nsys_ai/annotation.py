"""
annotation.py — Evidence annotation schema.

Agents produce findings (bottleneck highlights, time-range markers, etc.)
that overlay onto the timeline viewer for human verification.

This module also defines the v0.1 evidence schema models that downstream
surfaces (CLI, GUI, agent, diff) share:

    EvidenceRow      — one row of evidence backing a Finding
    TraceSelection   — a region in a profile (time, GPU, rank, stream, NVTX)
    DiffLineage      — links a Finding to the diff that surfaced it
    Diagnostic       — an agent's summarized diagnosis with verification command
    SkippedAnalysis  — an analysis that could not run on this profile, and why

``Finding`` carries optional v0.1 fields (id, category, confidence,
evidence rows, selection, diff lineage, etc.) that the new surfaces
populate. Existing producers/consumers that ignore the new fields keep
working unchanged.

``EvidenceReport`` JSON output carries an additive envelope
(``schema_version``, ``producer``, ``producer_version``) so downstream
tools can detect format compatibility.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from functools import cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any, Literal

# Field names on Finding that hold nested dataclass instances; serialized
# separately via each nested type's to_dict() to avoid a wasted deep copy
# through asdict() (which would materialize them only to be discarded).
_FINDING_NESTED_FIELDS = frozenset({"evidence", "selection", "diff_lineage"})

#: Current evidence-artifact schema version.
#:
#: Bumped on breaking changes to the JSON envelope or required fields.
#: Backward-compatible additions (new optional fields) do not bump this.
SCHEMA_VERSION = "0.1"

#: Producer identifier embedded in evidence-artifact JSON envelopes.
PRODUCER = "nsys-ai"


@cache
def _producer_version() -> str:
    """Return the installed nsys-ai package version, or a dev marker.

    Reads the distribution metadata directly via ``importlib.metadata`` so
    ``EvidenceReport.to_dict`` stays self-contained — it does not pull in
    ``nsys_ai/__init__.py`` (which would eagerly import the rest of the
    package and reintroduce circular-import risk).

    Cached for the lifetime of the process: the installed package version
    is constant per interpreter, and ``importlib.metadata.version`` reads
    distribution metadata from disk on every call without caching.
    """
    try:
        return _pkg_version("nsys-ai")
    except PackageNotFoundError:
        return "0.0.0+dev"


@dataclass
class Finding:
    """A single agent-authored finding to overlay on the timeline.

    Required fields (``type``, ``label``, ``start_ns``) define a minimum
    visual overlay. v0.1 optional fields enrich the finding with structured
    evidence, category, confidence, source location, and diff provenance.

    All v0.1 fields default to ``None`` and are dropped from
    :meth:`to_dict` when unset, so legacy JSON output remains compact.
    """

    type: str  # "highlight" | "region" | "marker"
    label: str
    start_ns: int
    end_ns: int | None = None  # None for marker type
    stream: str | None = None  # target stream ID (for highlight)
    gpu_id: int | None = None
    color: str = "rgba(255,68,68,0.3)"
    severity: str = "info"  # "critical" | "warning" | "info"
    note: str = ""

    # v0.1 additive fields — all optional, all drop from to_dict when None.
    id: str | None = None
    category: "FindingCategory | None" = None
    confidence: float | None = None
    evidence: list["EvidenceRow"] | None = None
    selection: "TraceSelection | None" = None
    explanation: str | None = None
    suggested_actions: list[str] | None = None
    false_positive_notes: list[str] | None = None
    provenance: dict[str, Any] | None = None
    diff_lineage: "DiffLineage | None" = None
    # Potential recoverable time (ms) if this finding's inefficiency were
    # removed — the optimization *opportunity*, used by :func:`rank_findings`
    # to order findings by upside rather than severity alone.
    headroom_ms: float | None = None
    # What span ``headroom_ms`` covers. Ranking compares the raw magnitudes, so
    # producers must agree on the span or the ordering is meaningless — a
    # per-instance value loses to a capture-wide one for reasons that have
    # nothing to do with opportunity. Every producer currently emits
    # ``"capture_total"``; the field exists so a new one has to state its basis
    # rather than diverge silently.
    headroom_basis: str | None = None

    def to_dict(self) -> dict:
        # Walk fields() directly for scalar / primitive fields; nested
        # dataclass fields are serialized via their own to_dict() to
        # preserve each nested type's None-drop convention. Avoids the
        # recursive deep copy that asdict() would perform on the nested
        # fields only to be discarded here.
        d: dict = {}
        for f in fields(self):
            if f.name in _FINDING_NESTED_FIELDS:
                continue
            v = getattr(self, f.name)
            if v is None:
                continue
            # Shallow defensive copy for mutable container fields:
            # top-level mutation of the returned dict (e.g.
            # ``d["suggested_actions"].append(...)``) does not affect
            # the source. Nested mutable values inside ``provenance`` /
            # ``values`` etc. are still shared by reference — deep
            # copies are intentionally avoided to keep ``to_dict``
            # cheap, since the output is normally consumed by JSON
            # serialization rather than mutated.
            if isinstance(v, list):
                d[f.name] = list(v)
            elif isinstance(v, dict):
                d[f.name] = dict(v)
            else:
                d[f.name] = v
        if self.evidence is not None:
            d["evidence"] = [e.to_dict() for e in self.evidence]
        if self.selection is not None:
            d["selection"] = self.selection.to_dict()
        if self.diff_lineage is not None:
            d["diff_lineage"] = self.diff_lineage.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        # Rehydrate nested dataclass fields when present.
        if filtered.get("evidence") is not None:
            filtered["evidence"] = [EvidenceRow.from_dict(e) for e in filtered["evidence"]]
        if filtered.get("selection") is not None:
            filtered["selection"] = TraceSelection.from_dict(filtered["selection"])
        if filtered.get("diff_lineage") is not None:
            filtered["diff_lineage"] = DiffLineage.from_dict(filtered["diff_lineage"])
        return cls(**filtered)


def headroom_sort_prefix(headroom_ms: float | None) -> tuple[int, float]:
    """Sort-key prefix that orders by optimization opportunity.

    Items with a numeric ``headroom_ms`` come first, largest headroom first;
    ``None`` (or any non-numeric value that survived deserialization) sorts
    after. Shared by :func:`rank_findings` (Finding objects) and the guided
    loop's dict-based ranking so both stay consistent.
    """
    hv = headroom_ms if isinstance(headroom_ms, (int, float)) else None
    return (0 if hv is not None else 1, -(hv or 0.0))


def rank_findings(findings: list["Finding"]) -> list["Finding"]:
    """Order findings by optimization opportunity (largest headroom first).

    Findings carrying a numeric ``headroom_ms`` sort ahead of those without,
    largest first, so the biggest *recoverable* win surfaces first regardless
    of how severe a finding merely looks. The sort is stable, so findings
    without a headroom keep their original relative order and — when **no**
    finding carries one — the input order is returned unchanged.
    """
    return sorted(findings, key=lambda f: headroom_sort_prefix(f.headroom_ms))


@dataclass
class SkippedAnalysis:
    """One analysis that could not run on this profile, and why.

    A ``Finding`` asserts something about the profile's *performance*; this
    asserts something about the *analysis's coverage*. They are kept apart
    deliberately — ranking a "could not run" among headroom-bearing findings
    would put bookkeeping in the middle of a priority list, which is the same
    reason ``EvidenceBuilder`` never hands an abstention row to a
    ``to_findings_fn``.

    Two names, because they are two different handles and a reader needs the
    one that matches the surface they are on:

    * ``analyzer`` — the key ``EvidenceBuilder._SKILL_PIPELINE`` runs it under,
      which is what ``evidence build --analyzers`` accepts. This is the name a
      user can act on.
    * ``skill`` — the skill that actually abstained, which is what
      ``skill run`` accepts. Several analyzers can share one skill, so this
      alone does not say which coverage was lost.

    ``reason`` is the text the skill passed to ``skills.base.abstain``, unless
    the abstention row carried none — see ``EvidenceBuilder.build``, which
    substitutes a short placeholder rather than emitting an empty string.
    """

    analyzer: str
    skill: str
    reason: str

    def to_dict(self) -> dict:
        return {"analyzer": self.analyzer, "skill": self.skill, "reason": self.reason}

    @classmethod
    def from_dict(cls, d: dict) -> "SkippedAnalysis":
        return cls(
            analyzer=str(d.get("analyzer", "")),
            skill=str(d.get("skill", "")),
            reason=str(d.get("reason", "")),
        )


@dataclass
class EvidenceReport:
    """A collection of findings for a profile, produced by an AI agent.

    The :meth:`to_dict` output carries the v0.1 envelope
    (``schema_version``, ``producer``, ``producer_version``). The
    :meth:`from_dict` reader accepts both v0.1 envelopes and legacy
    (envelope-free) JSON payloads.

    .. note::
       New envelope fields must be added as ``field(..., kw_only=True)``
       (see ``profile_id`` below). Inserting a non-kw-only field before
       the existing positional ones would silently shift the positional
       signature and rebind old callers.
    """

    title: str
    # ``profile_id`` is keyword-only so adding it after the original
    # ``title`` / ``profile_path`` fields does not shift the positional
    # signature — pre-v0.1 callers using ``EvidenceReport("T", "/p")``
    # still get ``profile_path="/p"``, not ``profile_id="/p"``.
    profile_path: str = ""
    findings: list[Finding] = field(default_factory=list)
    profile_id: str = field(default="", kw_only=True)
    # Analyses that could not run, beside the findings rather than among
    # them. Keyword-only by the class convention noted above, which is a
    # convention and not a fix for a live hazard: appending a positional
    # field here could not rebind an existing caller, because a fourth
    # positional argument to ``EvidenceReport`` is a TypeError today. The
    # convention exists so the *next* field is not inserted before
    # ``findings``, where it would rebind.
    skipped: list[SkippedAnalysis] = field(default_factory=list, kw_only=True)

    def __post_init__(self) -> None:
        # Callers occasionally hand in ``pathlib.Path`` even though the
        # field is typed ``str``. Coerce now so ``to_dict()`` /
        # ``save_findings`` downstream can JSON-dump without
        # ``TypeError: Object of type PosixPath is not JSON serializable``.
        if self.profile_path:
            import os

            self.profile_path = os.fspath(self.profile_path)

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "producer": PRODUCER,
            "producer_version": _producer_version(),
            "title": self.title,
            "profile_id": self.profile_id,
            "profile_path": self.profile_path,
            "findings": [f.to_dict() for f in self.findings],
            # Always emitted, empty list included: a consumer can then tell
            # "nothing was skipped" from "this producer predates the field"
            # by the key's presence, instead of guessing from its absence.
            "skipped": [s.to_dict() for s in self.skipped],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceReport":
        # Envelope fields (schema_version / producer / producer_version)
        # are informational only — readers ignore them. Pre-profile_id
        # payloads load with an empty profile_id (additive, not breaking);
        # pre-``skipped`` payloads load with an empty skipped list.
        findings = [Finding.from_dict(f) for f in d.get("findings", [])]
        skipped = [SkippedAnalysis.from_dict(s) for s in d.get("skipped") or []]
        return cls(
            title=d.get("title", "Untitled"),
            profile_id=d.get("profile_id", ""),
            profile_path=d.get("profile_path", ""),
            findings=findings,
            skipped=skipped,
        )


def load_findings(path: str) -> EvidenceReport:
    """Load an evidence report from a JSON file."""
    with open(path) as f:
        return EvidenceReport.from_dict(json.load(f))


def save_findings(report: EvidenceReport, path: str) -> None:
    """Save an evidence report to a JSON file."""
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)


# ──────────────────────────────────────────────────────────────────────
# v0.1 evidence schema models
# ──────────────────────────────────────────────────────────────────────

# FindingCategory: step-time category for findings.
# The first four values map to the step-time decomposition
# ``Step Time = Compute + Communication + Launch/Overhead + Idle``.
# The remaining values are orthogonal tags, not step-time buckets.
FindingCategory = Literal[
    "compute",
    "communication",
    "launch_overhead",
    "idle",
    "memory",
    "sync",
    "nvtx",
    "profile_quality",
    "kernel_internal",
    "framework",
]


@dataclass
class EvidenceRow:
    """One row of evidence backing a Finding.

    A skill emits zero or more ``EvidenceRow`` instances; an evidence-citing
    ``Finding`` references them either by id (via ``selection_id``-style
    pointers) or by embedding.
    """

    id: str
    source_skill: str
    values: dict[str, Any] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    selection_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceRow":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        # Normalize JSON null → {} for dict-typed fields so the dict invariant
        # holds even when callers serialize an explicit ``null`` value.
        for key in ("values", "units", "provenance"):
            if filtered.get(key) is None and key in filtered:
                filtered[key] = {}
        return cls(**filtered)


@dataclass
class TraceSelection:
    """A region in a profile.

    ``profile_id`` is the canonical fingerprint of the source profile
    (see ``nsys_ai.fingerprint.get_fingerprint``); two surfaces looking at
    the same ``.sqlite`` will agree on this id without depending on the
    filesystem path.

    All location fields are optional. A selection may be time-only,
    GPU-only, NVTX-only, or any combination.

    ``source`` records who produced the selection, using the convention
    ``"skill:<name>"`` | ``"gui"`` | ``"user"`` | ``"diff"``.
    """

    id: str
    profile_id: str
    source: str
    start_ns: int | None = None
    end_ns: int | None = None
    gpu_ids: list[int] | None = None
    rank_ids: list[int] | None = None
    stream_ids: list[int] | None = None
    nvtx_path: list[str] | None = None
    event_ids: list[str] | None = None
    label: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "TraceSelection":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass
class DiffLineage:
    """Links a Finding to the diff that surfaced it.

    Lets a Finding inside an ``after.sqlite`` profile carry "I am regression
    #2 of the YYYY-MM-DD diff against baseline:v1.0". Agent and GUI use
    this for provenance and narration.
    """

    diff_id: str
    role: Literal["regression", "improvement", "stable"]
    rank: int  # 0-indexed position in top_regressions / top_improvements
    baseline_profile_id: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DiffLineage":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


_FINDING_FIELDS = {item.name for item in fields(Finding)}
_FINDING_REQUIRED_FIELDS = {"type", "label", "start_ns", "color", "severity", "note"}
_SELECTION_FIELDS = {item.name for item in fields(TraceSelection)}
_SELECTION_REQUIRED_FIELDS = {"id", "profile_id", "source"}
_EVIDENCE_ROW_FIELDS = {item.name for item in fields(EvidenceRow)}
_EVIDENCE_ROW_REQUIRED_FIELDS = {"id", "source_skill", "values", "units", "provenance"}
_DIFF_LINEAGE_FIELDS = {item.name for item in fields(DiffLineage)}
_REPORT_FIELDS = {
    "schema_version",
    "producer",
    "producer_version",
    "title",
    "profile_id",
    "profile_path",
    "findings",
    "skipped",
}


def _artifact_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _artifact_exact_keys(
    value: Mapping[str, Any], allowed: set[str], required: set[str], label: str
) -> None:
    actual = set(value)
    missing = required - actual
    unknown = actual - allowed
    if missing or unknown:
        raise ValueError(
            f"{label} fields do not match schema; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _artifact_string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _artifact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _artifact_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label} must be at most {maximum}")
    return number


def _artifact_string_list(value: Any, label: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an array of strings")


def validate_trace_selection_payload(
    payload: Any, *, label: str = "selection"
) -> TraceSelection:
    value = _artifact_object(payload, label)
    _artifact_exact_keys(value, _SELECTION_FIELDS, _SELECTION_REQUIRED_FIELDS, label)
    _artifact_string(value["id"], f"{label}.id")
    _artifact_string(value["profile_id"], f"{label}.profile_id", allow_empty=True)
    _artifact_string(value["source"], f"{label}.source")
    for field_name in ("start_ns", "end_ns"):
        if field_name in value:
            _artifact_int(value[field_name], f"{label}.{field_name}", minimum=0)
    if (
        "start_ns" in value
        and "end_ns" in value
        and value["end_ns"] < value["start_ns"]
    ):
        raise ValueError(f"{label}.end_ns must not precede start_ns")
    for field_name in ("gpu_ids", "rank_ids", "stream_ids"):
        if field_name not in value:
            continue
        numbers = value[field_name]
        if not isinstance(numbers, list):
            raise ValueError(f"{label}.{field_name} must be an array")
        for index, item in enumerate(numbers):
            _artifact_int(item, f"{label}.{field_name}[{index}]", minimum=0)
    for field_name in ("nvtx_path", "event_ids"):
        if field_name in value:
            _artifact_string_list(value[field_name], f"{label}.{field_name}")
    if "label" in value:
        _artifact_string(value["label"], f"{label}.label", allow_empty=True)
    return TraceSelection.from_dict(dict(value))


def validate_evidence_row_payload(
    payload: Any, *, label: str = "evidence row"
) -> EvidenceRow:
    value = _artifact_object(payload, label)
    _artifact_exact_keys(
        value, _EVIDENCE_ROW_FIELDS, _EVIDENCE_ROW_REQUIRED_FIELDS, label
    )
    _artifact_string(value["id"], f"{label}.id")
    _artifact_string(value["source_skill"], f"{label}.source_skill")
    for field_name in ("values", "units", "provenance"):
        _artifact_object(value[field_name], f"{label}.{field_name}")
    if any(
        not isinstance(key, str) or not isinstance(unit, str)
        for key, unit in value["units"].items()
    ):
        raise ValueError(f"{label}.units must map strings to strings")
    if "selection_id" in value:
        _artifact_string(value["selection_id"], f"{label}.selection_id")
    return EvidenceRow.from_dict(dict(value))


def validate_finding_payload(payload: Any, *, label: str = "finding") -> Finding:
    value = _artifact_object(payload, label)
    _artifact_exact_keys(value, _FINDING_FIELDS, _FINDING_REQUIRED_FIELDS, label)
    if value["type"] not in {"highlight", "region", "marker"}:
        raise ValueError(f"{label}.type must be highlight, region, or marker")
    _artifact_string(value["label"], f"{label}.label", allow_empty=True)
    _artifact_int(value["start_ns"], f"{label}.start_ns", minimum=0)
    if "end_ns" in value:
        _artifact_int(value["end_ns"], f"{label}.end_ns", minimum=0)
        if value["end_ns"] < value["start_ns"]:
            raise ValueError(f"{label}.end_ns must not precede start_ns")
    for field_name in ("stream", "color", "note"):
        if field_name in value:
            _artifact_string(value[field_name], f"{label}.{field_name}", allow_empty=True)
    if "gpu_id" in value:
        _artifact_int(value["gpu_id"], f"{label}.gpu_id", minimum=0)
    if value["severity"] not in {"critical", "warning", "info"}:
        raise ValueError(f"{label}.severity is invalid")
    if "id" in value:
        _artifact_string(value["id"], f"{label}.id", allow_empty=True)
    if "category" in value and value["category"] not in {
        "compute",
        "communication",
        "launch_overhead",
        "idle",
        "memory",
        "sync",
        "nvtx",
        "profile_quality",
        "kernel_internal",
        "framework",
    }:
        raise ValueError(f"{label}.category is invalid")
    if "confidence" in value:
        _artifact_number(value["confidence"], f"{label}.confidence", minimum=0, maximum=1)
    if "evidence" in value:
        rows = value["evidence"]
        if not isinstance(rows, list):
            raise ValueError(f"{label}.evidence must be an array")
        for index, row in enumerate(rows):
            validate_evidence_row_payload(row, label=f"{label}.evidence[{index}]")
    if "selection" in value:
        validate_trace_selection_payload(value["selection"], label=f"{label}.selection")
    for field_name in ("explanation", "headroom_basis"):
        if field_name in value:
            _artifact_string(value[field_name], f"{label}.{field_name}", allow_empty=True)
    for field_name in ("suggested_actions", "false_positive_notes"):
        if field_name in value:
            _artifact_string_list(value[field_name], f"{label}.{field_name}")
    if "provenance" in value:
        _artifact_object(value["provenance"], f"{label}.provenance")
    if "diff_lineage" in value:
        lineage = _artifact_object(value["diff_lineage"], f"{label}.diff_lineage")
        _artifact_exact_keys(
            lineage,
            _DIFF_LINEAGE_FIELDS,
            _DIFF_LINEAGE_FIELDS,
            f"{label}.diff_lineage",
        )
        for field_name in ("diff_id", "baseline_profile_id"):
            _artifact_string(
                lineage[field_name], f"{label}.diff_lineage.{field_name}"
            )
        if lineage["role"] not in {"regression", "improvement", "stable"}:
            raise ValueError(f"{label}.diff_lineage.role is invalid")
        _artifact_int(lineage["rank"], f"{label}.diff_lineage.rank", minimum=0)
    if "headroom_ms" in value:
        _artifact_number(value["headroom_ms"], f"{label}.headroom_ms", minimum=0)
    return Finding.from_dict(dict(value))


def validate_evidence_report_payload(payload: Any) -> EvidenceReport:
    """Validate and rehydrate the current evidence artifact contract."""
    value = _artifact_object(payload, "evidence report")
    _artifact_exact_keys(value, _REPORT_FIELDS, _REPORT_FIELDS, "evidence report")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"evidence report schema_version must be {SCHEMA_VERSION!r}")
    if value["producer"] != PRODUCER:
        raise ValueError(f"evidence report producer must be {PRODUCER!r}")
    _artifact_string(value["producer_version"], "evidence report.producer_version")
    for field_name in ("title", "profile_id", "profile_path"):
        _artifact_string(
            value[field_name], f"evidence report.{field_name}", allow_empty=True
        )
    findings = value["findings"]
    if not isinstance(findings, list):
        raise ValueError("evidence report.findings must be an array")
    for index, finding in enumerate(findings):
        validate_finding_payload(finding, label=f"findings[{index}]")
    skipped = value["skipped"]
    if not isinstance(skipped, list):
        raise ValueError("evidence report.skipped must be an array")
    for index, item in enumerate(skipped):
        entry = _artifact_object(item, f"skipped[{index}]")
        _artifact_exact_keys(
            entry, {"analyzer", "skill", "reason"}, {"analyzer", "skill", "reason"}, f"skipped[{index}]"
        )
        for field_name in ("analyzer", "skill", "reason"):
            _artifact_string(entry[field_name], f"skipped[{index}].{field_name}")

    report = EvidenceReport.from_dict(dict(value))
    regenerated = report.to_dict()
    regenerated["producer_version"] = value["producer_version"]
    if regenerated != dict(value):
        raise ValueError("evidence report changes when rehydrated")
    return report


@dataclass
class Diagnostic:
    """An agent-authored diagnosis with a runnable verification command.

    ``verification_command`` is the runnable ``nsys-ai`` command the user
    should run to confirm whether the proposed fix works. Narration is
    not verification; if no runnable command can be constructed the agent
    should say so explicitly rather than emit prose here.
    """

    id: str
    summary: str
    recommendation: str
    verification_command: str
    confidence: float
    primary_findings: list[Finding] = field(default_factory=list)
    root_cause_hypotheses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "summary": self.summary,
            "recommendation": self.recommendation,
            "verification_command": self.verification_command,
            "confidence": self.confidence,
            "primary_findings": [f.to_dict() for f in self.primary_findings],
            "root_cause_hypotheses": list(self.root_cause_hypotheses),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Diagnostic":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        # Normalize JSON null / missing → empty list for the nested-Finding path.
        if "primary_findings" in filtered:
            raw = filtered["primary_findings"] or []
            filtered["primary_findings"] = [Finding.from_dict(f) for f in raw]
        if "root_cause_hypotheses" in filtered and filtered["root_cause_hypotheses"] is None:
            filtered["root_cause_hypotheses"] = []
        return cls(**filtered)
