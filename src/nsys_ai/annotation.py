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

``Finding`` carries optional v0.1 fields (id, category, confidence,
evidence rows, selection, diff lineage, etc.) that the new surfaces
populate. Existing producers/consumers that ignore the new fields keep
working unchanged.

``EvidenceReport`` JSON output carries an additive envelope
(``schema_version``, ``producer``, ``producer_version``) so downstream
tools can detect format compatibility.
"""

import json
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
    # removed — the optimization *opportunity*. Lets consumers rank by upside
    # rather than only severity: a small idle gap with large headroom should
    # outrank a dramatic-looking finding with little room to improve. Optional;
    # drops from to_dict when None, and ranking is a no-op when no finding
    # carries one (see :func:`rank_findings`).
    headroom_ms: float | None = None

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


def rank_findings(findings: list["Finding"]) -> list["Finding"]:
    """Order findings by optimization opportunity (largest headroom first).

    Findings carrying a ``headroom_ms`` sort ahead of those without, largest
    headroom first, so the biggest *recoverable* win surfaces first regardless
    of how severe a finding merely looks. Findings without a headroom keep
    their original relative order and follow the headroom-bearing ones.

    The ranking is a deliberate no-op when **no** finding carries a headroom:
    the input order is returned unchanged, so behaviour on legacy findings is
    identical to before. The sort is stable.
    """
    if not any(f.headroom_ms is not None for f in findings):
        return list(findings)

    def _key(item: tuple[int, "Finding"]) -> tuple:
        idx, f = item
        has_headroom = f.headroom_ms is not None
        return (0 if has_headroom else 1, -(f.headroom_ms or 0.0), idx)

    return [f for _, f in sorted(enumerate(findings), key=_key)]


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
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceReport":
        # Envelope fields (schema_version / producer / producer_version)
        # are informational only — readers ignore them. Pre-profile_id
        # payloads load with an empty profile_id (additive, not breaking).
        findings = [Finding.from_dict(f) for f in d.get("findings", [])]
        return cls(
            title=d.get("title", "Untitled"),
            profile_id=d.get("profile_id", ""),
            profile_path=d.get("profile_path", ""),
            findings=findings,
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
