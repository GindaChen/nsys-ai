"""
annotation.py — Evidence annotation schema for AI agent findings.

Agents produce findings (bottleneck highlights, time-range markers, etc.)
that overlay onto the timeline viewer for human verification.
"""

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Finding:
    """A single agent-authored finding to overlay on the timeline."""

    type: str  # "highlight" | "region" | "marker"
    label: str
    start_ns: int
    end_ns: int | None = None  # None for marker type
    stream: str | None = None  # target stream ID (for highlight)
    gpu_id: int | None = None
    color: str = "rgba(255,68,68,0.3)"
    severity: str = "info"  # "critical" | "warning" | "info"
    note: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass
class EvidenceReport:
    """A collection of findings for a profile, produced by an AI agent."""

    title: str
    profile_path: str = ""
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "profile_path": self.profile_path,
            "findings": [f.to_dict() for f in self.findings],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceReport":
        findings = [Finding.from_dict(f) for f in d.get("findings", [])]
        return cls(
            title=d.get("title", "Untitled"),
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
