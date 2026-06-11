"""loop_state.py — Shared guided-loop state for Direction 2 workflows.

This module orchestrates the five-step workflow:
diagnose -> propose -> reprofile -> diff -> accept.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Phase = Literal["diagnose", "propose", "reprofile", "diff", "accept"]
Decision = Literal["accept", "reject"]
Scope = Literal["global", "gpu", "iteration", "region"]

PHASES: tuple[Phase, ...] = ("diagnose", "propose", "reprofile", "diff", "accept")
SEVERITY_WEIGHT = {"critical": 4, "warning": 3, "info": 2}


def _phase_index(phase: str) -> int:
    try:
        return PHASES.index(phase)  # type: ignore[arg-type]
    except ValueError:
        return -1


def _normalize_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank findings so workflow surfaces high-signal rows first."""
    ranked: list[tuple[int, dict[str, Any]]] = []
    for idx, f in enumerate(findings):
        severity = str(f.get("severity") or "").lower()
        kind = str(f.get("type") or "").lower()
        score = SEVERITY_WEIGHT.get(severity, 1) * 1000
        if "overlap" in kind or "nccl" in kind:
            score += 250
        if "idle" in kind:
            score += 100
        confidence = f.get("confidence")
        if isinstance(confidence, (int, float)):
            score += int(confidence * 100)
        ranked.append((score - idx, f))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in ranked]


def normalize_profile_path(path: str, *, label: str = "profile") -> str:
    """Expand and validate a profile path for loop/diff operations.

    Symlinks are not followed so Hugging Face snapshot paths keep human-readable
    names (e.g. ``profiles/perf_h100_sp1.sqlite``) instead of blob hashes.
    """
    raw = (path or "").strip()
    if not raw:
        raise ValueError(f"{label} path is required")
    p = Path(raw).expanduser()
    if p.is_symlink():
        resolved = str(p if p.is_absolute() else p.absolute())
    else:
        resolved = str(p.resolve(strict=False))
    candidate = Path(resolved)
    if not candidate.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    if not candidate.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    return resolved


def same_profile_path(left: str, right: str) -> bool:
    """True when two paths refer to the same on-disk profile file."""
    if not left or not right:
        return False
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except OSError:
        return left == right


H100_PRESET_DATASET = "rich7421/fastvideo-wan-h100-sp1-nsys"
H100_PRESET_CACHE = (
    "~/.cache/huggingface/hub/datasets--rich7421--fastvideo-wan-h100-sp1-nsys/snapshots"
)
H100_BEFORE_FILE = "perf_h100_sp1.sqlite"
H100_AFTER_FILE = "perf_h100_sp1_fa3.sqlite"
_BLOB_STEM = re.compile(r"^[a-f0-9]{32,64}$", re.IGNORECASE)


def h100_preset_download_hint() -> str:
    """CLI help text when --h100-preset profiles are missing locally."""
    return (
        f"Download the FA2/FA3 replay pair from Hugging Face (~340 MB):\n"
        f"  hf download {H100_PRESET_DATASET} --repo-type dataset \\\n"
        f"    profiles/perf_h100_sp1.sqlite profiles/perf_h100_sp1_fa3.sqlite\n"
        f"(requires the `hf` CLI: pip install -U huggingface_hub)\n"
        f"Expected cache layout: {H100_PRESET_CACHE}/<rev>/profiles/*.sqlite\n"
        f"Or pass paths explicitly:\n"
        f"  nsys-ai loop /path/to/perf_h100_sp1.sqlite "
        f"--after /path/to/perf_h100_sp1_fa3.sqlite"
    )


def detect_h100_replay_preset() -> dict[str, str] | None:
    """Return before/after paths if the known H100 dataset is present locally."""
    base = Path(
        "~/.cache/huggingface/hub/datasets--rich7421--fastvideo-wan-h100-sp1-nsys/snapshots"
    ).expanduser()
    if not base.exists():
        return None
    snapshots = [p for p in base.iterdir() if p.is_dir()]
    if not snapshots:
        return None
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for snap in snapshots:
        before = snap / "profiles" / H100_BEFORE_FILE
        after = snap / "profiles" / H100_AFTER_FILE
        if before.exists() and after.exists():
            return {
                "before_path": normalize_profile_path(str(before), label="before"),
                "after_path": normalize_profile_path(str(after), label="after"),
            }
    return None


def profile_display_name(path: str, preset: dict[str, str] | None = None) -> str:
    """Human-readable profile filename for loop UI labels."""
    raw = (path or "").strip()
    if not raw:
        return "—"
    preset = preset or detect_h100_replay_preset()
    if preset:
        if same_profile_path(raw, preset["before_path"]):
            return H100_BEFORE_FILE
        if same_profile_path(raw, preset["after_path"]):
            return H100_AFTER_FILE
    name = Path(raw).expanduser().name
    if name.endswith(".sqlite") and not _BLOB_STEM.match(Path(name).stem):
        return name
    return name or "—"


def reconcile_h100_loop_paths(state: DiffLoopState) -> None:
    """Point loop paths at HF snapshot files when they refer to the H100 preset blobs."""
    preset = detect_h100_replay_preset()
    if not preset:
        return
    if state.before_path and same_profile_path(state.before_path, preset["before_path"]):
        state.before_path = preset["before_path"]
    if state.after_path and same_profile_path(state.after_path, preset["after_path"]):
        state.after_path = preset["after_path"]


@dataclass
class DiffLoopState:
    before_path: str = ""
    after_path: str = ""
    phase: Phase = "diagnose"
    selected_scope: Scope = "global"
    proposal: str = ""
    expected_impact: str = ""
    decision: Decision | None = None
    decision_reason: str = ""
    diagnose_ran: bool = False
    diagnose_findings_count: int = 0
    top_findings: list[dict[str, Any]] = field(default_factory=list)
    diff_summary: dict[str, Any] | None = None
    comparability_confidence: float | None = None
    verdict: str = "neutral"
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        preset = detect_h100_replay_preset()
        return {
            "before_path": self.before_path,
            "after_path": self.after_path,
            "before_label": profile_display_name(self.before_path, preset),
            "after_label": profile_display_name(self.after_path, preset),
            "phase": self.phase,
            "selected_scope": self.selected_scope,
            "proposal": self.proposal,
            "expected_impact": self.expected_impact,
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "diagnose_ran": self.diagnose_ran,
            "diagnose_findings_count": self.diagnose_findings_count,
            "top_findings": self.top_findings,
            "diff_summary": self.diff_summary,
            "comparability_confidence": self.comparability_confidence,
            "verdict": self.verdict,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DiffLoopState:
        state = cls(
            before_path=str(payload.get("before_path") or ""),
            after_path=str(payload.get("after_path") or ""),
            phase=payload.get("phase") or "diagnose",
            selected_scope=payload.get("selected_scope") or "global",
            proposal=str(payload.get("proposal") or ""),
            expected_impact=str(payload.get("expected_impact") or ""),
            decision=payload.get("decision"),
            decision_reason=str(payload.get("decision_reason") or ""),
            diagnose_ran=bool(payload.get("diagnose_ran")),
            diagnose_findings_count=int(payload.get("diagnose_findings_count") or 0),
            top_findings=list(payload.get("top_findings") or []),
            diff_summary=payload.get("diff_summary"),
            comparability_confidence=payload.get("comparability_confidence"),
            verdict=str(payload.get("verdict") or "neutral"),
            last_error=str(payload.get("last_error") or ""),
        )
        if _phase_index(state.phase) < 0:
            state.phase = "diagnose"
        if state.selected_scope not in ("global", "gpu", "iteration", "region"):
            state.selected_scope = "global"
        return state

    def set_phase(self, new_phase: Phase, *, allow_backtrack: bool = True) -> None:
        cur = _phase_index(self.phase)
        nxt = _phase_index(new_phase)
        if nxt < 0:
            raise ValueError(f"Unknown phase: {new_phase}")
        if not allow_backtrack and nxt < cur:
            raise ValueError(f"Cannot move backward from {self.phase} to {new_phase}")
        self.phase = new_phase

    def set_proposal(self, text: str, expected_impact: str = "") -> None:
        self.proposal = (text or "").strip()
        self.expected_impact = (expected_impact or "").strip()
        self.last_error = ""
        if self.proposal:
            self.phase = "propose"

    def record_reprofile_artifact(self, after_path: str) -> None:
        self.after_path = normalize_profile_path(after_path, label="after")
        reconcile_h100_loop_paths(self)
        self.phase = "reprofile"
        self.last_error = ""

    def sync_before_path(self, path: str) -> None:
        """Align stored before_path with the profile currently loaded in the viewer."""
        self.before_path = normalize_profile_path(path, label="before")
        reconcile_h100_loop_paths(self)
        self.last_error = ""

    def run_diagnose(
        self,
        prof,
        *,
        device: int = 0,
        trim: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        from .evidence_builder import EvidenceBuilder

        builder = EvidenceBuilder(prof, device=device, trim=trim)
        report = builder.build()
        findings = [f.to_dict() for f in report.findings]
        ranked = _normalize_findings(findings)
        self.diagnose_ran = True
        self.diagnose_findings_count = len(ranked)
        self.top_findings = ranked[:15]
        self.phase = "diagnose"
        self.last_error = ""
        return ranked

    def run_diff(
        self,
        *,
        gpu: int | None = None,
        trim: tuple[int, int] | None = None,
        limit: int = 15,
        sort: str = "delta",
        baseline_prof=None,
    ) -> dict[str, Any]:
        if not self.after_path:
            raise ValueError("after_path is not set; register the candidate profile first")

        from . import profile as _profile
        from .diff import diff_profiles
        from .diff_render import to_diff_json

        after_path = normalize_profile_path(self.after_path, label="after")
        self.after_path = after_path

        if baseline_prof is not None:
            baseline_path = getattr(baseline_prof, "path", "") or self.before_path
            if baseline_path:
                self.before_path = normalize_profile_path(baseline_path, label="before")
            with _profile.open(after_path) as after_prof:
                summary = diff_profiles(
                    baseline_prof,
                    after_prof,
                    gpu=gpu,
                    trim=trim,
                    limit=limit,
                    sort=sort,
                )
        else:
            if not self.before_path:
                raise ValueError("before_path is not set")
            before_path = normalize_profile_path(self.before_path, label="before")
            self.before_path = before_path
            with _profile.open(before_path) as before_prof, _profile.open(after_path) as after_prof:
                summary = diff_profiles(
                    before_prof,
                    after_prof,
                    gpu=gpu,
                    trim=trim,
                    limit=limit,
                    sort=sort,
                )
        payload = json.loads(to_diff_json(summary))
        self.diff_summary = payload
        self.verdict = str(payload.get("verdict") or "neutral")
        confidence = payload.get("comparability_confidence")
        if isinstance(confidence, (int, float)):
            self.comparability_confidence = float(confidence)
        self.phase = "diff"
        self.last_error = ""
        return payload

    def set_decision(self, decision: Decision, reason: str = "") -> None:
        if decision not in ("accept", "reject"):
            raise ValueError("decision must be 'accept' or 'reject'")
        self.decision = decision
        self.decision_reason = (reason or "").strip()
        self.phase = "accept"
        self.last_error = ""
