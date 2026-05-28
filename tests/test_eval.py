"""test_eval.py — finding-correctness regression tests against labeled profiles.

This is the **eval seed**. Each label JSON in ``tests/eval/labels/*.json``
attaches expected / must-not finding assertions to a profile, providing the
first ground truth for ``EvidenceBuilder`` output. See ``tests/eval/README.md``.

Scoring (seed scope — informational by default, one hard gate):

  - ``expect[]``     unmet  -> false negative   (informational; printed, NOT gated)
  - ``must_not[]``   matched-> violation         (HARD gate: fails CI)
  - quality metrics (total, % categorized, category histogram) -> informational
  - ``known_gaps[]`` documented-but-unfixed observations         -> printed only

Why only ``must_not`` gates: positive expectations and quality ratios are
still being calibrated; gating them now would freeze unverified behavior
(roadmap §5). A false positive that a label explicitly forbids is, by
contrast, a stable regression signal. Gating can tighten once labels settle.

Profiles are resolved relative to the repo root. A label whose profile file is
absent is skipped, so only the committed ``tests/fixtures/mock.sqlite`` runs in
CI; large real profiles stay local-only.

Usage::

    pytest tests/test_eval.py -v          # metrics are printed via capsys.disabled()
    pytest tests/test_eval.py -k mock
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

_LABELS_DIR = _REPO_ROOT / "tests" / "eval" / "labels"

# Severity ladder spanning both the legacy Finding scale
# ("info" | "warning" | "critical") and the v0.1 Severity literal
# ("info" | "low" | "medium" | "high" | "critical"). "warning" and "medium"
# share a rung so a label can use either vocabulary.
_SEV_ORDER = {"info": 0, "low": 1, "warning": 2, "medium": 2, "high": 3, "critical": 4}


# ---------------------------------------------------------------------------
# Label loading + profile resolution
# ---------------------------------------------------------------------------


def _load_labels() -> list[dict]:
    if not _LABELS_DIR.is_dir():
        return []
    return [json.loads(p.read_text()) for p in sorted(_LABELS_DIR.glob("*.json"))]


def _resolve_profile(rel: str) -> Path | None:
    p = _REPO_ROOT / rel
    return p if p.is_file() else None


# ---------------------------------------------------------------------------
# Assertion matching
# ---------------------------------------------------------------------------


def _sev_ge(sev: str | None, minimum: str) -> bool:
    return _SEV_ORDER.get(sev or "info", 0) >= _SEV_ORDER.get(minimum, 0)


def _window_overlaps(window, fstart, fend) -> bool:
    """Loose overlap: assertion ``window_ns`` (``[start, end]``) vs a finding's
    ``[start_ns, end_ns]``. No window in the assertion means "don't constrain"."""
    if not window:
        return True
    a0 = window[0]
    a1 = window[1] if len(window) > 1 else a0
    if fstart is None:
        return False
    fend = fend if fend is not None else fstart
    return a0 <= fend and fstart <= a1


def _matches(finding: dict, assertion: dict) -> bool:
    cat = assertion.get("category")
    if cat is not None and finding.get("category") != cat:
        return False
    minsev = assertion.get("min_severity")
    if minsev and not _sev_ge(finding.get("severity"), minsev):
        return False
    if not _window_overlaps(assertion.get("window_ns"), finding.get("start_ns"), finding.get("end_ns")):
        return False
    return True


# ---------------------------------------------------------------------------
# Build + score
# ---------------------------------------------------------------------------


def _build_findings(label: dict, prof_path: Path) -> list[dict]:
    from nsys_ai.evidence_builder import EvidenceBuilder
    from nsys_ai.profile import Profile

    with Profile(str(prof_path)) as prof:
        report = EvidenceBuilder(prof, device=label.get("device", 0)).build()
    return [f.to_dict() for f in report.findings]


def _score(label: dict, findings: list[dict]) -> dict:
    unmet = [a for a in label.get("expect", []) if not any(_matches(f, a) for f in findings)]
    # An expect marked ``"required": true`` is a calibrated, gating assertion;
    # the rest are informational false negatives while labels stabilize.
    required_failures = [a for a in unmet if a.get("required")]
    false_negatives = [a for a in unmet if not a.get("required")]
    violations = []
    for a in label.get("must_not", []):
        hits = sum(1 for f in findings if _matches(f, a))
        if hits:
            violations.append({"assertion": a, "n_hits": hits})
    categorized = sum(1 for f in findings if f.get("category"))
    hist: dict = {}
    for f in findings:
        hist[f.get("category")] = hist.get(f.get("category"), 0) + 1
    return {
        "required_failures": required_failures,
        "false_negatives": false_negatives,
        "violations": violations,
        "categorized_pct": (100.0 * categorized / len(findings)) if findings else 0.0,
        "category_hist": hist,
    }


def _report(label: dict, findings: list[dict], score: dict) -> str:
    lines = [
        f"\n[eval:{label['id']}] {label['profile']}",
        f"  findings={len(findings)}  categorized={score['categorized_pct']:.0f}%"
        f"  by_category={score['category_hist']}",
    ]
    if score["required_failures"]:
        lines.append(f"  REQUIRED EXPECTATIONS UNMET (gated): {score['required_failures']}")
    if score["false_negatives"]:
        lines.append(f"  false negatives (expected but absent; informational): {score['false_negatives']}")
    if score["violations"]:
        lines.append(f"  MUST-NOT VIOLATIONS (gated): {score['violations']}")
    for g in label.get("known_gaps", []):
        lines.append(f"  known gap: {g.get('observation')}  [{g.get('ref', '')}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

_LABELS = _load_labels()


@pytest.mark.skipif(not _LABELS, reason="no eval labels in tests/eval/labels/")
@pytest.mark.parametrize("label", _LABELS, ids=[lbl["id"] for lbl in _LABELS])
def test_eval_label(label: dict, capsys) -> None:
    prof_path = _resolve_profile(label["profile"])
    if prof_path is None:
        pytest.skip(f"profile not available locally: {label['profile']}")

    findings = _build_findings(label, prof_path)
    score = _score(label, findings)

    with capsys.disabled():
        print(_report(label, findings, score))

    # Seed gates: forbidden findings (must_not) and calibrated required
    # expectations. Non-required expectation misses and quality ratios stay
    # informational while labels are being calibrated.
    assert not score["violations"], (
        f"{label['id']}: forbidden findings present — {score['violations']}"
    )
    assert not score["required_failures"], (
        f"{label['id']}: required findings missing — {score['required_failures']}"
    )


# ---------------------------------------------------------------------------
# Harness unit tests — validate the matcher/scorer itself (no profile needed,
# so these always run in CI even when every labeled profile is absent).
# ---------------------------------------------------------------------------


class TestMatcher:
    def test_sev_ge_ladder(self) -> None:
        assert _sev_ge("critical", "warning")
        assert _sev_ge("warning", "warning")
        assert not _sev_ge("info", "warning")
        assert _sev_ge("medium", "warning")  # "medium" and "warning" share a rung
        assert not _sev_ge(None, "low")  # missing severity treated as "info"

    def test_window_overlap(self) -> None:
        assert _window_overlaps([10, 20], 15, 18)  # finding inside window
        assert _window_overlaps([10, 20], 18, 25)  # partial overlap
        assert not _window_overlaps([10, 20], 30, 40)  # disjoint
        assert _window_overlaps(None, 0, 0)  # no window constraint
        assert _window_overlaps([10], 10, None)  # single-point window vs marker
        assert not _window_overlaps([10, 20], None, None)  # finding has no start

    def test_matches_category_severity_window(self) -> None:
        f = {"category": "idle", "severity": "warning", "start_ns": 100, "end_ns": 200}
        assert _matches(f, {"category": "idle"})
        assert _matches(f, {"category": "idle", "min_severity": "warning"})
        assert _matches(f, {"category": "idle", "window_ns": [150, 160]})
        assert not _matches(f, {"category": "communication"})
        assert not _matches(f, {"category": "idle", "min_severity": "critical"})
        assert not _matches(f, {"category": "idle", "window_ns": [300, 400]})


class TestScore:
    def test_required_vs_informational_and_must_not(self) -> None:
        findings = [{"category": "idle", "severity": "warning", "start_ns": 1, "end_ns": 2}]
        label = {
            "expect": [
                {"category": "idle", "required": True},  # met
                {"category": "communication"},  # unmet, informational
            ],
            "must_not": [{"category": "memory"}],  # no memory finding -> no violation
        }
        score = _score(label, findings)
        assert score["required_failures"] == []
        assert len(score["false_negatives"]) == 1
        assert score["false_negatives"][0]["category"] == "communication"
        assert score["violations"] == []
        assert score["categorized_pct"] == 100.0

    def test_required_failure_and_violation_detected(self) -> None:
        findings = [{"category": "communication", "severity": "critical", "start_ns": 5}]
        label = {
            "expect": [{"category": "idle", "required": True}],  # unmet -> gating failure
            "must_not": [{"category": "communication"}],  # present -> violation
        }
        score = _score(label, findings)
        assert len(score["required_failures"]) == 1
        assert len(score["violations"]) == 1
