"""One shape for `diff.json`, decided or not, and a versioned envelope.

The decided record was never a different *shape* from the undecided one — it is
the same object plus a populated `decision`. What differed is that the undecided
payload omitted the key entirely, so every consumer had to `.get()` it and the
one artifact read as two. It is now always present, `null` until decided.

On versioning: nothing in this repo reads a stored `diff.json` back. `diff_web`
regenerates it, and every use of `decision_path` stores or prints the path. So
there is no load path on which to enforce a compatibility check, and adding one
would invent a consumer that does not exist. The protection that does apply to a
write-only artifact is pinning the envelope here, so a change to it has to be
deliberate.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BEFORE = REPO / "tests" / "fixtures" / "mfu_2gpu_before.sqlite"
AFTER = REPO / "tests" / "fixtures" / "mfu_2gpu_after.sqlite"


@pytest.fixture(scope="module")
def undecided() -> dict:
    """The payload the CLI and the web view emit before a decision."""
    from nsys_ai.diff import diff_profiles
    from nsys_ai.diff_render import to_diff_dict
    from nsys_ai.profile import Profile

    with Profile(str(BEFORE)) as b, Profile(str(AFTER)) as a:
        summary = diff_profiles(b, a, gpu=0)
    return to_diff_dict(summary)


def test_the_decision_key_is_always_present(undecided):
    """`null` until decided, rather than absent.

    Omitting it is what made one artifact look like two: a consumer reading the
    undecided payload had to know the key might not be there, while the same
    consumer reading a decided one could index it directly.
    """
    assert "decision" in undecided, "the undecided payload omits the key again"
    assert undecided["decision"] is None


def test_deciding_adds_only_the_decision(undecided):
    """The decided record is the undecided one plus a populated decision.

    Asserted directly, because the issue behind this change claimed the two were
    structurally different envelopes. They are not, and a future change that
    made them diverge would be the actual defect.
    """
    from nsys_ai.diff_decision import build_diff_decision_record_from_diff_dict

    decided, _warnings = build_diff_decision_record_from_diff_dict(
        undecided, decision="accepted", reason="shape test"
    )
    assert set(decided) == set(undecided), (
        f"deciding changed the key set: "
        f"added {sorted(set(decided) - set(undecided))}, "
        f"dropped {sorted(set(undecided) - set(decided))}"
    )
    assert decided["decision"] is not None
    assert decided["decision"]["status"] == "accepted"


def test_whitespace_environment_identity_falls_back_before_building_decision(
    monkeypatch,
):
    import nsys_ai.diff_decision as diff_decision

    def unavailable_git(*_args, **_kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(diff_decision.subprocess, "run", unavailable_git)
    monkeypatch.setenv("USER", "   ")
    monkeypatch.setenv("USERNAME", "\t")
    payload, _warnings = diff_decision.build_diff_decision_record_from_diff_dict(
        {
            "before": {"profile_id": "before"},
            "after": {"profile_id": "after"},
            "warnings": [],
        },
        decision="accepted",
        reason="verified",
    )

    assert payload["decision"]["decider"] == "unknown"


def test_the_envelope_is_pinned(undecided):
    """A write-only artifact has no reader to reject a bad version, so the
    envelope is pinned here instead.

    These are the fields a consumer outside this repo would key off. Adding an
    optional field is fine and does not bump the version; removing or renaming
    one of these is the breaking change the version exists to signal.
    """
    from nsys_ai.annotation import PRODUCER, SCHEMA_VERSION

    required = {
        "schema_version",
        "producer",
        "producer_version",
        "diff_id",
        "before",
        "after",
        "verdict",
        "comparability_confidence",
        "decision",
    }
    missing = required - set(undecided)
    assert not missing, f"the diff.json envelope lost required fields: {sorted(missing)}"
    assert undecided["schema_version"] == SCHEMA_VERSION
    assert undecided["producer"] == PRODUCER
    for side in ("before", "after"):
        assert "profile_id" in undecided[side]
        assert "path" in undecided[side]


def test_it_is_json_serialisable(undecided):
    """The artifact is written to disk; a value that cannot serialise is a bug
    that would only surface at write time."""
    assert json.loads(json.dumps(undecided, default=str))["decision"] is None


def test_canonical_diff_passes_the_session_artifact_validator(undecided):
    from nsys_ai.session_store import _validate_diff_payload

    _validate_diff_payload(undecided, require_undecided=True)


# ── The two version constants are separate on purpose ───────────────────────


def test_the_doctor_version_is_not_the_evidence_version():
    """They version different artifacts and merely share a value today.

    The issue behind this change asked for a single definition. That would be
    wrong: `annotation.SCHEMA_VERSION` versions the evidence and diff envelope,
    the doctor constant versions the doctor report. Merging them would make a
    breaking change to one falsely bump the other. The rename makes the
    separation intentional rather than a duplicate that invites merging.
    """
    import nsys_ai.doctor as doctor

    assert hasattr(doctor, "DOCTOR_SCHEMA_VERSION")
    assert not hasattr(doctor, "SCHEMA_VERSION"), (
        "the bare name is back, which is what made this look like a duplicate"
    )

    src = (REPO / "src" / "nsys_ai" / "doctor.py").read_text()
    assert "annotation.SCHEMA_VERSION" in src, "the reason for the split is undocumented"


def test_the_evidence_version_documents_when_it_bumps():
    """The version is a breaking-change counter, not semver.

    Semver would let a 0.y differ freely — 'anything MAY change at any time' —
    but this repo's own rule is narrower: bumped only on breaking envelope
    changes, with backward-compatible additions not bumping it. That is what
    makes the pin above meaningful, so the rule has to stay written down.
    """
    src = (REPO / "src" / "nsys_ai" / "annotation.py").read_text()
    assert "Bumped on breaking changes" in src
    assert "do not bump this" in src
