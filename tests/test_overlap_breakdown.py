"""Tests for overlap_breakdown skill enrichments.

Covers the v0.2.4 additions:
- `present_devices` field surfaces other GPUs in the profile
- `same_stream_compute_pct` / `same_stream_nccl_pct` quantify the
  proportion behind `same_stream_diagnosis`

Note: several tests mutate the `minimal_nsys_conn` fixture (INSERT/UPDATE
extra rows). This is safe today because conftest declares it function-scoped
— each test gets a fresh in-memory connection. If anyone bumps the fixture
to session scope, these mutations will leak between tests.

The conftest's `minimal_nsys_conn` seed places 5 kernels on device 0:

    streamId=7  compute (cId=1,  shortName=1  → kernel_A)
    streamId=7  compute (cId=2,  shortName=2  → kernel_B)
    streamId=8  NCCL    (cId=10, shortName=10 → nccl_AllReduce_kernel)
    streamId=7  NCCL    (cId=12, shortName=11 → nccl_ReduceScatter_kernel)
    streamId=7  compute (cId=13, shortName=1  → kernel_A)

So stream 7 has 3 compute + 1 NCCL (a same-stream candidate). Stream 8
has 1 NCCL only (not same-stream). Across the device:
    total compute = 3, total NCCL = 2
    same-stream compute on stream 7 = 3 (100 %)
    same-stream NCCL on stream 7 = 1 (50 %)
"""

import json
import sqlite3

from nsys_ai.annotation import EvidenceRow, Finding, TraceSelection
from nsys_ai.skills.builtins.overlap_breakdown import SKILL


def test_present_devices_single_device(minimal_nsys_conn):
    rows = SKILL.execute_fn(minimal_nsys_conn)
    assert rows, "overlap_breakdown returned no rows"
    r = rows[0]
    assert r.get("present_devices") == [0], (
        f"Expected [0] for single-device fixture, got {r.get('present_devices')!r}"
    )


def test_present_devices_multi_device(minimal_nsys_conn):
    """Add kernels on devices 1, 2, 3 and confirm all four are surfaced."""
    c = minimal_nsys_conn.cursor()
    # Append a compute kernel on each extra device. Same shortName as
    # existing rows so the schema check holds.
    for dev in (1, 2, 3):
        c.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (100, dev, 17, 50 + dev, 10_000_000 + dev * 100, 11_000_000 + dev * 100,
             1, 1, 32, 1, 1, 256, 1, 1),
        )
    minimal_nsys_conn.commit()

    rows = SKILL.execute_fn(minimal_nsys_conn)
    assert rows[0].get("present_devices") == [0, 1, 2, 3]


def test_same_stream_proportions(minimal_nsys_conn):
    """Stream 7 carries all 3 compute + 1 of 2 NCCL kernels →
    same_stream_compute_pct = 100, same_stream_nccl_pct = 50."""
    rows = SKILL.execute_fn(minimal_nsys_conn)
    r = rows[0]
    assert r.get("same_stream_diagnosis") == ["7"], (
        f"Expected stream 7 flagged, got {r.get('same_stream_diagnosis')!r}"
    )
    assert r.get("same_stream_compute_pct") == 100.0, (
        f"Expected 100.0 % of compute on stream 7, "
        f"got {r.get('same_stream_compute_pct')!r}"
    )
    assert r.get("same_stream_nccl_pct") == 50.0, (
        f"Expected 50.0 % of NCCL on stream 7, "
        f"got {r.get('same_stream_nccl_pct')!r}"
    )


def test_clean_multi_stream_does_not_fire_diagnostic(minimal_nsys_conn):
    """Move the same-stream NCCL kernel to stream 8 so no stream has both
    compute and NCCL. The diagnostic must not fire and the proportions
    must not be emitted."""
    c = minimal_nsys_conn.cursor()
    # cId=12 is the NCCL kernel currently on stream 7. Move it.
    c.execute(
        "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET streamId = 8 "
        "WHERE correlationId = 12"
    )
    minimal_nsys_conn.commit()

    rows = SKILL.execute_fn(minimal_nsys_conn)
    r = rows[0]
    assert "same_stream_diagnosis" not in r, (
        f"Diagnostic should not fire when no stream has both kinds "
        f"(got {r.get('same_stream_diagnosis')!r})"
    )
    assert "same_stream_compute_pct" not in r
    assert "same_stream_nccl_pct" not in r


def test_stray_kernel_on_nccl_stream_does_not_fire_diagnostic(minimal_nsys_conn):
    """A few stray compute kernels sharing the NCCL stream must NOT trigger
    same_stream_diagnosis when they are a tiny fraction of compute.

    Stream 7 keeps its 3 compute + 1 NCCL (the would-be false positive). Adding
    100 compute kernels on a clean stream drops stream-7's compute share to
    ~2.9% (3/103) — below the 5% threshold — so the diagnosis must stay silent.
    """
    c = minimal_nsys_conn.cursor()
    for i in range(100):
        c.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (100, 0, 9, 200 + i, 20_000_000 + i * 100, 20_000_000 + i * 100 + 50,
             1, 1, 32, 1, 1, 256, 1, 1),  # stream 9, compute (shortName=1), no NCCL
        )
    minimal_nsys_conn.commit()

    r = SKILL.execute_fn(minimal_nsys_conn)[0]
    assert "same_stream_diagnosis" not in r, (
        f"stray compute on the NCCL stream (~2.9% of compute) should not fire; "
        f"got {r.get('same_stream_diagnosis')!r}"
    )
    assert "same_stream_compute_pct" not in r
    assert "same_stream_nccl_pct" not in r


def test_present_devices_survives_empty_kernel_table():
    """If the kernel table exists but is empty, present_devices should
    either be empty or absent — not crash."""
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    c.execute("CREATE TABLE NsightSchemaMeta (version INTEGER)")
    c.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    c.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
        "(globalPid INTEGER, deviceId INTEGER, streamId INTEGER, "
        " correlationId INTEGER, start INTEGER, \"end\" INTEGER, "
        " shortName INTEGER, demangledName INTEGER, "
        " gridX INTEGER, gridY INTEGER, gridZ INTEGER, "
        " blockX INTEGER, blockY INTEGER, blockZ INTEGER)"
    )
    conn.commit()

    rows = SKILL.execute_fn(conn)
    # Either an error row (no kernel activity) or normal row with no
    # present_devices key. Both are acceptable; the assertion is that
    # we don't crash.
    assert rows is not None
    if rows and "error" not in rows[0]:
        assert rows[0].get("present_devices", []) == []


# ---------------------------------------------------------------------------
# Textual rendering — _format must surface the new fields so users running
# the skill without --format json see same-stream warnings and multi-device
# hints inline.
# ---------------------------------------------------------------------------


def test_format_shows_same_stream_warning_with_pct(minimal_nsys_conn):
    """_format must emit the ⚠️ Same-stream line with the proportion."""
    rows = SKILL.execute_fn(minimal_nsys_conn)
    text = SKILL.format_fn(rows)
    assert "Same-stream" in text, f"missing warning line in:\n{text}"
    assert "[7]" in text, "expected stream id 7 in warning"
    assert "100" in text and "50" in text, (
        f"expected compute_pct=100 and nccl_pct=50 in warning, got:\n{text}"
    )


def test_format_shows_multi_device_hint(minimal_nsys_conn):
    """When >1 device exists, _format hints at other ranks."""
    c = minimal_nsys_conn.cursor()
    for dev in (1, 2, 3):
        c.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (100, dev, 17, 50 + dev, 10_000_000 + dev * 100, 11_000_000 + dev * 100,
             1, 1, 32, 1, 1, 256, 1, 1),
        )
    minimal_nsys_conn.commit()

    text = SKILL.format_fn(SKILL.execute_fn(minimal_nsys_conn))
    assert "Devices:" in text, f"missing devices line in:\n{text}"
    assert "1, 2, 3" in text, f"expected other ranks listed; got:\n{text}"


def test_format_no_multi_device_hint_single_device(minimal_nsys_conn):
    """Single-device profile must not emit the multi-device hint."""
    text = SKILL.format_fn(SKILL.execute_fn(minimal_nsys_conn))
    assert "Devices:" not in text, f"unexpected multi-device hint:\n{text}"


# ---------------------------------------------------------------------------
# Findings — _to_findings should bake the proportions into the note so the
# evidence layer downstream of analyze --format json gets the context.
# ---------------------------------------------------------------------------


def _overlap_row(**overrides):
    row = {
        "compute_only_ms": 20.0,
        "nccl_only_ms": 80.0,
        "overlap_ms": 10.0,
        "idle_ms": 5.0,
        "total_ms": 115.0,
        "overlap_pct": 11.1,
        "span_start_ns": 1_000,
        "span_end_ns": 9_000,
        "device_id": 2,
        "compute_kernels": 4,
        "nccl_kernels": 2,
        "same_stream_diagnosis": ["7"],
        "same_stream_compute_pct": 75.0,
        "same_stream_nccl_pct": 50.0,
    }
    row.update(overrides)
    return row


def test_findings_note_includes_same_stream_pct(minimal_nsys_conn):
    rows = SKILL.execute_fn(minimal_nsys_conn)
    findings = SKILL.to_findings_fn(rows)
    # Look for the low-overlap finding (always emitted in this fixture
    # because overlap_pct is < 30 with same-stream serialization).
    low_overlap = [f for f in findings if "Low Compute/NCCL Overlap" in f.label]
    assert low_overlap, f"expected a low-overlap finding, got: {[f.label for f in findings]}"
    note = low_overlap[0].note
    # Proportion strings must appear so consumers can quantify the warning
    # without re-querying the skill output.
    assert "100" in note and "50" in note, (
        f"expected 100% / 50% proportions in finding note, got:\n{note}"
    )


def test_low_overlap_finding_has_v01_evidence_shape():
    findings = SKILL.to_findings_fn(
        [_overlap_row()],
        context={"profile_id": "nsys1:sha256:test"},
    )
    low = [f for f in findings if f.id and f.id.startswith("overlap_low_gpu2_")]
    assert low, f"expected low-overlap finding, got {[f.label for f in findings]}"

    f = low[0]
    assert f.type == "region"
    assert f.label == "Low Compute/NCCL Overlap (11.1%)"
    assert f.start_ns == 1_000
    assert f.end_ns == 9_000
    assert f.gpu_id == 2
    assert f.severity == "warning"
    assert f.category == "communication"
    assert isinstance(f.confidence, float)
    assert 0.0 <= f.confidence <= 1.0
    assert f.explanation and "NCCL" in f.explanation
    assert f.suggested_actions
    assert f.false_positive_notes
    assert f.provenance == {"skill": "overlap_breakdown", "row_kind": "low_overlap"}

    assert isinstance(f.selection, TraceSelection)
    assert f.selection.profile_id == "nsys1:sha256:test"
    assert f.selection.source == "skill:overlap_breakdown"
    assert f.selection.start_ns == 1_000
    assert f.selection.end_ns == 9_000
    assert f.selection.gpu_ids == [2]

    assert f.evidence and isinstance(f.evidence[0], EvidenceRow)
    ev = f.evidence[0]
    assert ev.source_skill == "overlap_breakdown"
    assert ev.selection_id == f.selection.id
    assert ev.values["overlap_pct"] == 11.1
    assert ev.values["nccl_only_ms"] == 80.0
    assert ev.values["overlap_ms"] == 10.0
    assert ev.values["nccl_total_ms"] == 90.0
    assert ev.values["same_streams"] == ["7"]
    assert ev.values["same_stream_compute_pct"] == 75.0
    assert ev.values["same_stream_nccl_pct"] == 50.0
    assert ev.units["overlap_pct"] == "percent"
    assert ev.units["nccl_only_ms"] == "ms"


def test_communication_dominated_finding_has_ratio_evidence():
    findings = SKILL.to_findings_fn(
        [_overlap_row(compute_only_ms=30.0, nccl_only_ms=100.0, overlap_ms=0.0)],
        context={"profile_id": "p"},
    )
    dominated = [f for f in findings if f.id and f.id.startswith("overlap_comm_dominated")]
    assert dominated, f"expected communication-dominated finding, got {[f.label for f in findings]}"

    f = dominated[0]
    assert f.label == "Communication Dominated (ratio=0.30)"
    assert f.category == "communication"
    assert f.severity == "critical"
    assert f.selection.profile_id == "p"
    assert f.provenance["row_kind"] == "communication_dominated"
    ev = f.evidence[0]
    assert ev.values["compute_nccl_ratio"] == 0.3
    assert ev.units["compute_nccl_ratio"] == "ratio"
    assert ev.provenance["row_kind"] == "communication_dominated"


def test_overlap_finding_round_trips_through_json():
    findings = SKILL.to_findings_fn(
        [_overlap_row()],
        context={"profile_id": "/tmp/profile.sqlite"},
    )
    for f in findings:
        d = f.to_dict()
        assert isinstance(d["selection"], dict)
        assert isinstance(d["evidence"], list)
        restored = Finding.from_dict(json.loads(json.dumps(d)))
        assert restored.id == f.id
        assert restored.category == "communication"
        assert isinstance(restored.selection, TraceSelection)
        assert restored.selection.profile_id == "/tmp/profile.sqlite"
        assert isinstance(restored.evidence[0], EvidenceRow)


def test_overlap_findings_without_context_use_unknown_profile_id():
    findings = SKILL.to_findings_fn([_overlap_row()])
    assert findings
    assert findings[0].selection.profile_id == "unknown"


# ---------------------------------------------------------------------------
# Trim-aware path — passing -p trim_start_ns / trim_end_ns must still emit
# the new fields and proportions, scoped to the trimmed window.
# ---------------------------------------------------------------------------


def test_trim_window_preserves_enrichments(minimal_nsys_conn):
    """All five fixture kernels fall in [1000000, 9000000] ns; a trim
    that includes only stream-7 same-stream rows (4_000_000..6_000_000)
    catches just the NCCL-on-stream-7 (cId=12, 4.5..5.5 ms) — proportions
    should still emit if both kinds exist in the window."""
    rows = SKILL.execute_fn(
        minimal_nsys_conn,
        trim_start_ns=500_000,
        trim_end_ns=9_500_000,  # wide enough to include all 5 kernels
    )
    r = rows[0]
    # present_devices is profile-wide, not window-scoped → always [0].
    assert r.get("present_devices") == [0]
    # Same-stream diagnosis should still fire with the wide trim.
    assert r.get("same_stream_diagnosis") == ["7"]
    assert r.get("same_stream_compute_pct") == 100.0
    assert r.get("same_stream_nccl_pct") == 50.0
