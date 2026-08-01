"""Tests for v0.1 evidence schema models in annotation.py.

Covers EvidenceRow, TraceSelection, DiffLineage, Diagnostic.
Existing Finding / EvidenceReport tests live in test_evidence_build.py.
"""

import json

from nsys_ai.annotation import (
    Diagnostic,
    DiffLineage,
    EvidenceReport,
    EvidenceRow,
    Finding,
    TraceSelection,
)


class TestEvidenceRow:
    def test_minimal_construction(self):
        row = EvidenceRow(id="r0", source_skill="gpu_idle_gaps")
        assert row.id == "r0"
        assert row.source_skill == "gpu_idle_gaps"
        assert row.values == {}
        assert row.units == {}
        assert row.selection_id is None
        assert row.provenance == {}

    def test_full_construction(self):
        row = EvidenceRow(
            id="r1",
            source_skill="overlap_breakdown",
            values={"overlap_ratio": 0.12, "exposed_ms": 48},
            units={"overlap_ratio": "ratio", "exposed_ms": "ms"},
            selection_id="sel_001",
            provenance={"trim_start_ns": 0, "trim_end_ns": 1_000_000},
        )
        assert row.values["overlap_ratio"] == 0.12
        assert row.selection_id == "sel_001"
        assert row.provenance["trim_end_ns"] == 1_000_000

    def test_round_trip(self):
        original = EvidenceRow(
            id="r2",
            source_skill="top_kernels",
            values={"count": 42},
            units={"count": "count"},
        )
        restored = EvidenceRow.from_dict(original.to_dict())
        assert restored == original

    def test_round_trip_json(self):
        row = EvidenceRow(id="r3", source_skill="nccl_breakdown", values={"a": 1})
        restored = EvidenceRow.from_dict(json.loads(json.dumps(row.to_dict())))
        assert restored == row

    def test_from_dict_ignores_unknown_keys(self):
        row = EvidenceRow.from_dict({"id": "r4", "source_skill": "x", "future_field": "ignored"})
        assert row.id == "r4"
        assert row.source_skill == "x"

    def test_to_dict_drops_none(self):
        row = EvidenceRow(id="r5", source_skill="x")
        d = row.to_dict()
        assert "selection_id" not in d


class TestTraceSelection:
    def test_minimal_construction(self):
        sel = TraceSelection(id="s0", profile_id="abc123", source="user")
        assert sel.id == "s0"
        assert sel.profile_id == "abc123"
        assert sel.source == "user"
        assert sel.start_ns is None
        assert sel.gpu_ids is None

    def test_full_construction(self):
        sel = TraceSelection(
            id="s1",
            profile_id="abc123",
            source="skill:gpu_idle_gaps",
            start_ns=1_000_000,
            end_ns=2_000_000,
            gpu_ids=[0, 1],
            rank_ids=[5],
            stream_ids=[7, 13],
            nvtx_path=["iteration_142", "backward"],
            event_ids=["evt_99"],
            label="NCCL exposed",
        )
        assert sel.gpu_ids == [0, 1]
        assert sel.nvtx_path[-1] == "backward"
        assert sel.label == "NCCL exposed"

    def test_round_trip_minimal(self):
        sel = TraceSelection(id="s2", profile_id="def", source="gui")
        restored = TraceSelection.from_dict(sel.to_dict())
        assert restored == sel

    def test_round_trip_full(self):
        sel = TraceSelection(
            id="s3",
            profile_id="def",
            source="diff",
            start_ns=10,
            end_ns=20,
            gpu_ids=[0],
            nvtx_path=["a", "b"],
            label="region",
        )
        restored = TraceSelection.from_dict(sel.to_dict())
        assert restored == sel

    def test_round_trip_json(self):
        sel = TraceSelection(id="s4", profile_id="x", source="user", start_ns=1, gpu_ids=[0, 1])
        restored = TraceSelection.from_dict(json.loads(json.dumps(sel.to_dict())))
        assert restored == sel

    def test_to_dict_drops_none_optionals(self):
        sel = TraceSelection(id="s5", profile_id="x", source="user")
        d = sel.to_dict()
        # Optional fields with None values are not serialized
        assert "start_ns" not in d
        assert "gpu_ids" not in d
        assert "label" not in d
        # Required fields are
        assert d["id"] == "s5"
        assert d["source"] == "user"

    def test_empty_list_distinct_from_none(self):
        """Empty list ≠ None: empty list means 'selection contains zero gpus'."""
        sel = TraceSelection(id="s6", profile_id="x", source="user", gpu_ids=[])
        d = sel.to_dict()
        assert d["gpu_ids"] == []

    def test_from_dict_ignores_unknown_keys(self):
        sel = TraceSelection.from_dict(
            {
                "id": "s7",
                "profile_id": "x",
                "source": "user",
                "unknown": 123,
            }
        )
        assert sel.id == "s7"


class TestDiffLineage:
    def test_construction(self):
        lin = DiffLineage(
            diff_id="diff_20260511",
            role="regression",
            rank=2,
            baseline_profile_id="base_v1_0",
        )
        assert lin.role == "regression"
        assert lin.rank == 2
        assert lin.baseline_profile_id == "base_v1_0"

    def test_round_trip(self):
        lin = DiffLineage(diff_id="d", role="improvement", rank=0, baseline_profile_id="b")
        restored = DiffLineage.from_dict(lin.to_dict())
        assert restored == lin

    def test_round_trip_json(self):
        lin = DiffLineage(diff_id="d", role="stable", rank=5, baseline_profile_id="b")
        restored = DiffLineage.from_dict(json.loads(json.dumps(lin.to_dict())))
        assert restored == lin

    def test_from_dict_ignores_unknown_keys(self):
        lin = DiffLineage.from_dict(
            {
                "diff_id": "d",
                "role": "stable",
                "rank": 0,
                "baseline_profile_id": "b",
                "extra": "ignored",
            }
        )
        assert lin.diff_id == "d"


class TestDiagnostic:
    def test_minimal_construction(self):
        diag = Diagnostic(
            id="diag_0",
            summary="Test summary",
            recommendation="Do nothing",
            verification_command="nsys-ai diff a.sqlite b.sqlite",
            confidence=0.9,
        )
        assert diag.primary_findings == []
        assert diag.root_cause_hypotheses == []
        assert diag.confidence == 0.9

    def test_with_findings_and_hypotheses(self):
        f = Finding(type="marker", label="L", start_ns=0)
        diag = Diagnostic(
            id="diag_1",
            summary="S",
            recommendation="R",
            verification_command="V",
            confidence=0.7,
            primary_findings=[f],
            root_cause_hypotheses=["h1", "h2"],
        )
        assert len(diag.primary_findings) == 1
        assert diag.primary_findings[0].label == "L"
        assert diag.root_cause_hypotheses == ["h1", "h2"]

    def test_to_dict_serializes_nested_findings(self):
        f1 = Finding(type="region", label="A", start_ns=10, end_ns=20)
        f2 = Finding(type="marker", label="B", start_ns=30)
        diag = Diagnostic(
            id="diag_2",
            summary="S",
            recommendation="R",
            verification_command="V",
            confidence=0.5,
            primary_findings=[f1, f2],
        )
        d = diag.to_dict()
        # Nested findings should be dicts, not Finding objects
        assert isinstance(d["primary_findings"][0], dict)
        assert d["primary_findings"][0]["label"] == "A"
        assert d["primary_findings"][1]["start_ns"] == 30

    def test_round_trip_with_nested_findings(self):
        f1 = Finding(type="region", label="A", start_ns=10, end_ns=20)
        f2 = Finding(type="marker", label="B", start_ns=30)
        diag = Diagnostic(
            id="diag_3",
            summary="S",
            recommendation="R",
            verification_command="V",
            confidence=0.5,
            primary_findings=[f1, f2],
            root_cause_hypotheses=["h1"],
        )
        restored = Diagnostic.from_dict(diag.to_dict())
        assert len(restored.primary_findings) == 2
        assert restored.primary_findings[0].label == "A"
        assert restored.primary_findings[1].start_ns == 30
        assert restored.root_cause_hypotheses == ["h1"]

    def test_round_trip_json(self):
        f = Finding(type="region", label="L", start_ns=1)
        diag = Diagnostic(
            id="diag_4",
            summary="S",
            recommendation="R",
            verification_command="nsys-ai diff x y",
            confidence=0.6,
            primary_findings=[f],
            root_cause_hypotheses=["h"],
        )
        restored = Diagnostic.from_dict(json.loads(json.dumps(diag.to_dict())))
        assert restored.id == diag.id
        assert restored.confidence == diag.confidence
        assert restored.verification_command == diag.verification_command
        assert restored.primary_findings[0].label == "L"

    def test_from_dict_ignores_unknown_keys(self):
        diag = Diagnostic.from_dict(
            {
                "id": "d",
                "summary": "s",
                "recommendation": "r",
                "verification_command": "v",
                "confidence": 0.5,
                "future_field": "ignored",
            }
        )
        assert diag.id == "d"
        assert diag.primary_findings == []


class TestDiagnosticEdgeCases:
    """Edge cases for Diagnostic: None lists, mutation safety, unicode, zero values."""

    def _minimal(self, **overrides):
        defaults = dict(
            id="d",
            summary="s",
            recommendation="r",
            verification_command="v",
            confidence=0.5,
        )
        defaults.update(overrides)
        return Diagnostic.from_dict(defaults)

    def test_from_dict_with_null_primary_findings(self):
        """JSON null → empty list, not a crash."""
        diag = self._minimal(primary_findings=None)
        assert diag.primary_findings == []

    def test_from_dict_with_null_root_cause_hypotheses(self):
        diag = self._minimal(root_cause_hypotheses=None)
        assert diag.root_cause_hypotheses == []

    def test_from_dict_with_missing_optional_lists(self):
        """Missing keys use dataclass defaults."""
        diag = self._minimal()
        assert diag.primary_findings == []
        assert diag.root_cause_hypotheses == []

    def test_to_dict_does_not_share_list_state(self):
        """Mutating the returned dict's lists does not affect the Diagnostic."""
        f = Finding(type="region", label="L", start_ns=0)
        diag = Diagnostic(
            id="d",
            summary="s",
            recommendation="r",
            verification_command="v",
            confidence=0.5,
            primary_findings=[f],
            root_cause_hypotheses=["h"],
        )
        d = diag.to_dict()
        d["root_cause_hypotheses"].append("mutation")
        d["primary_findings"].append({"injected": True})
        assert diag.root_cause_hypotheses == ["h"]
        assert len(diag.primary_findings) == 1

    def test_to_dict_repeated_calls_produce_independent_results(self):
        """Two consecutive to_dict() calls return independent dicts."""
        diag = Diagnostic(
            id="d",
            summary="s",
            recommendation="r",
            verification_command="v",
            confidence=0.5,
            root_cause_hypotheses=["a"],
        )
        d1 = diag.to_dict()
        d2 = diag.to_dict()
        d1["root_cause_hypotheses"].append("mutation")
        assert d2["root_cause_hypotheses"] == ["a"]

    def test_full_json_round_trip_with_unicode(self):
        f1 = Finding(
            type="region",
            label="A",
            start_ns=10,
            end_ns=20,
            severity="warning",
            note="中文 note — em-dash",
        )
        f2 = Finding(type="marker", label="B", start_ns=30)
        diag = Diagnostic(
            id="diag",
            summary="unicode: 中文 ✓ — em-dash",
            recommendation="action",
            verification_command="nsys-ai diff a b",
            confidence=0.0,  # edge: zero confidence
            primary_findings=[f1, f2],
            root_cause_hypotheses=["h1", "h2"],
        )
        s = json.dumps(diag.to_dict(), indent=2, ensure_ascii=False)
        restored = Diagnostic.from_dict(json.loads(s))
        assert restored == diag

    def test_confidence_boundary_values(self):
        """confidence accepts 0.0 and 1.0; no validation enforced (yet)."""
        for c in (0.0, 0.5, 1.0):
            diag = self._minimal(confidence=c)
            assert diag.confidence == c

    def test_verification_command_can_be_multiline(self):
        """Verification commands may be multi-line shell snippets."""
        cmd = (
            "nsys-ai diff before.sqlite after.sqlite \\\n"
            "  --gate exposed_comm:+10ms \\\n"
            "  --exit-on regression"
        )
        diag = self._minimal(verification_command=cmd)
        restored = Diagnostic.from_dict(json.loads(json.dumps(diag.to_dict())))
        assert restored.verification_command == cmd


class TestDiagnosticIO:
    """save_diagnostic / load_diagnostic round-trip (diagnostics.json, issue #207)."""

    def _diag(self) -> Diagnostic:
        f = Finding(
            type="region",
            label="GPU Idle Gap (18.25ms)",
            start_ns=10,
            end_ns=20,
            severity="warning",
            id="idle_gap_gpu0_stream7_10",
            category="idle",
            confidence=0.85,
            evidence=[
                EvidenceRow(
                    id="ev1",
                    source_skill="gpu_idle_gaps",
                    values={"gap_ms": 18.25},
                    units={"gap_ms": "ms"},
                )
            ],
        )
        return Diagnostic(
            id="diag_test",
            summary="S — unicode ✓",
            recommendation="Use pinned memory",
            verification_command="nsys-ai skill run gpu_idle_gaps p.sqlite --format json",
            confidence=0.75,
            primary_findings=[f],
            root_cause_hypotheses=["Pageable Memory in Async Memcpy"],
        )

    def test_save_load_round_trip(self, tmp_path):
        from nsys_ai.annotation import load_diagnostic, save_diagnostic

        diag = self._diag()
        out = tmp_path / "diagnostics.json"
        save_diagnostic(diag, str(out))
        assert load_diagnostic(str(out)) == diag

    def test_saved_json_matches_to_dict_shape(self, tmp_path):
        from nsys_ai.annotation import save_diagnostic

        diag = self._diag()
        out = tmp_path / "diagnostics.json"
        save_diagnostic(diag, str(out))
        raw = json.loads(out.read_text())
        assert raw == diag.to_dict()
        # No findings-report envelope keys leak into the diagnostics schema.
        assert "findings" not in raw
        assert "schema_version" not in raw
        assert "profile_path" not in raw

    def test_default_filename_constant(self):
        from nsys_ai.annotation import DEFAULT_DIAGNOSTICS_FILENAME

        assert DEFAULT_DIAGNOSTICS_FILENAME == "diagnostics.json"


class TestEvidenceRowEdgeCases:
    def test_nested_dict_in_values(self):
        row = EvidenceRow(
            id="r",
            source_skill="x",
            values={"nested": {"a": 1, "b": [2, 3]}},
        )
        restored = EvidenceRow.from_dict(json.loads(json.dumps(row.to_dict())))
        assert restored.values["nested"]["b"] == [2, 3]

    def test_unicode_in_values(self):
        row = EvidenceRow(
            id="r",
            source_skill="x",
            values={"label": "中文 — em-dash"},
        )
        s = json.dumps(row.to_dict(), ensure_ascii=False)
        restored = EvidenceRow.from_dict(json.loads(s))
        assert restored.values["label"] == "中文 — em-dash"

    def test_empty_dicts_preserved(self):
        """Empty dicts (default values) round-trip; only None is dropped."""
        row = EvidenceRow(id="r", source_skill="x")
        d = row.to_dict()
        assert d["values"] == {}
        assert d["units"] == {}
        assert d["provenance"] == {}
        assert "selection_id" not in d  # None → dropped

    def test_to_dict_mutation_safety(self):
        row = EvidenceRow(id="r", source_skill="x", values={"a": 1})
        d = row.to_dict()
        d["values"]["a"] = 999
        assert row.values["a"] == 1  # original unchanged

    def test_from_dict_normalizes_null_dict_fields(self):
        """JSON null for values / units / provenance is coerced to {}.

        Preserves the dict invariant on these fields so callers never see
        ``None`` where a dict is expected.
        """
        row = EvidenceRow.from_dict(
            {
                "id": "r",
                "source_skill": "x",
                "values": None,
                "units": None,
                "provenance": None,
            }
        )
        assert row.values == {}
        assert row.units == {}
        assert row.provenance == {}

    def test_from_dict_null_dict_fields_round_trip(self):
        """A payload with null dict-fields normalizes and round-trips cleanly."""
        row = EvidenceRow.from_dict({"id": "r", "source_skill": "x", "values": None})
        restored = EvidenceRow.from_dict(json.loads(json.dumps(row.to_dict())))
        assert restored == row
        assert restored.values == {}

    def test_from_dict_mixed_null_and_present_dict_fields(self):
        """Only null dict fields are normalized; present ones are preserved."""
        row = EvidenceRow.from_dict(
            {
                "id": "r",
                "source_skill": "x",
                "values": {"a": 1},
                "units": None,
                "provenance": {"trim_start_ns": 0},
            }
        )
        assert row.values == {"a": 1}
        assert row.units == {}
        assert row.provenance == {"trim_start_ns": 0}


class TestTraceSelectionEdgeCases:
    def test_all_lists_empty_distinct_from_none(self):
        sel = TraceSelection(
            id="s",
            profile_id="x",
            source="user",
            gpu_ids=[],
            rank_ids=[],
            stream_ids=[],
            nvtx_path=[],
            event_ids=[],
        )
        d = sel.to_dict()
        assert d["gpu_ids"] == []
        assert d["rank_ids"] == []
        assert d["stream_ids"] == []
        assert d["nvtx_path"] == []
        assert d["event_ids"] == []
        restored = TraceSelection.from_dict(d)
        assert restored == sel

    def test_zero_timestamps_preserved(self):
        """start_ns=0 is meaningful (not None) and must round-trip."""
        sel = TraceSelection(
            id="s",
            profile_id="x",
            source="user",
            start_ns=0,
            end_ns=0,
        )
        d = sel.to_dict()
        assert d["start_ns"] == 0
        assert d["end_ns"] == 0
        restored = TraceSelection.from_dict(d)
        assert restored.start_ns == 0
        assert restored.end_ns == 0

    def test_to_dict_mutation_safety(self):
        sel = TraceSelection(id="s", profile_id="x", source="user", gpu_ids=[0, 1])
        d = sel.to_dict()
        d["gpu_ids"].append(99)
        assert sel.gpu_ids == [0, 1]

    def test_unicode_label(self):
        sel = TraceSelection(id="s", profile_id="x", source="user", label="iteration_142 / 反向")
        s = json.dumps(sel.to_dict(), ensure_ascii=False)
        restored = TraceSelection.from_dict(json.loads(s))
        assert restored.label == "iteration_142 / 反向"


class TestDiffLineageEdgeCases:
    def test_rank_zero(self):
        lin = DiffLineage(diff_id="d", role="regression", rank=0, baseline_profile_id="b")
        restored = DiffLineage.from_dict(lin.to_dict())
        assert restored.rank == 0

    def test_large_rank(self):
        lin = DiffLineage(diff_id="d", role="improvement", rank=999_999, baseline_profile_id="b")
        restored = DiffLineage.from_dict(lin.to_dict())
        assert restored.rank == 999_999

    def test_all_three_roles_round_trip(self):
        for role in ("regression", "improvement", "stable"):
            lin = DiffLineage(diff_id="d", role=role, rank=0, baseline_profile_id="b")
            restored = DiffLineage.from_dict(json.loads(json.dumps(lin.to_dict())))
            assert restored.role == role


class TestModuleExports:
    """All v0.1 schema models are importable from nsys_ai.annotation."""

    def test_imports(self):
        from nsys_ai.annotation import (
            PRODUCER,
            SCHEMA_VERSION,
            Diagnostic,
            DiffLineage,
            EvidenceReport,
            EvidenceRow,
            Finding,
            FindingCategory,
            TraceSelection,
            load_findings,
            save_findings,
        )

        # Touch each binding so a future accidental removal fails this test.
        assert Diagnostic is not None
        assert DiffLineage is not None
        assert EvidenceReport is not None
        assert EvidenceRow is not None
        assert Finding is not None
        assert FindingCategory is not None
        assert TraceSelection is not None
        assert callable(load_findings)
        assert callable(save_findings)
        assert isinstance(SCHEMA_VERSION, str) and SCHEMA_VERSION
        assert PRODUCER == "nsys-ai"


class TestFindingV01Fields:
    """The additive v0.1 optional fields on Finding."""

    def _legacy(self, **overrides) -> Finding:
        kwargs = dict(type="region", label="L", start_ns=0)
        kwargs.update(overrides)
        return Finding(**kwargs)

    def test_legacy_construction_still_works(self):
        """No new field is required; pre-v0.1 callers see no behavior change."""
        f = self._legacy()
        assert f.id is None
        assert f.category is None
        assert f.confidence is None
        assert f.evidence is None
        assert f.selection is None
        assert f.explanation is None
        assert f.suggested_actions is None
        assert f.false_positive_notes is None
        assert f.provenance is None
        assert f.diff_lineage is None

    def test_legacy_to_dict_drops_all_new_fields(self):
        """Default Finding produces the same dict shape as before v0.1."""
        d = self._legacy().to_dict()
        # New fields, all None by default, must not appear.
        for k in (
            "id",
            "category",
            "confidence",
            "evidence",
            "selection",
            "explanation",
            "suggested_actions",
            "false_positive_notes",
            "provenance",
            "diff_lineage",
        ):
            assert k not in d, f"new field {k} leaked into legacy to_dict"

    def test_scalar_new_fields_round_trip(self):
        f = self._legacy(
            id="f0",
            category="idle",
            confidence=0.83,
            explanation="GPU stalls after backward pass.",
            suggested_actions=["inspect DDP overlap"],
            false_positive_notes=["short inference may be fine"],
            provenance={"trim_start_ns": 0},
        )
        restored = Finding.from_dict(f.to_dict())
        assert restored == f

    def test_nested_evidence_round_trip(self):
        rows = [
            EvidenceRow(
                id="r0",
                source_skill="gpu_idle_gaps",
                values={"gap_ms": 12.5},
                units={"gap_ms": "ms"},
            ),
            EvidenceRow(id="r1", source_skill="overlap_breakdown"),
        ]
        f = self._legacy(id="f1", category="idle", evidence=rows)
        restored = Finding.from_dict(f.to_dict())
        assert restored.evidence == rows
        # Nested type is rehydrated, not a list of dicts.
        assert isinstance(restored.evidence[0], EvidenceRow)

    def test_nested_selection_round_trip(self):
        sel = TraceSelection(
            id="sel_0",
            profile_id="abc",
            source="skill:gpu_idle_gaps",
            start_ns=10,
            end_ns=20,
            gpu_ids=[0],
        )
        f = self._legacy(id="f2", selection=sel)
        restored = Finding.from_dict(f.to_dict())
        assert restored.selection == sel
        assert isinstance(restored.selection, TraceSelection)

    def test_nested_diff_lineage_round_trip(self):
        lin = DiffLineage(
            diff_id="diff_x",
            role="regression",
            rank=1,
            baseline_profile_id="base_v1",
        )
        f = self._legacy(id="f3", diff_lineage=lin)
        restored = Finding.from_dict(f.to_dict())
        assert restored.diff_lineage == lin
        assert isinstance(restored.diff_lineage, DiffLineage)

    def test_full_v01_finding_json_round_trip(self):
        rows = [EvidenceRow(id="r0", source_skill="overlap_breakdown")]
        sel = TraceSelection(id="sel", profile_id="p", source="diff", start_ns=1)
        lin = DiffLineage(diff_id="d", role="improvement", rank=0, baseline_profile_id="b")
        f = self._legacy(
            id="f",
            category="communication",
            confidence=0.9,
            evidence=rows,
            selection=sel,
            explanation="exposed NCCL improved",
            suggested_actions=["keep change"],
            false_positive_notes=["small N"],
            provenance={"trim_start_ns": 0},
            diff_lineage=lin,
        )
        restored = Finding.from_dict(json.loads(json.dumps(f.to_dict())))
        assert restored == f

    def test_legacy_json_loads_with_defaults(self):
        """A pre-v0.1 JSON payload (no new keys) loads cleanly."""
        legacy = {
            "type": "region",
            "label": "L",
            "start_ns": 100,
            "end_ns": 200,
            "severity": "warning",
            "note": "old payload",
        }
        f = Finding.from_dict(legacy)
        assert f.label == "L"
        assert f.severity == "warning"
        # New fields keep their None defaults.
        assert f.id is None
        assert f.category is None
        assert f.evidence is None

    def test_nested_to_dict_drops_inner_none(self):
        """Nested EvidenceRow.to_dict drops None inside the Finding payload."""
        rows = [EvidenceRow(id="r", source_skill="x")]  # selection_id is None
        f = self._legacy(id="f", evidence=rows)
        d = f.to_dict()
        assert d["evidence"][0]["id"] == "r"
        # selection_id was None on the row → must not leak into JSON output.
        assert "selection_id" not in d["evidence"][0]

    def test_to_dict_mutation_safety_for_mutable_fields(self):
        """Mutating mutable containers in to_dict output must not mutate the source."""
        f = self._legacy(
            id="f",
            provenance={"a": 1},
            suggested_actions=["action_a"],
            false_positive_notes=["note_a"],
        )
        d = f.to_dict()
        d["provenance"]["a"] = 999
        d["suggested_actions"].append("mutated")
        d["false_positive_notes"].clear()
        assert f.provenance == {"a": 1}
        assert f.suggested_actions == ["action_a"]
        assert f.false_positive_notes == ["note_a"]

    def test_to_dict_avoids_wasted_nested_materialization(self, monkeypatch):
        """Refactored to_dict reads nested fields once via their own to_dict.

        Regression guard for the perf fix that replaced ``asdict(self)`` with
        a hand-rolled ``fields()`` walk: confirms each nested type's
        ``to_dict`` is invoked exactly once for a Finding with one of each
        nested kind populated.

        Implemented with ``monkeypatch`` + manual call counters rather than
        ``unittest.mock.patch.object(autospec=True, wraps=...)`` because the
        ``autospec + wraps`` combination behaves inconsistently across
        Python 3.10 / 3.11 / 3.12 mock implementations.
        """
        rows = [EvidenceRow(id="r0", source_skill="x")]
        sel = TraceSelection(id="s", profile_id="p", source="diff")
        lin = DiffLineage(diff_id="d", role="stable", rank=0, baseline_profile_id="b")
        f = self._legacy(evidence=rows, selection=sel, diff_lineage=lin)

        calls = {"row": 0, "sel": 0, "lin": 0}

        def _counted(key: str, original):
            def wrapper(inner_self):
                calls[key] += 1
                return original(inner_self)

            return wrapper

        monkeypatch.setattr(EvidenceRow, "to_dict", _counted("row", EvidenceRow.to_dict))
        monkeypatch.setattr(TraceSelection, "to_dict", _counted("sel", TraceSelection.to_dict))
        monkeypatch.setattr(DiffLineage, "to_dict", _counted("lin", DiffLineage.to_dict))

        d = f.to_dict()

        assert calls == {"row": 1, "sel": 1, "lin": 1}
        assert isinstance(d["evidence"][0], dict)
        assert isinstance(d["selection"], dict)
        assert isinstance(d["diff_lineage"], dict)


class TestEvidenceReportEnvelope:
    """v0.1 envelope on EvidenceReport JSON output."""

    def test_to_dict_contains_envelope(self):
        report = EvidenceReport(title="Test")
        d = report.to_dict()
        assert d["schema_version"] == "0.1"
        assert d["producer"] == "nsys-ai"
        assert isinstance(d["producer_version"], str) and d["producer_version"]

    def test_envelope_does_not_displace_legacy_fields(self):
        """Legacy keys (title, profile_path, findings) still appear unchanged."""
        report = EvidenceReport(title="T", profile_path="/p/x.sqlite")
        d = report.to_dict()
        assert d["title"] == "T"
        assert d["profile_path"] == "/p/x.sqlite"
        assert d["findings"] == []
        # ``profile_id`` is additive — always present (empty default when
        # not supplied), so consumers can rely on the key existing.
        assert "profile_id" in d
        assert d["profile_id"] == ""

    def test_envelope_carries_profile_id(self):
        """When set, profile_id is serialized verbatim in the envelope."""
        report = EvidenceReport(title="T", profile_id="nsys1:sha256:abc", profile_path="/p")
        d = report.to_dict()
        assert d["profile_id"] == "nsys1:sha256:abc"

    def test_from_dict_preserves_profile_id(self):
        restored = EvidenceReport.from_dict(
            {"title": "T", "profile_id": "nsys1:sha256:deadbeef", "profile_path": "/p"}
        )
        assert restored.profile_id == "nsys1:sha256:deadbeef"

    def test_from_dict_defaults_profile_id_when_missing(self):
        """Pre-profile_id payloads (no key) load with profile_id=''."""
        restored = EvidenceReport.from_dict({"title": "T", "profile_path": "/p"})
        assert restored.profile_id == ""

    def test_post_init_coerces_pathlib_profile_path(self):
        """``EvidenceReport.profile_path`` is typed ``str``, but callers
        sometimes hand in ``pathlib.Path``. ``__post_init__`` coerces so
        downstream JSON serialization doesn't choke on PosixPath.
        """
        import json
        from pathlib import Path

        report = EvidenceReport(title="T", profile_path=Path("/x.sqlite"))
        assert isinstance(report.profile_path, str)
        assert report.profile_path == "/x.sqlite"
        # to_dict + json.dumps must work end-to-end
        json.dumps(report.to_dict())

    def test_post_init_leaves_empty_profile_path_alone(self):
        """Default empty ``profile_path`` must stay an empty string after
        ``__post_init__`` — don't crash, don't replace with ``"."``."""
        report = EvidenceReport(title="T")
        assert report.profile_path == ""

    def test_profile_id_is_keyword_only(self):
        """Regression for Copilot review: ``profile_id`` was originally
        inserted before ``profile_path`` in the dataclass fields, which
        silently broke positional callers (``EvidenceReport("T", "/p")``
        would have bound ``/p`` to ``profile_id`` instead of
        ``profile_path``). Marking it ``kw_only=True`` preserves the
        positional signature and forces explicit naming.
        """
        import pytest

        # Positional binding of ``profile_id`` is forbidden:
        with pytest.raises(TypeError):
            EvidenceReport("T", "/p", [], "nsys1:sha256:abc")  # type: ignore[misc]

        # Old positional usage (title, profile_path[, findings]) still
        # binds to the same fields it always did.
        report = EvidenceReport("T", "/some/profile.sqlite")
        assert report.title == "T"
        assert report.profile_path == "/some/profile.sqlite"
        assert report.profile_id == ""  # default kept

    def test_from_dict_accepts_v01_envelope(self):
        """Round-trip through to_dict / from_dict preserves the report."""
        f = Finding(type="region", label="L", start_ns=10)
        report = EvidenceReport(title="T", profile_path="/p", findings=[f])
        restored = EvidenceReport.from_dict(report.to_dict())
        assert restored.title == "T"
        assert restored.profile_path == "/p"
        assert len(restored.findings) == 1
        assert restored.findings[0].label == "L"

    def test_from_dict_accepts_legacy_payload(self):
        """A pre-envelope payload (no schema_version) loads correctly."""
        legacy = {
            "title": "Legacy",
            "profile_path": "/x.sqlite",
            "findings": [{"type": "region", "label": "L", "start_ns": 0, "severity": "info"}],
        }
        report = EvidenceReport.from_dict(legacy)
        assert report.title == "Legacy"
        assert len(report.findings) == 1
        assert report.findings[0].label == "L"

    def test_from_dict_ignores_envelope_metadata(self):
        """Envelope fields are informational; from_dict drops them silently."""
        payload = {
            "schema_version": "9.9",  # any value, even unknown
            "producer": "ghostwriter",
            "producer_version": "0.0.0",
            "title": "T",
            "profile_path": "",
            "findings": [],
        }
        report = EvidenceReport.from_dict(payload)
        assert report.title == "T"
        assert report.findings == []

    def test_json_round_trip_with_v01_findings(self):
        rows = [EvidenceRow(id="r", source_skill="x")]
        sel = TraceSelection(id="s", profile_id="p", source="diff")
        f = Finding(
            type="region",
            label="L",
            start_ns=10,
            id="f",
            category="communication",
            evidence=rows,
            selection=sel,
        )
        report = EvidenceReport(title="T", findings=[f])
        s = json.dumps(report.to_dict(), indent=2)
        restored = EvidenceReport.from_dict(json.loads(s))
        assert restored.title == "T"
        assert restored.findings[0].id == "f"
        assert restored.findings[0].evidence[0].id == "r"
        assert restored.findings[0].selection.profile_id == "p"

    def test_save_load_round_trip(self, tmp_path):
        from nsys_ai.annotation import load_findings, save_findings

        f = Finding(type="region", label="L", start_ns=10, id="f0", category="idle")
        report = EvidenceReport(title="Persist", findings=[f])
        path = tmp_path / "findings.json"
        save_findings(report, str(path))
        loaded = load_findings(str(path))
        assert loaded.title == "Persist"
        assert loaded.findings[0].id == "f0"
        assert loaded.findings[0].category == "idle"
