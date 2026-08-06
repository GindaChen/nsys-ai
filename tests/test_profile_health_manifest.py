"""Tests for profile_health_manifest skill and --max-rows truncation.

Uses the minimal_nsys_conn and duckdb_conn fixtures from conftest.py.
"""

import pytest

from nsys_ai.skills.builtins.profile_health_manifest import (
    _AUTO_TRIM_MIN_WINDOW_NS,
    _AUTO_TRIM_TARGET_WINDOW_NS,
    _auto_select_trim_window,
)
from nsys_ai.skills.registry import get_skill

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def manifest_skill():
    skill = get_skill("profile_health_manifest")
    assert skill is not None, "profile_health_manifest skill not registered"
    return skill


# ── Manifest skill tests ─────────────────────────────────────────


class TestManifestExecute:
    def test_returns_single_row(self, minimal_nsys_conn, manifest_skill):
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        assert isinstance(rows, list)
        assert len(rows) == 1

    def test_has_required_keys(self, minimal_nsys_conn, manifest_skill):
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        m = rows[0]
        required_keys = [
            "gpu",
            "profile_span_ms",
            "top_kernels",
            "total_kernel_ms",
            "overlap",
            "nccl",
            "communicators",
            "idle",
            "root_cause_count",
            "root_causes",
            "data_quality",
        ]
        for key in required_keys:
            assert key in m, f"Missing key '{key}' in manifest"

    def test_top_kernels_is_list(self, minimal_nsys_conn, manifest_skill):
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        m = rows[0]
        assert isinstance(m["top_kernels"], list)

    def test_top_kernels_carry_dominant_convention_aliases(
        self, minimal_nsys_conn, manifest_skill
    ):
        """Manifest top_kernels expose kernel_name/invocations (the convention
        used by the top_kernels skill / planner / tool_dispatch) in addition to
        the legacy name/count, additively."""
        m = manifest_skill.execute(minimal_nsys_conn, device=0)[0]
        for k in m["top_kernels"]:
            assert k["kernel_name"] == k["name"]
            assert k["invocations"] == k["count"]

    def test_nccl_summary_has_streams(self, minimal_nsys_conn, manifest_skill):
        """Our test data has NCCL kernels, so streams > 0."""
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        m = rows[0]
        assert m["nccl"]["streams"] > 0

    def test_communicator_summary_defaults_without_payloads(self, minimal_nsys_conn, manifest_skill):
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        m = rows[0]
        assert m["communicators"]["communicators"] == 0
        assert m["communicators"]["collective_rows"] == 0

    def test_root_causes_capped(self, minimal_nsys_conn, manifest_skill):
        """root_causes list should never exceed 5 entries."""
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        m = rows[0]
        assert len(m["root_causes"]) <= 5

    def test_empty_device_returns_manifest(self, minimal_nsys_conn, manifest_skill):
        """Device with no data should still return a valid manifest."""
        rows = manifest_skill.execute(minimal_nsys_conn, device=99)
        assert isinstance(rows, list)
        assert len(rows) == 1
        m = rows[0]
        assert m["nccl"]["streams"] == 0

    def test_data_quality_metrics(self, minimal_nsys_conn, manifest_skill):
        """Ensure data_quality metrics are properly computed."""
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        dq = rows[0]["data_quality"]

        assert "profiler_overhead_ms" in dq
        assert "overhead_pct" in dq

        assert isinstance(dq["profiler_overhead_ms"], (int, float))
        assert isinstance(dq["overhead_pct"], (int, float))


class TestManifestAutoTrim:
    """Auto-trim picks a representative window on long profiles so the
    manifest doesn't grind through 10-minute / 2 GB exports.

    Boundary: profiles shorter than _AUTO_TRIM_PROFILE_SPAN_THRESHOLD_NS
    pass through untouched; longer profiles get the middle steady-state
    instance of the top non-aten:: NVTX range, or the middle 20 s as a
    fallback when no NVTX iteration markers are present.
    """

    @staticmethod
    def _make_profile_stub(span_ns: int, nvtx_rows: list[tuple] | None = None):
        """Lightweight Profile stub sufficient for `_auto_select_trim_window`.

        ``nvtx_rows`` is the post-filter set (i.e. these rows survive the
        CTE's ``NOT LIKE 'aten::%'`` exclusion). ``None`` means "no NVTX
        iteration markers" — the selector should hit the middle-of-span
        fallback path.
        """
        from types import SimpleNamespace

        rows = list(nvtx_rows or [])

        def fake_query(sql, params=None):
            normalized = " ".join(sql.split()).lower()
            # Probe queries — bare `SELECT 1 FROM <view> LIMIT 1` — say "yes".
            if normalized.startswith("select 1 from") and normalized.endswith("limit 1"):
                return [{"1": 1}]
            # The CTE query is the only other shape this function emits.
            if "with ranged" in normalized and rows:
                return [{"text": r[0], "start": r[1], "end": r[2]} for r in rows]
            return []

        return SimpleNamespace(
            meta=SimpleNamespace(time_range=(0, span_ns)),
            _duckdb_query=fake_query,
        )

    def test_short_profile_returns_none(self):
        """Profile span below threshold → no auto-trim."""
        prof = self._make_profile_stub(span_ns=30 * 10**9)
        assert _auto_select_trim_window(prof) is None

    def test_long_profile_picks_middle_nvtx_instance(self):
        """3 stage instances → middle one (skip JIT warmup at idx 0)."""
        # Three short ranges (each < 20 s) so the selector returns them
        # verbatim rather than slicing to a 20 s sub-window.
        nvtx_rows = [
            # (text, start_ns, end_ns)
            ("stage::DenoisingStage", 100_000_000_000, 110_000_000_000),
            ("stage::DenoisingStage", 200_000_000_000, 210_000_000_000),
            ("stage::DenoisingStage", 300_000_000_000, 310_000_000_000),
        ]
        prof = self._make_profile_stub(span_ns=500 * 10**9, nvtx_rows=nvtx_rows)
        picked = _auto_select_trim_window(prof)
        # Index 1 (the middle of 3, skipping idx 0)
        assert picked == (200_000_000_000, 210_000_000_000)

    def test_long_profile_trims_wide_range_to_middle_20s(self):
        """Range wider than 20 s gets narrowed to its middle 20 s."""
        # One 200 s range × 3 instances; middle instance spans
        # 400 s → 600 s. Auto-trim should return middle ± 10 s = 490-510 s.
        nvtx_rows = [
            ("stage::DenoisingStage", 100_000_000_000, 300_000_000_000),
            ("stage::DenoisingStage", 400_000_000_000, 600_000_000_000),
            ("stage::DenoisingStage", 700_000_000_000, 900_000_000_000),
        ]
        prof = self._make_profile_stub(span_ns=1000 * 10**9, nvtx_rows=nvtx_rows)
        picked = _auto_select_trim_window(prof)
        assert picked is not None
        lo, hi = picked
        assert hi - lo == _AUTO_TRIM_TARGET_WINDOW_NS
        # Centered on the middle of (400e9, 600e9) = 500e9
        assert lo == 500_000_000_000 - _AUTO_TRIM_TARGET_WINDOW_NS // 2
        assert hi == 500_000_000_000 + _AUTO_TRIM_TARGET_WINDOW_NS // 2

    def test_long_profile_no_nvtx_falls_back_to_middle(self):
        """No qualifying NVTX iteration markers → middle 20 s of span."""
        prof = self._make_profile_stub(span_ns=400 * 10**9)
        picked = _auto_select_trim_window(prof)
        assert picked is not None
        lo, hi = picked
        assert hi - lo == _AUTO_TRIM_TARGET_WINDOW_NS
        # Span midpoint is 200 s; window is [190 s, 210 s]
        assert lo == 200 * 10**9 - _AUTO_TRIM_TARGET_WINDOW_NS // 2
        assert hi == 200 * 10**9 + _AUTO_TRIM_TARGET_WINDOW_NS // 2

    def test_narrow_nvtx_iteration_falls_back_to_middle(self):
        """Sub-floor NVTX picks (e.g. per-op markers) must not produce a
        microscopic trim window — downstream ratios would be undefined.

        Pre-fix behaviour: a 9 µs window was selected on a single-iteration
        H100 capture, which made ``overhead_pct`` blow up to ~3.5 M %.
        Post-fix: anything under ``_AUTO_TRIM_MIN_WINDOW_NS`` is rejected
        and the function falls through to the middle-of-span fallback.
        """
        narrow = _AUTO_TRIM_MIN_WINDOW_NS // 100  # well below the floor
        nvtx_rows = [
            ("op::tiny_marker", 100_000_000_000, 100_000_000_000 + narrow),
            ("op::tiny_marker", 200_000_000_000, 200_000_000_000 + narrow),
            ("op::tiny_marker", 300_000_000_000, 300_000_000_000 + narrow),
        ]
        prof = self._make_profile_stub(span_ns=400 * 10**9, nvtx_rows=nvtx_rows)
        picked = _auto_select_trim_window(prof)
        assert picked is not None
        lo, hi = picked
        # Middle-of-span fallback, not the narrow NVTX pick
        assert hi - lo == _AUTO_TRIM_TARGET_WINDOW_NS
        assert lo == 200 * 10**9 - _AUTO_TRIM_TARGET_WINDOW_NS // 2

    def test_iteration_exactly_at_min_width_accepted(self):
        """An iteration at exactly ``_AUTO_TRIM_MIN_WINDOW_NS`` wide is
        accepted (inclusive boundary). Pins the floor so a future refactor
        cannot silently turn it into a strict ``>``.
        """
        at_floor = _AUTO_TRIM_MIN_WINDOW_NS
        nvtx_rows = [
            ("stage::Step", 100_000_000_000, 100_000_000_000 + at_floor),
            ("stage::Step", 200_000_000_000, 200_000_000_000 + at_floor),
            ("stage::Step", 300_000_000_000, 300_000_000_000 + at_floor),
        ]
        prof = self._make_profile_stub(span_ns=400 * 10**9, nvtx_rows=nvtx_rows)
        picked = _auto_select_trim_window(prof)
        assert picked == (200_000_000_000, 200_000_000_000 + at_floor)

    def test_degenerate_time_range_returns_none(self):
        """None / inverted / zero-width time_range → no auto-trim."""
        from types import SimpleNamespace

        # Missing endpoint
        prof = SimpleNamespace(
            meta=SimpleNamespace(time_range=(None, None)),
            _duckdb_query=lambda *a, **k: [],
        )
        assert _auto_select_trim_window(prof) is None
        # Inverted endpoints
        prof = SimpleNamespace(
            meta=SimpleNamespace(time_range=(200, 100)),
            _duckdb_query=lambda *a, **k: [],
        )
        assert _auto_select_trim_window(prof) is None
        # meta missing entirely
        prof = SimpleNamespace(meta=SimpleNamespace())
        assert _auto_select_trim_window(prof) is None

    def test_explicit_trim_disables_auto_trim_on_long_profile(
        self, monkeypatch, minimal_nsys_conn, manifest_skill
    ):
        """Even when span is over threshold, explicit trim_start_ns /
        trim_end_ns from the caller must not be replaced by auto-trim."""
        # Lower the threshold so the minimal fixture profile counts as
        # "long" without us needing a real 2 GB sqlite. The point is to
        # exercise the explicit-trim guard, not the threshold itself.
        monkeypatch.setattr(
            "nsys_ai.skills.builtins.profile_health_manifest._AUTO_TRIM_PROFILE_SPAN_THRESHOLD_NS",
            0,
        )
        rows = manifest_skill.execute(
            minimal_nsys_conn,
            device=0,
            trim_start_ns=0,
            trim_end_ns=10_000_000,
        )
        assert "auto_trim" not in rows[0]["data_quality"]

    def test_short_profile_no_auto_trim_in_manifest(self, minimal_nsys_conn, manifest_skill):
        """Fixture profile spans ~24 ms — well below threshold; no auto-trim."""
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        assert "auto_trim" not in rows[0]["data_quality"]

    @pytest.mark.parametrize("token", ["0", "false", "False", "no", "OFF", " 0 "])
    def test_env_var_disables_auto_trim(
        self, token, monkeypatch, minimal_nsys_conn, manifest_skill
    ):
        """NSYS_AI_MANIFEST_AUTO_TRIM accepts the same off-tokens as the rest
        of the repo (0 / false / no / off, case-insensitive, trim-tolerant).
        """
        monkeypatch.setenv("NSYS_AI_MANIFEST_AUTO_TRIM", token)
        # Also lower the threshold so the fixture profile would normally
        # qualify for auto-trim — proves the env var is what stopped it.
        monkeypatch.setattr(
            "nsys_ai.skills.builtins.profile_health_manifest._AUTO_TRIM_PROFILE_SPAN_THRESHOLD_NS",
            0,
        )
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        assert "auto_trim" not in rows[0]["data_quality"]

    @pytest.mark.parametrize("token", ["1", "true", "yes", "on", ""])
    def test_env_var_truthy_keeps_auto_trim_enabled(
        self, token, monkeypatch, minimal_nsys_conn, manifest_skill
    ):
        """Any value that is NOT in the off-token set leaves auto-trim on."""
        monkeypatch.setenv("NSYS_AI_MANIFEST_AUTO_TRIM", token)
        monkeypatch.setattr(
            "nsys_ai.skills.builtins.profile_health_manifest._AUTO_TRIM_PROFILE_SPAN_THRESHOLD_NS",
            0,
        )
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        # auto_trim will be applied (threshold=0) — proves the env var
        # didn't switch it off.
        assert rows[0]["data_quality"].get("auto_trim", {}).get("applied") is True

    def test_auto_trim_applied_block_present_in_manifest(
        self, monkeypatch, minimal_nsys_conn, manifest_skill
    ):
        """Positive coverage: when the selector returns a window, the
        manifest emits a complete `data_quality.auto_trim` block.

        Stubs the selector so the test doesn't depend on the fixture
        actually having NVTX iteration markers — focuses on the
        `_execute()` plumbing.
        """
        fake_t0, fake_t1 = 1_000_000, 5_000_000  # 1 ms – 5 ms within fixture
        monkeypatch.setattr(
            "nsys_ai.skills.builtins.profile_health_manifest._auto_select_trim_window",
            lambda _prof: (fake_t0, fake_t1),
        )
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        dq = rows[0]["data_quality"]
        assert "auto_trim" in dq, "expected auto_trim block when selector returns a window"
        at = dq["auto_trim"]
        assert at["applied"] is True
        assert at["trim_start_ns"] == fake_t0
        assert at["trim_end_ns"] == fake_t1
        assert at["window_ms"] == round((fake_t1 - fake_t0) / 1e6, 1)
        # profile_full_span_ms must show the ORIGINAL profile length,
        # not the trim window — that's the whole point of carrying it.
        assert at["profile_full_span_ms"] >= at["window_ms"]
        # And the top-level profile_span_ms should reflect the window.
        assert rows[0]["profile_span_ms"] == at["window_ms"]

    def test_partial_trim_does_not_block_auto_trim(
        self, monkeypatch, minimal_nsys_conn, manifest_skill
    ):
        """Only one trim endpoint supplied → treat as no explicit trim.

        Otherwise long profiles would silently scan in full because the
        partial trim flag-set toggles `explicit_trim` to True but the
        sub-skills' filters need BOTH endpoints to fire.
        """
        monkeypatch.setattr(
            "nsys_ai.skills.builtins.profile_health_manifest._AUTO_TRIM_PROFILE_SPAN_THRESHOLD_NS",
            0,
        )
        # Only start, no end — partial trim
        rows = manifest_skill.execute(
            minimal_nsys_conn,
            device=0,
            trim_start_ns=500_000,
        )
        # Auto-trim should kick in because explicit_trim was False
        # (partial trim got discarded).
        assert rows[0]["data_quality"].get("auto_trim", {}).get("applied") is True

    def test_span_exactly_at_threshold_returns_none(self):
        """Boundary contract: span == threshold → no trim.

        Matches the docstring (≤ threshold returns None) and protects
        against the off-by-one between < / ≤ that PR review caught.
        """
        from nsys_ai.skills.builtins.profile_health_manifest import (
            _AUTO_TRIM_PROFILE_SPAN_THRESHOLD_NS,
        )

        prof = self._make_profile_stub(span_ns=_AUTO_TRIM_PROFILE_SPAN_THRESHOLD_NS)
        assert _auto_select_trim_window(prof) is None
        # And one ns over the threshold should trip the auto-trim path.
        prof = self._make_profile_stub(span_ns=_AUTO_TRIM_PROFILE_SPAN_THRESHOLD_NS + 1)
        assert _auto_select_trim_window(prof) is not None


class TestManifestFormat:
    def test_format_includes_header(self, minimal_nsys_conn, manifest_skill):
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        text = manifest_skill.format_rows(rows)
        assert "Profile Health Manifest" in text

    def test_format_includes_gpu(self, minimal_nsys_conn, manifest_skill):
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        text = manifest_skill.format_rows(rows)
        assert "GPU:" in text


class TestManifestDuckDB:
    def test_duckdb_path(self, duckdb_conn, manifest_skill):
        rows = manifest_skill.execute(duckdb_conn, device=0)
        assert isinstance(rows, list)
        assert len(rows) == 1
        assert "gpu" in rows[0]


# ── Token budget protection (--max-rows) ─────────────────────────


class TestMaxRowsTruncation:
    def test_truncation_applied(self):
        from nsys_ai.cli.handlers import _apply_max_rows_truncation

        rows = [{"id": i} for i in range(10)]
        max_rows = 3
        truncated = _apply_max_rows_truncation(rows, max_rows)
        assert len(truncated) == max_rows + 1
        assert truncated[-1]["_truncated"] is True
        assert truncated[-1]["_total_rows"] == 10
        assert truncated[-1]["_shown_rows"] == 3

    def test_no_truncation_when_under_limit(self):
        from nsys_ai.cli.handlers import _apply_max_rows_truncation

        rows = [{"id": i} for i in range(3)]
        truncated = _apply_max_rows_truncation(rows, 100)
        assert len(truncated) == 3
        assert not any(r.get("_truncated") for r in truncated)

    def test_negative_max_rows_raises(self):
        from nsys_ai.cli.handlers import _apply_max_rows_truncation

        rows = [{"id": i} for i in range(3)]
        with pytest.raises(ValueError, match="non-negative integer"):
            _apply_max_rows_truncation(rows, -1)

    def test_error_payload_not_truncated_when_max_rows_zero(self):
        """Error payloads (e.g., [{'error': ...}]) should not be dropped by max-rows."""
        from nsys_ai.cli.handlers import _apply_max_rows_truncation

        rows = [{"error": "Something went wrong"}]
        truncated = _apply_max_rows_truncation(rows, 0)

        # The single error row should be preserved and not replaced by a truncation marker.
        assert isinstance(truncated, list)
        assert len(truncated) == 1
        assert "error" in truncated[0]
        assert truncated[0]["error"] == "Something went wrong"
        # Error payloads should not be marked as truncated.
        assert not truncated[0].get("_truncated")


# ── compute_profiler_overhead_ns helper ──────────────────────────


class TestComputeProfilerOverheadNs:
    """Direct tests for the helper that computes overhead union duration.

    The helper is shared between ``Skill.execute`` (injects ``overhead_ns``
    for sub-skills) and ``profile_health_manifest._execute`` (re-aligns
    overhead to the effective window after auto-trim).
    """

    def _make_conn_with_overhead(self, intervals: list[tuple[int, int]]):
        """Build an in-memory sqlite with a profiler_overhead table only."""
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute(
            'CREATE TABLE profiler_overhead (start BIGINT NOT NULL, "end" BIGINT NOT NULL)'
        )
        conn.executemany(
            'INSERT INTO profiler_overhead (start, "end") VALUES (?, ?)',
            intervals,
        )
        return conn

    def test_returns_zero_when_table_absent(self, minimal_nsys_conn):
        """Profiles without profiler_overhead table → 0 (probe falls through)."""
        from nsys_ai.skills.base import compute_profiler_overhead_ns

        assert compute_profiler_overhead_ns(minimal_nsys_conn) == 0

    def test_returns_union_without_window(self):
        """Three disjoint intervals → sum of widths."""
        from nsys_ai.skills.base import compute_profiler_overhead_ns

        conn = self._make_conn_with_overhead(
            [(0, 100), (200, 350), (500, 600)]
        )
        assert compute_profiler_overhead_ns(conn) == 100 + 150 + 100

    def test_unions_overlapping_intervals(self):
        """Overlapping intervals collapse — the result is the union, not the sum."""
        from nsys_ai.skills.base import compute_profiler_overhead_ns

        conn = self._make_conn_with_overhead([(0, 200), (100, 300), (250, 400)])
        # [0,200] ∪ [100,300] ∪ [250,400] = [0,400] = 400 ns
        assert compute_profiler_overhead_ns(conn) == 400

    def test_clips_intervals_to_window(self):
        """Intervals straddling the window are clipped; outside-window dropped.

        This is the scenario PR-side realignment depends on: when manifest
        auto-trim narrows the window, the helper must report only the
        overhead falling inside that window.
        """
        from nsys_ai.skills.base import compute_profiler_overhead_ns

        # Three intervals: one entirely before the window, one straddling
        # the start edge, one entirely inside.
        conn = self._make_conn_with_overhead(
            [(0, 100), (450, 600), (700, 900)]
        )
        # Window [500, 800]:
        # - (0,100)   → outside, dropped
        # - (450,600) → clipped to (500,600) = 100 ns
        # - (700,900) → clipped to (700,800) = 100 ns
        # Union = 200 ns
        result = compute_profiler_overhead_ns(
            conn, trim_start_ns=500, trim_end_ns=800
        )
        assert result == 200

    def test_window_with_no_overlap_returns_zero(self):
        """All intervals outside the window → 0."""
        from nsys_ai.skills.base import compute_profiler_overhead_ns

        conn = self._make_conn_with_overhead([(0, 100), (200, 300)])
        assert compute_profiler_overhead_ns(
            conn, trim_start_ns=1000, trim_end_ns=2000
        ) == 0


# ── Overhead realignment on auto-trim ────────────────────────────


class TestManifestOverheadRealignment:
    """When auto-trim narrows the analysis window, the manifest must
    re-query ``overhead_ns`` aligned with that window so the resulting
    ratio is well-defined.

    Pre-fix behaviour: ``Skill.execute`` injected ``overhead_ns``
    computed over the full profile, ``_execute`` then narrowed the
    denominator via auto-trim, and the ratio could blow up to millions
    of percent. Post-fix: when ``auto_trim_meta`` is set, ``_execute``
    re-queries overhead clipped to ``effective_start_ns``/``effective_end_ns``.
    """

    def test_auto_trim_realigns_overhead_to_effective_window(
        self, monkeypatch, minimal_nsys_conn, manifest_skill
    ):
        """The realignment branch must override the initially-injected
        overhead with one recomputed against the auto-trim window.

        Strategy: pick an auto-trim window *inside* the fixture's profile
        range so the effective window doesn't collapse to zero after
        clamping (the fixture's time_range is roughly 1 ms - 9 ms). Then
        return *distinguishable* values from the helper for the
        ``(None, None)`` injection path vs the windowed realignment path
        so the final ``profiler_overhead_ms`` directly reveals which
        path produced it.
        """
        from nsys_ai.skills.builtins import profile_health_manifest as phm

        # Inside the fixture's profile range (1 ms - 9 ms); leaves headroom
        # on both sides so we can also verify the clamp doesn't shrink it.
        fake_t0 = 2_000_000
        fake_t1 = 8_000_000  # 6 ms wide
        monkeypatch.setattr(
            phm, "_auto_select_trim_window", lambda prof: (fake_t0, fake_t1)
        )

        windowed_calls: list[tuple[int, int]] = []

        def fake_compute(conn, *, trim_start_ns=None, trim_end_ns=None):
            if trim_start_ns is None and trim_end_ns is None:
                # Initial injection from Skill.execute — baseline value.
                return 100_000_000  # 100 ms
            # Sub-skill calls AND the realignment call share this branch;
            # record the realignment-shaped window so we can assert it.
            if (trim_start_ns, trim_end_ns) == (fake_t0, fake_t1):
                windowed_calls.append((trim_start_ns, trim_end_ns))
            return 1_000_000  # 1 ms — distinct from the baseline above

        from nsys_ai.skills import base as base_module

        monkeypatch.setattr(base_module, "compute_profiler_overhead_ns", fake_compute)

        rows = manifest_skill.execute(minimal_nsys_conn, device=0)

        # The helper must have been called with the auto-trim window at
        # least once (sub-skills + realignment all hit this shape).
        assert windowed_calls, (
            "compute_profiler_overhead_ns was never called with the auto-trim window"
        )

        # If realignment ran, the manifest's ``overhead_ns`` was overwritten
        # to the windowed return value (1 ms). If realignment had been
        # skipped — the bug shape — the value would still be 100 ms from
        # the initial injection.
        dq = rows[0].get("data_quality", {})
        assert dq.get("profiler_overhead_ms") == 1.0, (
            "realignment did not override the injected overhead_ns "
            f"(profiler_overhead_ms={dq.get('profiler_overhead_ms')})"
        )
        # Sanity: effective window had positive width, so the branch was reachable.
        assert rows[0].get("profile_span_ms", 0) > 0

    def test_auto_trim_window_outside_profile_range_skips_realignment(
        self, monkeypatch, minimal_nsys_conn, manifest_skill
    ):
        """When the auto-trim picker returns a window outside the actual
        profile time_range, the clamp collapses the effective span to 0
        and the realignment branch is correctly skipped — the injected
        overhead value flows through unchanged. Pins the guard so a
        zero-width window can't produce a division-by-zero or a stale
        realigned value.
        """
        from nsys_ai.skills.builtins import profile_health_manifest as phm

        # Far outside the fixture's range (1 ms - 9 ms) so the clamp
        # makes effective_end < effective_start.
        monkeypatch.setattr(
            phm,
            "_auto_select_trim_window",
            lambda prof: (1_000_000_000_000, 1_020_000_000_000),
        )

        def fake_compute(conn, *, trim_start_ns=None, trim_end_ns=None):
            if trim_start_ns is None and trim_end_ns is None:
                return 100_000_000  # 100 ms — what survives if realignment is skipped
            return 1_000_000  # 1 ms — what would appear if realignment fired

        from nsys_ai.skills import base as base_module

        monkeypatch.setattr(base_module, "compute_profiler_overhead_ns", fake_compute)

        rows = manifest_skill.execute(minimal_nsys_conn, device=0)

        # Effective window collapsed → realignment must be skipped, so
        # the initial injection value survives.
        assert rows[0].get("profile_span_ms") == 0
        dq = rows[0].get("data_quality", {})
        assert dq.get("profiler_overhead_ms") == 100.0, (
            "realignment branch must be skipped when effective span is 0 "
            f"(profiler_overhead_ms={dq.get('profiler_overhead_ms')})"
        )
