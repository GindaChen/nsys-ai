"""Regression tests for iteration-noise classification.

A loose NVTX marker substring can match many short op-level ranges in addition
to the true iteration boundary. Those tiny ranges contaminated the median, so
every real iteration looked thousands of percent slow. `_flag_real_iterations`
separates the real iterations from the sub-ms tail, and the timing/detail
skills compute variance over the real rows only.
"""

from nsys_ai.overlap import _flag_real_iterations
from nsys_ai.skills.builtins.iteration_timing import _format, _to_findings


def _row(dur_ms, **extra):
    row = {
        "iteration": extra.get("iteration", 0),
        "duration_ms": dur_ms,
        "gpu_start_ns": extra.get("gpu_start_ns", 0),
        "gpu_end_ns": extra.get("gpu_end_ns", int(dur_ms * 1e6)),
        "kernel_count": extra.get("kernel_count", 1),
        "nccl_count": 0,
        "compute_ms": extra.get("compute_ms", dur_ms),
    }
    row.update(extra)
    return row


def test_flag_real_iterations_splits_bimodal():
    """16 real iterations (~500ms) + 968 sub-ms op markers → only 16 are real."""
    rows = [_row(500.0, iteration=i) for i in range(16)]
    rows += [_row(0.3, iteration=100 + i) for i in range(968)]
    _flag_real_iterations(rows)

    real = [r for r in rows if r["is_real_iteration"]]
    assert len(real) == 16, f"expected 16 real iterations, got {len(real)}"
    assert all(r["duration_ms"] == 500.0 for r in real)


def test_flag_real_iterations_drops_low_compute_ghost():
    """A heuristic-fallback 'ghost' — a huge wall-clock span with almost no GPU
    work — must NOT count as a real iteration, even though its duration is large.
    Classification is by compute, not duration."""
    rows = [_row(500.0, iteration=i, compute_ms=480.0) for i in range(8)]
    # Ghost: longest duration of all, but negligible compute (a long idle gap).
    rows.append(_row(9000.0, iteration=99, compute_ms=0.6, kernel_count=2))
    _flag_real_iterations(rows)

    ghost = next(r for r in rows if r["iteration"] == 99)
    assert ghost["is_real_iteration"] is False, "ghost (high span, ~0 compute) must be dropped"
    assert all(r["is_real_iteration"] for r in rows if r["iteration"] != 99)


def test_flag_real_iterations_keeps_uniform_all_real():
    """A clean profile (no tiny tail) must keep every iteration real."""
    rows = [_row(500.0, iteration=i) for i in range(16)]
    # Normal variance (one slow iteration) must not be misread as noise.
    rows.append(_row(900.0, iteration=99))
    _flag_real_iterations(rows)
    assert all(r["is_real_iteration"] for r in rows)


def test_flag_real_iterations_empty_and_zero_duration():
    assert _flag_real_iterations([]) == []
    rows = [_row(0.0), _row(0.0)]
    _flag_real_iterations(rows)
    assert all(r["is_real_iteration"] for r in rows)  # can't distinguish → all real


def test_findings_ignore_noise_and_use_real_median():
    """With 968 noise rows, the contaminated median used to flag every real
    iteration as ~+390,000% slow. After the fix, findings come only from real
    iterations judged against the real median."""
    rows = [_row(500.0, iteration=i) for i in range(15)]
    rows.append(_row(900.0, iteration=15))  # a genuinely slow real iteration
    rows += [_row(0.3, iteration=100 + i) for i in range(968)]
    _flag_real_iterations(rows)

    findings = _to_findings(rows)
    # Exactly one slow-iteration finding: the 900ms one (> 1.5 * median 500).
    assert len(findings) == 1, f"expected 1 finding, got {len(findings)}: {[f.label for f in findings]}"
    f = findings[0]
    assert "Slow Iteration 15" in f.label
    # The note must reference the real median (~500ms), not a sub-ms noise value.
    assert "median 500" in f.note, f"expected real median in note, got: {f.note!r}"
    # And no absurd thousands-of-percent figure from a contaminated median.
    assert "390" not in f.note


def test_format_shows_real_iterations_and_filtered_count():
    """The text view lists real iterations and reports the noise it hid, instead
    of dumping hundreds of sub-iteration ranges."""
    rows = [_row(500.0, iteration=i) for i in range(16)]
    rows += [_row(0.3, iteration=100 + i) for i in range(968)]
    _flag_real_iterations(rows)

    text = _format(rows)
    # 16 real iteration lines, not 984.
    iter_lines = [ln for ln in text.splitlines() if ln.strip().startswith("iter ")]
    assert len(iter_lines) == 16, f"expected 16 iter lines, got {len(iter_lines)}"
    assert "968 sub-iteration NVTX range(s) filtered as noise" in text
    # The average must reflect real iterations (~500ms), not the noise.
    assert "Average: 500.0ms" in text


def test_findings_empty_when_fewer_than_three_real():
    """Two real iterations buried in noise → not enough real data to judge."""
    rows = [_row(500.0, iteration=0), _row(500.0, iteration=1)]
    rows += [_row(0.3, iteration=100 + i) for i in range(50)]
    _flag_real_iterations(rows)
    assert _to_findings(rows) == []


# ── Heuristic-fallback phantom iterations (issue #215) ──────────────────────
# When no user NVTX iteration marker matches, detect_iterations falls back to
# synthesising segments from inter-kernel gaps. On a non-looping capture those
# are pauses, not iterations, and must not drive a confident count, variance
# verdict, or slow-iteration finding.


def test_to_findings_ignores_heuristic_iterations():
    """Heuristic gap-segments produce no findings even when they would survive
    the compute-noise filter. Uniform compute (so is_real_iteration keeps them
    all) plus one genuinely-slower segment: the ONLY thing suppressing the
    finding is the heuristic flag, so this isolates it from the compute filter."""
    rows = [_row(500.0, iteration=i, compute_ms=480.0, heuristic=True) for i in range(6)]
    rows.append(_row(1500.0, iteration=99, compute_ms=1450.0, heuristic=True))
    _flag_real_iterations(rows)
    assert all(r["is_real_iteration"] for r in rows), "compute filter must keep them"
    assert _to_findings(rows) == [], "but heuristic detection must yield no findings"


def test_to_findings_still_fires_for_nvtx_iterations():
    """The NVTX-marker path (heuristic absent/False) is unchanged: a genuine
    slow iteration among uniform ones still yields a finding."""
    rows = [_row(500.0, iteration=i) for i in range(6)]  # no heuristic key -> False
    rows.append(_row(1500.0, iteration=99))
    _flag_real_iterations(rows)
    finds = _to_findings(rows)
    assert finds, "a real slow NVTX iteration must still be reported"
    assert any("Slow Iteration" in f.label for f in finds)


def test_manifest_summary_drops_heuristic_iterations():
    from nsys_ai.skills.builtins.profile_health_manifest import _summarize_iterations

    rows = [_row(2.0, iteration=i, heuristic=True) for i in range(20)]
    rows.append(_row(5000.0, iteration=99, heuristic=True))
    _flag_real_iterations(rows)
    # No count, no median, no slowest -> the variance-spike verdict cannot fire.
    assert _summarize_iterations(rows) == {}


def test_manifest_summary_counts_nvtx_iterations():
    from nsys_ai.skills.builtins.profile_health_manifest import _summarize_iterations

    rows = [_row(500.0, iteration=i) for i in range(5)]  # NVTX-based (heuristic False)
    rows.append(_row(900.0, iteration=99))
    _flag_real_iterations(rows)
    summary = _summarize_iterations(rows)
    assert summary["iteration_count"] == 6
    assert summary["median_iter_ms"] == 500.0  # warm-up (iter 0) skipped
    assert summary["slowest_iter_ms"] == 900.0


def test_detect_iterations_flags_source(minimal_nsys_conn):
    """detect_iterations tags rows heuristic=False when a user NVTX marker
    matched, True when it fell back to gap-segments — the flag the consumers
    above rely on."""
    from nsys_ai.overlap import detect_iterations
    from nsys_ai.profile import Profile

    prof = Profile._from_conn(minimal_nsys_conn)
    matched = detect_iterations(prof, 0, marker="train_step")
    assert matched and all(r.get("heuristic") is False for r in matched)

    fallback = detect_iterations(prof, 0, marker="zzz_no_such_marker_zzz")
    assert fallback and all(r.get("heuristic") is True for r in fallback)
