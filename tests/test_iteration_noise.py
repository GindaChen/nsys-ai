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
