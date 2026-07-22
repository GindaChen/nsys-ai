"""End-to-end golden test for the optimization loop (issue #185).

One controlled before/after pair is run through the whole diff → verdict →
diff.json serialization, and the verdict, category attribution, and on-disk
record are frozen. A per-skill change that quietly breaks the diff or the
verdict record fails here rather than in production.

The pair is synthetic but exercises the real stages: overlap_analysis, the HTA
category attribution (overlap counts as compute, only exposed NCCL is comm),
export-schema comparability (#244), the verdict thresholds, and the diff.json
envelope. It mirrors the shape of a real regression — the "after" ran the same
compute but with more idle and more exposed communication, so step time grows.

Self-contained (no cross-file test imports) so it is robust to pytest rootdir.
"""

import json
import sqlite3

from nsys_ai.diff import diff_profiles
from nsys_ai.diff_render import to_diff_dict, to_diff_json
from nsys_ai.profile import Profile

_MS = 1_000_000
_STRINGS = {1: "gemm", 2: "gemm_dem", 10: "nccl_AllReduce", 11: "nccl_AllReduce_dem"}

# kernels: (start_ns, end_ns, deviceId, streamId, correlationId, shortName, demangledName)
# before: 2x 10ms compute on stream 7 with a 5ms gap; two 5ms NCCL collectives on
# stream 8 — one hidden behind compute ([5,10] overlaps the first GEMM), one
# exposed ([25,30], after compute). So: compute_only 15, overlap 5, nccl_only 5,
# idle 5. Under the HTA convention the hidden 5ms folds into compute (20ms
# compute category), only the exposed 5ms is communication.  step = 30ms.
_BEFORE_KERNELS = [
    (0, 10 * _MS, 0, 7, 1, 1, 2),
    (15 * _MS, 25 * _MS, 0, 7, 2, 1, 2),
    (5 * _MS, 10 * _MS, 0, 8, 3, 10, 11),
    (25 * _MS, 30 * _MS, 0, 8, 4, 10, 11),
]
# after: same compute and same hidden collective, but the gap widens (idle
# 5->10ms) and the exposed collective doubles (5->10ms) -> step 40ms, a ~33%
# regression. Compute category stays 20ms (still folds the 5ms overlap).
_AFTER_KERNELS = [
    (0, 10 * _MS, 0, 7, 1, 1, 2),
    (20 * _MS, 30 * _MS, 0, 7, 2, 1, 2),
    (5 * _MS, 10 * _MS, 0, 8, 3, 10, 11),
    (30 * _MS, 40 * _MS, 0, 8, 4, 10, 11),
]


def _make_profile(path, kernels):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE StringIds(id INT PRIMARY KEY, value TEXT)")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INT, [end] INT, deviceId INT, "
        "streamId INT, correlationId INT, shortName INT, demangledName INT)"
    )
    conn.execute("CREATE TABLE NVTX_EVENTS(text TEXT, globalTid INT, start INT, [end] INT)")
    conn.execute("CREATE TABLE META_DATA_EXPORT(name TEXT, value TEXT)")
    conn.executemany("INSERT INTO StringIds VALUES(?,?)", list(_STRINGS.items()))
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start, [end], deviceId, streamId, "
        "correlationId, shortName, demangledName) VALUES(?,?,?,?,?,?,?)",
        kernels,
    )
    conn.executemany(
        "INSERT INTO META_DATA_EXPORT VALUES(?,?)",
        [("EXPORT_SCHEMA_VERSION", "3.24.14"), ("EXPORT_PRODUCT_VERSION", "2026.1.1.204")],
    )
    conn.commit()
    conn.close()


def _run_loop(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    bp = tmp_path / "before.sqlite"
    ap = tmp_path / "after.sqlite"
    _make_profile(str(bp), _BEFORE_KERNELS)
    _make_profile(str(ap), _AFTER_KERNELS)
    with Profile(str(bp)) as b, Profile(str(ap)) as a:
        return diff_profiles(b, a, gpu=0)


def test_golden_verdict_and_category_attribution(tmp_path):
    """The headline: a same-compute, more-idle, more-comm 'after' is a likely
    regression, and the four category buckets are frozen."""
    data = to_diff_dict(_run_loop(tmp_path))

    assert data["verdict"] == "regression_likely"
    assert data["comparability_confidence"] == 1.0  # same export schema, same GPU
    assert data["warnings"] == []

    step = data["step_time"]
    assert step["before_ms"] == 30.0
    assert step["after_ms"] == 40.0
    assert step["delta_pct"] == 33.33  # well past the 5% regression gate

    cats = {c["category"]: c for c in data["category_attribution"]}
    # Compute is unchanged; the regression is entirely idle + exposed comm.
    # The 20ms compute category INCLUDES the 5ms hidden collective folded in by
    # the HTA convention — if overlap were miscounted as comm, compute would read
    # 15 and communication 10, and these assertions would fail.
    assert cats["compute"]["before_ms"] == 20.0 and cats["compute"]["delta_ms"] == 0.0
    assert cats["communication"]["before_ms"] == 5.0 and cats["communication"]["after_ms"] == 10.0
    assert cats["idle"]["before_ms"] == 5.0 and cats["idle"]["after_ms"] == 10.0
    assert cats["launch_overhead"]["after_ms"] == 0.0

    # Overlap is real in this pair (a hidden collective), so the HTA folding is
    # exercised, not bypassed: 5ms overlapped in both, half the NCCL before.
    assert data["overlap"]["before"]["overlap_ms"] == 5.0
    assert data["overlap"]["before"]["overlap_pct"] == 50.0


def test_golden_diff_json_envelope_and_on_disk_record(tmp_path):
    """The serialized diff.json carries the loop's auditable envelope, and the
    on-disk record round-trips unchanged."""
    data = _run_loop(tmp_path)
    payload = json.loads(to_diff_json(data))

    for key in ("schema_version", "producer", "diff_id", "verdict", "category_attribution"):
        assert key in payload, f"diff.json envelope missing {key}"
    assert payload["verdict"] == "regression_likely"
    assert payload["diff_id"].startswith("diff")

    # Write the record and read it back — the on-disk artifact is stable.
    out = tmp_path / "diff.json"
    out.write_text(to_diff_json(data))
    assert json.loads(out.read_text()) == payload


def test_golden_diff_id_is_deterministic(tmp_path):
    """Same before/after content must produce the same diff_id — the record is
    reproducible, so a re-run over an unchanged pair is idempotent."""
    first = to_diff_dict(_run_loop(tmp_path / "a"))["diff_id"]
    second = to_diff_dict(_run_loop(tmp_path / "b"))["diff_id"]
    assert first == second


def test_golden_improvement_direction(tmp_path):
    """Symmetry check: swapping before/after flips the verdict to an improvement,
    so the loop is not hard-coded to regressions."""
    bp = tmp_path / "before.sqlite"
    ap = tmp_path / "after.sqlite"
    _make_profile(str(bp), _BEFORE_KERNELS)
    _make_profile(str(ap), _AFTER_KERNELS)
    # after (slow) as the baseline, before (fast) as the new profile -> improvement
    with Profile(str(ap)) as slow, Profile(str(bp)) as fast:
        data = to_diff_dict(diff_profiles(slow, fast, gpu=0))
    assert data["verdict"] == "improvement_likely"
    assert data["step_time"]["delta_pct"] < -5.0
