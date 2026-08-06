"""Tests for the critical_path bound-classification skill.

Builds minimal Nsight-shaped SQLite fixtures with KNOWN bottlenecks and
asserts the emitted bound class, on-path breakdown, and confidence behave.
"""

import sqlite3

import pytest

_SCHEMA_SQL = """\
CREATE TABLE StringIds (
    id      INTEGER PRIMARY KEY,
    value   TEXT NOT NULL
);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
    globalPid       INTEGER DEFAULT 0,
    deviceId        INTEGER DEFAULT 0,
    streamId        INTEGER DEFAULT 0,
    correlationId   INTEGER DEFAULT 0,
    start           INTEGER NOT NULL,
    end             INTEGER NOT NULL,
    shortName       INTEGER NOT NULL,
    demangledName   INTEGER DEFAULT 0
);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
    globalTid       INTEGER DEFAULT 0,
    correlationId   INTEGER DEFAULT 0,
    start           INTEGER NOT NULL,
    end             INTEGER NOT NULL,
    nameId          INTEGER DEFAULT 0
);
"""

# shortName / demangledName ids reused across every fixture.
_STRINGS = [
    (1, "compute_gemm_kernel"),
    (2, "compute_elementwise_kernel"),
    (10, "nccl_AllReduce_kernel"),
    (24, "cudaLaunchKernel"),
]


def _make_conn(kernels, runtime=None):
    """Build an in-memory Nsight-shaped DB.

    kernels: list of (deviceId, streamId, correlationId, start, end, shortName)
             demangledName is set equal to shortName.
    runtime: optional list of (correlationId, start, end, nameId).
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA_SQL)
    conn.executemany("INSERT INTO StringIds VALUES (?, ?)", _STRINGS)
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
        "(deviceId, streamId, correlationId, start, end, shortName, demangledName) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(d, s, c, st, en, sn, sn) for (d, s, c, st, en, sn) in kernels],
    )
    if runtime:
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME "
            "(correlationId, start, end, nameId) VALUES (?, ?, ?, ?)",
            runtime,
        )
    conn.commit()
    return conn


def _get_skill():
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("critical_path")
    assert skill is not None
    return skill


def _run(conn):
    skill = _get_skill()
    rows = skill.execute(conn)
    assert isinstance(rows, list)
    assert len(rows) == 1
    return skill, rows[0]


MS = 1_000_000


# ---------------------------------------------------------------------------
# Registration / discovery
# ---------------------------------------------------------------------------


def test_critical_path_registered():
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("critical_path")
    assert skill is not None
    assert skill.name == "critical_path"
    assert skill.category == "system"
    assert skill.execute_fn is not None


def test_registry_autodiscovery_intact():
    """Adding the new builtin must not break discovery of the others."""
    from nsys_ai.skills import list_skills

    names = list_skills()
    assert "critical_path" in names
    # Sanity: several unrelated builtins still discovered.
    for other in ("top_kernels", "overlap_breakdown", "gpu_idle_gaps"):
        assert other in names


# ---------------------------------------------------------------------------
# Known-bound classification
# ---------------------------------------------------------------------------


def test_gpu_compute_bound():
    """Back-to-back large kernels, no NCCL, ~no idle -> gpu-compute-bound."""
    kernels = [
        (0, 7, 1, 0, 10 * MS, 1),
        (0, 7, 2, 10 * MS, 20 * MS, 1),
        (0, 7, 3, 20 * MS, 30 * MS, 1),
    ]
    conn = _make_conn(kernels)
    _, r = _run(conn)
    conn.close()

    assert r["bound_class"] == "gpu-compute-bound"
    b = r["breakdown"]
    assert b["gpu_compute_share"] > 0.9
    assert b["comm_ms"] == 0.0
    assert r["confidence"] > 0.5
    # Top on-path compute kernel present with a time selection.
    assert r["top_compute_kernels"]
    top = r["top_compute_kernels"][0]
    assert top["start_ns"] is not None and top["end_ns"] is not None
    assert top["end_ns"] > top["start_ns"]


def test_cpu_bound():
    """Tiny kernels with huge gaps -> cpu-bound (GPU idle dominates)."""
    kernels = [
        (0, 7, 1, 0, 100_000, 1),
        (0, 7, 2, 20 * MS, 20 * MS + 100_000, 1),
        (0, 7, 3, 40 * MS, 40 * MS + 100_000, 1),
    ]
    conn = _make_conn(kernels)
    _, r = _run(conn)
    conn.close()

    assert r["bound_class"] == "cpu-bound"
    b = r["breakdown"]
    assert b["cpu_share"] > 0.9
    assert r["confidence"] > 0.5


def test_comm_bound():
    """Large exposed NCCL, small non-overlapping compute -> comm-bound."""
    kernels = [
        (0, 7, 1, 0, 1 * MS, 1),  # 1ms compute
        (0, 8, 2, 2 * MS, 20 * MS, 10),  # 18ms exposed NCCL
    ]
    conn = _make_conn(kernels)
    _, r = _run(conn)
    conn.close()

    assert r["bound_class"] == "comm-bound"
    b = r["breakdown"]
    assert b["comm_share"] > 0.8
    assert b["comm_ms"] > b["gpu_compute_ms"]
    assert r["top_collectives"]
    assert "nccl" in r["top_collectives"][0]["name"].lower()


def test_mixed_on_near_tie():
    """~Equal compute and idle -> mixed with low confidence."""
    kernels = [
        (0, 7, 1, 0, 15 * MS, 1),  # 15ms compute
        (0, 7, 2, 30 * MS, 30 * MS + 10_000, 1),  # negligible; idle 15ms before it
    ]
    conn = _make_conn(kernels)
    _, r = _run(conn)
    conn.close()

    assert r["bound_class"] == "mixed"
    assert r["confidence"] < 0.15


# ---------------------------------------------------------------------------
# Breakdown invariants and degenerate handling
# ---------------------------------------------------------------------------


def test_breakdown_sums_to_critical_path():
    kernels = [
        (0, 7, 1, 0, 5 * MS, 1),
        (0, 8, 2, 6 * MS, 12 * MS, 10),
        (0, 7, 3, 15 * MS, 18 * MS, 2),
    ]
    conn = _make_conn(kernels)
    _, r = _run(conn)
    conn.close()

    b = r["breakdown"]
    total = b["gpu_compute_ms"] + b["comm_ms"] + b["cpu_ms"]
    assert total == pytest.approx(r["critical_path_ms"], abs=0.05)
    shares = b["gpu_compute_share"] + b["comm_share"] + b["cpu_share"]
    assert shares == pytest.approx(1.0, abs=1e-3)


def test_degenerate_no_kernels():
    conn = _make_conn([])
    _, r = _run(conn)
    conn.close()

    assert r["bound_class"] == "n/a"
    assert r["confidence"] == 0.0
    assert r["critical_path_ms"] == 0.0


def test_format_is_text():
    kernels = [(0, 7, 1, 0, 10 * MS, 1), (0, 7, 2, 10 * MS, 20 * MS, 1)]
    conn = _make_conn(kernels)
    skill, r = _run(conn)
    text = skill.format_rows([r])
    conn.close()
    assert isinstance(text, str)
    assert "Critical-Path" in text
    assert "Bound class" in text


# ---------------------------------------------------------------------------
# Structured Finding
# ---------------------------------------------------------------------------


def test_finding_category_valid_for_confident_class():
    from nsys_ai.annotation import FindingCategory

    valid = set(FindingCategory.__args__)

    kernels = [
        (0, 7, 1, 0, 10 * MS, 1),
        (0, 7, 2, 10 * MS, 20 * MS, 1),
        (0, 7, 3, 20 * MS, 30 * MS, 1),
    ]
    conn = _make_conn(kernels)
    skill, r = _run(conn)
    findings = skill.to_findings_fn([r])
    conn.close()

    assert len(findings) == 1
    assert findings[0].category in valid
    assert findings[0].category == "compute"


def test_no_finding_for_mixed():
    kernels = [
        (0, 7, 1, 0, 15 * MS, 1),
        (0, 7, 2, 30 * MS, 30 * MS + 10_000, 1),
    ]
    conn = _make_conn(kernels)
    skill, r = _run(conn)
    findings = skill.to_findings_fn([r])
    conn.close()
    assert findings == []


# ---------------------------------------------------------------------------
# HTA convention: overlapping communication is hidden and counts as compute
# ---------------------------------------------------------------------------


def test_overlap_hidden_counts_as_compute():
    """A 20ms compute kernel with a 20ms NCCL kernel that overlaps its second
    half: the overlapped 10ms is folded into gpu-compute (hidden comm), and
    only the exposed 10ms NCCL tail is comm."""
    kernels = [
        (0, 7, 1, 0, 20 * MS, 1),  # compute [0, 20ms]
        (0, 8, 2, 10 * MS, 30 * MS, 10),  # NCCL   [10ms, 30ms] — half hidden
    ]
    conn = _make_conn(kernels)
    _, r = _run(conn)
    conn.close()

    b = r["breakdown"]
    # compute_only (10ms) + overlap (10ms) folded together == 20ms.
    assert b["gpu_compute_ms"] == pytest.approx(20.0, abs=0.05)
    # Only the exposed NCCL tail (10ms) is comm; the hidden half is not.
    assert b["comm_ms"] == pytest.approx(10.0, abs=0.05)
    assert r["bound_class"] == "gpu-compute-bound"


# ---------------------------------------------------------------------------
# Trim: GPU-idle at the window edges is host-wait (cpu), not invisible
# ---------------------------------------------------------------------------


def test_trim_edge_idle_counts_as_cpu():
    """One 30ms compute kernel late in a 100ms iteration window. Without a
    trim the span is the kernel extent (compute-bound); with the iteration
    window trimmed, the 70ms of leading GPU-idle is host-wait -> cpu-bound."""
    kernels = [(0, 7, 1, 70 * MS, 100 * MS, 1)]  # compute only in the last 30ms

    conn = _make_conn(kernels)
    skill = _get_skill()
    untrimmed = skill.execute(conn)[0]
    trimmed = skill.execute(conn, trim_start_ns=0, trim_end_ns=100 * MS)[0]
    conn.close()

    # Kernel-extent view sees only back-to-back compute.
    assert untrimmed["bound_class"] == "gpu-compute-bound"
    # Iteration-window view credits the 70ms edge idle to cpu.
    assert trimmed["bound_class"] == "cpu-bound"
    assert trimmed["breakdown"]["cpu_ms"] == pytest.approx(70.0, abs=0.1)
    assert trimmed["breakdown"]["gpu_compute_ms"] == pytest.approx(30.0, abs=0.1)


# ---------------------------------------------------------------------------
# Multi-device: busiest device is the spine by default, overridable
# ---------------------------------------------------------------------------


def test_busiest_device_selected_and_overridable():
    kernels = [
        # device 1 — 30ms of back-to-back compute (busiest)
        (1, 7, 1, 0, 10 * MS, 1),
        (1, 7, 2, 10 * MS, 20 * MS, 1),
        (1, 7, 3, 20 * MS, 30 * MS, 1),
        # device 0 — a single 5ms compute kernel
        (0, 7, 4, 0, 5 * MS, 1),
    ]
    conn = _make_conn(kernels)
    skill = _get_skill()
    auto = skill.execute(conn)[0]
    forced = skill.execute(conn, device=0)[0]
    conn.close()

    assert auto["device"] == 1  # busiest by kernel time
    assert auto["bound_class"] == "gpu-compute-bound"
    assert forced["device"] == 0  # explicit override honored


# ---------------------------------------------------------------------------
# Conservative refusal: no majority, and insufficient on-path time
# ---------------------------------------------------------------------------


def test_no_majority_reported_mixed():
    """~46% compute / 28% comm / 26% cpu: a plurality but not a majority, so
    the skill refuses to crown a winner and reports mixed."""
    kernels = [
        (0, 7, 1, 0, 45 * MS, 1),  # 45ms compute
        (0, 8, 2, 45 * MS, 73 * MS, 10),  # 28ms exposed NCCL, back-to-back
        (0, 7, 3, 99 * MS, 100 * MS, 1),  # 1ms compute -> idle [73,99] = 26ms
    ]
    conn = _make_conn(kernels)
    _, r = _run(conn)
    conn.close()

    assert r["breakdown"]["gpu_compute_share"] < 0.5
    assert r["margin"] >= 0.15  # a clear plurality, but...
    assert r["bound_class"] == "mixed"  # ...no majority -> refuse to commit
    assert "majority" in r["note"]


def test_insufficient_on_path_time_reported_mixed():
    """A single sub-millisecond kernel is too little signal to classify."""
    kernels = [(0, 7, 1, 0, 100_000, 1)]  # 0.1ms
    conn = _make_conn(kernels)
    skill, r = _run(conn)
    findings = skill.to_findings_fn([r])
    conn.close()

    assert r["bound_class"] == "mixed"
    assert "nsufficient" in r["note"]
    assert findings == []  # no confident class -> no finding


# ---------------------------------------------------------------------------
# CPU attribution grounds the cpu bucket in host-side causes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kernels,expected_class,bucket",
    [
        (
            [
                (0, 7, 1, 0, 100_000, 1),
                (0, 7, 2, 20 * MS, 20 * MS + 100_000, 1),
                (0, 7, 3, 40 * MS, 40 * MS + 100_000, 1),
            ],
            "cpu-bound",
            "cpu_ms",
        ),
        (
            [
                (0, 7, 1, 0, 1 * MS, 1),  # 1ms compute
                (0, 8, 2, 2 * MS, 20 * MS, 10),  # 18ms exposed NCCL
            ],
            "comm-bound",
            "comm_ms",
        ),
    ],
)
def test_reports_no_headroom_to_avoid_double_counting(kernels, expected_class, bucket):
    """The bound class is a verdict, not a new pool of recoverable time.

    Both committed branches must be covered: the cpu bucket is the idle
    gpu_idle_gaps already claims, and the comm bucket the exposed NCCL
    overlap_breakdown already claims. Testing only cpu-bound left the comm
    half — cited as half the justification for this change — unguarded.
    """
    conn = _make_conn(kernels)
    skill, r = _run(conn)
    findings = skill.to_findings_fn([r])
    conn.close()

    assert r["bound_class"] == expected_class
    assert findings[0].headroom_ms is None
    assert findings[0].headroom_basis is None
    # The measurement itself is still reported, just not claimed as headroom.
    assert findings[0].evidence[0].values[bucket] == r["breakdown"][bucket]


def test_cpu_attribution_grounds_cpu_bucket():
    """A cpu-bound run whose dispatches lag the GPU by >1ms surfaces the
    dispatch-starvation count, so the cpu class is grounded in *why*."""
    kernels = [
        (0, 7, 1, 0, 100_000, 1),
        (0, 7, 2, 20 * MS, 20 * MS + 100_000, 1),
        (0, 7, 3, 40 * MS, 40 * MS + 100_000, 1),
    ]
    # Launch API finishes >1ms before each late kernel starts -> starvation.
    runtime = [
        (2, 10 * MS, 14 * MS, 24),  # queue delay 6ms before kernel 2
        (3, 30 * MS, 34 * MS, 24),  # queue delay 6ms before kernel 3
    ]
    conn = _make_conn(kernels, runtime=runtime)
    _, r = _run(conn)
    conn.close()

    assert r["bound_class"] == "cpu-bound"
    assert r["cpu_attribution"].get("dispatch_starvation_events") == 2
