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


def _run(conn):
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("critical_path")
    assert skill is not None
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
