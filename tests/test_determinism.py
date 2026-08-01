"""Analysis must be deterministic: the same profile yields the same answer.

`ORDER BY` on a non-unique key leaves tie resolution to the engine — SQLite
documents that "the order in which two rows for which all ORDER BY expressions
evaluate to equal values are returned is undefined". Skills send the *same* SQL
to both the SQLite and DuckDB adapters, so a non-total order can produce two
different answers for one profile depending on whether the parquet cache exists.

The naive check — run it twice and compare — passes trivially, because one
engine with one plan over one dataset usually returns the same arbitrary order.
These tests instead use a **tie-dense** fixture, where every sort key that skills
order on is deliberately duplicated, and compare across the two backends.

This is not hypothetical. On a real two-GPU H100 capture whose devices had
byte-identical kernel counts (210765 each), the device-selection query returned
device 1 under SQLite and device 0 under DuckDB — so the whole root-cause
analysis targeted a different GPU depending on whether the parquet cache had
been built. Adding the `deviceId` tiebreak made both engines return device 0.
"""

import sqlite3

import pytest

# ── Tie-dense fixture ───────────────────────────────────────────────────────
#
# Everything that a skill might sort on is duplicated on purpose:
#   * two devices with byte-identical kernel counts AND identical busy time
#     (the symmetric data-parallel case, where device selection ties)
#   * kernels sharing a start timestamp on one stream (window-function ties)
#   * kernels with identical durations (top-N ordering ties)

_SCHEMA = """
CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
    start INTEGER, "end" INTEGER, deviceId INTEGER, streamId INTEGER,
    correlationId INTEGER, shortName INTEGER, demangledName INTEGER
);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
    start INTEGER, "end" INTEGER, correlationId INTEGER,
    globalTid INTEGER, nameId INTEGER
);
"""


def _seed(conn):
    conn.executescript(_SCHEMA)
    # Two distinct names sorting differently from their insertion order, so a
    # name tiebreak is observable.
    conn.executemany(
        "INSERT INTO StringIds VALUES (?, ?)",
        [(1, "zeta_kernel"), (2, "alpha_kernel"), (3, "cudaLaunchKernel")],
    )
    rows = []
    corr = 100
    # Devices 0 and 1 get identical work: same count, same total duration.
    for device in (0, 1):
        for i in range(4):
            start = 1_000 + i * 10_000
            # Every kernel lasts exactly 5_000ns -> all durations tie.
            rows.append((start, start + 5_000, device, 7, corr, 1 if i % 2 else 2, 0))
            corr += 1
        # A second kernel sharing a start timestamp on the same stream: makes
        # "the previous kernel" ambiguous for LAG() unless the order is total.
        rows.append((1_000, 6_000, device, 7, corr, 2, 0))
        corr += 1
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)", rows
    )
    conn.commit()


@pytest.fixture
def tie_dense_sqlite(tmp_path):
    """A profile whose every sort key is deliberately duplicated."""
    path = tmp_path / "ties.sqlite"
    conn = sqlite3.connect(str(path))
    _seed(conn)
    conn.close()
    return str(path)


# ── Device selection: the highest-impact tie ────────────────────────────────


def test_busiest_device_is_stable_when_devices_tie(tie_dense_sqlite):
    """Equal-busy devices must resolve to the same device every time.

    `critical_path` picks the busiest device and runs the *entire* analysis
    against it, so an unbroken tie silently retargets the whole report.
    """
    from nsys_ai.connection import wrap_connection
    from nsys_ai.skills.builtins.critical_path import _busiest_device

    picks = set()
    for _ in range(5):
        conn = sqlite3.connect(tie_dense_sqlite)
        try:
            picks.add(_busiest_device(wrap_connection(conn), "CUPTI_ACTIVITY_KIND_KERNEL"))
        finally:
            conn.close()
    assert len(picks) == 1, f"busiest device varied across runs: {picks}"
    # The tiebreak is documented as "lowest device id wins".
    assert picks == {0}


def test_busiest_device_query_declares_a_total_order():
    """Guard the tiebreak itself, so removing it fails loudly.

    Without this, the behavioural test above can keep passing on a single
    engine while the underlying order is once again undefined.
    """
    from pathlib import Path

    src = Path("src/nsys_ai/skills/builtins/critical_path.py").read_text()
    assert "ORDER BY busy DESC, deviceId ASC" in src

    rcm = Path("src/nsys_ai/skills/builtins/root_cause_matcher.py").read_text()
    assert "ORDER BY c DESC, deviceId ASC" in rcm


# ── Window functions: a tie changes a computed value, not a row position ────


def test_idle_gap_values_are_stable_when_starts_tie(tie_dense_sqlite):
    """Kernels sharing a start make LAG()'s "previous kernel" ambiguous.

    That changes the *computed gap*, so this is a wrong-number bug rather
    than a cosmetic ordering one.

    Note this repeated-run check is a weak guard by construction — one engine
    with one plan tends to return the same arbitrary order, which is exactly
    why it is paired with the structural assertion below and the cross-backend
    test further down. On its own it does not catch a removed tiebreak.
    """
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("gpu_idle_gaps")
    seen = []
    for _ in range(5):
        conn = sqlite3.connect(tie_dense_sqlite)
        try:
            rows = skill.execute(conn, min_gap_ns=1, limit=50, device=0)
        finally:
            conn.close()
        seen.append([(r.get("gap_ns"), r.get("before_kernel"), r.get("after_kernel")) for r in rows])
    assert all(s == seen[0] for s in seen), f"idle gaps varied across runs: {seen}"


def test_window_functions_declare_a_total_order():
    """Every window ORDER BY that feeds a computed value must be total.

    A `LAG`/`LEAD` over a non-unique order picks an arbitrary neighbour, so the
    resulting gap or dispatch interval is engine-dependent. The equivalent
    aggregate orderings in the NVTX path decide which label wins.
    """
    from pathlib import Path

    b = Path("src/nsys_ai/skills/builtins")

    gaps = (b / "gpu_idle_gaps.py").read_text()
    # Both the row query and the aggregate that sums total_gap_ns.
    assert gaps.count("ORDER BY k.start, k.correlationId") == 3
    assert "PARTITION BY k.deviceId, k.streamId ORDER BY k.start\n" not in gaps

    pattern = (b / "kernel_launch_pattern.py").read_text()
    assert pattern.count("ORDER BY k.start, k.correlationId") == 2

    nvtx = (b / "nvtx_layer_breakdown.py").read_text()
    # FIRST(...) and every string_agg(...) path get the label tiebreak.
    assert nvtx.count("n_start ASC, nvtx_text ASC") == 4
    assert "ORDER BY n_dur ASC, n_start ASC)" not in nvtx
    assert "ORDER BY n_dur DESC, n_start ASC)" not in nvtx


# ── Cross-backend agreement ─────────────────────────────────────────────────


def _duckdb_view_of(sqlite_path):
    """Same data, DuckDB engine — the parquet-cache path's tie resolution."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{sqlite_path}' AS src (TYPE sqlite);")
    con.execute("USE src;")
    return con


@pytest.mark.parametrize(
    "skill_name,kwargs",
    [
        ("top_kernels", {"limit": 10}),
        ("gpu_idle_gaps", {"min_gap_ns": 1, "limit": 20, "device": 0}),
    ],
)
def test_same_profile_same_answer_on_both_backends(tie_dense_sqlite, skill_name, kwargs):
    """A user must not get a different answer because the cache was built.

    Skills issue identical SQL to both adapters; only a total order makes the
    two engines agree when keys tie.
    """
    from nsys_ai.skills.registry import get_skill

    skill = get_skill(skill_name)

    conn = sqlite3.connect(tie_dense_sqlite)
    try:
        via_sqlite = skill.execute(conn, **kwargs)
    finally:
        conn.close()

    try:
        con = _duckdb_view_of(tie_dense_sqlite)
    except Exception as exc:  # sqlite_scanner unavailable in this environment
        pytest.skip(f"duckdb sqlite attach unavailable: {exc}")
    try:
        via_duckdb = skill.execute(con, **kwargs)
    except Exception as exc:
        pytest.skip(f"{skill_name} unsupported on duckdb attach path: {exc}")
    finally:
        con.close()

    def ordered_rows(rows):
        """The ordering-sensitive payload only.

        Summary rows are excluded deliberately: some fields there reflect
        backend *capability* rather than ordering — `device_idle_ms` is
        computed by a sweep-line that only the DuckDB path provides, so it is
        legitimately None on SQLite. Comparing it would test the wrong thing.
        """
        return [
            tuple(sorted((k, str(v)) for k, v in r.items()))
            for r in rows
            if not r.get("_summary")
        ]

    assert ordered_rows(via_sqlite) == ordered_rows(via_duckdb), (
        f"{skill_name} disagreed between SQLite and DuckDB on a tie-dense profile"
    )
