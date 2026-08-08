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
        #
        # It must differ in `end` AND name from its start-partner, or the tie is
        # unobservable: LAG would return the same prev_end/prev_kernel whichever
        # peer it picked, and a test built on it would pass with the tiebreak
        # removed. The partner at i==0 is (1_000, 6_000, name=alpha); this one
        # ends later and carries the other name, so which peer wins is visible
        # in both the computed gap and the reported previous kernel.
        rows.append((1_000, 8_000, device, 7, corr, 1, 0))
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

    b = Path(__file__).resolve().parent.parent / "src/nsys_ai/skills/builtins"
    src = (b / "critical_path.py").read_text()
    assert "ORDER BY busy DESC, deviceId ASC" in src

    rcm = (b / "root_cause_matcher.py").read_text()
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

    b = Path(__file__).resolve().parent.parent / "src/nsys_ai/skills/builtins"

    gaps = (b / "gpu_idle_gaps.py").read_text()
    # Both the row query and the aggregate that sums total_gap_ns.
    # Count, not substring-absence: an earlier version asserted a string that
    # was absent from the buggy code too, so it passed on what it should reject.
    assert gaps.count("ORDER BY k.start, k.correlationId") == 3

    pattern = (b / "kernel_launch_pattern.py").read_text()
    assert pattern.count("ORDER BY k.start, k.correlationId") == 2

    nvtx = (b / "nvtx_layer_breakdown.py").read_text()
    # FIRST(...) and every string_agg(...) path get the label tiebreak.
    assert nvtx.count("n_start ASC, nvtx_text ASC") == 4
    assert "ORDER BY n_dur ASC, n_start ASC)" not in nvtx
    assert "ORDER BY n_dur DESC, n_start ASC)" not in nvtx


def test_the_sweep_resolves_label_ties_by_data_not_by_input_order():
    """Two enclosing ranges with an identical span are both innermost.

    The sweep used to take whichever the stack yielded, so the answer followed
    the order the rows arrived in rather than the data: feeding the same two
    ranges as (zzz, aaa) chose `aaa` and as (aaa, zzz) chose `zzz`. That is the
    "different answer for the same profile" failure this module exists to catch,
    and it does not show up on real captures because exact-span ties are rare
    there -- a full 3.5GB comparison found none.

    Asserted on behaviour rather than on the source text. The previous version
    of this test grepped parquet_cache.py for a fragment of the IEJoin SQL, and
    went red the moment that query was replaced by the sweep even though the
    property it cared about was intact.
    """
    from nsys_ai.parquet_cache import _sweep_nvtx_kernel_map

    kr = [(1, 100, 200, 100, 200, "k1")]
    answers = set()
    for order in (("zzz", "aaa"), ("aaa", "zzz")):
        nvtx = sorted(((1, 0, 300, text) for text in order), key=lambda r: (r[0], r[1]))
        rows = _sweep_nvtx_kernel_map(kr, nvtx)
        assert rows, "the sweep attributed nothing"
        answers.add((rows[0]["nvtx_text"], rows[0]["nvtx_path"]))

    assert len(answers) == 1, f"input order changed the answer: {answers}"
    # The order is the one nvtx_attribution uses: innermost by
    # (duration ASC, start ASC, text ASC); path outermost-first.
    assert answers == {("aaa", "aaa > zzz")}


def test_the_sweep_does_not_depend_on_the_order_kernels_arrive_in():
    """The kernel side may arrive in any order; only the NVTX side must be sorted.

    The kernel-side query used to carry `ORDER BY r.globalTid, r.start` and a
    comment claiming the sweep "advances one index per thread, so this must
    arrive sorted". Half of that is true: the sweep does advance a single index
    over the *ranges*, but `_sweep_nvtx_kernel_map` re-sorts the *kernels* itself
    (`kr_list.sort(...)`), so the SQL order was being redone in Python. This test
    is what stops that self-sort from being dropped as "redundant" later, which
    would silently make the map's contents depend on DuckDB's join order.

    It guards `_sweep_nvtx_kernel_map`, which is the on-demand builder's entry
    point (`_ensure_nvtx_kernel_map_in_memory`) and the only caller that buckets
    rows in Python. The cached builder no longer comes through here: it
    partitions by thread in SQL and streams both sides, so it *does* ask for
    `ORDER BY r.start` per thread — a stream has nowhere to put a later sort.
    That is not this ORDER BY coming back; it is a different query, on a
    different axis, feeding a consumer that no longer sorts.

    Asserted on *content*, canonically ordered, not on the returned list order.
    The sweep visits threads in the order they first appear in its input, so
    interleaving two threads differently permutes the rows between thread
    blocks without changing a single attribution — and the writer emits
    ``COPY ... ORDER BY k_start, k_end, kernel_name`` anyway, so list order is
    not what reaches the cache.

    Checked against a real capture before being reduced to this size: on an
    88MB profile (449,043 kernel-runtime rows, 466,398 ranges), 200 shuffles of
    the kernel side produced one distinct result, while 5 shuffles of the NVTX
    side produced 5 distinct results, none matching the unshuffled one. Kept
    synthetic here so it runs in milliseconds and needs no fixture profile.
    """
    import random

    from nsys_ai.parquet_cache import _sweep_nvtx_kernel_map

    # Two threads, nested ranges, kernels landing at several depths. Distinct
    # r_start values throughout, so the correct answer is a single total order
    # and any dependence on arrival order shows up as an exact mismatch.
    nvtx = [
        (1, 0, 1000, "step"),
        (1, 100, 400, "fwd"),
        (1, 150, 250, "attn"),
        (1, 500, 900, "bwd"),
        (2, 0, 1000, "step"),
        (2, 200, 800, "allreduce"),
    ]
    kr = [
        (1, 160, 240, 1600, 2400, "kA"),
        (1, 300, 380, 3000, 3800, "kB"),
        (1, 420, 480, 4200, 4800, "kC"),  # inside step only
        (1, 600, 700, 6000, 7000, "kD"),
        (2, 350, 450, 3500, 4500, "kE"),
        (2, 850, 950, 8500, 9500, "kF"),  # inside step only
    ]
    nvtx_sorted = sorted(nvtx, key=lambda r: (r[0], r[1]))

    def canonical(rows):
        return sorted(
            (r["k_start"], r["k_end"], r["kernel_name"], r["nvtx_text"], r["nvtx_path"])
            for r in rows
        )

    expected = canonical(_sweep_nvtx_kernel_map(list(kr), list(nvtx_sorted)))
    # Guard the guard: a sweep that attributed nothing, or that attributed
    # everything to the outermost range, would make the comparison below pass
    # for the wrong reason.
    assert len(expected) == len(kr)
    assert [row[-1] for row in expected] == [
        "step > fwd > attn",  # kA, k_start 1600
        "step > fwd",  # kB, 3000
        "step > allreduce",  # kE, 3500
        "step",  # kC, 4200 — inside `step` only
        "step > bwd",  # kD, 6000
        "step",  # kF, 8500 — inside `step` only
    ]

    for seed in range(50):
        shuffled = list(kr)
        random.Random(seed).shuffle(shuffled)
        assert canonical(_sweep_nvtx_kernel_map(shuffled, list(nvtx_sorted))) == expected, (
            f"kernel arrival order changed the sweep's answer (seed {seed})"
        )


def test_the_direct_attach_builder_keeps_the_same_total_order():
    """`nvtx_attribution` builds the map on a direct-attached profile, where
    there is no parquet cache. Its ordering must match the sweep's, or cached
    and uncached runs disagree."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "src/nsys_ai"
    attribution = (root / "nvtx_attribution.py").read_text()
    assert attribution.count("n_start ASC, nvtx_text ASC") == 2

    cache = (root / "parquet_cache.py").read_text()
    # The persisted map must not be reused across a change in how it is built.
    assert "_CACHE_VERSION = 16" in cache


# ── Cross-backend agreement ─────────────────────────────────────────────────


def _duckdb_view_of(sqlite_path):
    """Same data, DuckDB engine — the parquet-cache path's tie resolution."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{sqlite_path}' AS src (TYPE sqlite);")
    con.execute("USE src;")
    return con


def _ordered_payload(rows):
    """Compare every non-summary field, including derived floats.

    Summary rows are excluded deliberately: some fields there reflect backend
    *capability* rather than ordering — `device_idle_ms` is computed by a
    sweep-line that only the DuckDB path provides, so it is legitimately None
    on SQLite. Comparing it would test the wrong thing.
    """
    return [
        tuple(sorted((k, str(v)) for k, v in r.items()))
        for r in rows
        if not r.get("_summary")
    ]


@pytest.fixture
def float_cast_sqlite(tmp_path):
    """Uneven dispatch counts whose percentages are not exact in float32.

    11/21 and 10/21 round to 52.4 and 47.6. DuckDB's REAL (float32) cannot
    represent those tenths, so CAST(... AS REAL) + ROUND yields
    52.400001525878906 / 47.599998474121094 while SQLite's REAL (float64)
    yields 52.4 / 47.6. That is the #276 symptom on a minimal fixture.
    """
    path = tmp_path / "float_cast.sqlite"
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    conn.execute("INSERT INTO StringIds VALUES (1, 'k'), (2, 'cudaLaunchKernel')")
    kernels = []
    runtimes = []
    corr = 1
    # Thread 1001: 11 launches; thread 1002: 10. CPU end must precede GPU start.
    for tid, n in ((1001, 11), (1002, 10)):
        for i in range(n):
            gpu_start = 1_000_000 + corr * 10_000
            kernels.append((gpu_start, gpu_start + 5_000, 0, 7, corr, 1, 1))
            runtimes.append((gpu_start - 2_000, gpu_start - 1_000, corr, tid, 2))
            corr += 1
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)", kernels
    )
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)", runtimes
    )
    conn.commit()
    conn.close()
    return str(path)


def test_skill_sql_casts_ratios_as_double_not_real():
    """SQLite REAL is float64; DuckDB REAL is float32. Skills must say DOUBLE.

    Width is pinned at the skill SQL cast, not by post-query Python rounding:
    findings and golden fixtures consume the SQL values directly.
    """
    from pathlib import Path

    builtins = Path(__file__).resolve().parent.parent / "src/nsys_ai/skills/builtins"
    offenders = []
    for path in builtins.glob("*.py"):
        text = path.read_text()
        if "AS REAL" in text:
            offenders.append(path.name)
    assert not offenders, (
        f"CAST(... AS REAL) is float32 on DuckDB — use DOUBLE: {offenders}"
    )


def test_pct_of_dispatches_is_float64_on_both_backends(float_cast_sqlite):
    """Pin the schema-visible percentage, not merely 'same type on both sides'.

    Before the DOUBLE cast, DuckDB returned 52.400001525878906 for the leading
    thread — enough to disagree with SQLite and to poison a golden fixture.
    """
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("cpu_gpu_pipeline")

    conn = sqlite3.connect(float_cast_sqlite)
    try:
        via_sqlite = skill.execute(conn, limit=10)
    finally:
        conn.close()

    try:
        con = _duckdb_view_of(float_cast_sqlite)
    except Exception as exc:
        pytest.skip(f"duckdb sqlite attach unavailable: {exc}")
    try:
        via_duckdb = skill.execute(con, limit=10)
    except Exception as exc:
        pytest.skip(f"cpu_gpu_pipeline unsupported on duckdb attach path: {exc}")
    finally:
        con.close()

    assert [(r["cpu_tid"], r["dispatches"], r["pct_of_dispatches"]) for r in via_sqlite] == [
        (1001, 11, 52.4),
        (1002, 10, 47.6),
    ]
    assert [(r["cpu_tid"], r["dispatches"], r["pct_of_dispatches"]) for r in via_duckdb] == [
        (1001, 11, 52.4),
        (1002, 10, 47.6),
    ]


@pytest.mark.parametrize(
    "skill_name,kwargs",
    [
        ("top_kernels", {"limit": 10}),
        ("gpu_idle_gaps", {"min_gap_ns": 1, "limit": 20, "device": 0}),
        ("kernel_launch_pattern", {"limit": 10}),
        ("cpu_gpu_pipeline", {"limit": 10}),
    ],
)
def test_same_profile_same_answer_on_both_backends(tie_dense_sqlite, float_cast_sqlite, skill_name, kwargs):
    """A user must not get a different answer because the cache was built.

    Skills issue identical SQL to both adapters; only a total order makes the
    two engines agree when keys tie. Derived floats (pct_of_dispatches,
    dispatch_rate_per_ms) must agree too — that requires float64 casts, not
    DuckDB REAL.
    """
    from nsys_ai.skills.registry import get_skill

    skill = get_skill(skill_name)
    # cpu_gpu_pipeline needs correlated runtime rows; the float-cast fixture
    # supplies them. Other skills keep using the tie-dense profile.
    profile = float_cast_sqlite if skill_name == "cpu_gpu_pipeline" else tie_dense_sqlite

    conn = sqlite3.connect(profile)
    try:
        via_sqlite = skill.execute(conn, **kwargs)
    finally:
        conn.close()

    try:
        con = _duckdb_view_of(profile)
    except Exception as exc:  # sqlite_scanner unavailable in this environment
        pytest.skip(f"duckdb sqlite attach unavailable: {exc}")
    try:
        via_duckdb = skill.execute(con, **kwargs)
    except Exception as exc:
        pytest.skip(f"{skill_name} unsupported on duckdb attach path: {exc}")
    finally:
        con.close()

    assert _ordered_payload(via_sqlite) == _ordered_payload(via_duckdb), (
        f"{skill_name} disagreed between SQLite and DuckDB on a tie-dense profile"
    )
