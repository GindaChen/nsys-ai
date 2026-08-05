"""On-demand nvtx_kernel_map builder (issues #257, #319).

The four skills that attribute kernels to NVTX regions depend on a precomputed
`nvtx_kernel_map`. Without it each falls to an in-file IEJoin that hangs DuckDB's
sqlite_scanner, so `ensure_nvtx_kernel_map` builds it on demand via the shared
Python sort-merge (`_sweep_nvtx_kernel_map`).

It started (#257) as cover for the direct-attach no-cache path, where the map was
the one artifact nothing produced. It is now the *only* producer: #319 took the
map out of the cache build, because it was the most expensive artifact there and
the fewest runs read it. Two builds sit behind the one accessor — a streaming one
that writes Parquet back into the cache, and the original in-memory one for
backends with no cache directory — and the tests below cover both.
"""

import concurrent.futures
import json
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from nsys_ai import parquet_cache
from nsys_ai.parquet_cache import _sweep_nvtx_kernel_map, ensure_nvtx_kernel_map

# ── The sort-merge sweep (shared containment core) ──────────────────────────


def test_sweep_attributes_kernel_to_innermost_nvtx():
    """A kernel is credited to its innermost enclosing NVTX range, with the full
    outer>inner path and the nesting depth."""
    # nvtx rows sorted by (tid, start): train_step[0,100] ⊃ forward[10,50]
    nvtx = [(1, 0, 100, "train_step"), (1, 10, 50, "forward")]
    # kr rows: (tid, r_start, r_end, k_start, k_end, kernel_name)
    kr = [
        (1, 15, 20, 1000, 1005, "gemm"),  # inside forward (⊂ train_step)
        (1, 60, 65, 2000, 2010, "add"),  # inside train_step only (forward closed)
    ]
    out = {r["kernel_name"]: r for r in _sweep_nvtx_kernel_map(kr, nvtx)}

    assert out["gemm"]["nvtx_text"] == "forward"
    assert out["gemm"]["nvtx_depth"] == 1
    assert out["gemm"]["nvtx_path"] == "train_step > forward"
    assert out["gemm"]["k_dur_ns"] == 5

    assert out["add"]["nvtx_text"] == "train_step"
    assert out["add"]["nvtx_depth"] == 0
    assert out["add"]["nvtx_path"] == "train_step"
    assert out["add"]["k_dur_ns"] == 10


def test_sweep_drops_kernels_with_no_enclosing_range():
    nvtx = [(1, 0, 10, "phase")]
    kr = [(1, 50, 60, 100, 110, "orphan")]  # runtime after the range closed
    assert _sweep_nvtx_kernel_map(kr, nvtx) == []


def test_sweep_isolates_by_thread():
    """A range on one thread never encloses a kernel launched on another."""
    nvtx = [(1, 0, 100, "t1_range")]
    kr = [(2, 10, 20, 500, 505, "other_thread_kernel")]
    assert _sweep_nvtx_kernel_map(kr, nvtx) == []


# ── ensure_nvtx_kernel_map materialisation ──────────────────────────────────


def _duckdb_profile():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE StringIds(id BIGINT, value VARCHAR)")
    con.execute(
        'CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start BIGINT, "end" BIGINT, deviceId INT, '
        'streamId INT, correlationId BIGINT, shortName BIGINT, demangledName BIGINT)'
    )
    con.execute(
        'CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(globalTid BIGINT, correlationId BIGINT, '
        'start BIGINT, "end" BIGINT)'
    )
    con.execute(
        'CREATE TABLE NVTX_EVENTS(globalTid BIGINT, start BIGINT, "end" BIGINT, text VARCHAR, '
        "eventType INT, textId BIGINT)"
    )
    # gemm (short id 1 / demangled id 2), add (3/4)
    con.execute(
        "INSERT INTO StringIds VALUES (1,'gemm'),(2,'void gemm<float>'),(3,'add'),(4,'void add<float>')"
    )
    # kernels + their runtime launches on tid 1
    con.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
        "(1000,1005,0,7,101,1,2), (2000,2010,0,7,102,3,4)"
    )
    con.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1,101,15,20), (1,102,60,65)"
    )
    # NVTX PushPop ranges (eventType 59) on tid 1
    con.execute(
        "INSERT INTO NVTX_EVENTS VALUES (1,0,100,'train_step',59,NULL), (1,10,50,'forward',59,NULL)"
    )
    return con


def test_ensure_builds_map_with_demangled_names():
    con = _duckdb_profile()
    assert ensure_nvtx_kernel_map(con) is True

    rows = con.execute(
        "SELECT nvtx_text, nvtx_depth, kernel_name, k_dur_ns, is_tc_eligible, uses_tc "
        "FROM nvtx_kernel_map ORDER BY k_start"
    ).fetchall()
    assert rows == [
        # demangled name (not shortName); the 9-col schema carries embedded TC —
        # "gemm" is TC-eligible, "add" is not — so consumers take the map-only path.
        ("forward", 1, "void gemm<float>", 5, 1, 0),
        ("train_step", 0, "void add<float>", 10, 0, 0),
    ]
    # path_dict populated and joinable
    paths = dict(
        con.execute(
            "SELECT d.nvtx_path, m.nvtx_text FROM nvtx_kernel_map m "
            "JOIN nvtx_path_dict d ON m.path_id = d.path_id"
        ).fetchall()
    )
    assert paths == {"train_step > forward": "forward", "train_step": "train_step"}


def test_ensure_orders_nvtx_so_paths_are_correct():
    """The nvtx fetch must ORDER BY start: the sweep advances a single index over
    nvtx per thread without re-sorting, so out-of-order rows would build the path
    inner>outer. Insert the ranges reversed and require the correct outer>inner."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE StringIds(id BIGINT, value VARCHAR)")
    con.execute(
        'CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start BIGINT, "end" BIGINT, deviceId INT, '
        'streamId INT, correlationId BIGINT, shortName BIGINT, demangledName BIGINT)'
    )
    con.execute(
        'CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(globalTid BIGINT, correlationId BIGINT, '
        'start BIGINT, "end" BIGINT)'
    )
    con.execute(
        'CREATE TABLE NVTX_EVENTS(globalTid BIGINT, start BIGINT, "end" BIGINT, text VARCHAR, '
        "eventType INT, textId BIGINT)"
    )
    con.execute("INSERT INTO StringIds VALUES (1,'gemm'),(2,'void gemm<float>')")
    con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1000,1005,0,7,101,1,2)")
    con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1,101,15,20)")
    # inner range inserted BEFORE the outer one — reverse of start order
    con.execute(
        "INSERT INTO NVTX_EVENTS VALUES (1,10,50,'forward',59,NULL), (1,0,100,'train_step',59,NULL)"
    )
    ensure_nvtx_kernel_map(con)
    path = con.execute(
        "SELECT d.nvtx_path FROM nvtx_kernel_map m JOIN nvtx_path_dict d ON m.path_id = d.path_id"
    ).fetchone()[0]
    assert path == "train_step > forward"


def test_ensure_is_noop_when_map_present():
    con = _duckdb_profile()
    assert ensure_nvtx_kernel_map(con) is True
    before = con.execute("SELECT COUNT(*) FROM nvtx_kernel_map").fetchone()[0]
    # Second call must see the existing table and not rebuild/duplicate.
    assert ensure_nvtx_kernel_map(con) is True
    after = con.execute("SELECT COUNT(*) FROM nvtx_kernel_map").fetchone()[0]
    assert before == after == 2


def test_ensure_returns_false_for_sqlite_connection():
    # Non-DuckDB connection: unchanged, caller keeps its own path.
    assert ensure_nvtx_kernel_map(sqlite3.connect(":memory:")) is False


def test_the_map_is_visible_to_a_thread_local_cursor():
    """The on-demand map must not be a TEMP table.

    DuckDB requires each thread to work through its own ``.cursor()`` handle —
    a shared connection serves concurrent queries wrong results, silently. A
    TEMP table is visible only to the connection that created it, so building
    the map as one left every worker thread unable to see it and quietly
    falling back to the slower on-the-fly attribution. The fallback returns
    rows, so nothing raised and nothing looked broken; only the source of the
    numbers changed.

    This bites specifically on the direct-attach path a large profile takes,
    where there is no ``nvtx_kernel_map.parquet`` to expose as a view.
    """
    con = _duckdb_profile()
    assert ensure_nvtx_kernel_map(con) is True
    expected = con.execute("SELECT COUNT(*) FROM nvtx_kernel_map").fetchone()[0]

    for name in ("nvtx_kernel_map", "nvtx_path_dict"):
        cur = con.cursor()
        got = cur.execute(f"SELECT COUNT(*) FROM {name}").fetchone()
        assert got is not None, f"{name} is invisible to a cursor — is it TEMP again?"

    assert con.cursor().execute("SELECT COUNT(*) FROM nvtx_kernel_map").fetchone()[0] == expected


# ── The map is deferred out of the cache build, then persisted (issue #319) ──
#
# ``build_cache`` used to produce all sixteen Parquets before returning. On the
# project's 881 MB reference capture that was 19.8 s of which the map was 11.6 s,
# and ``overlap_breakdown`` — which reads none of it — then took 0.3 s. On the
# 3.5 GB capture the map is 59.9 s of a 93.2 s build. So the map is built on the
# first query that needs it, and written back into the cache so the next process
# reads it instead of rebuilding it.
#
# The two halves have to land together. Deferring alone would have moved the cost
# onto ``ensure_nvtx_kernel_map``'s in-memory build, which ``fetchall``s where the
# cache builder streams: measured on the 881 MB capture, a second
# ``nvtx_layer_breakdown`` run went from 7.6 s to 16.1 s and gained 0.5 GB of peak
# RSS, because the in-memory map died with the process. The tests below pin both
# halves.

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"


@pytest.fixture
def cached_profile(tmp_path):
    """A copy of the NVTX fixture with a freshly built Parquet cache."""
    profile = tmp_path / "p.sqlite"
    shutil.copy(FIXTURE, profile)
    cache_dir = Path(parquet_cache.build_cache(str(profile)))
    return profile, cache_dir


def test_building_a_cache_does_not_build_the_map(cached_profile):
    """The issue's own regression test: opening must not produce the map.

    Both Parquets absent and the stamp saying so, which is the difference
    between "not built yet" and "built wrong" that #312's partial-publication
    guard depends on. A cache without the map is a complete, valid cache — every
    consumer probes for the map rather than assuming it — which is why this
    needed no ``_CACHE_VERSION`` bump.
    """
    profile, cache_dir = cached_profile

    assert not (cache_dir / "nvtx_kernel_map.parquet").exists()
    assert not (cache_dir / "nvtx_path_dict.parquet").exists()
    assert parquet_cache.is_cache_valid(str(profile)) is True

    stamp = json.loads((cache_dir / ".cache_version").read_text())
    assert stamp["nvtx_kernel_map_ready"] is False
    assert stamp["deferred_nvtx_kernel_map"] is True


def test_a_skill_that_does_not_attribute_nvtx_never_builds_the_map(cached_profile):
    """``overlap_breakdown`` is the skill the issue measured at 0.3 s against a
    19.8 s open. It must not pay for the map, and it must still answer."""
    from nsys_ai.profile import Profile
    from nsys_ai.skills.base import is_abstention
    from nsys_ai.skills.registry import get_skill

    profile, cache_dir = cached_profile
    with Profile(str(profile), cache_mode="parquet") as prof:
        rows = get_skill("overlap_breakdown").execute(prof.query_conn())

    assert not is_abstention(rows) and rows, "the control skill produced nothing"
    assert not (cache_dir / "nvtx_kernel_map.parquet").exists(), (
        "a skill that never attributes kernels to NVTX ranges paid for the map"
    )


def test_the_first_nvtx_query_publishes_the_map_into_the_cache(cached_profile):
    """The persisting half. Without it the deferral is a memory regression."""
    from nsys_ai.profile import Profile

    profile, cache_dir = cached_profile
    with Profile(str(profile), cache_mode="parquet") as prof:
        assert ensure_nvtx_kernel_map(prof.db) is True
        rows = prof.db.execute("SELECT count(*) FROM nvtx_kernel_map").fetchone()[0]

    assert rows > 0
    assert (cache_dir / "nvtx_kernel_map.parquet").is_file()
    assert (cache_dir / "nvtx_path_dict.parquet").is_file()

    stamp = json.loads((cache_dir / ".cache_version").read_text())
    assert stamp["nvtx_kernel_map_ready"] is True
    assert stamp["deferred_nvtx_kernel_map"] is False
    assert parquet_cache.is_cache_valid(str(profile)) is True, (
        "rewriting the stamp must not invalidate the cache — a torn or "
        "unparseable stamp costs a full rebuild"
    )

    # No staging directory survives a successful publish.
    assert [p.name for p in cache_dir.iterdir() if p.is_dir()] == []

    # A later open finds it by glob, with no accessor call at all.
    with Profile(str(profile), cache_mode="parquet") as prof2:
        again = prof2.db.execute("SELECT count(*) FROM nvtx_kernel_map").fetchone()[0]
    assert again == rows


def _age_the_stamp(cache_dir, seconds=5.0):
    """Move ``.cache_version``'s mtime *backwards* so the cache reads as stale.

    Read this before rewriting it. The obvious alternative — pushing the source
    SQLite's mtime forwards — produces a test that passes with the fix reverted:
    a source dated into the future is also newer than the publish's "now", so
    ``is_cache_valid`` stays False either way and nothing is being measured. The
    window this test exists for is ``build < source change < publish``, and with
    the build already done the only way to reach that ordering without sleeping
    is to age the stamp, leaving the source's real, present-day mtime in front
    of it and the publish's later still.
    """
    stamp = cache_dir / ".cache_version"
    aged = os.stat(stamp).st_mtime_ns - int(seconds * 1e9)
    os.utime(stamp, ns=(aged, aged))
    return aged


def test_a_stale_cache_is_not_resurrected_by_the_lazy_map_publish(cached_profile):
    """Publishing the map must not re-date the cache's freshness token (#323).

    ``is_cache_valid`` decides staleness from ``.cache_version``'s mtime, so the
    query-time stamp rewrite that records "map ready" is not the inert
    bookkeeping it looks like: an ``os.replace`` alone stamps it "now", which is
    newer than any source change that happened while this connection was open,
    and the cache this function had correctly rejected passes again.

    The two informational keys are asserted as well, so the fix cannot be met by
    skipping the publish — the stamp must still stop saying "deferred".
    """
    profile, cache_dir = cached_profile

    # A connection opened while the cache was still valid — the live web/tui/
    # chat/loop session in the issue.
    db = parquet_cache.open_cached_db(str(profile))
    try:
        _age_the_stamp(cache_dir)
        assert parquet_cache.is_cache_valid(str(profile)) is False, (
            "the aged stamp should already read as stale before the publish"
        )

        assert ensure_nvtx_kernel_map(db) is True
    finally:
        db.close()

    assert parquet_cache.is_cache_valid(str(profile)) is False, (
        "publishing the lazily built map resurrected a cache that was stale "
        "against its source; every later process is now served the old Parquet"
    )

    stamp = json.loads((cache_dir / ".cache_version").read_text())
    assert stamp["nvtx_kernel_map_ready"] is True
    assert stamp["deferred_nvtx_kernel_map"] is False


def test_a_changed_source_is_reread_after_a_lazy_map_publish(cached_profile):
    """The issue's reproduction, at the level a user feels it (#323).

    Re-capturing to the same path while a cached connection is live is the
    ordinary way a profile changes. The live connection keeps its own Parquet
    views for its lifetime — that is unchanged and out of scope — but the *next*
    process must see the new capture, and it does not if the map publish has
    quietly revalidated the old cache.
    """
    profile, cache_dir = cached_profile

    db = parquet_cache.open_cached_db(str(profile))
    try:
        cached_rows = db.execute("SELECT count(*) FROM nvtx").fetchone()[0]

        # Re-capture: same path, twice the NVTX events.
        con = sqlite3.connect(str(profile))
        con.execute("INSERT INTO NVTX_EVENTS SELECT * FROM NVTX_EVENTS")
        con.commit()
        source_rows = con.execute("SELECT count(*) FROM NVTX_EVENTS").fetchone()[0]
        con.close()
        assert source_rows == 2 * cached_rows > 0

        _age_the_stamp(cache_dir)
        assert ensure_nvtx_kernel_map(db) is True
    finally:
        db.close()

    later = parquet_cache.open_cached_db(str(profile))
    try:
        served = later.execute("SELECT count(*) FROM nvtx").fetchone()[0]
    finally:
        later.close()

    assert served == source_rows, (
        f"a later open was served {served} stale rows; the source has {source_rows}"
    )


def test_the_map_is_absent_from_schema_discovery_until_it_is_built(cached_profile):
    """Deferral hides the map from the catalog, not just from the cache dir.

    The skill system does not care — both consumers call
    ``ensure_nvtx_kernel_map`` before probing for the map — but the text-to-SQL
    surface reads the catalog directly: ``ai/backend/profile_db_tool.py``
    rewrites ``sqlite_master`` to ``SHOW TABLES`` and gives the result to the
    model, and the ``schema_inspect`` skill reads
    ``information_schema.columns``. Before this change the map was listed the
    moment a connection was handed out, because ``build_cache`` wrote its
    Parquet and ``open_cached_db``'s glob viewed it; now, on a cold cache, it is
    not there and a ``SELECT`` against it raises a Catalog Error where it used
    to return rows.

    That is a discoverability and performance change, not a correctness one —
    nothing in the agent's prompt names the map — but it was unpinned, so this
    test states both ends of the window: invisible before the first build,
    visible after it, and visible at open to every later process because the
    published Parquet becomes a view and DuckDB's ``SHOW TABLES`` lists views.
    """
    from nsys_ai.connection import DB_ERRORS
    from nsys_ai.profile import Profile

    def _catalog(db):
        listed = {r[0] for r in db.execute("SHOW TABLES").fetchall()}
        described = {
            r[0]
            for r in db.execute(
                "SELECT DISTINCT table_name FROM information_schema.columns"
            ).fetchall()
        }
        return listed, described

    profile, _cache_dir = cached_profile
    with Profile(str(profile), cache_mode="parquet") as prof:
        listed, described = _catalog(prof.db)
        assert "nvtx_kernel_map" not in listed, "the deferred map is still advertised"
        assert "nvtx_path_dict" not in listed, "the deferred dictionary is still advertised"
        assert "nvtx_kernel_map" not in described
        with pytest.raises(DB_ERRORS):
            prof.db.execute("SELECT count(*) FROM nvtx_kernel_map").fetchone()

        assert ensure_nvtx_kernel_map(prof.db) is True
        listed, described = _catalog(prof.db)
        assert {"nvtx_kernel_map", "nvtx_path_dict"} <= listed, (
            "the built map must be discoverable — SHOW TABLES lists DuckDB views too"
        )
        assert "nvtx_kernel_map" in described

    with Profile(str(profile), cache_mode="parquet") as prof2:
        listed, _ = _catalog(prof2.db)
        assert {"nvtx_kernel_map", "nvtx_path_dict"} <= listed, (
            "a published map must be visible at open, with no accessor call"
        )


def test_the_lazily_built_map_is_identical_to_the_eager_one(tmp_path, monkeypatch):
    """Same rows, same dictionary, whichever builder produced them.

    Both go through ``_build_nvtx_kernel_map_from_parquet``, so this is a check
    that the split into it did not change what the sweep is fed — the on-demand
    caller has no attached SQLite and no ``src_tables`` set, and if that mattered
    the rows would differ here. The two differ only in ``out_dir``: one writes
    into the cache directory, the other into a staging directory that is then
    renamed over it.
    """
    import duckdb

    from nsys_ai.profile import Profile

    lazy = tmp_path / "lazy.sqlite"
    eager = tmp_path / "eager.sqlite"
    shutil.copy(FIXTURE, lazy)
    shutil.copy(FIXTURE, eager)

    lazy_dir = Path(parquet_cache.build_cache(str(lazy)))
    with Profile(str(lazy), cache_mode="parquet") as prof:
        assert ensure_nvtx_kernel_map(prof.db) is True

    monkeypatch.setenv("NSYS_AI_ALWAYS_BUILD_NVTX_KERNEL_MAP", "1")
    eager_dir = Path(parquet_cache.build_cache(str(eager)))
    assert (eager_dir / "nvtx_kernel_map.parquet").is_file(), (
        "control: the env var must still force the eager build"
    )

    db = duckdb.connect()
    try:
        for name in ("nvtx_kernel_map", "nvtx_path_dict"):
            got = db.execute(
                f"SELECT * FROM '{lazy_dir / name}.parquet' ORDER BY ALL"
            ).fetchall()
            want = db.execute(
                f"SELECT * FROM '{eager_dir / name}.parquet' ORDER BY ALL"
            ).fetchall()
            assert got, f"{name} is empty on the lazy side"
            assert got == want, f"{name} differs between the lazy and eager builders"
    finally:
        db.close()


@pytest.mark.skipif(
    parquet_cache._fcntl is None,
    reason="build-lock degrades to no-op without POSIX fcntl; this assertion "
    "only holds on platforms where the lock is real.",
)
def test_concurrent_cursors_build_the_map_once(cached_profile):
    """Four threads, four ``.cursor()`` handles, one build.

    This race is not hypothetical and predates the deferral: run against a
    direct-attached profile before the per-database lock existed, all four
    threads ran the whole fetch-and-sweep and three then lost the ``CREATE
    TABLE`` to a DuckDB "Catalog write-write conflict on create"
    TransactionException. It was invisible because both production call sites
    swallow it under ``except DB_ERRORS: pass`` and then find the winner's
    table. Deferring the build makes it reachable far more often, since it now
    fires on whichever skill happens to run first.

    ``CREATE VIEW IF NOT EXISTS`` does not fix it — the conflict is raised
    before the existence check — so the DDL is serialised in Python instead.

    On this path two things could be doing the serialising: ``_MAP_BUILD_LOCK``
    and, inside it, ``_build_lock``'s flock. Measured with the Python lock
    removed, this test still passes — flock blocks threads within a process as
    well as across them, and the re-check inside it finds the published Parquet.
    So this test pins the *outcome* for the cached path; the lock itself is
    pinned by its in-memory counterpart below, where there is no lock file.
    """
    from nsys_ai.profile import Profile

    profile, cache_dir = cached_profile
    threads = 4
    calls = []
    calls_lock = threading.Lock()
    original = parquet_cache._build_nvtx_kernel_map_from_parquet

    def counting(db, src_dir, out_dir=None, **kwargs):
        with calls_lock:
            calls.append(1)
        # Long enough that every other thread is queued on the lock before this
        # one publishes; the sweep on this fixture is a few milliseconds.
        time.sleep(0.3)
        return original(db, src_dir, out_dir, **kwargs)

    parquet_cache._build_nvtx_kernel_map_from_parquet = counting
    try:
        with Profile(str(profile), cache_mode="parquet") as prof:
            barrier = threading.Barrier(threads)

            def runner():
                barrier.wait(timeout=10)
                # query_conn(), not db.cursor(): a worker thread's cursor only
                # resolves to the owning connection's cache directory because
                # query_conn registers it (#301). A raw cursor finds no cache
                # dir and silently takes the in-memory build instead — which is
                # what the first draft of this test measured.
                cur = prof.query_conn()
                assert ensure_nvtx_kernel_map(cur) is True
                return cur.execute("SELECT count(*) FROM nvtx_kernel_map").fetchone()[0]

            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
                futures = [pool.submit(runner) for _ in range(threads)]
                counts = [f.result(timeout=60) for f in futures]
    finally:
        parquet_cache._build_nvtx_kernel_map_from_parquet = original

    assert len(calls) == 1, (
        f"{len(calls)} threads ran the sweep; they are not being serialised"
    )
    assert len(set(counts)) == 1 and counts[0] > 0, (
        f"threads disagreed about the map: {counts}"
    )
    assert (cache_dir / "nvtx_kernel_map.parquet").is_file()


def test_a_read_only_cache_directory_degrades_to_the_in_memory_build(cached_profile):
    """``open_cached_db``'s fast path exists so a prebuilt cache works on a
    read-only mount. The lazy build must not break that: it cannot create its
    lock file or its staging directory there, and the only acceptable answer is
    to fall back to the in-memory build, not to raise into the caller.
    """
    from nsys_ai.profile import Profile

    profile, cache_dir = cached_profile
    with Profile(str(profile), cache_mode="parquet") as prof:
        mode = cache_dir.stat().st_mode
        parent_mode = cache_dir.parent.stat().st_mode
        # The lock file lives beside the cache dir, the staging dir inside it.
        os.chmod(cache_dir, 0o500)
        os.chmod(cache_dir.parent, 0o500)
        try:
            assert ensure_nvtx_kernel_map(prof.db) is True
            rows = prof.db.execute("SELECT count(*) FROM nvtx_kernel_map").fetchone()[0]
        finally:
            os.chmod(cache_dir.parent, parent_mode)
            os.chmod(cache_dir, mode)

    assert rows > 0, "the in-memory fallback produced no map"
    assert not (cache_dir / "nvtx_kernel_map.parquet").exists(), (
        "something was written into a read-only cache directory"
    )
    assert [p.name for p in cache_dir.iterdir() if p.is_dir()] == [], (
        "a staging directory was left behind"
    )


def test_concurrent_cursors_build_the_in_memory_map_once():
    """The same guarantee on the backend that has no cache to write into.

    This race is real and predates the deferral. Reproduced on this fixture with
    the lock removed: four threads on their own ``.cursor()`` handles all ran the
    whole fetch-and-sweep, and three then lost the ``CREATE TABLE`` to a DuckDB
    ``TransactionException: Catalog write-write conflict on create`` — four times
    the work and four times the memory to produce one table. It looked harmless
    because both production call sites swallow the exception under ``except
    DB_ERRORS: pass`` and then find the winner's table.

    The cached path is serialised by ``_build_lock``'s flock, which works across
    threads as well as processes. This path has no such file, so the Python lock
    is the only thing standing between four threads and four sweeps — and
    ``ensure_nvtx_kernel_map`` is reached far more often now that the map is not
    precomputed at open.
    """
    con = _duckdb_profile()
    threads = 4
    calls = []
    calls_lock = threading.Lock()
    original = parquet_cache._sweep_nvtx_kernel_map

    def counting(kr_rows, nvtx_rows):
        with calls_lock:
            calls.append(1)
        time.sleep(0.3)
        return original(kr_rows, nvtx_rows)

    parquet_cache._sweep_nvtx_kernel_map = counting
    try:
        barrier = threading.Barrier(threads)

        def runner():
            barrier.wait(timeout=10)
            cur = con.cursor()
            assert ensure_nvtx_kernel_map(cur) is True
            return cur.execute("SELECT count(*) FROM nvtx_kernel_map").fetchone()[0]

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
            futures = [pool.submit(runner) for _ in range(threads)]
            counts = [f.result(timeout=60) for f in futures]
    finally:
        parquet_cache._sweep_nvtx_kernel_map = original

    assert len(calls) == 1, (
        f"{len(calls)} of {threads} threads ran the sweep; _MAP_BUILD_LOCK is not "
        "serialising the in-memory build"
    )
    assert counts == [2] * threads, f"threads disagreed about the map: {counts}"
