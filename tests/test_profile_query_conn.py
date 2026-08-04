"""`Profile.query_conn()` — one accessor for two decisions that were one.

Which connection to query was spelled out in eleven places across eight modules,
in three spellings, while `Profile.__init__` had already decided it. Which
*handle* to query through was decided nowhere, and the default was wrong under
threads: DuckDB keeps the pending result set on the connection, so `execute`
and `fetch` are individually atomic but not atomic as a pair. Two threads on one
handle clobber each other and return wrong rows with nothing raised.

Measured against a plain DuckDB connection, outside this repo entirely: a shared
connection was wrong in 6 of 6 trials, per-thread cursors in 0 of 6. The failure
mode includes `fetchone()` returning None because another thread had already
consumed the result.
"""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nsys_ai.profile import Profile
from nsys_ai.skills.registry import get_skill

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"


def test_the_owning_thread_keeps_the_connection_itself():
    """Existing single-threaded callers must stay on exactly the path they were on.

    This is not cosmetic. Scratch tables made with `CREATE TEMP TABLE` are
    visible only to the handle that created them, so handing the owning thread a
    cursor would change behaviour for code that has always worked.
    """
    with Profile(str(FIXTURE)) as prof:
        if prof.db is None:  # pragma: no cover - cache unavailable
            pytest.skip("DuckDB not in use")
        assert prof.query_conn() is prof.db


def test_worker_threads_get_their_own_reused_handle():
    """One cursor per thread, reused — not one per call.

    Per-call cursors would work too, but every cursor is a distinct key in the
    per-connection memo from #295, so churning them churns the cache. Bound to
    the pool size, the fragmentation is bounded too.
    """
    with Profile(str(FIXTURE)) as prof:
        if prof.db is None:  # pragma: no cover
            pytest.skip("DuckDB not in use")
        seen: set[int] = set()
        lock = threading.Lock()

        def grab(_):
            handle = prof.query_conn()
            with lock:
                seen.add(id(handle))
            return handle is prof.db

        with ThreadPoolExecutor(max_workers=4) as ex:
            for _ in range(10):  # ten waves over one persistent pool
                results = list(ex.map(grab, range(8)))

        assert not any(results), "a worker thread was handed the shared connection"
        assert len(seen) <= 4, f"expected at most one handle per worker, got {len(seen)}"


def test_repeated_memo_reads_hand_out_independent_copies():
    """Every read of one memoized answer must get its own row dicts.

    This test used to claim it verified concurrent skill runs against a
    sequential oracle, on a thread pool. It did neither, and both claims were
    measured false. Recorded here rather than quietly deleted, because the way it
    failed is the way any concurrency test in this repo will fail:

    * Its oracle was the cache. Across all forty concurrent calls the process
      issued exactly *one* distinct SQL statement — everything after the
      sequential baseline was served by the per-connection skill memo (#295),
      shared with every derived cursor (#302). It was comparing copies of the
      baseline against the baseline.
    * Its bound did not bind. The comment said a regression must fail the build
      and not hang it, and ``f.result(timeout=60)`` looked like it enforced
      that. It does not: the TimeoutError leaves the ``with
      ThreadPoolExecutor(...)`` block, ``__exit__`` calls ``shutdown(wait=True)``
      and joins the deadlocked workers forever. Mutated to hand every thread the
      shared connection, this test ran past 120 seconds of wall clock on a couple
      of seconds of CPU before being killed by hand — and pytest collects this
      file seven files *before* the bounded probe that is supposed to catch that
      regression, so under it CI hung to the job timeout instead of failing.

    The real property — concurrent skill answers equal the sequential ones, with
    the memo cleared so the calls reach the database, bounded by a subprocess
    kill so a deadlock is a failure — lives in ``tests/test_skill_concurrency.py``.

    What is left here needs no threads at all, and therefore cannot hang: the
    memo's isolation is fully observable from repeated sequential reads. Both
    halves of it are asserted, because they are two different copies guarding two
    different directions:

    * ``Skill.execute`` copies on the way *out* of the cache, so no two readers
      hold the same dict. Without it, one reader's in-place edit is visible to
      every later reader.
    * It copies on the way *in* as well, so the first caller — which is handed
      the freshly built list, not a copy of it — cannot reach the cache either.
      That one is invisible to an identity check and only shows up if you
      actually mutate what the first call returned, which is what the last block
      does.
    """
    with Profile(str(FIXTURE)) as prof:
        if prof.db is None:  # pragma: no cover
            pytest.skip("DuckDB not in use")
        skill = get_skill("top_kernels")

        # The first call builds and stores; every later one is a memo hit. Three
        # reads, because the copy-on-read defect only shows between two *hits* —
        # the builder's own list is distinct from the cache either way.
        first = skill.execute(prof.query_conn(), limit=5)
        assert first, "the fixture returned no kernels, so nothing below is tested"
        reads = [first] + [skill.execute(prof.query_conn(), limit=5) for _ in range(2)]

        for rows in reads[1:]:
            assert rows == first, "a memoized read returned different values"

        seen_ids: set[int] = set()
        for rows in reads:
            for row in rows:
                assert id(row) not in seen_ids, (
                    "two readers were handed the same row dict — the memo stopped "
                    "copying on the way out"
                )
                seen_ids.add(id(row))

        # Copy-on-store: poison what the *builder* was handed. If the cache kept
        # that same list, the poison is served to everyone after it.
        marker = "poisoned-by-the-first-caller"
        first[0]["kernel_name"] = marker
        after = skill.execute(prof.query_conn(), limit=5)
        assert after[0].get("kernel_name") != marker, (
            "the first caller's in-place edit reached the cache — the memo stopped "
            "copying on the way in"
        )


def test_a_sqlite_only_profile_gets_the_sqlite_connection():
    """No cursor games where none are needed — sqlite3 is opened with
    `check_same_thread=False` and serialises internally."""
    conn = sqlite3.connect(FIXTURE, check_same_thread=False)
    try:
        prof = Profile._from_conn(conn)
        assert prof.db is None
        assert prof.query_conn() is conn
    finally:
        conn.close()


def test_the_accessor_works_during_construction():
    """`_discover()` queries through this before the owning thread is recorded.

    Constructing a Profile at all is the assertion; an ordering mistake here
    raises AttributeError from deep inside __init__.
    """
    with Profile(str(FIXTURE)) as prof:
        assert prof.meta is not None
        assert prof.query_conn() is not None


# "A second handle must not mean a second cache" — the property that worker
# threads share the owning connection's bag (#302) — used to be asserted here,
# in-process, on a ThreadPoolExecutor with f.result(timeout=60). It has moved to
# tests/test_skill_concurrency.py, which runs it in a killed subprocess.
#
# It is not memo-only work, which is what made it unsafe to keep here.
# Skill.execute calls compute_profiler_overhead_ns *before* the memo lookup, and
# that queries the database on every call — so eight futures issue eight real
# concurrent statements. Under the regression this whole file exists to catch,
# they contend for one DuckDB connection's pending result set and never return:
# measured hanging past 120 seconds, with the result() timeout doing nothing
# because the TimeoutError leaves the executor's with-block and
# shutdown(wait=True) joins the stuck workers forever. pytest collects this file
# well before the bounded probe, so the hang arrived first and CI would have run
# to its job timeout rather than failing.
#
# Everything left in this file is either single-threaded or touches no database
# from a thread, so this file cannot hang.


def test_close_drops_the_per_thread_handles():
    """Cursors are handles on the connection being closed.

    Left in the thread-local, a worker thread outliving the profile would be
    handed a dead cursor rather than an error or a fresh one.
    """
    prof = Profile(str(FIXTURE))
    if prof.db is None:  # pragma: no cover
        prof.close()
        pytest.skip("DuckDB not in use")

    def touch():
        prof.query_conn()

    worker = threading.Thread(target=touch)
    worker.start()
    worker.join()

    prof.close()
    assert getattr(prof._thread_handles, "conn", None) is None, (
        "close() left a cursor in the thread-local"
    )


def test_a_type_check_does_not_materialise_a_cursor():
    """`_has_duckdb` asks which engine is in use; answering it should not
    allocate a handle as a side effect."""
    from nsys_ai.overlap import _has_duckdb

    with Profile(str(FIXTURE)) as prof:
        if prof.db is None:  # pragma: no cover
            pytest.skip("DuckDB not in use")
        seen = {}

        def probe():
            _has_duckdb(prof)
            seen["handle"] = getattr(prof._thread_handles, "conn", None)

        worker = threading.Thread(target=probe)
        worker.start()
        worker.join()

    assert seen["handle"] is None, "a type check created a thread-local cursor"
