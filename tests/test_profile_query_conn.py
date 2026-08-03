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


def test_concurrent_skill_runs_agree_with_the_sequential_answer():
    """The behaviour the accessor exists to protect.

    Run through a shared connection this returns wrong row counts silently; the
    sequential baseline is the oracle.
    """
    with Profile(str(FIXTURE)) as prof:
        if prof.db is None:  # pragma: no cover
            pytest.skip("DuckDB not in use")
        skill = get_skill("top_kernels")
        baseline = skill.execute(prof.query_conn(), limit=5)

        # Bounded wait on purpose. Removing the per-thread handle does not just
        # return wrong rows — threads contending for one connection's result set
        # deadlocked and ran past ten minutes when this was mutation-checked. A
        # regression must fail the build, not hang it.
        batches = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            for _ in range(5):
                futures = [
                    ex.submit(lambda: skill.execute(prof.query_conn(), limit=5)) for _ in range(8)
                ]
                batches.append([f.result(timeout=60) for f in futures])

    for batch in batches:
        for rows in batch:
            assert rows == baseline, "a concurrent run disagreed with the sequential answer"


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
