"""Child process for ``test_skill_concurrency.py``. Not collected by pytest.

Runs each named skill sequentially to get an oracle, then runs it again from a
thread pool and reports how many concurrent answers disagreed. Prints one JSON
object on stdout.

It lives in a subprocess because the failure mode this guards is a *deadlock*,
not a mismatch. Two threads sharing one DuckDB connection contend for the single
pending result set on it; when that regression was mutation-checked the run went
past four minutes of wall time on 2.5 seconds of CPU, and one variant dumped
core. ``future.result(timeout=...)`` does not bound that: the TimeoutError
propagates out of the ``with ThreadPoolExecutor(...)`` block, ``__exit__`` calls
``shutdown(wait=True)``, and that joins the stuck workers forever. A subprocess
with a kill timeout is the only bound that holds.

The workdir is supplied by the parent rather than made here, for the same
reason: the parent kills this process on timeout, a killed process never runs
its ``finally``, and a deadlock probe that leaks a 2.5 MB temp directory on
every failed run is a probe that fills /tmp during a bisect.

Usage: ``python tests/_concurrency_probe.py <profile.sqlite> <workdir> <skill,skill,...>``
"""

import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WAVES = 3
FUTURES_PER_WAVE = 8
WORKERS = 4

# Skills whose defaults return too much to compare cheaply get a small limit.
_LIMITED = {"top_kernels", "gpu_idle_gaps"}


def _clear_probe_bag(conn) -> None:
    """Drop the per-connection memo bag so the next call really queries.

    Without this the test measures the memo, not the database. #295 memoizes
    skill results per connection keyed on resolved parameters, and #302 shares
    one bag between a connection and every cursor derived from it — so a naive
    concurrent run issues one statement and reads it back 39 times. Clearing the
    bag between waves is what makes the eight racing calls actually contend.

    Note what is cleared: the *whole* bag, not just the ``skill:`` entries. The
    bag also holds schema-resolution results (``resolve_activity_tables``,
    ``detect_nvtx_text_id``, the ``nvtx_kernel_map`` shape probe), so every wave
    races those as well as the skill query. That is deliberate — they are read
    on the same connection from the same threads and are part of what a
    concurrent skill call actually does — but it means a mismatch reported here
    is not necessarily in the skill's own SQL.
    """
    from nsys_ai import connection as conn_mod

    owner = conn_mod._cache_owner(conn)
    bag = conn_mod._duck_probe_bags.get(owner)
    if bag is not None:
        bag.clear()


def main() -> int:
    source = Path(sys.argv[1])
    workdir = Path(sys.argv[2])
    names = sys.argv[3].split(",")

    work = workdir / source.name
    shutil.copy(source, work)

    from nsys_ai.profile import Profile
    from nsys_ai.skills import base as skills_base
    from nsys_ai.skills.registry import get_skill

    # Count real executions: set_cached_skill_rows runs only on a memo miss,
    # so it is an exact count of the calls that went to the database.
    executions: dict[str, int] = {name: 0 for name in names}
    exec_lock = threading.Lock()
    original_set = skills_base.set_cached_skill_rows

    def counting_set(conn, key, rows):
        skill_name = key.split(":")[1] if key.startswith("skill:") else ""
        if skill_name in executions:
            with exec_lock:
                executions[skill_name] += 1
        return original_set(conn, key, rows)

    skills_base.set_cached_skill_rows = counting_set

    started = time.perf_counter()
    with Profile(str(work)) as prof:
        if prof.db is None:
            print(json.dumps({"engine": "sqlite3"}))
            return 0
        open_s = time.perf_counter() - started

        # ── Phase 1: worker threads must share the owning connection's bag ──
        # Runs before anything clears the memo, because that is the whole point:
        # the owner populates it, and every worker thread must find it. Keyed on
        # the handle instead of the owning connection, each thread starts empty
        # and re-executes — measured as four extra executions across four
        # threads before #302.
        #
        # This lives in the probe rather than in-process next to the accessor
        # tests it belongs with, because it is not memo-only work: Skill.execute
        # calls compute_profiler_overhead_ns *before* the memo lookup, and that
        # queries the database on every single call. Under the shared-connection
        # regression eight of those contend for one pending result set and never
        # return — measured hanging past 120s, with f.result(timeout=60) doing
        # nothing about it, in a file pytest collects before this one. Only a
        # killed subprocess bounds it.
        share_skill = get_skill(names[0])
        share_kwargs = {"limit": 5} if names[0] in _LIMITED else {}
        share_skill.execute(prof.query_conn(), **share_kwargs)
        after_owner = executions[names[0]]
        share_pool = ThreadPoolExecutor(max_workers=WORKERS)
        try:
            share_futures = [
                share_pool.submit(lambda: share_skill.execute(prof.query_conn(), **share_kwargs))
                for _ in range(FUTURES_PER_WAVE)
            ]
            for future in share_futures:
                future.result()
        finally:
            share_pool.shutdown(wait=True)
        cache_sharing = {
            "skill": names[0],
            "owner_executions": after_owner,
            "worker_extra_executions": executions[names[0]] - after_owner,
        }

        # ── Phase 2: concurrent answers must equal the sequential ones ──
        # Executions are reported as a delta from here, so phase 1's run does
        # not inflate the count the parent uses to prove phase 2 reached the
        # database.
        before_phase2 = dict(executions)
        mismatches: dict[str, int] = {}
        # Baseline shape, reported so the parent can refuse to compare [] to [].
        # A skill that returns nothing on this fixture makes its agreement case
        # a tautology, which is how nccl_breakdown sat in this list asserting
        # nothing at all.
        baseline_rows: dict[str, int] = {}
        for name in names:
            skill = get_skill(name)
            kwargs = {"limit": 5} if name in _LIMITED else {}

            def run():
                return skill.execute(prof.query_conn(), **kwargs)

            _clear_probe_bag(prof.query_conn())
            baseline = run()
            baseline_rows[name] = -1 if skills_base.is_abstention(baseline) else len(baseline)

            bad = 0
            pool = ThreadPoolExecutor(max_workers=WORKERS)
            try:
                for _ in range(WAVES):
                    _clear_probe_bag(prof.query_conn())
                    futures = [pool.submit(run) for _ in range(FUTURES_PER_WAVE)]
                    for future in futures:
                        if future.result() != baseline:
                            bad += 1
            finally:
                pool.shutdown(wait=True)
            mismatches[name] = bad

    print(
        json.dumps(
            {
                "engine": "duckdb",
                "open_s": round(open_s, 3),
                "total_s": round(time.perf_counter() - started, 3),
                "mismatches": mismatches,
                "executions": {name: executions[name] - before_phase2[name] for name in names},
                "baseline_rows": baseline_rows,
                "cache_sharing": cache_sharing,
                "waves": WAVES,
                "futures_per_wave": FUTURES_PER_WAVE,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
