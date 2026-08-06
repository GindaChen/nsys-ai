"""Child process for ``test_analysis_memory.py``. Not collected by pytest.

Runs one fixed analysis workload under ``tracemalloc`` and prints the peak
Python-heap allocation as JSON on stdout. Lives in its own process because the
number is only reproducible from a fresh interpreter: inside a pytest session the
first measurement pays for lazily-imported skill modules and DuckDB, and later
ones do not, which was measured as 36.35 / 4.88 / 4.84 MB for three identical
in-process repetitions.

**Three windows, not one.** The peak is reported separately for opening the
profile, for running the skills cold, and for re-running attribution once the
nvtx_kernel_map build has been paid — because they are not the same measurement
and an earlier single-number version of this probe silently conflated the first
two. Measured on ``h100_2gpu_1s.sqlite``: with ``SKILLS`` emptied the single
number was 4.99 MB against 5.00 MB with all three skills — 100% of it was the
Parquet cache ETL inside ``Profile.__init__``, and a skill-layer memory
regression was invisible under it. Split, and with the one-time costs below
taken out, the same run reports 0.06 / 4.93 / 0.27 MB on ``cache_mode="auto"``
and 0.06 / 6.62 / 0.27 MB on ``cache_mode="direct"``. Each has a mutation that
moves it — see the test module.

The third window was added when the second stopped bounding what it used to; the
comment at that window says why, and it is not optional bookkeeping.

The workdir is supplied by the parent rather than made here. The parent kills
this process on timeout, and a killed process does not run its ``finally``, so
whoever can still clean up has to own the directory.

Usage: ``python tests/_memory_probe.py <profile.sqlite> <workdir> <cache_mode>``
"""

import json
import shutil
import sys
import time
import tracemalloc
from pathlib import Path

# Three skills with different shapes: top_kernels aggregates, gpu_idle_gaps
# sweeps the kernel timeline, and nvtx_kernel_map attributes kernels to NVTX
# ranges. The last is the expensive one, and both cache modes now pay for it in
# the *skills* window — but by different routes, which is why both are measured:
#
#   cache_mode="auto"   — the map is deferred out of Profile.__init__, so the
#                         skill's first call reaches
#                         parquet_cache.materialize_cached_nvtx_kernel_map. That
#                         streams its NVTX ranges out of the cached Parquet and
#                         writes the result back into the cache, so it is paid
#                         once per profile rather than once per process.
#   cache_mode="direct" — there is no cache directory to write into, so the same
#                         call falls through to ensure_nvtx_kernel_map's
#                         in-memory build, which fetchall()s the whole NVTX table
#                         and the kernel/runtime join into Python tuples. That is
#                         the largest single allocation this repo makes.
#
# The gap between the two — 4.93 MB against 6.62 MB on this fixture — is the
# streaming build measured against the materialising one, on identical output.
SKILLS = ("top_kernels", "gpu_idle_gaps", "nvtx_kernel_map")


def main() -> int:
    source = Path(sys.argv[1])
    workdir = Path(sys.argv[2])
    cache_mode = sys.argv[3]

    # Two copies. The cache is built fresh for each, because measuring against a
    # cache that may or may not already exist is how this number stops being
    # reproducible.
    warm = workdir / f"{cache_mode}-warmup-{source.name}"
    work = workdir / f"{cache_mode}-{source.name}"
    shutil.copy(source, warm)
    shutil.copy(source, work)

    from nsys_ai.nvtx_attribution import attribute_kernels_to_nvtx
    from nsys_ai.profile import Profile
    from nsys_ai.skills.base import is_abstention
    from nsys_ai.skills.registry import get_skill

    # Import cost is deliberately outside the measurement, and getting it out is
    # most of the work. Measured against a probe that only imported nsys_ai
    # first, 23 of the 33.9 MB peak was module bytecode from duckdb and pyarrow
    # being imported lazily during Profile construction — so the data term was a
    # third of the number, and a doubling of it would have sat under any
    # plausible ceiling. Importing them up front leaves a measurement that is
    # mostly the profile.
    skills = {name: get_skill(name) for name in SKILLS}
    import duckdb  # noqa: F401
    import pyarrow  # noqa: F401
    import pyarrow.parquet  # noqa: F401

    from nsys_ai import parquet_cache, sql_compat  # noqa: F401

    # pandas is not a declared dependency but duckdb reaches for it, and it is
    # 18 MB of module objects when it lands inside the window. Hoisted when
    # present; when it is absent it never enters the window at all, so either
    # way the measurement is of the profile and not of an import.
    try:
        import pandas  # noqa: F401
    except ImportError:
        pass

    # One complete throwaway pass on a separate copy, for the same reason the
    # imports above are hoisted: everything a first run pays once has to be paid
    # before the window opens, or the number is a property of the machine's
    # state rather than of the code.
    #
    # The explicit import list above cannot cover this on its own. Modules
    # imported lazily *inside* skill execution — nvtx_attribution and what it
    # reaches — were still landing in the measured window, and compiling their
    # bytecode cost 0.61 MB: the skills window measured 1.96 MB with a cold
    # __pycache__ and 1.35 MB with a warm one, on identical code. CI checks out
    # clean, so without this the first number would be the CI number and the
    # second the local one, and the difference is 46%.
    with Profile(str(warm), cache_mode=cache_mode) as warmup_prof:
        for name, skill in skills.items():
            skill.execute(
                warmup_prof.query_conn(), **({"limit": 20} if name == "top_kernels" else {})
            )

    tracemalloc.start()
    started = time.perf_counter()
    prof = Profile(str(work), cache_mode=cache_mode)
    try:
        engine = "duckdb" if prof.db is not None else "sqlite3"
        _, open_peak = tracemalloc.get_traced_memory()
        open_s = time.perf_counter() - started

        # Second window. reset_peak leaves the live blocks in place — the open
        # window's retained objects still count against this one, which is what
        # a caller experiences.
        tracemalloc.reset_peak()
        # Nothing is swallowed here. A skill that raises or abstains would leave
        # the measured window smaller than it looks, so the parent is told which
        # rows each skill produced and asserts they are real.
        rows: dict[str, int] = {}
        for name, skill in skills.items():
            kwargs = {"limit": 20} if name == "top_kernels" else {}
            out = skill.execute(prof.query_conn(), **kwargs)
            rows[name] = -1 if is_abstention(out) else len(out)
        _, skills_peak = tracemalloc.get_traced_memory()

        # Third window: attribution against a map that already exists.
        #
        # It exists because the second window stopped bounding what it used to.
        # Both cache modes now build nvtx_kernel_map inside the skills window
        # — "auto" since the build was deferred out of Profile.__init__ — and
        # that build's peak (4.93 MB cached, 6.62 MB direct) sits so far above
        # the attribution query's that the query's own regression vanished
        # underneath it. Measured, on this fixture: dropping the LIMIT push-down
        # in attribute_kernels_to_nvtx takes auto/skills from 0.27 to 1.53 MB
        # before the deferral and moves it 0.00 MB after.
        #
        # ``attribute_kernels_to_nvtx`` directly, not the skill a second time.
        # Skills memoize their rows per connection and resolved parameters
        # (#295), so re-executing one measures a dict lookup — the first draft of
        # this window did exactly that and reported 0.28 MB whether the push-down
        # was there or not, which is the same "guard that guards nothing" this
        # file was written to avoid.
        tracemalloc.reset_peak()
        attrib_rows = len(attribute_kernels_to_nvtx(prof.query_conn(), limit=50))
        _, attrib_peak = tracemalloc.get_traced_memory()
    finally:
        prof.close()
    elapsed = time.perf_counter() - started
    tracemalloc.stop()

    print(
        json.dumps(
            {
                "engine": engine,
                "cache_mode": cache_mode,
                "input_mb": round(source.stat().st_size / 1e6, 3),
                "open_peak_mb": round(open_peak / 1e6, 2),
                "skills_peak_mb": round(skills_peak / 1e6, 2),
                "attrib_peak_mb": round(attrib_peak / 1e6, 2),
                "rows": rows,
                "attrib_rows": attrib_rows,
                "open_s": round(open_s, 2),
                "workload_s": round(elapsed, 2),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
