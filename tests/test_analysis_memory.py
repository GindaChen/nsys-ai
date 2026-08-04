"""Ceilings on how much Python heap an analysis run is allowed to allocate.

Nothing in this repo bounded memory anywhere. The working set of a run is
roughly two to three times the capture size, and that ratio is what decides
whether a tens-of-GB profile can be opened at all — but it lived in session
notes, so a change that doubled it would have gone through CI green.

What is measured, and what each number actually covers
-----------------------------------------------------

Four numbers, from two child processes. The split matters: an earlier version of
this file reported one peak for "open plus three skills" and claimed it bounded
both. It did not. Re-run with the skill list emptied, that single number was
4.99 MB against 5.00 MB with all three skills — every measurable byte of it was
the Parquet-cache ETL inside ``Profile.__init__``, and a skill-layer regression
was invisible underneath it. Each window now carries its own ceiling and its own
mutation:

===============  ========  =======  ==================================================
window           baseline  ceiling  mutation that moves it (all measured here)
===============  ========  =======  ==================================================
auto / open       4.85 MB   5.5 MB  ``list(_stream_nvtx_ranges(...))`` -> 6.09 MB
auto / skills     0.27 MB   0.8 MB  drop the ``LIMIT`` push-down in
                                    ``nvtx_attribution`` -> 1.53 MB
direct / open     0.06 MB   1.0 MB  route ``cache_mode="direct"`` through
                                    ``open_cached_db`` -> 4.85 MB
direct / skills   6.62 MB   7.4 MB  dict instead of tuple buckets in
                                    ``_sweep_nvtx_kernel_map`` -> 8.06 MB
===============  ========  =======  ==================================================

``cache_mode="direct"`` is the second case because it is the path a real
tens-of-GB capture takes: ``Profile.__init__`` sends anything over 50 MB to
``open_direct_sqlite``, which does no ETL at all. The two cases are near
mirror images, and the shape is worth stating plainly because it is where the
scale risk lives:

* the cached path spends its memory *opening* (4.85 MB) and almost none running
  skills (0.27 MB), because ``_build_nvtx_kernel_map`` streams and the skills
  then read a precomputed map;
* the direct path spends almost nothing opening (0.06 MB — the zero-ETL promise
  of that path, in a number) and 6.62 MB running skills, because the first
  NVTX-attribution call reaches ``ensure_nvtx_kernel_map``, which ``fetchall()``s
  the entire NVTX table and the whole kernel/runtime join into Python tuples.

So the path taken by the largest captures is the one that materialises most, at
2.9x the input against the cached path's 2.1x. That is not asserted as good; it
is pinned so it does not get worse silently, and so that fixing it (streaming
the on-demand build the way the cache build already streams) shows up as a
deliberate change to a stated number.

Why tracemalloc and not peak RSS
--------------------------------

The obvious test — assert peak RSS stays under a multiple of the input size —
was built and rejected on measurement. The reasons are recorded so it does not
get proposed again:

* Peak RSS on a fixture this size is a *floor*, not a ratio: 176.6 MB for a
  0.057 MB profile, 212.2 MB for a 2.437 MB one. ~85% of it is interpreter and
  DuckDB startup, so the apparent "ratio" spans 3230x to 11.4x purely because
  the denominator moves.
* Environment noise swamps the signal. DuckDB thread count alone moved peak RSS
  by ±7% (185.8 MB at one thread, 212.1 MB at twelve). CI has 4 vCPU; this was
  measured at 12.
* The denominator is not stable either: analysing a profile in place grows the
  .sqlite file by 71%, because it builds indices in the source database.
* It failed its own mutation check. Materialising a 320k-row NVTX generator
  into a list gave peak RSS of 334.7-372.8 MB against a clean 284.3-360.7 MB —
  overlapping distributions, no threshold between them.

``tracemalloc`` peak, taken from a fresh interpreter with the imports hoisted
out of the measured window, is the instrument that works. It counts Python
allocations only, so it is blind to DuckDB's C++ arena and to interpreter
startup, and every one of the four numbers above reproduced to within 0.01 MB on
five consecutive runs, under ``taskset -c 0,1``, and with eight CPU hogs
competing for the machine. That is the axis that killed the RSS version.

Being blind to the C++ arena is a real limitation and not a footnote: a
regression that moves work into DuckDB, numpy or pyarrow buffers will not show
up here at all. One measured example — swapping ``fetchall()`` for
``fetchdf().to_records()`` in ``ensure_nvtx_kernel_map`` moved the direct/skills
number by 0.01 MB while genuinely allocating more, because the data landed in
numpy. These ceilings bound the Python heap, which is where this repo's own row
handling lives, and nothing more.

Getting one-time costs out of the window is most of the work, not a detail, and
it took three attempts:

* A first version measured 33.92 MB, of which 23 MB was module bytecode from
  duckdb, pyarrow and pandas being imported lazily *during* ``Profile``
  construction. Hence the explicit import block at the top of the probe.
* Explicit imports are not enough, because the skills reach for modules the
  probe does not name. With a cold ``__pycache__`` the skills window measured
  1.96 MB and with a warm one 1.35 MB — 46% apart on identical code, and CI
  always checks out cold. Hence the throwaway warm-up pass in the probe, after
  which the same measurement is 0.27 MB either way.
* Three identical repetitions inside one pytest session returned 36.35 / 4.88 /
  4.84 MB, because the first pays for those imports and the rest do not. Hence
  the subprocess.

Each of those made the number smaller and the data term a larger share of it.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PROBE = Path(__file__).resolve().parent / "_memory_probe.py"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"

# Baselines and ceilings, measured on Python 3.12.3 / DuckDB 1.5.0 /
# pyarrow 22.0.0 / pandas 2.3.3 against the 2.269 MB fixture. Each ceiling sits
# above its baseline and below the mutation named in the module docstring.
#
#   cache_mode -> (open ceiling MB, skills ceiling MB, measured open, measured skills)
#
# The measured pair is carried so the failure message can say how far the number
# moved, not just that it is over. Only one interpreter was available where this
# was set and CI runs three, so every number is printed on every run — including
# a green one. If a CI baseline comes in materially above the measured value,
# re-derive the ceiling from the printed numbers rather than nudging one until
# it passes.
#
# Headroom is deliberately uneven, because it is set by the gap to the mutation
# rather than by taste. direct/skills is the tightest at 12% over baseline and
# 9% under its mutation; if that one ever flakes on another interpreter it is the
# first to re-derive, and the honest fix is a bigger gap, not a bigger ceiling.
CASES = {
    "auto": (5.5, 0.8, 4.85, 0.27),
    "direct": (1.0, 7.4, 0.06, 6.62),
}

PROBE_TIMEOUT_S = 120


def _emit(request, message: str) -> None:
    """Write to the terminal even when the test passes.

    pytest captures stdout for a passing test, and the whole point of printing
    the number is that a *green* build reports it. The terminal reporter writes
    outside the capture.
    """
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(message)
    else:  # pragma: no cover - only under a stripped-down pytest
        print(message)


@pytest.fixture(scope="module", params=sorted(CASES))
def measured(request):
    """One child process per cache mode.

    The working directory belongs to this fixture, not to the child. On timeout
    the child is killed, and a killed process never reaches its own cleanup — so
    the side that survives has to be the side that owns the directory.
    """
    cache_mode = request.param
    workdir = tempfile.mkdtemp(prefix="nsys-mem-probe-")
    try:
        result = subprocess.run(
            [sys.executable, str(PROBE), str(FIXTURE), workdir, cache_mode],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(workdir, ignore_errors=True)
        pytest.fail(f"the memory probe did not finish within {PROBE_TIMEOUT_S}s")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if result.returncode != 0:
        pytest.fail(
            f"the memory probe exited {result.returncode}\n"
            f"stdout: {result.stdout.strip()}\nstderr: {result.stderr.strip()[-2000:]}"
        )

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if payload["engine"] != "duckdb":
        pytest.skip("requires duckdb — the ceilings were measured on the DuckDB path")
    payload["ceilings"] = CASES[cache_mode]
    return payload


def test_the_measured_workload_actually_ran(measured, request):
    """Guards the guard: an empty workload has a very low peak.

    Every ceiling below is an upper bound, so anything that stops the workload
    doing its work makes them pass harder. A skill that abstained, returned
    nothing, or was quietly skipped would leave a comfortably green build
    measuring almost nothing — which is the failure mode that made the previous
    single-window version of this file measure the cache ETL while claiming to
    measure skills.
    """
    rows = measured["rows"]
    _emit(
        request,
        f"[memory:{measured['cache_mode']}] rows "
        + ", ".join(f"{name}={count}" for name, count in sorted(rows.items())),
    )
    abstained = sorted(name for name, count in rows.items() if count < 0)
    assert not abstained, (
        f"{abstained} abstained on this fixture, so the measured window does not "
        "include their work. Replace them with skills that run here, or the "
        "ceilings below bound nothing."
    )
    empty = sorted(name for name, count in rows.items() if count == 0)
    assert not empty, (
        f"{empty} returned no rows, so the measured window is mostly the "
        "round trip and not the data"
    )


def test_opening_a_profile_stays_under_the_python_heap_ceiling(measured, request):
    """Peak Python heap for ``Profile(...)`` alone, from a cold start.

    On ``cache_mode="auto"`` this is the Parquet-cache ETL, and it is the number
    that catches a streamed scan being turned back into a materialised list:
    reverting the NVTX generator in ``parquet_cache._build_nvtx_kernel_map``
    takes it from 4.85 MB to 6.09 MB. That change leaves every other assertion in
    the suite passing, because the answers are identical — only the memory
    differs, and nothing else looks at memory.

    On ``cache_mode="direct"`` this is the zero-ETL claim stated as a number:
    0.06 MB, against 4.85 MB if the mode is ever routed through the cache build.
    """
    open_ceiling, _, want_open, _ = measured["ceilings"]
    peak = measured["open_peak_mb"]
    _emit(
        request,
        f"[memory:{measured['cache_mode']}] open {peak:.2f} MB "
        f"(ceiling {open_ceiling:.1f}, measured {want_open:.2f}, "
        f"input {measured['input_mb']:.3f} MB, {measured['open_s']:.2f}s, "
        f"python {sys.version_info.major}.{sys.version_info.minor})",
    )
    assert peak < open_ceiling, (
        f"opening a profile with cache_mode={measured['cache_mode']!r} peaked at "
        f"{peak:.2f} MB of Python heap against a {want_open:.2f} MB baseline, over "
        f"the {open_ceiling:.1f} MB ceiling. Either something now holds rows that "
        "used to stream, or the ceiling needs raising with a reason. Do not raise "
        "it without one — this number is what decides whether a tens-of-GB "
        "capture can be opened at all."
    )


def test_running_skills_stays_under_the_python_heap_ceiling(measured, request):
    """Peak Python heap for three skills on an already-open profile.

    Separate from the open window because on the cached path the open dominates
    so completely that this was invisible: with the skills removed entirely the
    combined number moved by 0.01 MB. Measured alone it is 0.27 MB, and dropping
    the ``LIMIT`` push-down in ``nvtx_attribution.attribute_kernels_to_nvtx`` —
    the optimisation its own docstring describes as keeping unrelated kernels
    out of Python — takes it to 1.53 MB.

    On ``cache_mode="direct"`` this window also contains the on-demand
    ``ensure_nvtx_kernel_map`` build, which is the largest single Python
    allocation in the codebase at 6.62 MB for a 2.27 MB capture. Turning that
    sweep's per-thread buckets from tuples into dicts takes it to 8.06 MB.
    """
    _, skills_ceiling, _, want_skills = measured["ceilings"]
    peak = measured["skills_peak_mb"]
    _emit(
        request,
        f"[memory:{measured['cache_mode']}] skills {peak:.2f} MB "
        f"(ceiling {skills_ceiling:.1f}, measured {want_skills:.2f}, "
        f"total {measured['workload_s']:.2f}s)",
    )
    assert peak < skills_ceiling, (
        f"running skills with cache_mode={measured['cache_mode']!r} peaked at "
        f"{peak:.2f} MB of Python heap against a {want_skills:.2f} MB baseline, "
        f"over the {skills_ceiling:.1f} MB ceiling. Something a skill reads now "
        "lands in Python objects that did not before — check the LIMIT push-downs "
        "and the on-demand nvtx_kernel_map build before raising this."
    )
