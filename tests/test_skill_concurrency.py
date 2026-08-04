"""Concurrent skill runs must give the sequential answer — as a standing property.

``tests/test_profile_query_conn.py`` covers the accessor that decides which
handle a thread queries through. It does not establish agreement at the skill
layer, and agreement is the thing that actually matters: DuckDB keeps the
pending result set on the connection, so ``execute`` and ``fetch`` are each
atomic but not atomic as a pair. Two threads on one handle read each other's
rows, and nothing raises.

Three design points are load-bearing and were all settled by measurement.

**The memo has to be cleared, or this measures the memo.** #295 memoizes skill
results per connection keyed on resolved parameters, and #302 shares one bag
between a connection and every cursor derived from it. Left alone, forty
concurrent skill calls issue *one* distinct SQL statement and read the answer
back thirty-nine times — the concurrent calls never touch the database, so the
oracle is being compared against copies of itself. The probe clears the bag
before each wave and reports the resulting execution count; the test asserts on
that count, so if the memo ever swallows the concurrency again this fails rather
than passing vacuously.

**The skills have to return rows, or agreement is a tautology.** Two of the six
skills originally listed here did not, on this fixture: ``nccl_breakdown``
returns 0 rows (the capture has no NCCL kernels at all — measured, despite the
fixture's name) so its case asserted ``[] == []``, and ``thread_utilization``
abstains for want of a ``COMPOSITE_EVENTS`` table, so its "answer" is a constant
marker built without touching the data. Neither can disagree under any
regression. They are replaced by two skills that do read the tables and do
return rows, and the probe now reports each baseline's row count so a future
fixture change turns the same vacuity into a failure instead of a green tick.

**The bound has to be a process kill, not a future timeout.** The regression
mode here is a hang, not a wrong answer. Under the mutation that hands every
thread the shared connection, the run went past 120 seconds of wall clock on a
couple of seconds of CPU, and one variant dumped core. Bounding it with
``future.result(timeout=...)`` does nothing: the TimeoutError leaves the ``with
ThreadPoolExecutor(...)`` block, ``__exit__`` calls ``shutdown(wait=True)``, and
that joins the deadlocked workers forever. Only ``subprocess.run(timeout=...)``,
which kills the child, turns the hang into a build failure — measured here as a
red build in 31s under that mutation.

One child process covers everything and returns per-skill counts; a
module-scoped fixture plus parametrize gives per-skill failure attribution for
the price of one subprocess (six separate children would cost 4.6s instead of
1.2s for the same information).
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PROBE = Path(__file__).resolve().parent / "_concurrency_probe.py"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"

# Every skill here reads the kernel/runtime/NVTX tables of this fixture and
# returns rows — measured, and asserted below so it stays true. Row counts at
# the limits the probe applies: top_kernels 5, gpu_idle_gaps 6,
# memory_transfers 3, kernel_launch_overhead 20, nvtx_layer_breakdown 21,
# kernel_overlap_matrix 10.
#
# Left out on purpose: schema_inspect reads the catalog rather than the data;
# nvtx_kernel_map is covered by the memory probe; nccl_breakdown and
# thread_utilization return nothing here (see the module docstring).
SKILLS = (
    "top_kernels",
    "gpu_idle_gaps",
    "memory_transfers",
    "kernel_launch_overhead",
    "nvtx_layer_breakdown",
    "kernel_overlap_matrix",
)

# Generous: the clean run is ~1.2s wall including interpreter start and a cold
# cache build. The deadlock this bounds does not resolve at any timeout, so the
# only thing a larger number buys is a slower failure.
PROBE_TIMEOUT_S = 30


@pytest.fixture(scope="module")
def probe_result():
    """One child process, shared by every parametrized case.

    The working directory belongs to this fixture. On timeout the child is
    SIGKILLed, and a killed process never runs its own cleanup — so the 2.3 MB
    copy it made would be left behind on exactly the runs that happen most
    during a bisect.
    """
    workdir = tempfile.mkdtemp(prefix="nsys-conc-probe-")
    try:
        result = subprocess.run(
            [sys.executable, str(PROBE), str(FIXTURE), workdir, ",".join(SKILLS)],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"the concurrent skill run did not finish within {PROBE_TIMEOUT_S}s. "
            "This is the deadlock signature: threads contending for one DuckDB "
            "connection's pending result set never make progress. Check that "
            "Profile.query_conn() still hands worker threads their own cursor."
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if result.returncode != 0:
        pytest.fail(
            f"the concurrency probe exited {result.returncode}\n"
            f"stdout: {result.stdout.strip()}\nstderr: {result.stderr.strip()[-2000:]}"
        )

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if payload.get("engine") != "duckdb":
        pytest.skip("requires duckdb — a sqlite3 connection serialises internally")
    return payload


def test_worker_threads_share_the_owning_connection_s_cache(probe_result):
    """A second handle must not mean a second cache.

    The per-connection bag backs both the probe cache and the skill memo from
    #295. Keyed on the handle, every worker thread starts empty and re-executes
    work the owner already did — measured as four extra executions across four
    threads. Keyed on the owning connection (#302), the handle a thread happens
    to hold stops mattering.

    Sharing is safe because the bag stores results, not handles: one skill with
    one set of resolved parameters over an immutable capture has one answer.

    Lived in ``tests/test_profile_query_conn.py`` until it was measured hanging
    past 120s under the shared-connection regression — see the note left in that
    file. It is here because here there is a kill timeout.
    """
    sharing = probe_result["cache_sharing"]
    assert sharing["owner_executions"] == 1, (
        f"the owner's single call to {sharing['skill']} executed "
        f"{sharing['owner_executions']} times, so the count below measures "
        "something other than what the workers added"
    )
    assert sharing["worker_extra_executions"] == 0, (
        f"worker threads re-executed {sharing['skill']} "
        f"{sharing['worker_extra_executions']} times — the cache is keyed on the "
        "handle again, not the owning connection"
    )


@pytest.mark.parametrize("skill_name", SKILLS)
def test_the_sequential_answer_is_worth_comparing_against(probe_result, skill_name):
    """Guards the guard, part one: an empty oracle cannot disagree.

    ``nccl_breakdown`` sat in this list for a while asserting ``[] == []``,
    because the fixture has no NCCL kernels; ``thread_utilization`` asserted that
    forty threads all produced the same abstention marker, which is built without
    reading anything. Both were green under every mutation. A skill that stops
    returning rows on this fixture — because the fixture changed, or because the
    skill did — has to fail here rather than quietly making its agreement case
    meaningless.
    """
    rows = probe_result["baseline_rows"][skill_name]
    assert rows > 0, (
        f"{skill_name}'s sequential baseline is "
        + ("an abstention" if rows < 0 else "empty")
        + " on this fixture, so comparing forty concurrent runs against it "
        "proves nothing. Give it parameters that return rows, or drop it."
    )


@pytest.mark.parametrize("skill_name", SKILLS)
def test_concurrent_runs_agree_with_the_sequential_answer(probe_result, skill_name):
    """Eight threads asking the same question must get the sequential answer.

    Fails silently otherwise: a shared result set returns short or interleaved
    rows, so a skill reports a plausible-looking wrong number and every other
    assertion in the suite still passes.
    """
    bad = probe_result["mismatches"][skill_name]
    assert bad == 0, (
        f"{skill_name}: {bad} of "
        f"{probe_result['waves'] * probe_result['futures_per_wave']} concurrent "
        "runs disagreed with the sequential answer"
    )


@pytest.mark.parametrize("skill_name", SKILLS)
def test_the_concurrent_runs_actually_queried_the_database(probe_result, skill_name):
    """Guards the guard, part two: without this the agreement test can pass vacuously.

    The per-connection memo will serve every concurrent call from the sequential
    baseline's result if nothing clears it, and then agreement is a tautology —
    which is exactly the state the older concurrency test was in, at one distinct
    statement for forty calls. Each wave must produce at least one real
    execution beyond the baseline.
    """
    executions = probe_result["executions"][skill_name]
    waves = probe_result["waves"]
    assert executions >= waves + 1, (
        f"{skill_name} executed only {executions} times for {waves} waves plus a "
        "baseline — the concurrent calls were served from the memo, so the "
        "agreement assertion proves nothing"
    )
