"""Determinism for the orderings outside the skills package.

Companion to `test_determinism.py`. The sites guarded here differ from the ones
there in an important way: they do not merely sort a list, they **select an
entity** — which thread the tree is built for, which launch configuration is
called dominant, which two profiles get diffed, which NVTX region an occurrence
index resolves to. A tie therefore changes what gets analysed, not the order it
is presented in.

Most of these choices are made deep inside functions that need a real profile,
a cache, or a filesystem layout to reach. Rather than build elaborate fixtures
that would themselves need maintaining, the guards below assert the orderings
structurally — the same approach `test_determinism.py` settled on after a
behavioural test proved unable to observe a tie on a single engine.
"""

from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "nsys_ai"


def _read(rel: str) -> str:
    return (SRC / rel).read_text()


# ── Entity selection: a tie changes what is analysed ────────────────────────


def test_primary_thread_selection_is_total():
    """`_find_primary_thread` takes rows[0] of a launch-count ranking.

    Symmetric launcher threads tie routinely, and the choice feeds iteration
    detection — so an unbroken tie retargets the whole analysis. Same shape as
    the device-selection bug fixed for `root_cause_matcher`.
    """
    src = _read("nvtx_tree.py")
    assert "GROUP BY r.globalTid ORDER BY cnt DESC, r.globalTid ASC" in src


def test_thread_name_lookup_is_total():
    """A thread may carry several names at equal priority."""
    src = _read("nvtx_tree.py")
    assert "ORDER BY t.priority DESC, s.value ASC LIMIT 1" in src


def test_dominant_launch_config_ordering_includes_the_group_keys():
    """`rows[0]` becomes the reported dominant config, and the caller turns it
    into a causal claim about occupancy. Neither `total_ns` nor `sample_count`
    is a group key, so the group keys must complete the order."""
    src = _read("diff_tools.py")
    assert 'sql += " ORDER BY total_ns DESC, sample_count DESC"' in src
    assert 'sql += ", " + ", ".join(f"{g} ASC" for g in group_parts) + ", matched_name ASC"' in src


def test_snapshot_selection_is_total():
    """mtime is not unique for co-extracted directories, and this decides
    which two profiles get diffed."""
    src = _read("loop_state.py")
    assert "key=lambda p: (-p.stat().st_mtime, p.name)" in src
    assert "key=lambda p: p.stat().st_mtime, reverse=True" not in src


# ── Order-dependent sweeps: a tie changes a computed number ─────────────────


def test_launch_overhead_sweep_is_total():
    """The loop advances a running maximum and tests `ks > prev_end`, so tied
    starts shift a reported millisecond figure.

    The kernel keys alone are not enough. `correlationId` is the join key and
    the runtime side fans out — on a real H100 capture 212k correlationIds
    carry several runtime rows — so many joined rows share all three kernel
    keys while differing in the `rs`/`re` the loop actually reads. Before the
    runtime columns were added, this returned values spanning ~20% between runs
    on identical input (712 / 702 / 565 ms). With them the spread is zero.
    """
    src = _read("overlap.py")
    assert "ORDER BY k.start, k.[end], k.correlationId, r.start, r.[end]" in src


def test_kernel_launch_overhead_orders_the_fanned_out_runtime_side():
    """Same defect, same cause, in the skill that reports per-launch overhead.

    Its join is on correlationId too, so the runtime columns are required for
    the order to be total.
    """
    skill = (
        Path(__file__).resolve().parent.parent
        / "src/nsys_ai/skills/builtins/kernel_launch_overhead.py"
    ).read_text()
    assert "r.start ASC, r.[end] ASC" in skill


def test_iteration_extraction_prefers_the_enclosing_range():
    """This tiebreak is semantic, not merely deterministic.

    The greedy filter keeps the first range at a given start. On a tie the
    longer range is the enclosing one, so `end DESC` is required — ordering the
    shorter one first would let a nested range mask its parent and silently
    shrink the detected iteration.
    """
    src = _read("overlap.py")
    assert "ORDER BY n.start, n.[end] DESC" in src


def test_region_occurrence_selection_is_total():
    """`occurrence_index` indexes into these rows, so a tie selects a different
    region and reports a different MFU percentage."""
    src = _read("region_mfu.py")
    assert 'base_sql += "ORDER BY start_ns, end_ns, text, global_tid"' in src


def test_nsys_kernel_list_is_ordered():
    """Not for the LCS tie — that iterates the cutracer side, which is already
    sorted. The order matters because `cutracer_analysis` builds the reverse
    map behind a first-wins guard, so when several nsys kernels match one
    cutracer kernel the first one seen wins, deciding SASS attachment.
    """
    src = _read("cutracer/correlator.py")
    assert "SELECT DISTINCT name FROM kernels WHERE name IS NOT NULL ORDER BY name" in src

    # The first-wins guard this protects. If it ever stops being first-wins,
    # the ordering above is no longer load-bearing and this test should say so.
    analysis = _read("skills/builtins/cutracer_analysis.py")
    assert "if ct_k and ct_k not in ct_to_nsys:" in analysis


# ── Top-N with LIMIT: a tie changes which rows survive ──────────────────────


@pytest.mark.parametrize(
    "fragment",
    [
        # aggregate_kernels — grouped by name/demangled
        'sql += " ORDER BY total_ns DESC, name ASC, demangled ASC"',
        # aggregate_nvtx_ranges and search_nvtx_names — grouped by text
        'sql += " ORDER BY total_ns DESC, text ASC"',
    ],
)
def test_profile_aggregates_order_by_the_group_key(fragment):
    """Paired with LIMIT, a tie at the cut-off changes which rows survive.

    `diff.py` calls `aggregate_nvtx_ranges(limit=200)`, so this decides the
    `new`/`removed` classification in the emitted diff.
    """
    assert fragment in _read("profile.py")


def test_no_bare_total_ns_ordering_remains_in_profile():
    """Guard against a fourth aggregate being added with the old pattern."""
    src = _read("profile.py")
    assert 'sql += " ORDER BY total_ns DESC"' not in src


# ── Recorded non-defects, so they are not "fixed" by mistake ────────────────


def test_overlap_sweepline_is_already_total_by_construction():
    """The sweep-line groups by ts *before* the window, so ts is unique there.

    Adding a tiebreak would be noise at best. This test exists to stop a future
    determinism sweep from "fixing" it.
    """
    src = _read("overlap.py")
    agg_then_window = src.index("GROUP BY ts") < src.index("LEAD(ts) OVER (ORDER BY ts)")
    assert agg_then_window, "sweep-line no longer aggregates by ts before the window"


def test_sol_gate_first_row_is_backed_by_single_row_skill():
    """`sol_gate` takes rows[0]; that is safe only while region_mfu returns one row."""
    assert "return [result]" in _read("skills/builtins/region_mfu.py")
