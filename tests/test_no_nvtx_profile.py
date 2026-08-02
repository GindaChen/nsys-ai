"""A capture taken without NVTX is valid, and must not crash the commands.

`analyze` and `export-csv` exited 1 with a raw SQL error naming NVTX_EVENTS on
any profile lacking that table. Nothing about such a profile is wrong — NVTX
annotation is optional, and everything else in it is analysable — so the correct
behaviour is to produce the rest of the report and leave out the part that needs
annotation. `analyze` already reports the absence through the health manifest's
"Insufficient NVTX coverage" finding, so once the crash is gone the user is told.

The same code also hardcoded the table name, which would fail on the _V2 and _V3
suffixed variants newer Nsight exports use even when NVTX is present. Resolving
the table fixes both, which is why the tests below check resolution as well as
absence.
"""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NO_NVTX = REPO / "tests" / "fixtures" / "healthy_1pct.sqlite"
WITH_NVTX = REPO / "tests" / "fixtures" / "h100_2gpu_1s.sqlite"


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "nsys_ai", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def test_the_fixture_really_has_no_nvtx():
    """Guard the premise, so these cannot go vacuous."""
    conn = sqlite3.connect(NO_NVTX)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert not any(t.startswith("NVTX_EVENTS") for t in tables)


def test_analyze_succeeds_without_nvtx():
    r = _run("analyze", str(NO_NVTX), "--gpu", "0", "--trim", "0.0", "0.4")
    assert r.returncode == 0, f"analyze failed:\n{r.stderr[-600:]}"
    assert "NVTX_EVENTS" not in r.stderr, "leaked a database error naming the table"
    # It must still analyse everything that does not need annotation.
    assert "Kernels:" in r.stdout


def test_export_csv_succeeds_without_nvtx(tmp_path):
    out = tmp_path / "k.csv"
    r = _run("export-csv", str(NO_NVTX), "--gpu", "0", "--trim", "0.0", "0.4", "-o", str(out))
    assert r.returncode == 0, f"export-csv failed:\n{r.stderr[-600:]}"
    assert out.exists() and out.read_text().count("\n") > 1, "wrote no rows"


def test_iteration_detection_falls_back_to_the_heuristic():
    """Without markers there is nothing to match, but the gap-based heuristic
    needs no annotation.

    This crashed before. An intermediate fix returned early instead, which
    withheld a result the profile can genuinely support — the issue asked for
    everything that does not require annotation, and this qualifies.
    """
    from nsys_ai.overlap import detect_iterations
    from nsys_ai.profile import Profile

    with Profile(str(NO_NVTX)) as prof:
        iters = detect_iterations(prof, 0, (0, 400_000_000))
    assert iters, "the heuristic fallback was skipped"
    assert all(i.get("heuristic") for i in iters), "claimed real iterations without NVTX"


def test_nvtx_tree_returns_nothing_rather_than_raising():
    """The shared entry point for both commands' NVTX work."""
    from nsys_ai.nvtx_tree import build_nvtx_tree
    from nsys_ai.profile import Profile

    with Profile(str(NO_NVTX)) as prof:
        assert build_nvtx_tree(prof, 0, (0, 400_000_000)) == []


@pytest.mark.parametrize(
    "method,args",
    [
        ("aggregate_nvtx_ranges", ()),
        ("search_nvtx_names", ("anything",)),
    ],
)
def test_profile_nvtx_accessors_return_empty(method, args):
    """`[]` deliberately, not an abstention row.

    `Profile` is a data accessor, where "there are no NVTX ranges" is the
    truthful answer. Abstention is a skill-layer concept; pushing it here would
    force all six callers to handle a sentinel row.

    This behaviour predates the change — it is pinned, not claimed as new.
    """
    from nsys_ai.profile import Profile

    with Profile(str(NO_NVTX)) as prof:
        assert getattr(prof, method)(*args) == []


# ── The other half: a resolved table name, not a hardcoded one ──────────────


@pytest.fixture
def v2_profile(tmp_path):
    """An annotated profile whose table carries the _V2 suffix newer exports use."""
    import shutil

    dst = tmp_path / "v2.sqlite"
    shutil.copy(WITH_NVTX, dst)
    conn = sqlite3.connect(dst)
    try:
        conn.execute("ALTER TABLE NVTX_EVENTS RENAME TO NVTX_EVENTS_V2")
        conn.commit()
    finally:
        conn.close()
    return dst


def test_a_versioned_nvtx_table_is_still_read(v2_profile):
    """The half of this change that a source-text check could not verify.

    An earlier version asserted that a literal table name was absent from one
    file. It passed while the fix was inert — a redundant guard sat above the
    hardcoded one that actually short-circuited — and while sibling files kept
    their own literals. Renaming the table on a real annotated profile is the
    only check that distinguishes "resolved" from "looks resolved": the wrong
    answer here is silence, not a crash.
    """
    from nsys_ai.nvtx_tree import build_nvtx_tree
    from nsys_ai.profile import Profile

    # Pure-SQLite path: the fallback users on older exports actually hit.
    conn = sqlite3.connect(v2_profile)
    try:
        prof = Profile._from_conn(conn)
        assert prof.aggregate_nvtx_ranges(limit=5), "annotated _V2 profile reported no ranges"
        assert prof.search_nvtx_names("a", limit=5) is not None
        assert build_nvtx_tree(prof, 0, prof.meta.time_range), "_V2 tree came back empty"
        assert prof.meta.nvtx_count > 0, "_V2 export reported nvtx_count=0"
    finally:
        conn.close()


def test_a_profile_with_nvtx_is_unaffected():
    """The guards must not short-circuit a profile that does have annotation."""
    from nsys_ai.nvtx_tree import build_nvtx_tree
    from nsys_ai.profile import Profile

    if not WITH_NVTX.exists():  # pragma: no cover - fixture is committed
        pytest.skip("two-GPU fixture not available")

    with Profile(str(WITH_NVTX)) as prof:
        assert prof.aggregate_nvtx_ranges(limit=5), "NVTX ranges vanished"
        tree = build_nvtx_tree(prof, 0, prof.meta.time_range)
    assert tree, "the NVTX tree came back empty on an annotated profile"
