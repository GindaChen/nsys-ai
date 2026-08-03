"""Nsight versions its activity tables, and thirteen queries still named them literally.

`resolve_activity_tables` exists because newer exports suffix these tables `_V2`
or `_V3`. Most readers go through `Profile._duckdb_query`, and `parquet_cache`
creates an alias view for the unversioned name over whichever versioned table it
finds, so a literal answers correctly there and no test written against those
readers can fail. `region_mfu.get_region_kernels` takes a plain `sqlite3`
connection, gets no alias layer, and raises.

That asymmetry is the whole problem with this defect class: it has now been found
six times — #285, #288, #291, #292, and twice here — always by reading, never by
a failing test, because whichever path someone happens to exercise is usually the
aliased one.

So this file has two jobs. The first tests the reader that genuinely breaks. The
second is a source check over every query in the package, which is the only thing
that catches a literal on a path no test exercises. A source check is a poor
substitute for a behavioural test of a fix — #288 shipped one that passed while
its fix was inert — but it is the right tool for "this must not reappear", which
is a different job from "this is fixed".
"""

import re
import shutil
import sqlite3
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "nsys_ai"
ANNOTATED = REPO / "tests" / "fixtures" / "h100_2gpu_1s.sqlite"

# The tables resolve_activity_tables() resolves by prefix — the repo's own answer
# to which names carry a version suffix. StringIds, ThreadNames and the
# TARGET_INFO_* tables are not versioned and are deliberately absent.
VERSIONED = (
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
    "ENUM_CUPTI_SYNC_TYPE",
    "NVTX_EVENTS",
)

# The modules that do the resolving, and so must name the tables literally.
RESOLVERS = {"connection.py", "parquet_cache.py"}

_LITERAL_IN_SQL = re.compile(r"\b(?:FROM|JOIN)\s+(" + "|".join(VERSIONED) + r")\b")


def _readonly(path: Path) -> sqlite3.Connection:
    """Open a committed fixture without letting the run modify it.

    Several readers call ``ensure_performance_indexes``, which writes
    ``_nsysai_*`` indexes into whatever profile it is handed. Against a checked-in
    fixture that leaves the working tree dirty and the file changed for every
    later run — a test that edits its own input is not reproducible.
    """
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


# ── The reader that actually breaks ─────────────────────────────────────────


@pytest.fixture(scope="module")
def runtime_v3(tmp_path_factory) -> Path:
    """The annotated fixture with CUPTI_ACTIVITY_KIND_RUNTIME renamed to _V3."""
    dst = tmp_path_factory.mktemp("versioned") / "runtime_v3.sqlite"
    shutil.copy(ANNOTATED, dst)
    conn = sqlite3.connect(dst)
    try:
        conn.execute(
            "ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME "
            "RENAME TO CUPTI_ACTIVITY_KIND_RUNTIME_V3"
        )
        conn.commit()
    finally:
        conn.close()
    return dst


def test_the_fixture_has_runtime_rows():
    """Premise: without runtime rows the assertions below pass vacuously."""
    conn = _readonly(ANNOTATED)
    try:
        n = conn.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()[0]
    finally:
        conn.close()
    assert n > 0


def test_get_region_kernels_reads_the_versioned_runtime_table(runtime_v3):
    """The assertion that fails while the table name is a literal.

    This one raises rather than returning nothing: it is handed a plain
    connection, so there is no alias view to fall back on.
    """
    from nsys_ai.region_mfu import get_region_kernels

    conn = sqlite3.connect(runtime_v3)
    try:
        rows = get_region_kernels(
            conn, nvtx_start_ns=0, nvtx_end_ns=10**18, global_tid=None, device_id=None
        )
    finally:
        conn.close()

    assert rows, "no kernels resolved from a profile whose runtime table is _V3"


def test_get_region_kernels_agrees_across_both_table_names(runtime_v3):
    """Resolution must change where the rows come from, not how many."""
    from nsys_ai.region_mfu import get_region_kernels

    kw = dict(nvtx_start_ns=0, nvtx_end_ns=10**18, global_tid=None, device_id=None)
    original = _readonly(ANNOTATED)
    renamed = sqlite3.connect(runtime_v3)
    try:
        before = get_region_kernels(original, **kw)
        after = get_region_kernels(renamed, **kw)
    finally:
        original.close()
        renamed.close()

    assert before, "premise: the unrenamed profile must return kernels"
    assert len(after) == len(before)


# ── The check that stops it coming back ─────────────────────────────────────


# An escape hatch for the rare query that must keep the literal. It suppresses
# the line that follows it, and it costs a written reason — see fingerprint.py,
# where resolving the name would silently change a stable content hash.
EXEMPTION = "literal-table-ok:"


def _sql_literals() -> list[str]:
    """Every `FROM`/`JOIN <versioned table>` outside the resolver modules."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name in RESOLVERS:
            continue
        lines = path.read_text().splitlines()
        exempt_until = -1
        for lineno, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                if EXEMPTION in stripped:
                    # Suppress the next non-comment line.
                    exempt_until = lineno
                    for ahead in range(lineno, len(lines)):
                        if not lines[ahead].strip().startswith("#"):
                            exempt_until = ahead + 1
                            break
                continue
            if lineno == exempt_until:
                continue
            match = _LITERAL_IN_SQL.search(line)
            if match:
                found.append(f"{path.relative_to(SRC.parent)}:{lineno}  {match.group(1)}")
    return found


def test_no_query_names_a_versioned_table_literally():
    """A literal here reads the wrong table on a _V2 / _V3 export.

    Resolve it instead:

        tables = wrap_connection(conn).resolve_activity_tables()
        runtime_table = tables.get("runtime")

    or, in a Skill SQL template, use the `{runtime_table}` placeholder that
    `Skill.execute` substitutes.

    Whether a given site is currently broken depends on whether its connection
    has parquet_cache's alias views, which is a property of how the profile was
    opened rather than anything the reader controls. That is why this check does
    not try to distinguish them.
    """
    offenders = _sql_literals()
    assert not offenders, (
        "queries naming a versioned Nsight table literally:\n  "
        + "\n  ".join(offenders)
    )


def test_the_check_can_actually_fail():
    """Guard the guard: a regex that matches nothing would pass silently."""
    probe = "            FROM CUPTI_ACTIVITY_KIND_RUNTIME r\n"
    assert _LITERAL_IN_SQL.search(probe), "the detector stopped detecting"
    assert not _LITERAL_IN_SQL.search("FROM {runtime_table} r")
    assert not _LITERAL_IN_SQL.search("FROM StringIds s")
