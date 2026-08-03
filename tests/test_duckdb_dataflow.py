"""Invariants of the DuckDB path that are currently held only by discipline.

The repo runs every profile through one of two engines: DuckDB over a Parquet
cache (or over SQLite via sqlite_scanner), and plain sqlite3 as the fallback.
Three things keep that dual path honest, and none of them was enforced:

1. SQL written in SQLite dialect must go through ``DuckDBAdapter.execute``, which
   applies ``sqlite_to_duckdb``. Code may also issue DuckDB-native SQL directly —
   that is legitimate — but only if it genuinely needs no rewriting. Measured
   over a full EvidenceBuilder build, 11 executes bypass the adapter and 0 of
   them need rewriting. The test below pins the 0, not the 11: adding a direct
   call is fine, adding one carrying ``[end]`` or a hex literal is not.

2. The Parquet cache creates an alias view for *every* known variant of a table
   (base, _V2, _V3), so any table-resolution strategy lands on the same data.
   That is the reason a second, differently-ordered resolver in
   ``sync_cost_analysis`` has never produced a wrong answer.

3. A profile whose tables carry an unknown future suffix must still analyse.

These are characterisation tests: they encode what was measured, so a change
that breaks the arrangement fails here rather than silently on a user's profile.
"""

import shutil
import sqlite3
import traceback
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from nsys_ai.profile import Profile  # noqa: E402
from nsys_ai.skills.registry import get_skill  # noqa: E402
from nsys_ai.sql_compat import sqlite_to_duckdb  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"


def _classify(stack) -> str:
    """Which of the three legitimate routes did this execute take?"""
    files = [f.filename for f in stack]
    if any(
        f.endswith("nsys_ai/connection.py") and fr.name == "execute"
        for f, fr in zip(files, stack)
    ):
        return "rewritten"
    if any(f.endswith("nsys_ai/connection.py") for f in files):
        return "adapter_internal"  # SHOW TABLES / DESCRIBE — engine-specific by design
    if any(f.endswith("nsys_ai/parquet_cache.py") for f in files):
        return "cache_internal"  # DuckDB-native DDL that builds the session
    return "direct"


def test_sql_bypassing_the_rewriter_never_needs_rewriting():
    """The guard that keeps the dual-dialect arrangement safe.

    17 skill files write ``k.[end]`` — SQLite bracket syntax DuckDB rejects — and
    13 of those also define an ``execute_fn`` that can call the connection
    directly. Nothing structurally prevents bracket SQL from taking the direct
    route; today it simply never does. Without this test that is an unstated
    convention, and the failure mode is a DuckDB parse error on a user's profile.
    """
    offenders = []
    original = duckdb.DuckDBPyConnection.execute

    def patched(self, sql, *args, **kwargs):
        if isinstance(sql, str):
            stack = traceback.extract_stack()[:-1]
            if _classify(stack) == "direct" and sqlite_to_duckdb(sql) != sql:
                caller = next(
                    (f for f in reversed(stack) if "nsys_ai" in f.filename), stack[-1]
                )
                offenders.append(f"{caller.filename}:{caller.lineno} -> {sql[:120]}")
        return original(self, sql, *args, **kwargs)

    duckdb.DuckDBPyConnection.execute = patched
    try:
        from nsys_ai.evidence_builder import EvidenceBuilder

        with Profile(str(FIXTURE)) as prof:
            EvidenceBuilder(prof, device=0).build()
    finally:
        duckdb.DuckDBPyConnection.execute = original

    assert not offenders, (
        "SQL needing dialect translation reached DuckDB without going through "
        "DuckDBAdapter.execute:\n  " + "\n  ".join(offenders)
    )


def test_every_known_table_variant_is_reachable_on_the_cached_path():
    """The alias views are the actual compatibility layer.

    ``parquet_cache._ALIASES`` maps each short name to every Nsight variant, and
    ``_create_existing_alias_views`` materialises all of them over the same
    Parquet. This is why two different table resolvers in this repo can pick
    different names and still read identical data — a coincidence worth pinning,
    since removing the aliases would turn it into a silent wrong answer.

    Checked by querying each alias rather than by comparing names: DuckDB
    identifiers are case-insensitive, so an alias differing from its short name
    only in case is the *same* object and appears once in ``SHOW TABLES``.
    Name comparison reports those as missing when they are perfectly reachable.
    """
    from nsys_ai.parquet_cache import _ALIASES

    unreachable = []
    with Profile(str(FIXTURE)) as prof:
        if prof.db is None:  # pragma: no cover - cache build failed
            pytest.skip("DuckDB cache unavailable")
        present = {r[0] for r in prof.db.execute("SHOW TABLES").fetchall()}
        for short_name, aliases in _ALIASES.items():
            if short_name not in present:
                continue  # this profile genuinely lacks that data
            for alias in aliases:
                try:
                    prof.db.execute(f'SELECT 1 FROM "{alias}" LIMIT 0')
                except duckdb.Error as exc:
                    unreachable.append(f"{alias} ({exc.__class__.__name__})")

    assert not unreachable, (
        f"alias views absent, so resolution is no longer variant-safe: {unreachable}"
    )


def test_the_two_table_resolvers_agree_wherever_only_one_variant_exists():
    """A second resolver lives in ``sync_cost_analysis`` and orders variants the
    opposite way (``ORDER BY name DESC`` vs ``sorted()[0]``).

    On a real capture only one variant is present — verified across every
    committed fixture and a 3.7GB production capture — so the two agree. They
    diverge only when variants coexist, which happens exclusively on the cached
    path where all variants alias the same data. This test states the condition
    under which the duplication is harmless, so that if it stops holding the
    reason is on record.
    """
    from nsys_ai.connection import wrap_connection
    from nsys_ai.skills.builtins.sync_cost_analysis import _resolve_table_name

    conn = sqlite3.connect(FIXTURE)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        variants = sorted(t for t in tables if t.startswith("CUPTI_ACTIVITY_KIND_KERNEL"))
        assert len(variants) == 1, f"fixture premise broke; found {variants}"

        official = wrap_connection(conn).resolve_activity_tables().get("kernel")
        local = _resolve_table_name(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
    finally:
        conn.close()

    assert official == local == variants[0]


def test_an_unknown_future_table_suffix_still_analyses(tmp_path):
    """Nsight adds versioned tables between releases; _ALIASES stops at _V3.

    A capture whose kernel table is ``_V4`` must still produce results — the ETL
    resolves the source table by prefix rather than from the hardcoded list, so
    the cache is built from it and the analysis proceeds.
    """
    profile = tmp_path / "v4.sqlite"
    shutil.copy(FIXTURE, profile)
    conn = sqlite3.connect(profile)
    try:
        conn.execute(
            "ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL RENAME TO CUPTI_ACTIVITY_KIND_KERNEL_V4"
        )
        conn.commit()
    finally:
        conn.close()

    with Profile(str(profile)) as prof:
        conn = prof.db if prof.db is not None else prof.conn
        rows = get_skill("top_kernels").execute(conn, limit=3)

    assert rows, "a _V4 kernel table produced no kernels"
    assert rows[0].get("kernel_name"), f"kernels came back unnamed: {rows[0]}"


def test_both_engines_return_the_same_answer_for_a_dual_path_skill():
    """``sync_cost_analysis`` has a DuckDB route and a SQLite route.

    Divergence between the two is the defect class that produced the cached-vs-
    uncached bug fixed in #274, and a green unit suite did not catch that one.
    """
    skill = get_skill("sync_cost_analysis")

    with Profile(str(FIXTURE)) as prof:
        if prof.db is None:  # pragma: no cover
            pytest.skip("DuckDB cache unavailable")
        via_duckdb = skill.execute(prof.db, device=0)

    conn = sqlite3.connect(FIXTURE)
    try:
        via_sqlite = skill.execute(conn, device=0)
    finally:
        conn.close()

    assert via_duckdb == via_sqlite, "the DuckDB and SQLite paths disagree"
