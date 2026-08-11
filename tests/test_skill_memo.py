"""Skill results are memoized per connection, keyed on the resolved parameters.

An `EvidenceBuilder.build()` reaches skills both directly and through the
health manifest and root-cause matcher, which re-run some skills the pipeline
has already requested. Exact repeats should reuse the first result.

The parameters are the reason this needs care rather than a dictionary keyed on
the skill name. Most of the repeats are *not* identical — `gpu_idle_gaps` is
called at `limit=1` and at `limit=5` within a single build, and those return
genuinely different results — so a name-only key would serve one caller the
other's rows and change findings without any error.
"""

import sqlite3

import pytest

from nsys_ai.skills.registry import get_skill


@pytest.fixture(scope="module")
def fixture(profile_copy):
    """A writable copy: opening a profile writes helper indexes into the file."""
    return profile_copy("h100_2gpu_1s.sqlite")


def test_identical_params_hit_the_cache(fixture):
    skill = get_skill("top_kernels")
    calls = []
    original = skill.execute_fn

    def counting(conn, **kwargs):
        calls.append(kwargs)
        return original(conn, **kwargs)

    skill.execute_fn = counting
    conn = sqlite3.connect(fixture)
    try:
        first = skill.execute(conn, limit=5)
        second = skill.execute(conn, limit=5)
    finally:
        skill.execute_fn = original
        conn.close()

    assert len(calls) == 1, "the second identical call re-executed"
    assert first == second


def test_different_params_are_never_served_a_cached_result(fixture):
    """The failure a name-only key would cause, asserted directly.

    `gpu_idle_gaps` is called at limit=1 and limit=5 in the same build. If the
    key ignored parameters the second caller would receive the first's rows.
    """
    skill = get_skill("gpu_idle_gaps")
    conn = sqlite3.connect(fixture)
    try:
        few = [r for r in skill.execute(conn, device=0, limit=1) if not r.get("_summary")]
        many = [r for r in skill.execute(conn, device=0, limit=5) if not r.get("_summary")]
    finally:
        conn.close()

    assert len(few) <= 1
    assert len(many) > len(few), (
        f"limit=5 returned {len(many)} rows, limit=1 returned {len(few)} — "
        "the cache ignored the parameters"
    )


def test_a_defaulted_parameter_shares_an_entry_with_an_explicit_one(fixture):
    """The key is built from resolved parameters, after defaults are applied.

    Keying on raw kwargs would treat an omitted `limit` and an explicit `limit`
    set to the same default as different, and re-execute for nothing.
    """
    skill = get_skill("top_kernels")
    default_limit = next(p.default for p in skill.params if p.name == "limit")
    calls = []
    original = skill.execute_fn

    def counting(conn, **kwargs):
        calls.append(kwargs)
        return original(conn, **kwargs)

    skill.execute_fn = counting
    conn = sqlite3.connect(fixture)
    try:
        skill.execute(conn)
        skill.execute(conn, limit=default_limit)
    finally:
        skill.execute_fn = original
        conn.close()

    assert len(calls) == 1, f"defaulted and explicit {default_limit} did not share an entry"


def test_a_cached_result_cannot_be_corrupted_by_its_consumer(fixture):
    """Rows are copied out, because consumers mutate them in place.

    Without the copy, one caller adding a key would change what every later
    caller sees — a bug that would surface far from here.
    """
    skill = get_skill("top_kernels")
    conn = sqlite3.connect(fixture)
    try:
        first = skill.execute(conn, limit=3)
        first[0]["_injected_by_consumer"] = True
        second = skill.execute(conn, limit=3)
    finally:
        conn.close()

    assert "_injected_by_consumer" not in second[0], "the cache handed out a shared row"


def test_a_second_reader_does_not_share_rows_with_the_first(fixture):
    """The read-side copy, which the store-side test does not reach.

    That test poisons the *first*, uncached list, so it only exercises the copy
    made on store. Removing the copy made on read left all seven tests passing —
    the classic "would pass with the body deleted".
    """
    skill = get_skill("top_kernels")
    conn = sqlite3.connect(fixture)
    try:
        skill.execute(conn, limit=3)          # populate
        second = skill.execute(conn, limit=3)  # first cached read
        second[0]["_injected_by_second_reader"] = True
        third = skill.execute(conn, limit=3)   # second cached read
    finally:
        conn.close()

    assert "_injected_by_second_reader" not in third[0], (
        "two cached readers were handed the same row object"
    )


def test_sql_only_skills_are_cached_too(fixture):
    """The store used to sit inside the execute_fn branch only.

    SQL-only skills therefore built a cache key and then took a guaranteed miss
    on every call, paying the cost without the benefit. Keep one registered
    SQL-only skill pinned here as skills move between execution paths.
    """
    skill = get_skill("stream_concurrency")
    assert skill.execute_fn is None, "this skill is no longer SQL-only; pick another"
    conn = sqlite3.connect(fixture)
    try:
        skill.execute(conn, limit=5)
        import nsys_ai.connection as connection

        cached = [
            k
            for bag in connection._sqlite_probe_bags.values()
            for k in bag
            if k.startswith("skill:stream_concurrency")
        ]
    finally:
        conn.close()
    assert cached, "a SQL-only skill produced no cache entry"


def test_separate_connections_do_not_share_results(fixture, profile_copy):
    """The cache is per connection, so two profiles cannot bleed into each other."""
    other = profile_copy("healthy_1pct.sqlite")
    skill = get_skill("top_kernels")

    a = sqlite3.connect(fixture)
    b = sqlite3.connect(other)
    try:
        rows_a = skill.execute(a, limit=3)
        rows_b = skill.execute(b, limit=3)
    finally:
        a.close()
        b.close()

    names_a = {r.get("kernel_name") or r.get("name") for r in rows_a}
    names_b = {r.get("kernel_name") or r.get("name") for r in rows_b}
    assert names_a != names_b, "two different profiles returned the same kernels"


@pytest.mark.parametrize("profile", ["h100_2gpu_1s", "healthy_1pct"])
def test_the_build_executes_fewer_skills_without_changing_findings(profile, profile_copy):
    """The whole point: less work, same answer.

    Findings were compared byte-for-byte against the pre-memo implementation on
    three committed profiles; this pins the execution count so a regression that
    reintroduces the duplicates is visible.
    """
    import collections

    from nsys_ai.evidence_builder import EvidenceBuilder
    from nsys_ai.profile import Profile
    from nsys_ai.skills import registry

    counts: collections.Counter = collections.Counter()
    originals = {}
    for sk in registry.all_skills():
        if sk.execute_fn is None:
            continue
        originals[sk.name] = sk.execute_fn

        def make(name, fn):
            def wrapped(conn, **kwargs):
                counts[name] += 1
                return fn(conn, **kwargs)

            return wrapped

        sk.execute_fn = make(sk.name, sk.execute_fn)

    try:
        with Profile(str(profile_copy(f"{profile}.sqlite"))) as prof:
            report = EvidenceBuilder(prof, device=0).build()
    finally:
        for sk in registry.all_skills():
            if sk.name in originals:
                sk.execute_fn = originals[sk.name]

    assert report.findings, "the build produced nothing"
    total = sum(counts.values())
    # Pins the count so a change that reintroduces duplicates is visible.
    # Deliberately not asserting that every remaining repeat is a genuinely
    # different parameter set — that is false today. `sync_cost_analysis` and
    # `overlap_breakdown` still repeat under keys differing only by parameters
    # the skill does not declare and does not read (`communicator_data` is
    # forwarded through `root_cause_matcher`), so three of the remaining
    # executions are semantically identical calls the memo fails to absorb.
    # Both independently enrolled default-pack skills add one distinct execution
    # rather than a duplicate. Narrowing the key to declared parameters would
    # take this to 20; that is a separate change with its own correctness surface.
    assert counts["kernel_launch_overhead"] == 1
    assert counts["nccl_compile_context_breakdown"] == 1
    assert total <= 23, f"{total} skill executions — duplicates came back: {dict(counts)}"
