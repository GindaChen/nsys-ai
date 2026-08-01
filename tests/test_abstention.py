"""A skill that cannot run must say so, not vanish.

Silence has three possible meanings and callers could not tell them apart:
the skill ran and found nothing; the skill could not run because the profile
lacks a table it needs; or the skill raised and a caller swallowed it.

The last was the real state of affairs. `EvidenceBuilder` catches `Exception`
and logs, so on a profile captured without NVTX the four annotation-dependent
skills raised `no such table: NVTX_EVENTS` and simply disappeared from the
findings. A user reading that output sees no NVTX findings and no explanation,
which is indistinguishable from a clean bill of health — the failure mode a
verify-first tool can least afford.

`tests/fixtures/mock.sqlite` genuinely has no `NVTX_EVENTS` table, so it is the
no-NVTX negative fixture called for in #262; nothing had asserted against it.
"""

import sqlite3
from pathlib import Path

import pytest

from nsys_ai.skills.base import abstain
from nsys_ai.skills.registry import get_skill

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mock.sqlite"

# The skills that need NVTX annotation to say anything at all.
NVTX_DEPENDENT = [
    "iteration_timing",
    "nccl_compile_context_breakdown",
    "nvtx_kernel_map",
    "nvtx_layer_breakdown",
]

# Not NVTX, same contract: a missing table means the skill cannot run. This one
# had the identical guard shape returning a bare `[]`, which read as "ran, every
# thread idle" rather than "CPU sampling was never captured".
CANNOT_RUN_ON_FIXTURE = [*NVTX_DEPENDENT, "thread_utilization"]


def test_the_fixture_really_has_no_nvtx():
    """Guard the premise: if NVTX_EVENTS ever appears here, these tests are vacuous."""
    conn = sqlite3.connect(FIXTURE)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert "NVTX_EVENTS" not in tables


@pytest.mark.parametrize("skill_name", NVTX_DEPENDENT)
def test_nvtx_skills_abstain_with_a_reason_instead_of_raising(skill_name):
    skill = get_skill(skill_name)
    assert skill is not None, f"{skill_name} is not registered"

    conn = sqlite3.connect(FIXTURE)
    try:
        rows = skill.execute(conn)  # must not raise
    finally:
        conn.close()

    assert rows, f"{skill_name} returned nothing — abstention must be visible"
    assert rows[0].get("_abstained") is True, (
        f"{skill_name} returned rows without the abstention marker: {rows[0]}"
    )
    reason = rows[0].get("reason", "")
    assert reason, f"{skill_name} abstained with no reason"
    # The reason has to be actionable, not merely present.
    assert "NVTX" in reason
    assert "re-capture" in reason.lower() or "annotate" in reason.lower()


@pytest.mark.parametrize("skill_name", NVTX_DEPENDENT)
def test_abstention_is_distinguishable_from_an_empty_result(skill_name):
    """The whole point: `[]` and "could not run" must not look alike."""
    conn = sqlite3.connect(FIXTURE)
    try:
        rows = skill.execute(conn) if (skill := get_skill(skill_name)) else []
    finally:
        conn.close()
    assert rows != [], f"{skill_name} is back to returning an ambiguous empty list"


def test_abstain_helper_shape():
    rows = abstain("because", table="NVTX_EVENTS")
    assert rows == [{"_abstained": True, "reason": "because", "table": "NVTX_EVENTS"}]


def test_evidence_builder_no_longer_loses_these_skills_to_an_exception():
    """The regression that motivated all of this.

    Before, these four raised and `EvidenceBuilder` caught and logged, so the
    skill left no trace. Abstaining keeps them in the pipeline, which is what
    lets a caller report "no NVTX annotation" instead of saying nothing.
    """
    conn = sqlite3.connect(FIXTURE)
    try:
        for name in NVTX_DEPENDENT:
            skill = get_skill(name)
            rows = skill.execute(conn)
            # A skill that abstains contributes no findings, but it does so
            # visibly — the rows survive to the caller.
            assert rows[0]["_abstained"] is True
            if skill.to_findings_fn:
                # Same invoker EvidenceBuilder uses: to_findings_fn signatures
                # vary across skills, and this is the path production takes.
                from nsys_ai.evidence_builder import _invoke_to_findings

                findings = _invoke_to_findings(
                    skill.to_findings_fn, rows, {"profile_id": "p"}
                )
                assert findings == [], (
                    f"{name} produced findings from an abstention row: {findings}"
                )
    finally:
        conn.close()


# ── The other half of #262: a healthy profile must stay quiet ───────────────


HEALTHY = Path(__file__).resolve().parent / "fixtures" / "healthy_1pct.sqlite"


def test_a_one_percent_overhead_profile_claims_no_recoverable_time():
    """Calling ~1% dispatch cost a bottleneck is the credibility-ending failure.

    The fixture is 400 back-to-back kernels separated by 10us gaps: 4.0ms of
    idle across a 404ms span. Describing the run is fine — which kernel
    dominates, what the bound class is. Claiming time back is not.
    """
    from nsys_ai.evidence_builder import EvidenceBuilder
    from nsys_ai.profile import Profile

    with Profile(str(HEALTHY)) as prof:
        report = EvidenceBuilder(prof, device=0).build()

    claimed = [(f.label, f.headroom_ms) for f in report.findings if f.headroom_ms]
    assert claimed == [], f"a healthy profile was told it had recoverable time: {claimed}"


def test_the_healthy_fixture_really_is_healthy():
    """Guard the premise, so the assertion above cannot go vacuous."""
    conn = sqlite3.connect(HEALTHY)
    try:
        rows = conn.execute(
            "SELECT MIN(start), MAX([end]), COUNT(*), SUM([end]-start) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()
    finally:
        conn.close()
    lo, hi, n, busy = rows
    idle_share = 1.0 - busy / (hi - lo)
    assert n == 400
    assert 0.005 < idle_share < 0.015, f"fixture idle share drifted to {idle_share:.2%}"

    # The kernels must be VISIBLE to the analysis, not merely present in the
    # table. Profile.kernels() inner-joins demangledName to StringIds, so a
    # dangling id silently yields an empty profile — and the "claims no
    # headroom" assertion would then pass because nothing was analysed at all.
    from nsys_ai.profile import Profile

    with Profile(str(HEALTHY)) as prof:
        assert len(prof.kernels(0)) == 400, "the analysis surface cannot see the kernels"


# ── Consumers must not render an abstention as data ────────────────────────
#
# The original version of this change had a green suite and a broken CLI:
# nothing exercised the formatters or the agent's evidence path, so an
# abstention row reached code that indexes data columns.


@pytest.mark.parametrize("skill_name", CANNOT_RUN_ON_FIXTURE)
def test_formatting_an_abstention_does_not_crash(skill_name):
    """`skill run <name> <no-nvtx-profile>` must print the reason, not traceback.

    Each skill's own `format_fn` indexes its data columns, so this is handled
    once in `Skill.format_rows` rather than in four formatters that could each
    forget.
    """
    skill = get_skill(skill_name)
    conn = sqlite3.connect(FIXTURE)
    try:
        text = skill.run(conn)  # execute + format, the CLI's path
    finally:
        conn.close()

    assert "not applicable to this profile" in text
    # The reason must name the table that is missing, so the user can tell
    # which capture option to turn on rather than guessing.
    assert "NVTX_EVENTS" in text or "COMPOSITE_EVENTS" in text
    # The actionable half of the reason has to survive to the user.
    assert "re-capture" in text.lower() or "annotate" in text.lower()


def test_agent_does_not_cite_an_abstention_as_a_measurement():
    """An unavailable skill is not evidence for anything.

    Routing it through the metric path produced
    `metric=row_present=true`, dressing an absence up as a measurement —
    precisely the ungrounded claim the answer contract exists to prevent.
    """
    from nsys_ai.agent.loop import Agent

    agent = Agent(str(FIXTURE))
    try:
        lines = agent._evidence_lines(
            {
                "nvtx_kernel_map": [{"_abstained": True, "reason": "no NVTX here"}],
                "top_kernels": [{"kernel_name": "gemm", "total_ms": 12.3}],
            }
        )
    finally:
        agent.close()

    abstained = [ln for ln in lines if "nvtx_kernel_map" in ln]
    assert abstained, "the unavailable skill vanished instead of being reported"
    assert "unavailable:" in abstained[0]
    assert "no NVTX here" in abstained[0]
    assert "metric=" not in abstained[0], "an absence was rendered as a measurement"

    # Real evidence is unaffected.
    real = [ln for ln in lines if "top_kernels" in ln]
    assert real and "metric=" in real[0]


def test_is_abstention_helper():
    from nsys_ai.skills.base import is_abstention

    assert is_abstention(abstain("x")) is True
    assert is_abstention([]) is False
    assert is_abstention(None) is False
    assert is_abstention([{"kernel_name": "gemm"}]) is False


# ── Abstention must not become a number, a confidence, or a verify target ──
#
# Each of these was a real leak: the row is truthy and dict-shaped, so code
# that checks `if rows:` or `row.get(key, default)` treats it as data.


def test_manifest_does_not_invent_an_iteration_from_an_abstention():
    """The sharpest instance: a fabricated measurement on the finding that
    exists to report there is nothing to measure.

    `_summarize_iterations` filtered on `is_real_iteration` and `heuristic`.
    An abstention row has neither, so it took both defaults and was counted as
    one genuine 0ms iteration — leaving the manifest asserting
    `has_nvtx: false` and `iteration_count: 1` in the same object.
    """
    skill = get_skill("profile_health_manifest")
    conn = sqlite3.connect(FIXTURE)
    try:
        rows = skill.execute(conn)
    finally:
        conn.close()

    nvtx = rows[0].get("nvtx") or {}
    assert nvtx.get("has_nvtx") is False
    assert "iteration_count" not in nvtx, f"fabricated iteration data: {nvtx}"
    assert "median_iter_ms" not in nvtx


def test_an_unavailable_skill_does_not_raise_the_answers_confidence():
    """A tool that sells grounding cannot let a skill that could not run
    improve its own stated confidence."""
    from nsys_ai.agent.loop import Agent

    agent = Agent(str(FIXTURE))
    try:
        only_abstentions = agent._confidence_label(
            {"nvtx_kernel_map": [{"_abstained": True, "reason": "x"}]}, None
        )
        real = agent._confidence_label({"top_kernels": [{"kernel_name": "g"}]}, None)
    finally:
        agent.close()

    assert only_abstentions.startswith("0.20"), only_abstentions
    assert "no skill returned usable evidence" in only_abstentions
    # Real evidence still scores higher, so the guard is not a blanket downgrade.
    assert real.startswith("0.60")


def test_verify_command_never_points_at_an_unavailable_skill():
    """`## Verify` must be runnable and actually verify something."""
    from nsys_ai.agent.loop import Agent

    agent = Agent(str(FIXTURE))
    try:
        picked = agent._choose_verify_skill(
            {
                "nvtx_kernel_map": [{"_abstained": True, "reason": "x"}],
                "top_kernels": [{"kernel_name": "g"}],
            },
            ["nvtx_kernel_map", "top_kernels"],
        )
        none_usable = agent._choose_verify_skill(
            {"nvtx_kernel_map": [{"_abstained": True}]}, ["nvtx_kernel_map"]
        )
    finally:
        agent.close()

    assert picked == "top_kernels"
    assert none_usable is None


def test_guard_uses_the_table_resolver_not_an_exact_name():
    """Nsight ships versioned variants such as NVTX_EVENTS_V2, and the parquet
    backend registers views by filename.

    An exact-name check told users holding an annotated profile to "re-capture
    with NVTX enabled" — a functional regression carrying a false message.
    """
    import tempfile

    path = Path(tempfile.mkdtemp()) / "v2.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,"end" INTEGER,
          deviceId INTEGER, streamId INTEGER, correlationId INTEGER,
          shortName INTEGER, demangledName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(globalTid INTEGER,
          correlationId INTEGER, start INTEGER,"end" INTEGER, nameId INTEGER);
        CREATE TABLE NVTX_EVENTS_V2(globalTid INTEGER, start INTEGER,"end" INTEGER,
          text TEXT, eventType INTEGER, textId INTEGER);
        """
    )
    conn.execute("INSERT INTO StringIds VALUES (1,'gemm_nn_128x128')")
    conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1000,5000,0,7,10,1,1)")
    conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1,10,500,900,1)")
    conn.execute("INSERT INTO NVTX_EVENTS_V2 VALUES (1,0,9000,'my_layer',59,NULL)")
    conn.commit()
    conn.close()

    from nsys_ai.skills.base import is_abstention

    conn = sqlite3.connect(path)
    try:
        rows = get_skill("nvtx_kernel_map").execute(conn)
    finally:
        conn.close()
    assert not is_abstention(rows), "told a profile that HAS NVTX to re-capture with NVTX"


def test_a_missing_table_abstains_even_when_it_is_not_nvtx():
    """The contract is "cannot run", not "no NVTX".

    `thread_utilization` guards on COMPOSITE_EVENTS and previously returned a
    bare `[]`, which reads as "ran, every thread idle" — a claim about the
    workload rather than about the capture.
    """
    from nsys_ai.skills.base import is_abstention

    conn = sqlite3.connect(FIXTURE)
    try:
        rows = get_skill("thread_utilization").execute(conn)
    finally:
        conn.close()

    assert is_abstention(rows)
    assert "COMPOSITE_EVENTS" in rows[0]["reason"]
    assert "sampling" in rows[0]["reason"].lower()


# ── Structural guards: the contract must hold without each consumer opting in ──
#
# Six leaks happened because `abstain()` was a convention that eight separate
# consumers each had to remember, checking variously `if rows:`,
# `"error" in row`, `"bucket" in r`, or `row.get(key, default)`. These tests
# pin the places where the contract is now enforced centrally instead.


def test_to_findings_never_sees_an_abstention_row():
    """Filtered in the invoker, so no `to_findings_fn` can mint a finding.

    Before, the skills that were safe were safe only through unrelated guards —
    an early return on a row count, a length check — which a refactor could
    remove without anyone noticing the connection.
    """
    from nsys_ai.evidence_builder import _invoke_to_findings

    def explode(rows, context=None):  # must never be called
        raise AssertionError(f"to_findings_fn received an abstention: {rows}")

    assert _invoke_to_findings(explode, abstain("no NVTX"), {"profile_id": "p"}) == []

    # Real rows still reach it.
    seen = {}

    def record(rows, context=None):
        seen["rows"] = rows
        return []

    _invoke_to_findings(record, [{"kernel_name": "gemm"}], {"profile_id": "p"})
    assert seen["rows"] == [{"kernel_name": "gemm"}]


def test_the_sol_gate_fails_rather_than_passing_on_an_unmeasured_region():
    """A gate cannot pass on a measurement that was never taken.

    `sol_gate` reads `rows[0]` to decide CI pass/fail and checked only for an
    `error` key. An abstention has no such key, so it would have been read as a
    measurement — an unmeasurable profile reporting "no regression" and letting
    a real one through.
    """
    import nsys_ai.sol_gate as sg

    assert 'row.get("_abstained")' in Path(sg.__file__).read_text(), (
        "the gate no longer distinguishes an abstention from a measurement"
    )


def test_the_llm_is_not_handed_abstentions_as_analysis_data():
    """The one path a type check cannot protect.

    Serialised beside real rows under a header calling it analysis data, a
    model can reasonably narrate "could not run" as a property of the workload.
    They are split out and labelled as unavailable instead.
    """
    from nsys_ai.agent import loop as agent_loop

    src = Path(agent_loop.__file__).read_text()
    assert "usable, unavailable = {}, {}" in src
    assert "json.dumps(usable" in src, "abstentions are still serialised as data"
    assert "could NOT run" in src


@pytest.mark.parametrize(
    "skill_name,kwargs",
    [
        ("code_attribution_candidates", {"start_ns": 0, "end_ns": 10**9}),
        ("iteration_detail", {"iteration": 0}),
        ("nccl_payload_breakdown", {}),
    ],
)
def test_the_remaining_skills_abstain_rather_than_raise(skill_name, kwargs):
    """Contract completeness: `is_abstention()` is now sufficient.

    These three previously raised or returned an error row, so a consumer had
    to handle three shapes to learn one thing.
    """
    from nsys_ai.skills.base import is_abstention

    skill = get_skill(skill_name)
    conn = sqlite3.connect(FIXTURE)
    try:
        rows = skill.execute(conn, **kwargs)
    finally:
        conn.close()
    assert is_abstention(rows), f"{skill_name} did not abstain: {rows[:1]}"


JUDGED = Path(__file__).resolve().parent / "fixtures" / "healthy_judged_1pct.sqlite"


def test_idle_the_pipeline_actually_sees_is_still_not_called_recoverable():
    """The sibling fixture tests the judgement, not the filter.

    `healthy_1pct.sqlite` has 10us gaps, below the 1ms floor, so its idle never
    reaches a threshold — it passes because nothing weighed it. Here the gaps
    are 2ms, above the floor, so the pipeline sees 19 of them totalling 38ms
    and must still decline to call 0.99% recoverable.
    """
    from nsys_ai.evidence_builder import EvidenceBuilder
    from nsys_ai.profile import Profile
    from nsys_ai.skills.registry import get_skill

    conn = sqlite3.connect(JUDGED)
    try:
        seen = get_skill("gpu_idle_gaps").execute(conn, device=0)
    finally:
        conn.close()
    summary = next((r for r in seen if r.get("_summary")), {})
    assert summary.get("gap_count", 0) > 0, "the gaps were filtered out — tests the wrong thing"

    with Profile(str(JUDGED)) as prof:
        report = EvidenceBuilder(prof, device=0).build()

    claimed = [(f.label, f.headroom_ms) for f in report.findings if f.headroom_ms]
    assert claimed == [], f"idle it saw and weighed was called recoverable: {claimed}"

    # Documenting current behaviour rather than blessing it: severity is judged
    # per gap with no reference to the share of the run, so a healthy profile
    # still emits warnings. Tracked separately; pinned here so a change is
    # deliberate rather than accidental.
    idle_warnings = [
        f for f in report.findings if f.category == "idle" and f.severity == "warning"
    ]
    assert len(idle_warnings) <= 5, f"idle warning noise grew to {len(idle_warnings)}"
    assert not [f for f in report.findings if f.category == "idle" and f.severity == "critical"]
