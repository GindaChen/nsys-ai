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
                "kernel_overlap_matrix": [{"error": "kernel query failed"}],
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

    failed = [ln for ln in lines if "kernel_overlap_matrix" in ln]
    assert failed and "unavailable: kernel query failed" in failed[0]
    assert "metric=" not in failed[0]

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
        only_errors = agent._confidence_label(
            {"kernel_overlap_matrix": [{"error": "query failed"}]}, None
        )
        real = agent._confidence_label({"top_kernels": [{"kernel_name": "g"}]}, None)
    finally:
        agent.close()

    assert only_abstentions.startswith("0.20"), only_abstentions
    assert "no skill returned usable evidence" in only_abstentions
    assert only_errors.startswith("0.20"), only_errors
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
        error_only = agent._choose_verify_skill(
            {"kernel_overlap_matrix": [{"error": "query failed"}]},
            ["kernel_overlap_matrix"],
        )
    finally:
        agent.close()

    assert picked == "top_kernels"
    assert none_usable is None
    assert error_only is None


def test_unavailable_root_cause_cannot_become_the_primary_diagnosis():
    from nsys_ai.agent.loop import Agent

    agent = Agent(str(FIXTURE))
    unavailable = {
        "pattern": "Fabricated root cause",
        "recommendation": "Change the workload",
        "_abstained": True,
        "reason": "required table is absent",
    }
    try:
        diagnosis_row = agent._first_actionable_row([unavailable])
        answer = agent._format_evidence_first_answer(
            "why slow?",
            {"root_cause_matcher": [unavailable]},
            ["root_cause_matcher"],
        )
    finally:
        agent.close()

    assert diagnosis_row is None
    assert "Fabricated root cause" not in answer
    assert "Change the workload" not in answer
    assert "cannot answer this profile question" in answer


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

    class _Abstaining:
        name = "region_mfu"

        def execute(self, conn, **kwargs):
            return abstain("no NVTX in this profile")

    # Exercised, not grepped: an earlier version asserted a source string,
    # which passes whether or not the branch is reachable. Substituting an
    # abstaining skill proves the gate raises rather than reading the row as a
    # measurement. Note this branch is defensive today — no skill the gate runs
    # abstains — so the substitution is what makes it testable at all.
    # get_skill is imported inside the function, so patch it at the registry.
    import nsys_ai.skills.registry as registry

    original = registry.get_skill
    registry.get_skill = lambda name: _Abstaining()
    try:
        with pytest.raises(sg.SolGateError) as exc:
            sg.evaluate_sol_gates(
                sqlite3.connect(":memory:"),
                [sg.parse_sol_gate("myregion:50")],
                theoretical_flops=1e12,
            )
    finally:
        registry.get_skill = original
    assert "no NVTX in this profile" in str(exc.value)


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

    # Scoped to idle: an unscoped check passes on a fixture where the idle
    # logic never ran, which is exactly what the sibling fixture does.
    idle_claimed = [
        (f.label, f.headroom_ms)
        for f in report.findings
        if f.category == "idle" and f.headroom_ms
    ]
    assert idle_claimed == [], f"idle it saw and weighed was called recoverable: {idle_claimed}"
    assert [f for f in report.findings if f.category == "idle"], (
        "no idle findings at all — the gaps were filtered, not judged"
    )

    # Documenting current behaviour rather than blessing it: severity is judged
    # per gap with no reference to the share of the run, so a healthy profile
    # still emits warnings. Tracked separately; pinned here so a change is
    # deliberate rather than accidental.
    idle_warnings = [
        f for f in report.findings if f.category == "idle" and f.severity == "warning"
    ]
    assert len(idle_warnings) <= 5, f"idle warning noise grew to {len(idle_warnings)}"
    assert not [f for f in report.findings if f.category == "idle" and f.severity == "critical"]


# ── Structure: the marker stays private to the module that defines it ───────


def test_the_marker_is_not_re_implemented_across_the_codebase():
    """`_abstained` is an implementation detail of `skills/base.py`.

    It leaked into eight hand-rolled checks — `isinstance(row, dict) and
    row.get("_abstained")` repeated in the agent, the gate and the manifest —
    which is the same "every consumer re-implements the convention" shape that
    produced the original six leaks. Consumers use the predicates instead, so a
    change to the representation touches one file.
    """
    src_root = Path(__file__).resolve().parent.parent / "src" / "nsys_ai"
    offenders = []
    for path in src_root.rglob("*.py"):
        if path.name == "base.py" and path.parent.name == "skills":
            continue  # the definition lives here
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or '"""' in stripped:
                continue
            if "_abstained" in stripped:
                offenders.append(f"{path.relative_to(src_root)}:{lineno}: {stripped}")
    assert not offenders, "raw marker checks outside skills/base.py:\n" + "\n".join(offenders)


# ── #307: Start/End (eventType 60) ranges are not attributed, and say so ───
#
# Every attribution path filters `eventType = 59`. A profile annotated only
# with Start/End ranges therefore produced an empty result and a formatter
# asking "are NVTX annotations present?" — while `profile_health_manifest`
# named the very same regions on the very same profile, because
# `Profile.aggregate_nvtx_ranges` does not filter eventType. Two skills in one
# run disagreeing, and the wrong one phrased as a question whose answer is yes.
#
# The decision is NOT to widen the filter. NVIDIA keeps push/pop ranges on a
# per-thread stack, so they nest strictly by construction — which is precisely
# what the sweep exploits. Start/End ranges "expose arbitrary concurrency (not
# just nesting)" and "the start of a range can occur on a different thread than
# the end". Feeding those to a nesting stack yields wrong parents, not more
# coverage. So the four Push/Pop-dependent skills abstain with a stated reason.

# The four skills whose attribution is a per-thread nesting sweep. The two
# iteration skills are deliberately absent: `overlap.detect_iterations` never
# filters eventType, so they work on a Start/End profile and guarding them
# would be a regression. `test_iteration_timing_still_runs_on_start_end_ranges`
# pins that.
PUSHPOP_DEPENDENT = [
    ("nvtx_kernel_map", {}),
    ("nvtx_layer_breakdown", {}),
    ("code_attribution_candidates", {"start_ns": 0, "end_ns": 10**9}),
    ("nccl_compile_context_breakdown", {}),
]


def _build_nvtx_profile(path: Path, rows: list[tuple]) -> Path:
    """A minimal attributable profile whose NVTX rows are exactly ``rows``.

    Kernels and their correlated runtime launches sit inside the ranges, so the
    only reason a skill can fail to attribute is the eventType — otherwise a
    passing abstention assertion would prove nothing.

    ``endGlobalTid`` is present because that is how real exports carry a
    Start/End pair that begins and ends on different threads: one row with two
    thread ids. Stated plainly — no capture available here has a populated one
    (it is NULL across all 15.9M NVTX rows of the largest trace on this
    machine), so the cross-thread row below is a reasoned reconstruction of the
    documented shape, not an observed sample.
    """
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(globalPid INTEGER, deviceId INTEGER,
          streamId INTEGER, correlationId INTEGER, start INTEGER,"end" INTEGER,
          shortName INTEGER, demangledName INTEGER,
          gridX INTEGER, gridY INTEGER, gridZ INTEGER,
          blockX INTEGER, blockY INTEGER, blockZ INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(globalTid INTEGER,
          correlationId INTEGER, start INTEGER,"end" INTEGER, nameId INTEGER);
        CREATE TABLE NVTX_EVENTS(globalTid INTEGER, endGlobalTid INTEGER,
          start INTEGER,"end" INTEGER, text TEXT, eventType INTEGER,
          rangeId INTEGER, textId INTEGER);
        """
    )
    conn.executemany(
        "INSERT INTO StringIds VALUES (?,?)",
        [(1, "gemm_nn_128x128"), (2, "ncclDevKernel_AllReduce"), (24, "cudaLaunchKernel")],
    )
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,1,1,1,128,1,1)",
        [
            (100, 0, 7, 1, 1_000_000, 2_000_000, 1, 1),
            (100, 0, 7, 2, 2_200_000, 3_000_000, 2, 2),
        ],
    )
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,24)",
        [(100, 1, 900_000, 950_000), (100, 2, 2_100_000, 2_150_000)],
    )
    conn.executemany(
        "INSERT INTO NVTX_EVENTS(globalTid, endGlobalTid, start, [end], text, "
        "eventType, rangeId, textId) VALUES (?,?,?,?,?,?,?,NULL)",
        rows,
    )
    conn.commit()
    conn.close()
    return path


# Two ranges on ONE thread that overlap without nesting — impossible under
# push/pop, which is the whole point — plus a pair whose start and end are on
# different threads.
_START_END_ROWS = [
    (100, None, 500_000, 2_500_000, "train_step", 60, 0),
    (100, None, 2_000_000, 3_500_000, "overlapping_phase", 60, 1),
    (100, 101, 800_000, 2_800_000, "cross_thread_span", 60, 2),
]
_PUSH_POP_ROWS = [
    (100, None, 500_000, 3_500_000, "train_step", 59, 0),
    (100, None, 900_000, 2_500_000, "forward", 59, 1),
]
# eventType 34 is NvtxMark in the profile's own ENUM_NSYS_EVENT_TYPE table:
# an instant, with no duration to attribute a kernel to.
_MARKS_ONLY_ROWS = [
    (100, None, 900_000, -1, "step_begin", 34, 0),
    (100, None, 2_600_000, -1, "step_end", 34, 1),
]


@pytest.fixture
def start_end_profile(tmp_path):
    return _build_nvtx_profile(tmp_path / "start_end.sqlite", _START_END_ROWS)


@pytest.fixture
def push_pop_profile(tmp_path):
    return _build_nvtx_profile(tmp_path / "push_pop.sqlite", _PUSH_POP_ROWS)


def test_the_start_end_fixture_really_is_start_end_only(start_end_profile):
    """Guard the premise, so the abstention assertions cannot go vacuous."""
    conn = sqlite3.connect(start_end_profile)
    try:
        types = {r[0] for r in conn.execute("SELECT DISTINCT eventType FROM NVTX_EVENTS")}
        cross = conn.execute(
            "SELECT count(*) FROM NVTX_EVENTS WHERE endGlobalTid IS NOT NULL "
            "AND endGlobalTid != globalTid"
        ).fetchone()[0]
    finally:
        conn.close()
    assert types == {60}, f"the fixture is no longer Start/End-only: {types}"
    assert cross == 1, "the cross-thread row that push/pop cannot express is gone"


@pytest.mark.parametrize("skill_name,kwargs", PUSHPOP_DEPENDENT)
def test_start_end_only_nvtx_abstains_instead_of_reporting_nothing(
    skill_name, kwargs, start_end_profile
):
    """The bug: silence that reads as "this profile has no annotation"."""
    from nsys_ai.skills.base import is_abstention

    conn = sqlite3.connect(start_end_profile)
    try:
        rows = get_skill(skill_name).execute(conn, **kwargs)
    finally:
        conn.close()

    assert is_abstention(rows), f"{skill_name} reported nothing instead of abstaining: {rows[:1]}"
    reason = rows[0]["reason"]
    # Name the kind and the number, so the user can check it against their own
    # annotation calls rather than guessing what "unsupported" means.
    assert "Start/End" in reason, reason
    assert "60" in reason, reason
    assert "59" in reason, reason
    # And say what to do about it.
    assert "nvtxRangePush" in reason, reason
    assert rows[0].get("nvtx_event_types") == "60"


@pytest.mark.parametrize("skill_name,kwargs", PUSHPOP_DEPENDENT)
def test_push_pop_nvtx_is_not_swept_up_by_the_new_guard(skill_name, kwargs, push_pop_profile):
    """A guard that is too broad is the real failure mode.

    Same fixture builder, same kernels, same ranges — only eventType differs —
    so this pins that the abstention above is caused by the event kind and not
    by the fixture being unattributable in the first place.
    """
    from nsys_ai.skills.base import is_abstention

    conn = sqlite3.connect(push_pop_profile)
    try:
        rows = get_skill(skill_name).execute(conn, **kwargs)
    finally:
        conn.close()
    assert not is_abstention(rows), (
        f"{skill_name} told a Push/Pop profile its annotation was the wrong kind: {rows[0]}"
    )


def test_the_push_pop_fixture_is_actually_attributable(push_pop_profile):
    """Otherwise the test above passes on an empty result and proves nothing."""
    conn = sqlite3.connect(push_pop_profile)
    try:
        rows = get_skill("nvtx_kernel_map").execute(conn)
    finally:
        conn.close()
    assert rows, "the positive fixture attributes no kernels — the negative test is vacuous"
    assert {r["nvtx_text"] for r in rows} & {"forward", "train_step"}


def test_iteration_timing_still_runs_on_start_end_ranges(start_end_profile):
    """Deliberate scope. `overlap.detect_iterations` is eventType-agnostic, so
    these two skills work on Start/End ranges and must not be guarded."""
    from nsys_ai.skills.base import is_abstention

    conn = sqlite3.connect(start_end_profile)
    try:
        rows = get_skill("iteration_timing").execute(conn, marker="train_step")
    finally:
        conn.close()
    assert not is_abstention(rows), (
        "iteration_timing was guarded on Push/Pop; it does not need it and "
        f"profiles that work today would stop: {rows[0]}"
    )


def test_marks_only_nvtx_abstains_with_its_own_reason(tmp_path):
    """Neither 59 nor 60: a third reason, and it reports the type it measured."""
    from nsys_ai.skills.base import is_abstention

    path = _build_nvtx_profile(tmp_path / "marks.sqlite", _MARKS_ONLY_ROWS)
    conn = sqlite3.connect(path)
    try:
        rows = get_skill("nvtx_kernel_map").execute(conn)
    finally:
        conn.close()

    assert is_abstention(rows), rows[:1]
    reason = rows[0]["reason"]
    assert rows[0].get("nvtx_event_types") == "34", reason
    assert "34" in reason, reason
    # It must still name what it needs, or the user cannot act on it.
    assert "59" in reason and "nvtxRangePush" in reason, reason


def test_nvtxt_imported_ranges_are_not_called_marks(tmp_path):
    """The fallthrough must not describe rows it never inspected.

    A profile built by `nsys import` from an .nvtxt annotation file carries its
    ranges as eventType 70/71 (NvtxtPushPopRange / NvtxtStartEndRange in the
    profile's own ENUM_NSYS_EVENT_TYPE). Those are not attributed here — the
    sweep filters eventType 59 and nothing else — so an abstention is right.
    But an earlier version of this branch probed only "does any row exist" and
    then told the user the table held "only marks, categories or domain
    records, which are instants": two claims about their data, both false, from
    a branch that had inspected neither. Reporting `nvtx_event_types="none"`
    was wrong there for the same reason.
    """
    from nsys_ai.skills.base import is_abstention

    for event_type in (70, 71):
        path = _build_nvtx_profile(
            tmp_path / f"nvtxt_{event_type}.sqlite",
            [(100, None, 500_000, 3_000_000, "imported_step", event_type, 0)],
        )
        conn = sqlite3.connect(path)
        try:
            rows = get_skill("nvtx_kernel_map").execute(conn)
        finally:
            conn.close()

        assert is_abstention(rows), f"eventType {event_type}: {rows[:1]}"
        reason = rows[0]["reason"]
        # The claims that were false.
        assert "instants" not in reason, reason
        assert "only marks" not in reason, reason
        assert "no range rows" not in reason, reason
        # And the measured truth, in both the prose and the machine field.
        assert str(event_type) in reason, reason
        assert rows[0].get("nvtx_event_types") == str(event_type), rows[0]


def test_the_fallthrough_names_event_types_from_the_profiles_own_enum(tmp_path):
    """`ENUM_NSYS_EVENT_TYPE` is the authority for the export in hand.

    Naming the type from a list hard-coded here would be the same mistake this
    branch exists to correct, one Nsight release later. The bare number is the
    documented fallback when the profile does not carry the table, which the
    fixtures above exercise.
    """
    from nsys_ai.skills.base import requires_pushpop_nvtx

    path = _build_nvtx_profile(
        tmp_path / "nvtxt_named.sqlite",
        [(100, None, 500_000, 3_000_000, "imported_step", 70, 0)],
    )
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE ENUM_NSYS_EVENT_TYPE(id INTEGER PRIMARY KEY, label TEXT);"
        "INSERT INTO ENUM_NSYS_EVENT_TYPE VALUES (70, 'NvtxtPushPopRange');"
    )
    conn.commit()
    try:
        guard = requires_pushpop_nvtx(conn, needs="Region attribution")
    finally:
        conn.close()

    assert guard is not None
    assert "70 (NvtxtPushPopRange)" in guard[0]["reason"], guard[0]["reason"]


def test_an_empty_nvtx_table_is_left_alone_by_the_push_pop_guard(tmp_path):
    """Deliberate scope, and the one case the guard must NOT claim.

    A present-but-empty NVTX_EVENTS table is a third state, distinct from
    "Start/End only" and "marks only", and it is not what #307 is about. The
    "only marks, categories or domain records" reason would be plainly false
    there — the table holds no records at all — and `code_attribution_candidates`
    already answers this case with a structured row carrying the requested
    window and the skill's limitations, which a reason string would replace with
    something less useful.
    """
    from nsys_ai.skills.base import is_abstention, requires_pushpop_nvtx

    path = _build_nvtx_profile(tmp_path / "empty_nvtx.sqlite", [])
    conn = sqlite3.connect(path)
    try:
        guard = requires_pushpop_nvtx(conn, needs="Region attribution")
        rows = get_skill("nvtx_layer_breakdown").execute(conn)
    finally:
        conn.close()

    assert guard is None, f"an empty NVTX table was claimed as the wrong kind: {guard}"
    assert not is_abstention(rows)


def test_the_push_pop_guard_fails_open_when_there_is_no_eventType_column(tmp_path):
    """A schema surprise must not be reported as "your annotation is the wrong
    kind".

    Not hypothetical: eight modules in this suite (test_baseline, test_diff,
    test_e2e_golden_loop, test_fingerprint, test_profile_resolve,
    test_region_mfu, test_skills, test_tools_profile) build NVTX_EVENTS with no
    eventType column at all. The count is only there to say the shape is
    common; if it drifts, nothing about this test does.
    """
    from nsys_ai.skills.base import is_abstention, requires_pushpop_nvtx

    path = tmp_path / "no_eventtype.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE NVTX_EVENTS(globalTid INTEGER, start INTEGER,"end" INTEGER, text TEXT);
        INSERT INTO NVTX_EVENTS VALUES (100, 0, 9000, 'my_layer');
        """
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(path)
    try:
        guard = requires_pushpop_nvtx(conn, needs="Region attribution")
    finally:
        conn.close()
    assert guard is None, f"a missing column was turned into a false abstention: {guard}"
    assert not is_abstention(guard or [])


def test_a_non_numeric_eventType_column_fails_open_too(tmp_path):
    """SQLite columns are dynamically typed, so the census can meet text.

    Same contract as the missing-column case: a schema surprise must produce no
    claim at all, not a claim about the user's annotation, and certainly not a
    ValueError out of a guard.
    """
    from nsys_ai.skills.base import requires_pushpop_nvtx

    path = tmp_path / "text_eventtype.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE NVTX_EVENTS(globalTid INTEGER, start INTEGER,"end" INTEGER,
          text TEXT, eventType TEXT);
        INSERT INTO NVTX_EVENTS VALUES (100, 0, 9000, 'my_layer', 'NvtxMark');
        """
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(path)
    try:
        guard = requires_pushpop_nvtx(conn, needs="Region attribution")
    finally:
        conn.close()
    assert guard is None, f"a text eventType was turned into a false abstention: {guard}"


def test_the_push_pop_guard_is_defined_once():
    """Same reason as the NVTX guard below: an inlined eventType probe would
    drift, and one that hard-codes NVTX_EVENTS misses NVTX_EVENTS_V2."""
    src_root = Path(__file__).resolve().parent.parent / "src" / "nsys_ai"
    copies = [
        p.relative_to(src_root)
        for p in (src_root / "skills" / "builtins").rglob("*.py")
        if "eventType = 60" in p.read_text() or "eventType=60" in p.read_text()
    ]
    assert copies == [], f"an eventType-60 probe was inlined in: {copies}"


def test_the_nvtx_guard_is_defined_once():
    """Six skills needed the same resolve-and-abstain block.

    Each had its own copy, and one copy shipped with an exact name match that
    missed `NVTX_EVENTS_V2` — the kind of divergence duplication guarantees.
    """
    src_root = Path(__file__).resolve().parent.parent / "src" / "nsys_ai"
    copies = [
        p.relative_to(src_root)
        for p in (src_root / "skills" / "builtins").rglob("*.py")
        if 'resolve_activity_tables().get("nvtx")' in p.read_text()
    ]
    assert copies == [], f"the NVTX guard was re-inlined in: {copies}"
