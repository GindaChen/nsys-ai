"""Tests for the shared agent runner contract."""


def test_runner_uses_canonical_diagnose_pack(monkeypatch):
    from nsys_ai.agent import runner

    class FakeSkill:
        def execute(self, conn, **kwargs):
            return [{"skill": conn, "kwargs": kwargs}]

    monkeypatch.setattr(runner, "DIAGNOSE_DEFAULT", ["fake_skill"])
    monkeypatch.setattr(runner, "get_skill", lambda name: FakeSkill())

    evidence = runner.run_diagnose_pack("connection", trim_kwargs={"limit": 2})

    assert evidence == {"fake_skill": [{"skill": "connection", "kwargs": {"limit": 2}}]}


def test_runner_selection_is_deterministic_and_bounded():
    from nsys_ai.agent.runner import select_skills_for_question

    selected = select_skills_for_question(
        "why is the nccl overlap slow?",
        keyword_map={
            "nccl": ["nccl_breakdown", "overlap_breakdown"],
            "slow": ["top_kernels"],
        },
        fallback=["top_kernels"],
    )

    assert selected == ["nccl_breakdown", "overlap_breakdown", "top_kernels"]
    assert len(
        select_skills_for_question(
            "unknown",
            fallback=["a", "b", "c", "d", "e"],
        )
    ) == 4


def test_runner_does_not_repeat_root_cause_triage(monkeypatch):
    from nsys_ai.agent import runner

    calls = []

    def fake_execute(_conn, names, _trim):
        calls.append(names)
        return {name: [{"value": 1}] for name in names}

    monkeypatch.setattr(runner, "_execute_pack", fake_execute)
    evidence, selected = runner.run_question_evidence("conn", "why is this slow?")

    assert calls[0] == ["root_cause_matcher"]
    assert "root_cause_matcher" not in calls[1]
    assert "root_cause_matcher" in evidence
    assert "root_cause_matcher" not in selected


def test_runner_formats_abstention_as_unavailable_evidence():
    from nsys_ai.agent.runner import format_evidence_first_answer

    answer = format_evidence_first_answer(
        "why is this slow?",
        {"nvtx_kernel_map": [{"_abstained": True, "reason": "no NVTX events"}]},
        ["nvtx_kernel_map"],
        profile_path="profile.sqlite",
    )

    assert "cannot answer this profile question" in answer
    assert "unavailable: no NVTX events" in answer
    assert "metric=row_present=true" not in answer


def test_runner_does_not_synthesize_without_usable_evidence(monkeypatch):
    from nsys_ai.agent import runner

    monkeypatch.setitem(__import__("sys").modules, "litellm", object())
    assert runner.synthesize_evidence(
        "why?", {"top_kernels": [{"_abstained": True, "reason": "missing"}]}
    ) is None


def test_runner_answer_question_is_the_shared_transport_contract(monkeypatch):
    from nsys_ai.agent import runner

    calls = {}

    def fake_evidence(conn, question, *, trim_kwargs, use_llm):
        calls["evidence"] = (conn, question, trim_kwargs, use_llm)
        return {"root_cause_matcher": [{"pattern": "slow"}]}, ["top_kernels"]

    def fake_format(question, evidence, selected, **kwargs):
        calls["format"] = (question, evidence, selected, kwargs)
        return "answer"

    monkeypatch.setattr(runner, "run_question_evidence", fake_evidence)
    monkeypatch.setattr(runner, "format_evidence_first_answer", fake_format)

    answer, evidence, selected = runner.answer_question(
        "conn",
        "why?",
        profile_path="profile.nsys-rep",
        trim_kwargs={"trim_start_ns": 1, "trim_end_ns": 2},
        use_llm=False,
    )

    assert answer == "answer"
    assert evidence == {"root_cause_matcher": [{"pattern": "slow"}]}
    assert selected == ["root_cause_matcher", "top_kernels"]
    assert calls["evidence"] == (
        "conn",
        "why?",
        {"trim_start_ns": 1, "trim_end_ns": 2},
        False,
    )
    assert calls["format"][2] == ["root_cause_matcher", "top_kernels"]
