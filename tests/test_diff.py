import json
import sqlite3
import subprocess
import sys


def _make_profile(path: str, *, kernels: list[tuple], nvtx: list[tuple] | None = None):
    """
    Create a minimal Nsight-like SQLite export sufficient for Profile().

    kernels entries: (start_ns, end_ns, deviceId, streamId, correlationId, shortNameId, demangledId)
    nvtx entries: (text, globalTid, start_ns, end_ns)
    """
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE StringIds(id INT PRIMARY KEY, value TEXT)")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
        "start INT, [end] INT, deviceId INT, streamId INT, correlationId INT, "
        "shortName INT, demangledName INT)"
    )
    conn.execute("CREATE TABLE NVTX_EVENTS(text TEXT, globalTid INT, start INT, [end] INT)")

    # StringIds
    strings = {
        1: "kA",
        2: "kA_dem",
        3: "kB",
        4: "kB_dem",
        5: "kC",
        6: "kC_dem",
    }
    conn.executemany("INSERT INTO StringIds(id, value) VALUES(?,?)", list(strings.items()))

    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start, [end], deviceId, streamId, correlationId, shortName, demangledName) "
        "VALUES(?,?,?,?,?,?,?)",
        kernels,
    )

    if nvtx:
        conn.executemany(
            "INSERT INTO NVTX_EVENTS(text, globalTid, start, [end]) VALUES(?,?,?,?)",
            nvtx,
        )

    conn.commit()
    conn.close()


def _make_profile_with_runtime(
    path: str,
    *,
    marker: str = "step",
    tid: int = 1,
):
    """Minimal profile with RUNTIME + NVTX so detect_iterations finds one iteration."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE StringIds(id INT PRIMARY KEY, value TEXT)")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
        "start INT, [end] INT, deviceId INT, streamId INT, correlationId INT, "
        "shortName INT, demangledName INT)"
    )
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(globalTid INT, correlationId INT, start INT, [end] INT)"
    )
    conn.execute("CREATE TABLE NVTX_EVENTS(text TEXT, globalTid INT, start INT, [end] INT)")
    conn.execute("INSERT INTO StringIds(id, value) VALUES (1,'k'), (2,'k_dem')")
    # One kernel 1000–2000 ns, correlationId 1
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start, [end], deviceId, streamId, correlationId, shortName, demangledName) "
        "VALUES (1000, 2000, 0, 7, 1, 1, 2)"
    )
    # NVTX range that contains the kernel launch; RUNTIME 500–1000 so kernel 1000–2000 is inside
    conn.execute(
        "INSERT INTO NVTX_EVENTS(text, globalTid, start, [end]) VALUES (?, ?, 500, 2500)",
        (marker, tid),
    )
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME(globalTid, correlationId, start, [end]) VALUES (?, 1, 900, 1000)",
        (tid,),
    )
    conn.commit()
    conn.close()


def test_diff_engine_math(tmp_path):
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"

    # before:
    # - kA: 2 calls, 10ns each => 20ns
    # - kB: 1 call, 30ns => 30ns
    _make_profile(
        str(before),
        kernels=[
            (0, 10, 0, 7, 1, 1, 2),
            (20, 30, 0, 7, 2, 1, 2),
            (40, 70, 0, 7, 3, 3, 4),
        ],
        nvtx=[
            ("step", 1, 0, 100),
            ("warmup", 1, 0, 10),
        ],
    )

    # after:
    # - kA: 2 calls, 20ns each => 40ns (regression +20ns)
    # - kC: 1 call, 5ns => 5ns (new)
    _make_profile(
        str(after),
        kernels=[
            (0, 20, 0, 7, 1, 1, 2),
            (30, 50, 0, 7, 2, 1, 2),
            (60, 65, 0, 7, 3, 5, 6),
        ],
        nvtx=[
            ("step", 1, 0, 120),
        ],
    )

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        d = diff_profiles(b, a, gpu=0, trim=None, limit=10, sort="delta")

    # total GPU time = sum of aggregated kernel durations
    assert d.before.total_gpu_ns == 50
    assert d.after.total_gpu_ns == 45

    # kA regression should be present
    kA = [k for k in d.kernel_diffs if k.name == "kA"][0]
    assert kA.before_total_ns == 20
    assert kA.after_total_ns == 40
    assert kA.delta_ns == 20
    assert kA.classification == "regression"

    # kB removed
    kB = [k for k in d.kernel_diffs if k.name == "kB"][0]
    assert kB.before_total_ns == 30
    assert kB.after_total_ns == 0
    assert kB.classification == "removed"

    # kC new
    kC = [k for k in d.kernel_diffs if k.name == "kC"][0]
    assert kC.before_total_ns == 0
    assert kC.after_total_ns == 5
    assert kC.classification == "new"


def test_diff_cli_json_output(tmp_path):
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(
        str(before),
        kernels=[
            (0, 10, 0, 1, 1, 1, 2),
        ],
    )
    _make_profile(
        str(after),
        kernels=[
            (0, 20, 0, 1, 1, 1, 2),
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nsys_ai",
            "diff",
            str(before),
            str(after),
            "--gpu",
            "0",
            "--format",
            "json",
            "--no-ai",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["before"]["total_gpu_ns"] == 10
    assert payload["after"]["total_gpu_ns"] == 20
    assert payload["top_regressions"][0]["delta_ns"] == 10


def test_diff_with_trim_before_trim_after(tmp_path):
    """Phase C: diff_profiles supports trim_before/trim_after for iteration diff."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff import diff_profiles

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(
        str(before),
        kernels=[
            (100, 110, 0, 7, 1, 1, 2),
            (200, 230, 0, 7, 2, 3, 4),
        ],
        nvtx=[("step", 1, 0, 300)],
    )
    _make_profile(
        str(after),
        kernels=[
            (100, 130, 0, 7, 1, 1, 2),
            (250, 260, 0, 7, 2, 3, 4),
        ],
        nvtx=[("step", 1, 0, 300)],
    )
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        # Same window in both: 0–300 ns
        d = diff_profiles(
            b, a,
            gpu=0,
            trim_before=(0, 300),
            trim_after=(0, 300),
            limit=10,
        )
    assert d.before.total_gpu_ns == 40  # 10 + 30
    assert d.after.total_gpu_ns == 40   # 30 + 10
    kA = [k for k in d.kernel_diffs if k.name == "kA"][0]
    assert kA.delta_ns == 20  # 30 - 10
    kB = [k for k in d.kernel_diffs if k.name == "kB"][0]
    assert kB.delta_ns == -20  # 10 - 30


def test_diff_tools_search_nvtx_regions(tmp_path):
    """Phase C: search_nvtx_regions returns merged before/after NVTX names."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, search_nvtx_regions

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(
        str(before),
        kernels=[(0, 10, 0, 7, 1, 1, 2)],
        nvtx=[("Attention", 1, 0, 50), ("forward", 1, 0, 100)],
    )
    _make_profile(
        str(after),
        kernels=[(0, 10, 0, 7, 1, 1, 2)],
        nvtx=[("Attention", 1, 0, 60), ("backward", 1, 0, 80)],
    )
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        out = search_nvtx_regions(ctx, "Att", limit=10)
    assert "regions" in out
    assert out["query"] == "Att"
    names = [r["text"] for r in out["regions"]]
    assert "Attention" in names
    for r in out["regions"]:
        assert "in_before" in r and "in_after" in r
        assert "total_ns_before" in r and "total_ns_after" in r


def test_diff_tools_get_iteration_boundaries_shape(tmp_path):
    """Phase C: get_iteration_boundaries returns is_aligned and boundaries list."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_iteration_boundaries

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    # detect_iterations needs RUNTIME + NVTX with marker; use _make_profile_with_runtime
    _make_profile_with_runtime(str(before), marker="step", tid=1)
    _make_profile_with_runtime(str(after), marker="step", tid=1)
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="step")
        out = get_iteration_boundaries(ctx, marker="step", target_gpu=0)
    assert "is_aligned" in out
    assert "boundaries" in out
    assert "iteration_count_before" in out and "iteration_count_after" in out
    for bnd in out["boundaries"]:
        assert "before" in bnd and "after" in bnd
        assert "start_ns" in bnd["before"] or bnd["before"]["start_ns"] is None


def test_diff_cli_iteration_and_marker_help():
    """Phase C: diff --help shows --iteration and --marker."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--iteration" in result.stdout
    assert "iteration" in result.stdout.lower()
    assert "--marker" in result.stdout
    assert "sample_0" in result.stdout or "marker" in result.stdout.lower()


def test_diff_cli_chat_help():
    """Stage 6: diff --help shows --chat."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--chat" in result.stdout
    assert "chat" in result.stdout.lower()


def test_diff_tools_run_diff_tool_and_openai_tools(tmp_path):
    """Stage 6: run_diff_tool dispatches; TOOLS_DIFF_OPENAI and build_phase_c_system_prompt exist."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import (
        TOOLS_DIFF_OPENAI,
        DiffContext,
        build_phase_c_system_prompt,
        run_diff_tool,
    )

    assert len(TOOLS_DIFF_OPENAI) >= 10
    names = [t["function"]["name"] for t in TOOLS_DIFF_OPENAI]
    assert "search_nvtx_regions" in names
    assert "get_iteration_boundaries" in names
    assert "get_iteration_diff" in names

    before = tmp_path / "b.sqlite"
    after = tmp_path / "a.sqlite"
    _make_profile_with_runtime(str(before), marker="step")
    _make_profile_with_runtime(str(after), marker="step")
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="step")
        out = run_diff_tool(ctx, "get_iteration_boundaries", {})
    assert "boundaries" in out
    assert isinstance(out["boundaries"], list)

    prompt = build_phase_c_system_prompt(ctx, "/before.sqlite", "/after.sqlite", snapshot=None)
    assert "Before profile:" in prompt and "After profile:" in prompt
    assert "/before.sqlite" in prompt and "/after.sqlite" in prompt


def test_diff_tools_phase_c_prompt_export():
    """Phase C: system prompt and tool descriptions are exported for agent use."""
    from nsys_ai.diff_tools import PHASE_C_SYSTEM_PROMPT, TOOL_DESCRIPTIONS

    assert "Never guess names" in PHASE_C_SYSTEM_PROMPT
    assert "search_nvtx_regions" in PHASE_C_SYSTEM_PROMPT
    assert "get_launch_config_diff" in PHASE_C_SYSTEM_PROMPT or "Explain" in PHASE_C_SYSTEM_PROMPT
    assert "search_nvtx_regions" in TOOL_DESCRIPTIONS
    assert "get_iteration_diff" in TOOL_DESCRIPTIONS
    assert "get_global_diff" in TOOL_DESCRIPTIONS
    assert "get_region_diff" in TOOL_DESCRIPTIONS
    assert "get_gpu_imbalance_stats" in TOOL_DESCRIPTIONS
    assert "get_memory_profile_diff" in TOOL_DESCRIPTIONS


def test_diff_tools_stage5_warning_flags(tmp_path):
    """Stage 5: get_iteration_diff sets JIT_Compilation_Warning for iteration 0; payload has Hardware_Warning."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import DiffContext, get_iteration_diff

    before = tmp_path / "b.sqlite"
    after = tmp_path / "a.sqlite"
    _make_profile_with_runtime(str(before), marker="step")
    _make_profile_with_runtime(str(after), marker="step")
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="step")
        out = get_iteration_diff(ctx, 0, marker="step", target_gpu=0)
    assert "error" not in out or "iteration_index" in out
    assert "JIT_Compilation_Warning" in out
    assert out["JIT_Compilation_Warning"] is True  # iteration_index == 0
    assert "Hardware_Warning" in out
    assert isinstance(out["Hardware_Warning"], bool)


def test_diff_tools_region_diff_and_stubs(tmp_path):
    """Phase C: get_region_diff, get_launch_config_diff, get_memory_profile_diff return expected shape or error."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.diff_tools import (
        DiffContext,
        get_launch_config_diff,
        get_memory_profile_diff,
        get_region_diff,
    )

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10, 0, 7, 1, 1, 2)], nvtx=[("Attention", 1, 0, 50)])
    _make_profile(str(after), kernels=[(0, 20, 0, 7, 1, 1, 2)], nvtx=[("Attention", 1, 0, 60)])
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        out = get_region_diff(ctx, "Attention", target_gpu=0)
    assert "nvtx_exact_match" in out or "error" in out
    assert "wall_clock_ms" in out or "error" in out

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        launch = get_launch_config_diff(ctx, "kA", target_gpu=0)
    assert "error" in launch or "kernel_name" in launch
    assert "uses_tensor_core_likely" in launch or "error" in launch

    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        ctx = DiffContext(before=b, after=a, trim=None, marker="sample_0")
        mem = get_memory_profile_diff(ctx, target_gpu=0)
    assert "error" in mem


# ---------------------------------------------------------------------------
# AI narrative and executive summary (diff report augmentation)
# ---------------------------------------------------------------------------


def test_diff_build_executive_summary_with_tmp_path(tmp_path):
    """build_executive_summary with tmp_path fixture (stable content)."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.ai.diff_narrative import build_executive_summary
    from nsys_ai.diff import diff_profiles

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 20, 0, 7, 1, 1, 2)])
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        summary = diff_profiles(b, a, gpu=0, limit=10)
    text = build_executive_summary(summary)
    assert "slower" in text or "faster" in text
    assert "+10" in text or "10" in text


def test_diff_generate_narrative_no_model_returns_warning(tmp_path, monkeypatch):
    """generate_diff_narrative with no LLM configured returns warning, no exception."""
    import nsys_ai.chat_config as chat_config_mod
    from nsys_ai import profile as profile_mod
    from nsys_ai.ai.diff_narrative import DiffNarrative, generate_diff_narrative
    from nsys_ai.diff import diff_profiles
    monkeypatch.setattr(chat_config_mod, "_get_model_and_key", lambda _=None: (None, None), raising=False)

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 20, 0, 7, 1, 1, 2)])
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        summary = diff_profiles(b, a, gpu=0, limit=10)

    narrative = generate_diff_narrative(summary)
    assert isinstance(narrative, DiffNarrative)
    assert narrative.executive_summary
    assert narrative.ai_narrative is None
    assert narrative.warning is not None
    assert "No LLM" in narrative.warning or "no-ai" in narrative.warning.lower() or "API" in narrative.warning


def test_diff_format_terminal_with_narrative(tmp_path):
    """format_diff_terminal with narrative includes Executive Summary and optional AI block."""
    from nsys_ai import profile as profile_mod
    from nsys_ai.ai.diff_narrative import DiffNarrative
    from nsys_ai.diff import diff_profiles
    from nsys_ai.diff_render import format_diff_terminal

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 20, 0, 7, 1, 1, 2)])
    with profile_mod.open(str(before)) as b, profile_mod.open(str(after)) as a:
        summary = diff_profiles(b, a, gpu=0, limit=10)
    narrative = DiffNarrative(
        executive_summary="Total GPU time increased by +10.00us.",
        ai_narrative="The main regression is in kernel kA.",
        model="test",
        warning=None,
    )
    out = format_diff_terminal(summary, narrative=narrative)
    assert "Executive Summary" in out
    assert "Total GPU time increased" in out
    assert "AI Narrative" in out
    assert "main regression" in out
    out_no_ai = format_diff_terminal(summary, narrative=None)
    assert "Executive Summary" not in out_no_ai
    assert "AI Narrative" not in out_no_ai


def test_diff_cli_terminal_no_ai_shows_executive_summary(tmp_path):
    """diff --format terminal --no-ai shows Executive Summary and no AI section."""
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 20, 0, 7, 1, 1, 2)])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nsys_ai",
            "diff",
            str(before),
            str(after),
            "--gpu",
            "0",
            "--format",
            "terminal",
            "--no-ai",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Executive Summary" in result.stdout
    assert "Profile Diff" in result.stdout
    assert "Top regressions" in result.stdout
    # With --no-ai we do not call the LLM; Note section may appear only if we tried AI and failed
    # So we only require that the numeric report is present
    assert "10" in result.stdout and "20" in result.stdout


def test_diff_cli_json_structure_unchanged(tmp_path):
    """diff --format json output does not include narrative fields (contract unchanged)."""
    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    _make_profile(str(before), kernels=[(0, 10, 0, 7, 1, 1, 2)])
    _make_profile(str(after), kernels=[(0, 20, 0, 7, 1, 1, 2)])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nsys_ai",
            "diff",
            str(before),
            str(after),
            "--gpu",
            "0",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "before" in payload and "after" in payload and "top_regressions" in payload
    assert "executive_summary" not in payload
    assert "ai_narrative" not in payload
