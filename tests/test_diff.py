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
    conn.execute(
        "CREATE TABLE NVTX_EVENTS(text TEXT, globalTid INT, start INT, [end] INT)"
    )

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

