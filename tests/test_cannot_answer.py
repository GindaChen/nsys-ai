"""The matrix: every way this tool says "I cannot answer", driven end to end.

"There is no problem here" and "I could not look" are different answers, and a
consumer that cannot tell them apart reads the second as the first — a clean
bill of health issued by an analysis that never ran. An automated pass over the
installed package counted eight encodings of "could not look", five of them at
exit 0 (#400).

`docs/notes/cannot-answer.md` decides the shape:

* one marker — `_abstained: True` plus a non-empty `reason`, on the payload
  that could not answer, at whatever level that is;
* one predicate — `nsys_ai.cannot_answer.is_cannot_answer(payload)`;
* one exit code — **0**, because an abstention is a correct and complete answer
  to the question that was asked. Exit 1 stays "the command did not finish",
  exit 2 stays "the command could not accept what it was given".

This file is the product of that decision, not a footnote to it. One test per
row of #400's table, each driving the real path, each asserting the single
predicate and the chosen exit code. Rows that do not conform yet are `xfail`
with the issue that will fix them, `strict=True` so that fixing one turns this
file red until the marker is removed — the ratchet is the point. Adding a ninth
encoding means adding a row here first, which is where someone notices there is
already a shape for it.

A ninth turned up while this was being written and is recorded in the note:
`nccl_payload_breakdown` returns `{"error": ..., "backend_limitation": True}`
and `{"error": ..., "binary_data_rows": 0}` — abstentions wearing an untyped
`error` key, the same class as #345's five. It needs a backend that raises on a
BLOB read to reproduce, so it is documented rather than driven here.
"""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from nsys_ai.cannot_answer import cannot_answer, cannot_answer_reason, is_cannot_answer

ROOT = Path(__file__).resolve().parent.parent
MOCK = ROOT / "tests" / "fixtures" / "mock.sqlite"

# The decision, as constants, so a test reads as the contract rather than as a
# magic number: a stated abstention is a successful run.
EXIT_ANSWERED = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE = 2


def _cli(*args: str, cwd: Path, timeout: float = 180.0) -> subprocess.CompletedProcess:
    """Run the CLI as a real process, against *this* worktree's sources.

    The package is installed editable from another checkout, so without the
    PYTHONPATH prefix a subprocess would measure main rather than the branch.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT / "src"), env.get("PYTHONPATH", "")) if part
    )
    return subprocess.run(
        [sys.executable, "-m", "nsys_ai", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _stdout_json(result: subprocess.CompletedProcess):
    """Parse stdout as JSON, failing with the actual output rather than a stack."""
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"stdout is not JSON ({exc}).\n"
            f"exit={result.returncode}\nstdout={result.stdout[:400]!r}\n"
            f"stderr={result.stderr[-800:]!r}"
        )


@pytest.fixture
def profile_without_activity_tables(tmp_path: Path) -> Path:
    """A profile whose CUPTI/NVTX tables are absent, not merely empty.

    The distinction is the whole subject: a table that is missing means the
    analysis *cannot run*, where a table that is present and empty means it ran
    and found nothing. Both are copied out of the fixture so no cache artifact
    lands beside the committed one.
    """
    path = tmp_path / "no_activity_tables.sqlite"
    shutil.copy(MOCK, path)
    conn = sqlite3.connect(path)
    try:
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        dropped = [n for n in names if n.startswith("CUPTI_") or "NVTX" in n]
        for name in dropped:
            conn.execute(f'DROP TABLE "{name}"')
        conn.commit()
    finally:
        conn.close()
    if not dropped:
        raise AssertionError("fixture premise broken: mock.sqlite had no activity tables to drop")
    return path


@pytest.fixture
def profile_without_kernel_rows(tmp_path: Path) -> Path:
    """A profile whose kernel table exists and is empty — a capture that recorded nothing."""
    path = tmp_path / "no_kernel_rows.sqlite"
    shutil.copy(MOCK, path)
    conn = sqlite3.connect(path)
    try:
        conn.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture
def mock_copy(tmp_path: Path) -> Path:
    """mock.sqlite, copied, so cache artifacts never land beside the committed fixture."""
    path = tmp_path / "mock.sqlite"
    shutil.copy(MOCK, path)
    return path


# ---------------------------------------------------------------------------
# The predicate itself
# ---------------------------------------------------------------------------


def test_the_predicate_agrees_with_the_skill_layer_it_generalises():
    """One shape, not two. `abstain()` output must satisfy the general predicate.

    `skills.base.is_abstention` is the row-list predicate that already exists;
    `is_cannot_answer` is the same rule lifted to envelopes and route bodies. If
    the two ever disagree, consumers at different levels reach different
    conclusions about the same payload, which is the state #400 describes.
    """
    from nsys_ai.skills.base import abstain, is_abstention

    rows = abstain("no NVTX_EVENTS table")

    assert is_abstention(rows) is True
    assert is_cannot_answer(rows) is True
    assert is_cannot_answer(rows[0]) is True
    assert cannot_answer_reason(rows) == "no NVTX_EVENTS table"


def test_an_empty_result_is_not_an_abstention():
    """The distinction the marker exists to make, asserted directly."""
    assert is_cannot_answer([]) is False
    assert is_cannot_answer([{"kernel": "gemm", "total_ms": 0.0}]) is False
    assert is_cannot_answer({"findings": []}) is False
    assert is_cannot_answer(None) is False
    assert is_cannot_answer("could not run") is False


def test_an_untyped_error_key_is_not_the_marker():
    """A broken tool and an inapplicable analysis are different states.

    Teaching the predicate to accept `error` would merge them again and make
    every genuine failure look like a considered abstention.
    """
    assert is_cannot_answer({"error": "no such table: CUPTI_ACTIVITY_KIND_KERNEL"}) is False
    assert is_cannot_answer([{"total_ms": 0.0, "error": "tables not found"}]) is False


def test_a_marker_without_a_reason_cannot_be_built():
    """Silence, only typed, is still silence."""
    with pytest.raises(ValueError):
        cannot_answer("")
    with pytest.raises(ValueError):
        cannot_answer("   ")


# ---------------------------------------------------------------------------
# The matrix — one test per row of #400's table
# ---------------------------------------------------------------------------


def test_row_1_a_skill_that_cannot_run_abstains_with_a_reason(mock_copy: Path, tmp_path: Path):
    """Row 1: `abstain(reason)` — the intended encoding, and the reference for the rest.

    `mock.sqlite` has no NVTX table, so NVTX-dependent skills cannot run at all.
    Exit 0 is the decision, not an accident: the command ran and returned the
    correct answer to the question asked.
    """
    result = _cli("skill", "run", "nvtx_kernel_map", str(mock_copy), "--format", "json", cwd=tmp_path)

    assert result.returncode == EXIT_ANSWERED, result.stderr[-800:]
    payload = _stdout_json(result)
    assert is_cannot_answer(payload), payload
    assert "NVTX" in cannot_answer_reason(payload)


@pytest.mark.xfail(
    strict=True,
    reason="#345: execute_fn skills return a bare [] when the table is absent, "
    "which reads identically to 'ran, found nothing'",
)
def test_row_2_a_bare_empty_list_when_the_table_is_absent(
    profile_without_activity_tables: Path, tmp_path: Path
):
    """Row 2: `[]` in `--format json`, exit 0, indistinguishable from a real result.

    `gpu_idle_gaps` needs kernel activity to find a gap between kernels. With no
    kernel table at all it returns `[]` — the same output it gives for a profile
    with no idle gaps, which is a compliment rather than a failure to look.
    """
    result = _cli(
        "skill", "run", "gpu_idle_gaps", str(profile_without_activity_tables),
        "--format", "json", cwd=tmp_path,
    )

    assert result.returncode == EXIT_ANSWERED, result.stderr[-800:]
    payload = _stdout_json(result)
    assert is_cannot_answer(payload), payload


def test_row_3_an_error_key_beside_real_looking_zeros(
    profile_without_activity_tables: Path, tmp_path: Path
):
    """Row 3: the missing synchronization tables are a typed abstention."""
    result = _cli(
        "skill", "run", "sync_cost_analysis", str(profile_without_activity_tables),
        "--format", "json", cwd=tmp_path,
    )

    assert result.returncode == EXIT_ANSWERED, result.stderr[-800:]
    payload = _stdout_json(result)
    assert is_cannot_answer(payload), payload


@pytest.mark.xfail(
    strict=True,
    reason="#345: a missing table surfaces as {'error': ...} at exit 1, reporting a "
    "profile the tool cannot analyse as a tool that broke",
)
def test_row_4_a_missing_table_reported_as_a_runtime_error(
    profile_without_activity_tables: Path, tmp_path: Path
):
    """Row 4: `{"error": ...}` on stdout, exit 1.

    Nothing is wrong with the tool here: it was handed a profile that does not
    contain the activity it was asked about. Under the decision that is an
    abstention at exit 0, and exit 1 stays for the command that did not finish.
    """
    result = _cli(
        "skill", "run", "top_kernels", str(profile_without_activity_tables),
        "--format", "json", cwd=tmp_path,
    )

    payload = _stdout_json(result)
    assert is_cannot_answer(payload), payload
    assert result.returncode == EXIT_ANSWERED, result.stderr[-800:]


@pytest.mark.xfail(
    strict=True,
    reason="#401: a command that cannot answer prints nothing at all and exits 0, "
    "which is the loudest form of the same ambiguity",
)
def test_row_5_a_command_that_cannot_answer_says_nothing_at_all(
    profile_without_kernel_rows: Path, tmp_path: Path
):
    """Row 5: empty stdout, exit 0 — the human surface of the same bug.

    `summary` on a capture that recorded no kernel activity produces zero bytes
    on both streams and exits successfully. A person cannot tell that from a
    successful run whose report scrolled past, and no exit code would have told
    them either: what they need is the sentence.
    """
    result = _cli("summary", str(profile_without_kernel_rows), cwd=tmp_path)

    assert result.returncode == EXIT_ANSWERED, result.stderr[-800:]
    said = (result.stdout + result.stderr).strip()
    assert said, "the command exited 0 having said nothing on either stream"
    assert "kernel" in said.lower(), f"the reason is not stated: {said[:300]!r}"


def test_row_6_a_raw_traceback_breaks_the_json_contract(mock_copy: Path, tmp_path: Path):
    """Row 6: traceback, exit 1, empty stdout even under `--format json`.

    This one is a usage error rather than an abstention — the caller omitted a
    required parameter — so the decision puts it at exit 2 with a coded message.
    What it must not do is emit a stack trace where a JSON document was
    promised, because that fails the consumer before any predicate can run.
    """
    result = _cli("skill", "run", "region_mfu", str(mock_copy), "--format", "json", cwd=tmp_path)

    assert "Traceback" not in result.stderr, result.stderr[-800:]
    assert result.returncode == EXIT_USAGE, (
        f"exit={result.returncode}; stderr={result.stderr[-400:]!r}"
    )
    payload = _stdout_json(result)
    assert not is_cannot_answer(payload), "a usage error is not an abstention"


def test_row_7_the_web_session_route_carries_the_marker():
    """Row 7: the web `limitation` shape, converged onto the marker.

    Driven through the handler method that builds every one of those bodies —
    `_handle_loop_phase`, `_handle_loop_proposal`, `_handle_loop_reprofile`,
    `_handle_loop_diagnose` and `_handle_loop_diff` all route through it — with
    the response captured rather than sent, so the shape is asserted without
    standing a server up and publishing a session first.

    The status stays 400 and the aliases stay too: this is a convergence, not a
    rename, and existing browser code keeps working.
    """
    from nsys_ai.web import _ViewerHandler

    captured: dict = {}

    class _Recorder:
        def _json_response(self, obj, status=200):
            captured["body"] = obj
            captured["status"] = status

    _ViewerHandler._session_limitation(
        _Recorder(),
        "nsys-ai evidence",
        "session mode does not run diagnose in the browser",
    )

    body = captured["body"]
    assert is_cannot_answer(body), body
    assert cannot_answer_reason(body) == "session mode does not run diagnose in the browser"
    # Retained: the command that *can* do the job is what the marker cannot carry.
    assert body["cli"] == "nsys-ai evidence"
    # Retained aliases — pinned so the convergence is not silently a rename.
    assert body["limitation"] is True
    assert body["error"] == body["reason"]
    # The transport answer is separate from the analysis answer, on purpose.
    assert captured["status"] == 400


@pytest.mark.xfail(
    strict=True,
    reason="#399: diff computes verdict=inconclusive and comparability 0.0 and "
    "persists both, but the payload carries no marker a general consumer can apply",
)
def test_row_8_an_inconclusive_diff_carries_the_marker(
    mock_copy: Path, profile_without_kernel_rows: Path, tmp_path: Path
):
    """Row 8: `verdict: inconclusive` with comparability 0.0.

    This row is already honest to a reader who knows the diff vocabulary — the
    verdict and the warning both say so. The marker is for the reader who does
    not, which is every generic consumer of a JSON envelope. Exit stays 0: no
    gate was requested, so nothing was asserted and nothing failed.
    """
    result = _cli(
        "diff", str(mock_copy), str(profile_without_kernel_rows),
        "--format", "json", "--no-ai", cwd=tmp_path,
    )

    assert result.returncode == EXIT_ANSWERED, result.stderr[-800:]
    payload = _stdout_json(result)
    # Premise: this pair really is incomparable, or the row proves nothing.
    assert payload["verdict"] == "inconclusive", payload["verdict"]
    assert payload["comparability_confidence"] == 0.0
    assert is_cannot_answer(payload), sorted(payload)
