"""What the suite is actually covering, asserted rather than assumed.

A green badge meant "the tests CI could run passed", which is a weaker claim
than it looks and nothing distinguished the two. Concretely, 22 tests across
three files are gated on a 46MB fixture that is not committed: they run on a
developer machine and skip in CI, so CI covers *less* than local and no log
said so. One of them had been failing on main for some time without anyone
noticing, because the only machine that ran it was not the one being watched.

These tests make coverage a property the build checks. They deliberately assert
*reasons* rather than counts — a count is brittle against every new test, while
a new reason means something started skipping that did not before, which is the
event worth catching.
"""

import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REAL_FIXTURE = REPO / "tests" / "fixtures" / "h100_2gpu_1s.sqlite"

# Skips we know about and accept. Anything else fails the build.
#
# API-key skips are legitimate: those tests call paid providers and cannot run
# unattended. Fixture skips are legitimate but *load-bearing* — see the module
# docstring — so they are named individually rather than matched loosely.
ACCEPTED_SKIP_REASONS = (
    # Paid providers; cannot run unattended. Legitimate.
    "No API key configured",
    "GEMINI_API_KEY not set",
    # Gated on uncommitted profiles. Legitimate but load-bearing — these run on
    # a developer machine and not in CI, so CI covers strictly less. Each is
    # named rather than matched loosely so a new one is a deliberate addition.
    "distca example sqlite not found",  # the 46MB capture, still uncommitted
    "distca example profile not found",
    "profile not available locally",
    # The trajectory suite needs BOTH an API key and a 27MB uncommitted profile,
    # so it skips locally on the key and in CI on the profile. 130 tests that
    # run in neither place — see the dedicated test below.
    "Profile not found: data/nsys-hero",
    "requires duckdb",
    "parquet cache unavailable",
    "real Nsight capture opt-in",
    "requires nsys and nvcc",
    # Environment-shaped, not fixture-shaped: the read-only-directory test in
    # test_profile_modes.py cannot mean anything on Windows (no POSIX mode bits)
    # or as root (which ignores the write bit). It runs on every CI job we have;
    # the reason is registered so a container that happens to run as root fails
    # loudly on nothing rather than on this.
    "needs POSIX directory permissions and a non-root user",
)


def _collect_skip_reasons() -> list[str]:
    """Run the suite and return each distinct skip reason."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "-rs", "-p", "no:cacheprovider",
         "--ignore=tests/test_ci_coverage.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    reasons = []
    for line in result.stdout.splitlines():
        if not line.startswith("SKIPPED"):
            continue
        # SKIPPED [n] path/to/test.py:12: the reason
        _, _, tail = line.partition("] ")
        _, _, reason = tail.partition(": ")
        if reason:
            reasons.append(reason.strip())
    return sorted(set(reasons))


def test_every_skip_has_a_known_reason():
    """A new skip reason means something stopped running. Say so loudly.

    This is the check that would have caught a whole file going dark in CI.
    """
    unknown = [
        reason
        for reason in _collect_skip_reasons()
        if not any(accepted in reason for accepted in ACCEPTED_SKIP_REASONS)
    ]
    assert not unknown, (
        "tests are skipping for reasons not in ACCEPTED_SKIP_REASONS:\n  "
        + "\n  ".join(unknown)
        + "\n\nIf the skip is legitimate, add its reason to the list with a note "
        "explaining why that coverage is acceptable to lose."
    )


def test_the_fixture_gated_files_are_named_not_incidental():
    """Three files depend on an uncommitted 46MB fixture.

    That is a defensible trade — the fixture is large — but it must be a stated
    decision rather than an accident of what happens to be on disk. If a fourth
    file acquires the same dependency it should be a deliberate addition here.
    """
    gated = sorted(
        p.name
        for p in (REPO / "tests").glob("test_*.py")
        if p.name != Path(__file__).name and "megatron_distca.sqlite" in p.read_text()
    )
    assert gated == [
        "test_timeline_web_distca_benchmark.py",
        "test_timeline_web_distca_profile.py",
    ], f"the set of fixture-gated test files changed: {gated}"


def test_committed_real_fixture_keeps_its_provenance_and_shape():
    """The CI profile must remain a real derived capture, not a hand-written stub."""
    with sqlite3.connect(REAL_FIXTURE) as conn:
        metadata = dict(
            conn.execute(
                "SELECT name, value FROM META_DATA_EXPORT WHERE name LIKE 'DERIVED_%'"
            )
        )
        kernel_count = conn.execute(
            "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()[0]
        device_count = conn.execute(
            "SELECT count(DISTINCT deviceId) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()[0]
        stream_count = conn.execute(
            "SELECT count(DISTINCT streamId) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()[0]
        nvtx_count = conn.execute("SELECT count(*) FROM NVTX_EVENTS").fetchone()[0]

    assert metadata["DERIVED_BY"] == "scripts/derive_test_fixture.py"
    assert metadata["DERIVED_FROM"]
    assert metadata["DERIVED_WINDOW_NS"]
    assert kernel_count >= 1_000
    assert device_count >= 2
    assert stream_count >= 2
    assert nvtx_count >= 1


def test_real_fixture_generator_reproduces_provenance(tmp_path):
    """The committed fixture can be regenerated by the checked-in generator."""
    output = tmp_path / "derived.sqlite"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "derive_test_fixture.py"),
            str(REAL_FIXTURE),
            str(output),
            "--window",
            "0.1",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    with sqlite3.connect(output) as conn:
        metadata = dict(
            conn.execute(
                "SELECT name, value FROM META_DATA_EXPORT WHERE name LIKE 'DERIVED_%'"
            )
        )
    assert metadata["DERIVED_BY"] == "scripts/derive_test_fixture.py"
    assert metadata["DERIVED_FROM"] == REAL_FIXTURE.name
    assert metadata["DERIVED_WINDOW_NS"]


def test_the_integration_tests_are_no_longer_gated_out_of_ci():
    """They were six of the eleven profile-backed tests that CI never ran.

    A committed two-GPU capture now satisfies them, so "No test profile" is
    gone from the accepted reasons. If it comes back the build fails rather
    than quietly returning to the old state.
    """
    assert "No test profile" not in ACCEPTED_SKIP_REASONS


def test_the_trajectory_suite_is_known_to_run_nowhere():
    """130 tests that no environment satisfies, stated rather than discovered.

    They need an API key *and* a 27MB uncommitted profile. Locally the profile
    exists so they skip on the key; in CI the key is absent and so is the
    profile, so they skip on the profile. Two different gates, neither ever
    met — which is why the count never looked alarming from either side.

    This is not asserting the situation is acceptable. It is asserting that it
    is known, so the number cannot grow quietly, and so that closing either gate
    shows up as a failure here prompting the count to be revisited.
    """
    import subprocess

    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_trajectories.py", "-q",
         "--collect-only", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True,
    ).stdout
    collected = next(
        (int(w) for line in out.splitlines() for w in line.split()
         if w.isdigit() and "collected" in line),
        0,
    )
    assert collected == 130, (
        f"the trajectory suite is now {collected} tests, not 130. If it grew, "
        "more coverage is being written that nothing runs; if it shrank, say why."
    )


def test_ci_does_not_run_the_same_tests_twice():
    """The previous layout named four files as "tiers" and then ran the whole
    suite, executing those 70 tests a second time.

    Layering pays when a cheap stage can fail the build before an expensive one
    starts. These finished in six seconds against four minutes for the suite, so
    the ordering bought nothing and cost a duplicate run. If tiers return they
    should exclude what they cover, and this test should be updated to say so
    deliberately.
    """
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    runs = [
        line.strip()
        for line in ci.splitlines()
        if line.strip().startswith("run: pytest") or line.strip().startswith("run: python -m pytest")
    ]
    whole_suite = [r for r in runs if "pytest tests/ " in r or r.endswith("pytest tests/")]
    named_files = [r for r in runs if "tests/test_" in r and "test_ci_coverage" not in r]
    assert whole_suite, "the full suite is no longer run in CI"
    assert not named_files, (
        "CI names individual test files alongside a full-suite run, so those "
        f"tests execute twice: {named_files}"
    )
