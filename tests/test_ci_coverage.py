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

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Skips we know about and accept. Anything else fails the build.
#
# API-key skips are legitimate: those tests call paid providers and cannot run
# unattended. Fixture skips are legitimate but *load-bearing* — see the module
# docstring — so they are named individually rather than matched loosely.
ACCEPTED_SKIP_REASONS = (
    "No API key configured",
    "GEMINI_API_KEY not set",
    "distca example sqlite not found",
    "distca example profile not found",
    "requires duckdb",
    "parquet cache unavailable",
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
