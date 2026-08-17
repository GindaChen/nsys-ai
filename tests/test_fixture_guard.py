"""The fixture guard is this suite's own safety net, so it needs coverage.

Issue #347 is a bug that landed repeatedly: a test opens a committed profile
directly, `CREATE INDEX IF NOT EXISTS` rewrites it, and a multi-megabyte binary
diff rides along in the next commit. The guard in conftest.py is what stops
that. Without a test, a later cleanup can soften the raise into a warning and
nothing would notice -- which is how the original bug survived five times.
"""

from __future__ import annotations

import conftest
import pytest


def _drive(monkeypatch, fixture_dir, *, stale=()):
    """Run the session-scoped guard as a plain generator against a temp dir."""
    monkeypatch.setattr(conftest, "FIXTURE_DIR", fixture_dir)
    monkeypatch.setattr(conftest, "_already_modified_fixtures", lambda: list(stale))
    return conftest._committed_fixtures_are_left_alone.__wrapped__()


def test_a_rewritten_fixture_fails_the_session(monkeypatch, tmp_path):
    """The case the guard exists for: bytes changed while the suite ran."""
    fixture = tmp_path / "profile.sqlite"
    fixture.write_bytes(b"original")

    guard = _drive(monkeypatch, tmp_path)
    next(guard)
    fixture.write_bytes(b"grown by an index")

    with pytest.raises(AssertionError, match="rewrote committed fixtures in place"):
        next(guard, None)


def test_an_untouched_fixture_passes(monkeypatch, tmp_path):
    """A guard that fires on a clean run would be turned off within a week."""
    (tmp_path / "profile.sqlite").write_bytes(b"original")

    guard = _drive(monkeypatch, tmp_path)
    next(guard)

    assert next(guard, None) is None


def test_an_already_dirty_checkout_fails_before_the_suite_runs(monkeypatch, tmp_path):
    """Digests compare a session against itself, so a dirty tree hides writes.

    A contributor whose checkout is already grown -- from a run before the
    guard, or from a red run they simply repeated -- would otherwise get a
    green suite and commit the churn exactly as before.
    """
    (tmp_path / "profile.sqlite").write_bytes(b"already grown")

    guard = _drive(monkeypatch, tmp_path, stale=["profile.sqlite"])

    with pytest.raises(AssertionError, match="already modified before this session"):
        next(guard)


def test_a_vanished_fixture_is_reported(monkeypatch, tmp_path):
    """Reporting only names present at the end would miss a deletion entirely."""
    fixture = tmp_path / "profile.sqlite"
    fixture.write_bytes(b"original")

    guard = _drive(monkeypatch, tmp_path)
    next(guard)
    fixture.unlink()

    with pytest.raises(AssertionError, match="profile.sqlite"):
        next(guard, None)
