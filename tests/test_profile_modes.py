import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from nsys_ai import parquet_cache
from nsys_ai.parquet_cache import _cache_dir_for
from nsys_ai.profile import Profile


def test_invalid_cache_mode(minimal_nsys_db_path):
    with pytest.raises(ValueError, match="Unknown cache_mode: 'invalid'"):
        Profile(str(minimal_nsys_db_path), cache_mode="invalid")


def test_invalid_backend(minimal_nsys_db_path):
    with pytest.raises(ValueError, match="Unknown backend: 'invalid'"):
        Profile(str(minimal_nsys_db_path), backend="invalid")


def test_parquetdir_backend_rejects_non_auto_cache_mode(minimal_nsys_db_path):
    with pytest.raises(
        ValueError,
        match="cache_mode is not supported with backend='parquetdir'; use cache_mode='auto'.",
    ):
        Profile(str(minimal_nsys_db_path), backend="parquetdir", cache_mode="direct")


def test_cache_mode_parquet(minimal_nsys_db_path):
    with Profile(str(minimal_nsys_db_path), cache_mode="parquet") as prof:
        # Execute a query using alias view syntax (which Parquet mode supports via registration)
        res = prof.db.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()
        assert res[0] >= 0

    # Parquet cache MUST exist
    cache_dir = _cache_dir_for(str(minimal_nsys_db_path))
    assert cache_dir.exists()
    assert (cache_dir / ".cache_version").exists()


def test_cache_mode_direct(minimal_nsys_db_path, tmp_path):
    # Move the db so it doesn't have a cache from the previous test
    db_path = tmp_path / "test_direct.sqlite"
    with open(minimal_nsys_db_path, "rb") as src, open(db_path, "wb") as dst:
        dst.write(src.read())

    with Profile(str(db_path), cache_mode="direct") as prof:
        # Verify the alias view exists and we can query unqualified
        res = prof.db.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()
        assert res[0] >= 0

    # Parquet cache MUST NOT exist for direct scanning
    cache_dir = _cache_dir_for(str(db_path))
    assert not cache_dir.exists()


def test_cache_mode_auto_small(minimal_nsys_db_path, tmp_path):
    db_path = tmp_path / "test_auto_small.sqlite"
    with open(minimal_nsys_db_path, "rb") as src, open(db_path, "wb") as dst:
        dst.write(src.read())

    with Profile(str(db_path), cache_mode="auto"):
        pass

    cache_dir = _cache_dir_for(str(db_path))
    assert cache_dir.exists()


# ── The `auto` policy (issue #317) ───────────────────────────────────
#
# These pin what `auto` decides, in the order parquet_cache.open_auto_db asks:
# size does not decide, the env override does, disk does, and a profile that
# cannot be cached still gets a DuckDB engine rather than raw sqlite3. The last
# one pins that the override reaches `skill run`, which opens its connection
# without a Profile at all.


def _copy_fixture(minimal_nsys_db_path, dest):
    with open(minimal_nsys_db_path, "rb") as src, open(dest, "wb") as dst:
        dst.write(src.read())
    return dest


def test_cache_mode_auto_builds_for_a_large_profile(minimal_nsys_db_path, tmp_path):
    """Size alone must not send a profile to direct mode.

    Replaces test_cache_mode_auto_large_mocked, which asserted the opposite for
    the same 100 MB. The old 50 MB rule dated from the interval-join map builder
    (>15 min on a 3.5 GB capture); against the stack sweep that replaced it, an
    eight-skill run is faster end to end on a cold build at 93 MB, 235 MB,
    924 MB and 3.7 GB alike — the table lives in parquet_cache.open_auto_db.

    The patch below is global, and deliberately so. `mock.patch` on a dotted
    path resolves through the `os` module object, which `nsys_ai.profile` and
    `nsys_ai.parquet_cache` share with everything else, so patching
    "nsys_ai.profile.os.path.getsize" patches posixpath.getsize — verified, it
    is byte-for-byte the old test's "os.path.getsize". Nothing else consulted
    during this open changes its answer at 100 MB: `is_cache_valid` reads
    getmtime not getsize, `cache_is_affordable` clamps to the 128 MB floor
    either way, the build banner needs 500 MB, and
    `_should_defer_nvtx_kernel_map` returns True at any size unless
    NSYS_AI_DEFER_NVTX_KERNEL_MAP_MB is set, which it is not here. So the
    assertion below can only be satisfied by the size rule being gone.
    """
    db_path = _copy_fixture(minimal_nsys_db_path, tmp_path / "auto_large.sqlite")

    with mock.patch("os.path.getsize", return_value=100.0 * 1e6):
        with Profile(str(db_path), cache_mode="auto") as prof:
            res = prof.db.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()
            assert res[0] >= 0

    assert _cache_dir_for(str(db_path)).exists(), (
        "auto declined to build a cache for a 100MB profile — the size threshold is back"
    )


def test_cache_mode_auto_skips_build_when_disk_is_short(minimal_nsys_db_path, tmp_path):
    """No room for the cache: query directly, do not fail and do not half-build."""
    db_path = _copy_fixture(minimal_nsys_db_path, tmp_path / "auto_nodisk.sqlite")
    usage = mock.Mock(total=10**9, used=10**9, free=1024)

    with mock.patch("nsys_ai.parquet_cache.shutil.disk_usage", return_value=usage):
        with Profile(str(db_path), cache_mode="auto") as prof:
            assert prof.db is not None, "a full disk must not cost the DuckDB engine"
            res = prof.db.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()
            assert res[0] >= 0

    assert not _cache_dir_for(str(db_path)).exists()


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason="needs POSIX directory permissions and a non-root user",
)
def test_cache_mode_auto_read_only_directory_builds_nothing(minimal_nsys_db_path, tmp_path):
    """A profile on a read-only mount: no build *attempt*, and a working engine.

    Before #317 this reached `_build_lock`, which fails at os.open(O_CREAT) with
    PermissionError; Profile's blanket except then set `db = None`. Now the
    affordability check declines up front, so nothing tries to write.

    "Declines up front" is the whole point, so it is what the assertion has to
    say. Checking only that the profile still opens does not distinguish this
    from the build failing at the lock and the fallback rescuing it — that path
    is already covered by test_failed_cache_build_falls_back_to_duckdb_not_sqlite3,
    and a test that passes either way pins nothing here. Spying on build_cache
    is the difference: it must never be reached.
    """
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    db_path = _copy_fixture(minimal_nsys_db_path, ro_dir / "auto_ro.sqlite")
    ro_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with mock.patch(
            "nsys_ai.parquet_cache.build_cache", wraps=parquet_cache.build_cache
        ) as spy:
            with Profile(str(db_path), cache_mode="auto") as prof:
                assert prof.db is not None, (
                    "a read-only profile directory dropped the profile to raw sqlite3"
                )
                res = prof.db.execute(
                    "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME"
                ).fetchone()
                assert res[0] >= 0
        spy.assert_not_called()
        assert not _cache_dir_for(str(db_path)).exists()
    finally:
        ro_dir.chmod(stat.S_IRWXU)


def test_failed_cache_build_falls_back_to_duckdb_not_sqlite3(minimal_nsys_db_path, tmp_path):
    """A cache that cannot be built costs the cache, not the query engine.

    The old fallback set `self.db = None`, which drops the whole run onto raw
    `sqlite3`. That is not "the same, without Parquet acceleration": the
    enriched `kernels` view does not exist there, so tensor_core_usage fails
    outright rather than running slower, and on a 93MB profile the eight-skill
    query time was 8.15 s against 5.62 s on the direct DuckDB attach.

    The assertion below is the difference, stated as SQL that only the DuckDB
    path can answer.
    """
    db_path = _copy_fixture(minimal_nsys_db_path, tmp_path / "auto_buildfail.sqlite")

    def _boom(_path):
        raise RuntimeError("simulated build failure (ENOSPC mid-COPY, unreadable source, ...)")

    with mock.patch("nsys_ai.parquet_cache.open_cached_db", side_effect=_boom):
        with Profile(str(db_path), cache_mode="auto") as prof:
            assert prof.db is not None, (
                "a failed cache build dropped the profile to raw sqlite3; the enriched "
                "kernels view is gone and tensor_core_usage cannot run"
            )
            prof.db.execute("SELECT is_tc_eligible, uses_tc FROM kernels LIMIT 1").fetchall()


def test_open_profile_readonly_failed_cache_keeps_duckdb_kernels(
    minimal_nsys_db_path, tmp_path
):
    """chat / region_mfu must keep DuckDB when a cache build fails.

    Profile already recovers via open_direct_sqlite. open_profile_readonly used
    to drop straight to raw sqlite3 on the same failure, which loses the
    enriched kernels view (``no such table: kernels``) and makes
    tensor_core_usage fail outright. Force the cache open to raise, then
    assert the handle is DuckDB and that kernels resolves.
    """
    import sqlite3

    import duckdb

    from nsys_ai.ai.backend.profile_db_tool import open_profile_readonly

    db_path = _copy_fixture(minimal_nsys_db_path, tmp_path / "readonly_buildfail.sqlite")

    def _boom(_path):
        raise RuntimeError("simulated build failure")

    with mock.patch("nsys_ai.parquet_cache.open_cached_db", side_effect=_boom):
        conn = open_profile_readonly(str(db_path))
    try:
        assert not isinstance(conn, sqlite3.Connection), (
            "open_profile_readonly fell to raw sqlite3 after a cache build failure"
        )
        assert isinstance(conn, duckdb.DuckDBPyConnection)
        conn.execute("SELECT is_tc_eligible, uses_tc FROM kernels LIMIT 1").fetchall()
    finally:
        conn.close()


def test_skill_run_open_failed_cache_keeps_duckdb_kernels(minimal_nsys_db_path, tmp_path):
    """skill run uses the same three-tier chain as Profile.

    Mirrors the handler's open: open_with_direct_fallback(path, open_auto_db).
    A test that only checked 'no exception' would have passed before the fix.
    """
    import sqlite3

    import duckdb

    from nsys_ai.parquet_cache import open_auto_db, open_with_direct_fallback

    db_path = _copy_fixture(minimal_nsys_db_path, tmp_path / "skillrun_buildfail.sqlite")

    def _boom(_path):
        raise RuntimeError("simulated build failure")

    with mock.patch("nsys_ai.parquet_cache.open_cached_db", side_effect=_boom):
        conn, err = open_with_direct_fallback(str(db_path), open_auto_db)
    try:
        assert err is not None
        assert conn is not None, "skill-run open path returned None (raw sqlite3 tier)"
        assert not isinstance(conn, sqlite3.Connection)
        assert isinstance(conn, duckdb.DuckDBPyConnection)
        conn.execute("SELECT is_tc_eligible, uses_tc FROM kernels LIMIT 1").fetchall()
    finally:
        if conn is not None:
            conn.close()


def test_cache_mode_auto_env_override_direct(minimal_nsys_db_path, tmp_path, monkeypatch):
    """The escape hatch for the entry points that expose no cache flag."""
    db_path = _copy_fixture(minimal_nsys_db_path, tmp_path / "auto_env.sqlite")
    monkeypatch.setenv("NSYS_AI_CACHE_MODE", "direct")

    with Profile(str(db_path), cache_mode="auto") as prof:
        res = prof.db.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()
        assert res[0] >= 0

    assert not _cache_dir_for(str(db_path)).exists()


def test_cache_mode_auto_env_override_parquet_beats_affordability(
    minimal_nsys_db_path, tmp_path, monkeypatch
):
    """`=parquet` is the override in the other direction: build anyway.

    Pinned separately from `=direct` because it is the branch that has to win
    against the affordability check, not just against the default.
    """
    db_path = _copy_fixture(minimal_nsys_db_path, tmp_path / "auto_env_parquet.sqlite")
    monkeypatch.setenv("NSYS_AI_CACHE_MODE", "parquet")
    usage = mock.Mock(total=10**9, used=10**9, free=1024)

    with mock.patch("nsys_ai.parquet_cache.shutil.disk_usage", return_value=usage):
        with Profile(str(db_path), cache_mode="auto") as prof:
            res = prof.db.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()
            assert res[0] >= 0

    assert _cache_dir_for(str(db_path)).exists(), (
        "NSYS_AI_CACHE_MODE=parquet did not override the affordability check"
    )


def test_cache_mode_auto_env_override_garbage_warns_and_uses_the_default(
    minimal_nsys_db_path, tmp_path, monkeypatch, caplog
):
    """A typo must be loud and harmless, not silently read as `direct`."""
    db_path = _copy_fixture(minimal_nsys_db_path, tmp_path / "auto_env_junk.sqlite")
    monkeypatch.setenv("NSYS_AI_CACHE_MODE", "sqlite")

    with caplog.at_level("WARNING", logger="nsys_ai.parquet_cache"):
        with Profile(str(db_path), cache_mode="auto"):
            pass

    assert _cache_dir_for(str(db_path)).exists(), "an unrecognised value silently disabled the cache"
    assert any("Ignoring NSYS_AI_CACHE_MODE" in r.getMessage() for r in caplog.records), (
        "an unrecognised NSYS_AI_CACHE_MODE was ignored without saying so"
    )


def test_skill_run_honours_the_cache_mode_override(minimal_nsys_db_path, tmp_path):
    """`skill run` must obey the variable its own build banner advertises.

    It is the one command that opens a connection without building a Profile,
    so it used to call open_cached_db directly and build regardless — while
    _build_banner, printed from inside that build, told the user to set
    NSYS_AI_CACHE_MODE=direct to avoid it. Reproduced before the fix: the
    variable was exported, the banner printed, the cache was built anyway.
    """
    db_path = _copy_fixture(minimal_nsys_db_path, tmp_path / "skillrun_env.sqlite")
    env = {**os.environ, "NSYS_AI_CACHE_MODE": "direct"}

    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "skill", "run", "top_kernels", str(db_path)],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert not _cache_dir_for(str(db_path)).exists(), (
        "skill run built a cache with NSYS_AI_CACHE_MODE=direct set — the banner's "
        "advice is wrong on the subcommand that prints it"
    )


def test_open_profile_readonly_honours_the_cache_mode_override(
    minimal_nsys_db_path, tmp_path, monkeypatch
):
    """The chat / region_mfu entry point obeys the override too.

    `open_profile_readonly` is the other place that opens a connection with no
    Profile involved. It called `open_cached_db` directly, so the one escape
    hatch those entry points expose — they take no cache flag at all — did
    nothing there. Switching it to `open_auto_db` is a behaviour change, and
    this is what goes red if anyone switches it back: mutating that one call
    site makes this the only failing test across the profile-mode, tools,
    threading and trajectory suites.
    """
    from nsys_ai.ai.backend.profile_db_tool import open_profile_readonly

    db_path = _copy_fixture(minimal_nsys_db_path, tmp_path / "readonly_env.sqlite")
    monkeypatch.setenv("NSYS_AI_CACHE_MODE", "direct")

    conn = open_profile_readonly(str(db_path))
    try:
        # Query something, so this cannot pass merely by failing to open.
        res = conn.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()
        assert res[0] >= 0
    finally:
        conn.close()

    assert not _cache_dir_for(str(db_path)).exists(), (
        "open_profile_readonly built a cache with NSYS_AI_CACHE_MODE=direct set"
    )


def test_open_profile_readonly_honours_sqlite_ingest_policy(
    minimal_nsys_db_path, tmp_path, monkeypatch
):
    from nsys_ai.ai.backend.profile_db_tool import open_profile_readonly

    db_path = _copy_fixture(minimal_nsys_db_path, tmp_path / "readonly_ingest.sqlite")
    monkeypatch.setenv("NSYS_AI_INGEST", "sqlite")
    with mock.patch(
        "nsys_ai.parquet_cache.open_auto_db",
        side_effect=AssertionError("sqlite ingest policy must not build a cache"),
    ):
        conn = open_profile_readonly(str(db_path))
    try:
        assert conn.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()[0] >= 0
    finally:
        conn.close()
    assert not _cache_dir_for(str(db_path)).exists()


def test_a_declined_cache_says_so_at_warning(minimal_nsys_db_path, tmp_path, caplog):
    """A cache declined for disk reasons must be announced at WARNING, not INFO.

    The level *is* the feature here, which is why this test pins the level and
    not merely the text. Nothing under src/nsys_ai/ ever configures logging —
    no basicConfig, no dictConfig, no addHandler anywhere — so the only sink a
    record can reach is `logging.lastResort`, a StreamHandler fixed at WARNING.
    An INFO record is therefore dropped on the floor, and the user silently
    pays several times slower queries with nothing on stderr to say why.
    Measured on a profile in a chmod 500 directory: `skill run top_kernels`
    wrote 2059 bytes to stdout and 0 to stderr at INFO, 220 bytes at WARNING.

    `caplog.at_level("WARNING", ...)` sets the logger's level, so an INFO call
    is filtered out and the assertion below fails — which is exactly the
    revert-detection this needs.
    """
    db_path = _copy_fixture(minimal_nsys_db_path, tmp_path / "declined_warn.sqlite")
    usage = mock.Mock(total=10**9, used=10**9, free=1024)

    with caplog.at_level("WARNING", logger="nsys_ai.parquet_cache"):
        with mock.patch("nsys_ai.parquet_cache.shutil.disk_usage", return_value=usage):
            with Profile(str(db_path), cache_mode="auto") as prof:
                assert prof.db is not None

    assert any("Not building an analysis cache" in r.getMessage() for r in caplog.records), (
        "the declined-cache notice did not reach WARNING, so lastResort drops it and a "
        "silently degraded run produces nothing on stderr"
    )


def test_the_batch_audit_script_opens_the_same_way_skill_run_does():
    """scripts/batch_audit_skills.py claims parity with `nsys-ai skill run`.

    It exists to reproduce what that subcommand does across every registry
    skill, and its module docstring says so. When `skill run` moved to
    `open_auto_db` the script kept calling `open_cached_db`, so it silently
    ignored NSYS_AI_CACHE_MODE and built caches on profiles the CLI would have
    read in place — an audit of a path the CLI no longer takes, under a
    docstring asserting the opposite. Both now share
    `open_with_direct_fallback` so a failed build keeps DuckDB too.

    This is the fourth time in this file's neighbourhood that prose one level
    up from a change survived it and became false (#321, #325, #317). A grep
    over call sites in src/ does not catch it, because the script is not in
    src/. Pinning the two together in a test does.
    """
    root = Path(__file__).resolve().parent.parent
    script = (root / "scripts" / "batch_audit_skills.py").read_text()
    handlers = (root / "src" / "nsys_ai" / "cli" / "handlers.py").read_text()

    # What `skill run` actually calls, read from the handler rather than
    # assumed, so this test tracks the CLI instead of restating it.
    assert "open_with_direct_fallback(args.profile, primary)" in handlers, (
        "`skill run` no longer uses open_with_direct_fallback — update this "
        "test and the batch audit script together, they are supposed to agree"
    )
    assert "open_auto_db" in handlers and "open_direct_sqlite if no_cache else open_auto_db" in handlers

    assert "open_with_direct_fallback(profile_path, open_auto_db)" in script, (
        "batch_audit_skills.py does not open the way `skill run` does; its "
        "docstring claims it does, and NSYS_AI_CACHE_MODE does not reach it"
    )
    # The docstrings are the half that goes stale silently, so assert on them too.
    assert "open_cached_db()" not in script, (
        "batch_audit_skills.py still advertises the open_cached_db path it no longer takes"
    )
