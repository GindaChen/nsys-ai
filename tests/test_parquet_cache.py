"""Tests for parquet_cache module — DuckDB + Parquet cache lifecycle."""

import json
import os
import sqlite3
from pathlib import Path

import pytest

from nsys_ai import parquet_cache

# Minimal schema reused by tests that need to seed custom NVTX rows. Mirrors
# the production CUPTI/NVTX layout; intentionally narrow — just enough for
# build_cache() to produce kernels.parquet, runtime.parquet, nvtx.parquet,
# and nvtx_high.parquet.
_TEST_SQLITE_SCHEMA = """
    CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
    CREATE TABLE TARGET_INFO_GPU (
        id INTEGER PRIMARY KEY, name TEXT, busLocation TEXT DEFAULT '',
        totalMemory INTEGER DEFAULT 0, smCount INTEGER DEFAULT 0,
        chipName TEXT DEFAULT '', memoryBandwidth INTEGER DEFAULT 0
    );
    CREATE TABLE TARGET_INFO_CUDA_DEVICE (
        gpuId INTEGER, cudaId INTEGER, pid INTEGER DEFAULT 0,
        uuid TEXT DEFAULT '', numMultiprocessors INTEGER DEFAULT 0
    );
    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
        globalPid INTEGER DEFAULT 0, deviceId INTEGER DEFAULT 0,
        streamId INTEGER DEFAULT 0, correlationId INTEGER DEFAULT 0,
        start INTEGER NOT NULL, end INTEGER NOT NULL,
        shortName INTEGER NOT NULL, demangledName INTEGER DEFAULT 0,
        gridX INTEGER DEFAULT 1, gridY INTEGER DEFAULT 1, gridZ INTEGER DEFAULT 1,
        blockX INTEGER DEFAULT 1, blockY INTEGER DEFAULT 1, blockZ INTEGER DEFAULT 1
    );
    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
        globalTid INTEGER DEFAULT 0, correlationId INTEGER DEFAULT 0,
        start INTEGER NOT NULL, end INTEGER NOT NULL, nameId INTEGER DEFAULT 0
    );
    CREATE TABLE NVTX_EVENTS (
        globalTid INTEGER DEFAULT 0, start INTEGER NOT NULL,
        end INTEGER DEFAULT -1, text TEXT DEFAULT '',
        eventType INTEGER DEFAULT 59, rangeId INTEGER DEFAULT 0,
        textId INTEGER DEFAULT NULL
    );
"""

_TEST_SQLITE_SEED_FIXED = """
    INSERT INTO StringIds VALUES (1, 'gemm_kernel');
    INSERT INTO TARGET_INFO_GPU VALUES
        (0, 'Test', '', 8589934592, 108, 'TestChip', 0);
    INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (0, 0, 100, '', 108);
    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
        (100, 0, 7, 1, 1000000, 2000000, 1, 1, 1, 1, 1, 1, 1, 1);
    INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
        (100, 1, 900000, 1000000, 0);
"""


def _make_nsys_sqlite(tmp_path: Path, filename: str, nvtx_rows: list[tuple]) -> Path:
    """Create a minimal nsys-style sqlite for nvtx_high-related tests.

    Includes one kernel + matching runtime row, plus caller-supplied NVTX
    rows. Returns the file path. ``nvtx_rows`` items are
    ``(globalTid, start, end, text, eventType, rangeId)`` tuples — keep
    ranges loose enough that the single kernel lands inside them when the
    test cares about attribution.
    """
    db_path = tmp_path / filename
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_TEST_SQLITE_SCHEMA)
        conn.executescript(_TEST_SQLITE_SEED_FIXED)
        conn.executemany(
            "INSERT INTO NVTX_EVENTS (globalTid, start, end, text, eventType, rangeId) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            nvtx_rows,
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


class TestCacheValidation:
    """Test cache validity checks."""

    def test_no_cache_dir(self, tmp_path):
        fake_sqlite = str(tmp_path / "profile.sqlite")
        open(fake_sqlite, "w").close()
        assert parquet_cache.is_cache_valid(fake_sqlite) is False

    def test_empty_cache_dir(self, tmp_path):
        fake_sqlite = str(tmp_path / "profile.sqlite")
        open(fake_sqlite, "w").close()
        (tmp_path / "profile.nsys-cache").mkdir()
        assert parquet_cache.is_cache_valid(fake_sqlite) is False

    def test_wrong_version(self, tmp_path):
        fake_sqlite = str(tmp_path / "profile.sqlite")
        open(fake_sqlite, "w").close()
        cache_dir = tmp_path / "profile.nsys-cache"
        cache_dir.mkdir()
        (cache_dir / ".cache_version").write_text(json.dumps({"version": -1}))
        (cache_dir / "kernels.parquet").write_text("dummy")
        assert parquet_cache.is_cache_valid(fake_sqlite) is False


class TestBuildAndOpen:
    """End-to-end: build cache from test SQLite, then open."""

    def test_build_cache_creates_parquet_files(self, minimal_nsys_db_path):
        """Building a cache should create Parquet files in .nsys-cache/."""
        cache_dir = parquet_cache.build_cache(minimal_nsys_db_path)

        assert cache_dir.exists()
        assert (cache_dir / "kernels.parquet").exists()
        assert (cache_dir / "nvtx.parquet").exists()
        assert (cache_dir / "runtime.parquet").exists()
        assert (cache_dir / ".cache_version").exists()

        # Version stamp
        meta = json.loads((cache_dir / ".cache_version").read_text())
        assert meta["version"] == parquet_cache._CACHE_VERSION

    def test_cache_is_valid_after_build(self, minimal_nsys_db_path):
        """After building, is_cache_valid should return True."""
        parquet_cache.build_cache(minimal_nsys_db_path)
        assert parquet_cache.is_cache_valid(minimal_nsys_db_path) is True

    def test_open_cached_db_returns_duckdb(self, minimal_nsys_db_path):
        """open_cached_db should return a DuckDB connection with views."""
        import duckdb

        db = parquet_cache.open_cached_db(minimal_nsys_db_path)
        assert isinstance(db, duckdb.DuckDBPyConnection)

        # Should be able to query kernels
        result = db.execute("SELECT COUNT(*) FROM kernels").fetchone()
        assert result[0] > 0

        db.close()

    def test_kernel_names_resolved(self, minimal_nsys_db_path):
        """Kernels parquet should have pre-joined name column."""
        db = parquet_cache.open_cached_db(minimal_nsys_db_path)
        rows = db.execute("SELECT name FROM kernels WHERE name IS NOT NULL").fetchall()
        assert len(rows) > 0
        # At least one kernel should have a resolved name
        names = [r[0] for r in rows]
        assert any(n for n in names if n)
        db.close()

    def test_nvtx_kernel_map_generated(self, minimal_nsys_db_path):
        """nvtx_kernel_map.parquet should be generated by Tier 2 sort-merge."""
        cache_dir = parquet_cache.build_cache(minimal_nsys_db_path)
        map_file = cache_dir / "nvtx_kernel_map.parquet"
        # It's OK if it doesn't exist (if no NVTX→kernel attribution found)
        # but if it does, it should be queryable
        if map_file.exists():
            dict_file = cache_dir / "nvtx_path_dict.parquet"
            assert dict_file.is_file()
            db = parquet_cache.open_cached_db(minimal_nsys_db_path)
            result = db.execute(
                """
                SELECT m.path_id, d.nvtx_path
                FROM nvtx_kernel_map m
                JOIN nvtx_path_dict d USING (path_id)
                LIMIT 5
                """
            )
            joined = result.fetchall()
            assert isinstance(joined, list)
            assert [col[0] for col in result.description] == ["path_id", "nvtx_path"]
            if joined:
                assert len(joined[0]) == 2
                assert isinstance(joined[0][0], int)
                assert joined[0][1] is None or isinstance(joined[0][1], str)
            db.close()

    def test_nvtx_high_filters_aten_events(self, tmp_path):
        """nvtx_high.parquet should exclude aten::*/cudaLaunch%/cudaMemcpy%."""
        import duckdb

        nvtx_rows = [
            # (globalTid, start, end, text, eventType, rangeId)
            (100, 100, 5000, "stage::DenoisingStage", 59, 0),   # keep
            (100, 200, 4000, "FlashAttnFunc", 59, 1),           # keep
            (100, 300, 3500, "nccl:all_to_all", 59, 2),         # keep
            (100, 400, 3000, "aten::linear", 59, 3),            # DROP
            (100, 500, 2500, "aten::layer_norm", 59, 4),        # DROP
            (100, 600, 2400, "cudaLaunchKernel", 59, 5),        # DROP
            (100, 700, 2300, "cudaMemcpyAsync", 59, 6),         # DROP
        ]
        db_path = _make_nsys_sqlite(tmp_path, "phase4_nvtx_high.sqlite", nvtx_rows)
        cache_dir = parquet_cache.build_cache(str(db_path))
        nvtx_high = cache_dir / "nvtx_high.parquet"
        assert nvtx_high.is_file(), "nvtx_high.parquet should be created"

        db = duckdb.connect()
        try:
            rows = db.execute(
                f"SELECT text FROM read_parquet('{nvtx_high.as_posix()}') ORDER BY text"
            ).fetchall()
        finally:
            db.close()

        assert sorted(r[0] for r in rows) == sorted(
            ["FlashAttnFunc", "nccl:all_to_all", "stage::DenoisingStage"]
        ), f"unexpected nvtx_high rows: {[r[0] for r in rows]}"

    def test_nvtx_kernel_map_uses_full_nvtx_not_high(self, tmp_path):
        """Regression: nvtx_kernel_map.parquet must include kernels whose only
        enclosing NVTX ranges are aten::* (e.g. emit_nvtx-style traces).

        If _build_nvtx_kernel_map() sourced the IEJoin from nvtx_high.parquet
        instead of full nvtx.parquet, such kernels would silently disappear
        from the precomputed map and the fast path in nvtx_layer_breakdown
        would return zero attribution.
        """
        import duckdb

        # Single kernel inside two aten:: enclosing ranges. Both ranges match
        # the nvtx_high exclusion list — so the precomputed map MUST be built
        # from full nvtx.parquet to retain attribution.
        nvtx_rows = [
            (100, 100_000, 5_000_000, "aten::linear", 59, 0),
            (100, 200_000, 4_000_000, "aten::layer_norm", 59, 1),
        ]
        db_path = _make_nsys_sqlite(tmp_path, "kmap_aten_only.sqlite", nvtx_rows)
        cache_dir = parquet_cache.build_cache(str(db_path))

        kmap = cache_dir / "nvtx_kernel_map.parquet"
        if not kmap.is_file():
            pytest.skip("nvtx_kernel_map.parquet not built on this profile")

        check = duckdb.connect()
        try:
            n_rows = check.execute(
                f"SELECT COUNT(*) FROM read_parquet('{kmap.as_posix()}')"
            ).fetchone()[0]
            sample = check.execute(
                f"SELECT kernel_name, nvtx_text "
                f"FROM read_parquet('{kmap.as_posix()}') LIMIT 5"
            ).fetchall()
        finally:
            check.close()

        assert n_rows >= 1, (
            "nvtx_kernel_map.parquet is empty on aten::-only trace; "
            "_build_nvtx_kernel_map() must source from full nvtx.parquet, "
            "not the filtered nvtx_high.parquet"
        )
        # The leaf attribution should be one of the aten:: ranges we seeded.
        leaves = [r[1] for r in sample]
        assert any("aten::" in (text or "") for text in leaves), (
            f"expected an aten:: leaf in nvtx_kernel_map, got {leaves}"
        )

    def test_nvtx_high_empty_falls_back_in_layer_breakdown(self, tmp_path):
        """When nvtx_high.parquet exists but is empty (profile is all aten::*),
        nvtx_layer_breakdown should still return attribution by reading the
        full nvtx view instead of silently emitting [].
        """
        import duckdb

        from nsys_ai.skills.registry import get_skill

        # Every NVTX row matches an exclusion prefix → nvtx_high will be empty
        # but full nvtx will still enclose the kernel (correlationId 1 lives
        # in [1_000_000, 2_000_000] and both ranges below cover that).
        nvtx_rows = [
            (100, 100_000, 5_000_000, "aten::linear", 59, 0),
            (100, 200_000, 4_000_000, "aten::layer_norm", 59, 1),
        ]
        db_path = _make_nsys_sqlite(tmp_path, "all_aten.sqlite", nvtx_rows)
        cache_dir = parquet_cache.build_cache(str(db_path))
        nvtx_high = cache_dir / "nvtx_high.parquet"
        assert nvtx_high.is_file(), "nvtx_high.parquet should still be created"

        check = duckdb.connect()
        try:
            n_high = check.execute(
                f"SELECT COUNT(*) FROM read_parquet('{nvtx_high.as_posix()}')"
            ).fetchone()[0]
        finally:
            check.close()
        assert n_high == 0, f"expected empty nvtx_high, got {n_high} rows"

        # Fallback path: skill must still attribute the kernel to aten:: ranges.
        db = parquet_cache.open_cached_db(str(db_path))
        try:
            rows = get_skill("nvtx_layer_breakdown").execute(db, limit=10)
        finally:
            db.close()
        assert rows, "expected fallback to full nvtx to produce attribution"
        data = [r for r in rows if not r.get("_detection_meta")]
        assert any(
            "aten::" in (r.get("nvtx_path") or r.get("leaf_text") or "")
            for r in data
        ), f"expected an aten:: leaf in fallback result, got {data}"

    def test_invalidate_cache(self, minimal_nsys_db_path):
        """invalidate_cache should remove the cache directory."""
        parquet_cache.build_cache(minimal_nsys_db_path)
        assert parquet_cache.is_cache_valid(minimal_nsys_db_path) is True

        parquet_cache.invalidate_cache(minimal_nsys_db_path)
        assert parquet_cache.is_cache_valid(minimal_nsys_db_path) is False

    def test_rebuild_on_stale(self, minimal_nsys_db_path):
        """If SQLite is newer than cache, rebuild automatically."""
        parquet_cache.build_cache(minimal_nsys_db_path)
        assert parquet_cache.is_cache_valid(minimal_nsys_db_path) is True

        # Touch the SQLite to make it newer; sleep long enough to exceed
        # coarse filesystem mtime granularity (often 1s).
        import time

        time.sleep(1.1)
        os.utime(minimal_nsys_db_path, None)

        assert parquet_cache.is_cache_valid(minimal_nsys_db_path) is False

        # open_cached_db should auto-rebuild
        db = parquet_cache.open_cached_db(minimal_nsys_db_path)
        assert db is not None
        db.close()
        assert parquet_cache.is_cache_valid(minimal_nsys_db_path) is True


class TestUnrecognisedKernelTable:
    """A kernel table the builder cannot address must abort the build, not
    publish a cache without kernels.parquet that keeps on validating."""

    def test_unrecognised_kernel_table_falls_back_instead_of_poisoning_cache(
        self, minimal_nsys_db_path
    ):
        """A non-`_V` kernel suffix must leave no cache and open via sqlite3."""
        from nsys_ai.profile import Profile

        conn = sqlite3.connect(minimal_nsys_db_path)
        conn.execute(
            "ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL RENAME TO CUPTI_ACTIVITY_KIND_KERNEL_X1"
        )
        conn.commit()
        conn.close()

        # Twice: the second open proves the first left nothing poisoned behind.
        for _ in range(2):
            prof = Profile(minimal_nsys_db_path, cache_mode="auto")
            try:
                assert prof.db is None  # sqlite3 fallback engaged
                assert prof.schema.kernel_table == "CUPTI_ACTIVITY_KIND_KERNEL_X1"
                assert prof.meta.kernel_count > 0
            finally:
                prof.close()

        # Assert on the cache dir + validity, not on a listing of the parent:
        # `_build_lock` leaves a <name>.nsys-cache.build.lock file behind on
        # every path, which is pre-existing and does not affect validity.
        cache_dir = Path(minimal_nsys_db_path).with_suffix(".nsys-cache")
        assert not cache_dir.exists()
        assert parquet_cache.is_cache_valid(minimal_nsys_db_path) is False

    def test_unrecognised_kernel_table_raises_runtimeerror_not_schemaerror(
        self, minimal_nsys_db_path
    ):
        """The exception type is load-bearing and must stay a RuntimeError.

        Every caller that falls back to direct SQLite keys on
        ``(duckdb.Error, RuntimeError, OSError)`` — ``Profile.__init__`` and the
        skill-run handler both do. ``SchemaError`` does not inherit from
        ``RuntimeError``, so raising one here would turn a working degraded read
        into a hard "Error [SCHEMA_ERROR]" for the user. The Profile-level test
        above cannot see this: ``Profile.__init__`` catches bare ``Exception``
        around the cache attempt, so it falls back either way.
        """
        from nsys_ai.exceptions import SchemaError

        conn = sqlite3.connect(minimal_nsys_db_path)
        conn.execute(
            "ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL RENAME TO CUPTI_ACTIVITY_KIND_KERNEL_X1"
        )
        conn.commit()
        conn.close()

        with pytest.raises(RuntimeError) as excinfo:
            parquet_cache.build_cache(minimal_nsys_db_path)
        assert "CUPTI_ACTIVITY_KIND_KERNEL_X1" in str(excinfo.value)
        # Stated explicitly: SchemaError descends from Exception, not from
        # RuntimeError, so it would slip past every fallback except-clause.
        assert not issubclass(SchemaError, RuntimeError)

    def test_profile_without_any_kernel_table_still_caches(self, minimal_nsys_db_path):
        """Guard must not fire on a capture that simply has no kernel data.

        Nsight creates tables lazily, so "no kernel table at all" is a legitimate
        state that must keep its existing behaviour (cache builds, SchemaError
        surfaces from NsightSchema).

        The ENUM_ table is not decoration. Every real capture carries
        ``ENUM_CUDA_KERNEL_LAUNCH_TYPE`` — h100_2gpu_1s.sqlite and
        mfu_2gpu_before.sqlite both do — and its name contains "KERNEL". Without
        the ``ENUM_`` exclusion in ``_kernel_like_tables`` the guard would fire on
        it and strip the cache from every genuinely kernel-less real profile. The
        minimal conftest schema has no ENUM_ table, so this test creates one.
        """
        conn = sqlite3.connect(minimal_nsys_db_path)
        conn.execute("CREATE TABLE ENUM_CUDA_KERNEL_LAUNCH_TYPE (id INTEGER, label TEXT)")
        conn.execute("DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL")
        conn.commit()
        conn.close()

        cache_dir = parquet_cache.build_cache(minimal_nsys_db_path)
        assert cache_dir.exists()
        assert not (cache_dir / "kernels.parquet").exists()
        assert parquet_cache.is_cache_valid(minimal_nsys_db_path) is True

    def test_enum_tables_are_not_kernel_activity_tables(self):
        """Unit-level companion: `_kernel_like_tables` must skip ENUM_ tables.

        Mirrors ``NsightSchema._detect_kernel_table``, which applies the same
        exclusion. Pinned separately from the build-level test so the intent
        survives even if the fixture schema changes.
        """
        assert parquet_cache._kernel_like_tables({"ENUM_CUDA_KERNEL_LAUNCH_TYPE"}) == []
        assert parquet_cache._kernel_like_tables(
            {"ENUM_CUDA_KERNEL_LAUNCH_TYPE", "CUPTI_ACTIVITY_KIND_KERNEL_X1"}
        ) == ["CUPTI_ACTIVITY_KIND_KERNEL_X1"]


class TestDamagedInput:
    """What happens when the bytes on disk are wrong.

    The sibling class above covers "the cache was never published". This one
    covers the two cases after that: a cache that *was* published and then went
    bad on disk, and a profile that is itself truncated. Both are ordinary in
    practice — an interrupted build, a killed capture, a full filesystem, an
    rsync that stopped halfway — and only three or four files in the whole suite
    touched malformed input before this.

    Every assertion here is a behaviour that was measured first. Nothing in this
    class asserts what the code *should* do; where the current behaviour is
    wrong, it is pinned as wrong and said so.
    """

    def test_a_corrupt_kernels_parquet_falls_back_instead_of_failing(
        self, minimal_nsys_db_path
    ):
        """A damaged cache file must degrade to sqlite3, not take the profile down.

        Silent otherwise in the loudest possible way: the cache is an
        optimisation the user never asked for, so a byte-level fault inside it
        surfacing as an error on a perfectly good capture is the worst outcome
        available. Nothing else in the suite writes garbage into a published
        cache, so a change that let ``duckdb.Error`` escape
        ``Profile.__init__``'s fallback would go unnoticed.
        """
        from nsys_ai.profile import Profile
        from nsys_ai.skills.registry import get_skill

        with Profile(minimal_nsys_db_path, cache_mode="auto") as prof:
            if prof.db is None:
                pytest.skip("requires duckdb")
        cache_dir = Path(minimal_nsys_db_path).with_suffix(".nsys-cache")
        (cache_dir / "kernels.parquet").write_bytes(b"NOTAPARQUET" * 100)

        with Profile(minimal_nsys_db_path, cache_mode="auto") as prof:
            assert prof.db is None, "a corrupt cache file was accepted as a cache"
            rows = get_skill("top_kernels").execute(prof.query_conn(), limit=3)
        assert rows, "falling back to sqlite3 lost the kernel data"

    def test_a_corrupt_cache_is_never_repaired(self, minimal_nsys_db_path):
        """Pinned defect, not a virtue: the damage is permanent and invisible.

        ``is_cache_valid()`` checks for the cache directory and its manifest, not
        for readable Parquet, so a corrupt cache stays "valid" forever. Every
        subsequent open pays the fallback and gets the degraded answer — the
        sqlite3 path returns fewer columns than the cached one (``top_kernels``
        loses ``tc_eligible``/``uses_tc``, ``nvtx_kernel_map`` loses demangled
        names), so the user silently gets a thinner analysis on every run until
        someone deletes the directory by hand.

        This test exists so that fixing it — discard and rebuild on the first
        unreadable read — is a visible change to a stated behaviour rather than
        an accidental one. When that lands, this test should fail and be
        rewritten to assert the repair.
        """
        from nsys_ai.profile import Profile

        with Profile(minimal_nsys_db_path, cache_mode="auto") as prof:
            if prof.db is None:
                pytest.skip("requires duckdb")
        cache_dir = Path(minimal_nsys_db_path).with_suffix(".nsys-cache")
        damaged = cache_dir / "kernels.parquet"
        damaged.write_bytes(b"GARBAGE" * 50)
        damaged_size = damaged.stat().st_size

        for attempt in range(3):
            with Profile(minimal_nsys_db_path, cache_mode="auto") as prof:
                assert prof.db is None, f"open #{attempt + 1} unexpectedly used the cache"

        assert parquet_cache.is_cache_valid(minimal_nsys_db_path) is True, (
            "is_cache_valid now rejects a corrupt cache — good, but this test "
            "pins the old behaviour; rewrite it to assert the repair"
        )
        assert damaged.stat().st_size == damaged_size, (
            "the damaged file changed size — something is rebuilding now; "
            "rewrite this test to assert the repair"
        )

    def test_a_truncated_profile_raises_rather_than_answering(self, tmp_path):
        """Half a capture must fail loudly, not analyse whatever survived.

        The loud-fail path from #305 seen from the other end. Nothing about the
        truncation is detected as truncation: ``sqlite3.connect`` succeeds
        (connecting is lazy), the first read of ``sqlite_master`` raises
        ``DatabaseError: database disk image is malformed``, and every layer
        above swallows that on its way to concluding there is no kernel table.
        The user gets a ``SchemaError`` — the right *outcome*, reached for
        reasons the message gets wrong.

        Both halves are asserted, and the second is pinned deliberately. The
        message currently blames the capture ("may have been captured without
        CUDA kernel tracing") for a file that is simply cut in half, which is a
        wart worth fixing; pinning it means fixing it is a visible change. And
        without the first half, a change to kernel-table detection could turn a
        truncated profile into an empty-but-successful analysis, which reads to
        a user exactly like a profile with no GPU work.
        """
        from nsys_ai.exceptions import SchemaError
        from nsys_ai.profile import Profile

        source = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"
        truncated = tmp_path / "truncated.sqlite"
        data = source.read_bytes()
        truncated.write_bytes(data[: len(data) // 2])
        del data

        # Control: connecting is lazy and succeeds, so nothing before the first
        # read can tell this file apart from a healthy one.
        control = sqlite3.connect(str(truncated))
        try:
            with pytest.raises(sqlite3.DatabaseError):
                control.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        finally:
            control.close()

        with pytest.raises(SchemaError) as excinfo:
            with Profile(str(truncated)) as prof:
                prof.query_conn()
        assert "no suitable KERNEL table found" in str(excinfo.value)

    def test_an_unreadable_map_source_aborts_the_build_and_a_missing_one_skips(
        self, tmp_path
    ):
        """The two cases ``_build_nvtx_kernel_map``'s docstring distinguishes.

        They are easy to confuse, and the docstring got them backwards until
        this test existed: it claimed an unreadable source was "logged and the
        map skipped". It is not. Both source readers are generator functions, so
        their ``db.execute`` runs lazily inside ``_sweep_nvtx_kernel_map``,
        where nothing catches it — the error leaves the builder and
        ``build_cache`` throws the half-built temp dir away.

        That is the wanted behaviour, not a defect: the map's sources are the
        same Parquets the rest of the cache reads, so one of them being garbage
        means the cache is unusable, and publishing it minus the map would hide
        that. Pinned here so that adding a ``try/except`` around the sweep —
        which would look like a tidy-up — is a visible change to a documented
        behaviour instead of a silent one.

        A *missing* source is the opposite and stays a logged skip; that branch
        is unreachable in a normal build, so this is its only exercise.
        """
        import shutil

        import duckdb

        src = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"
        profile = tmp_path / "p.sqlite"
        shutil.copy(src, profile)

        cache_dir = Path(parquet_cache.build_cache(str(profile)))
        map_path = cache_dir / "nvtx_kernel_map.parquet"
        assert map_path.is_file(), "control: the healthy build must produce a map"

        con = sqlite3.connect(str(profile))
        try:
            src_tables = {
                r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
        finally:
            con.close()

        db = duckdb.connect()
        try:
            nvtx = cache_dir / "nvtx.parquet"
            healthy = nvtx.read_bytes()
            map_path.unlink()

            nvtx.write_bytes(b"NOTAPARQUET" * 100)
            with pytest.raises(duckdb.Error):
                parquet_cache._build_nvtx_kernel_map(db, src_tables, cache_dir)
            assert not map_path.exists(), "a map was written from an unreadable source"

            nvtx.unlink()
            parquet_cache._build_nvtx_kernel_map(db, src_tables, cache_dir)
            assert not map_path.exists(), "the missing-source skip wrote a map anyway"

            # Control: the same call on the restored source does build one, so
            # the two assertions above are about the damage, not about the
            # arguments being wrong.
            nvtx.write_bytes(healthy)
            parquet_cache._build_nvtx_kernel_map(db, src_tables, cache_dir)
            assert map_path.is_file(), "the builder no longer works on healthy sources"
        finally:
            db.close()


class TestConcurrentBuild:
    """Regression: two terminals opening the same profile concurrently
    must NOT both run the full ETL.

    Before the build-lock landed, ``is_cache_valid()`` returned False
    for every concurrent caller until the first finished its atomic
    rename, so every caller ran its own ``_build_cache_into``. On a
    296MB profile that meant ~10s of wasted ETL per duplicate runner.
    """

    @pytest.mark.skipif(
        parquet_cache._fcntl is None,
        reason="build-lock degrades to no-op without POSIX fcntl; this assertion "
        "only holds on platforms where the lock is real.",
    )
    def test_concurrent_threads_only_build_once(self, minimal_nsys_db_path, monkeypatch):
        import threading
        import time

        # Start clean — make sure no prior test left a valid cache for
        # this fixture (the cache lives next to the .sqlite file).
        parquet_cache.invalidate_cache(minimal_nsys_db_path)
        assert parquet_cache.is_cache_valid(minimal_nsys_db_path) is False

        num_threads = 4
        # Barrier so every runner enters ``build_cache`` essentially
        # simultaneously — without this, on lightly-loaded systems the
        # first thread can finish before the others even start, hiding
        # any locking bug.
        barrier = threading.Barrier(num_threads)
        call_count = 0
        count_lock = threading.Lock()
        original_build_into = parquet_cache._build_cache_into

        def counting_build_into(sqlite_path: str, tmp_dir: Path) -> None:
            nonlocal call_count
            with count_lock:
                call_count += 1
            # Hold the lock long enough that all other threads are
            # guaranteed to be queued on flock before this one releases.
            # ETL on the minimal fixture is ~50ms; 0.5s sleep gives a
            # ~10× safety margin.
            time.sleep(0.5)
            original_build_into(sqlite_path, tmp_dir)

        monkeypatch.setattr(parquet_cache, "_build_cache_into", counting_build_into)

        # Track caches and exceptions separately so a barrier timeout
        # or any other error can't masquerade as "all paths succeeded".
        caches: list[Path] = []
        errors: list[BaseException] = []
        results_lock = threading.Lock()

        def runner() -> None:
            try:
                barrier.wait(timeout=10)
                cache = parquet_cache.build_cache(str(minimal_nsys_db_path))
            except BaseException as e:  # pragma: no cover - defensive
                with results_lock:
                    errors.append(e)
                return
            with results_lock:
                caches.append(cache)

        threads = [threading.Thread(target=runner) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # If any thread is still alive the lock has deadlocked.
        # Surface that immediately rather than letting a zombie thread
        # leak into later tests where it could rebuild the cache and
        # confuse unrelated assertions.
        stuck = [t.name for t in threads if t.is_alive()]
        assert not stuck, f"threads did not finish within 30s: {stuck}"

        assert errors == [], f"runners raised: {errors!r}"
        assert len(caches) == num_threads, (
            f"all {num_threads} runners must complete; got {len(caches)}"
        )
        assert call_count == 1, (
            f"_build_cache_into should run exactly once for {num_threads} "
            f"concurrent callers; got {call_count}"
        )
        # Every caller returns the same cache directory.
        assert len(set(caches)) == 1, f"all callers must agree on cache_dir; got {set(caches)}"
        assert parquet_cache.is_cache_valid(minimal_nsys_db_path) is True


class TestTensorCorePatterns:
    """Regression coverage for the TC eligibility / active regex patterns.

    The patterns are interpolated into DuckDB `regexp_matches()`; these cases
    guard against drift that would cause Flash Attention / CUTLASS tensor-op
    kernels to be silently mis-classified as FP32 fallback.
    """

    @staticmethod
    def _strip(pattern: str) -> str:
        # Stored as SQL-quoted literal, e.g. "'(gemm|...)'" — strip the outer quotes
        # so Python's re module sees a plain pattern.
        return pattern.strip("'")

    def test_eligible_pattern_covers_flash_attention(self):
        import re

        elig = self._strip(parquet_cache._TC_ELIGIBLE_PATTERN)
        for name in [
            "flash_fwd_splitkv_kernel",
            "flash_bwd_dq_dk_dv_loop_seqk_parallel",
            "ampere_bf16_s1688gemm_bf16_128x128x32",
            "cutlass_80_tensorop_bf16_s16816gemm_something",
            "sm80_xmma_gemm_bf16",
        ]:
            assert re.search(elig, name.lower()), f"{name!r} should be TC-eligible"

    def test_active_pattern_covers_cutlass_and_flash(self):
        import re

        active = self._strip(parquet_cache._TC_ACTIVE_PATTERN)
        for name in [
            "flash_fwd_splitkv_kernel",
            "flash_bwd_dq_dk_dv_loop_seqk_parallel",
            "cutlass_80_tensorop_bf16_s16816gemm_something",
            "ampere_bf16_s1688gemm_bf16_128x128x32",
            "some_kernel_with_16816_in_name",
        ]:
            assert re.search(active, name.lower()), f"{name!r} should be TC-active"

    def test_non_tc_kernels_not_matched(self):
        import re

        elig = self._strip(parquet_cache._TC_ELIGIBLE_PATTERN)
        active = self._strip(parquet_cache._TC_ACTIVE_PATTERN)
        for name in [
            "vectorized_elementwise_kernel",
            "reduce_kernel",
            "memset_kernel",
        ]:
            assert not re.search(elig, name.lower()), f"{name!r} should NOT be eligible"
            assert not re.search(active, name.lower()), f"{name!r} should NOT be active"

    def test_fp32_sgemm_is_eligible_but_not_tc_active(self):
        """Classic FP32 sgemm: TC-eligible (it's a gemm) but not TC-active."""
        import re

        elig = self._strip(parquet_cache._TC_ELIGIBLE_PATTERN)
        active = self._strip(parquet_cache._TC_ACTIVE_PATTERN)
        name = "ampere_sgemm_128x128_nn"
        assert re.search(elig, name.lower())
        assert not re.search(active, name.lower())


def test_the_map_is_built_by_the_sweep_and_carries_all_nine_columns(tmp_path):
    """NVTX attribution is a stack sweep, not a general inequality join.

    NVIDIA documents eventType 59 as a Push/Pop range maintaining an nvtxRange
    stack per thread, so ranges on a thread are strictly nested by construction.
    A containment IEJoin cannot exploit that: on a 3.5 GB capture it ran over
    twenty minutes at 1170% CPU to produce a three-million-row result, where the
    sweep takes 19 s. The whole cache build for that capture went from
    twenty-plus minutes to 60.6 s.

    Output was compared row-for-row against the IEJoin's on that capture:
    3,042,699 rows, zero differences either direction, all nine columns.

    Nine columns is the other half, and this is now the only guard on it. A
    second, unreachable builder used to exist alongside the sweep; unifying the
    two on one nine-column writer fixed the shape but not the values — it went
    on filling is_tc_eligible/uses_tc with zeroes where the sweep writes 296/296
    on this fixture, and since consumers probe those columns by *presence*
    (connection.cached_nvtx_map_has_embedded_tc), nine zeroed columns read as
    "TC data available" where seven had correctly fallen back to a kernels-join.
    That builder is gone, so nothing but this test stands between a future
    change and an all-zero map.
    """
    import shutil

    src = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"
    profile = tmp_path / "p.sqlite"
    shutil.copy(src, profile)

    from nsys_ai.profile import Profile

    with Profile(str(profile), cache_mode="parquet") as prof:
        assert prof.db is not None, "cache build did not produce a DuckDB connection"
        cols = [c[0] for c in prof.db.execute("DESCRIBE SELECT * FROM nvtx_kernel_map").fetchall()]
        rows, tc_eligible, tc_used = prof.db.execute(
            "SELECT count(*), sum(is_tc_eligible), sum(uses_tc) FROM nvtx_kernel_map"
        ).fetchone()

    assert cols == [
        "path_id",
        "nvtx_text",
        "nvtx_depth",
        "kernel_name",
        "k_start",
        "k_end",
        "k_dur_ns",
        "is_tc_eligible",
        "uses_tc",
    ], f"map schema drifted: {cols}"
    assert rows > 0, "the sweep attributed nothing"
    # Guards the seven-column regression: a map without TC flags reads as all
    # zeroes and pushes nvtx_layer_breakdown onto a path that double-counts.
    assert tc_eligible > 0, "is_tc_eligible is all zero — the TC flags were dropped"
    assert tc_used > 0, "uses_tc is all zero — the TC flags were dropped"


def test_the_path_dictionary_matches_the_map(tmp_path):
    """Every path_id in the map must resolve, or attribution silently vanishes
    on the join downstream."""
    import shutil

    src = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"
    profile = tmp_path / "p.sqlite"
    shutil.copy(src, profile)

    from nsys_ai.profile import Profile

    with Profile(str(profile), cache_mode="parquet") as prof:
        orphans = prof.db.execute(
            "SELECT count(*) FROM nvtx_kernel_map m "
            "LEFT JOIN nvtx_path_dict d USING (path_id) WHERE d.nvtx_path IS NULL"
        ).fetchone()[0]
    assert orphans == 0, f"{orphans} rows carry a path_id with no dictionary entry"
