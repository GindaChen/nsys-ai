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

    def test_nvtx_kernel_map_generated(self, minimal_nsys_db_path, monkeypatch):
        """nvtx_kernel_map.parquet should be generated by Tier 2 sort-merge.

        Asks for the eager build explicitly. The default is now to defer the map
        to first use, and this test is about the artifact's shape rather than
        about when it is produced — the deferral itself is pinned in
        ``tests/test_nvtx_kernel_map_ondemand.py``.
        """
        monkeypatch.setenv("NSYS_AI_ALWAYS_BUILD_NVTX_KERNEL_MAP", "1")
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

    def test_nvtx_kernel_map_uses_full_nvtx_not_high(self, tmp_path, monkeypatch):
        """Regression: nvtx_kernel_map.parquet must include kernels whose only
        enclosing NVTX ranges are aten::* (e.g. emit_nvtx-style traces).

        If _build_nvtx_kernel_map_from_parquet() sourced the IEJoin from nvtx_high.parquet
        instead of full nvtx.parquet, such kernels would silently disappear
        from the precomputed map and the fast path in nvtx_layer_breakdown
        would return zero attribution.

        Asks for the eager build, since the map is otherwise deferred to first
        use — and asserts the map exists rather than skipping when it does not.
        A ``pytest.skip`` here turned this whole test into a no-op the moment the
        deferral landed, which is precisely the coverage this file must not lose
        quietly: the seeded trace has attribution to find, so an absent map is a
        failure, not an unmet precondition.
        """
        import duckdb

        monkeypatch.setenv("NSYS_AI_ALWAYS_BUILD_NVTX_KERNEL_MAP", "1")

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
        assert kmap.is_file(), (
            "the eager build produced no nvtx_kernel_map.parquet for a trace that "
            "has aten::-only attribution to find"
        )

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
            "_build_nvtx_kernel_map_from_parquet() must source from full nvtx.parquet, "
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
        self, tmp_path, monkeypatch
    ):
        """The two cases ``_build_nvtx_kernel_map_from_parquet``'s docstring distinguishes.

        They are easy to confuse, and the docstring got them backwards until
        this test existed: it claimed an unreadable source was "logged and the
        map skipped". It is not. A damaged runtime.parquet raises out of the
        eager partition query, before anything is staged; a damaged
        nvtx.parquet raises later, from the generator whose ``db.execute`` runs
        on first advance inside ``_attribute_thread``. Both cases are checked
        below, and in neither does anything catch the error — it leaves the
        builder and ``build_cache`` throws the half-built temp dir away.

        That is the wanted behaviour, not a defect: the map's sources are the
        same Parquets the rest of the cache reads, so one of them being garbage
        means the cache is unusable, and publishing it minus the map would hide
        that. Pinned here so that adding a ``try/except`` around the sweep —
        which would look like a tidy-up — is a visible change to a documented
        behaviour instead of a silent one.

        A *missing* source is the opposite and stays a logged skip. It is the
        builder's own ``is_file()`` check that does that, not the ``src_tables``
        guard its former wrapper used to carry: probing that guard with a raise
        left the whole suite green, so it never fired here either, and both the
        guard and the wrapper have been removed.
        """
        import shutil

        import duckdb

        src = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"
        profile = tmp_path / "p.sqlite"
        shutil.copy(src, profile)

        # ``build_cache`` defers the map, so ask for it explicitly: this test is
        # about what ``_build_nvtx_kernel_map_from_parquet`` does with damaged
        # sources, not about when it is called.
        monkeypatch.setenv("NSYS_AI_ALWAYS_BUILD_NVTX_KERNEL_MAP", "1")
        cache_dir = Path(parquet_cache.build_cache(str(profile)))
        map_path = cache_dir / "nvtx_kernel_map.parquet"
        assert map_path.is_file(), "control: the healthy build must produce a map"

        db = duckdb.connect()
        try:
            nvtx = cache_dir / "nvtx.parquet"
            healthy = nvtx.read_bytes()
            map_path.unlink()

            nvtx.write_bytes(b"NOTAPARQUET" * 100)
            with pytest.raises(duckdb.Error):
                parquet_cache._build_nvtx_kernel_map_from_parquet(db, cache_dir)
            assert not map_path.exists(), "a map was written from an unreadable source"
            nvtx.write_bytes(healthy)

            # The other source, which is read eagerly to list the partitions
            # and so raises from a different statement than the one above.
            runtime = cache_dir / "runtime.parquet"
            healthy_runtime = runtime.read_bytes()
            runtime.write_bytes(b"NOTAPARQUET" * 100)
            with pytest.raises(duckdb.Error):
                parquet_cache._build_nvtx_kernel_map_from_parquet(db, cache_dir)
            assert not map_path.exists(), "a map was written from an unreadable source"
            runtime.write_bytes(healthy_runtime)

            nvtx.unlink()
            parquet_cache._build_nvtx_kernel_map_from_parquet(db, cache_dir)
            assert not map_path.exists(), "the missing-source skip wrote a map anyway"

            # Control: the same call on the restored source does build one, so
            # the two assertions above are about the damage, not about the
            # arguments being wrong.
            nvtx.write_bytes(healthy)
            parquet_cache._build_nvtx_kernel_map_from_parquet(db, cache_dir)
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

    The explicit ``ensure_nvtx_kernel_map`` is the deferral, not a workaround:
    ``build_cache`` does not produce the map any more, and querying it straight
    off the connection raises a Catalog Error. Every real consumer calls the
    accessor first for the same reason.
    """
    import shutil

    src = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"
    profile = tmp_path / "p.sqlite"
    shutil.copy(src, profile)

    from nsys_ai.profile import Profile

    with Profile(str(profile), cache_mode="parquet") as prof:
        assert prof.db is not None, "cache build did not produce a DuckDB connection"
        assert parquet_cache.ensure_nvtx_kernel_map(prof.db), "the map could not be built"
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
    on the join downstream.

    Both relations are built and published together by the on-demand path, and
    this checks that pair once published — every path_id in the map resolves in
    the dictionary.

    It does not observe a publish in flight. Nothing here interrupts one, so the
    dictionary-first ``os.replace`` ordering in
    ``materialize_cached_nvtx_kernel_map`` (a reader globbing mid-publish sees a
    dictionary with no map, never a map with no dictionary) is argued for in
    that function's comment and is not covered by this test or any other.
    """
    import shutil

    src = Path(__file__).resolve().parent / "fixtures" / "h100_2gpu_1s.sqlite"
    profile = tmp_path / "p.sqlite"
    shutil.copy(src, profile)

    from nsys_ai.profile import Profile

    with Profile(str(profile), cache_mode="parquet") as prof:
        assert parquet_cache.ensure_nvtx_kernel_map(prof.db), "the map could not be built"
        orphans = prof.db.execute(
            "SELECT count(*) FROM nvtx_kernel_map m "
            "LEFT JOIN nvtx_path_dict d USING (path_id) WHERE d.nvtx_path IS NULL"
        ).fetchone()[0]
    assert orphans == 0, f"{orphans} rows carry a path_id with no dictionary entry"


class TestStreamedSweep:
    """The cached map builder sweeps one thread of the capture at a time.

    Peak memory then tracks the batch rather than the profile: measured on a
    3.5 GB capture (15.9 M ranges, 3.1 M kernel-runtime rows) the build went from
    6.95 GB peak / 51.2 s to 3.31 GB / 37.3 s, producing 3,042,699 rows with zero
    differences in either direction across all nine columns and the path
    dictionary.

    These tests assert *contents*, not that a build finished, because both known
    ways of getting this wrong are silent. A hand-vectorised version of the same
    attribution was wrong on 15% of rows. Sharing one DuckDB handle between the
    two Arrow readers truncates the sweep with no exception at all — 968,394 rows
    instead of 3,042,699 on that capture, and 2 instead of 1,643 on the fixture
    below.
    """

    FIXTURES = ("h100_2gpu_1s.sqlite", "mfu_2gpu_before.sqlite")

    @staticmethod
    def _cache_for(tmp_path, fixture):
        """Copy a fixture out of tests/fixtures and build its cache.

        The copy is not hygiene theatre: opening a profile in place builds
        indices in the source database and grows the file, and these fixtures
        are checked in.
        """
        import shutil

        src = Path(__file__).resolve().parent / "fixtures" / fixture
        profile = tmp_path / fixture
        shutil.copy(src, profile)
        return Path(parquet_cache.build_cache(str(profile)))

    @staticmethod
    def _streamed(db, cache_dir, out_dir):
        """Run the production builder into ``out_dir`` and read its nine columns
        back with the path text joined in."""
        out_dir.mkdir(parents=True, exist_ok=True)
        built = parquet_cache._build_nvtx_kernel_map_from_parquet(db, cache_dir, out_dir)
        if not built:
            return None
        mp = parquet_cache._safe_path(out_dir / "nvtx_kernel_map.parquet")
        dp = parquet_cache._safe_path(out_dir / "nvtx_path_dict.parquet")
        return db.execute(
            f"SELECT m.nvtx_text, m.nvtx_depth, d.nvtx_path, m.kernel_name, m.k_start, "
            f"m.k_end, m.k_dur_ns, m.is_tc_eligible, m.uses_tc "
            f"FROM read_parquet('{mp}') m JOIN read_parquet('{dp}') d USING (path_id)"
        ).fetchall()

    @staticmethod
    def _oracle(db, cache_dir):
        """The same answer via ``_sweep_nvtx_kernel_map``, which holds both sides
        in memory and is what the on-demand builder still uses.

        Not a reimplementation of the streamed builder: it reads both sources in
        one shot, in the shape the pre-streaming builder read them, and attaches
        the Tensor Core flags afterwards from a side table the way that builder
        did. Only the containment core is shared.
        """
        kps = parquet_cache._safe_path(cache_dir / "kernels.parquet")
        rps = parquet_cache._safe_path(cache_dir / "runtime.parquet")
        nps = parquet_cache._safe_path(cache_dir / "nvtx.parquet")
        kr = db.execute(
            f'SELECT r.globalTid, r.start, r."end", k.start, k."end", k.name, '
            f"COALESCE(CAST(k.is_tc_eligible AS INTEGER), 0), "
            f"COALESCE(CAST(k.uses_tc AS INTEGER), 0) "
            f"FROM read_parquet('{kps}') k "
            f"JOIN read_parquet('{rps}') r ON r.correlationId = k.correlationId"
        ).fetchall()
        nvtx = db.execute(
            f'SELECT globalTid, start, "end", CAST(text AS VARCHAR) '
            f"FROM read_parquet('{nps}') "
            f'WHERE eventType = 59 AND "end" > start AND text IS NOT NULL '
            f"ORDER BY globalTid, start"
        ).fetchall()
        tc = {(r[3], r[4], r[5]): (r[6], r[7]) for r in kr}
        rows = []
        for r in parquet_cache._sweep_nvtx_kernel_map([k[:6] for k in kr], nvtx):
            elig, used = tc.get((r["k_start"], r["k_end"], r["kernel_name"]), (0, 0))
            rows.append(
                (
                    r["nvtx_text"],
                    r["nvtx_depth"],
                    r["nvtx_path"],
                    r["kernel_name"],
                    r["k_start"],
                    r["k_end"],
                    r["k_dur_ns"],
                    elig,
                    used,
                )
            )
        return rows

    @pytest.mark.parametrize("fixture", FIXTURES)
    def test_the_streamed_builder_matches_the_list_sweep_row_for_row(self, tmp_path, fixture):
        """Row count first, then the rows themselves, in both directions."""
        import collections

        import duckdb

        cache_dir = self._cache_for(tmp_path, fixture)
        db = duckdb.connect()
        try:
            got = self._streamed(db, cache_dir, tmp_path / "streamed")
            want = self._oracle(db, cache_dir)
        finally:
            db.close()

        assert want, f"{fixture}: the oracle attributed nothing, so this compares nothing"
        assert got is not None, f"{fixture}: the streamed builder wrote no map"
        # Stated separately from the contents comparison because a truncated
        # sweep is the specific failure this class exists to catch, and a count
        # says so unambiguously.
        assert len(got) == len(want), (
            f"{fixture}: the streamed builder produced {len(got)} rows against the "
            f"sweep's {len(want)}"
        )
        got_counts = collections.Counter(got)
        want_counts = collections.Counter(want)
        assert got_counts == want_counts, (
            f"{fixture}: {sum((want_counts - got_counts).values())} rows missing and "
            f"{sum((got_counts - want_counts).values())} extra"
        )

    def test_shrinking_the_arrow_batch_does_not_change_the_answer(self, tmp_path, monkeypatch):
        """Every batch boundary exercised at once, and the truncation trap with it.

        ``_SWEEP_BATCH_ROWS`` sizes both input readers and the output buffer, so
        a small prime here puts range and kernel boundaries at unrelated offsets
        and splits the output across many row groups. Nothing about the answer
        may move: the open stack is per thread and a thread's ranges are strictly
        nested, so there is no carried state at a batch boundary to get wrong.

        This is also the regression test for the two-readers-one-connection
        truncation, and it is the *shrunk* batch that gives it teeth. At the
        default 262,144 rows this fixture's NVTX side is a single batch, so the
        reader has already handed every row to Python before the kernel reader
        opens and a shared handle loses nothing — measured, 1,643 rows either
        way. At batch size 7 the same mutation yields 2.
        """
        import duckdb

        cache_dir = self._cache_for(tmp_path, "h100_2gpu_1s.sqlite")
        db = duckdb.connect()
        try:
            full = self._streamed(db, cache_dir, tmp_path / "full")
            monkeypatch.setattr(parquet_cache, "_SWEEP_BATCH_ROWS", 7)
            tiny = self._streamed(db, cache_dir, tmp_path / "tiny")
        finally:
            db.close()

        assert full, "the control build attributed nothing"
        assert tiny is not None, "the shrunk-batch build wrote no map"
        assert len(tiny) == len(full), (
            f"batching changed the row count: {len(tiny)} at batch 7 against {len(full)}"
        )
        assert sorted(tiny) == sorted(full), "batching changed the attribution"

    def test_both_map_writers_emit_the_same_columns(self, tmp_path):
        """The two producers of a nvtx_kernel_map, column by column.

        There are two, and since the streamed builder stopped sharing a writer
        with the in-memory one they list their columns independently:
        ``_nvtx_map_arrow_tables`` (in-memory, reached through
        ``_materialize_nvtx_kernel_map``) and the ``SELECT`` inside
        ``_publish_nvtx_kernel_map_parquet`` (cached). Nothing else in the suite
        compares them — the oracle test above checks the cached map's *rows*
        against ``_sweep_nvtx_kernel_map`` and never calls the Arrow writer — so
        a tenth column added to one of them would land with the suite green.

        That is not cosmetic: ``connection.cached_nvtx_map_has_embedded_tc``
        decides by column presence, so a map missing ``is_tc_eligible`` /
        ``uses_tc`` reroutes every NVTX consumer onto a different aggregate
        rather than failing.
        """
        import duckdb

        cache_dir = self._cache_for(tmp_path, "h100_2gpu_1s.sqlite")
        out_dir = tmp_path / "published"
        out_dir.mkdir()
        db = duckdb.connect()
        try:
            built = parquet_cache._build_nvtx_kernel_map_from_parquet(db, cache_dir, out_dir)
            assert built, "the cached builder wrote no map to compare against"
            published = {
                name: [
                    (r[0], r[1])
                    for r in db.execute(
                        f"DESCRIBE SELECT * FROM "
                        f"read_parquet('{parquet_cache._safe_path(out_dir / name)}')"
                    ).fetchall()
                ]
                for name in ("nvtx_kernel_map.parquet", "nvtx_path_dict.parquet")
            }
        finally:
            db.close()

        # An empty result is enough: the schema is fixed by the writer, not by
        # the rows, and this keeps the comparison about column names and types.
        map_tbl, dict_tbl = parquet_cache._nvtx_map_arrow_tables([])
        arrow_to_duckdb = {"string": "VARCHAR", "int32": "INTEGER", "int64": "BIGINT"}

        def as_duckdb(table):
            return [(f.name, arrow_to_duckdb[str(f.type)]) for f in table.schema]

        assert as_duckdb(map_tbl) == published["nvtx_kernel_map.parquet"], (
            "the in-memory map writer and the cached one no longer agree; "
            "_nvtx_map_arrow_tables and _publish_nvtx_kernel_map_parquet list "
            "their columns separately and have to be edited together"
        )
        assert as_duckdb(dict_tbl) == published["nvtx_path_dict.parquet"], (
            "the two path-dictionary writers no longer agree"
        )

    def test_threads_with_only_one_side_are_swept_without_affecting_the_others(self, tmp_path):
        """The three partition shapes a real capture contains.

        On the 3.5 GB reference capture one thread carries 3,917 ranges and no
        kernels at all, eight carry kernels and no ranges, and four carry both.
        Built here directly as Parquet rather than through a SQLite fixture, so
        the thread layout is stated rather than hoped for.

        The kernel on thread 3 sits inside a range that opened before it and
        closes after it, which is the case a stream cannot recover from by
        re-reading: the range has to still be on the open stack when the kernel
        arrives.
        """
        import duckdb
        import pyarrow as pa
        import pyarrow.parquet as pq

        cache_dir = tmp_path / "synthetic.nsys-cache"
        cache_dir.mkdir()
        pq.write_table(
            pa.table(
                {
                    "correlationId": pa.array([1, 2, 3], pa.int64()),
                    "start": pa.array([1100, 2100, 3100], pa.int64()),
                    "end": pa.array([1200, 2200, 3200], pa.int64()),
                    "name": pa.array(["kA", "kB", "kC"], pa.string()),
                    "is_tc_eligible": pa.array([1, 0, 0], pa.int32()),
                    "uses_tc": pa.array([1, 0, 0], pa.int32()),
                }
            ),
            cache_dir / "kernels.parquet",
        )
        pq.write_table(
            pa.table(
                {
                    # tid 2 launches two kernels and has no NVTX ranges at all;
                    # tid 3 launches one, inside a range that opened long before
                    # it. tid 1 appears only on the NVTX side.
                    "correlationId": pa.array([1, 2, 3], pa.int64()),
                    "globalTid": pa.array([2, 2, 3], pa.int64()),
                    "start": pa.array([100, 200, 500], pa.int64()),
                    "end": pa.array([150, 250, 550], pa.int64()),
                }
            ),
            cache_dir / "runtime.parquet",
        )
        pq.write_table(
            pa.table(
                {
                    "globalTid": pa.array([1, 3, 3, 3], pa.int64()),
                    "start": pa.array([0, 0, 400, 900], pa.int64()),
                    "end": pa.array([999, 999, 600, 999], pa.int64()),
                    "text": pa.array(["orphan", "step", "fwd", "after"], pa.string()),
                    "eventType": pa.array([59, 59, 59, 59], pa.int32()),
                }
            ),
            cache_dir / "nvtx.parquet",
        )

        db = duckdb.connect()
        try:
            rows = self._streamed(db, cache_dir, tmp_path / "out")
        finally:
            db.close()

        # tid 2's kernels have no enclosing range and tid 1 has no kernels, so
        # exactly one row survives — attributed to the inner range, with the
        # outer one still open around it.
        assert rows == [("fwd", 1, "step > fwd", "kC", 3100, 3200, 100, 0, 0)], rows

    def test_a_capture_with_nothing_to_attribute_publishes_no_map(self, tmp_path):
        """No kernel inside any range is a skip, not an empty map and not a crash.

        The streamed builder cannot test its inputs for emptiness — they are
        generators opened per thread — so "no kernels", "no ranges" and "no
        overlap" all have to come out the same way: nothing written, False
        returned, and the scratch Parquet cleaned up behind it.
        """
        import duckdb
        import pyarrow as pa
        import pyarrow.parquet as pq

        cache_dir = tmp_path / "empty.nsys-cache"
        cache_dir.mkdir()
        pq.write_table(
            pa.table(
                {
                    "correlationId": pa.array([1], pa.int64()),
                    "start": pa.array([1100], pa.int64()),
                    "end": pa.array([1200], pa.int64()),
                    "name": pa.array(["kA"], pa.string()),
                    "is_tc_eligible": pa.array([0], pa.int32()),
                    "uses_tc": pa.array([0], pa.int32()),
                }
            ),
            cache_dir / "kernels.parquet",
        )
        pq.write_table(
            pa.table(
                {
                    "correlationId": pa.array([1], pa.int64()),
                    "globalTid": pa.array([9], pa.int64()),
                    "start": pa.array([100], pa.int64()),
                    "end": pa.array([150], pa.int64()),
                }
            ),
            cache_dir / "runtime.parquet",
        )
        pq.write_table(
            pa.table(
                {
                    "globalTid": pa.array([9], pa.int64()),
                    "start": pa.array([500], pa.int64()),
                    "end": pa.array([600], pa.int64()),
                    "text": pa.array(["elsewhere"], pa.string()),
                    "eventType": pa.array([59], pa.int32()),
                }
            ),
            cache_dir / "nvtx.parquet",
        )

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        db = duckdb.connect()
        try:
            built = parquet_cache._build_nvtx_kernel_map_from_parquet(db, cache_dir, out_dir)
        finally:
            db.close()

        assert built is False
        assert not (out_dir / "nvtx_kernel_map.parquet").exists()
        assert not (out_dir / "nvtx_path_dict.parquet").exists()
        assert list(out_dir.iterdir()) == [], (
            f"the sweep left scratch files behind: {sorted(p.name for p in out_dir.iterdir())}"
        )
