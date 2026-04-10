from unittest import mock

import pytest

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

    # small file (< 50MB) should build parquet
    cache_dir = _cache_dir_for(str(db_path))
    assert cache_dir.exists()


def test_cache_mode_auto_large_mocked(minimal_nsys_db_path, tmp_path):
    db_path = tmp_path / "test_auto_large.sqlite"
    with open(minimal_nsys_db_path, "rb") as src, open(db_path, "wb") as dst:
        dst.write(src.read())

    # Mock os.path.getsize to return 100MB
    with mock.patch("os.path.getsize", return_value=100.0 * 1e6):
        with Profile(str(db_path), cache_mode="auto") as prof:
            # Large file should fallback to direct scanning and NOT build parquet
            cache_dir = _cache_dir_for(str(db_path))
            assert not cache_dir.exists()

            # Alias views should still be accessible
            res = prof.db.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()
            assert res[0] >= 0
