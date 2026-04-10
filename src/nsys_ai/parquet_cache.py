"""parquet_cache.py — DuckDB + Parquet cache for Nsight Systems profiles.

Accelerates repeated profile analysis by exporting key tables from the
original SQLite export into Parquet files (ZSTD-compressed), then serving
queries via DuckDB views over those Parquet files.

Flow:
  1. First open: ``build_cache()`` attaches the SQLite DB via DuckDB,
     exports tables into a sibling cache directory named
     ``<profile_basename>.nsys-cache`` (e.g., ``profile.nsys-cache``) as
     Parquet, and runs the Tier 2 sort-merge to produce
     ``nvtx_kernel_map.parquet``.
  2. Subsequent opens: ``open_cached_db()`` creates a DuckDB connection
     with views pointing at the cached Parquet files in that
     ``<profile_basename>.nsys-cache`` directory — sub-second startup.

Cache invalidation uses mtime comparison + a version stamp file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from hashlib import sha256
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)

# Bump this when the cache schema changes (e.g., new columns, new tables).
_CACHE_VERSION = 7  # bumped: preserve NVTX blobs and payload schema tables

_SAFE_PARQUETDIR_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")

# Tables to export as-is from SQLite → Parquet.
# (view_name, source_table_name)
_BASE_TABLES = [
    ("runtime", "CUPTI_ACTIVITY_KIND_RUNTIME"),
    ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY"),
    ("memset", "CUPTI_ACTIVITY_KIND_MEMSET"),
    ("overhead", "CUPTI_ACTIVITY_KIND_OVERHEAD"),
    ("profiler_overhead", "PROFILER_OVERHEAD"),
    ("composite_events", "COMPOSITE_EVENTS"),
    ("string_ids", "StringIds"),
    ("gpu_info", "TARGET_INFO_GPU"),
    ("cuda_device", "TARGET_INFO_CUDA_DEVICE"),
    ("thread_names", "ThreadNames"),
    ("sync", "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"),
    ("sync_type", "ENUM_CUPTI_SYNC_TYPE"),
    ("nic_info", "TARGET_INFO_NIC_INFO"),
    ("nvtx_payload_schemas", "NVTX_PAYLOAD_SCHEMAS"),
    ("nvtx_payload_schema_entries", "NVTX_PAYLOAD_SCHEMA_ENTRIES"),
    ("nvtx_payload_enums", "NVTX_PAYLOAD_ENUMS"),
    ("nvtx_payload_enum_entries", "NVTX_PAYLOAD_ENUM_ENTRIES"),
]


def _cache_dir_for(sqlite_path: str) -> Path:
    """Return the cache directory path for a given SQLite profile."""
    return Path(sqlite_path).with_suffix(".nsys-cache")


def is_cache_valid(sqlite_path: str) -> bool:
    """Check whether the Parquet cache is up-to-date.

    Returns True if:
      - The cache directory exists
      - The version stamp matches ``_CACHE_VERSION``
      - The cache is at least as new as the SQLite file (mtime comparison)
    """
    cache_dir = _cache_dir_for(sqlite_path)
    version_file = cache_dir / ".cache_version"

    if not cache_dir.exists() or not version_file.exists():
        return False

    # Version check
    try:
        meta = json.loads(version_file.read_text())
        if meta.get("version") != _CACHE_VERSION:
            return False
        is_empty = meta.get("empty", False)
    except (json.JSONDecodeError, OSError):
        return False

    # Freshness check: cache must be newer than the source SQLite
    try:
        sqlite_mtime = os.path.getmtime(sqlite_path)
        cache_mtime = os.path.getmtime(version_file)
        if sqlite_mtime > cache_mtime:
            return False
    except OSError:
        return False

    # Quick sanity: at least one core Parquet (e.g., string_ids) must exist unless marked empty
    if not is_empty and not (cache_dir / "string_ids.parquet").exists():
        return False

    return True


def invalidate_cache(sqlite_path: str) -> None:
    """Remove the Parquet cache for a profile, forcing rebuild on next open."""
    import shutil

    cache_dir = _cache_dir_for(sqlite_path)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        log.info("Removed cache: %s", cache_dir)


def build_cache(sqlite_path: str) -> Path:
    """Build a Parquet cache from a SQLite profile (first-run ETL).

    Attaches the SQLite DB via DuckDB, exports key tables to Parquet with
    ZSTD compression, and generates ``nvtx_kernel_map.parquet`` using the
    Python Tier 2 sort-merge logic.

    Returns the cache directory path.
    """
    import shutil
    import tempfile

    cache_dir = _cache_dir_for(sqlite_path)
    # Build into a temp dir first, then atomically rename to avoid race
    # conditions when multiple threads/processes open the same profile.
    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix=".parquet_build_",
            dir=cache_dir.parent,
        )
    )
    try:
        _build_cache_into(sqlite_path, tmp_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    # Atomic swap: rename old cache aside, rename new into place, then clean up.
    # This avoids a window where the cache directory is missing for concurrent readers.
    # Use PID in the old-dir name so concurrent builders don't collide.
    old_dir = cache_dir.parent / f"{cache_dir.name}.old.{os.getpid()}"
    if old_dir.exists():
        shutil.rmtree(old_dir, ignore_errors=True)

    try:
        if cache_dir.exists():
            cache_dir.rename(old_dir)
        tmp_dir.rename(cache_dir)
    except (FileExistsError, OSError):
        # Another process won the race and created cache_dir first.
        # Discard our redundant build.
        shutil.rmtree(tmp_dir, ignore_errors=True)
        # Restore old_dir if cache_dir does not exist (failed for other transient reasons)
        if old_dir.exists() and not cache_dir.exists():
            try:
                old_dir.rename(cache_dir)
            except OSError:
                pass

    # Clean up the old cache (now renamed aside) if we have a robust valid cache.
    if old_dir.exists() and cache_dir.exists():
        shutil.rmtree(old_dir, ignore_errors=True)
    return cache_dir


def _safe_path(p: Path) -> str:
    """Safely format a Path into a single-quoted string for DuckDB COPY."""
    return p.as_posix().replace("'", "''")


# Column projections for large tables — only export columns that downstream
# skills actually use.  Reduces I/O and memory during cache build.
_TABLE_PROJECTIONS: dict[str, str] = {
    # Verified via full codebase audit: all 8 consumer files only use these 5.
    "CUPTI_ACTIVITY_KIND_RUNTIME": 'start, "end", correlationId, globalTid, nameId',
    "CUPTI_ACTIVITY_KIND_RUNTIME_V2": 'start, "end", correlationId, globalTid, nameId',
    "CUPTI_ACTIVITY_KIND_RUNTIME_V3": 'start, "end", correlationId, globalTid, nameId',
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION": 'start, "end", globalPid, syncType',
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION_V2": 'start, "end", globalPid, syncType',
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION_V3": 'start, "end", globalPid, syncType',
    "ENUM_CUPTI_SYNC_TYPE": "id, name",
}

_TC_ELIGIBLE_PATTERN = "'(gemm|conv|linear|attention|matmul)'"
_TC_ACTIVE_PATTERN = "'(xmma|mma_sync|16816|1688|884|ampere_bf16|sm80_tensor_op)'"


# Mapping from cache view name (e.g. "kernels") to the actual SQLite table names that
# consumer queries might request. We use this to create stable alias views so queries
# work regardless of which table string they use.
_ALIASES: dict[str, list[str]] = {
    "kernels": [
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_KERNEL_V2",
        "CUPTI_ACTIVITY_KIND_KERNEL_V3",
    ],
    "nvtx": ["NVTX_EVENTS"],
    "runtime": [
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "CUPTI_ACTIVITY_KIND_RUNTIME_V2",
        "CUPTI_ACTIVITY_KIND_RUNTIME_V3",
    ],
    "memcpy": [
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMCPY_V2",
        "CUPTI_ACTIVITY_KIND_MEMCPY_V3",
    ],
    "memset": [
        "CUPTI_ACTIVITY_KIND_MEMSET",
        "CUPTI_ACTIVITY_KIND_MEMSET_V2",
        "CUPTI_ACTIVITY_KIND_MEMSET_V3",
    ],
    "string_ids": ["StringIds"],
    "gpu_info": ["TARGET_INFO_GPU"],
    "cuda_device": ["TARGET_INFO_CUDA_DEVICE"],
    "nic_info": ["TARGET_INFO_NIC_INFO"],
    "thread_names": ["ThreadNames"],
    "overhead": ["CUPTI_ACTIVITY_KIND_OVERHEAD"],
    "composite_events": ["COMPOSITE_EVENTS"],
    "sync": ["CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"],
    "sync_type": ["ENUM_CUPTI_SYNC_TYPE"],
    "nvtx_payload_schemas": ["NVTX_PAYLOAD_SCHEMAS"],
    "nvtx_payload_schema_entries": ["NVTX_PAYLOAD_SCHEMA_ENTRIES"],
    "nvtx_payload_enums": ["NVTX_PAYLOAD_ENUMS"],
    "nvtx_payload_enum_entries": ["NVTX_PAYLOAD_ENUM_ENTRIES"],
}

_PARQUETDIR_BINARY_COLUMNS: dict[str, tuple[str, ...]] = {
    "NVTX_EVENTS": ("binaryData",),
}


def _build_cache_into(sqlite_path: str, cache_dir: Path) -> Path:
    """Internal: build the Parquet cache into the given directory."""

    log.info("Building analysis cache (first run only)...")
    t0 = time.monotonic()

    db = duckdb.connect()
    try:
        # ── DuckDB performance tuning ─────────────────────────────────
        # Disable insertion-order preservation to allow multi-threaded
        # output without synchronization — significant speedup for large
        # COPY operations.  Safe because we never rely on implicit row order.
        db.execute("SET preserve_insertion_order = false")
        # Disable DuckDB's built-in progress bar — it writes \r escape sequences
        # to stderr that interleave with our own \r-based progress reporting,
        # producing garbled output in terminals and CI logs.
        db.execute("SET enable_progress_bar = false")

        # Attach the SQLite database
        safe_sqlite_path = str(sqlite_path).replace("'", "''")
        try:
            db.execute(f"ATTACH '{safe_sqlite_path}' AS src (TYPE SQLITE, READ_ONLY)")
        except duckdb.Error:
            # Clean up partial attach before retry with permissive typing
            try:
                db.execute("DETACH src")
            except duckdb.Error:
                pass
            db.execute("SET sqlite_all_varchar = true")
            db.execute(f"ATTACH '{safe_sqlite_path}' AS src (TYPE SQLITE, READ_ONLY)")

        # Discover which tables actually exist in the source
        # Note: DuckDB doesn't expose sqlite_master from attached DBs.
        # Use SHOW ALL TABLES and filter by the attached database name.
        src_tables: set[str] = set()
        try:
            for row in db.execute("SHOW ALL TABLES").fetchall():
                # row format: (database, schema, name, column_names, column_types, temporary)
                if row[0] == "src":
                    src_tables.add(row[2])
        except duckdb.Error:
            # Fallback: try to list tables another way
            try:
                for row in db.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_catalog = 'src'"
                ).fetchall():
                    src_tables.add(row[0])
            except duckdb.Error:
                log.warning("Could not discover tables in attached SQLite")

        # ── Progress reporting ─────────────────────────────────────────
        # Count total steps for progress display
        total_steps = sum(1 for _, src_name in _BASE_TABLES if _find_table(src_tables, src_name))

        has_kernel = bool(_find_table(src_tables, "CUPTI_ACTIVITY_KIND_KERNEL"))
        has_nvtx = bool(_find_table(src_tables, "NVTX_EVENTS"))
        has_runtime = bool(_find_table(src_tables, "CUPTI_ACTIVITY_KIND_RUNTIME"))

        if has_kernel:
            total_steps += 1
        if has_nvtx:
            total_steps += 1
        if has_kernel and has_nvtx and has_runtime:
            total_steps += 1
        step = [0]

        def _progress(name: str) -> None:
            step[0] += 1
            elapsed = time.monotonic() - t0
            sys.stderr.write(
                f"\r[nsys-ai] Building cache [{step[0]}/{total_steps}] {name} ({elapsed:.0f}s)"
            )
            sys.stderr.flush()

        # ── Export pre-joined kernels table ────────────────────────────────
        kernel_table = _find_table(src_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        if kernel_table:
            _progress("kernels.parquet")
            db.execute(f"""
                COPY (
                    SELECT k.*, COALESCE(d.value, s.value, 'kernel_' || CAST(k.shortName AS VARCHAR)) AS name, d.value AS demangled,
                           CAST(CASE
                   WHEN regexp_matches(lower(COALESCE(d.value, s.value, '')), {_TC_ELIGIBLE_PATTERN})
                     OR regexp_matches(lower(COALESCE(d.value, s.value, '')), {_TC_ACTIVE_PATTERN})
                   THEN 1
                   ELSE 0
               END AS INTEGER) AS is_tc_eligible,
                           CAST(CASE WHEN regexp_matches(lower(COALESCE(d.value, s.value, '')), {_TC_ACTIVE_PATTERN}) THEN 1 ELSE 0 END AS INTEGER) AS uses_tc
                    FROM src.{kernel_table} k
                    LEFT JOIN src.StringIds s ON k.shortName = s.id
                    LEFT JOIN src.StringIds d ON k.demangledName = d.id
                ) TO '{_safe_path(cache_dir / "kernels.parquet")}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """)

        # ── Export NVTX with resolved text ────────────────────────────────
        nvtx_table = _find_table(src_tables, "NVTX_EVENTS")
        if nvtx_table:
            _progress("nvtx.parquet")
            _export_nvtx_with_blobs(sqlite_path, nvtx_table, cache_dir)

        for view_name, src_name in _BASE_TABLES:
            actual = _find_table(src_tables, src_name)
            if actual:
                _progress(f"{view_name}.parquet")
                projection = _TABLE_PROJECTIONS.get(actual, "*")
                if projection == "*":
                    db.execute(f"""
                        COPY src.{actual}
                        TO '{_safe_path(cache_dir / f"{view_name}.parquet")}' (FORMAT PARQUET, COMPRESSION ZSTD)
                    """)
                else:
                    db.execute(f"""
                        COPY (SELECT {projection} FROM src.{actual})
                        TO '{_safe_path(cache_dir / f"{view_name}.parquet")}' (FORMAT PARQUET, COMPRESSION ZSTD)
                    """)

        # ── Generate nvtx_kernel_map ──────────────────────────────────────
        if has_kernel and has_nvtx and has_runtime:
            _progress("nvtx_kernel_map.parquet")
            _build_nvtx_kernel_map(db, src_tables, cache_dir, sqlite_path)

        # Clear progress line
        elapsed = time.monotonic() - t0
        sys.stderr.write(f"\r[nsys-ai] Cache ready ({elapsed:.1f}s)" + " " * 40 + "\n")
        sys.stderr.flush()

        # ── Write version stamp ───────────────────────────────────────────
        meta = {
            "version": _CACHE_VERSION,
            "source": os.path.basename(sqlite_path),
            "empty": len(src_tables) == 0 or not _find_table(src_tables, "StringIds"),
        }
        (cache_dir / ".cache_version").write_text(json.dumps(meta))

        # ── Size report ───────────────────────────────────────────────────
        total_bytes = sum(f.stat().st_size for f in cache_dir.iterdir() if f.is_file())
        log.info(
            "Cache ready: %s/ (%.0fMB, %.1fs)",
            cache_dir.name,
            total_bytes / 1e6,
            elapsed,
        )

        _check_cache_size(cache_dir, sqlite_path)
    finally:
        db.close()
    return cache_dir


def open_cached_db(sqlite_path: str) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with views over the Parquet cache.

    If the cache doesn't exist or is stale, builds it first.

    Returns a DuckDB connection with views named after each cached table:
      ``kernels``, ``nvtx``, ``runtime``, ``memcpy``, ``memset``,
      ``string_ids``, ``gpu_info``, ``cuda_device``, ``nvtx_kernel_map``.
    """
    if not is_cache_valid(sqlite_path):
        build_cache(sqlite_path)

    cache_dir = _cache_dir_for(sqlite_path)

    # Validate that the cache actually contains parquet files.
    # If build_cache() ran against a non-Nsight DB, the cache may be empty.
    parquet_files = list(cache_dir.glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(
            f"Parquet cache at {cache_dir} is empty — "
            f"the source file may not be a valid Nsight Systems export"
        )

    db = duckdb.connect()

    # Create views over Parquet files
    for parquet_file in cache_dir.glob("*.parquet"):
        view_name = parquet_file.stem
        safe_fpath = str(parquet_file).replace("'", "''")
        db.execute(f"CREATE VIEW \"{view_name}\" AS SELECT * FROM '{safe_fpath}'")

    _create_existing_alias_views(db)

    return db


def open_parquetdir_db(parquetdir_path: str) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection over an Nsight `parquetdir` export."""
    parquet_dir = Path(parquetdir_path)
    if not parquet_dir.is_dir():
        raise RuntimeError(
            f"Parquet directory path does not exist or is not a directory: {parquet_dir}"
        )
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(
            f"Parquet directory at {parquet_dir} does not contain any .parquet files"
        )

    db = duckdb.connect()
    try:
        _register_parquetdir_tables(db, parquet_dir, parquet_files)
        _create_existing_alias_views(db)
    except Exception:
        try:
            db.close()
        except Exception:
            pass
        raise
    return db


# ── Internal helpers ────────────────────────────────────────────────


def _find_table(tables: set[str], prefix: str) -> str | None:
    """Find the actual table name, handling versioned variants (e.g., _V2)."""
    if prefix in tables:
        return prefix
    candidates = sorted(t for t in tables if t.startswith(prefix + "_V"))
    return candidates[0] if candidates else None


def _table_has_column(db: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
    """Check whether a table/view has a specific column."""
    try:
        cols = db.execute(f"DESCRIBE {table}").fetchall()
        return any(c[0] == column for c in cols)
    except duckdb.Error:
        return False


def _create_existing_alias_views(db: duckdb.DuckDBPyConnection) -> None:
    """Create stable aliases for whatever canonical tables already exist."""
    existing_views = {r[0] for r in db.execute("SHOW TABLES").fetchall()}
    for short_name, aliases in _ALIASES.items():
        actual = None
        if short_name in existing_views:
            actual = short_name
        else:
            for alias in aliases:
                if alias in existing_views:
                    actual = alias
                    break
            if actual is None and aliases:
                actual = _find_table(existing_views, aliases[0])
        if not actual:
            continue
        for alias in [short_name, *aliases]:
            if alias in existing_views:
                continue
            try:
                db.execute(f'CREATE VIEW "{alias}" AS SELECT * FROM "{actual}"')
                existing_views.add(alias)
            except duckdb.Error:
                pass


def _register_parquetdir_tables(
    db: duckdb.DuckDBPyConnection,
    parquet_dir: Path,
    parquet_files: list[Path],
) -> None:
    """Create views for a raw Nsight parquetdir export.

    Nsight 2026 marks `NVTX_EVENTS.binaryData` as a UTF-8 string in Parquet
    metadata even though it contains arbitrary bytes. DuckDB rejects those
    rows during direct Parquet scans, so we repair that column via PyArrow and
    register the resulting Arrow table with DuckDB. Other tables can stay on
    the normal `read_parquet()` path.
    """
    for parquet_file in parquet_files:
        table_name = parquet_file.stem
        # Validate table name to prevent SQL injection from unexpected filenames.
        if not _SAFE_PARQUETDIR_NAME_RE.match(table_name):
            log.warning("Skipping parquet file with unsafe name: %s", parquet_file.name)
            continue
        # Escape double-quotes in identifiers as defence-in-depth.
        safe_name = table_name.replace('"', '""')
        if table_name in _PARQUETDIR_BINARY_COLUMNS:
            try:
                repaired = _repair_parquet_binary_columns_to_disk(parquet_file, table_name)
                safe_fpath = str(repaired).replace("'", "''")
                db.execute(
                    f'CREATE VIEW "{safe_name}" AS SELECT * FROM read_parquet(\'{safe_fpath}\')'
                )
            except Exception as exc:
                log.warning(
                    "Falling back to in-memory Arrow repair for %s due to: %s",
                    parquet_file,
                    exc,
                )
                arrow_name = f"_parquetdir_{table_name}"
                table = _load_parquet_table_for_duckdb(parquet_file, table_name)
                db.register(arrow_name, table)
                db.execute(f'CREATE VIEW "{safe_name}" AS SELECT * FROM "{arrow_name}"')
            continue

        safe_fpath = str(parquet_file).replace("'", "''")
        db.execute(
            f'CREATE VIEW "{safe_name}" AS SELECT * FROM read_parquet(\'{safe_fpath}\')'
        )


def _load_parquet_table_for_duckdb(parquet_file: Path, table_name: str):
    """Load a Parquet file into Arrow and normalize binary payload columns."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    binary_columns = set(_PARQUETDIR_BINARY_COLUMNS.get(table_name, ()))
    if not binary_columns:
        return pq.read_table(parquet_file)

    parquet = pq.ParquetFile(parquet_file)
    cast_targets: dict[str, pa.DataType] = {}
    for field in parquet.schema_arrow:
        if field.name in binary_columns:
            cast_targets[field.name] = pa.large_binary()
    if not cast_targets:
        return parquet.read()

    # Process by record batch so we do not hold both pre-cast and post-cast
    # full tables at once for large NVTX payload datasets.
    batches = []
    for batch in parquet.iter_batches():
        batch_arrays = []
        for idx, field in enumerate(batch.schema):
            column = batch.column(idx)
            target_type = cast_targets.get(field.name)
            if target_type is not None:
                column = pc.cast(column, target_type, safe=False)
            batch_arrays.append(column)
        batches.append(pa.record_batch(batch_arrays, names=batch.schema.names))
    return pa.Table.from_batches(batches)


def _repair_parquet_binary_columns_to_disk(parquet_file: Path, table_name: str) -> Path:
    """Repair mis-typed binary columns into a cached parquet file on disk."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    binary_columns = set(_PARQUETDIR_BINARY_COLUMNS.get(table_name, ()))
    if not binary_columns:
        return parquet_file

    src_stat = parquet_file.stat()
    cache_key = (
        f"{parquet_file.resolve()}:{src_stat.st_mtime_ns}:{src_stat.st_size}:"
        f"{','.join(sorted(binary_columns))}"
    )
    digest = sha256(cache_key.encode("utf-8")).hexdigest()[:20]
    # Keep repaired artifacts scoped to the profile directory instead of
    # global /tmp, so lifecycle naturally tracks the source parquetdir.
    out_dir = parquet_file.parent / ".nsys_ai_parquetdir_repaired"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{parquet_file.stem}.{digest}.parquet"
    if out_path.exists():
        return out_path

    parquet = pq.ParquetFile(parquet_file)
    source_schema = parquet.schema_arrow
    fields = []
    cast_targets: dict[str, pa.DataType] = {}
    for field in source_schema:
        if field.name in binary_columns:
            cast_targets[field.name] = pa.large_binary()
            fields.append(
                pa.field(
                    field.name,
                    pa.large_binary(),
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
            )
        else:
            fields.append(field)
    target_schema = pa.schema(fields, metadata=source_schema.metadata)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with pq.ParquetWriter(tmp_path, target_schema, compression="zstd") as writer:
        for batch in parquet.iter_batches():
            arrays = []
            for idx, field in enumerate(batch.schema):
                col = batch.column(idx)
                target_type = cast_targets.get(field.name)
                if target_type is not None:
                    col = pc.cast(col, target_type, safe=False)
                arrays.append(col)
            writer.write_batch(pa.record_batch(arrays, schema=target_schema))
    tmp_path.replace(out_path)
    return out_path


def _export_nvtx_with_blobs(sqlite_path: str, nvtx_table: str, cache_dir: Path) -> None:
    """Export NVTX rows via a varchar-only attachment so mixed TEXT/BLOB columns survive.

    The regular typed SQLite scanner cannot read NVTX `binaryData` when the
    SQLite export mixes TEXT and BLOB affinity in that column. A separate
    DuckDB connection with `sqlite_all_varchar=true` avoids that issue; we
    cast numeric columns back to their intended types and store the blob as a
    hex string for cache portability.
    """
    safe_sqlite_path = sqlite_path.replace("'", "''")
    db = duckdb.connect()
    try:
        db.execute("SET sqlite_all_varchar = true")
        db.execute(f"ATTACH '{safe_sqlite_path}' AS srcv (TYPE SQLITE, READ_ONLY)")
        table_ref = f"srcv.{nvtx_table}"

        def _expr(column: str, sql_type: str, alias: str | None = None) -> str:
            alias = alias or column
            if _table_has_column(db, table_ref, column):
                return f'CAST(n."{column}" AS {sql_type}) AS "{alias}"'
            return f'CAST(NULL AS {sql_type}) AS "{alias}"'

        has_textid = _table_has_column(db, f"srcv.{nvtx_table}", "textId")
        has_text = _table_has_column(db, table_ref, "text")
        has_json_text = _table_has_column(db, table_ref, "jsonText")
        has_binary = _table_has_column(db, table_ref, "binaryData")
        binary_expr = "hex(n.binaryData) AS binaryData" if has_binary else "CAST(NULL AS VARCHAR) AS binaryData"
        json_text_expr = "n.jsonText AS jsonText" if has_json_text else "CAST(NULL AS VARCHAR) AS jsonText"
        text_expr = "n.text" if has_text else "CAST(NULL AS VARCHAR)"
        if has_textid:
            db.execute(f"""
                COPY (
                    SELECT {_expr("globalTid", "BIGINT")},
                           {_expr("start", "BIGINT")},
                           {_expr("end", "BIGINT")},
                           {_expr("eventType", "INTEGER")},
                           {_expr("rangeId", "BIGINT")},
                           {_expr("category", "BIGINT")},
                           {_expr("color", "BIGINT")},
                           {_expr("endGlobalTid", "BIGINT")},
                           {_expr("domainId", "BIGINT")},
                           {_expr("uint64Value", "BIGINT")},
                           {_expr("int64Value", "BIGINT")},
                           {_expr("doubleValue", "DOUBLE")},
                           {_expr("uint32Value", "BIGINT")},
                           {_expr("int32Value", "BIGINT")},
                           {_expr("floatValue", "DOUBLE")},
                           {_expr("jsonTextId", "BIGINT")},
                           {json_text_expr},
                           {binary_expr},
                           COALESCE({text_expr}, s.value) AS text,
                           {_expr("textId", "BIGINT")}
                    FROM srcv.{nvtx_table} n
                    LEFT JOIN srcv.StringIds s ON n.textId = s.id
                ) TO '{_safe_path(cache_dir / "nvtx.parquet")}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
        else:
            db.execute(f"""
                COPY (
                    SELECT {_expr("globalTid", "BIGINT")},
                           {_expr("start", "BIGINT")},
                           {_expr("end", "BIGINT")},
                           {_expr("eventType", "INTEGER")},
                           {_expr("rangeId", "BIGINT")},
                           {_expr("category", "BIGINT")},
                           {_expr("color", "BIGINT")},
                           {_expr("endGlobalTid", "BIGINT")},
                           {_expr("domainId", "BIGINT")},
                           {_expr("uint64Value", "BIGINT")},
                           {_expr("int64Value", "BIGINT")},
                           {_expr("doubleValue", "DOUBLE")},
                           {_expr("uint32Value", "BIGINT")},
                           {_expr("int32Value", "BIGINT")},
                           {_expr("floatValue", "DOUBLE")},
                           {_expr("jsonTextId", "BIGINT")},
                           {json_text_expr},
                           {binary_expr},
                           {text_expr} AS text
                    FROM srcv.{nvtx_table} n
                ) TO '{_safe_path(cache_dir / "nvtx.parquet")}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
    finally:
        db.close()


def _build_nvtx_kernel_map(
    db: duckdb.DuckDBPyConnection,
    src_tables: set[str],
    cache_dir: Path,
    sqlite_path: str,
) -> None:
    """Generate nvtx_kernel_map.parquet using pure DuckDB SQL.

    Uses DuckDB's native IEJoin algorithm for range predicates
    (n.start <= r.start AND n.end >= r.end), which is 20-100x faster
    than the previous Python sort-merge loop.

    Falls back to Python sort-merge if the SQL approach fails.
    """
    kernel_table = _find_table(src_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
    runtime_table = _find_table(src_tables, "CUPTI_ACTIVITY_KIND_RUNTIME")
    nvtx_table = _find_table(src_tables, "NVTX_EVENTS")

    if not all([kernel_table, runtime_table, nvtx_table]):
        log.info("Skipping nvtx_kernel_map: missing required tables")
        return

    has_textid = _table_has_column(db, f"src.{nvtx_table}", "textId")
    if has_textid:
        text_expr = "COALESCE(n.text, ns.value)"
        text_join = "LEFT JOIN src.StringIds ns ON n.textId = ns.id"
    else:
        text_expr = "n.text"
        text_join = ""

    # Pure SQL approach: DuckDB will use IEJoin for the range predicate.
    #
    # Strategy:
    #   1. Join kernel ↔ runtime via correlationId (equality, fast hash join)
    #   2. Join runtime ↔ NVTX via globalTid (equality) + range containment
    #      (n.start <= r.start AND n."end" >= r."end")
    #      DuckDB automatically picks IEJoin for this pattern.
    #   3. GROUP BY (k_start, globalTid) to aggregate all enclosing NVTX:
    #      - string_agg(... ORDER BY n_dur DESC) builds "outer > middle > inner" path
    #      - FIRST(... ORDER BY n_dur ASC) picks the innermost NVTX text
    #      - COUNT(*) - 1 gives the nesting depth
    sql = f"""
    COPY (
        WITH kr AS (
            SELECT r.globalTid, r.start AS r_start, r."end" AS r_end,
                   k.start AS k_start, k."end" AS k_end, k.shortName,
                   COALESCE(kd.value, ks.value, 'kernel_' || CAST(k.shortName AS VARCHAR)) AS kernel_name,
                   r.correlationId
            FROM src.{kernel_table} k
            JOIN src.{runtime_table} r ON r.correlationId = k.correlationId
            LEFT JOIN src.StringIds ks ON k.shortName = ks.id
            LEFT JOIN src.StringIds kd ON k.demangledName = kd.id
        ),
        enclosing AS (
            SELECT kr.k_start, kr.k_end, kr.kernel_name,
                   kr.r_start, kr.r_end, kr.globalTid,
                   kr.correlationId,
                   {text_expr} AS nvtx_text,
                   (n."end" - n.start) AS n_dur,
                   n.start AS n_start
            FROM kr
            JOIN src.{nvtx_table} n
              ON n.globalTid = kr.globalTid
              AND n.eventType = 59
              AND n."end" > n.start
              AND n.start <= kr.r_start
              AND n."end" >= kr.r_end
            {text_join}
            WHERE {text_expr} IS NOT NULL
        ),
        grouped AS (
            SELECT
                FIRST(nvtx_text ORDER BY n_dur ASC, n_start ASC) AS nvtx_text,
                CAST(COUNT(*) - 1 AS INTEGER) AS nvtx_depth,
                string_agg(nvtx_text, ' > ' ORDER BY n_dur DESC, n_start ASC) AS nvtx_path,
                kernel_name,
                k_start,
                k_end,
                (k_end - k_start) AS k_dur_ns
            FROM enclosing
            GROUP BY k_start, k_end, globalTid, kernel_name, correlationId
        )
        SELECT nvtx_text, nvtx_depth, nvtx_path, kernel_name,
               k_start, k_end, k_dur_ns
        FROM grouped
    ) TO '{_safe_path(cache_dir / "nvtx_kernel_map.parquet")}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    try:
        db.execute(sql)
        log.info("nvtx_kernel_map built via pure SQL (IEJoin)")
    except duckdb.Error as e:
        log.warning("Pure-SQL nvtx_kernel_map failed (%s), falling back to Python sort-merge", e)
        _build_nvtx_kernel_map_python(db, src_tables, cache_dir, sqlite_path)


def _build_nvtx_kernel_map_python(
    db: duckdb.DuckDBPyConnection,
    src_tables: set[str],
    cache_dir: Path,
    sqlite_path: str,
) -> None:
    """Generate nvtx_kernel_map.parquet using Python sort-merge (fallback).

    This is the original O(N+M) per-thread sweep algorithm.  Used only when the
    pure-SQL IEJoin approach fails (e.g., schema incompatibilities).
    """
    kernel_table = _find_table(src_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
    runtime_table = _find_table(src_tables, "CUPTI_ACTIVITY_KIND_RUNTIME")
    nvtx_table = _find_table(src_tables, "NVTX_EVENTS")

    # ── Load data from attached SQLite via DuckDB ─────────────────────
    kr_rows = db.execute(f"""
        SELECT r.globalTid, r.start, r."end",
               k.start AS ks, k."end" AS ke, k.shortName
        FROM src.{kernel_table} k
        JOIN src.{runtime_table} r ON r.correlationId = k.correlationId
        ORDER BY r.globalTid, r.start
    """).fetchall()

    if not kr_rows:
        return

    has_textid = _table_has_column(db, f"src.{nvtx_table}", "textId")
    if has_textid:
        text_expr = "COALESCE(n.text, s.value)"
        text_join = "LEFT JOIN src.StringIds s ON n.textId = s.id"
    else:
        text_expr = "n.text"
        text_join = ""

    nvtx_rows = db.execute(f"""
        SELECT n.globalTid, n.start, n."end", {text_expr} AS text
        FROM src.{nvtx_table} n
        {text_join}
        WHERE n.eventType = 59 AND n."end" > n.start
        ORDER BY n.globalTid, n.start
    """).fetchall()

    # StringIds lookup
    short_name_ids = {r[5] for r in kr_rows if r[5] is not None}
    sid_map: dict = {}
    if short_name_ids:
        placeholders = ",".join(str(int(i)) for i in short_name_ids)
        sid_rows = db.execute(
            f"SELECT id, value FROM src.StringIds WHERE id IN ({placeholders})"
        ).fetchall()
        sid_map = dict(sid_rows)

    # ── Sort-merge sweep ──────────────────────────────────────────────
    from collections import defaultdict

    nvtx_by_tid: dict[int, list[tuple]] = defaultdict(list)
    for n in nvtx_rows:
        nvtx_by_tid[n[0]].append((n[1], n[2], n[3]))

    kr_by_tid: dict[int, list[tuple]] = defaultdict(list)
    for r in kr_rows:
        kr_by_tid[r[0]].append((r[1], r[2], r[3], r[4], r[5]))

    results: list[dict] = []

    for tid in kr_by_tid:
        if tid not in nvtx_by_tid:
            continue

        nvtx_list = nvtx_by_tid[tid]
        kr_by_tid[tid].sort(key=lambda x: x[0])

        nvtx_idx = 0
        open_stack: list[tuple[int, int, str]] = []

        for r_start, r_end, k_start, k_end, short_name in kr_by_tid[tid]:
            while open_stack and open_stack[-1][1] < r_start:
                open_stack.pop()
            while nvtx_idx < len(nvtx_list) and nvtx_list[nvtx_idx][0] <= r_start:
                if nvtx_list[nvtx_idx][1] >= r_start:
                    open_stack.append(nvtx_list[nvtx_idx])
                nvtx_idx += 1

            best_nvtx = None
            best_idx = -1
            for i in range(len(open_stack) - 1, -1, -1):
                ns, ne, nt = open_stack[i]
                if ns <= r_start and ne >= r_end:
                    best_nvtx = nt
                    best_idx = i
                    break

            if best_nvtx is not None:
                enclosing = [
                    e for e in open_stack[: best_idx + 1] if e[0] <= r_start and e[1] >= r_end
                ]
                results.append(
                    {
                        "nvtx_text": best_nvtx,
                        "nvtx_depth": len(enclosing) - 1,
                        "nvtx_path": " > ".join(e[2] for e in enclosing),
                        "kernel_name": sid_map.get(short_name, f"kernel_{short_name}"),
                        "k_start": k_start,
                        "k_end": k_end,
                        "k_dur_ns": k_end - k_start,
                    }
                )

    if not results:
        return

    # ── Write to Parquet via DuckDB ───────────────────────────────────
    import pyarrow as pa

    schema = pa.schema(
        [
            ("nvtx_text", pa.string()),
            ("nvtx_depth", pa.int32()),
            ("nvtx_path", pa.string()),
            ("kernel_name", pa.string()),
            ("k_start", pa.int64()),
            ("k_end", pa.int64()),
            ("k_dur_ns", pa.int64()),
        ]
    )
    arrays = [
        pa.array([r["nvtx_text"] for r in results], type=pa.string()),
        pa.array([r["nvtx_depth"] for r in results], type=pa.int32()),
        pa.array([r["nvtx_path"] for r in results], type=pa.string()),
        pa.array([r["kernel_name"] for r in results], type=pa.string()),
        pa.array([r["k_start"] for r in results], type=pa.int64()),
        pa.array([r["k_end"] for r in results], type=pa.int64()),
        pa.array([r["k_dur_ns"] for r in results], type=pa.int64()),
    ]
    table = pa.table(arrays, schema=schema)
    try:
        db.register("_nvtx_kernel_map", table)
        db.execute(f"""
            COPY _nvtx_kernel_map
            TO '{_safe_path(cache_dir / "nvtx_kernel_map.parquet")}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    finally:
        db.unregister("_nvtx_kernel_map")
        del table


def _check_cache_size(cache_dir: Path, sqlite_path: str) -> None:
    """Warn if nvtx_kernel_map.parquet is suspiciously large."""
    map_file = cache_dir / "nvtx_kernel_map.parquet"
    if not map_file.exists():
        return

    try:
        sqlite_size = os.path.getsize(sqlite_path)
        map_size = map_file.stat().st_size
        if sqlite_size > 0 and map_size > 2 * sqlite_size:
            log.warning(
                "nvtx_kernel_map.parquet is %.0fMB (%.1f× SQLite). "
                "Consider using leaf-only NVTX or --rebuild-cache.",
                map_size / 1e6,
                map_size / sqlite_size,
            )
    except OSError:
        pass


# ── Direct SQLite mode (zero-ETL fast path) ─────────────────────────


def open_direct_sqlite(sqlite_path: str) -> duckdb.DuckDBPyConnection:
    """Open DuckDB with SQLite directly attached — zero ETL latency.

    Uses DuckDB's sqlite_scanner to query the original SQLite file
    in-place.  Analytical queries on large scans are slower than cached
    Parquet, but startup is instant.  Best for:

      - First access to large profiles (>50MB)
      - One-off queries that only touch 1-2 tables
      - ``--no-cache`` mode for quick diagnostics
    """
    db = duckdb.connect()
    safe_path = str(sqlite_path).replace("'", "''")
    try:
        try:
            db.execute(f"ATTACH '{safe_path}' AS src (TYPE SQLITE, READ_ONLY)")
        except duckdb.Error:
            try:
                db.execute("DETACH src")
            except duckdb.Error:
                pass
            db.execute("SET sqlite_all_varchar = true")
            db.execute(f"ATTACH '{safe_path}' AS src (TYPE SQLITE, READ_ONLY)")

        # Create alias views so consumer SQL (which uses original table names)
        # works unchanged.  Views point through to src.<table>.
        _create_sqlite_alias_views(db)
    except Exception:
        # Ensure we don't leak the DuckDB connection on initialization failure.
        try:
            db.close()
        except Exception:
            pass
        raise

    return db


def _tc_enriched_sql(table_name: str) -> str:
    """Return SQL for a kernel table enriched with Tensor Core metrics."""
    return f"""
        SELECT k.*,
               COALESCE(d.value, s.value, 'kernel_' || CAST(k.shortName AS VARCHAR)) AS name,
               d.value AS demangled,
               CAST(CASE
                   WHEN regexp_matches(lower(COALESCE(d.value, s.value, '')), {_TC_ELIGIBLE_PATTERN})
                     OR regexp_matches(lower(COALESCE(d.value, s.value, '')), {_TC_ACTIVE_PATTERN})
                   THEN 1
                   ELSE 0
               END AS INTEGER) AS is_tc_eligible,
               CAST(CASE WHEN regexp_matches(lower(COALESCE(d.value, s.value, '')), {_TC_ACTIVE_PATTERN}) THEN 1 ELSE 0 END AS INTEGER) AS uses_tc
        FROM src."{table_name}" k
        LEFT JOIN src.StringIds s ON k.shortName = s.id
        LEFT JOIN src.StringIds d ON k.demangledName = d.id
    """


def _create_sqlite_alias_views(db: duckdb.DuckDBPyConnection) -> None:
    """Create views that alias ``src.TABLE_NAME → TABLE_NAME`` for consumer SQL."""
    _log = logging.getLogger(__name__)
    src_tables: set[str] = set()
    try:
        for row in db.execute("SHOW ALL TABLES").fetchall():
            if row[0] == "src":
                src_tables.add(row[2])
    except duckdb.Error:
        try:
            for row in db.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_catalog = 'src'"
            ).fetchall():
                src_tables.add(row[0])
        except duckdb.Error:
            _log.warning(
                "_create_sqlite_alias_views: could not discover tables in attached SQLite; direct-mode queries may fail"
            )

    if not src_tables:
        _log.warning("_create_sqlite_alias_views: no tables found in attached 'src' database")

    # Set of known TC-eligible kernel table names (from _ALIASES)
    _known_kernel_tables = {
        t.upper() for aliases in _ALIASES.values() for t in aliases if "KERNEL" in t.upper()
    }

    for table_name in src_tables:
        escaped = table_name.replace('"', '""')
        is_kernel = table_name.upper() in _known_kernel_tables
        sql = _tc_enriched_sql(escaped) if is_kernel else f'SELECT * FROM src."{escaped}"'
        try:
            db.execute(f'CREATE VIEW IF NOT EXISTS "{escaped}" AS {sql}')
        except duckdb.Error as e:
            _log.debug("Could not create alias view for %r: %s", table_name, e)

    # For any table that exists in a versioned form, also create stable views
    # for its aliases (including the unversioned name) so queries work seamlessly.
    for short_name, aliases in _ALIASES.items():
        base_name = aliases[0]
        actual_table = _find_table(src_tables, base_name)
        if actual_table:
            actual_escaped = actual_table.replace('"', '""')
            is_kernel = "KERNEL" in base_name.upper()
            sql = (
                _tc_enriched_sql(actual_escaped)
                if is_kernel
                else f'SELECT * FROM src."{actual_escaped}"'
            )
            # Create view for versioned names (e.g. CUPTI_ACTIVITY_KIND_KERNEL_V2)
            for alias in aliases:
                alias_escaped = alias.replace('"', '""')
                try:
                    db.execute(f'CREATE VIEW IF NOT EXISTS "{alias_escaped}" AS {sql}')
                except duckdb.Error:
                    pass
            # Also create a short-name view (e.g. "kernels") so skills can use FROM kernels
            short_escaped = short_name.replace('"', '""')
            try:
                db.execute(f'CREATE VIEW IF NOT EXISTS "{short_escaped}" AS {sql}')
            except duckdb.Error as e:
                _log.debug("Could not create short-name alias view %r: %s", short_name, e)
