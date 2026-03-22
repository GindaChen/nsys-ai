"""parquet_cache.py — DuckDB + Parquet cache for Nsight Systems profiles.

Accelerates repeated profile analysis by exporting key tables from the
original SQLite export into Parquet files (ZSTD-compressed), then serving
queries via DuckDB views over those Parquet files.

Flow:
  1. First open: ``build_cache()`` attaches the SQLite DB via DuckDB,
     exports tables to ``.nsys-cache/`` as Parquet, and runs the Tier 2
     sort-merge to produce ``nvtx_kernel_map.parquet``.
  2. Subsequent opens: ``open_cached_db()`` creates a DuckDB connection
     with views pointing at the cached Parquet files — sub-second startup.

Cache invalidation uses mtime comparison + a version stamp file.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)

# Bump this when the cache schema changes (e.g., new columns, new tables).
_CACHE_VERSION = 1

# Tables to export as-is from SQLite → Parquet.
# (view_name, source_table_name)
_BASE_TABLES = [
    ("runtime", "CUPTI_ACTIVITY_KIND_RUNTIME"),
    ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY"),
    ("memset", "CUPTI_ACTIVITY_KIND_MEMSET"),
    ("string_ids", "StringIds"),
    ("gpu_info", "TARGET_INFO_GPU"),
    ("cuda_device", "TARGET_INFO_CUDA_DEVICE"),
    ("thread_names", "ThreadNames"),
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

    # Quick sanity: at least one core Parquet (e.g., string_ids) must exist
    if not (cache_dir / "string_ids.parquet").exists():
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
    tmp_dir = Path(tempfile.mkdtemp(
        prefix=".parquet_build_", dir=cache_dir.parent,
    ))
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
    if cache_dir.exists():
        cache_dir.rename(old_dir)
    tmp_dir.rename(cache_dir)
    # Clean up the old cache (now renamed aside).
    if old_dir.exists():
        shutil.rmtree(old_dir, ignore_errors=True)
    return cache_dir


def _build_cache_into(sqlite_path: str, cache_dir: Path) -> Path:
    """Internal: build the Parquet cache into the given directory."""

    log.info("Building analysis cache (first run only)...")
    t0 = time.monotonic()

    db = duckdb.connect()

    # Attach the SQLite database
    safe_sqlite_path = str(sqlite_path).replace("'", "''")
    try:
        db.execute(f"ATTACH '{safe_sqlite_path}' AS src (TYPE SQLITE)")
    except duckdb.Error:
        # Clean up partial attach before retry with permissive typing
        try:
            db.execute("DETACH src")
        except duckdb.Error:
            pass
        db.execute("SET sqlite_all_varchar = true")
        db.execute(f"ATTACH '{safe_sqlite_path}' AS src (TYPE SQLITE)")

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

    # ── Export pre-joined kernels table ────────────────────────────────
    kernel_table = _find_table(src_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
    if kernel_table:
        db.execute(f"""
            COPY (
                SELECT k.*, s.value AS name, d.value AS demangled
                FROM src.{kernel_table} k
                LEFT JOIN src.StringIds s ON k.shortName = s.id
                LEFT JOIN src.StringIds d ON k.demangledName = d.id
            ) TO '{cache_dir}/kernels.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

    # ── Export NVTX with resolved text ────────────────────────────────
    nvtx_table = _find_table(src_tables, "NVTX_EVENTS")
    if nvtx_table:
        # Detect whether textId column exists
        has_textid = _table_has_column(db, f"src.{nvtx_table}", "textId")
        if has_textid:
            db.execute(f"""
                COPY (
                    SELECT n.globalTid, n.start, n."end", n.eventType, n.rangeId,
                           COALESCE(n.text, s.value) AS text,
                           n.textId
                    FROM src.{nvtx_table} n
                    LEFT JOIN src.StringIds s ON n.textId = s.id
                ) TO '{cache_dir}/nvtx.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
        else:
            db.execute(f"""
                COPY (
                    SELECT n.globalTid, n.start, n."end", n.eventType, n.rangeId,
                           n.text
                    FROM src.{nvtx_table} n
                ) TO '{cache_dir}/nvtx.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
            """)

    for view_name, src_name in _BASE_TABLES:
        actual = _find_table(src_tables, src_name)
        if actual:
            db.execute(f"""
                COPY src.{actual}
                TO '{cache_dir}/{view_name}.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
            """)

    # ── Generate nvtx_kernel_map via Tier 2 sort-merge ────────────────
    _build_nvtx_kernel_map(db, src_tables, cache_dir, sqlite_path)

    # ── Write version stamp ───────────────────────────────────────────
    (cache_dir / ".cache_version").write_text(
        json.dumps({"version": _CACHE_VERSION, "source": os.path.basename(sqlite_path)})
    )

    # ── Size report ───────────────────────────────────────────────────
    total_bytes = sum(f.stat().st_size for f in cache_dir.iterdir() if f.is_file())
    elapsed = time.monotonic() - t0
    log.info(
        "Cache ready: %s/ (%.0fMB, %.1fs)",
        cache_dir.name, total_bytes / 1e6, elapsed,
    )

    _check_cache_size(cache_dir, sqlite_path)

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
    db = duckdb.connect()

    # Create views over Parquet files
    for parquet_file in cache_dir.glob("*.parquet"):
        view_name = parquet_file.stem
        safe_fpath = str(parquet_file).replace("'", "''")
        db.execute(
            f'CREATE VIEW "{view_name}" AS SELECT * FROM \'{safe_fpath}\''
        )

    # ── Create alias views for SQLite table names ─────────────────────
    # Consumer code uses original SQLite table names in SQL queries.
    # These aliases let those queries work unchanged over DuckDB.
    _ALIASES = {
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
        "thread_names": ["ThreadNames"],
    }

    existing_views = {r[0] for r in db.execute("SHOW TABLES").fetchall()}
    for parquet_name, aliases in _ALIASES.items():
        if parquet_name not in existing_views:
            continue
        for alias in aliases:
            if alias not in existing_views:
                try:
                    db.execute(
                        f'CREATE VIEW "{alias}" AS SELECT * FROM {parquet_name}'
                    )
                except duckdb.Error:
                    pass  # View already exists or name conflict

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


def _build_nvtx_kernel_map(
    db: duckdb.DuckDBPyConnection,
    src_tables: set[str],
    cache_dir: Path,
    sqlite_path: str,
) -> None:
    """Generate nvtx_kernel_map.parquet using Python Tier 2 sort-merge.

    Reads data from the DuckDB-attached SQLite (not from Parquet — avoids
    the bootstrap problem where Parquet files are still being built).
    """
    kernel_table = _find_table(src_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
    runtime_table = _find_table(src_tables, "CUPTI_ACTIVITY_KIND_RUNTIME")
    nvtx_table = _find_table(src_tables, "NVTX_EVENTS")

    if not all([kernel_table, runtime_table, nvtx_table]):
        log.info("Skipping nvtx_kernel_map: missing required tables")
        return

    # ── Load data from attached SQLite via DuckDB ─────────────────────
    # Kernel → Runtime join
    kr_rows = db.execute(f"""
        SELECT r.globalTid, r.start, r."end",
               k.start AS ks, k."end" AS ke, k.shortName
        FROM src.{kernel_table} k
        JOIN src.{runtime_table} r ON r.correlationId = k.correlationId
        ORDER BY r.globalTid, r.start
    """).fetchall()

    if not kr_rows:
        return

    # NVTX ranges
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

    # ── Sort-merge sweep (replicates nvtx_attribution._sort_merge_attribute) ──
    from collections import defaultdict

    nvtx_by_tid: dict[int, list[tuple]] = defaultdict(list)
    for n in nvtx_rows:
        nvtx_by_tid[n[0]].append((n[1], n[2], n[3]))  # start, end, text

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
                    e for e in open_stack[: best_idx + 1]
                    if e[0] <= r_start and e[1] >= r_end
                ]
                results.append({
                    "nvtx_text": best_nvtx,
                    "nvtx_depth": len(enclosing) - 1,
                    "nvtx_path": " > ".join(e[2] for e in enclosing),
                    "kernel_name": sid_map.get(short_name, f"kernel_{short_name}"),
                    "k_start": k_start,
                    "k_end": k_end,
                    "k_dur_ns": k_end - k_start,
                })

    if not results:
        return

    # ── Write to Parquet via DuckDB ───────────────────────────────────
    # Use DuckDB's native Python relation registration for zero-copy
    # bulk transfer — much faster than executemany INSERT for large mappings.
    import pyarrow as pa

    schema = pa.schema([
        ("nvtx_text", pa.string()),
        ("nvtx_depth", pa.int32()),
        ("nvtx_path", pa.string()),
        ("kernel_name", pa.string()),
        ("k_start", pa.int64()),
        ("k_end", pa.int64()),
        ("k_dur_ns", pa.int64()),
    ])
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
            TO '{cache_dir}/nvtx_kernel_map.parquet'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    finally:
        db.unregister("_nvtx_kernel_map")
        del table  # Allow Python GC to free Arrow memory


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
