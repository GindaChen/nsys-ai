"""parquet_cache.py — DuckDB + Parquet cache for Nsight Systems profiles.

Accelerates repeated profile analysis by exporting key tables from the
original SQLite export into Parquet files (ZSTD-compressed), then serving
queries via DuckDB views over those Parquet files.

Flow:
  1. First open: ``build_cache()`` attaches the SQLite DB via DuckDB and
     exports tables into a sibling cache directory named
     ``<profile_basename>.nsys-cache`` (e.g., ``profile.nsys-cache``) as
     Parquet.
  2. Subsequent opens: ``open_cached_db()`` creates a DuckDB connection
     with views pointing at the cached Parquet files in that
     ``<profile_basename>.nsys-cache`` directory — sub-second startup.
  3. First NVTX-attribution query: ``ensure_nvtx_kernel_map()`` runs the
     Tier 2 stack sweep to produce ``nvtx_kernel_map.parquet`` +
     ``nvtx_path_dict.parquet`` *into the existing cache*, then creates views
     over them. Every later open picks them up from the ``*.parquet`` glob in
     step 2, so the sweep runs once per profile, not once per process.

Why the map is not built in step 1: it is the single most expensive artifact in
the cache and most runs never read it. Measured on this project's reference
captures, ``nvtx_kernel_map`` is 11.6s of an 19.8s build (881 MB profile) and
59.9s of a 93.2s build (3.5 GB profile), while only four skills consume it. An
``overlap_breakdown`` run on the 881 MB capture went 19.9s → 8.8s end to end by
deferring it, and peak RSS on the 3.5 GB capture fell from 7.01 GB to 5.97 GB.
The 13 base tables are the opposite case — 8.6s of that 93.2s, wanted by nearly
every skill — so they stay eager and are deliberately not split up.

Cache invalidation uses mtime comparison + a version stamp file — the stamp's
mtime dates the step 1 build, and step 3 rewrites that stamp with its timestamps
preserved so publishing the map can never revalidate a cache that has since gone
stale against its source. A cache with
the map and one without are both legal, complete caches: every consumer probes
for the map rather than assuming it, which is why deferring it does not need a
``_CACHE_VERSION`` bump. "Not built yet" and "built wrong" stay distinct on
disk — the map is published by ``os.replace`` from a staging directory, so a
half-written one is never visible, and a cache that cannot be built at all is
still discarded whole by ``build_cache`` rather than published partial.

Environment (large profiles / DuckDB tuning):
  By default ``nvtx_kernel_map`` is deferred to first use, as described above.

  ``NSYS_AI_ALWAYS_BUILD_NVTX_KERNEL_MAP=1`` / ``NSYS_AI_DEFER_NVTX_KERNEL_MAP=0`` —
  build the map during ``build_cache`` instead, so ``cache ready`` means every
  artifact is on disk (useful for a warm-the-cache batch job).

  ``NSYS_AI_DEFER_NVTX_KERNEL_MAP_MB=<float>`` — defer only when the SQLite file
  is ≥ N MB; below the threshold the map is built eagerly. Overrides the default.

  ``NSYS_AI_DUCKDB_THREADS`` — optional ``SET threads = N`` for analytical sessions.

  ``NSYS_AI_DUCKDB_TEMP_DIRECTORY`` — optional spill directory for large aggregations
  (DuckDB ``temp_directory`` pragma).

  ``NSYS_AI_CACHE_MODE=direct|parquet`` — read by ``open_auto_db``, i.e. by every
  caller that does not ask for a specific mode. ``direct`` skips the build and
  queries the SQLite export in place (instant open, slower queries); ``parquet``
  forces the build even when the affordability check would have declined it.
  Every entry point honours it, including ``skill run`` — which is the point of
  the variable, since the TUI, the timeline, the web server and the one-shot
  subcommands expose no cache flag at all. ``skill run`` is the only one that
  also takes ``--no-cache``.

  Note it is consulted only when no valid cache exists yet:
  ``direct`` skips *building* one, it does not refuse an existing one.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from collections.abc import Callable, Iterator
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

if TYPE_CHECKING:  # pyarrow is imported lazily at the one call site that needs it
    import pyarrow as pa

from nsys_ai.connection import cache_dir_for_connection, register_cache_dir, resolve_table_variant
from nsys_ai.exceptions import ProfileNotFoundError

# fcntl is POSIX-only. On Windows we degrade to no-locking — concurrent
# builders may then race redundantly, but ``build_cache``'s tmp_dir +
# atomic rename still keeps the cache consistent.
try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


def _require_profile_exists(path: str) -> None:
    """Raise before opening so a missing path can't create an empty stub/cache."""
    if not os.path.exists(path):
        raise ProfileNotFoundError(f"profile not found: {path}")


@contextlib.contextmanager
def _build_lock(cache_dir: Path) -> Iterator[None]:
    """Serialize cache builds for one profile across threads and processes.

    Used by :func:`build_cache` to guarantee that two concurrent callers
    against the same profile (e.g. ``nsys-ai`` running in two terminals)
    do *not* both run the full ETL — the second waits on the lock, then
    re-checks :func:`is_cache_valid` inside the critical section and
    returns immediately when the first has already published the cache.

    The lock file lives at ``<cache_dir>.build.lock`` next to the cache.
    ``fcntl.flock`` is associated with the file descriptor, so the lock
    is released automatically when the fd closes (including on process
    crash).
    """
    if _fcntl is None:
        # Windows: no flock. Two builders may run redundantly, but the
        # tmp_dir + atomic rename below still keeps the cache consistent.
        yield
        return
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir.parent / f"{cache_dir.name}.build.lock"
    # ``flock`` does not require a writable fd, so open the lock
    # ``O_RDONLY``: a second user without write permission on a lock
    # file created by the first user (default ``0o644``) can still
    # acquire and release it. ``O_CLOEXEC`` prevents the fd leaking
    # into any subprocess we spawn during the build.
    fd = os.open(lock_path, os.O_RDONLY | os.O_CREAT | os.O_CLOEXEC, 0o644)
    try:
        _fcntl.flock(fd, _fcntl.LOCK_EX)  # blocks until acquired
        yield
    finally:
        # flock is held on the fd — closing releases it.
        os.close(fd)

# Bump this when the cache schema changes (e.g., new columns, new tables).
_CACHE_VERSION = 17  # bumped: the Tensor Core name patterns now recognise cuBLAS's Hopper
# ``nvjet`` GEMMs. is_tc_eligible/uses_tc are computed at build time and stored in
# kernels.parquet, so a version 16 cache keeps answering that every nvjet kernel is neither
# eligible nor active -- the exact blind spot this fixes -- until it is rebuilt.

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


# Headroom the affordability check demands, as a fraction of the SQLite size.
# Measured cache/profile ratios (re-measured 2026-08-05 after #322; unchanged from
# the 2026-08-03 figures, because deferring nvtx_kernel_map moves *when* it is
# written, not whether):
#   92.7 MB -> 16.7 MB (0.18), 235 MB -> 21.8 MB (0.093),
#   924 MB -> 84.1 MB (0.091), 3734 MB -> 276.6 MB (0.074).
# 0.25 is 1.4x the worst of those and 3.4x the large-profile ratio.
#
# Peak disk is *not* simply the final size, though it is close enough to sit
# inside that margin. build_cache writes into a sibling temp dir and renames, so
# the export itself never doubles; but since #322 the deferred map is staged in a
# .nvtx_map_build_* directory *inside* the published cache dir before os.replace,
# so peak = final + one copy of the map. On the 93 MB capture that is 3.3 MB of
# 16.7 MB — the cache is 13.4 MB until an NVTX skill runs, measured — i.e. ~20%
# of a ratio that already has 40% of slack above it.
_CACHE_SIZE_FRACTION = 0.25
_CACHE_MIN_FREE_BYTES = 128 * 1024 * 1024


def cache_is_affordable(sqlite_path: str) -> tuple[bool, str]:
    """Can a Parquet cache be built alongside this profile?

    Returns ``(True, "")`` when the cache directory's parent is writable and has
    room for the estimated cache, else ``(False, reason)`` with a reason short
    enough to log verbatim.

    This is the test ``open_auto_db`` uses in place of the old "profile is
    larger than 50 MB" rule. Size is the wrong axis: build cost and query
    speedup both scale with the profile, so a cold build comes out cheaper than
    direct on the very first run at every size from 93 MB to 3.7 GB (see the
    table in ``open_auto_db``). What actually stops a build is having nowhere to
    put it, which is what this measures.

    Estimates are deliberately generous — see ``_CACHE_SIZE_FRACTION``. A build
    that overruns the estimate anyway fails inside ``build_cache``, which
    discards its temp dir, and the caller falls back to direct SQLite.
    """
    cache_dir = _cache_dir_for(sqlite_path)
    parent = cache_dir.parent
    if not os.access(parent, os.W_OK):
        return False, f"{parent} is not writable"
    try:
        needed = max(int(os.path.getsize(sqlite_path) * _CACHE_SIZE_FRACTION), _CACHE_MIN_FREE_BYTES)
    except OSError as exc:
        return False, f"size of {sqlite_path} could not be determined ({exc})"
    try:
        free = shutil.disk_usage(parent).free
    except OSError as exc:
        return False, f"free space on {parent} could not be determined ({exc})"
    if free < needed:
        return False, f"{parent} has {free / 1e6:.0f}MB free, cache needs ~{needed / 1e6:.0f}MB"
    return True, ""


def is_cache_valid(sqlite_path: str) -> bool:
    """Check whether the Parquet cache is up-to-date.

    Returns True if:
      - The cache directory exists
      - The version stamp matches ``_CACHE_VERSION``
      - The cache is at least as new as the SQLite file (mtime comparison)

    The freshness test reads no cache contents: ``.cache_version``'s own mtime
    *is* the token, standing in for "when was this cache built". Every writer of
    that file must therefore preserve its timestamps unless it is publishing a
    genuinely new build — a writer that merely edits the JSON (see
    :func:`_publish_stamp_map_ready`) and lets ``os.replace`` date the result
    "now" revalidates a cache this function had correctly rejected, and the
    stale Parquet is then served silently.
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
    cache_dir = _cache_dir_for(sqlite_path)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        log.info("Removed cache: %s", cache_dir)


def build_cache(
    sqlite_path: str,
    *,
    env_escape: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Build a Parquet cache from a SQLite profile (first-run ETL).

    Attaches the SQLite DB via DuckDB and exports the base tables to Parquet
    with ZSTD compression. It does **not** build ``nvtx_kernel_map.parquet``:
    that one is derived, only four skills read it, and it is materialised on
    first use instead (see :func:`materialize_cached_nvtx_kernel_map`). A cache
    with the map and one without are both complete — consumers probe for it
    rather than assuming it.

    Concurrency: serialized by :func:`_build_lock` across threads and
    processes operating on the same profile, with a double-checked
    :func:`is_cache_valid` inside the critical section — when a second
    caller waits out the first, it returns the freshly-built cache
    without re-running the ETL.

    The lock is *only* acquired when a build is actually needed; an
    initial ``is_cache_valid`` fast-path return avoids opening the lock
    file at all when the cache is already good. This matters for
    read-only profile directories (NFS / read-only mounts) where the
    cache is readable but the lock file cannot be created.

    ``env_escape`` is passed straight to :func:`_build_banner` and only picks
    which escape hatch that banner names; it changes nothing about the build.

    ``progress``, when supplied, is called as ``progress(label, step, total)``
    after each export step and suppresses all build writes to ``sys.stderr``
    (banner, ``\\r`` redraws, and the final "Cache ready" line). Callers that
    are not a terminal — notably a Textual app, whose ``sys.stderr.isatty()``
    is unconditionally True — must pass one; without it the tty gate alone
    cannot tell them apart from a real CLI tty.

    Returns the cache directory path.
    """
    import tempfile

    cache_dir = _cache_dir_for(sqlite_path)
    # Fast path: cache already valid — no lock file creation needed.
    # Lets read-only profile directories work as long as the cache was
    # built earlier (e.g. by a previous writable session).
    if is_cache_valid(sqlite_path):
        return cache_dir
    with _build_lock(cache_dir):
        # Double-checked locking: another process may have built the
        # cache while we were waiting on the lock. Skip redundant ETL.
        if is_cache_valid(sqlite_path):
            return cache_dir

        # Build into a temp dir first, then atomically rename to avoid race
        # conditions when multiple threads/processes open the same profile.
        tmp_dir = Path(
            tempfile.mkdtemp(
                prefix=".parquet_build_",
                dir=cache_dir.parent,
            )
        )
        try:
            _build_cache_into(
                sqlite_path, tmp_dir, env_escape=env_escape, progress=progress
            )
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
            # Belt-and-braces: the build lock already prevents this race
            # for cooperating callers, but the tmp_dir + rename dance
            # is preserved for older readers that bypass the lock.
            shutil.rmtree(tmp_dir, ignore_errors=True)
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
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION": 'start, "end", deviceId, globalPid, syncType',
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION_V2": 'start, "end", deviceId, globalPid, syncType',
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION_V3": 'start, "end", deviceId, globalPid, syncType',
    "ENUM_CUPTI_SYNC_TYPE": "id, name",
}

# Kernel-name heuristics, and they go stale whenever a vendor renames a family.
# ``nvjet`` is what recent cuBLAS calls its Hopper GEMMs -- names like
# ``nvjet_tst_128x128_64x4_1x2_h_bz_coopA`` and ``nvjet_sm90_hss_320x128``. It
# matched neither pattern, so on an H100/H200 capture the family that dominates
# the profile scored is_tc_eligible = 0 and was dropped by tensor_core_usage's
# own ``WHERE is_tc_eligible = 1`` before any analysis saw it. It is in both
# patterns rather than only the first because these are warpgroup (wgmma)
# kernels: eligible-only would report every Hopper GEMM as a Tensor Core
# fallback, which is a louder wrong answer than the silence it replaced.
_TC_ELIGIBLE_PATTERN = "'(gemm|conv|linear|attention|matmul|flash|nvjet)'"
_TC_ACTIVE_PATTERN = (
    "'(xmma|mma_sync|16816|1688|884|ampere_bf16|sm80_tensor_op|tensorop|flash|nvjet)'"
)


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
    # NOTE: `nvtx_high` is NOT aliased here on purpose. It is a filtered subset
    # of `nvtx` (aten::* / cudaLaunch% / cudaMemcpy% removed). When the cache
    # provides nvtx_high.parquet, the view is created directly via the parquet
    # glob loop in open_cached_db(). Skills should probe for its existence and
    # fall back to `nvtx` when absent — never let _create_existing_alias_views
    # silently turn `nvtx_high` into a slow scan over the full table.
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


def _configure_duckdb_analytics_session(db: duckdb.DuckDBPyConnection) -> None:
    """Apply DuckDB session settings from the performance guide (large scans/joins).

    See: https://duckdb.org/docs/current/guides/performance/how_to_tune_workloads.html
    """
    try:
        db.execute("SET preserve_insertion_order = false")
    except duckdb.Error:
        pass
    try:
        db.execute("SET enable_progress_bar = false")
    except duckdb.Error:
        pass
    # Cache Parquet file metadata across queries within this session — every
    # skill that reads the same parquet pays the metadata-parse cost only once.
    # https://duckdb.org/docs/current/configuration/pragmas
    try:
        db.execute("PRAGMA enable_object_cache")
    except duckdb.Error:
        pass
    raw = os.environ.get("NSYS_AI_DUCKDB_THREADS", "").strip()
    if raw:
        try:
            n = int(raw)
            if n > 0:
                db.execute(f"SET threads = {n}")
        except (ValueError, duckdb.Error):
            pass
    # Spill directory for large GROUP BY / joins (see DuckDB "temp_directory" pragma).
    tmp = os.environ.get("NSYS_AI_DUCKDB_TEMP_DIRECTORY", "").strip()
    if tmp:
        try:
            os.makedirs(tmp, exist_ok=True)
            safe = tmp.replace("'", "''")
            db.execute(f"SET temp_directory = '{safe}'")
        except (OSError, duckdb.Error):
            pass


def _should_defer_nvtx_kernel_map(sqlite_path: str) -> bool:
    """Return True when nvtx_kernel_map should be skipped on first cache build.

    Default is **defer**. The map is the most expensive artifact in the cache
    and the least widely read (see module docstring for the measurements), and
    deferring it no longer costs anything on the NVTX side: the on-demand build
    now writes the same Parquet into the same cache directory, so the sweep runs
    once per profile either way. What changes is *when* — and a run that never
    touches NVTX attribution never pays for it.

    What it does cost is *discoverability*, for as long as the map is unbuilt.
    Before this, ``build_cache`` wrote nvtx_kernel_map.parquet and
    ``open_cached_db``'s glob turned it into a view, so the map was in the
    catalog from the moment a connection was handed out. Now, on a cache whose
    map has not been built yet, ``SHOW TABLES`` and ``information_schema`` list
    neither name and a ``SELECT`` against either raises a Catalog Error
    (measured on tests/fixtures/h100_2gpu_1s.sqlite, both before and after).
    The skill system does not notice — both consumers call
    ``ensure_nvtx_kernel_map`` before they probe — but the text-to-SQL surface
    does: ``ai/backend/profile_db_tool.py`` rewrites ``sqlite_master`` to
    ``SHOW TABLES`` and hands the result to the model, and the
    ``schema_inspect`` skill reads ``information_schema.columns``. On a cold
    cache the model therefore sees no map and writes raw nvtx/kernels/runtime
    joins instead of using it — slower, not wrong, and nothing in the prompt
    names the map. Once anything builds it the views appear on that connection,
    and every later process opening that cache sees them at open.
    ``test_the_map_is_absent_from_schema_discovery_until_it_is_built`` pins
    both ends of that window.

    ``NSYS_AI_ALWAYS_BUILD_NVTX_KERNEL_MAP=1`` (or ``NSYS_AI_DEFER_NVTX_KERNEL_MAP=0``)
    restores the eager build for callers that want ``cache ready`` to mean every
    artifact is present. ``NSYS_AI_DEFER_NVTX_KERNEL_MAP_MB`` makes the choice
    size-dependent, so small profiles — where the sweep is a fraction of a
    second — can stay eager.
    """
    env_always = os.environ.get("NSYS_AI_ALWAYS_BUILD_NVTX_KERNEL_MAP", "").strip().lower()
    if env_always in ("1", "true", "yes", "on"):
        return False
    env_defer = os.environ.get("NSYS_AI_DEFER_NVTX_KERNEL_MAP", "").strip().lower()
    if env_defer in ("0", "false", "no", "off"):
        return False

    raw_mb = os.environ.get("NSYS_AI_DEFER_NVTX_KERNEL_MAP_MB", "").strip()
    if raw_mb:
        try:
            threshold_mb = float(raw_mb)
        except ValueError:
            log.warning("Ignoring invalid NSYS_AI_DEFER_NVTX_KERNEL_MAP_MB=%r", raw_mb)
            return True
        try:
            size_mb = os.path.getsize(sqlite_path) / 1e6
        except OSError:
            return True
        return size_mb >= threshold_mb

    return True


def _order_clause_for(
    view_name: str,
    db: duckdb.DuckDBPyConnection,
    table_ref: str,
    has_device_col: bool,
) -> str:
    """Return an ORDER BY clause that lets DuckDB zonemaps prune time / device filters.

    Returns an empty string for tables where ordering doesn't help (small lookup
    tables, payload schemas, etc.).

    Each ordering is chosen so the leading column is the most common filter:
      - deviceId-scoped tables → ORDER BY deviceId, start
      - per-thread runtime/nvtx → ORDER BY globalTid, start
      - time-only event streams → ORDER BY start

    All ordering keys are wrapped in ``TRY_CAST(... AS BIGINT)``: the SQLite
    attach in ``_build_cache_into`` falls back to ``sqlite_all_varchar=true``
    on some Nsight schemas, which would make these numeric columns VARCHAR
    and turn ORDER BY into lexicographic sort (``'9999'`` after ``'10000'``)
    — defeating the zonemap intent. ``TRY_CAST`` is a no-op when the column
    is already integer-typed.
    """
    plan: dict[str, list[str]] = {
        "runtime": ["globalTid", "start"],
        "memcpy": ["deviceId", "start"],
        "memset": ["deviceId", "start"],
        "sync": ["deviceId", "start"] if has_device_col else ["start"],
        "overhead": ["start"],
        "profiler_overhead": ["start"],
        "composite_events": ["start"],
    }
    cols = plan.get(view_name)
    if not cols:
        return ""
    # Drop any column that doesn't exist on this particular nsys export variant.
    present = [c for c in cols if _table_has_column(db, table_ref, c)]
    if not present:
        return ""
    cast = ", ".join(f'TRY_CAST("{c}" AS BIGINT)' for c in present)
    return f"ORDER BY {cast}"


# Build throughput measured on this repo's reference captures (2026-08-05, 12 cores):
# 50 MB/s at 92.7 MB, 89 at 235 MB, 110 at 924 MB, 136 at 3734 MB. Throughput climbs with
# size because a small build is dominated by fixed setup cost.
#
# Fitted to the >=500 MB end deliberately: _build_banner is the only consumer and it does
# not fire below _BUILD_BANNER_MIN_MB, so the 50 and 89 MB/s rows are outside the range
# this constant is ever asked about. 110 predicts 8 s at 924 MB (actual 8.4) and 34 s at
# 3734 MB (actual 27.4) — i.e. it errs towards over-stating the wait, which is the right
# direction for an ETA.
#
# These figures postdate #322, which moved nvtx_kernel_map out of the build and onto
# first use. Any throughput number measured before that split described a build that also
# produced the map and is 2-3x too pessimistic; re-measure rather than reconciling.
_BUILD_MB_PER_S = 110.0
# Below this the build is a few seconds and an ETA banner is just noise.
_BUILD_BANNER_MIN_MB = 500.0
# Throughput of the *deferred* nvtx_kernel_map sweep, which since #322 is no longer part
# of the build and is paid by the first NVTX-attributing query instead. Measured the same
# day, as cold-minus-warm time for nvtx_layer_breakdown: 11.2 s at 924 MB (82 MB/s) and
# 49.7 s at 3734 MB (75 MB/s). Only the banner's size range matters here and it is tight
# across it, so one constant is honest. Used solely to say how long the *second* wait is;
# a profile with no NVTX data never pays it at all.
_MAP_BUILD_MB_PER_S = 80.0


def _stderr_is_tty() -> bool:
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, ValueError):
        return False


def _build_banner(sqlite_path: str, *, env_escape: bool = True) -> None:
    """Announce a long build before it starts, with the way out.

    The per-step progress line already exists, but it says nothing until the
    first step completes and nothing about how long the whole thing will take.
    On a 3.7 GB profile that is ~27 s of a user not knowing whether to wait.

    The ETA covers *this* build only, which since #322 is the Parquet export
    and not the ``nvtx_kernel_map`` sweep — that is deferred to the first NVTX
    query and costs a further 3.7 / 3.4 / 11.2 / 49.7 s at 93 MB / 235 MB /
    924 MB / 3.7 GB. Naming the second wait is the point of the third line:
    without it the banner promises ~34 s on the 3.7 GB capture and the user
    then sits through another ~50 s of silence inside a skill, because
    ``materialize_cached_nvtx_kernel_map`` prints nothing at all. Instrumenting
    that build is the real fix and belongs with issue #300; saying it out loud
    is the floor.

    ``env_escape`` picks which way out the last line names, because the two
    callers do not have the same one. Reached through ``open_auto_db`` — every
    entry point in this project — ``NSYS_AI_CACHE_MODE=direct`` genuinely skips
    the build, including from its own ``=parquet`` branch. Reached through
    ``open_cached_db`` directly, i.e. ``Profile(cache_mode="parquet")``, the
    variable is never consulted and printing it would be advice that does
    nothing; there the way out is the argument, so that is what it says.
    Advertising an inert escape hatch is the exact defect #317 set out to fix
    on ``skill run`` — narrowing it to one path is not fixing it.
    """
    try:
        size_mb = os.path.getsize(sqlite_path) / 1e6
    except OSError:
        return
    if size_mb < _BUILD_BANNER_MIN_MB:
        return
    escape = "Set NSYS_AI_CACHE_MODE=direct" if env_escape else 'Pass cache_mode="direct"'
    # "4-5x" rather than the 3-9x range the docstring quotes: this banner only
    # fires at >=500 MB, and in that region direct-vs-warm-cached query time
    # measured 3.9x (924 MB) and 5.1x (3.7 GB). Quoting the whole-range figure
    # here would promise the small-profile speedup to the one audience that
    # cannot get it.
    sys.stderr.write(
        f"[nsys-ai] Building analysis cache for a {size_mb:.0f}MB profile "
        f"(~{size_mb / _BUILD_MB_PER_S:.0f}s, once per profile; "
        f"queries afterwards are 4-5x faster).\n"
        f"[nsys-ai] The first NVTX-attributing query then builds the kernel map "
        f"(~{size_mb / _MAP_BUILD_MB_PER_S:.0f}s more, also once per profile).\n"
        f"[nsys-ai] {escape} to skip the build and query "
        f"the SQLite export in place.\n"
    )
    sys.stderr.flush()


def _build_cache_into(
    sqlite_path: str,
    cache_dir: Path,
    *,
    env_escape: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Internal: build the Parquet cache into the given directory."""

    log.info("Building analysis cache (first run only)...")
    # A supplied progress callback owns reporting; do not also paint stderr.
    if progress is None:
        _build_banner(sqlite_path, env_escape=env_escape)
    t0 = time.monotonic()

    db = duckdb.connect()
    try:
        _configure_duckdb_analytics_session(db)

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
        defer_nvtx_map = has_kernel and has_nvtx and has_runtime and _should_defer_nvtx_kernel_map(
            sqlite_path
        )
        if has_kernel and has_nvtx and has_runtime and not defer_nvtx_map:
            total_steps += 1
        step = [0]

        label = [""]

        # \r redraws are only legible on a terminal; piped or redirected, they
        # arrive as one long line of fragments. Off a tty the banner and the
        # final "Cache ready" line carry the whole story.
        #
        # When a progress callback is supplied it owns reporting and stderr
        # stays quiet — required inside Textual, where sys.stderr.isatty() is
        # True unconditionally (#332).
        #
        # Single-threaded by construction: every caller is the build's own
        # thread, between DuckDB statements, so no write+flush pair can
        # interleave with another on the same fd.
        def _draw() -> None:
            if not _stderr_is_tty():
                return
            elapsed = time.monotonic() - t0
            sys.stderr.write(
                f"\r[nsys-ai] Building cache [{step[0]}/{total_steps}] "
                f"{label[0]} ({elapsed:.0f}s)" + " " * 8
            )
            sys.stderr.flush()

        def _progress(name: str) -> None:
            step[0] += 1
            label[0] = name
            if progress is not None:
                progress(name, step[0], total_steps)
            else:
                _draw()

        # ── Export pre-joined kernels table ────────────────────────────────
        # ORDER BY (deviceId, start) is critical: it lets DuckDB's parquet
        # zonemaps prune row groups for both `-p device=N` filters and
        # `--trim S E` time-range filters. Without explicit ordering,
        # row-group min/max ranges overlap and pruning is ineffective.
        # See: https://duckdb.org/docs/current/guides/performance/indexing
        # TRY_CAST guards against the `sqlite_all_varchar=true` attach
        # fallback, which would otherwise give lexicographic ordering.
        kernel_table = _find_table(src_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        if kernel_table is None:
            # The source may carry kernel activity under a name this builder
            # cannot address. kernels.parquet would then be missing while the
            # stamp still claimed a complete cache — and every later open would
            # revalidate that poisoned cache and fail in NsightSchema. Raise
            # instead: build_cache discards the temp dir on any exception, so
            # nothing is published and the caller's except-clause falls back to
            # direct SQLite. That fallback is degraded, not at parity — it loses
            # the tensor-core flags and the demangled kernel names the cache
            # precomputes — but it reads the profile.
            #
            # RuntimeError, not SchemaError: callers that fall back catch
            # (duckdb.Error, RuntimeError, OSError), and SchemaError descends
            # from Exception, not from RuntimeError.
            #
            # This guard only stops a *new* poisoned cache from being published.
            # build_cache's is_cache_valid fast-path returns before we get here,
            # so an already-poisoned cache dir is never repaired. _CACHE_VERSION
            # is deliberately not bumped for that: poisoning requires a kernel
            # table name outside the exact/`_V` forms _find_table matches, which
            # no shipped Nsight export uses, so no cache in the field can be in
            # that state. Deleting the .nsys-cache directory recovers one.
            unrecognised = _kernel_like_tables(src_tables)
            if unrecognised:
                raise RuntimeError(
                    "cache build aborted: the profile carries a kernel activity table "
                    f"({', '.join(unrecognised)}) that this build does not recognise, "
                    "so kernels.parquet cannot be produced"
                )
        else:
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
                    ORDER BY TRY_CAST(k.deviceId AS BIGINT), TRY_CAST(k.start AS BIGINT)
                ) TO '{_safe_path(cache_dir / "kernels.parquet")}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """)

        # ── Export NVTX with resolved text ────────────────────────────────
        nvtx_table = _find_table(src_tables, "NVTX_EVENTS")
        if nvtx_table:
            _progress("nvtx.parquet")
            _export_nvtx_with_blobs(sqlite_path, nvtx_table, cache_dir)
            # nvtx_high.parquet: aten::%/cudaLaunch%/cudaMemcpy% filtered out.
            # ~95% of NVTX rows are aten:: on typical PyTorch traces, so this
            # ~20× smaller table is what most NVTX-attribution skills should
            # read. Skipping is harmless — skills probe for the view and fall
            # back to the full `nvtx` view when absent.
            try:
                _build_nvtx_high(db, cache_dir)
            except duckdb.Error as exc:
                log.warning(
                    "nvtx_high.parquet build failed (%s); NVTX skills will use "
                    "full nvtx.parquet (slower).",
                    exc,
                )

        for view_name, src_name in _BASE_TABLES:
            actual = _find_table(src_tables, src_name)
            if actual:
                _progress(f"{view_name}.parquet")
                projection = _TABLE_PROJECTIONS.get(actual, "*")
                # Legacy Nsight exports may lack `deviceId` on synchronization
                # tables. Drop it from the projection rather than failing the
                # cache build — sync_cost_analysis degrades to single-device mode.
                has_device_col = True
                if (
                    projection != "*"
                    and actual.startswith("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION")
                    and not _table_has_column(db, f"src.{actual}", "deviceId")
                ):
                    projection = projection.replace("deviceId, ", "")
                    has_device_col = False
                order_clause = _order_clause_for(view_name, db, f"src.{actual}", has_device_col)
                if projection == "*":
                    if order_clause:
                        db.execute(f"""
                            COPY (SELECT * FROM src.{actual} {order_clause})
                            TO '{_safe_path(cache_dir / f"{view_name}.parquet")}' (FORMAT PARQUET, COMPRESSION ZSTD)
                        """)
                    else:
                        db.execute(f"""
                            COPY src.{actual}
                            TO '{_safe_path(cache_dir / f"{view_name}.parquet")}' (FORMAT PARQUET, COMPRESSION ZSTD)
                        """)
                else:
                    db.execute(f"""
                        COPY (SELECT {projection} FROM src.{actual} {order_clause})
                        TO '{_safe_path(cache_dir / f"{view_name}.parquet")}' (FORMAT PARQUET, COMPRESSION ZSTD)
                    """)

        # ── Generate nvtx_kernel_map ──────────────────────────────────────
        if has_kernel and has_nvtx and has_runtime and not defer_nvtx_map:
            _progress("nvtx_kernel_map.parquet")
            _build_nvtx_kernel_map_from_parquet(db, cache_dir)
        elif defer_nvtx_map:
            # Not a degraded cache: the first NVTX-attribution query calls
            # ensure_nvtx_kernel_map, which builds the same two Parquets into
            # this same directory and publishes them atomically. Every other
            # skill skips a build step it would never have read.
            log.info(
                "Deferring nvtx_kernel_map to first NVTX query "
                "(set NSYS_AI_ALWAYS_BUILD_NVTX_KERNEL_MAP=1 to build it now)"
            )

        # The CR and the padding exist only to overwrite the progress line
        # _draw left on screen, so they belong to the same condition _draw does.
        # Emitted unconditionally they put a bare CR and 40 spaces into every
        # piped or redirected run — litter in exactly the case the tty gate was
        # added to clean up. A progress callback replaces the whole stderr
        # channel, so skip this line too.
        elapsed = time.monotonic() - t0
        if progress is None:
            ready = f"[nsys-ai] Cache ready ({elapsed:.1f}s)"
            if _stderr_is_tty():
                sys.stderr.write("\r" + ready + " " * 40 + "\n")
            else:
                sys.stderr.write(ready + "\n")
            sys.stderr.flush()

        # ── Write version stamp ───────────────────────────────────────────
        meta = {
            "version": _CACHE_VERSION,
            "source": os.path.basename(sqlite_path),
            "empty": len(src_tables) == 0 or not _find_table(src_tables, "StringIds"),
            "nvtx_kernel_map_ready": (cache_dir / "nvtx_kernel_map.parquet").exists(),
            "deferred_nvtx_kernel_map": bool(defer_nvtx_map),
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


def open_cached_db(
    sqlite_path: str,
    *,
    env_escape: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with views over the Parquet cache.

    If the cache doesn't exist or is stale, builds it first.

    ``env_escape=False`` says the caller reached here without consulting
    ``NSYS_AI_CACHE_MODE``, so the build banner must not advertise it. Only
    :class:`~nsys_ai.profile.Profile` with an explicit ``cache_mode="parquet"``
    passes it; ``open_auto_db`` leaves it at the default because on that path
    the variable really does work.

    ``progress`` is forwarded to :func:`build_cache` when a build is needed.

    Returns a DuckDB connection with views named after each cached table:
      ``kernels``, ``nvtx``, ``runtime``, ``memcpy``, ``memset``,
      ``string_ids``, ``gpu_info``, ``cuda_device``.

    ``nvtx_kernel_map`` and ``nvtx_path_dict`` are deliberately absent: the map
    is built on first use, so a connection from a cache that has not needed it
    yet carries neither view. Probe for them rather than assuming them —
    :func:`ensure_nvtx_kernel_map` is the supported way to ask.
    """
    _require_profile_exists(sqlite_path)
    if not is_cache_valid(sqlite_path):
        build_cache(sqlite_path, env_escape=env_escape, progress=progress)

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
    _configure_duckdb_analytics_session(db)

    # Create views over Parquet files
    for parquet_file in cache_dir.glob("*.parquet"):
        view_name = parquet_file.stem
        safe_fpath = str(parquet_file).replace("'", "''")
        db.execute(f"CREATE VIEW \"{view_name}\" AS SELECT * FROM '{safe_fpath}'")

    _create_existing_alias_views(db)

    # Tell the connection where its cache lives, so a later
    # ``ensure_nvtx_kernel_map`` can publish the map into it rather than
    # rebuilding it in memory once per process. Only this opener registers a
    # directory: direct-attach and parquetdir connections have nowhere to write.
    register_cache_dir(db, cache_dir)

    return db


def open_auto_db(
    sqlite_path: str,
    *,
    progress: Callable[[str, int, int], None] | None = None,
) -> duckdb.DuckDBPyConnection:
    """Open a profile under the ``auto`` cache policy: cache if one can be had.

    Order: an existing valid cache, then the ``NSYS_AI_CACHE_MODE`` override,
    then "can a cache be written here at all", then build.

    ``progress`` is forwarded to :func:`open_cached_db` / :func:`build_cache`
    when a build runs.

    It lives here rather than on ``Profile`` because it is not only
    ``Profile``'s decision. ``nsys-ai skill run`` and ``open_profile_readonly``
    (chat, region_mfu) open a connection with no ``Profile`` involved, and they
    used to call ``open_cached_db`` directly — so the build banner advertised
    ``NSYS_AI_CACHE_MODE=direct`` on a subcommand where setting it changed
    nothing. One policy, one function, one place to change it.

    Raises whatever ``open_cached_db`` and ``open_direct_sqlite`` raise; a
    caller that wants a raw-``sqlite3`` last resort catches it itself.

    Why the default is to build
    ---------------------------

    This used to send any profile over 50 MB to ``open_direct_sqlite``. That
    was correct when it was written — the old interval-join map builder ran
    past fifteen minutes on a 3.5 GB capture and OOM'd a 175 GB pod (#143) —
    and it stopped being correct when the stack sweep replaced that join.

    Measured on this machine (12 cores, 15 GB RAM, DuckDB 1.5.0, one process
    per row, cache deleted before every cold run, 2026-08-05) over the
    **eight-skill battery** named in ``scripts/bench_cache.py`` as
    ``AUTO_POLICY_SKILLS`` — top_kernels, gpu_idle_gaps, overlap_breakdown,
    memory_transfers, kernel_launch_overhead, stream_concurrency,
    tensor_core_usage, nvtx_layer_breakdown. The basket is part of the result:
    a different one moves these numbers a lot (see the light workload below),
    so quoting them without it is not reproducible.

    =========  =======  ==============  ========  =========  =======  ========
    profile    SQLite   mode            open      query      total    peak RSS
    =========  =======  ==============  ========  =========  =======  ========
    nano_vllm    93MB   direct           0.75 s     5.75 s    6.50 s   1.03 GB
                        parquet (cold)   1.86 s     4.31 s    6.17 s   0.96 GB
                        parquet (warm)   0.04 s     0.63 s    0.67 s   0.33 GB
    perf_sp1    235MB   direct           1.36 s    11.21 s   12.57 s   1.11 GB
                        parquet (cold)   2.64 s     7.07 s    9.71 s   1.09 GB
                        parquet (warm)   0.04 s     3.72 s    3.76 s   0.77 GB
    mfu_l40s2   924MB   direct           4.68 s    33.17 s   37.85 s   2.67 GB
                        parquet (cold)   8.37 s    19.66 s   28.03 s   2.12 GB
                        parquet (warm)   0.05 s     8.49 s    8.54 s   1.39 GB
    perf       3734MB   direct          12.75 s    83.43 s   96.18 s  10.18 GB
                        parquet (cold)  27.44 s    66.13 s   93.57 s   7.14 GB
                        parquet (warm)   0.06 s    16.27 s   16.33 s   1.78 GB
    =========  =======  ==============  ========  =========  =======  ========

    Read the cold row's ``query`` column carefully: since #322 the build no
    longer produces ``nvtx_kernel_map``, so part of what used to be ``open``
    now sits inside the first NVTX-attributing skill. Cold ``query`` minus warm
    ``query`` for ``nvtx_layer_breakdown`` alone is 3.7 / 3.4 / 11.2 / 49.7 s
    at the four sizes. The total is what the user experiences, which is why the
    table carries one.

    Total against total, a cold build is never *worse* than direct on the first
    run, and never heavier on peak RSS over this battery. So the argument for
    building by default is not "always faster on the first run" -- it is "never
    slower, sometimes much faster, and materially lighter on the large
    profiles", plus the warm row, which is the one a user meets on every command
    after the first.

    An earlier version of this docstring quoted a "runs-to-repay" ratio derived
    from the open/query split, which the #322 deferral made meaningless because
    the cold query now contains a build.
    Note also that the margins at 93 MB (6.17 vs 6.50) and 3734 MB (93.57 vs
    96.18) are 3-5%, and run-to-run spread on a loaded box is 15-25% — those
    two sizes are a wash on *time*, not a win. The decisive gaps are in the
    middle. What the two ends win instead is memory, and at 3734 MB decisively:
    7.14 GB against 10.18 GB, a 30% cut on the size where running out is a real
    outcome.

    None of this is a property of the profile size: both sides of the trade
    scale together, which is exactly why size was the wrong axis. What does
    discriminate is how much querying follows, and every consumer of this
    project bar a one-shot skill is multi-query — the direct path re-pays its
    cost on every later run forever, a build is paid once (warm total is 0.67 /
    3.76 / 8.54 / 16.33 s). Two cases where that reasoning does not hold:

    * **One cheap skill and nothing else.** On the 3.7 GB capture that is
      ~19 s direct (12.75 open + 5.84 for ``top_kernels``) against ~27 s with a
      build (27.44 + 0.04). Direct still wins, but by 1.5x — before the #322
      deferral this comparison was 18.7 s against 79.2 s, so the advice is much
      weaker than it used to be.

      Say plainly which callers those are, because the eight-skill basket did
      not measure them and they inherit this default anyway: nothing in this
      project passes ``cache_mode``, so the TUI, the timeline, the web server
      and the one-shot subcommands (``info``, ``kernels``, ``export``) all land
      on ``auto``. A TUI that used to open a 3.7 GB profile in ~13 s now waits
      ~27 s at startup. That is a real regression for a one-shot user and a
      real win for anyone who then queries, and the trade was chosen for the
      latter. ``_build_banner`` is the mitigation, not a refutation: at
      >=500 MB it says the wait is coming and how to opt out. If someone wants
      this reversed for the interactive entry points, the honest fix is to
      measure a TUI-shaped basket and let those callers pass ``cache_mode``,
      not to put the size rule back.
    * **A light workload on a small profile, when RAM is tight.** The same
      93 MB capture queried with four cheap skills that never touch NVTX
      attribution (``top_kernels``, ``gpu_idle_gaps``, ``memory_transfers``,
      ``kernel_launch_overhead``) is 0.73 + 1.65 s at 0.36 GB peak direct,
      against 1.79 + 0.56 s at 0.74 GB cold. Wall clock is a dead heat
      (2.38 vs 2.34 s) and the build costs twice the peak RSS, because its
      working set is a fixed price a light run never earns back. So do *not*
      read the table above as an unconditional peak-RSS win; issue #300 cares
      about that ceiling.

    Both are what ``NSYS_AI_CACHE_MODE=direct``, ``cache_mode="direct"`` and
    ``skill run --no-cache`` are for.

    If those numbers no longer hold, re-measure before nudging anything:
    ``scripts/bench_cache.py --basket auto-policy`` is the harness and
    reproduces this basket. Its *default* basket is a different, twelve-skill
    one, which will not reproduce these rows.
    """
    if is_cache_valid(sqlite_path):
        return open_cached_db(sqlite_path, progress=progress)

    env_mode = os.environ.get("NSYS_AI_CACHE_MODE", "").strip().lower()
    if env_mode == "direct":
        log.info("NSYS_AI_CACHE_MODE=direct — querying the SQLite export in place.")
        return open_direct_sqlite(sqlite_path)
    if env_mode == "parquet":
        return open_cached_db(sqlite_path, progress=progress)
    if env_mode:
        log.warning("Ignoring NSYS_AI_CACHE_MODE=%r; expected 'direct' or 'parquet'.", env_mode)

    affordable, reason = cache_is_affordable(sqlite_path)
    if not affordable:
        # WARNING, not INFO, and the level is the whole mechanism. Nothing under
        # src/nsys_ai/ ever calls basicConfig/dictConfig/addHandler, so the only
        # sink for this record is logging.lastResort, a StreamHandler pinned at
        # WARNING. At INFO the line was dropped: measured on a profile in a
        # chmod 500 directory, `nsys-ai skill run top_kernels` wrote 2059 bytes
        # to stdout and *zero* to stderr while silently taking the 3-9x slower
        # path. Raising the level makes the same command print it.
        #
        # It also deserves the level on its merits: this is a silently degraded
        # run, not a routine note. The `NSYS_AI_CACHE_MODE=direct` branch above
        # stays at INFO on purpose — there the user asked for it, so silence is
        # the correct outcome.
        log.warning(
            "Not building an analysis cache (%s); querying the SQLite export directly. "
            "Queries will be several times slower.",
            reason,
        )
        return open_direct_sqlite(sqlite_path)
    return open_cached_db(sqlite_path, progress=progress)


def open_with_direct_fallback(
    sqlite_path: str,
    primary,
    *,
    log: logging.Logger | None = None,
) -> tuple[duckdb.DuckDBPyConnection | None, BaseException | None]:
    """Open via ``primary``, then ``open_direct_sqlite``, then give up.

    Shared three-tier policy used by ``Profile.__init__``,
    ``open_profile_readonly`` (chat / region_mfu), and ``skill run``. Losing
    the Parquet cache must not cost the DuckDB engine: a failed primary open
    falls back to a direct SQLite attach (enriched ``kernels`` view intact)
    and reaches raw ``sqlite3`` only when that fails too.

    Returns ``(conn, primary_error)``:
      - primary succeeded → ``(duckdb_conn, None)``
      - primary failed, direct attach recovered → ``(duckdb_conn, primary_error)``
      - both DuckDB paths failed → ``(None, primary_error)``; caller opens
        raw ``sqlite3`` (read-only or not is the caller's choice)

    ``ProfileNotFoundError`` is re-raised: a missing file cannot be salvaged.
    ``open_direct_sqlite`` attaches ``READ_ONLY``, so the middle tier stays
    compatible with read-only callers.
    """
    _log = log if log is not None else logging.getLogger(__name__)
    try:
        return primary(sqlite_path), None
    except ProfileNotFoundError:
        raise
    except Exception as e:
        try:
            db = open_direct_sqlite(sqlite_path)
            _log.warning(
                "Parquet cache unavailable (%s); querying the SQLite export directly "
                "through DuckDB.",
                e,
            )
            return db, e
        except Exception as direct_err:
            _log.warning(
                "DuckDB unavailable (cache: %s; direct: %s), falling back to sqlite3.",
                e,
                direct_err,
            )
            return None, e


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
        _configure_duckdb_analytics_session(db)
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
    """Find the actual table name, handling versioned variants (e.g., _V2).

    Shares ``resolve_table_variant``'s ordering with the query-side resolver in
    ``connection.py`` so the cached and uncached engines cannot pick different
    source tables on a profile that carries several variants. Unlike that
    resolver this one stays strict about ``_V<n>``: it decides which table the
    ETL *copies*, so an unrecognised suffix should leave the entry uncached
    rather than cache a guess under the canonical name. (Both sides now reject a
    bare trailing digit — ``CUPTI_ACTIVITY_KIND_MEMCPY2`` is peer-to-peer
    memcpy, not a memcpy version — so the two differ only on suffixes neither
    has met, such as a hypothetical ``..._NEXTGEN``.)
    """
    return resolve_table_variant(tables, prefix)


def _kernel_like_tables(tables: set[str]) -> list[str]:
    """Source tables ``NsightSchema`` would accept as the kernel activity table.

    Mirrors the *set* ``NsightSchema._detect_kernel_table`` will accept: any
    non-``ENUM_`` table with ``KERNEL`` in its name. Which one of them that
    function picks is the shared resolver's business and not mirrored here —
    every name the resolver can return starts with
    ``CUPTI_ACTIVITY_KIND_KERNEL``, so it is a subset of this list either way.

    Used to tell "this profile has no kernel data at all" (a legitimate state —
    Nsight creates tables lazily) from "this profile has kernel data under a
    name ``_find_table`` did not match", which must not be cached.
    """
    return sorted(t for t in tables if "KERNEL" in t.upper() and not t.upper().startswith("ENUM_"))


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
    _create_enriched_kernel_view(db, existing_views)
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


def _create_enriched_kernel_view(
    db: duckdb.DuckDBPyConnection,
    existing_views: set[str],
) -> None:
    """Expose the cache-compatible ``kernels`` view for raw parquetdir data.

    The SQLite-to-Parquet cache builder writes an enriched ``kernels.parquet``
    containing resolved names and Tensor Core classification. A raw Nsight
    parquetdir export only contains the source activity table and string-id
    references. Keep that raw table untouched, but make the stable ``kernels``
    alias provide the same columns so skills can use either backend.
    """
    if "kernels" in existing_views or "StringIds" not in existing_views:
        return
    kernel_table = _find_table(
        existing_views,
        "CUPTI_ACTIVITY_KIND_KERNEL",
    )
    if kernel_table is None:
        return

    try:
        db.execute(
            f'''
            CREATE VIEW "kernels" AS
            SELECT k.*,
                   COALESCE(d.value, s.value, 'kernel_' || CAST(k.shortName AS VARCHAR)) AS name,
                   d.value AS demangled,
                   CAST(CASE
                       WHEN regexp_matches(lower(COALESCE(d.value, s.value, '')), {_TC_ELIGIBLE_PATTERN})
                         OR regexp_matches(lower(COALESCE(d.value, s.value, '')), {_TC_ACTIVE_PATTERN})
                       THEN 1 ELSE 0
                   END AS INTEGER) AS is_tc_eligible,
                   CAST(CASE
                       WHEN regexp_matches(lower(COALESCE(d.value, s.value, '')), {_TC_ACTIVE_PATTERN})
                       THEN 1 ELSE 0
                   END AS INTEGER) AS uses_tc
            FROM "{kernel_table}" k
            LEFT JOIN "StringIds" s ON k.shortName = s.id
            LEFT JOIN "StringIds" d ON k.demangledName = d.id
            '''
        )
        existing_views.add("kernels")
    except duckdb.Error:
        # A raw export with an incomplete kernel/string-id schema should still
        # open; the normal schema/skill diagnostics will report the missing
        # capability instead of making parquetdir opening fail here.
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
        _configure_duckdb_analytics_session(db)
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
        # NVTX is most commonly filtered by (globalTid, time-window). Ordering
        # by (globalTid, start) lets DuckDB zonemap-prune for per-thread time
        # queries (e.g. host-sync attribution, iteration_timing).
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
                    ORDER BY TRY_CAST(n.globalTid AS BIGINT), TRY_CAST(n.start AS BIGINT)
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
                    ORDER BY TRY_CAST(n.globalTid AS BIGINT), TRY_CAST(n.start AS BIGINT)
                ) TO '{_safe_path(cache_dir / "nvtx.parquet")}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
    finally:
        db.close()


# Prefix patterns that mark NVTX rows as "op-level noise" — kept out of
# nvtx_high.parquet. The filter is intentionally narrow: anything *not*
# matching these prefixes is preserved, so framework-, layer-, and
# distributed-comm-level ranges (stage::*, FSDP::*, FlashAttn*, AllToAll4D,
# nccl:*, transformer_blocks/*, record_param_comms, …) all survive.
#
# Skills that need aten::item / aten::_local_scalar_dense (e.g.
# host_sync_parent_ranges, §5.7 Step 1) must query the full `nvtx` view —
# never `nvtx_high`.
#
# Patterns are inlined as SQL string literals in _build_nvtx_high(); the
# assertion below blocks any future addition that would break that
# splicing or open an injection seam.
_NVTX_HIGH_EXCLUDE_PATTERNS: tuple[str, ...] = (
    "aten::%",
    "cudaLaunch%",
    "cudaMemcpy%",
)
# Raise (not assert) so the check survives `python -O`; asserts get stripped.
_bad_patterns = [p for p in _NVTX_HIGH_EXCLUDE_PATTERNS if "'" in p or "\\" in p]
if _bad_patterns:
    raise ValueError(
        "nvtx_high exclusion patterns must not contain single quotes or "
        f"backslashes (offending: {_bad_patterns}). The patterns are inlined "
        "as SQL string literals in _build_nvtx_high()."
    )
del _bad_patterns


def _build_nvtx_high(db: duckdb.DuckDBPyConnection, cache_dir: Path) -> None:
    """Write nvtx_high.parquet — nvtx.parquet minus op-level noise.

    On profiles where ~95% of NVTX events are ``aten::*`` (typical PyTorch
    traces), this shrinks the input set for ``nvtx_layer_breakdown``'s slow
    path — still a real range join — by ~20×. The ``nvtx_kernel_map`` build is
    a stack sweep and deliberately reads the unfiltered nvtx.parquet instead.

    Preserves the same (globalTid, start) sort as nvtx.parquet so DuckDB
    zonemaps stay effective.
    """
    src = cache_dir / "nvtx.parquet"
    dst = cache_dir / "nvtx_high.parquet"
    if not src.is_file():
        return

    src_path = _safe_path(src)
    dst_path = _safe_path(dst)
    # Build the filter as a single NOT LIKE chain so DuckDB can push it down
    # into the Parquet scan (no full materialisation before the filter).
    exclusions = " AND ".join(
        f"text NOT LIKE '{pat}'" for pat in _NVTX_HIGH_EXCLUDE_PATTERNS
    )
    db.execute(f"""
        COPY (
            SELECT *
            FROM read_parquet('{src_path}')
            WHERE text IS NOT NULL AND {exclusions}
            ORDER BY globalTid, start
        ) TO '{dst_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)


# Rows per Arrow batch on each of the two input streams, and rows buffered
# before one batch is written to the staged Parquet. One number for all three
# because they are the same trade — the size of the window the sweep holds in
# Python — and because a test that shrinks it exercises every batch boundary at
# once (tests/test_parquet_cache.py::TestStreamedSweep).
_SWEEP_BATCH_ROWS = 262_144


def _build_nvtx_kernel_map_from_parquet(
    db: duckdb.DuckDBPyConnection,
    cache_dir: Path,
    out_dir: Path | None = None,
) -> bool:
    """Generate nvtx_kernel_map.parquet with a per-thread stack sweep.

    There is one builder and one path. Each kernel is attributed to its
    innermost enclosing NVTX push/pop range by ``_attribute_thread``; no SQL
    range join is involved (the comment below records why one was tried and
    dropped), and there is no fallback.

    **One thread of the capture at a time**, so what the sweep holds in Python
    tracks the batch rather than the profile. ``eventType = 59`` is a per-thread
    Push/Pop stack, so a thread's ranges are strictly nested and its work is
    independent of every other thread's: partitioning on ``globalTid`` needs no
    carried state at a partition boundary at all. Within a thread both sides
    then arrive sorted by start and are consumed through a single advancing
    cursor, so neither side is ever a list. Measured on a 3.5 GB capture
    (15.9 M ranges, 3.1 M kernel-runtime rows), peak process RSS: 5.28 GB ->
    1.39 GB inside the sweep, 7.50 GB -> 3.55 GB for the whole build, with
    unchanged output (3,042,699 rows). The 3.55 GB is the publish step below,
    which still sorts the whole map in DuckDB; that is #339.

    Sources are the ``kernels``, ``runtime`` and ``nvtx`` Parquets already in
    ``cache_dir`` — written earlier in the same build, or published by an
    earlier one when the on-demand caller runs, which has no attached SQLite at
    all. Either way the heavy read never scans the SQLite ``NVTX_EVENTS`` table.

    ``out_dir`` defaults to ``cache_dir``; the on-demand caller points it at a
    staging directory so the two Parquets can be published with ``os.replace``
    instead of appearing half-written under a concurrent reader's glob.

    Returns True when a map was written. False means the sweep found nothing to
    attribute or a source was missing — both are logged skips, and neither is an
    error.

    A source that exists but cannot be *read* is neither: the ``read_parquet``
    error propagates out of this function, and what happens next is the caller's,
    because the two callers are in different situations.

    From ``_build_cache_into`` it propagates on to ``build_cache``, which deletes
    the half-built temp dir rather than publishing it. That is what every export
    of a primary source in this build does; the one export that swallows its error,
    ``_build_nvtx_high``, is a derived convenience view that callers already probe
    for and fall back from, which is precisely what these three sources are not.
    They are shared with the rest of the cache, so an unreadable one is not a
    reason to drop the map and keep the cache, it is a reason to have no cache.
    ``Profile.__init__`` then logs and falls back to raw SQLite; the profile stays
    readable, just without Parquet acceleration.

    On the on-demand path there is no half-built cache to discard — the cache is
    already published and in use — and nothing here converts the error into a
    False return. ``materialize_cached_nvtx_kernel_map`` catches ``OSError`` only,
    so a ``duckdb.Error`` escapes it and ``ensure_nvtx_kernel_map``, and both call
    sites (``nvtx_layer_breakdown`` and ``nvtx_attribution``) swallow it with
    ``except DB_ERRORS: pass``. The query then takes its no-map fallback and the
    cache is kept. Deliberate to the extent that a published cache is not this
    function's to invalidate, but it does mean a corrupt ``nvtx.parquet`` is loud
    at build time and silent afterwards.
    """
    out_dir = cache_dir if out_dir is None else out_dir
    kp = cache_dir / "kernels.parquet"
    rp = cache_dir / "runtime.parquet"
    np = cache_dir / "nvtx.parquet"
    # The sweep's NVTX source must be the full nvtx.parquet, NOT nvtx_high.parquet.
    # nvtx_kernel_map is the canonical "kernel → enclosing NVTX range" mapping
    # used by attribute_kernels_to_nvtx() and downstream skills via the fast
    # path. Using a filtered source would silently drop kernels whose only
    # enclosing ranges are aten::* — e.g. emit_nvtx-style PyTorch traces with
    # no higher-level wrappers — and downstream callers won't fall back to
    # full nvtx when nvtx_kernel_map exists but is empty/incomplete.
    # The slow path in nvtx_layer_breakdown still benefits from nvtx_high
    # (see _pick_nvtx_view in that file).
    # All three are written unconditionally earlier in this same build, under
    # the very predicates (has_kernel/has_nvtx/has_runtime) that gate the call
    # to this function, and none of those writes is wrapped in a try/except — a
    # failure there aborts the whole build. So this branch cannot normally fire;
    # it exists so a future reordering degrades to a logged skip rather than a
    # duckdb.Error out of read_parquet.
    if not (kp.is_file() and rp.is_file() and np.is_file()):
        log.warning("nvtx_kernel_map: parquet sources missing; skipping map")
        return False

    kps = _safe_path(kp)
    rps = _safe_path(rp)
    nps = _safe_path(np)

    # nvtx.parquet already stores resolved text (export path); no StringIds join.

    # Attribution is a stack sweep, not a general inequality join.
    #
    # NVIDIA documents eventType 59 as a Push/Pop range maintaining an
    # nvtxRange stack per thread, so the ranges on a thread are strictly nested
    # by construction. A containment IEJoin cannot know that and pays the
    # general-case cost: on a 3.5 GB capture (2.19 M kernels, 15.9 M ranges) it
    # ran over twenty minutes, saturating twelve cores, to produce a 3.0 M-row
    # result. The sweep below does the same work in 35 s on that capture, and
    # its output was compared row-for-row against the IEJoin's -- 3,042,699
    # rows, zero differences in either direction.
    #
    # The equality key is why the join had so little to work with: on the NVTX
    # side globalTid has five distinct values, so IEJoin partitioned into five
    # buckets and range-joined millions against millions inside each. The same
    # skew is what makes the thread a usable *chunk* here: four of those five
    # threads hold 99.97% of the 15.9 M ranges in near-equal quarters (measured
    # on the 3.5 GB capture, from nvtx.parquet).
    #
    # The partition list below is taken from the runtime side, which on that
    # capture has 51 distinct globalTid -- 12 of which launched kernels, and 4
    # of which also carry push/pop ranges. So most partitions attribute
    # nothing and still issue two filtered scans. Only one of the two prunes
    # cheaply: nvtx.parquet is written sorted by globalTid, so its row-group
    # statistics narrow the scan to the matching groups. The kernel side does
    # not -- kernels.parquet carries no globalTid column at all, so the filter
    # applies to runtime.parquet and the correlationId join still reads the
    # kernel rows it pairs with. Either way the count of scans is linear in
    # thread count, which is what would bite. A capture with several
    # hundred CUDA-calling threads would want this list narrowed to the threads
    # present on both sides.
    #
    # Sources stay Parquet-only, as before, so the heavy read never touches the
    # attached SQLite.
    #
    # On errors: the partition query below is eager, so an unreadable
    # runtime.parquet raises there, before anything is staged. Everything after
    # it goes through the two stream helpers, which are generator functions, so
    # their ``execute`` runs on first advance inside ``_attribute_thread`` --
    # under the ``try`` whose ``finally`` removes the staged file. Either way a
    # duckdb.Error leaves this function and no map is published, which is what
    # tests/test_parquet_cache.py pins. The ``try`` blocks below are there to
    # release the writer, the cursors and the staged file, and none of them
    # swallows an error; adding one that did would be a change to a documented
    # behaviour.
    import pyarrow.parquet as pq

    # The partitions are the threads that launched work, so this reads the
    # runtime side: a thread with ranges and no kernels attributes nothing and
    # does not need a partition. A NULL globalTid is excluded rather than
    # partitioned on, because a per-thread nesting stack over rows carrying no
    # thread id is not defined. This is the one place the two builders can
    # disagree: ``_sweep_nvtx_kernel_map``, which the on-demand builder still
    # uses, buckets those rows under a single ``None`` key and attributes them
    # to each other, which is a made-up answer rather than a better one. No
    # export seen here has the case at all -- the column is the thread that
    # made the CUDA API call, and runtime.parquet on the 3.5 GB reference
    # capture has zero NULLs in it.
    tids = [
        row[0]
        for row in db.execute(
            f"SELECT DISTINCT globalTid FROM read_parquet('{rps}') "
            f"WHERE globalTid IS NOT NULL ORDER BY 1"
        ).fetchall()
    ]

    # A DuckDB connection holds ONE result. Both streams are live at the same
    # time — the NVTX reader keeps a batch of lookahead while the kernel reader
    # advances — and opening the second on the same handle invalidates the
    # first, after which the sweep just stops early with no exception at all.
    # Measured on the 3.5 GB capture: 968,394 rows instead of 3,042,699. Same
    # failure class as the shared-connection bugs in #300/#301, and equally
    # silent, so each stream gets its own cursor.
    nvtx_cur = db.cursor()
    kernel_cur = db.cursor()

    # Staged first, published second. The sweep cannot know the path dictionary
    # until it has seen every row, and the map's ``path_id`` column is defined
    # against that dictionary — so the nine columns land in a scratch Parquet
    # carrying the path text itself, and the two published files are derived
    # from it in SQL. It also means a build that dies part-way leaves no
    # readable nvtx_kernel_map.parquet behind.
    staged = out_dir / f".nvtx_map_sweep_{os.getpid()}.parquet"
    schema = _staged_map_schema()
    n_written = 0
    try:
        writer = pq.ParquetWriter(staged, schema, compression="zstd")
        try:
            buf: list[tuple] = []
            for tid in tids:
                nvtx_rows = _stream_thread_nvtx(nvtx_cur, nps, tid)
                kr_rows = _stream_thread_kernels(kernel_cur, kps, rps, tid)
                try:
                    for row in _attribute_thread(kr_rows, nvtx_rows):
                        buf.append(row)
                        if len(buf) >= _SWEEP_BATCH_ROWS:
                            _write_staged_batch(writer, schema, buf)
                            n_written += len(buf)
                            buf = []
                finally:
                    # The kernel side is always drained; the NVTX side usually is
                    # not, because the last kernel on a thread rarely sits at the
                    # last range. Close it so its Arrow reader releases the
                    # cursor's result now rather than at the next collection.
                    nvtx_rows.close()
                    kr_rows.close()
            if buf:
                _write_staged_batch(writer, schema, buf)
                n_written += len(buf)
        finally:
            writer.close()

        # Covers "no kernels", "no ranges" and "no overlap" alike: all three
        # reach here having written nothing.
        if n_written == 0:
            log.info("nvtx_kernel_map produced no NVTX/kernel attribution; skipping map creation")
            return False

        _publish_nvtx_kernel_map_parquet(db, staged, out_dir)
    finally:
        nvtx_cur.close()
        kernel_cur.close()
        staged.unlink(missing_ok=True)

    log.info("nvtx_kernel_map built via streamed stack sweep (%d rows)", n_written)
    return True


def _staged_map_schema() -> pa.Schema:
    """Column order of the scratch Parquet the streamed sweep writes.

    Not the published schema: this one carries ``nvtx_path`` as text, which
    ``_publish_nvtx_kernel_map_parquet`` replaces with the ``path_id`` of the
    dictionary it derives. Field order matches what ``_attribute_thread``
    yields, so a row is written without being reordered.
    """
    import pyarrow as pa

    return pa.schema(
        [
            ("nvtx_text", pa.string()),
            ("nvtx_depth", pa.int32()),
            ("nvtx_path", pa.string()),
            ("kernel_name", pa.string()),
            ("k_start", pa.int64()),
            ("k_end", pa.int64()),
            ("k_dur_ns", pa.int64()),
            ("is_tc_eligible", pa.int32()),
            ("uses_tc", pa.int32()),
        ]
    )


def _write_staged_batch(writer, schema: pa.Schema, rows: list[tuple]) -> None:
    """Append one batch of sweep rows to the staged Parquet."""
    import pyarrow as pa

    cols = list(zip(*rows))
    writer.write_batch(
        pa.RecordBatch.from_arrays(
            [pa.array(cols[i], schema.field(i).type) for i in range(len(cols))],
            schema=schema,
        )
    )


def _publish_nvtx_kernel_map_parquet(db, staged: Path, out_dir: Path) -> None:
    """Derive nvtx_path_dict.parquet + nvtx_kernel_map.parquet from ``staged``.

    ``path_id`` is assigned by ``row_number() OVER (ORDER BY nvtx_path)``, which
    is 1-based over the distinct paths in ascending order — the same ids the
    in-memory builder's ``sorted(...)`` + ``i + 1`` produces, so a cached map and
    an on-demand one still agree. DuckDB orders VARCHAR by UTF-8 bytes and
    Python by code point, which coincide for UTF-8.

    The published map's ``ORDER BY k_start, k_end, kernel_name`` clusters it by
    kernel start, which is the axis its readers filter on. It is *not* a total
    order and does not make two builds byte-comparable: one kernel reached
    through more than one runtime row yields several rows sharing that key, and
    on the 3.5 GB reference capture 759,299 keys are shared by 1.6 M of the
    3.1 M kernel-runtime rows. What is pinned is the row multiset
    (``TestStreamedSweep`` in tests/test_parquet_cache.py), not the byte layout.
    This sort is also the step that dominates peak memory on a large capture,
    tracked separately in #339.
    """
    sps = _safe_path(staged)
    dps = _safe_path(out_dir / "nvtx_path_dict.parquet")
    db.execute(f"""
        COPY (
            SELECT row_number() OVER (ORDER BY nvtx_path) AS path_id, nvtx_path
            FROM (SELECT DISTINCT nvtx_path FROM read_parquet('{sps}'))
        ) TO '{dps}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    db.execute(f"""
        COPY (
            SELECT d.path_id, m.nvtx_text, m.nvtx_depth, m.kernel_name,
                   m.k_start, m.k_end, m.k_dur_ns, m.is_tc_eligible, m.uses_tc
            FROM read_parquet('{sps}') m
            JOIN read_parquet('{dps}') d ON d.nvtx_path = m.nvtx_path
            ORDER BY m.k_start, m.k_end, m.kernel_name
        ) TO '{_safe_path(out_dir / "nvtx_kernel_map.parquet")}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)
    """)


def _arrow_reader(cur, sql: str, params: list):
    """Open a lazily-consumed Arrow record-batch reader on ``cur``.

    ``to_arrow_reader`` is the current name; ``fetch_record_batch`` is its
    deprecated alias and the only one present on the duckdb>=1.0 floor this
    package declares. Both take the batch size as their first argument.
    """
    result = cur.execute(sql, params)
    make_reader = getattr(result, "to_arrow_reader", None) or result.fetch_record_batch
    return make_reader(_SWEEP_BATCH_ROWS)


def _stream_thread_nvtx(cur, nvtx_parquet: str, tid):
    """Yield one thread's ``(start, end, text)`` PushPop ranges, in start order.

    ``eventType = 59`` deliberately excludes Start/End ranges (eventType 60)
    rather than merely overlooking them: the consumer is a per-thread nesting
    stack, valid only for push/pop ranges. See
    ``skills.base.requires_pushpop_nvtx`` for why widening this is not the fix.

    The labels are *not* interned, unlike the kernel names below, and that is
    measured rather than an oversight: on the 3.5 GB reference capture a single
    thread carries 3,850,911 distinct texts across 3,969,967 ranges, so the pool
    dedupes almost nothing and costs 0.19 GB to hold. NVTX labels on an
    ``emit_nvtx``-style PyTorch capture carry per-iteration identifiers; the
    assumption that they repeat does not survive contact with one.
    """
    for batch in _arrow_reader(
        cur,
        f"""
        SELECT start, "end", CAST(text AS VARCHAR) AS text
        FROM read_parquet('{nvtx_parquet}')
        WHERE eventType = 59 AND "end" > start AND text IS NOT NULL
          AND globalTid = ?
        ORDER BY start
        """,
        [tid],
    ):
        starts = batch.column(0).to_pylist()
        ends = batch.column(1).to_pylist()
        texts = batch.column(2).to_pylist()
        for i, text in enumerate(texts):
            yield starts[i], ends[i], text


def _stream_thread_kernels(cur, kernels_parquet: str, runtime_parquet: str, tid):
    """Yield one thread's ``(r_start, r_end, k_start, k_end, name, elig, used)``.

    Sorted by ``r_start``, which #318 and #327 removed from this query and this
    restores — per thread, and on ``r.start`` only. Those removals were right for
    the builder they were made against: it bucketed the rows by thread and sorted
    each bucket in Python, so a SQL order was redone. A stream cannot be
    re-sorted after the fact, so under streaming that Python sort has nowhere to
    live and the ORDER BY comes back. ``_sweep_nvtx_kernel_map``, which the
    on-demand builder still uses, keeps the old contract and still sorts its own
    buckets; tests/test_determinism.py pins that.

    The Tensor Core flags ride along in the row. They used to be dropped here and
    rebuilt afterwards from a ``(k_start, k_end, name) -> (elig, used)`` dict —
    2,193,776 entries of three-tuples on the 3.5 GB capture, existing only
    because the sweep discarded two columns it was already reading.

    Kernel names *are* interned, and here the pool pays: 3,077,650 rows carry
    109 distinct names averaging 253 characters, so as separate str objects that
    is ~0.9 GB and interned it is a few kilobytes.
    """
    pool: dict[str, str] = {}
    for batch in _arrow_reader(
        cur,
        f"""
        SELECT r.start, r."end", k.start, k."end", k.name,
               COALESCE(CAST(k.is_tc_eligible AS INTEGER), 0) AS is_tc_eligible,
               COALESCE(CAST(k.uses_tc AS INTEGER), 0) AS uses_tc
        FROM read_parquet('{kernels_parquet}') k
        JOIN read_parquet('{runtime_parquet}') r ON r.correlationId = k.correlationId
        WHERE r.globalTid = ?
        ORDER BY r.start
        """,
        [tid],
    ):
        cols = [batch.column(i).to_pylist() for i in range(7)]
        r_starts, r_ends, k_starts, k_ends, names, elig, used = cols
        for i, name in enumerate(names):
            shared = pool.get(name)
            if shared is None:
                shared = pool[name] = name
            yield r_starts[i], r_ends[i], k_starts[i], k_ends[i], shared, elig[i], used[i]


def _nvtx_map_arrow_tables(results: list[dict]) -> tuple[pa.Table, pa.Table]:
    """Build the ``(nvtx_kernel_map, nvtx_path_dict)`` Arrow pair from sweep rows.

    The in-memory builder's writer, and its only caller is
    ``_materialize_nvtx_kernel_map``. The cached builder streams straight to
    Parquet instead (``_publish_nvtx_kernel_map_parquet``) and never holds the
    rows, so the two maps' columns are now two lists that have to be edited
    together. The list here and the ``SELECT`` there are compared column by
    column in ``TestStreamedSweep.test_both_map_writers_emit_the_same_columns``
    (tests/test_parquet_cache.py); nothing else catches a divergence, because
    the row-for-row oracle in that class compares the *cached* map against
    ``_sweep_nvtx_kernel_map``'s rows and never calls this function. All nine
    columns are
    emitted, including ``is_tc_eligible``/``uses_tc``: consumers probe for those
    two by presence (``connection.cached_nvtx_map_has_embedded_tc``), so a map
    missing them is not merely thinner, it routes every reader onto a different
    aggregate. Rows arriving without the flags default to 0 rather than failing.
    """
    import pyarrow as pa

    distinct_paths = sorted({r["nvtx_path"] for r in results})
    path_to_id = {path: i + 1 for i, path in enumerate(distinct_paths)}

    map_table = pa.table(
        {
            "path_id": pa.array([path_to_id[r["nvtx_path"]] for r in results], pa.int64()),
            "nvtx_text": pa.array([r["nvtx_text"] for r in results], pa.string()),
            "nvtx_depth": pa.array([r["nvtx_depth"] for r in results], pa.int32()),
            "kernel_name": pa.array([r["kernel_name"] for r in results], pa.string()),
            "k_start": pa.array([r["k_start"] for r in results], pa.int64()),
            "k_end": pa.array([r["k_end"] for r in results], pa.int64()),
            "k_dur_ns": pa.array([r["k_dur_ns"] for r in results], pa.int64()),
            "is_tc_eligible": pa.array([r.get("is_tc_eligible", 0) for r in results], pa.int32()),
            "uses_tc": pa.array([r.get("uses_tc", 0) for r in results], pa.int32()),
        }
    )
    dict_table = pa.table(
        {
            "path_id": pa.array([path_to_id[p] for p in distinct_paths], pa.int64()),
            "nvtx_path": pa.array(list(distinct_paths), pa.string()),
        }
    )
    return map_table, dict_table


def _attribute_thread(kr_rows, nvtx_rows):
    """Attribute one thread's kernels to their innermost enclosing NVTX range.

    The single containment implementation in this package. Both builders reach
    it: the cached one feeds it two Arrow-backed generators per thread, the
    on-demand one feeds it two lists per thread through
    :func:`_sweep_nvtx_kernel_map`. There is deliberately no second copy of this
    logic — a hand-vectorised rewrite of it was wrong on 15% of rows, which is
    the kind of divergence #300 spent its effort removing.

    Both arguments are consumed **in one forward pass** and neither has to be a
    list. ``eventType = 59`` ranges on a thread are strictly nested, so a stack
    plus one range of lookahead is the whole state; measured depth on a 3.5 GB
    capture is <= 7.

    kr_rows: iterable of ``(r_start, r_end, k_start, k_end, kernel_name, *extra)``
    sorted by ``r_start``. Anything after the fifth field is passed through to
    the output untouched, which is how the cached builder carries the Tensor Core
    flags without a second pass.
    nvtx_rows: iterable of ``(start, end, text)`` sorted by ``start``.

    Yields ``(nvtx_text, nvtx_depth, nvtx_path, kernel_name, k_start, k_end,
    k_dur_ns, *extra)``. Kernels with no enclosing range are dropped.
    """
    nvtx_iter = iter(nvtx_rows)
    # One range of lookahead. The rest of the NVTX side stays in Arrow.
    pending = next(nvtx_iter, None)
    open_stack: list[tuple] = []

    for row in kr_rows:
        r_start, r_end, k_start, k_end, kernel_name = row[:5]
        while open_stack and open_stack[-1][1] < r_start:
            open_stack.pop()
        while pending is not None and pending[0] <= r_start:
            if pending[1] >= r_start:
                open_stack.append(pending)
            pending = next(nvtx_iter, None)

        best_idx = -1
        for i in range(len(open_stack) - 1, -1, -1):
            if open_stack[i][0] <= r_start and open_stack[i][1] >= r_end:
                best_idx = i
                break
        if best_idx < 0:
            continue

        enclosing = [e for e in open_stack[: best_idx + 1] if e[0] <= r_start and e[1] >= r_end]
        # Total order, matching nvtx_attribution's
        # ``ORDER BY n_dur ASC, n_start ASC, nvtx_text ASC``. Stack position
        # alone is not enough: two ranges with an identical span are both
        # innermost, and which one the stack yields depends on the order they
        # arrived in, so the answer followed the input rather than the data.
        # That is the divergence tests/test_determinism.py exists to prevent.
        # Most kernels sit inside a single range, and sorting a one-element list
        # twice cost 36% of the build on a 3.5GB capture (60.6s -> 82.4s) for no
        # reordering at all.
        if len(enclosing) == 1:
            by_inner = by_outer = enclosing
        else:
            by_inner = sorted(enclosing, key=lambda e: (e[1] - e[0], e[0], e[2]))
            by_outer = sorted(enclosing, key=lambda e: (-(e[1] - e[0]), e[0], e[2]))
        yield (
            by_inner[0][2],
            len(enclosing) - 1,
            " > ".join(e[2] for e in by_outer),
            kernel_name,
            k_start,
            k_end,
            k_end - k_start,
        ) + tuple(row[5:])


def _sweep_nvtx_kernel_map(kr_rows, nvtx_rows) -> list[dict]:
    """Bucket rows by thread and attribute each bucket with
    :func:`_attribute_thread`, materialising the result as a list of dicts.

    The on-demand builder's entry point (issue #257). It exists because that
    builder has both sides in memory already — it ``fetchall``s them off a
    backend with no Parquet cache to stream from — so partitioning them here is
    the cheapest way to reach the shared containment core. The cached builder
    does not come through here at all: it partitions in SQL and streams.

    kr_rows: iterable of ``(globalTid, r_start, r_end, k_start, k_end,
    kernel_name)`` — the kernel name already resolved by the caller (so each can
    match its own map's convention), **in any order**: this function buckets
    them by thread and sorts each bucket by ``r_start`` itself.
    nvtx_rows: ``(globalTid, start, end, text)``
    (PushPop ranges), which *must* arrive sorted by ``(globalTid, start)`` —
    they are swept with a single advancing index per thread. Returns rows:
    ``{nvtx_text, nvtx_depth, nvtx_path, kernel_name, k_start, k_end, k_dur_ns}``.
    """
    from collections import defaultdict

    nvtx_by_tid: dict[int, list[tuple]] = defaultdict(list)
    for n in nvtx_rows:
        nvtx_by_tid[n[0]].append((n[1], n[2], n[3]))

    kr_by_tid: dict[int, list[tuple]] = defaultdict(list)
    for r in kr_rows:
        kr_by_tid[r[0]].append((r[1], r[2], r[3], r[4], r[5]))

    results: list[dict] = []
    for tid, kr_list in kr_by_tid.items():
        nvtx_list = nvtx_by_tid.get(tid)
        if not nvtx_list:
            continue
        kr_list.sort(key=lambda x: x[0])
        for text, depth, path, name, k_start, k_end, dur in _attribute_thread(kr_list, nvtx_list):
            results.append(
                {
                    "nvtx_text": text,
                    "nvtx_depth": depth,
                    "nvtx_path": path,
                    "kernel_name": name,
                    "k_start": k_start,
                    "k_end": k_end,
                    "k_dur_ns": dur,
                }
            )
    return results


def _materialize_nvtx_kernel_map(db, results: list[dict]) -> None:
    """Create the ``nvtx_kernel_map`` + ``nvtx_path_dict`` tables on a
    DuckDB connection from sweep ``results``. Emits the full 9-column
    cache-built schema, including the embedded ``is_tc_eligible``/``uses_tc``,
    so consumers take their map-only fast path (a TC-less map would force
    nvtx_layer_breakdown into a (start,end) kernels-join that double-counts
    timestamp-colliding kernels). The nine columns are listed in
    ``_nvtx_map_arrow_tables`` above, and again in the cached builder's
    ``_publish_nvtx_kernel_map_parquet``; the two lists are separate and
    ``TestStreamedSweep.test_both_map_writers_emit_the_same_columns`` is what
    keeps them equal."""
    map_tbl, dict_tbl = _nvtx_map_arrow_tables(results)
    db.register("_odm_nkm", map_tbl)
    db.register("_odm_npd", dict_tbl)
    try:
        # Deliberately not TEMP. A temp table is visible only to the connection
        # that created it, and DuckDB requires each thread to work through its
        # own ``.cursor()`` handle — so a temp table here leaves every worker
        # thread unable to see the map and silently falling back to the slower
        # on-the-fly attribution. These two are the shared artifacts other code
        # reads; the scratch tables above stay TEMP because they never escape
        # the function that builds them.
        db.execute("CREATE TABLE nvtx_kernel_map AS SELECT * FROM _odm_nkm")
        db.execute("CREATE TABLE nvtx_path_dict AS SELECT * FROM _odm_npd")
    finally:
        db.unregister("_odm_nkm")
        db.unregister("_odm_npd")


# Catalog DDL on a DuckDB database is not safe to race. Two threads issuing
# ``CREATE VIEW nvtx_kernel_map`` through their own ``.cursor()`` handles get a
# "Catalog write-write conflict on create" TransactionException from all but the
# winner — and ``CREATE VIEW IF NOT EXISTS`` does *not* help, because the
# conflict is raised by the transaction manager before the existence check
# matters. Measured: with four threads, three lost. So the DDL is serialized
# here in Python, and the loser of any residual race re-probes rather than
# raising.
_CATALOG_DDL_LOCK = threading.Lock()

# One process-wide lock around the map build, so N threads that all want the map
# do the sweep once rather than N times.
#
# Process-wide rather than per-database on purpose. Keying it per database needs
# a way to tell that two handles are the same database, and for a bare
# ``.cursor()`` there is none: DuckDB exposes no parent pointer, and
# ``duckdb_databases()`` names every in-memory database "memory" (see the note
# in connection.py). A registry of cursors exists — ``register_derived_handle``,
# populated by ``Profile.query_conn`` — but relying on it would leave any thread
# that made its own cursor building a private copy, which is what a per-database
# version of this lock was measured doing: four threads, four sweeps.
#
# The cost of the coarser lock is that two *different* profiles building their
# maps concurrently in one process queue up instead of overlapping. That is
# latency on a rare path, never a wrong answer, and it buys a guarantee that
# does not depend on how the caller obtained its handle.
#
# This is the in-process half. The cross-process half is ``_build_lock``'s
# flock, taken inside ``materialize_cached_nvtx_kernel_map``. Lock order is
# always this one first, then flock, so the two cannot deadlock against
# each other.
_MAP_BUILD_LOCK = threading.Lock()


def _nvtx_map_present(db) -> bool:
    """True when both map relations are queryable on ``db``."""
    try:
        db.execute("SELECT 1 FROM nvtx_kernel_map LIMIT 0")
        db.execute("SELECT 1 FROM nvtx_path_dict LIMIT 0")
        return True
    except duckdb.Error:
        return False


def _create_nvtx_map_views(db, cache_dir: Path) -> bool:
    """Point ``nvtx_kernel_map``/``nvtx_path_dict`` views at the cached Parquet.

    Returns True when both are queryable afterwards. Views, not tables: the rows
    stay on disk, so a connection that only needs a slice of the map does not
    pay to load all of it, and a view created on any handle is visible to every
    ``.cursor()`` of the same database — which is what makes this work under the
    per-thread cursors #301 introduced.
    """
    map_path = cache_dir / "nvtx_kernel_map.parquet"
    dict_path = cache_dir / "nvtx_path_dict.parquet"
    if not (map_path.is_file() and dict_path.is_file()):
        return False
    with _CATALOG_DDL_LOCK:
        for view_name, parquet_path in (
            ("nvtx_path_dict", dict_path),
            ("nvtx_kernel_map", map_path),
        ):
            try:
                db.execute(
                    f'CREATE VIEW "{view_name}" AS '
                    f"SELECT * FROM '{_safe_path(parquet_path)}'"
                )
            except duckdb.Error:
                # Already created — by open_cached_db's glob, by an earlier call
                # on this database, or by a thread that beat us to the lock.
                # Whether that is true is decided by the probe below, not here.
                pass
    return _nvtx_map_present(db)


def _publish_stamp_map_ready(cache_dir: Path) -> None:
    """Record in ``.cache_version`` that the map is now on disk.

    The two keys are informational — nothing reads them to decide behaviour.
    The *file* they live in is not: its mtime is the cache's freshness token,
    compared against the source SQLite's mtime by :func:`is_cache_valid`. This
    rewrite happens at query time, arbitrarily long after the build that the
    token is supposed to date, so an ``os.replace`` alone would stamp the token
    "now" and revalidate a cache that had correctly gone stale — the source can
    change while a cached connection is live (a re-capture to the same path),
    and every later process would then be served the old Parquet. Hence the
    ``os.utime`` restore below: the file's contents advance, its timestamps do
    not, and a stale cache stays stale. The stat and the replace both run under
    ``_build_lock`` (see :func:`materialize_cached_nvtx_kernel_map`), so a
    concurrent rebuild cannot slip between them and have its fresh stamp aged
    backwards.

    Written temp-file + ``os.replace`` because a torn stamp fails the JSON parse
    in ``is_cache_valid`` and costs a full rebuild.

    A failure here is not a build failure: the Parquet is already published and
    the next open finds it by glob regardless.
    """
    stamp = cache_dir / ".cache_version"
    try:
        meta = json.loads(stamp.read_text())
        before = os.stat(stamp)
    except (json.JSONDecodeError, OSError):
        return
    meta["nvtx_kernel_map_ready"] = True
    meta["deferred_nvtx_kernel_map"] = False
    tmp = cache_dir / ".cache_version.tmp"
    replaced = False
    try:
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, stamp)
        replaced = True
    except OSError as exc:
        log.debug("could not refresh cache stamp after lazy map build (%s)", exc)
        with contextlib.suppress(OSError):
            tmp.unlink()

    if replaced:
        # Its own except, and its own message: folding this into the block
        # above would report a silently revalidated stale cache as "could not
        # refresh cache stamp" and would try to unlink a tmp that os.replace
        # has already consumed. ``ns=`` rather than the float fields — the
        # float form rounds, and the invariant is an exact one.
        try:
            os.utime(stamp, ns=(before.st_atime_ns, before.st_mtime_ns))
        except OSError as exc:
            log.warning(
                "could not restore the freshness token on %s (%s); a cache that "
                "is stale against its source may now pass is_cache_valid until "
                "it is rebuilt",
                stamp,
                exc,
            )

    # The oversized-map warning used to fire only from _build_cache_into, which
    # no longer builds the map — so without this it would never fire again. The
    # stamp is the only record of which profile this cache belongs to; the
    # connection does not carry the path.
    source = meta.get("source")
    if source:
        _check_cache_size(cache_dir, str(cache_dir.parent / source))


# Outcomes of :func:`materialize_cached_nvtx_kernel_map_outcome`. Only
# MAP_MATERIALIZED leaves the map queryable; the rest are the distinct reasons
# the bool-returning wrapper collapses into a single False.
MAP_MATERIALIZED = "materialized"  # views exist over a published map
MAP_NO_CACHE_DIR = "no-cache-dir"  # connection is not backed by a Parquet cache
MAP_SOURCES_MISSING = "sources-missing"  # a source Parquet the sweep reads is absent
MAP_NO_ATTRIBUTION = "no-attribution"  # the sweep ran and produced no rows
MAP_NOT_WRITABLE = "not-writable"  # an OSError while locking, staging or publishing
MAP_VIEWS_FAILED = "views-failed"  # published, but the views would not create


def materialize_cached_nvtx_kernel_map(conn) -> bool:
    """True when ``nvtx_kernel_map`` is queryable on ``conn`` off the cache.

    Thin wrapper over :func:`materialize_cached_nvtx_kernel_map_outcome`, which
    is the one to call when the *reason* for a False matters — the six outcomes
    it distinguishes range from "this profile has nothing to attribute" to "the
    cache directory could not be written", and a caller that only sees False
    cannot tell a warm profile from a failed one.
    """
    return materialize_cached_nvtx_kernel_map_outcome(conn)[0] == MAP_MATERIALIZED


def materialize_cached_nvtx_kernel_map_outcome(conn) -> tuple[str, str]:
    """Build ``nvtx_kernel_map`` into ``conn``'s Parquet cache, then view it.

    The persisted half of the deferred-map design. ``build_cache`` no longer
    runs the stack sweep (see :func:`_should_defer_nvtx_kernel_map`); this does,
    on the first query that actually needs the map, and it writes the result
    into the cache directory so the next process finds it by glob at open time
    instead of repeating the sweep. Without that persistence, deferring would
    merely move the cost — and onto the more expensive in-memory path, which
    ``fetchall``s where this streams.

    Returns ``(outcome, detail)``. ``outcome`` is one of the ``MAP_*`` constants
    above; ``detail`` is a human-readable elaboration, empty when there is
    nothing to add. Everything other than ``MAP_MATERIALIZED`` leaves the caller
    falling back to the in-memory build, which is exactly the behaviour those
    backends had before.

    That is *not* the same as leaving the cache directory unchanged, and
    ``MAP_VIEWS_FAILED`` is the exception that matters: the two Parquets were
    written, and only the views over them could not be created. The next process
    finds them by glob and skips the sweep. So it is a failure of this call, not
    a failure to persist, and code that groups it with the "could not write"
    outcomes is wrong about what is on disk.

    Note that ``MAP_NO_ATTRIBUTION`` publishes nothing, so it is *not* a state
    the next process inherits: a profile that reaches it re-runs the full sweep
    on every call.

    Never call this from inside ``build_cache``: that holds ``_build_lock`` for
    the same cache directory, and flock is not reentrant across the fds involved
    — the process would wedge. The only callers are query-time.
    """
    from .connection import DuckDBAdapter, forget_nvtx_map_probes, wrap_connection

    adapter = wrap_connection(conn)
    if not isinstance(adapter, DuckDBAdapter):
        return (MAP_NO_CACHE_DIR, "the connection is not DuckDB-backed")
    db = adapter.raw_conn

    registered = cache_dir_for_connection(db)
    if registered is None:
        return (MAP_NO_CACHE_DIR, "no cache directory is registered for this connection")
    cache_dir = Path(registered)
    if not cache_dir.is_dir():
        return (MAP_NO_CACHE_DIR, f"{cache_dir} is not a directory")

    # Cheap path first: a previous process already published the map, and this
    # connection just has not created the views yet. No lock, no write — which
    # is also what makes a read-only cache directory work.
    if _create_nvtx_map_views(db, cache_dir):
        forget_nvtx_map_probes(db)
        return (MAP_MATERIALIZED, "")

    import shutil
    import tempfile

    try:
        with _build_lock(cache_dir):
            # Another process may have published while we waited.
            if _create_nvtx_map_views(db, cache_dir):
                forget_nvtx_map_probes(db)
                return (MAP_MATERIALIZED, "")

            # Staged inside the cache directory so the os.replace below is a
            # rename within one filesystem rather than a copy. A process killed
            # between here and the rmtree leaves a `.nvtx_map_build_*`
            # directory behind and nothing prunes it. That is inert rather than
            # corrupting: open_cached_db's `cache_dir.glob("*.parquet")` is
            # non-recursive so a half-written file never becomes a view,
            # build_cache's size report counts `is_file()` entries only, and
            # the next build stages into a fresh mkdtemp. It costs disk until
            # the cache directory is removed.
            staging = Path(
                tempfile.mkdtemp(prefix=".nvtx_map_build_", dir=cache_dir)
            )
            try:
                built = _build_nvtx_kernel_map_from_parquet(db, cache_dir, staging)
                if not built:
                    # That helper answers False for exactly two reasons and does
                    # not say which. Re-test the cheaper one — the same
                    # ``is_file()`` predicate it checked a moment ago, on files
                    # nothing creates in between — so the caller can tell a
                    # degraded cache from a profile with nothing to attribute.
                    missing = [
                        name
                        for name in ("kernels.parquet", "runtime.parquet", "nvtx.parquet")
                        if not (cache_dir / name).is_file()
                    ]
                    if missing:
                        return (MAP_SOURCES_MISSING, f"missing from the cache: {', '.join(missing)}")
                    return (MAP_NO_ATTRIBUTION, "")
                # Publish the dictionary first. A concurrent opener globbing
                # mid-publish then sees a dict with no map and falls back, which
                # is correct; the reverse order would give it a map whose
                # path_id join resolves to nothing.
                os.replace(
                    staging / "nvtx_path_dict.parquet",
                    cache_dir / "nvtx_path_dict.parquet",
                )
                os.replace(
                    staging / "nvtx_kernel_map.parquet",
                    cache_dir / "nvtx_kernel_map.parquet",
                )
            finally:
                shutil.rmtree(staging, ignore_errors=True)

            _publish_stamp_map_ready(cache_dir)
    except OSError as exc:
        # Read-only cache directory, a full disk, an unwritable lock file in the
        # cache's *parent* — anything that stops the lock/stage/publish sequence.
        # Degrade to the in-memory build rather than failing the query, and hand
        # the caller the OSError's own words so it can name the real path.
        log.info("cannot persist nvtx_kernel_map into %s (%s)", cache_dir, exc)
        return (MAP_NOT_WRITABLE, str(exc))

    if _create_nvtx_map_views(db, cache_dir):
        forget_nvtx_map_probes(db)
        return (MAP_MATERIALIZED, "")
    return (MAP_VIEWS_FAILED, f"the map was published into {cache_dir} but no view could be created")


def ensure_nvtx_kernel_map(conn) -> bool:
    """Make ``nvtx_kernel_map`` + ``nvtx_path_dict`` queryable on a DuckDB
    connection when they are absent, so NVTX-attribution skills take their fast
    map-backed path instead of an in-file IEJoin that hangs DuckDB's
    ``sqlite_scanner`` on a direct-attached profile with no parquet cache (#257).

    Two builds sit behind this, and they leave different objects in the catalog.
    On a Parquet-cache connection the map is built *into the cache* by
    :func:`materialize_cached_nvtx_kernel_map` and exposed as **views** over the
    published Parquet, so it survives the process. On every other backend —
    direct SQLite attach, ``parquetdir``, a bare test connection — there is
    nowhere to persist it, so it is built as ordinary **tables** in memory as
    before, once per connection. Either way a ``SELECT`` against the two names
    works afterwards, and DuckDB's ``SHOW TABLES`` lists both shapes (it lists
    views too); the difference matters only to code that inspects object types.

    Returns True when the map is present afterwards (already there, or just
    built). Returns False — changing nothing — for a non-DuckDB connection or
    when the source tables are missing, so the caller keeps its existing path.
    The in-memory build is a flat fetch (fast on every backend) plus the shared
    Python sort-merge; it never issues the range join that chokes the scanner.

    Concurrency: serialized **process-wide** by ``_MAP_BUILD_LOCK``, not per
    database — two different profiles building their maps in one process queue
    up instead of overlapping, which is a deliberate trade (see the note at the
    lock's definition: a bare ``.cursor()`` cannot be traced back to its
    database, and a per-database key was measured letting all four threads
    through). Before that lock existed, four threads on their own ``.cursor()``
    handles each ran the whole fetch-and-sweep and then three of them lost the
    ``CREATE TABLE`` to a catalog write-write conflict — four times the work and
    four times the memory to produce one table. Both call sites swallowed the
    exception under ``except DB_ERRORS``, so it was invisible.

    The wait is silent, and stays so here. It moved out of ``build_cache`` and
    into this call — stderr during it captures as exactly ``''`` — and reporting
    it needs a channel from here back to the caller, which is #332.
    """
    from .connection import DB_ERRORS, DuckDBAdapter, wrap_connection

    adapter = wrap_connection(conn)
    if not isinstance(adapter, DuckDBAdapter):
        return False

    db = adapter.raw_conn
    # Already present — the parquet-cache path, or a prior call on this conn.
    try:
        db.execute("SELECT 1 FROM nvtx_kernel_map LIMIT 1")
        return True
    except DB_ERRORS:
        pass

    with _MAP_BUILD_LOCK:
        # Re-check under the lock: a thread that was building while we waited
        # has published by now, and repeating its work would be pure waste.
        if _nvtx_map_present(db):
            return True
        # Persisted build, when this connection has a cache to write into.
        if materialize_cached_nvtx_kernel_map(db):
            return True
        return _ensure_nvtx_kernel_map_in_memory(adapter, db)


def _ensure_nvtx_kernel_map_in_memory(adapter, db) -> bool:
    """Build the map as ordinary tables on ``db``, for backends with no cache.

    Kept for direct-attach and ``parquetdir`` connections, which have no
    directory to publish Parquet into. It ``fetchall``s both sources where the
    cached builder streams them, so it costs more memory — that is the reason
    the cached path exists and is tried first, not a defect here.

    The caller holds ``_MAP_BUILD_LOCK``, which is process-wide rather than
    per-database.
    """
    from .connection import DB_ERRORS

    tables = adapter.resolve_activity_tables()
    kernel_table = tables.get("kernel")
    runtime_table = tables.get("runtime")
    nvtx_table = tables.get("nvtx", "NVTX_EVENTS")
    if not kernel_table or not runtime_table:
        return False

    if adapter.detect_nvtx_text_id():
        text_expr = "COALESCE(n.text, s.value)"
        text_join = "LEFT JOIN StringIds s ON n.textId = s.id"
    else:
        text_expr = "n.text"
        text_join = ""

    # kernel_name and the embedded TC flags resolved exactly as the cache-built
    # map's TC-enriched ``kernels`` view (_tc_enriched_sql), so those three
    # columns are a drop-in for the cached map's: name = COALESCE(demangled,
    # short, 'kernel_'||id), TC eligibility/use by the same name regexes.
    tc_active = _TC_ACTIVE_PATTERN
    tc_elig = _TC_ELIGIBLE_PATTERN
    lname = "lower(COALESCE(sd.value, ss.value, ''))"
    try:
        kr_rows = db.execute(
            f'SELECT r.globalTid, r.start, r."end", k.start, k."end", '
            f"COALESCE(sd.value, ss.value, 'kernel_' || CAST(k.shortName AS VARCHAR)) AS kernel_name, "
            f"CAST(CASE WHEN regexp_matches({lname}, {tc_elig}) "
            f"OR regexp_matches({lname}, {tc_active}) THEN 1 ELSE 0 END AS INTEGER) AS is_tc_eligible, "
            f"CAST(CASE WHEN regexp_matches({lname}, {tc_active}) THEN 1 ELSE 0 END AS INTEGER) AS uses_tc "
            f"FROM {kernel_table} k "
            f"JOIN {runtime_table} r ON r.correlationId = k.correlationId "
            f"LEFT JOIN StringIds sd ON k.demangledName = sd.id "
            f"LEFT JOIN StringIds ss ON k.shortName = ss.id "
        ).fetchall()
        # Only the NVTX side has to arrive sorted. The sweep advances a single
        # index over the ranges per thread without re-sorting them, but it does
        # sort the kernel side itself (``kr_by_tid[tid].sort(...)``), so asking
        # the kernel query above for an order it will redo buys nothing. That
        # asymmetry is what tests/test_determinism.py pins.
        nvtx_rows = db.execute(
            f'SELECT n.globalTid, n.start, n."end", {text_expr} AS text '
            f"FROM {nvtx_table} n {text_join} "
            f'WHERE n.eventType = 59 AND n."end" > n.start '
            f"ORDER BY n.globalTid, n.start"
        ).fetchall()
    except DB_ERRORS as exc:
        log.debug("ensure_nvtx_kernel_map: source fetch failed (%s)", exc)
        return False

    # Per-kernel TC flags (by k_start, k_end, name) to attach after the
    # name-agnostic sweep, which only reads the first six fields.
    tc_by_kernel = {(r[3], r[4], r[5]): (r[6], r[7]) for r in kr_rows}
    results = _sweep_nvtx_kernel_map(kr_rows, nvtx_rows)
    for r in results:
        r["is_tc_eligible"], r["uses_tc"] = tc_by_kernel.get(
            (r["k_start"], r["k_end"], r["kernel_name"]), (0, 0)
        )
    # Materialize even when empty: the existence check then passes and consumers
    # take the (empty) map path rather than re-entering the hanging IEJoin.
    with _CATALOG_DDL_LOCK:
        try:
            _materialize_nvtx_kernel_map(db, results)
        except DB_ERRORS as exc:
            # A caller that reached here without _MAP_BUILD_LOCK held (this
            # helper is module-private, but the DDL lock is the only thing
            # guarding the catalog) can still lose a write-write conflict. The
            # winner's tables are as good as ours, so re-probe instead of
            # raising into the call sites' bare ``except DB_ERRORS: pass``,
            # where a genuine failure would be indistinguishable from a lost
            # race.
            log.debug("nvtx_kernel_map CREATE lost a catalog race (%s)", exc)
            return _nvtx_map_present(db)
    return True


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
    in-place.  Analytical queries on large scans are slower than a warm cache
    (3-9x on the four captures measured for issue #317, from 93 MB to 3.7 GB:
    3.0x at 235 MB, 3.9x at 924 MB, 5.1x at 3.7 GB, 9.1x at 93 MB), but startup
    is instant.  Used for:

      - One-off queries that only touch 1-2 tables, where the build would not
        repay itself: on a 3.7 GB profile one cheap skill costs ~19 s direct
        against ~27 s including a build. That margin used to be 18.7 s against
        79.2 s; #322 deferred nvtx_kernel_map out of the build, so direct now
        wins this case by 1.5x rather than 4x
      - ``cache_mode="direct"``, ``--no-cache`` and ``NSYS_AI_CACHE_MODE=direct``
      - the fallback when a cache cannot be built or read — an unwritable
        profile directory, no disk space, or a failed build

    It is *not* the default for large profiles any more. That rule predated the
    stack-sweep map builder and is gone; see ``open_auto_db``.
    """
    _require_profile_exists(sqlite_path)
    db = duckdb.connect()
    _configure_duckdb_analytics_session(db)
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
