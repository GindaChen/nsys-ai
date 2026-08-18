"""
profile.py — Ingest and query Nsight Systems profiles.

Provides a thin wrapper around profile backends with typed accessors
for kernels, NVTX events, CUDA runtime calls, and metadata.
"""

import functools
import logging
import os
import re
import shutil
import sqlite3
import subprocess  # nosec B404
import threading
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# subprocess: nsys export (.nsys-rep→.sqlite) only; argv list, no shell.
if typing.TYPE_CHECKING:
    from .fingerprint import ProfileFingerprint


from nsys_ai import parquet_cache
from nsys_ai.connection import DB_ERRORS
from nsys_ai.exceptions import (
    ExportError,
    ExportTimeoutError,
    ExportToolMissingError,
    ProfileNotFoundError,
    SchemaError,
)
from nsys_ai.profile_reference import inspect_local_parquetdir

# Regex for safe SQL identifiers (table/column names).
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

IngestPolicy = Literal["auto", "parquetdir", "sqlite"]
StorageKind = Literal["nsys-rep", "parquetdir", "sqlite"]


@dataclass(frozen=True)
class ProfileResolution:
    """The single ingest decision shared by every profile entry point."""

    source_path: str
    resolved_path: str
    storage_kind: StorageKind
    backend: Literal["sqlite", "parquetdir"]
    cache_mode: Literal["auto", "direct"]


def resolve_ingest_policy(policy: str | None = None) -> IngestPolicy:
    """Return the explicit ingest policy, honoring ``NSYS_AI_INGEST``."""
    value = (
        policy
        if policy is not None
        else os.environ.get("NSYS_AI_INGEST", "auto").strip().lower()
    )
    if value not in {"auto", "parquetdir", "sqlite"}:
        raise ValueError(
            f"Unknown ingest policy: {value!r}. Expected 'auto', 'parquetdir', or 'sqlite'."
        )
    return value  # type: ignore[return-value]


def _validate_table_name(name: str) -> str:
    """Ensure a table name contains only safe SQL identifier characters.

    Table names come from ``sqlite_master`` (not user input), but this
    provides defence-in-depth against accidental SQL injection if schema
    detection logic ever changes.
    """
    if not _SAFE_IDENTIFIER_RE.match(name):
        raise ValueError(f"Unsafe table name from schema: {name!r}")
    return name


class NsightSchema:
    """
    Lightweight schema/metadata helper for Nsight Systems SQLite exports.

    Detects available tables, attempts to infer the Nsight Systems version,
    and exposes canonical table choices (e.g., kernel activity table).
    """

    def __init__(self, conn, meta_conn=None):
        self._conn = conn
        from .connection import wrap_connection

        self._adapter = wrap_connection(conn)
        # ``META_DATA_*`` lives only in the SQLite export — the Parquet/DuckDB
        # cache materializes the analysis tables and not the metadata ones. When
        # the schema is built on the cache (the normal path), those lookups must
        # fall back to the original SQLite connection or every metadata field
        # silently reads as absent.
        self._meta_adapter = (
            wrap_connection(meta_conn) if meta_conn is not None and meta_conn is not conn else None
        )

        self.tables = list(self._adapter.get_table_names())

        # Read the metadata tables once; both version fields derive from it.
        self._meta: dict[str, str] = {}
        for table in ("META_DATA_EXPORT", "META_DATA_CAPTURE"):
            self._meta.update(self._read_kv_table(table))

        self.version: str | None = self._detect_version()
        # The export schema version is what actually changes shape between
        # releases — NVIDIA documents that the SQLite schema "can and will
        # change" — so this, not the product version, is what compatibility
        # handling should key off.
        self.schema_version: str | None = self._meta.get("EXPORT_SCHEMA_VERSION") or None
        kt = self._detect_kernel_table()
        self.kernel_table: str | None = _validate_table_name(kt) if kt else None

    # ── Version detection ──────────────────────────────────────────────

    def _read_kv_table(self, table: str) -> dict[str, str]:
        """
        Best-effort reader for META_DATA_* style tables which may use
        slightly different column names across Nsight versions.
        """
        adapter = self._adapter
        if table not in self.tables:
            # Not in the primary connection — try the metadata source before
            # giving up (see the note in __init__).
            if self._meta_adapter is None:
                return {}
            try:
                if table not in self._meta_adapter.get_table_names():
                    return {}
            except DB_ERRORS:
                return {}
            adapter = self._meta_adapter

        try:
            cols = adapter.get_table_columns(table)
        except DB_ERRORS:
            return {}
        key_col = None
        val_col = None

        # Common patterns seen in Nsight exports. Matching is case-insensitive
        # because the real exports use lowercase ``name``/``value`` while older
        # probes only listed ``NAME``/``Name`` — the omission silently produced
        # an empty mapping, so every consumer of these tables (version
        # detection, GPU clock) degraded to None on every real profile.
        lowered = {str(c).lower(): c for c in cols}
        for cand in ("name", "key"):
            if cand in lowered:
                key_col = lowered[cand]
                break
        for cand in ("value", "val"):
            if cand in lowered:
                val_col = lowered[cand]
                break

        if not key_col or not val_col:
            return {}

        kv: dict[str, str] = {}
        try:
            cur = adapter.execute(f"SELECT {key_col}, {val_col} FROM {table}")  # noqa: S608
            rows = cur.fetchall()
        except DB_ERRORS:
            return {}
        for k, v in rows:
            if k is not None and v is not None:
                kv[str(k)] = str(v)
        return kv

    def _detect_version(self) -> str | None:
        """Infer the Nsight Systems version from the metadata tables."""
        meta = self._meta

        # The canonical key in current exports. Checked first and by exact name
        # so the product *name* ("NVIDIA Nsight Systems") can never be mistaken
        # for the version, which the old value-substring fallback allowed.
        for key in ("EXPORT_PRODUCT_VERSION", "PRODUCT_VERSION"):
            if meta.get(key):
                return meta[key]

        # Heuristic keys that might carry version information
        for key in meta:
            lk = key.lower()
            if "nsight systems version" in lk or "exporter version" in lk:
                return meta[key]
        # Fallback: a value carrying 'Nsight Systems X.Y'. Require a digit so a
        # bare product name is not returned as a version.
        for val in meta.values():
            if "Nsight Systems" in val and any(ch.isdigit() for ch in val):
                return val
        return None

    # ── Table detection ────────────────────────────────────────────────

    def _detect_kernel_table(self) -> str | None:
        """
        Pick an appropriate kernel activity table, if present.

        Today this is usually CUPTI_ACTIVITY_KIND_KERNEL, but we keep
        the detection logic resilient to future renames.

        Variants of the canonical name are chosen by the shared resolver, never
        here. This used to be its own "exact name, else sorted(candidates)[0]"
        scan, which picks the *oldest* variant: on a profile carrying both
        ``..._KERNEL_V2`` and ``..._KERNEL_V3`` it returned ``_V2`` while
        ``resolve_activity_tables()["kernel"]`` returned ``_V3``, so
        ``launch_overhead_ms`` — whose SQL joins ``kernel_table`` against the
        resolved runtime table — read two tables that did not describe the same
        window. Measured on a fixture with all three activity tables doubled and
        the ``_V2`` copies truncated to a quarter: 0.0 ms / 0 iterations against
        the ``_V3``-only control's 0.05 ms / 28.

        The substring sweep below stays as a last resort for a rename the
        resolver cannot see (a name that does not start with the canonical
        prefix at all). Its ``sorted`` is only for determinism between such
        unrelated names — by then there is no version ordering left to get wrong.
        """
        from .connection import resolve_table_variant

        resolved = resolve_table_variant(
            set(self.tables), "CUPTI_ACTIVITY_KIND_KERNEL", allow_other_suffixes=True
        )
        if resolved:
            return resolved

        # Fallback: any non-enum table with KERNEL in the name
        candidates = [
            t for t in self.tables if "KERNEL" in t.upper() and not t.upper().startswith("ENUM_")
        ]
        if candidates:
            # Deterministic order
            candidates.sort()
            return candidates[0]

        return None

    # ── Schema contract ────────────────────────────────────────────────
    # Columns the core analysis path selects by name. Their absence is not a
    # graceful degradation but a crash or a silently-wrong number, so a future
    # Nsight export that drops or renames one should fail loudly and by name
    # rather than surface as a user bug report (issue #237). NVIDIA documents
    # that the SQLite export schema "can and will change". Optional surface —
    # NVTX_EVENTS, META_DATA_* — is deliberately excluded: skills degrade around
    # it, so a --trace=cuda capture with no NVTX must not trip the contract.
    _REQUIRED_KERNEL_COLUMNS = (
        "deviceId",
        "streamId",
        "start",
        "end",
        "shortName",
        "demangledName",
        "correlationId",
    )
    _REQUIRED_STRINGIDS_COLUMNS = ("id", "value")

    def _resolve_table(self, name: str) -> str | None:
        """Actual table name matching ``name`` case-insensitively, or None."""
        lname = name.lower()
        for t in self.tables:
            if t.lower() == lname:
                return t
        return None

    def _missing_columns(self, table: str, required) -> list[str]:
        try:
            present = {str(c).lower() for c in self._adapter.get_table_columns(table)}
        except DB_ERRORS:
            return [f"{table} (columns unreadable)"]
        return [f"{table}.{c}" for c in required if c.lower() not in present]

    def missing_required_columns(self) -> list[str]:
        """Descriptors of hard-required tables/columns absent from this export.

        Empty when the schema is compatible with the core analysis path. Each
        entry names exactly what is missing (``CUPTI_ACTIVITY_KIND_KERNEL.start``,
        or a table name) so a diagnostic points straight at the drift.
        """
        missing: list[str] = []

        if not self.kernel_table:
            missing.append(
                "kernel activity table (CUPTI_ACTIVITY_KIND_KERNEL or a *KERNEL* table)"
            )
        else:
            missing += self._missing_columns(self.kernel_table, self._REQUIRED_KERNEL_COLUMNS)

        stringids = self._resolve_table("StringIds")
        if not stringids:
            missing.append("StringIds")
        else:
            missing += self._missing_columns(stringids, self._REQUIRED_STRINGIDS_COLUMNS)

        return missing


@dataclass
class GpuInfo:
    """Hardware metadata for one GPU."""

    device_id: int
    name: str = ""
    pci_bus: str = ""
    sm_count: int = 0
    memory_bytes: int = 0
    kernel_count: int = 0
    streams: list[int] = field(default_factory=list)


@dataclass
class ProfileMeta:
    """Discovered metadata from an Nsight profile."""

    devices: list[int]  # active deviceIds
    streams: dict[int, list[int]]  # deviceId -> [streamId, ...]
    time_range: tuple[int, int]  # (min_start_ns, max_end_ns)
    kernel_count: int
    nvtx_count: int
    tables: list[str]
    gpu_info: dict[int, GpuInfo] = field(default_factory=dict)  # deviceId -> GpuInfo
    #: deviceId -> summed kernel duration in ns. Comes free from the pass that
    #: already groups by device, and lets a caller ask how much of the capture
    #: a device actually accounts for rather than only whether it appears.
    device_kernel_ns: dict[int, int] = field(default_factory=dict)


class Profile:
    """Handle to an opened Nsight Systems profile.

    Exposes two database connections:
      - ``self.conn`` (sqlite3.Connection | duckdb.DuckDBPyConnection): the primary
        backend connection. For ``backend='sqlite'`` this is the original SQLite DB;
        for ``backend='parquetdir'`` this is the DuckDB connection over Parquet.
      - ``self.db`` (duckdb.DuckDBPyConnection): DuckDB over Parquet cache — the
        primary query path for all analytical queries.
    """

    _log = logging.getLogger(__name__)

    #: The exception that sent this Profile down the SQLite fallback, or None.
    #: ``db is None`` says only *that* the cache is unavailable; commands whose
    #: job is the cache itself (``warm``) need to say *why*, and re-running the
    #: build to find out would repeat an ETL that can take minutes. Set only in
    #: the fallback branch below; every other construction path leaves the class
    #: default in place.
    cache_error: Exception | None = None

    def __init__(
        self,
        path: str,
        *,
        cache_mode: str = "auto",
        backend: str = "auto",
        ingest_policy: str | None = None,
        progress: typing.Callable[[str, int, int], None] | None = None,
    ):
        if cache_mode not in ("auto", "parquet", "direct"):
            raise ValueError(
                f"Unknown cache_mode: {cache_mode!r}. Expected 'auto', 'parquet', or 'direct'."
            )
        if backend not in ("auto", "sqlite", "parquetdir"):
            raise ValueError(
                f"Unknown backend: {backend!r}. Expected 'auto', 'sqlite', or 'parquetdir'."
            )
        if backend == "parquetdir" and cache_mode != "auto":
            raise ValueError(
                "cache_mode is not supported with backend='parquetdir'; use cache_mode='auto'."
            )
        resolution = resolve_profile(
            path,
            backend=backend,
            ingest_policy=ingest_policy,
        )
        if resolution.backend == "parquetdir" and cache_mode != "auto":
            raise ValueError(
                "cache_mode is not supported with backend='parquetdir'; use cache_mode='auto'."
            )
        if resolution.cache_mode == "direct" and cache_mode == "parquet":
            raise ValueError(
                "ingest policy 'sqlite' requires direct SQLite access; "
                "cache_mode='parquet' is incompatible"
            )
        self.path = resolution.resolved_path
        self.input_path = resolution.source_path
        self.resolved_path = resolution.resolved_path
        self.storage_kind = resolution.storage_kind
        self.backend = resolution.backend
        self._lock = threading.Lock()
        self._owns_conn = True
        if resolution.backend == "parquetdir":
            self.db = parquet_cache.open_parquetdir_db(resolution.resolved_path)
            self.conn = self.db
        else:
            self.conn = sqlite3.connect(resolution.resolved_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            # Three-tier open (cache / direct attach / raw sqlite3) lives in
            # parquet_cache.open_with_direct_fallback — also used by
            # open_profile_readonly and skill run, so a failed build cannot
            # drop only some entry points onto raw sqlite3 (issue #333).
            effective_cache_mode = (
                "direct" if resolution.cache_mode == "direct" else cache_mode
            )
            if effective_cache_mode == "direct":
                # Force direct SQLite via DuckDB — zero ETL, instant startup
                primary = parquet_cache.open_direct_sqlite
            elif effective_cache_mode == "parquet":
                # Original behaviour: block until cache is built.
                # env_escape=False: this branch never reads
                # NSYS_AI_CACHE_MODE, so the build banner must not tell the
                # user to set it — on this path it does nothing. It names
                # cache_mode="direct" instead, which is the way out here.
                primary = functools.partial(
                    parquet_cache.open_cached_db, env_escape=False, progress=progress
                )
            else:
                # auto: cache when one can be had, direct SQLite when it
                # cannot. The policy (and the measurements behind it) lives
                # in parquet_cache.open_auto_db, because `skill run` and
                # open_profile_readonly reach it without a Profile.
                primary = functools.partial(parquet_cache.open_auto_db, progress=progress)
            self.db, err = parquet_cache.open_with_direct_fallback(
                resolution.resolved_path, primary, log=self._log
            )
            if err is not None:
                self.cache_error = err
            # self.db is None → keep self.conn (raw sqlite3); already opened above
        from .connection import wrap_connection

        self.adapter = wrap_connection(self.db if self.db is not None else self.conn)
        self.schema = NsightSchema(
            self.db if self.db is not None else self.conn,
            # The SQLite export is the only place META_DATA_* exists.
            meta_conn=self.conn,
        )
        self.meta = self._discover()
        self._nvtx_has_text_id = self.adapter.detect_nvtx_text_id()
        self._owner_thread = threading.get_ident()
        self._thread_handles = threading.local()

    @classmethod
    def _from_conn(cls, conn: sqlite3.Connection) -> "Profile":
        """Wrap an existing connection as a Profile without opening a new file.

        The connection is borrowed — ``close()`` will NOT close it.
        Supports both SQLite and DuckDB connections.
        """
        from .connection import DuckDBAdapter, wrap_connection

        adapter = wrap_connection(conn)
        is_duckdb = isinstance(adapter, DuckDBAdapter)
        if not is_duckdb:
            conn.row_factory = sqlite3.Row
        obj = cls.__new__(cls)
        obj.conn = conn
        obj._lock = threading.Lock()
        obj._owns_conn = False
        obj.path = ""
        obj.backend = "sqlite"
        obj.db = conn if is_duckdb else None  # type: ignore[assignment]
        obj.adapter = adapter
        obj.schema = NsightSchema(conn)
        obj.meta = obj._discover()
        obj._nvtx_has_text_id = obj.adapter.detect_nvtx_text_id()
        obj._owner_thread = threading.get_ident()
        obj._thread_handles = threading.local()
        return obj

    def _discover(self) -> ProfileMeta:
        tables = self.schema.tables

        if not self.schema.kernel_table:
            ver_msg = f" (Nsight version: {self.schema.version})" if self.schema.version else ""
            raise SchemaError(
                "This profile does not contain GPU kernel activity "
                f"(no suitable KERNEL table found){ver_msg}. "
                "It may have been captured without CUDA kernel tracing, "
                "or exported with a schema layout this version of nsys-ai "
                "does not yet understand."
            )

        kernel_table = self.schema.kernel_table

        # Single pass over the kernel table for devices, per-device streams,
        # time range, total kernel count, and per-device kernel counts. These
        # were five separate full scans; in direct-SQLite mode (large profiles)
        # each scan is seconds, so collapsing them noticeably speeds up open.
        # ORDER BY preserves the previous device/stream ordering exactly.
        devices: list[int] = []
        streams: dict[int, list[int]] = {}
        kcounts: dict[int, int] = {}
        device_ns: dict[int, int] = {}
        kernel_count = 0
        min_start = None
        max_end = None
        for dev, stream, mn, mx, cnt, dur in self.adapter.execute(
            f"SELECT deviceId, streamId, MIN(start), MAX([end]), COUNT(*), SUM([end] - start) "
            f"FROM {kernel_table} GROUP BY deviceId, streamId ORDER BY deviceId, streamId"
        ).fetchall():
            if dev not in streams:
                streams[dev] = []
                kcounts[dev] = 0
                device_ns[dev] = 0
                devices.append(dev)
            streams[dev].append(stream)
            kcounts[dev] += cnt
            device_ns[dev] += int(dur or 0)
            kernel_count += cnt
            if mn is not None:
                min_start = mn if min_start is None else min(min_start, mn)
            if mx is not None:
                max_end = mx if max_end is None else max(max_end, mx)

        _meta_nvtx = self.adapter.resolve_activity_tables().get("nvtx")
        nc = (
            self.adapter.execute(f"SELECT COUNT(*) FROM {_meta_nvtx}").fetchone()[0]  # noqa: S608
            if _meta_nvtx
            else 0
        )

        return ProfileMeta(
            devices=devices,
            streams=streams,
            time_range=(min_start or 0, max_end or 0),
            kernel_count=kernel_count,
            nvtx_count=nc,
            tables=tables,
            gpu_info=self._gpu_info(devices, streams, tables, kcounts),
            device_kernel_ns=device_ns,
        )

    def query_conn(self):
        """The connection this thread should run queries through.

        Two things in one accessor, because they are the same decision.

        Which connection: DuckDB when a cache or direct attach is in use, the
        SQLite connection when it is not. That choice was spelled out in eleven
        places across eight modules, in three different spellings including an
        inverted one, while ``__init__`` had already made it.

        Which handle: DuckDB keeps the pending result set on the connection, so
        ``execute`` and ``fetch`` are individually atomic but not atomic as a
        pair. Two threads sharing one handle therefore clobber each other and
        return wrong rows with nothing raised — measured wrong in 6 of 6 trials
        against a plain DuckDB connection. The documented remedy is one
        ``.cursor()`` per thread, so worker threads get a thread-local cursor,
        reused across calls rather than created per query.

        The thread that opened the profile keeps the connection object itself.
        That leaves every existing single-threaded caller on exactly the path it
        was on, and confines the new behaviour to threads that did not exist
        before. It also matters for correctness: scratch tables created with
        ``CREATE TEMP TABLE`` are visible only to the handle that made them.

        SQLite needs none of this — the connection is opened with
        ``check_same_thread=False`` and serialises internally.
        """
        if self.db is None:
            return self.conn
        # During __init__ the owner is not recorded yet, and _discover() queries
        # through here. Construction is single-threaded, so the raw handle is
        # the right answer and there is nothing to guard.
        owner = getattr(self, "_owner_thread", None)
        if owner is None or threading.get_ident() == owner:
            return self.db
        handle = getattr(self._thread_handles, "conn", None)
        if handle is None:
            handle = self._thread_handles.conn = self.db.cursor()
            # Point the per-connection caches at the owning connection's bag.
            # Without this each worker thread starts with an empty bag and
            # re-executes every memoized skill once per thread.
            from .connection import register_derived_handle

            register_derived_handle(handle, self.db)
        return handle

    def _gpu_info(self, devices, streams, tables, kcounts) -> dict[int, GpuInfo]:
        """Build per-GPU metadata. Per-device kernel counts come from the
        single scan in :meth:`_discover`; only hardware metadata is queried here."""
        info: dict[int, GpuInfo] = {}
        query_conn = self.db if self.db is not None else self.conn

        # Hardware info from TARGET_INFO_GPU + TARGET_INFO_CUDA_DEVICE
        hw = {}
        if "TARGET_INFO_GPU" in tables and "TARGET_INFO_CUDA_DEVICE" in tables:
            for r in query_conn.execute("""
                SELECT c.cudaId as dev, g.name, g.busLocation,
                       g.smCount as sms, g.totalMemory as mem,
                       g.chipName, g.memoryBandwidth as bw
                FROM TARGET_INFO_GPU g
                JOIN TARGET_INFO_CUDA_DEVICE c ON g.id = c.gpuId
                GROUP BY c.cudaId, g.name, g.busLocation, g.smCount, g.totalMemory, g.chipName, g.memoryBandwidth
            """).fetchall():
                hw[r[0]] = dict(
                    name=r[1] or "",
                    pci_bus=r[2] or "",
                    sm_count=r[3] or 0,
                    memory_bytes=r[4] or 0,
                )

        for dev in devices:
            h = hw.get(dev, {})
            info[dev] = GpuInfo(
                device_id=dev,
                name=h.get("name", ""),
                pci_bus=h.get("pci_bus", ""),
                sm_count=h.get("sm_count", 0),
                memory_bytes=h.get("memory_bytes", 0),
                kernel_count=kcounts.get(dev, 0),
                streams=streams.get(dev, []),
            )
        return info

    @functools.cached_property
    def fingerprint(self) -> "ProfileFingerprint":  # type: ignore[name-defined]
        from .fingerprint import get_fingerprint

        return get_fingerprint(self.conn)

    def kernels(self, device: int | None, trim: tuple[int, int] | None = None) -> list[dict]:
        """All kernels on a device (or all devices if None), optionally trimmed to a time window."""
        sql = """
            SELECT k.start, k.[end], k.deviceId, k.streamId, k.correlationId,
                   s.value as name, d.value as demangled
            FROM {kernel_table} k
            JOIN StringIds s ON k.shortName = s.id
            JOIN StringIds d ON k.demangledName = d.id"""
        sql = sql.format(kernel_table=self.schema.kernel_table)
        params: list = []
        if device is not None:
            sql += "\n            WHERE k.deviceId = ?"
            params.append(device)
        else:
            sql += "\n            WHERE 1=1"
        if trim:
            sql += " AND k.start >= ? AND k.[end] <= ?"
            params += list(trim)
        sql += " ORDER BY k.start"
        return self._duckdb_query(sql, params)

    def aggregate_kernels(
        self,
        device: int | None,
        trim: tuple[int, int] | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """
        Aggregate kernels by (demangled,name) using SQL GROUP BY.
        If device is None, aggregates across all devices.

        Returns rows sorted by total_ns descending:
          {name, demangled, total_ns, count, avg_ns, min_ns, max_ns}
        """
        sql = """
            SELECT
                s.value AS name,
                d.value AS demangled,
                SUM(k.[end] - k.start) AS total_ns,
                COUNT(*) AS count,
                AVG(k.[end] - k.start) AS avg_ns,
                MIN(k.[end] - k.start) AS min_ns,
                MAX(k.[end] - k.start) AS max_ns
            FROM {kernel_table} k
            JOIN StringIds s ON k.shortName = s.id
            JOIN StringIds d ON k.demangledName = d.id"""
        sql = sql.format(kernel_table=self.schema.kernel_table)
        params: list = []
        if device is not None:
            sql += "\n            WHERE k.deviceId = ?"
            params.append(device)
        else:
            sql += "\n            WHERE 1=1"
        if trim:
            sql += " AND k.start >= ? AND k.[end] <= ?"
            params += list(trim)
        sql += " GROUP BY s.value, d.value"
        # Group keys complete the order: paired with LIMIT, a tie at the
        # cut-off changes which rows survive, not merely their order.
        sql += " ORDER BY total_ns DESC, name ASC, demangled ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return self._duckdb_query(sql, params)

    def aggregate_nvtx_ranges(
        self,
        trim: tuple[int, int] | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """
        Aggregate NVTX ranges by text using SQL GROUP BY.

        Note: This is a *range duration* aggregation (CPU-side wall time of NVTX ranges),
        not "enclosed GPU kernel time". It's intended as a lightweight v1 diff signal.

        Returns rows sorted by total_ns descending:
          {text, total_ns, count, avg_ns}
        """
        # Resolve rather than hardcode: Nsight suffixes this table _V2/_V3 on
        # newer exports. The check this replaced named the table literally, so a
        # _V2 export reported no ranges on a fully annotated profile — silence
        # rather than an error, and the harder failure to notice. An absent
        # table is the other case, and there the empty answer is the true one.
        nvtx_table = self.adapter.resolve_activity_tables().get("nvtx")
        if not nvtx_table:
            return []

        if self._nvtx_has_text_id:
            sql = f"""
                SELECT
                    COALESCE(n.text, s.value) AS text,
                    SUM(n.[end] - n.start) AS total_ns,
                    COUNT(*) AS count,
                    AVG(n.[end] - n.start) AS avg_ns
                FROM {nvtx_table} n
                LEFT JOIN StringIds s ON n.textId = s.id
                WHERE (n.text IS NOT NULL OR s.value IS NOT NULL)
                  AND n.[end] > n.start
            """
        else:
            sql = f"""
                SELECT
                    n.text AS text,
                    SUM(n.[end] - n.start) AS total_ns,
                    COUNT(*) AS count,
                    AVG(n.[end] - n.start) AS avg_ns
                FROM {nvtx_table} n
                WHERE n.text IS NOT NULL
                  AND n.[end] > n.start
            """

        params: list = []
        if trim:
            sql += " AND n.start >= ? AND n.[end] <= ?"
            params += list(trim)
        sql += " GROUP BY COALESCE(n.text, s.value)" if self._nvtx_has_text_id else " GROUP BY text"
        sql += " ORDER BY total_ns DESC, text ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return self._duckdb_query(sql, params)

    def search_nvtx_names(
        self,
        pattern: str,
        limit: int | None = 50,
        use_glob: bool = False,
        trim: tuple[int, int] | None = None,
    ) -> list[dict]:
        """
        Discover NVTX range names by fuzzy match (LIKE or GLOB).

        Use before any region diff so the agent has exact strings.
        pattern: substring to match; for LIKE we wrap with %; for GLOB pass a full pattern.
        Returns rows: {text, total_ns, count} sorted by total_ns descending.
        """
        # Resolve rather than hardcode: Nsight suffixes this table _V2/_V3 on
        # newer exports. The check this replaced named the table literally, so a
        # _V2 export reported no ranges on a fully annotated profile — silence
        # rather than an error, and the harder failure to notice. An absent
        # table is the other case, and there the empty answer is the true one.
        nvtx_table = self.adapter.resolve_activity_tables().get("nvtx")
        if not nvtx_table:
            return []
        match_val = (
            pattern
            if (use_glob and "*" in pattern) or (not use_glob and "%" in pattern)
            else f"%{pattern}%"
            if not use_glob
            else f"*{pattern}*"
        )
        if self._nvtx_has_text_id:
            sql = f"""
                SELECT
                    COALESCE(n.text, s.value) AS text,
                    SUM(n.[end] - n.start) AS total_ns,
                    COUNT(*) AS count
                FROM {nvtx_table} n
                LEFT JOIN StringIds s ON n.textId = s.id
                WHERE (n.text IS NOT NULL OR s.value IS NOT NULL)
                  AND n.[end] > n.start
                  AND COALESCE(n.text, s.value) """
            sql += "GLOB ?" if use_glob else "LIKE ?"
            params: list = [match_val]
        else:
            sql = f"""
                SELECT
                    n.text AS text,
                    SUM(n.[end] - n.start) AS total_ns,
                    COUNT(*) AS count
                FROM {nvtx_table} n
                WHERE n.text IS NOT NULL AND n.[end] > n.start
                  AND n.text """
            sql += "GLOB ?" if use_glob else "LIKE ?"
            params = [match_val]
        if trim:
            sql += " AND n.start >= ? AND n.[end] <= ?"
            params += list(trim)
        sql += " GROUP BY COALESCE(n.text, s.value)" if self._nvtx_has_text_id else " GROUP BY text"
        sql += " ORDER BY total_ns DESC, text ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return self._duckdb_query(sql, params)

    def memcpy_in_window(
        self,
        device: int | None,
        trim: tuple[int, int],
    ) -> dict:
        """
        Sum memcpy time in window by direction (H2D=1, D2H=2, D2D=8).
        ``device=None`` aggregates across all devices (matches ``kernels``).
        Returns {h2d_ns, d2h_ns, d2d_ns, total_ns}; 0 when table or window empty.
        """
        out = {"h2d_ns": 0, "d2h_ns": 0, "d2d_ns": 0, "total_ns": 0}
        memcpy_table = self.adapter.resolve_activity_tables().get("memcpy")
        if not memcpy_table:
            return out
        sql = f"""
            SELECT copyKind, SUM([end] - start) AS total_ns
            FROM {memcpy_table}
            WHERE start >= ? AND [end] <= ?
        """
        params: list = [trim[0], trim[1]]
        if device is not None:
            sql += " AND deviceId = ?"
            params.append(device)
        sql += " GROUP BY copyKind"
        rows = self._duckdb_query(sql, params)
        for r in rows:
            kind = int(r["copyKind"])
            ns = int(r["total_ns"] or 0)
            if kind == 1:
                out["h2d_ns"] = ns
            elif kind == 2:
                out["d2h_ns"] = ns
            elif kind == 8:
                out["d2d_ns"] = ns
            out["total_ns"] += ns
        return out

    def kernel_map(self, device: int) -> dict[int, dict]:
        """Build correlationId -> kernel info for ALL kernels on a device."""
        return {
            r["correlationId"]: dict(
                start=r["start"],
                end=r["end"],
                stream=r["streamId"],
                name=r["name"],
                demangled=r["demangled"],
            )
            for r in self._duckdb_query(
                f"""
                    SELECT k.start, k.[end], k.streamId, k.correlationId,
                           s.value as name, d.value as demangled
                    FROM {self.schema.kernel_table} k
                    JOIN StringIds s ON k.shortName = s.id
                    JOIN StringIds d ON k.demangledName = d.id
                    WHERE k.deviceId = ?  ORDER BY k.start
                """,
                [device],
            )
        }

    def gpu_threads(self, device: int) -> set[int]:
        """Find all CPU threads (globalTid) that launch kernels on this device."""
        runtime_table = self.adapter.resolve_activity_tables().get("runtime")
        if not runtime_table:
            return set()
        return {
            r["globalTid"]
            for r in self._duckdb_query(
                f"""
            SELECT DISTINCT r.globalTid
            FROM {runtime_table} r
            JOIN {self.schema.kernel_table} k ON r.correlationId = k.correlationId
            WHERE k.deviceId = ?
        """,
                [device],
            )
        }

    def runtime_index(self, threads: set[int], window: tuple[int, int]) -> dict[int, list]:
        """Load CUDA runtime calls for threads, indexed by globalTid."""
        runtime_table = self.adapter.resolve_activity_tables().get("runtime")
        if not runtime_table:
            return {}
        idx = {}
        for tid in threads:
            idx[tid] = self._duckdb_query(
                f"""
                SELECT start, [end], correlationId FROM {runtime_table}
                WHERE globalTid = ? AND start >= ? AND [end] <= ?  ORDER BY start
            """,
                [tid, window[0], window[1]],
            )
        return idx

    def nvtx_events(self, threads: set[int], window: tuple[int, int]) -> list:
        """Load NVTX push/pop events for given threads in a time window.

        Handles both schema variants:
          - Legacy: NVTX_EVENTS.text holds the annotation string inline.
          - Newer:  NVTX_EVENTS.textId references StringIds; text may be NULL.
        """
        # Resolve rather than hardcode: Nsight suffixes this table _V2/_V3 on
        # newer exports. The check this replaced named the table literally, so a
        # _V2 export reported no ranges on a fully annotated profile — silence
        # rather than an error, and the harder failure to notice. An absent
        # table is the other case, and there the empty answer is the true one.
        nvtx_table = self.adapter.resolve_activity_tables().get("nvtx")
        if not nvtx_table or not threads:
            return []
        tids = ",".join(map(str, threads))
        if self._nvtx_has_text_id:
            return self._duckdb_query(
                f"""
                SELECT COALESCE(n.text, s.value) AS text,
                       n.globalTid, n.start, n.[end]
                FROM {nvtx_table} n
                LEFT JOIN StringIds s ON n.textId = s.id
                WHERE (n.text IS NOT NULL OR s.value IS NOT NULL)
                  AND n.[end] > n.start
                  AND n.start >= ? AND n.start <= ?
                  AND n.globalTid IN ({tids})
                ORDER BY n.start
            """,
                list(window),
            )
        else:
            return self._duckdb_query(
                f"""
                SELECT text, globalTid, start, [end] FROM {nvtx_table}
                WHERE text IS NOT NULL AND [end] > start
                  AND start >= ? AND start <= ?
                  AND globalTid IN ({tids})
                ORDER BY start
            """,
                list(window),
            )

    def _duckdb_query(self, sql: str, params=None) -> list[dict]:
        """Execute a SQL query via DuckDB, falling back to SQLite.

        Translates SQLite-dialect SQL (``[end]``) to DuckDB (``"end"``).
        Returns results as a list of dicts.
        """
        conn = self.db if self.db is not None else self.conn

        from .connection import DuckDBAdapter, wrap_connection

        # Always wrap the actual connection being used — self.adapter may
        # reference self.conn (SQLite) while conn here is self.db (DuckDB).
        adapter = wrap_connection(conn)

        is_duckdb = isinstance(adapter, DuckDBAdapter)

        if not is_duckdb and not getattr(self, "_warned_sqlite_fallback", False):
            self._log.warning(
                "DuckDB cache unavailable or not in use; falling back to SQLite (slower)"
            )
            self._warned_sqlite_fallback = True

        with self._lock:
            cur = adapter.execute(sql, params or [])
            if is_duckdb:
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
            else:
                return [dict(r) for r in cur.fetchall()]

    def close(self):
        # Drop the per-thread cursors first. They are handles on self.db, so
        # after it closes they raise "Connection already closed" — and being
        # held in a thread-local, a worker thread that outlives the profile
        # would be handed the dead one instead of an error or a fresh handle.
        # Replacing the container clears every thread's slot, not just this
        # thread's, which is the point: the owner is usually the one closing.
        if getattr(self, "_thread_handles", None) is not None:
            self._thread_handles = threading.local()

        # Close the primary connection only if we own it.
        if getattr(self, "_owns_conn", True):
            try:
                self.conn.close()
            except Exception:
                pass

        db = getattr(self, "db", None)
        if db is None:
            return

        if db is self.conn:
            return

        try:
            db.close()
        except Exception:
            pass

    def __enter__(self) -> "Profile":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def resolve_profile(
    path: str,
    *,
    backend: str = "auto",
    ingest_policy: str | None = None,
    nsys_executable: str = "nsys",
) -> ProfileResolution:
    """Resolve one profile input according to the shared ingest policy."""
    path = os.fspath(path)
    if not isinstance(path, str):
        raise TypeError("profile path must be a string or path-like object")
    if not os.path.exists(path):
        raise ProfileNotFoundError(f"profile not found: {path}")

    policy = resolve_ingest_policy(ingest_policy)
    if backend not in {"auto", "sqlite", "parquetdir"}:
        raise ValueError(
            f"Unknown backend: {backend!r}. Expected 'auto', 'sqlite', or 'parquetdir'."
        )
    selected = policy if backend == "auto" else backend
    if os.path.isdir(path) or Path(path).suffix.lower() in {".parquetdir", ".nsys-cache"}:
        try:
            inspect_local_parquetdir(os.path.abspath(path), allow_missing=False)
        except ValueError as exc:
            raise ExportError(str(exc)) from exc
        if selected == "sqlite":
            raise ExportError(
                "SQLite ingest policy cannot open a parquetdir; use NSYS_AI_INGEST=parquetdir."
            )
        return ProfileResolution(path, path, "parquetdir", "parquetdir", "auto")

    if path.lower().endswith(".nsys-rep"):
        if selected in {"auto", "parquetdir"}:
            resolved = _resolve_parquetdir_path(path, nsys_executable=nsys_executable)
            return ProfileResolution(path, resolved, "nsys-rep", "parquetdir", "auto")
        resolved = _resolve_sqlite_path(path, nsys_executable=nsys_executable)
        return ProfileResolution(path, resolved, "nsys-rep", "sqlite", "direct")

    if selected == "parquetdir":
        raise ExportError(
            "Parquetdir ingest requires a parquetdir directory or a .nsys-rep input."
        )
    cache_mode: Literal["auto", "direct"] = "direct" if selected == "sqlite" else "auto"
    return ProfileResolution(path, path, "sqlite", "sqlite", cache_mode)


def resolve_profile_path(
    path: str, *, backend: str = "auto", nsys_executable: str = "nsys"
) -> str:
    """
    Resolve a profile path for the selected backend.

    The default ``auto`` policy resolves `.nsys-rep` to a Parquet directory and
    opens `.sqlite` through the cache/direct fallback. Explicit ``sqlite`` and
    ``parquetdir`` backends remain available for compatibility and diagnostics.

    Exports always pass `--include-blobs=true` so NVTX payload-dependent
    analysis (for example communicator-aware NCCL diagnostics) remains available.

    Raises ProfileNotFoundError when *path* does not exist (e.g. a missing
    `.nsys-rep`, before any export is attempted).
    """
    return resolve_profile(
        path,
        backend=backend,
        nsys_executable=nsys_executable,
    ).resolved_path


def _resolve_sqlite_path(path: str, *, nsys_executable: str = "nsys") -> str:
    """Resolve a ``.nsys-rep`` to its compatibility SQLite export."""

    # Reuse an existing up-to-date SQLite export if possible.
    out = path[:-9] + ".sqlite"  # .nsys-rep -> .sqlite
    reuse_by_mtime = (
        os.path.exists(path)
        and os.path.exists(out)
        and os.path.getsize(out) > 0
        and os.path.getmtime(out) >= os.path.getmtime(path)
    )
    if reuse_by_mtime:
        if not _sqlite_needs_blob_reexport(out):
            return out
        # Missing NVTX payload blobs: re-export only when nsys is available so we
        # do not regress users with a valid sidecar .sqlite but no Nsight install.
        if not shutil.which(nsys_executable):
            logging.getLogger(__name__).warning(
                "Reusing existing SQLite export at %r without NVTX payload blobs; "
                "communicator-aware analysis and other payload-dependent features may be incomplete. "
                "Install Nsight Systems and re-export with: nsys export --type sqlite "
                "--include-blobs=true -o <out.sqlite> --force-overwrite=true <file.nsys-rep>",
                out,
            )
            return out

    nsys_exe = shutil.which(nsys_executable)
    if not nsys_exe:
        raise ExportToolMissingError(
            "Profile is .nsys-rep; conversion requires 'nsys' (NVIDIA Nsight Systems) on PATH. "
            "Install Nsight Systems or export manually: "
            "nsys export --type sqlite --include-blobs=true -o <out.sqlite> "
            "--force-overwrite=true <file.nsys-rep>"
        )

    try:
        # path/out passed as list args to nsys, no shell; caller-controlled paths only
        result = subprocess.run(  # nosec B603
            [
                nsys_exe,
                "export",
                "--type=sqlite",
                # Always explicitly request payload blobs so communicator
                # analysis and other payload-dependent features function.
                "--include-blobs=true",
                "-o",
                out,
                "--force-overwrite=true",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as e:
        raise ExportTimeoutError(
            "nsys export timed out after 300 seconds. This may indicate that nsys is waiting "
            "for interactive input (for example, a license prompt) or that the .nsys-rep file "
            "is corrupted. Try running the export manually to see the full output:\n"
            f"  nsys export --type sqlite --include-blobs=true -o {out} --force-overwrite=true {path}\n"
        ) from e
    except subprocess.CalledProcessError as e:
        raise ExportError(
            f"nsys export failed: {e.stderr or e.stdout or str(e)}. "
            "Export manually: nsys export --type sqlite --include-blobs=true "
            "-o <out.sqlite> --force-overwrite=true <file.nsys-rep>"
        ) from e
    if not (os.path.exists(out) and os.path.getsize(out) > 0):
        stdout = getattr(result, "stdout", None) or "(empty)"
        stderr = getattr(result, "stderr", None) or "(empty)"
        raise ExportError(
            f"nsys export completed without error but did not produce a usable SQLite file at '{out}'. "
            "This may indicate that nsys wrote output elsewhere or hit an unexpected condition.\n"
            f"nsys stdout:\n{stdout}\nnsys stderr:\n{stderr}"
        )
    return out


def _resolve_parquetdir_path(path: str, *, nsys_executable: str = "nsys") -> str:
    """Return a path to an Nsight `parquetdir` export."""
    if os.path.isdir(path):
        try:
            inspect_local_parquetdir(str(os.path.abspath(path)), allow_missing=False)
        except ValueError as exc:
            raise ExportError(str(exc)) from exc
        return path

    if not path.lower().endswith(".nsys-rep"):
        return path

    out = path[:-9] + ".parquetdir"
    if (
        os.path.exists(path)
        and os.path.isdir(out)
        and os.path.getmtime(out) >= os.path.getmtime(path)
    ):
        try:
            inspect_local_parquetdir(str(os.path.abspath(out)), allow_missing=False)
        except ValueError as exc:
            raise ExportError(str(exc)) from exc
        return out

    nsys_exe = shutil.which(nsys_executable)
    if not nsys_exe:
        raise ExportToolMissingError(
            "Profile is .nsys-rep; conversion requires 'nsys' (NVIDIA Nsight Systems) on PATH. "
            "Install Nsight Systems or export manually: "
            "nsys export --type parquetdir --include-blobs=true -o <out.parquetdir> "
            "--force-overwrite=true <file.nsys-rep>"
        )

    try:
        subprocess.run(  # nosec B603
            [
                nsys_exe,
                "export",
                "--type=parquetdir",
                "--include-blobs=true",
                "-o",
                out,
                "--force-overwrite=true",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as e:
        raise ExportTimeoutError(
            "nsys export timed out after 300 seconds while producing a parquetdir export. "
            "Try running the export manually to inspect the full output:\n"
            f"  nsys export --type parquetdir --include-blobs=true --force-overwrite=true -o {out} {path}\n"
        ) from e
    except subprocess.CalledProcessError as e:
        raise ExportError(
            f"nsys parquetdir export failed: {e.stderr or e.stdout or str(e)}. "
            "Export manually: nsys export --type parquetdir --include-blobs=true "
            "-o <out.parquetdir> --force-overwrite=true <file.nsys-rep>"
        ) from e

    try:
        inspect_local_parquetdir(str(os.path.abspath(out)), allow_missing=False)
    except ValueError as exc:
        raise ExportError(
            f"nsys export completed without error: {exc}"
        ) from exc
    return out


def _sqlite_needs_blob_reexport(path: str) -> bool:
    """Check whether a SQLite export is missing NVTX payload schema/blob support.

    This is a capability check (schema/tables present), not a content check.
    Some profiles legitimately contain no NVTX payload rows.
    """
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return True
    try:
        with sqlite3.connect(path) as conn:
            cur = conn.cursor()
            tables = {
                row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "NVTX_EVENTS" not in tables or "NVTX_PAYLOAD_SCHEMAS" not in tables:
                # Missing payload tables means the export was not done with
                # --include-blobs=true → needs re-export.
                return True
            cols = {row[1] for row in cur.execute("PRAGMA table_info(NVTX_EVENTS)")}
            # Require binaryData column presence; row values may legitimately
            # all be NULL on profiles without payload-bearing NVTX ranges.
            return "binaryData" not in cols
    except sqlite3.Error:
        # If the file is unreadable or corrupt, we cannot use it
        return True


def get_first_gpu_name(conn) -> str:
    """Return the first GPU name from TARGET_INFO_GPU (for peak TFLOPS lookup). Empty if tables missing.

    Accepts both sqlite3.Connection and duckdb.DuckDBPyConnection.
    """
    from .connection import wrap_connection

    adapter = wrap_connection(conn)
    tables = adapter.get_table_names()
    if "TARGET_INFO_GPU" not in tables and "gpu_info" not in tables:
        return ""
    if "TARGET_INFO_CUDA_DEVICE" not in tables and "cuda_device" not in tables:
        return ""
    # Use Parquet view names if available, otherwise original SQLite names
    gpu_tbl = "gpu_info" if "gpu_info" in tables else "TARGET_INFO_GPU"
    dev_tbl = "cuda_device" if "cuda_device" in tables else "TARGET_INFO_CUDA_DEVICE"
    row = adapter.execute(f"""
        SELECT g.name
        FROM {gpu_tbl} g
        JOIN {dev_tbl} c ON g.id = c.gpuId
        ORDER BY c.cudaId
        LIMIT 1
    """).fetchone()
    return (row[0] or "").strip() if row else ""


def open(
    path: str,
    *,
    backend: str = "auto",
    cache_mode: str = "auto",
    ingest_policy: str | None = None,
    progress: typing.Callable[[str, int, int], None] | None = None,
) -> Profile:
    """Open an Nsight Systems profile using the requested backend."""
    return Profile(
        path,
        cache_mode=cache_mode,
        backend=backend,
        ingest_policy=ingest_policy,
        progress=progress,
    )
