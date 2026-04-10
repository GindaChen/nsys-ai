"""
profile.py — Open and query Nsight Systems SQLite databases.

Provides a thin wrapper around the SQLite export with typed accessors
for kernels, NVTX events, CUDA runtime calls, and metadata.
"""

import functools
import logging
import os
import re
import shutil
import sqlite3
import subprocess  # nosec B404 — only for nsys export .nsys-rep→.sqlite, list args no shell
import threading
import typing
from dataclasses import dataclass, field

if typing.TYPE_CHECKING:
    from .fingerprint import ProfileFingerprint


from nsys_ai import parquet_cache
from nsys_ai.exceptions import (
    ExportError,
    ExportTimeoutError,
    ExportToolMissingError,
    SchemaError,
)

# Regex for safe SQL identifiers (table/column names).
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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

    def __init__(self, conn):
        self._conn = conn
        from .connection import wrap_connection

        self._adapter = wrap_connection(conn)

        self.tables = list(self._adapter.get_table_names())

        self.version: str | None = self._detect_version()
        kt = self._detect_kernel_table()
        self.kernel_table: str | None = _validate_table_name(kt) if kt else None

    # ── Version detection ──────────────────────────────────────────────

    def _read_kv_table(self, table: str) -> dict[str, str]:
        """
        Best-effort reader for META_DATA_* style tables which may use
        slightly different column names across Nsight versions.
        """
        if table not in self.tables:
            return {}

        cols = self._adapter.get_table_columns(table)
        key_col = None
        val_col = None

        # Common patterns seen in Nsight exports
        for cand in ("key", "Key", "NAME", "Name"):
            if cand in cols:
                key_col = cand
                break
        for cand in ("value", "Value", "VAL", "Val"):
            if cand in cols:
                val_col = cand
                break

        if not key_col or not val_col:
            return {}

        kv: dict[str, str] = {}
        cur = self._adapter.execute(f"SELECT {key_col}, {val_col} FROM {table}")
        for k, v in cur.fetchall():
            if k is not None and v is not None:
                kv[str(k)] = str(v)
        return kv

    def _detect_version(self) -> str | None:
        """Try to infer Nsight Systems version from META_DATA tables."""
        meta: dict[str, str] = {}
        for table in ("META_DATA_EXPORT", "META_DATA_CAPTURE"):
            meta.update(self._read_kv_table(table))

        # Heuristic keys that might carry version information
        for key in meta:
            lk = key.lower()
            if "nsight systems version" in lk or "exporter version" in lk:
                return meta[key]
        # Fallback: sometimes the value itself contains 'Nsight Systems X.Y'
        for val in meta.values():
            if "Nsight Systems" in val:
                return val
        return None

    # ── Table detection ────────────────────────────────────────────────

    def _detect_kernel_table(self) -> str | None:
        """
        Pick an appropriate kernel activity table, if present.

        Today this is usually CUPTI_ACTIVITY_KIND_KERNEL, but we keep
        the detection logic resilient to future renames.
        """
        # Preferred legacy/known name
        if "CUPTI_ACTIVITY_KIND_KERNEL" in self.tables:
            return "CUPTI_ACTIVITY_KIND_KERNEL"

        # Fallback: any non-enum table with KERNEL in the name
        candidates = [
            t for t in self.tables if "KERNEL" in t.upper() and not t.upper().startswith("ENUM_")
        ]
        if candidates:
            # Deterministic order
            candidates.sort()
            return candidates[0]

        return None


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


class Profile:
    """Handle to an opened Nsight Systems SQLite database.

    Exposes two database connections:
      - ``self.conn`` (sqlite3.Connection | duckdb.DuckDBPyConnection): the primary
        backend connection. For ``backend='sqlite'`` this is the original SQLite DB;
        for ``backend='parquetdir'`` this is the DuckDB connection over Parquet.
      - ``self.db`` (duckdb.DuckDBPyConnection): DuckDB over Parquet cache — the
        primary query path for all analytical queries.
    """

    _log = logging.getLogger(__name__)

    def __init__(self, path: str, *, cache_mode: str = "auto", backend: str = "sqlite"):
        if cache_mode not in ("auto", "parquet", "direct"):
            raise ValueError(
                f"Unknown cache_mode: {cache_mode!r}. Expected 'auto', 'parquet', or 'direct'."
            )
        if backend not in ("sqlite", "parquetdir"):
            raise ValueError(f"Unknown backend: {backend!r}. Expected 'sqlite' or 'parquetdir'.")
        if backend == "parquetdir" and cache_mode != "auto":
            raise ValueError(
                "cache_mode is not supported with backend='parquetdir'; use cache_mode='auto'."
            )
        self.path = path
        self.backend = backend
        self._lock = threading.Lock()
        self._owns_conn = True
        if backend == "parquetdir":
            self.db = parquet_cache.open_parquetdir_db(path)
            self.conn = self.db
        else:
            self.conn = sqlite3.connect(path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            try:
                if cache_mode == "direct":
                    # Force direct SQLite via DuckDB — zero ETL, instant startup
                    self.db = parquet_cache.open_direct_sqlite(path)
                elif cache_mode == "parquet":
                    # Original behaviour: block until cache is built
                    self.db = parquet_cache.open_cached_db(path)
                else:
                    # "auto" mode: use cache if valid, else smart fallback
                    if parquet_cache.is_cache_valid(path):
                        self.db = parquet_cache.open_cached_db(path)
                    else:
                        size_mb = os.path.getsize(path) / 1e6
                        if size_mb > 50:
                            self._log.info(
                                "Large profile (%.0fMB), using direct query mode (instant startup).",
                                size_mb,
                            )
                            self._log.info(
                                "To build a Parquet cache for faster repeated queries, re-run with "
                                "cache_mode='parquet' or pre-build the cache."
                            )
                            self.db = parquet_cache.open_direct_sqlite(path)
                        else:
                            # Small file — build cache now (seconds)
                            self.db = parquet_cache.open_cached_db(path)
            except Exception as e:
                self._log.warning("DuckDB cache unavailable, falling back to SQLite: %s", e)
                self.db = None  # type: ignore[assignment]
        from .connection import wrap_connection

        self.adapter = wrap_connection(self.db if self.db is not None else self.conn)
        self.schema = NsightSchema(self.db if self.db is not None else self.conn)
        self.meta = self._discover()
        self._nvtx_has_text_id = self.adapter.detect_nvtx_text_id()

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

        devices = [
            r[0]
            for r in self.adapter.execute(
                f"SELECT DISTINCT deviceId FROM {kernel_table} ORDER BY deviceId"
            ).fetchall()
        ]

        streams: dict[int, list[int]] = {}
        for r in self.adapter.execute(
            f"SELECT DISTINCT deviceId, streamId FROM {kernel_table} ORDER BY deviceId, streamId"
        ).fetchall():
            streams.setdefault(r[0], []).append(r[1])

        tr = self.adapter.execute(f"SELECT MIN(start), MAX([end]) FROM {kernel_table}").fetchone()

        kc = self.adapter.execute(f"SELECT COUNT(*) FROM {kernel_table}").fetchone()[0]
        nc = (
            self.adapter.execute("SELECT COUNT(*) FROM NVTX_EVENTS").fetchone()[0]
            if "NVTX_EVENTS" in tables
            else 0
        )

        return ProfileMeta(
            devices=devices,
            streams=streams,
            time_range=(tr[0] or 0, tr[1] or 0),
            kernel_count=kc,
            nvtx_count=nc,
            tables=tables,
            gpu_info=self._gpu_info(devices, streams, tables),
        )

    def _gpu_info(self, devices, streams, tables) -> dict[int, GpuInfo]:
        """Query hardware metadata per GPU."""
        info: dict[int, GpuInfo] = {}

        # Kernel counts per device
        kcounts = {}
        query_conn = self.db if self.db is not None else self.conn
        for r in query_conn.execute(
            f"SELECT deviceId, COUNT(*) FROM {self.schema.kernel_table} GROUP BY deviceId"
        ).fetchall():
            kcounts[r[0]] = r[1]

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
            SELECT k.start, k.[end], k.streamId, k.correlationId,
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
        sql += " ORDER BY total_ns DESC"
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
        if "NVTX_EVENTS" not in self.schema.tables:
            return []

        if self._nvtx_has_text_id:
            sql = """
                SELECT
                    COALESCE(n.text, s.value) AS text,
                    SUM(n.[end] - n.start) AS total_ns,
                    COUNT(*) AS count,
                    AVG(n.[end] - n.start) AS avg_ns
                FROM NVTX_EVENTS n
                LEFT JOIN StringIds s ON n.textId = s.id
                WHERE (n.text IS NOT NULL OR s.value IS NOT NULL)
                  AND n.[end] > n.start
            """
        else:
            sql = """
                SELECT
                    n.text AS text,
                    SUM(n.[end] - n.start) AS total_ns,
                    COUNT(*) AS count,
                    AVG(n.[end] - n.start) AS avg_ns
                FROM NVTX_EVENTS n
                WHERE n.text IS NOT NULL
                  AND n.[end] > n.start
            """

        params: list = []
        if trim:
            sql += " AND n.start >= ? AND n.[end] <= ?"
            params += list(trim)
        sql += " GROUP BY text"
        sql += " ORDER BY total_ns DESC"
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
        if "NVTX_EVENTS" not in self.schema.tables:
            return []
        match_val = (
            pattern
            if (use_glob and "*" in pattern) or (not use_glob and "%" in pattern)
            else f"%{pattern}%"
            if not use_glob
            else f"*{pattern}*"
        )
        if self._nvtx_has_text_id:
            sql = """
                SELECT
                    COALESCE(n.text, s.value) AS text,
                    SUM(n.[end] - n.start) AS total_ns,
                    COUNT(*) AS count
                FROM NVTX_EVENTS n
                LEFT JOIN StringIds s ON n.textId = s.id
                WHERE (n.text IS NOT NULL OR s.value IS NOT NULL)
                  AND n.[end] > n.start
                  AND COALESCE(n.text, s.value) """
            sql += "GLOB ?" if use_glob else "LIKE ?"
            params: list = [match_val]
        else:
            sql = """
                SELECT
                    n.text AS text,
                    SUM(n.[end] - n.start) AS total_ns,
                    COUNT(*) AS count
                FROM NVTX_EVENTS n
                WHERE n.text IS NOT NULL AND n.[end] > n.start
                  AND n.text """
            sql += "GLOB ?" if use_glob else "LIKE ?"
            params = [match_val]
        if trim:
            sql += " AND n.start >= ? AND n.[end] <= ?"
            params += list(trim)
        sql += " GROUP BY text ORDER BY total_ns DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return self._duckdb_query(sql, params)

    def memcpy_in_window(
        self,
        device: int,
        trim: tuple[int, int],
    ) -> dict:
        """
        Sum memcpy time in window by direction (H2D=1, D2H=2, D2D=8).
        Returns {h2d_ns, d2h_ns, d2d_ns, total_ns}; 0 when table or window empty.
        """
        out = {"h2d_ns": 0, "d2h_ns": 0, "d2d_ns": 0, "total_ns": 0}
        if "CUPTI_ACTIVITY_KIND_MEMCPY" not in self.schema.tables:
            return out
        rows = self._duckdb_query(
            """
            SELECT copyKind, SUM([end] - start) AS total_ns
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE deviceId = ? AND start >= ? AND [end] <= ?
            GROUP BY copyKind
            """,
            [device, trim[0], trim[1]],
        )
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
        return {
            r["globalTid"]
            for r in self._duckdb_query(
                f"""
            SELECT DISTINCT r.globalTid
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN {self.schema.kernel_table} k ON r.correlationId = k.correlationId
            WHERE k.deviceId = ?
        """,
                [device],
            )
        }

    def runtime_index(self, threads: set[int], window: tuple[int, int]) -> dict[int, list]:
        """Load CUDA runtime calls for threads, indexed by globalTid."""
        idx = {}
        for tid in threads:
            idx[tid] = self._duckdb_query(
                """
                SELECT start, [end], correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME
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
        if "NVTX_EVENTS" not in self.schema.tables or not threads:
            return []
        tids = ",".join(map(str, threads))
        if self._nvtx_has_text_id:
            return self._duckdb_query(
                f"""
                SELECT COALESCE(n.text, s.value) AS text,
                       n.globalTid, n.start, n.[end]
                FROM NVTX_EVENTS n
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
                SELECT text, globalTid, start, [end] FROM NVTX_EVENTS
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


def resolve_profile_path(path: str, *, backend: str = "sqlite") -> str:
    """
    Resolve a profile path for the selected backend.

    `backend='sqlite'` returns a `.sqlite` file, exporting from `.nsys-rep`
    when needed. `backend='parquetdir'` returns a Parquet directory, exporting
    from `.nsys-rep` when needed.

    Exports always pass `--include-blobs=true` so NVTX payload-dependent
    analysis (for example communicator-aware NCCL diagnostics) remains available.
    """
    if backend == "parquetdir":
        return _resolve_parquetdir_path(path)
    if not path.lower().endswith(".nsys-rep"):
        return path

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
        if not shutil.which("nsys"):
            logging.getLogger(__name__).warning(
                "Reusing existing SQLite export at %r without NVTX payload blobs; "
                "communicator-aware analysis and other payload-dependent features may be incomplete. "
                "Install Nsight Systems and re-export with: nsys export --type sqlite "
                "--include-blobs=true -o <out.sqlite> --force-overwrite=true <file.nsys-rep>",
                out,
            )
            return out

    nsys_exe = shutil.which("nsys")
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


def _resolve_parquetdir_path(path: str) -> str:
    """Return a path to an Nsight `parquetdir` export."""
    if os.path.isdir(path):
        parquet_files = [name for name in os.listdir(path) if name.endswith(".parquet")]
        if parquet_files:
            return path
        raise ExportError(f"Parquet directory '{path}' does not contain any .parquet files.")

    if not path.lower().endswith(".nsys-rep"):
        return path

    out = path[:-9] + ".parquetdir"
    if (
        os.path.exists(path)
        and os.path.isdir(out)
        and any(name.endswith(".parquet") for name in os.listdir(out))
        and os.path.getmtime(out) >= os.path.getmtime(path)
    ):
        return out

    nsys_exe = shutil.which("nsys")
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

    if not (os.path.isdir(out) and any(name.endswith(".parquet") for name in os.listdir(out))):
        raise ExportError(
            f"nsys export completed without error but did not produce a usable parquetdir at '{out}'."
        )
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


def open(path: str, *, backend: str = "sqlite", cache_mode: str = "auto") -> Profile:
    """Open an Nsight Systems profile using the requested backend."""
    path = resolve_profile_path(path, backend=backend)
    # Heuristic: if the given path is an empty .sqlite stub but a sibling
    # file without the .sqlite suffix exists and is a non-empty SQLite DB,
    # prefer the sibling. This helps when users accidentally point nsys-ai
    # at a placeholder file instead of the real Nsight export.
    if backend == "sqlite" and path.endswith(".sqlite") and os.path.exists(path) and not os.path.getsize(path):
        base = path[:-7]
        if os.path.exists(base) and os.path.getsize(base) > 0:
            path = base

    return Profile(path, cache_mode=cache_mode, backend=backend)
