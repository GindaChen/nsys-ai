"""
profile.py — Open and query Nsight Systems SQLite databases.

Provides a thin wrapper around the SQLite export with typed accessors
for kernels, NVTX events, CUDA runtime calls, and metadata.
"""
import os
import shutil
import sqlite3
import subprocess  # nosec B404 — only for nsys export .nsys-rep→.sqlite, list args no shell
from dataclasses import dataclass, field
from typing import Optional, Dict, List


class NsightSchema:
    """
    Lightweight schema/metadata helper for Nsight Systems SQLite exports.

    Detects available tables, attempts to infer the Nsight Systems version,
    and exposes canonical table choices (e.g., kernel activity table).
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self.tables: List[str] = [
            r[0] for r in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]
        self.version: Optional[str] = self._detect_version()
        self.kernel_table: Optional[str] = self._detect_kernel_table()

    # ── Version detection ──────────────────────────────────────────────

    def _read_kv_table(self, table: str) -> Dict[str, str]:
        """
        Best-effort reader for META_DATA_* style tables which may use
        slightly different column names across Nsight versions.
        """
        if table not in self.tables:
            return {}

        cur = self._conn.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cur.fetchall()]  # 1 = name
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

        kv: Dict[str, str] = {}
        cur = self._conn.execute(
            f"SELECT {key_col}, {val_col} FROM {table}"
        )
        for k, v in cur.fetchall():
            if k is not None and v is not None:
                kv[str(k)] = str(v)
        return kv

    def _detect_version(self) -> Optional[str]:
        """Try to infer Nsight Systems version from META_DATA tables."""
        meta: Dict[str, str] = {}
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

    def _detect_kernel_table(self) -> Optional[str]:
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
            t for t in self.tables
            if "KERNEL" in t.upper() and not t.upper().startswith("ENUM_")
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
    devices: list[int]          # active deviceIds
    streams: dict[int, list[int]]  # deviceId -> [streamId, ...]
    time_range: tuple[int, int]    # (min_start_ns, max_end_ns)
    kernel_count: int
    nvtx_count: int
    tables: list[str]
    gpu_info: dict[int, GpuInfo] = field(default_factory=dict)  # deviceId -> GpuInfo


class Profile:
    """Handle to an opened Nsight Systems SQLite database."""

    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.schema = NsightSchema(self.conn)
        self.meta = self._discover()

    def _discover(self) -> ProfileMeta:
        tables = self.schema.tables

        if not self.schema.kernel_table:
            ver_msg = f" (Nsight version: {self.schema.version})" if self.schema.version else ""
            raise RuntimeError(
                "This profile does not contain GPU kernel activity "
                f"(no suitable KERNEL table found){ver_msg}. "
                "It may have been captured without CUDA kernel tracing, "
                "or exported with a schema layout this version of nsys-ai "
                "does not yet understand."
            )

        kernel_table = self.schema.kernel_table

        devices = [r[0] for r in self.conn.execute(
            f"SELECT DISTINCT deviceId FROM {kernel_table} ORDER BY deviceId")]

        streams: dict[int, list[int]] = {}
        for r in self.conn.execute(
            f"SELECT DISTINCT deviceId, streamId FROM {kernel_table} "
            "ORDER BY deviceId, streamId"):
            streams.setdefault(r[0], []).append(r[1])

        tr = self.conn.execute(
            f"SELECT MIN(start), MAX([end]) FROM {kernel_table}").fetchone()

        kc = self.conn.execute(
            f"SELECT COUNT(*) FROM {kernel_table}").fetchone()[0]
        nc = self.conn.execute("SELECT COUNT(*) FROM NVTX_EVENTS").fetchone()[0] if "NVTX_EVENTS" in tables else 0

        return ProfileMeta(
            devices=devices, streams=streams,
            time_range=(tr[0] or 0, tr[1] or 0),
            kernel_count=kc, nvtx_count=nc, tables=tables,
            gpu_info=self._gpu_info(devices, streams, tables))

    def _gpu_info(self, devices, streams, tables) -> dict[int, GpuInfo]:
        """Query hardware metadata per GPU."""
        info: dict[int, GpuInfo] = {}

        # Kernel counts per device
        kcounts = {}
        for r in self.conn.execute(
            f"SELECT deviceId, COUNT(*) FROM {self.schema.kernel_table} GROUP BY deviceId"):
            kcounts[r[0]] = r[1]

        # Hardware info from TARGET_INFO_GPU + TARGET_INFO_CUDA_DEVICE
        hw = {}
        if "TARGET_INFO_GPU" in tables and "TARGET_INFO_CUDA_DEVICE" in tables:
            for r in self.conn.execute("""
                SELECT c.cudaId as dev, g.name, g.busLocation,
                       g.smCount as sms, g.totalMemory as mem,
                       g.chipName, g.memoryBandwidth as bw
                FROM TARGET_INFO_GPU g
                JOIN TARGET_INFO_CUDA_DEVICE c ON g.id = c.gpuId
                GROUP BY c.cudaId
            """):
                hw[r["dev"]] = dict(name=r["name"] or "", pci_bus=r["busLocation"] or "",
                                    sm_count=r["sms"] or 0, memory_bytes=r["mem"] or 0)

        for dev in devices:
            h = hw.get(dev, {})
            info[dev] = GpuInfo(
                device_id=dev, name=h.get("name", ""), pci_bus=h.get("pci_bus", ""),
                sm_count=h.get("sm_count", 0), memory_bytes=h.get("memory_bytes", 0),
                kernel_count=kcounts.get(dev, 0),
                streams=streams.get(dev, []))
        return info

    def kernels(self, device: int, trim: Optional[tuple[int, int]] = None) -> list[dict]:
        """All kernels on a device, optionally trimmed to a time window."""
        sql = """
            SELECT k.start, k.[end], k.streamId, k.correlationId, s.value as name
            FROM {kernel_table} k
            JOIN StringIds s ON k.shortName = s.id
            WHERE k.deviceId = ?"""
        sql = sql.format(kernel_table=self.schema.kernel_table)
        params: list = [device]
        if trim:
            sql += " AND k.start >= ? AND k.[end] <= ?"
            params += list(trim)
        sql += " ORDER BY k.start"
        return [dict(r) for r in self.conn.execute(sql, params)]

    def kernel_map(self, device: int) -> dict[int, dict]:
        """Build correlationId -> kernel info for ALL kernels on a device."""
        return {r["correlationId"]: dict(start=r["start"], end=r["end"],
                stream=r["streamId"], name=r["name"],
                demangled=r["demangled"])
                for r in self.conn.execute("""
                    SELECT k.start, k.[end], k.streamId, k.correlationId,
                           s.value as name, d.value as demangled
                    FROM {kernel_table} k
                    JOIN StringIds s ON k.shortName = s.id
                    JOIN StringIds d ON k.demangledName = d.id
                    WHERE k.deviceId = ?  ORDER BY k.start
                """.format(kernel_table=self.schema.kernel_table), (device,))}

    def gpu_threads(self, device: int) -> set[int]:
        """Find all CPU threads (globalTid) that launch kernels on this device."""
        return {r[0] for r in self.conn.execute("""
            SELECT DISTINCT r.globalTid
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN {kernel_table} k ON r.correlationId = k.correlationId
            WHERE k.deviceId = ?
        """.format(kernel_table=self.schema.kernel_table), (device,))}

    def runtime_index(self, threads: set[int],
                      window: tuple[int, int]) -> dict[int, list]:
        """Load CUDA runtime calls for threads, indexed by globalTid."""
        idx = {}
        for tid in threads:
            idx[tid] = self.conn.execute("""
                SELECT start, [end], correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME
                WHERE globalTid = ? AND start >= ? AND [end] <= ?  ORDER BY start
            """, (tid, window[0], window[1])).fetchall()
        return idx

    def nvtx_events(self, threads: set[int],
                    window: tuple[int, int]) -> list:
        """Load NVTX push/pop events for given threads in a time window."""
        return self.conn.execute("""
            SELECT text, globalTid, start, [end] FROM NVTX_EVENTS
            WHERE text IS NOT NULL AND [end] > start
              AND start >= ? AND start <= ?
              AND globalTid IN ({})
            ORDER BY start
        """.format(",".join(map(str, threads))), window).fetchall()

    def close(self):
        self.conn.close()


def resolve_profile_path(path: str) -> str:
    """
    Return a path to a .sqlite profile. If path is .nsys-rep, export via
    `nsys export --type sqlite` and return the path to the resulting .sqlite.
    (NVIDIA Nsight Systems exporter: docs.nvidia.com/nsight-systems/nsys-exporter)
    """
    if not path.lower().endswith(".nsys-rep"):
        return path

    # Reuse an existing up-to-date SQLite export if possible.
    out = path[:-9] + ".sqlite"  # .nsys-rep -> .sqlite
    if (
        os.path.exists(path)
        and os.path.exists(out)
        and os.path.getsize(out) > 0
        and os.path.getmtime(out) >= os.path.getmtime(path)
    ):
        return out

    nsys_exe = shutil.which("nsys")
    if not nsys_exe:
        raise RuntimeError(
            "Profile is .nsys-rep; conversion requires 'nsys' (NVIDIA Nsight Systems) on PATH. "
            "Install Nsight Systems or export manually: nsys export --type sqlite -o <out.sqlite> --force-overwrite true <file.nsys-rep>"
        )

    try:
        # path/out passed as list args to nsys, no shell; caller-controlled paths only
        result = subprocess.run(  # nosec B603
            [nsys_exe, "export", "--type=sqlite", "-o", out, "--force-overwrite=true", path],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "nsys export timed out after 300 seconds. This may indicate that nsys is waiting "
            "for interactive input (for example, a license prompt) or that the .nsys-rep file "
            "is corrupted. Try running the export manually to see the full output:\n"
            f"  nsys export --type sqlite -o {out} {path}\n"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"nsys export failed: {e.stderr or e.stdout or str(e)}. "
            "Export manually: nsys export --type sqlite -o <out.sqlite> --force-overwrite true <file.nsys-rep>"
        )
    if not (os.path.exists(out) and os.path.getsize(out) > 0):
        stdout = getattr(result, "stdout", None) or "(empty)"
        stderr = getattr(result, "stderr", None) or "(empty)"
        raise RuntimeError(
            f"nsys export completed without error but did not produce a usable SQLite file at '{out}'. "
            "This may indicate that nsys wrote output elsewhere or hit an unexpected condition.\n"
            f"nsys stdout:\n{stdout}\nnsys stderr:\n{stderr}"
        )
    return out


def open(path: str) -> Profile:
    """Open an Nsight Systems SQLite database."""
    path = resolve_profile_path(path)
    # Heuristic: if the given path is an empty .sqlite stub but a sibling
    # file without the .sqlite suffix exists and is a non-empty SQLite DB,
    # prefer the sibling. This helps when users accidentally point nsys-ai
    # at a placeholder file instead of the real Nsight export.
    if path.endswith(".sqlite") and os.path.exists(path) and not os.path.getsize(path):
        base = path[:-7]
        if os.path.exists(base) and os.path.getsize(base) > 0:
            path = base

    return Profile(path)
