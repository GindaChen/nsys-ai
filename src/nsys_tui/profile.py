"""
profile.py — Open and query Nsight Systems SQLite databases.

Provides a thin wrapper around the SQLite export with typed accessors
for kernels, NVTX events, CUDA runtime calls, and metadata.
"""
import sqlite3
from dataclasses import dataclass, field
from typing import Optional


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
        self.meta = self._discover()

    def _discover(self) -> ProfileMeta:
        tables = [r[0] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]

        devices = [r[0] for r in self.conn.execute(
            "SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY deviceId")]

        streams: dict[int, list[int]] = {}
        for r in self.conn.execute(
            "SELECT DISTINCT deviceId, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY deviceId, streamId"):
            streams.setdefault(r[0], []).append(r[1])

        tr = self.conn.execute(
            "SELECT MIN(start), MAX([end]) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()

        kc = self.conn.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
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
            "SELECT deviceId, COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY deviceId"):
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
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName = s.id
            WHERE k.deviceId = ?"""
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
                    FROM CUPTI_ACTIVITY_KIND_KERNEL k
                    JOIN StringIds s ON k.shortName = s.id
                    JOIN StringIds d ON k.demangledName = d.id
                    WHERE k.deviceId = ?  ORDER BY k.start
                """, (device,))}

    def gpu_threads(self, device: int) -> set[int]:
        """Find all CPU threads (globalTid) that launch kernels on this device."""
        return {r[0] for r in self.conn.execute("""
            SELECT DISTINCT r.globalTid
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON r.correlationId = k.correlationId
            WHERE k.deviceId = ?
        """, (device,))}

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


def open(path: str) -> Profile:
    """Open an Nsight Systems SQLite database."""
    return Profile(path)
