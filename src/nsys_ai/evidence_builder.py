"""
evidence_builder.py — Convert profile analysis into visual Finding overlays.

Each method queries individual kernel instances (not aggregates)
to produce findings with exact nanosecond timestamps for timeline overlay.
"""

import logging
import statistics

from .annotation import EvidenceReport, Finding
from .profile import Profile

_log = logging.getLogger(__name__)


class EvidenceBuilder:
    """Generates findings from a profile using direct SQL queries.

    Usage::

        with Profile("profile.sqlite") as prof:
            builder = EvidenceBuilder(prof, device=0)
            report = builder.build()
            # report.findings is a list of Finding objects
    """

    def __init__(
        self,
        prof: Profile,
        device: int = 0,
        trim: tuple[int, int] | None = None,
    ):
        self.prof = prof
        self.device = device
        self.trim = trim or tuple(prof.meta.time_range)

    def build(self) -> EvidenceReport:
        """Run all analyzers and return a combined EvidenceReport."""
        findings: list[Finding] = []
        findings += self._slow_iterations()
        findings += self._gpu_idle_gaps()
        findings += self._nccl_stalls()
        findings += self._kernel_hotspots()
        findings += self._overlap_ratio()
        findings += self._memory_anomalies()
        findings += self._h2d_spikes()
        return EvidenceReport(
            title="Auto-Analysis",
            profile_path=getattr(self.prof, "path", ""),
            findings=findings,
        )

    # ------------------------------------------------------------------
    # Analyzers
    # ------------------------------------------------------------------

    def _overlap_ratio(self) -> list[Finding]:
        """Flag poor compute/NCCL overlap and communication dominance."""
        from .overlap import overlap_analysis

        result = overlap_analysis(self.prof, self.device, self.trim)
        if "error" in result:
            return []

        findings = []
        nccl_ms = result.get("nccl_only_ms", 0) + result.get("overlap_ms", 0)
        compute_ms = result.get("compute_only_ms", 0)
        overlap_pct = result.get("overlap_pct", 0)
        total_ms = result.get("total_ms", 1)

        # Low overlap: NCCL not well hidden behind compute
        if nccl_ms > 0 and overlap_pct < 30:
            note = (
                f"Only {overlap_pct}% of NCCL time overlaps with compute. "
                f"NCCL-only: {result['nccl_only_ms']:.1f}ms out of "
                f"{total_ms:.1f}ms total span."
            )
            # Try same-stream diagnosis for richer note
            try:
                kernel_tbl = self.prof.schema.kernel_table
                if kernel_tbl:
                    same_stream = self.prof._duckdb_query(
                        f"""
                        SELECT k.streamId,
                            SUM(CASE WHEN s.value LIKE '%nccl%' OR s.value LIKE '%NCCL%'
                                THEN 1 ELSE 0 END) AS nccl_count,
                            SUM(CASE WHEN NOT (s.value LIKE '%nccl%' OR s.value LIKE '%NCCL%')
                                THEN 1 ELSE 0 END) AS compute_count
                        FROM {kernel_tbl} k
                        JOIN StringIds s ON k.shortName = s.id
                        WHERE k.deviceId = ?
                        GROUP BY k.streamId
                        HAVING nccl_count > 0 AND compute_count > 0
                        """,
                        (self.device,),
                    )
                    if same_stream:
                        streams = [str(r["streamId"]) for r in same_stream]
                        note += (
                            f" Streams [{', '.join(streams)}] run both NCCL and "
                            f"compute — overlap is impossible on same stream."
                        )
            except Exception:
                _log.debug("Same-stream enrichment query failed", exc_info=True)

            findings.append(
                Finding(
                    type="region",
                    label=f"Low Compute/NCCL Overlap ({overlap_pct}%)",
                    start_ns=self.trim[0],
                    end_ns=self.trim[1],
                    gpu_id=self.device,
                    severity="warning",
                    note=note,
                )
            )

        # Communication dominated: NCCL > compute
        if nccl_ms > 0 and compute_ms > 0:
            ratio = compute_ms / nccl_ms
            if ratio < 0.5:
                findings.append(
                    Finding(
                        type="region",
                        label=f"Communication Dominated (ratio={ratio:.2f})",
                        start_ns=self.trim[0],
                        end_ns=self.trim[1],
                        gpu_id=self.device,
                        severity="critical",
                        note=(
                            f"Compute/Communication ratio is {ratio:.2f} "
                            f"(healthy > 2.0). Compute: {compute_ms:.1f}ms, "
                            f"NCCL: {nccl_ms:.1f}ms. Consider reducing "
                            f"tensor parallelism degree."
                        ),
                    )
                )

        return findings

    def _slow_iterations(self) -> list[Finding]:
        """Iterations with duration >1.5× median → region findings."""
        from .overlap import detect_iterations

        iters = detect_iterations(self.prof, self.device, self.trim)
        if len(iters) < 3:
            return []
        durs = [it["duration_ms"] for it in iters]
        med = statistics.median(durs)
        if med <= 0:
            return []
        findings = []
        for it in iters:
            if it["duration_ms"] > 1.5 * med:
                pct = 100 * it["duration_ms"] / med
                findings.append(
                    Finding(
                        type="region",
                        label=f"Slow Iteration {it['iteration']}",
                        start_ns=int(it["gpu_start_s"] * 1e9),
                        end_ns=int(it["gpu_end_s"] * 1e9),
                        gpu_id=self.device,
                        severity="warning",
                        note=(
                            f"{it['duration_ms']:.1f}ms "
                            f"({pct:.0f}% of median {med:.1f}ms), "
                            f"{it['kernel_count']} kernels"
                        ),
                    )
                )
        return findings

    def _gpu_idle_gaps(self, top_n: int = 5, min_gap_ns: int = 1_000_000) -> list[Finding]:
        """Top N idle gaps between consecutive kernels → region findings.

        Enriched with total idle time statistics and CPU attribution
        from CUDA Runtime API calls during each gap window.
        """
        sql = f"""\
WITH ordered AS (
    SELECT k.streamId, k.deviceId,
           k.start, k.[end],
           LAG(k.[end]) OVER (
               PARTITION BY k.streamId ORDER BY k.start
           ) AS prev_end
    FROM {self.prof.schema.kernel_table} k
    WHERE k.deviceId = ? AND k.[end] >= ? AND k.start <= ?
)
SELECT streamId, deviceId, prev_end AS gap_start, start AS gap_end,
       (start - prev_end) AS gap_ns
FROM ordered
WHERE prev_end IS NOT NULL AND (start - prev_end) > ?
ORDER BY gap_ns DESC
LIMIT ?"""
        rows = self.prof._duckdb_query(
            sql,
            (self.device, self.trim[0], self.trim[1], min_gap_ns, top_n),
        )

        # Compute total idle stats for note enrichment
        total_idle_ns = sum(r["gap_ns"] for r in rows)
        profile_span = self.trim[1] - self.trim[0]
        # Normalize by number of active streams so percentage is not misleading
        # on multi-stream workloads.  Query the kernel table directly instead of
        # deriving from the limited top-N rows (which would undercount streams).
        try:
            active_streams_rows = self.prof._duckdb_query(
                    f"""
                    SELECT COUNT(DISTINCT k.streamId) AS n
                    FROM {self.prof.schema.kernel_table} k
                    WHERE k.deviceId = ? AND k.[end] >= ? AND k.start <= ?
                    """,
                    (self.device, self.trim[0], self.trim[1]),
                )
            active_streams = (active_streams_rows[0]["n"] or 0) if active_streams_rows else 0
            active_streams = active_streams or 1
        except Exception:
            _log.debug("Active streams query failed", exc_info=True)
            active_streams = 1
        pct = (
            round(100 * total_idle_ns / (profile_span * active_streams), 1)
            if profile_span > 0
            else 0
        )

        findings = []
        for r in rows:
            gap_ms = r["gap_ns"] / 1e6
            note = f"Stream {r['streamId']}: {gap_ms:.2f}ms idle"

            # CPU attribution: find dominant CUDA Runtime API during gap
            try:
                runtime_tables = [
                    t
                    for t in self.prof.schema.tables
                    if t.startswith("CUPTI_ACTIVITY_KIND_RUNTIME")
                ]
                if runtime_tables:
                    rt_tbl = runtime_tables[0]
                    api_rows = self.prof._duckdb_query(
                        f"""
                        SELECT s.value AS api_name,
                               SUM(r.[end] - r.start) AS total_ns
                        FROM {rt_tbl} r
                        JOIN StringIds s ON r.nameId = s.id
                        WHERE r.start < ? AND r.[end] > ?
                        GROUP BY s.value
                        ORDER BY total_ns DESC
                        LIMIT 1
                        """,
                        (int(r["gap_end"]), int(r["gap_start"])),
                    )
                    if api_rows:
                        api = api_rows[0]
                        api_name = api["api_name"].split("_v")[0]  # strip version
                        api_ms = api["total_ns"] / 1e6
                        note += f" — CPU: {api_name} ({api_ms:.1f}ms)"
            except Exception:
                _log.debug("CPU attribution query failed", exc_info=True)

            findings.append(
                Finding(
                    type="region",
                    label=f"GPU Idle Gap ({gap_ms:.2f}ms)",
                    start_ns=int(r["gap_start"]),
                    end_ns=int(r["gap_end"]),
                    gpu_id=self.device,
                    stream=str(r["streamId"]),
                    severity="warning",
                    note=note,
                )
            )

        # Add summary finding if significant idle time
        if pct > 5 and len(rows) > 0:
            findings.append(
                Finding(
                    type="region",
                    label=f"GPU Idle Summary ({pct}% of profile)",
                    start_ns=self.trim[0],
                    end_ns=self.trim[1],
                    gpu_id=self.device,
                    severity="info",
                    note=(
                        f"Total: {total_idle_ns / 1e6:.1f}ms idle across top "
                        f"{len(rows)} gaps ({pct}% of profiled span)"
                    ),
                )
            )

        return findings

    def _nccl_stalls(self, top_n: int = 3) -> list[Finding]:
        """Longest individual NCCL kernel instances → highlight findings."""
        sql = f"""\
SELECT k.start, k.[end], k.streamId, k.deviceId,
       s.value AS name, (k.[end] - k.start) AS dur_ns
FROM {self.prof.schema.kernel_table} k
JOIN StringIds s ON k.shortName = s.id
WHERE k.deviceId = ?
  AND (s.value LIKE '%nccl%' OR s.value LIKE '%NCCL%')
  AND k.[end] >= ? AND k.start <= ?
ORDER BY dur_ns DESC
LIMIT ?"""
        rows = self.prof._duckdb_query(
                sql,
                (self.device, self.trim[0], self.trim[1], top_n),
            )
        return [
            Finding(
                type="highlight",
                label=f"Long NCCL ({r['dur_ns'] / 1e6:.2f}ms)",
                start_ns=int(r["start"]),
                end_ns=int(r["end"]),
                gpu_id=self.device,
                stream=str(r["streamId"]),
                severity="critical" if r["dur_ns"] > 5_000_000 else "warning",
                note=f"{r['name'][:60]}: {r['dur_ns'] / 1e6:.2f}ms",
            )
            for r in rows
        ]

    def _kernel_hotspots(self, top_n: int = 3) -> list[Finding]:
        """Top longest non-NCCL kernel instances → highlight."""
        sql = f"""\
SELECT s.value AS name, k.start, k.[end], k.streamId,
       (k.[end] - k.start) AS dur_ns
FROM {self.prof.schema.kernel_table} k
JOIN StringIds s ON k.shortName = s.id
WHERE k.deviceId = ?
  AND NOT (s.value LIKE '%nccl%' OR s.value LIKE '%NCCL%')
  AND k.[end] >= ? AND k.start <= ?
ORDER BY dur_ns DESC
LIMIT ?"""
        rows = self.prof._duckdb_query(
                sql,
                (self.device, self.trim[0], self.trim[1], top_n),
            )
        return [
            Finding(
                type="highlight",
                label=f"Hotspot: {r['name'][:30]}",
                start_ns=int(r["start"]),
                end_ns=int(r["end"]),
                gpu_id=self.device,
                stream=str(r["streamId"]),
                severity="info",
                note=f"{r['name'][:60]}: {r['dur_ns'] / 1e6:.2f}ms",
            )
            for r in rows
        ]

    def _memory_anomalies(self, min_bytes: int = 10_000_000, top_n: int = 5) -> list[Finding]:
        """Flag large memory transfers that may stall the GPU."""
        # Resolve memcpy table dynamically (may be versioned)
        memcpy_table = None
        for t in self.prof.schema.tables:
            if t == "CUPTI_ACTIVITY_KIND_MEMCPY" or t.startswith("CUPTI_ACTIVITY_KIND_MEMCPY"):
                memcpy_table = t
                break
        if memcpy_table is None:
            return []
        sql = f"""\
SELECT copyKind, bytes, start, [end], ([end] - start) AS dur_ns
FROM {memcpy_table}
WHERE deviceId = ? AND bytes > ? AND [end] >= ? AND start <= ?
ORDER BY dur_ns DESC
LIMIT ?"""
        kind_names = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}
        rows = self.prof._duckdb_query(
                sql, (self.device, min_bytes, self.trim[0], self.trim[1], top_n)
            )
        findings = []
        for r in rows:
            kind = kind_names.get(r["copyKind"], f"kind{r['copyKind']}")
            mb = r["bytes"] / 1e6
            dur_ms = r["dur_ns"] / 1e6
            findings.append(
                Finding(
                    type="highlight",
                    label=f"Large {kind} Transfer ({mb:.1f}MB)",
                    start_ns=int(r["start"]),
                    end_ns=int(r["end"]),
                    gpu_id=self.device,
                    severity="warning" if dur_ms > 1.0 else "info",
                    note=(
                        f"{kind}: {mb:.1f}MB in {dur_ms:.2f}ms "
                        f"({r['bytes'] / max(r['dur_ns'], 1) * 1e9 / 1e9:.1f}GB/s)"
                    ),
                )
            )
        return findings

    def _h2d_spikes(self) -> list[Finding]:
        """Detect H2D transfer spike windows and mark on timeline."""
        memcpy_table = None
        for t in self.prof.schema.tables:
            if t == "CUPTI_ACTIVITY_KIND_MEMCPY" or t.startswith("CUPTI_ACTIVITY_KIND_MEMCPY"):
                memcpy_table = t
                break
        if memcpy_table is None:
            return []

        # Group H2D by 1-second windows
        sql = f"""\
WITH baseline AS (
    SELECT MIN(start) AS min_start FROM {memcpy_table}
    WHERE copyKind = 1 AND deviceId = ? AND [end] >= ? AND start <= ?
)
SELECT
    CAST((m.start - b.min_start) / 1000000000.0 AS INT) AS second,
    COUNT(*) AS ops,
    SUM(m.bytes) AS total_bytes,
    MIN(m.start) AS window_start,
    MAX(m.[end]) AS window_end
FROM {memcpy_table} m CROSS JOIN baseline b
WHERE m.copyKind = 1 AND m.deviceId = ? AND m.[end] >= ? AND m.start <= ?
GROUP BY 1
ORDER BY total_bytes DESC"""
        try:
            rows = self.prof._duckdb_query(
                    sql,
                    (
                        self.device, self.trim[0], self.trim[1],
                        self.device, self.trim[0], self.trim[1],
                    ),
                )
        except Exception:
            _log.debug("H2D spike query failed", exc_info=True)
            return []

        if len(rows) < 3:
            return []

        # Find spike windows (> 3× median bytes)
        sorted_bytes = sorted(r["total_bytes"] for r in rows)
        median_bytes = sorted_bytes[len(sorted_bytes) // 2]
        if median_bytes <= 0:
            return []

        findings = []
        for r in rows:
            if r["total_bytes"] > 3 * median_bytes:
                mb = r["total_bytes"] / 1e6
                findings.append(
                    Finding(
                        type="region",
                        label=f"H2D Spike ({mb:.1f}MB at t={r['second']}s)",
                        start_ns=int(r["window_start"]),
                        end_ns=int(r["window_end"]),
                        gpu_id=self.device,
                        severity="info",
                        note=(
                            f"{r['ops']} H2D ops, {mb:.1f}MB "
                            f"(median {median_bytes / 1e6:.1f}MB/s window)"
                        ),
                    )
                )
                if len(findings) >= 3:  # cap at top 3 spikes
                    break

        return findings
