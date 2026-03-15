"""
evidence_builder.py — Convert profile analysis into visual Finding overlays.

Each method queries individual kernel instances (not aggregates)
to produce findings with exact nanosecond timestamps for timeline overlay.
"""

import statistics

from .annotation import EvidenceReport, Finding
from .profile import Profile


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
        return EvidenceReport(
            title="Auto-Analysis",
            profile_path=getattr(self.prof, "path", ""),
            findings=findings,
        )

    # ------------------------------------------------------------------
    # Analyzers
    # ------------------------------------------------------------------

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
                findings.append(Finding(
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
                ))
        return findings

    def _gpu_idle_gaps(
        self, top_n: int = 5, min_gap_ns: int = 1_000_000
    ) -> list[Finding]:
        """Top N idle gaps between consecutive kernels → region findings."""
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
        with self.prof._lock:
            rows = self.prof.conn.execute(
                sql,
                (self.device, self.trim[0], self.trim[1], min_gap_ns, top_n),
            ).fetchall()
        return [
            Finding(
                type="region",
                label=f"GPU Idle Gap ({r['gap_ns'] / 1e6:.2f}ms)",
                start_ns=int(r["gap_start"]),
                end_ns=int(r["gap_end"]),
                gpu_id=self.device,
                stream=str(r["streamId"]),
                severity="warning",
                note=f"Stream {r['streamId']}: {r['gap_ns'] / 1e6:.2f}ms idle",
            )
            for r in rows
        ]

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
        with self.prof._lock:
            rows = self.prof.conn.execute(
                sql,
                (self.device, self.trim[0], self.trim[1], top_n),
            ).fetchall()
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
        with self.prof._lock:
            rows = self.prof.conn.execute(
                sql,
                (self.device, self.trim[0], self.trim[1], top_n),
            ).fetchall()
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
