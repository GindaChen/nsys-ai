"""Top GPU kernels by total execution time."""

from nsys_ai.connection import DB_ERRORS

from ..base import Skill, SkillParam


def _format(rows):
    if not rows:
        return "(No kernels found)"
    lines = [
        "── Top GPU Kernels by Total Time ──",
        f"{'TC':<4s}  {'Kernel':<57s}  {'Count':>7s}  {'Total(ms)':>10s}  {'Avg(ms)':>9s}  {'Min(ms)':>9s}  {'Max(ms)':>9s}",
        "-" * 116,
    ]
    for r in rows:
        name = r["kernel_name"]
        if len(name) > 55:
            name = name[:52] + "..."

        tc_status = "[-]"
        if r.get("tc_eligible") is None:
            tc_status = "N/A "
        elif r.get("tc_eligible"):
            tc_status = "[✓]" if r.get("tc_active") else "[⚠️]"

        lines.append(
            f"{tc_status:<4s}  {name:<57s}  {r['invocations']:>7d}  {r['total_ms']:>10.2f}  "
            f"{r['avg_ms']:>9.2f}  {r['min_ms']:>9.2f}  {r['max_ms']:>9.2f}"
        )
    return "\n".join(lines)


def _execute(conn, **kwargs):
    limit = int(kwargs.get("limit", 15))
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")

    try:
        conn.execute("SELECT 1 FROM kernels LIMIT 1")
        has_kernels = True
    except DB_ERRORS:
        has_kernels = False

    params = []
    if has_kernels:
        trim_clause = ""
        if trim_start is not None and trim_end is not None:
            trim_clause = 'AND start >= ? AND "end" <= ?'
            params.extend([trim_start, trim_end])

        sql = f"""
            SELECT name AS kernel_name,
                   COUNT(*) AS invocations,
                   ROUND(SUM("end" - start) / 1e6, 2) AS total_ms,
                   ROUND(AVG("end" - start) / 1e6, 2) AS avg_ms,
                   ROUND(MIN("end" - start) / 1e6, 2) AS min_ms,
                   ROUND(MAX("end" - start) / 1e6, 2) AS max_ms,
                   SUM(is_tc_eligible) > 0 AS tc_eligible,
                   SUM(uses_tc) > 0 AS tc_active
            FROM kernels
            WHERE 1=1 {trim_clause}
            GROUP BY name
            ORDER BY total_ms DESC
            LIMIT {limit}
        """
        rows = conn.execute(sql, params).fetchall()
        cols = [
            "kernel_name",
            "invocations",
            "total_ms",
            "avg_ms",
            "min_ms",
            "max_ms",
            "tc_eligible",
            "tc_active",
        ]
        return [dict(zip(cols, r)) for r in rows]
    else:
        # Pure SQLite fallback (lacks TC eligibility analysis)
        from nsys_ai.connection import wrap_connection

        tables = wrap_connection(conn).resolve_activity_tables()
        kernel_table = tables.get("kernel", "CUPTI_ACTIVITY_KIND_KERNEL")

        trim_clause = ""
        if trim_start is not None and trim_end is not None:
            trim_clause = 'AND k.start >= ? AND k."end" <= ?'
            params.extend([trim_start, trim_end])

        sql = f"""
            SELECT COALESCE(d.value, s.value, 'kernel_' || CAST(k.shortName AS VARCHAR)) AS kernel_name,
                   COUNT(*) AS invocations,
                   ROUND(SUM(k."end" - k.start) / 1e6, 2) AS total_ms,
                   ROUND(AVG(k."end" - k.start) / 1e6, 2) AS avg_ms,
                   ROUND(MIN(k."end" - k.start) / 1e6, 2) AS min_ms,
                   ROUND(MAX(k."end" - k.start) / 1e6, 2) AS max_ms,
                   NULL AS tc_eligible,
                   NULL AS tc_active
            FROM {kernel_table} k
            LEFT JOIN StringIds s ON k.shortName = s.id
            LEFT JOIN StringIds d ON k.demangledName = d.id
            WHERE 1=1 {trim_clause}
            GROUP BY COALESCE(d.value, s.value, 'kernel_' || CAST(k.shortName AS VARCHAR))
            ORDER BY total_ms DESC
            LIMIT {limit}
        """
        rows = conn.execute(sql, params).fetchall()
        cols = [
            "kernel_name",
            "invocations",
            "total_ms",
            "avg_ms",
            "min_ms",
            "max_ms",
            "tc_eligible",
            "tc_active",
        ]
        return [dict(zip(cols, r)) if isinstance(r, tuple) else dict(r) for r in rows]


SKILL = Skill(
    name="top_kernels",
    title="Top GPU Kernels by Total Time",
    description=(
        "Lists the heaviest GPU kernels ranked by cumulative execution time. "
        "Use this to identify hotspots. TC column shows Tensor Core usage "
        "(✓=Active, ⚠️=Eligible but Fallback, -=Ineligible)."
    ),
    category="kernels",
    execute_fn=_execute,
    params=[SkillParam("limit", "Max number of kernels to return", "int", False, 15)],
    format_fn=_format,
    tags=["hotspot", "kernel", "duration", "performance", "top", "tensor_core"],
)
