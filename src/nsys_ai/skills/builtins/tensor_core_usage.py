"""Tensor Core utilization analysis.

Queries the profile for kernels that are mathematically eligible for Tensor Core
acceleration (e.g. GEMM, Convolutions) and checks whether they successfully
utilized the hardware Tensor Cores. Helps find FP32 fallbacks and alignment errors.
"""

from ..base import Skill, SkillParam


def _execute(conn, **kwargs):
    """Execute Tensor Core eligibility analysis."""
    limit = int(kwargs.get("limit", 20))
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")

    trim_clause = ""
    params = []
    if trim_start is not None and trim_end is not None:
        trim_clause = 'AND start >= ? AND "end" <= ?'
        params = [trim_start, trim_end]

    # Only include TC-eligible kernels
    sql = f"""
        SELECT
            name,
            SUM("end" - start) AS total_ns,
            COUNT(*) AS call_count,
            SUM(CASE WHEN uses_tc = 1 THEN ("end" - start) ELSE 0 END) AS tc_active_ns,
            SUM(CASE WHEN uses_tc = 1 THEN 1 ELSE 0 END) AS tc_active_calls
        FROM kernels
        WHERE is_tc_eligible = 1 {trim_clause}
        GROUP BY name
        ORDER BY total_ns DESC
        LIMIT {limit}
    """
    try:
        rows = conn.execute(sql, params).fetchall()
        cols = ["name", "total_ns", "call_count", "tc_active_ns", "tc_active_calls"]
        rows = [dict(zip(cols, r)) for r in rows]
    except Exception:
        return [
            {
                "error": (
                    "Tensor Core analysis requires a database connection that exposes "
                    "a 'kernels' view with 'is_tc_eligible' and 'uses_tc' columns, or a "
                    "compatible profiling backend."
                )
            }
        ]

    if not rows:
        return []

    results = []
    for r in rows:
        total_ns = r["total_ns"]
        call_count = r["call_count"]
        tc_active_ns = r["tc_active_ns"]
        tc_active_calls = r["tc_active_calls"]

        tc_pct_time = (tc_active_ns / total_ns * 100.0) if total_ns > 0 else 0.0
        tc_pct_calls = (tc_active_calls / call_count * 100.0) if call_count > 0 else 0.0

        results.append(
            {
                "kernel_name": r["name"],
                "total_gpu_ms": total_ns / 1e6,
                "call_count": call_count,
                "tc_active_ms": tc_active_ns / 1e6,
                "tc_active_calls": tc_active_calls,
                "tc_achieved_pct": tc_pct_time,
                "tc_calls_pct": tc_pct_calls,
                "is_outlier": tc_pct_time
                < 50.0,  # Flag any kernel with < 50% TC uptime as an outlier/fallback
            }
        )

    return results


def _format(rows):
    if not rows:
        return "(No Tensor Core eligible kernels found)"
    if "error" in rows[0]:
        return f"Error: {rows[0]['error']}"

    lines = [
        "── Tensor Core Utilization (Eligible Kernels Only) ──",
        f"{'Kernel Name':<50s}  {'Calls':>8s}  {'TC Calls':>10s}  {'Time(ms)':>10s}  {'TC Ops%':>8s}  {'Flag':>6s}",
        "─" * 100,
    ]

    total_eligible_ms = sum(r["total_gpu_ms"] for r in rows)
    total_active_ms = sum(r["tc_active_ms"] for r in rows)
    global_pct = (total_active_ms / total_eligible_ms * 100.0) if total_eligible_ms > 0 else 0.0

    for r in rows:
        name = r["kernel_name"]
        if len(name) > 48:
            name = "..." + name[-45:]

        flag = " ⚠️" if r["is_outlier"] else ""
        lines.append(
            f"{name:<50s}  {r['call_count']:>8d}  {r['tc_calls_pct']:>9.1f}%"
            f"  {r['total_gpu_ms']:>10.2f}  {r['tc_achieved_pct']:>7.1f}%{flag:>6s}"
        )

    lines.append("-" * 100)
    lines.append(f"Global Top-K TC Achieved: {global_pct:.1f}%")
    if global_pct < 100.0:
        lines.append(
            "Note: ⚠️ flags kernels that frequently fallback to FP32. Check alignment and padding."
        )

    return "\n".join(lines)


SKILL = Skill(
    name="tensor_core_usage",
    title="Tensor Core Utilization",
    description=(
        "Analyzes kernels that are eligible for Tensor Core acceleration "
        "(GEMM, Convolutions, MatMul) and reports the percentage of time "
        "they actually utilized Tensor Cores. Flags kernels that silently "
        "fell back to non-TC generic ALUs due to shape/alignment issues."
    ),
    category="kernels",
    execute_fn=_execute,
    format_fn=_format,
    params=[
        SkillParam("limit", "Max kernels to return", "int", False, 20),
    ],
)
