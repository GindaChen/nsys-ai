#!/usr/bin/env python3
# ruff: noqa: E501
"""Run all built-in skills once against an Nsight profile (single DB connection).

Uses the same ``open_cached_db()`` path as ``nsys-ai skill run`` (DuckDB/Parquet
with SQLite fallback). Writes per-skill JSON, logs, and ``_batch_summary.json``.

Usage: python scripts/batch_audit_skills.py <profile.sqlite> [output_dir]

Default output_dir: audit/l40s-perf_compile (override for other profiles).

Records every registry skill (currently 35): all executed except explicit
SKIP for ``cutracer_analysis`` (needs trace_dir).
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

SKIP = {"cutracer_analysis"}

# Extra kwargs for batch runs (CLI would prompt for required params).
BATCH_EXTRA: dict[str, dict] = {
    "iteration_detail": {"iteration": 0, "device": 0},
}

# Run diagnosis-critical skills first (overlap/NCCL before slow host-sync SQL).
PRIORITY_FIRST = [
    "overlap_breakdown",
    "nccl_payload_breakdown",
    "nccl_breakdown",
    "nccl_communicator_analysis",
    "profile_health_manifest",
    "root_cause_matcher",
    "iteration_timing",
    "iteration_detail",
    "nvtx_layer_breakdown",
    "pipeline_bubble_metrics",
    "top_kernels",
    "kernel_overlap_matrix",
    "stream_concurrency",
    "sync_cost_analysis",
]


def _open_skill_connection(profile_path: str):
    """Match ``nsys-ai skill run`` connection setup (cached DuckDB, else SQLite)."""
    import duckdb

    from nsys_ai.parquet_cache import open_cached_db

    try:
        return open_cached_db(profile_path)
    except (duckdb.Error, RuntimeError, OSError):
        return sqlite3.connect(profile_path)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: batch_audit_skills.py <profile.sqlite> [output_dir]", file=sys.stderr)
        return 2

    profile_path = (sys.argv[1] or "").strip()
    if not profile_path:
        print(
            "error: profile path is empty (is L40S_PROFILE set?)\n"
            "  export L40S_PROFILE=~/.cache/nsys-ai-datasets/fastvideo-wan-l40s-nsys/profiles/perf_compile.sqlite\n"
            "  python scripts/batch_audit_skills.py \"$L40S_PROFILE\"",
            file=sys.stderr,
        )
        return 2
    profile_path = str(Path(profile_path).expanduser().resolve())
    if not Path(profile_path).is_file():
        print(f"error: profile not found: {profile_path}", file=sys.stderr)
        return 2

    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "audit/l40s-perf_compile")
    json_dir = out_dir / "json"
    log_dir = out_dir / "logs"
    json_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    from nsys_ai.skills import list_skills
    from nsys_ai.skills.registry import get_skill

    all_names = list_skills()
    ordered = [n for n in PRIORITY_FIRST if n in all_names]
    ordered += [n for n in all_names if n not in ordered]

    summary = []
    conn = _open_skill_connection(profile_path)
    execute_kwargs = {"_sqlite_path": profile_path}

    try:
        for name in ordered:
            if name in SKIP:
                out_path = json_dir / f"{name}.json"
                skip_row = {
                    "skill": name,
                    "category": "utility",
                    "exit_code": -1,
                    "runtime_s": 0,
                    "json_path": str(out_path),
                    "status_hint": "SKIP",
                    "skip_reason": "requires trace_dir (CUTracer histograms)",
                }
                summary.append(skip_row)
                out_path.write_text(
                    json.dumps([{"error": skip_row["skip_reason"]}], indent=2)
                )
                print(f"{name}: SKIP", file=sys.stderr)
                continue

            skill = get_skill(name)
            if skill is None:
                summary.append(
                    {"skill": name, "exit_code": 1, "runtime_s": 0, "status_hint": "NOT_FOUND"}
                )
                continue

            t0 = time.perf_counter()
            hint = "OK"
            exit_code = 0
            rows: list = []
            err_text = ""
            run_kwargs = {**execute_kwargs, **BATCH_EXTRA.get(name, {})}
            try:
                rows = skill.execute(conn, **run_kwargs)
            except ValueError as exc:
                exit_code = 1
                hint = "ERROR_JSON"
                err_text = str(exc)
                rows = [{"error": err_text}]
            except Exception as exc:
                exit_code = 1
                hint = "EXCEPTION"
                err_text = str(exc)
                rows = [{"error": err_text}]

            elapsed = time.perf_counter() - t0
            out_path = json_dir / f"{name}.json"
            with out_path.open("w") as f:
                json.dump(rows, f, indent=2, default=str)

            log_path = log_dir / f"{name}.log"
            log_path.write_text(err_text or skill.format_rows(rows))

            if rows and isinstance(rows[0], dict) and "error" in rows[0]:
                hint = "ERROR_JSON"
            elif not rows:
                hint = "EMPTY"

            summary.append(
                {
                    "skill": name,
                    "category": skill.category,
                    "exit_code": exit_code,
                    "runtime_s": round(elapsed, 2),
                    "json_path": str(out_path),
                    "status_hint": hint,
                }
            )
            print(f"{name}: {hint} ({elapsed:.1f}s)", file=sys.stderr)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    summary_path = json_dir / "_batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path} ({len(summary)} skills)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
