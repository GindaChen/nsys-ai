"""
indexing.py — Centralized performance index management for Nsight SQLite profiles.

Merges index creation logic previously duplicated in:
  - skills/base.py  (``ensure_indexes``)
  - region_mfu.py   (``_ensure_kernel_indexes``)

All callers should use ``ensure_performance_indexes()`` which is idempotent,
silent on read-only connections, and tolerant of missing tables.
"""

import logging
import sqlite3

_log = logging.getLogger(__name__)

# Track connections that have already been indexed to avoid repeated work.
_indexed_connections: set[int] = set()


def _resolve_activity_tables(conn: sqlite3.Connection) -> dict[str, str]:
    """Resolve Nsight activity table names (kernel/runtime/NVTX/memcpy/memset).

    Nsight may emit versioned table names such as
    ``CUPTI_ACTIVITY_KIND_KERNEL_V2``.
    This helper finds the first matching table for each logical kind.
    """
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    except Exception:
        return {}

    def _find_by_prefix(prefix: str) -> str | None:
        if prefix in tables:
            return prefix
        candidates = sorted(t for t in tables if t.startswith(prefix))
        return candidates[0] if candidates else None

    kernel_table = _find_by_prefix("CUPTI_ACTIVITY_KIND_KERNEL")
    runtime_table = _find_by_prefix("CUPTI_ACTIVITY_KIND_RUNTIME")
    memcpy_table = _find_by_prefix("CUPTI_ACTIVITY_KIND_MEMCPY")
    memset_table = _find_by_prefix("CUPTI_ACTIVITY_KIND_MEMSET")
    if "NVTX_EVENTS" in tables:
        nvtx_table: str | None = "NVTX_EVENTS"
    else:
        nvtx_table = _find_by_prefix("NVTX_EVENTS")

    resolved: dict[str, str] = {}
    if kernel_table:
        resolved["kernel"] = kernel_table
    if runtime_table:
        resolved["runtime"] = runtime_table
    if memcpy_table:
        resolved["memcpy"] = memcpy_table
    if memset_table:
        resolved["memset"] = memset_table
    if nvtx_table:
        resolved["nvtx"] = nvtx_table

    return resolved


def ensure_performance_indexes(conn: sqlite3.Connection) -> None:
    """Create all performance indexes needed by skills, MFU, and evidence analysis.

    This is safe to call repeatedly — indexes use ``CREATE INDEX IF NOT EXISTS``
    and the function tracks which connections have been processed.  Each index
    creation is wrapped in try/except so missing tables don't block the rest.

    Index naming convention: ``_nsysai_<table_kind>_<column(s)>``
    """
    conn_id = id(conn)
    if conn_id in _indexed_connections:
        return

    tables = _resolve_activity_tables(conn)

    index_stmts: list[str] = []

    kernel_table = tables.get("kernel")
    if kernel_table:
        index_stmts.extend([
            f"CREATE INDEX IF NOT EXISTS _nsysai_kernel_start ON {kernel_table}(start)",
            f"CREATE INDEX IF NOT EXISTS _nsysai_kernel_corr  ON {kernel_table}(correlationId)",
            # shortName index — critical for kernel name lookups (region_mfu, skills)
            f"CREATE INDEX IF NOT EXISTS _nsysai_kernel_short ON {kernel_table}(shortName)",
            # Streamwise index for window-function skills (gpu_idle_gaps, kernel_launch_pattern)
            f"CREATE INDEX IF NOT EXISTS _nsysai_kernel_stream ON {kernel_table}(streamId, start)",
        ])

    runtime_table = tables.get("runtime")
    if runtime_table:
        index_stmts.extend([
            f"CREATE INDEX IF NOT EXISTS _nsysai_runtime_corr ON {runtime_table}(correlationId)",
            f"CREATE INDEX IF NOT EXISTS _nsysai_runtime_tid  ON {runtime_table}(globalTid, start)",
        ])

    nvtx_table = tables.get("nvtx")
    if nvtx_table:
        index_stmts.extend([
            f"CREATE INDEX IF NOT EXISTS _nsysai_nvtx_start   ON {nvtx_table}(start)",
            f"CREATE INDEX IF NOT EXISTS _nsysai_nvtx_tid     ON {nvtx_table}(globalTid, start)",
            # Compound index for NVTX join queries (nvtx_layer_breakdown, nvtx_kernel_map)
            f"CREATE INDEX IF NOT EXISTS _nsysai_nvtx_range   ON {nvtx_table}(globalTid, start, [end])",
        ])

    memcpy_table = tables.get("memcpy")
    if memcpy_table:
        index_stmts.append(
            f"CREATE INDEX IF NOT EXISTS _nsysai_memcpy_corr ON {memcpy_table}(correlationId)"
        )

    memset_table = tables.get("memset")
    if memset_table:
        index_stmts.append(
            f"CREATE INDEX IF NOT EXISTS _nsysai_memset_corr ON {memset_table}(correlationId)"
        )

    for stmt in index_stmts:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            # Table doesn't exist in this profile — skip silently.
            pass
        except Exception as exc:
            # Read-only filesystem, locked DB, etc.
            _log.debug("ensure_performance_indexes: %s — %s", stmt.split("ON")[0].strip(), exc)

    try:
        conn.commit()
    except Exception:
        pass

    _indexed_connections.add(conn_id)
