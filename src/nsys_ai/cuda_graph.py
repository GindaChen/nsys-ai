"""Optional CUDA Graph schema support.

Nsight Systems adds CUDA Graph columns and tables independently of the core
kernel activity schema.  Keep the probes in one small module so SQLite,
DuckDB, and the Parquet cache agree on what "graph-aware" means without
turning graph tracing into a hard requirement for older reports.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

GRAPH_TABLE_NAMES = (
    "CUDA_GRAPH_EVENTS",
    "CUDA_GRAPH_NODE_EVENTS",
    "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
)

# Canonical output names used by nsys-ai.  The values are the names NVIDIA
# uses in the kernel activity export.  Keeping the mapping explicit also lets
# us tolerate harmless case changes in future exports.
GRAPH_KERNEL_COLUMNS = {
    "graph_node_id": "graphNodeId",
    "graph_id": "graphId",
}


def actual_table(tables: Iterable[str], name: str) -> str | None:
    """Return the case-preserving table name matching *name*, if present."""
    wanted = name.lower()
    return next((table for table in tables if str(table).lower() == wanted), None)


def actual_columns(columns: Iterable[str]) -> dict[str, str]:
    """Return lower-case column names mapped to their actual spelling."""
    return {str(column).lower(): str(column) for column in columns}


def graph_tables(tables: Iterable[str]) -> dict[str, str]:
    """Resolve the optional graph tables present in an export."""
    return {
        name: table
        for name in GRAPH_TABLE_NAMES
        if (table := actual_table(tables, name)) is not None
    }


def kernel_graph_columns(columns: Iterable[str]) -> dict[str, str]:
    """Resolve graph columns on a kernel table to canonical output names."""
    by_lower = actual_columns(columns)
    resolved: dict[str, str] = {}
    for canonical, source in GRAPH_KERNEL_COLUMNS.items():
        # Accept both NVIDIA's camelCase export and the normalized names used
        # by the Parquet cache/raw parquetdir adapter.
        for candidate in (source, canonical):
            if candidate.lower() in by_lower:
                resolved[canonical] = by_lower[candidate.lower()]
                break
    return resolved


def graph_capability(
    tables: Iterable[str], kernel_columns: Mapping[str, str]
) -> dict[str, object]:
    """Describe optional graph support without claiming more than the data.

    ``kernel_attribution`` is true only when kernel rows carry a graph ID.  A
    graph metadata table alone is useful for diagnostics, but it cannot be
    joined to kernel timing without inventing an attribution.
    """
    resolved_tables = graph_tables(tables)
    return {
        "kernel_attribution": bool(kernel_columns),
        "kernel_columns": dict(kernel_columns),
        "tables": resolved_tables,
        "available": bool(kernel_columns or resolved_tables),
    }
