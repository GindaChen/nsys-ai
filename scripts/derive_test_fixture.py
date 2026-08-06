#!/usr/bin/env python3
"""Derive a small, committable test profile from a real Nsight capture.

Hand-written fixtures cannot reproduce the things that actually break. Recent
defects needed a correlationId fan-out across hundreds of thousands of rows, two
devices with byte-identical kernel counts, and an NVTX regime that only appears
inside a trimmed window of a real training step. None of that is expressible by
writing INSERT statements; all of it survives a time slice of a genuine export.

So this takes a real profile and keeps a window of it, preserving the full
schema, the real kernel and NVTX names, and the export metadata. The result is a
genuine nsys export that happens to be short, rather than a synthetic imitation.

Committed fixtures live in git history forever, so the goal is the smallest
window that still exercises the structure: several thousand kernels, more than
one device, more than one stream, real NVTX nesting.

    python scripts/derive_test_fixture.py SOURCE.sqlite OUT.sqlite --window 1.0

Provenance is written into the output's META_DATA_EXPORT so the fixture can say
where it came from without a README nobody reads.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

# Tables copied whole rather than sliced. They are lookups the sliced rows join
# against — dropping unreferenced entries would save little and risks breaking a
# join the analysis makes but this script does not know about.
_COPY_WHOLE_PREFIXES = (
    "ThreadNames",
    "TARGET_INFO",
    "META_DATA",
    "ANALYSIS_DETAILS",
    "EXPORT_META",
    "NVTX_PAYLOAD_SCHEMAS",
    "NVTX_PAYLOAD_SCHEMA_ENTRIES",
)


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]


def _tables(conn: sqlite3.Connection) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]


def derive(src_path: str, out_path: str, window_s: float, offset_frac: float) -> dict:
    src = sqlite3.connect(src_path)
    lo, hi = src.execute(
        "SELECT MIN(start), MAX([end]) FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()
    if lo is None:
        raise SystemExit("source has no kernels")

    # Start partway in: the opening of a capture is warm-up, which is not what
    # the analysis is usually pointed at.
    start = lo + int((hi - lo) * offset_frac)
    end = start + int(window_s * 1e9)

    if os.path.exists(out_path):
        os.unlink(out_path)
    out = sqlite3.connect(out_path)

    kept: dict[str, int] = {}
    for table in _tables(src):
        ddl = src.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()[0]
        out.execute(ddl)

        cols = _columns(src, table)
        # StringIds is deliberately NOT copied whole. It holds every string the
        # capture ever saw — 1.5M rows on a 265s profile — which dominated the
        # output at 75MB for 3400 kernels. Only the ids the retained rows
        # actually reference are kept, below.
        if table == "StringIds":
            continue
        # Some event tables key on `timestamp` rather than `start`; without
        # this they fall through to "copy whole" and drag in 117k rows.
        time_col = "start" if "start" in cols else ("timestamp" if "timestamp" in cols else None)
        whole = table.startswith(_COPY_WHOLE_PREFIXES) or time_col is None
        if whole:
            rows = src.execute(f'SELECT * FROM "{table}"').fetchall()  # noqa: S608
        elif time_col == "timestamp":
            q = f'SELECT * FROM "{table}" WHERE timestamp >= ? AND timestamp <= ?'  # noqa: S608
            rows = src.execute(q, (start, end)).fetchall()
        elif "end" in cols:
            q = f'SELECT * FROM "{table}" WHERE start >= ? AND [end] <= ?'  # noqa: S608
            rows = src.execute(q, (start, end)).fetchall()
        else:
            q = f'SELECT * FROM "{table}" WHERE start >= ? AND start <= ?'  # noqa: S608
            rows = src.execute(q, (start, end)).fetchall()

        if rows:
            placeholders = ",".join("?" * len(cols))
            out.executemany(
                f'INSERT INTO "{table}" VALUES ({placeholders})',  # noqa: S608
                rows,
            )
        kept[table] = len(rows)

    # StringIds last, restricted to ids the retained rows reference. Any column
    # ending in Name/nameId/textId is a StringIds foreign key by convention.
    # The table itself was already created by the loop above; only its rows
    # were deferred.
    referenced: set[int] = set()
    for table in _tables(src):
        if table == "StringIds":
            continue
        for col in _columns(src, table):
            if not (col.endswith(("Name", "nameId", "textId", "NameId")) or col == "id"):
                continue
            if col == "id" and not table.startswith("StringIds"):
                continue
            try:
                referenced.update(
                    r[0]
                    for r in out.execute(f'SELECT DISTINCT "{col}" FROM "{table}"')  # noqa: S608
                    if r[0] is not None
                )
            except sqlite3.Error:
                continue
    if referenced:
        marks = ",".join("?" * len(referenced))
        kept_ids = src.execute(
            f"SELECT * FROM StringIds WHERE id IN ({marks})",  # noqa: S608
            tuple(referenced),
        ).fetchall()
        out.executemany("INSERT INTO StringIds VALUES (?,?)", kept_ids)
        kept["StringIds"] = len(kept_ids)

    # Record where this came from, in the file itself.
    try:
        out.executemany(
            "INSERT INTO META_DATA_EXPORT VALUES (?, ?)",
            [
                ("DERIVED_FROM", os.path.basename(src_path)),
                ("DERIVED_WINDOW_NS", f"{start}-{end}"),
                ("DERIVED_BY", "scripts/derive_test_fixture.py"),
            ],
        )
    except sqlite3.Error:
        pass  # older exports may lack the table; provenance is best-effort

    out.commit()
    out.execute("VACUUM")
    out.close()
    src.close()

    return {
        "kernels": kept.get("CUPTI_ACTIVITY_KIND_KERNEL", 0),
        "nvtx": kept.get("NVTX_EVENTS", 0),
        "tables": len(kept),
        "bytes": os.path.getsize(out_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("--window", type=float, default=1.0, help="seconds to keep")
    ap.add_argument(
        "--offset",
        type=float,
        default=0.5,
        help="where in the capture to start, as a fraction (default: midpoint)",
    )
    args = ap.parse_args()

    stats = derive(args.source, args.output, args.window, args.offset)
    print(
        f"{args.output}: {stats['bytes'] / 2**20:.1f}MB  "
        f"tables={stats['tables']} kernels={stats['kernels']} nvtx={stats['nvtx']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
