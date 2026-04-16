"""Parse CUTracer output files into structured Python objects.

Supported formats
-----------------
* ``proton_instr_histogram`` CSV — ``kernel_<name>_<hash>_hist.csv``
  Columns: ``warp_id, region_id, instruction, count, cycles``

* NDJSON / NDJSON.zst (opcode_only, reg_trace, mem_addr_trace …)
  Each line is a JSON record; zstd-compressed variants auto-detected.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)

# Filename patterns:
#   1) kernel_<name>_<hex-hash>_hist.csv
#   2) kernel_<hex-hash>_<name>_hist.csv
_HIST_NAME_HASH_RE = re.compile(r"^kernel_(.+?)_([0-9a-f]{8,})_hist\.csv$", re.IGNORECASE)
_HIST_HASH_NAME_RE = re.compile(r"^kernel_([0-9a-f]{8,})_(.+?)_hist\.csv$", re.IGNORECASE)


def _extract_kernel_name_from_hist_filename(path: Path) -> str:
    """Extract kernel name from supported histogram filename layouts."""
    m = _HIST_NAME_HASH_RE.match(path.name)
    if m:
        return m.group(1)
    m = _HIST_HASH_NAME_RE.match(path.name)
    if m:
        return m.group(2)
    return path.stem


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class KernelHistogram:
    """Aggregated instruction histogram for one kernel."""

    kernel_name: str
    """Kernel name extracted from the CUTracer output filename."""

    instruction_counts: dict[str, int] = field(default_factory=dict)
    """Opcode → total invocation count across all warps/regions."""

    instruction_cycles: dict[str, int] = field(default_factory=dict)
    """Opcode → total cycles spent across all warps/regions."""

    warp_count: int = 0
    """Number of distinct warp IDs seen (proxy for active warps)."""

    @property
    def total_count(self) -> int:
        return sum(self.instruction_counts.values())

    @property
    def total_cycles(self) -> int:
        return sum(self.instruction_cycles.values())

    def cycles_per_instruction(self, opcode: str) -> float:
        count = self.instruction_counts.get(opcode, 0)
        cycles = self.instruction_cycles.get(opcode, 0)
        return cycles / count if count else 0.0


@dataclass
class TraceRecord:
    """Single record from an NDJSON CUTracer trace."""

    record_type: str
    opcode_id: int | None
    sass: str | None
    warp: int | None
    cta: list[int] | None
    timestamp: int | None
    raw: dict


# ---------------------------------------------------------------------------
# Histogram CSV parser
# ---------------------------------------------------------------------------


def parse_histogram_csv(path: Path) -> KernelHistogram | None:
    """Parse one ``*_hist.csv`` file into a :class:`KernelHistogram`.

    Returns ``None`` if the file cannot be parsed or is empty.
    """
    kernel_name = _extract_kernel_name_from_hist_filename(path)

    counts: dict[str, int] = {}
    cycles: dict[str, int] = {}
    warps: set[int] = set()

    try:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                opcode = row.get("instruction", "").strip().upper()
                if not opcode:
                    continue
                cnt = int(row.get("count", 0))
                cyc = int(row.get("cycles", 0))
                counts[opcode] = counts.get(opcode, 0) + cnt
                cycles[opcode] = cycles.get(opcode, 0) + cyc
                try:
                    warps.add(int(row.get("warp_id", -1)))
                except ValueError:
                    pass
    except (OSError, csv.Error, ValueError) as exc:
        _log.warning("Failed to parse histogram %s: %s", path, exc)
        return None

    if not counts:
        return None

    return KernelHistogram(
        kernel_name=kernel_name,
        instruction_counts=counts,
        instruction_cycles=cycles,
        warp_count=len(warps),
    )


def parse_histogram_dir(trace_dir: Path) -> dict[str, KernelHistogram]:
    """Scan *trace_dir* for histogram data and parse all of it.

    Supports two formats (auto-detected):

    1. ``*_hist.csv`` — pre-computed histograms (produced by the Modal pipeline
       or by a previous run of this function with ``--analyze``).
    2. ``*.ndjson`` + ``*.cubin`` pairs — raw CUTracer traces.  The function
       calls :func:`sass_resolve_dir` to resolve ``opcode_id`` → SASS mnemonic
       via ``cutracer sass`` and aggregates instruction counts on the fly.

    Returns a dict mapping the CUTracer kernel name → :class:`KernelHistogram`.
    """
    trace_dir = Path(trace_dir)
    result: dict[str, KernelHistogram] = {}

    hist_files = sorted(trace_dir.glob("**/*_hist.csv"))
    ndjson_files = sorted(trace_dir.glob("**/*.ndjson"))

    if not hist_files and ndjson_files:
        # Only raw ndjson traces found — resolve via SASS and build histograms.
        _log.debug("No *_hist.csv in %s; resolving %d ndjson file(s) via SASS",
                   trace_dir, len(ndjson_files))
        result = sass_resolve_dir(trace_dir)
        _log.debug("SASS-resolved %d kernel histogram(s)", len(result))
        return result

    if not hist_files:
        _log.debug("No *_hist.csv or *.ndjson files found in %s", trace_dir)
        return result

    for f in hist_files:
        hist = parse_histogram_csv(f)
        if hist is not None:
            if hist.kernel_name in result:
                existing = result[hist.kernel_name]
                for op, cnt in hist.instruction_counts.items():
                    existing.instruction_counts[op] = existing.instruction_counts.get(op, 0) + cnt
                for op, cyc in hist.instruction_cycles.items():
                    existing.instruction_cycles[op] = existing.instruction_cycles.get(op, 0) + cyc
                existing.warp_count = max(existing.warp_count, hist.warp_count)
            else:
                result[hist.kernel_name] = hist

    _log.debug("Parsed %d kernel histogram(s) from %s", len(result), trace_dir)
    return result


def sass_resolve_dir(trace_dir: Path) -> dict[str, KernelHistogram]:
    """Resolve raw CUTracer ndjson traces → :class:`KernelHistogram` objects.

    Requires ``cutracer sass`` (ships with the ``cutracer`` PyPI package) and
    ``nvdisasm`` (part of the CUDA toolkit) to disassemble the ``.cubin`` files.

    For each ``*.cubin`` in *trace_dir*, finds matching ``*_iter*.ndjson``
    files, builds an ``opcode_id`` → SASS mnemonic table by parsing the SASS
    disassembly, then counts instruction executions across all iterations.

    Returns a dict mapping kernel arch name → :class:`KernelHistogram`.
    Kernels whose cubin cannot be disassembled are skipped with a warning.
    """
    import json
    import subprocess
    from collections import defaultdict

    trace_dir = Path(trace_dir)
    cubins = {f.stem: f for f in trace_dir.glob("*.cubin")}
    if not cubins:
        _log.debug("sass_resolve_dir: no *.cubin files in %s", trace_dir)
        return {}

    # Group ndjson files by kernel base name (strip _iter<N>)
    kernel_groups: dict[str, list[Path]] = defaultdict(list)
    for ndf in sorted(trace_dir.glob("*.ndjson")):
        base = re.sub(r"_iter\d+", "", ndf.stem)
        kernel_groups[base].append(ndf)

    result: dict[str, KernelHistogram] = {}

    for base_name, ndjson_files in kernel_groups.items():
        cubin = cubins.get(base_name)
        if cubin is None:
            _log.warning("sass_resolve_dir: no cubin for %s — skipping", base_name)
            continue

        # Disassemble cubin → opcode_id table for the traced kernel function
        try:
            sass_r = subprocess.run(
                ["cutracer", "sass", str(cubin),
                 "--stdout", "--no-source-info", "--no-line-info"],
                capture_output=True, text=True, timeout=120,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            _log.warning("sass_resolve_dir: cutracer sass failed for %s: %s", cubin.name, exc)
            continue

        # Parse SASS: build per-function {sequential_idx: mnemonic} tables
        func_tables: dict[str, dict[int, str]] = {}
        current_fn: str | None = None
        fn_idx = 0
        for line in sass_r.stdout.splitlines():
            m = re.match(r"^\.text\.(\S+):", line)
            if m:
                current_fn = m.group(1)
                fn_idx = 0
                func_tables[current_fn] = {}
                continue
            m2 = re.match(r"^\s+/\*[0-9a-fA-F]+\*/\s+(?:@[!P\w]+ )?(\w[\w.]*)", line)
            if m2 and current_fn is not None:
                func_tables[current_fn][fn_idx] = m2.group(1).split(".")[0]
                fn_idx += 1

        # Find the matching SASS function (arch suffix after hash in base_name)
        arch_suffix = "_".join(base_name.split("_")[2:])
        matching = [k for k in func_tables if arch_suffix in k]
        tbl: dict[int, str] = func_tables[matching[0]] if matching else {}
        if not tbl:
            _log.warning("sass_resolve_dir: no SASS function matching '%s' in %s",
                         arch_suffix, cubin.name)

        # Aggregate opcode counts across all iterations
        raw_counts: dict[int, int] = defaultdict(int)
        for ndf in ndjson_files:
            try:
                with ndf.open() as fh:
                    for raw_line in fh:
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue
                        try:
                            obj = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue
                        oid = obj.get("opcode_id")
                        if oid is not None:
                            raw_counts[int(oid)] += int(obj.get("count", 1))
            except OSError as exc:
                _log.warning("sass_resolve_dir: could not read %s: %s", ndf, exc)

        if not raw_counts:
            _log.debug("sass_resolve_dir: no opcode records in %s", base_name)
            continue

        # Map opcode_id → mnemonic
        mnemonic_counts: dict[str, int] = defaultdict(int)
        for oid, cnt in raw_counts.items():
            mnemonic_counts[tbl.get(oid, f"OPCODE_{oid}")] += cnt

        kernel_name = arch_suffix or base_name
        result[kernel_name] = KernelHistogram(
            kernel_name=kernel_name,
            instruction_counts=dict(mnemonic_counts),
            warp_count=0,
        )

    return result


# ---------------------------------------------------------------------------
# NDJSON trace parser (streaming, zstd-aware)
# ---------------------------------------------------------------------------


def _open_trace(path: Path):
    """Open *path* for reading, decompressing zstd on the fly if needed."""
    if path.suffix == ".zst":
        try:
            import zstandard as zstd  # type: ignore[import]

            ctx = zstd.ZstdDecompressor()
            fh = path.open("rb")
            return ctx.stream_reader(fh)
        except ImportError:
            _log.warning(
                "zstandard package not installed; cannot decompress %s. "
                "Install it with: pip install zstandard",
                path,
            )
            raise
    return path.open("r", encoding="utf-8")


def parse_ndjson_trace(path: Path) -> Iterator[TraceRecord]:
    """Stream-parse an NDJSON (or NDJSON.zst) CUTracer trace file.

    Yields :class:`TraceRecord` objects one at a time to avoid loading the
    entire file into memory.
    """
    binary_mode = str(path).endswith(".zst")
    with _open_trace(path) as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.decode("utf-8") if binary_mode else raw_line
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                _log.debug("NDJSON parse error at line %d in %s: %s", lineno, path, exc)
                continue
            yield TraceRecord(
                record_type=obj.get("type", "unknown"),
                opcode_id=obj.get("opcode_id"),
                sass=obj.get("sass"),
                warp=obj.get("warp"),
                cta=obj.get("cta"),
                timestamp=obj.get("timestamp"),
                raw=obj,
            )
