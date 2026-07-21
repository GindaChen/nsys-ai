"""
fingerprint.py — Detects the ML framework and network topology from Nsight SQLite traces.

Extracts a ProfileFingerprint efficiently using O(1) string pool queries.
Also exposes ``get_profile_id`` — a content-derived stable identifier
for a profile, used as the canonical ``profile_id`` in evidence
artefacts (envelope and ``TraceSelection``).
"""

import hashlib
import json
import os
import re
import typing
from dataclasses import dataclass, field

from .connection import DB_ERRORS, wrap_connection

# Characters kept when trace-derived text is placed into prompt context or a
# report: word characters, whitespace and light punctuation. Anything else
# (markup, control characters, template/chat-template markers) is dropped.
_SCRUB_ALLOWED = re.compile(r"[^\w\s\-.,():'/]")


def _scrub(text: str, limit: int) -> str:
    """Normalize untrusted trace text for display: strip exotic characters,
    collapse all whitespace to single spaces (so no newline can forge an extra
    line in the prompt), and cap the length."""
    if not text:
        return ""
    return " ".join(_SCRUB_ALLOWED.sub("", str(text)).split())[:limit]


def _evidence_snippet(matched: str, keyword: str | None, width: int = 60) -> str:
    """A short window of the matched string, centred on the keyword.

    Captured strings are often whole log blocks — kilobytes long — so truncating
    from the start typically cuts the match off entirely and shows the reader
    nothing relevant. Centring the window on the hit is what makes the evidence
    actually reviewable.
    """
    flat = " ".join(str(matched or "").split())
    pos = flat.lower().find(keyword) if keyword else -1
    if pos < 0:
        return flat[:width]
    start = max(0, pos - width // 3)
    end = min(len(flat), start + width)
    return ("…" if start else "") + flat[start:end] + ("…" if end < len(flat) else "")


def _like_literal(keyword: str) -> str:
    """Escape SQL ``LIKE`` wildcards so a keyword matches literally.

    ``_`` matches any single character in SQL but is a literal in Python, so
    without escaping, SQL matches strings (``optimizer.step`` for the keyword
    ``optimizer_step``) that the Python-side keyword recovery cannot confirm —
    which previously produced evidence naming a keyword that never matched.
    """
    return keyword.replace("\\", r"\\").replace("%", r"\%").replace("_", r"\_")


@dataclass
class ProfileFingerprint:
    framework: str
    distributed: bool
    multi_node: bool
    nic_summary: str = ""
    precision_notes: list[str] = field(default_factory=list)
    # What the framework claim is actually based on: the keyword that matched
    # and a snippet of the string it matched in. Empty when no keyword matched
    # (framework is then the generic fallback). Kept so the claim is auditable
    # — a match against a log line is much weaker evidence than one against a
    # framework symbol, and a reader can now see which it was.
    framework_evidence: str = ""

    def to_prompt_string(self) -> str:
        # Both fields below carry text lifted out of the trace (NIC names, log
        # lines, environment variables, mangled symbols) into the agent's prompt
        # context, so both are scrubbed and length-capped the same way.
        framework_line = f"Framework: {self.framework}"
        safe_evidence = _scrub(self.framework_evidence, 120)
        if safe_evidence:
            # Labelled as trace-derived so the agent treats it as observed data
            # rather than instruction.
            framework_line += f' (matched, from trace text: "{safe_evidence}")'
        lines = [
            framework_line,
            f"Distributed training: {'yes' if self.distributed else 'no'}",
            f"Multi-node (RDMA): {'yes' if self.multi_node else 'no'}",
        ]
        safe_nic = _scrub(self.nic_summary, 200)
        if safe_nic:
            lines.append(f"Network: {safe_nic}")
        if self.precision_notes:
            lines.append("Notes: " + "; ".join(self.precision_notes))
        return "\n".join(lines)


# Ranked by priority / specificity.
# An environment matching multiple (e.g. Megatron and PyTorch) resolves to the first match.
FRAMEWORK_PRIORITY = [
    ("vLLM", ["paged_attention", "vllm", "SamplerOutput", "ModelRunner"]),
    ("SGLang", ["sglang", "RadixAttention", "TokenAttention"]),
    ("Megatron-LM", ["Megatron", "p2p_comm", "FlushGroups", "MegatronModule"]),
    # Keywords must be *distinctive*: they are matched case-insensitively as
    # substrings against every captured string (kernel symbols, NVTX text, and
    # also environment variables and log lines), so a generic token produces
    # confident nonsense. "ZeRO" and "offload" were removed for exactly that
    # reason — they matched `at::zeros`-style symbols and an unrelated
    # `OFFLOAD_TARGET_NAMES=...` env var, making DeepSpeed the reported
    # framework for most PyTorch traces. Prefer whole, framework-specific
    # identifiers over words that occur in ordinary CUDA/PyTorch output.
    ("DeepSpeed", ["DeepSpeed", "DeepSpeedEngine", "zero_optimization", "zero_stage"]),
    ("PyTorch", ["forward", "backward", "optimizer_step", "flash_attn"]),
]

_LOWERCASE_FRAMEWORK_PRIORITY = [
    (fw, [kw.lower() for kw in keywords]) for fw, keywords in FRAMEWORK_PRIORITY
]

# Known high-performance interconnect vendors
KNOWN_NIC_VENDORS = {
    5555: "Mellanox / NVIDIA",
    5348: "Broadcom",
    6082: "Cray",
    32902: "Intel",
}


def get_fingerprint(conn: typing.Any) -> ProfileFingerprint:
    adapter = wrap_connection(conn)
    tables = adapter.get_table_names()

    framework = "Generic CUDA"
    framework_evidence = ""

    def _check_framework(table: str, column: str) -> tuple[str, str] | None:
        """First framework (in priority order) with a keyword hit, plus evidence.

        One OR-ed sweep per framework, stopped by ``LIMIT 1`` — probing each
        keyword separately would cost a full LIKE scan per keyword, seconds on a
        multi-million-row StringIds table. The matched *value* is selected rather
        than a literal 1 so the specific keyword is recovered in Python for free,
        which is what makes the framework claim auditable rather than asserted.
        """
        try:
            for fw, lower_keywords in _LOWERCASE_FRAMEWORK_PRIORITY:
                like_conds = " OR ".join(
                    f"{column} LIKE '%{_like_literal(kw)}%' ESCAPE '\\'" for kw in lower_keywords
                )
                cur = adapter.execute(f"SELECT {column} FROM {table} WHERE {like_conds} LIMIT 1")
                row = cur.fetchone()
                if row:
                    matched = str(row[0] or "")
                    lowered = matched.lower()
                    kw = next((k for k in lower_keywords if k in lowered), None)
                    snippet = _evidence_snippet(matched, kw)
                    # Never name a keyword that cannot be confirmed inside the
                    # matched string: fabricated evidence would be worse than
                    # none, and this field exists precisely to be trustworthy.
                    origin = f"'{kw}' in {table}" if kw else table
                    return fw, f"{origin}: {snippet}"
        except DB_ERRORS:
            pass
        return None

    if "StringIds" in tables:
        found = _check_framework("StringIds", "value")
        if found:
            framework, framework_evidence = found

    if framework == "Generic CUDA" and "NVTX_EVENTS" in tables:
        # Fallback to direct event traces if no canonical stringIds are hit
        found = _check_framework("NVTX_EVENTS", "text")
        if found:
            framework, framework_evidence = found

    # Step B: Topology Search.
    # `distributed` is established first because multi-node is only meaningful
    # for a run that is distributed at all (see the NIC gating below).
    distributed = False
    if "NVTX_PAYLOAD_SCHEMAS" in tables:
        try:
            cur = adapter.execute(
                "SELECT 1 FROM NVTX_PAYLOAD_SCHEMAS WHERE name LIKE '%NCCL%' LIMIT 1"
            )
            if cur.fetchone():
                distributed = True
        except DB_ERRORS:
            pass

    # Fallback: NVTX_PAYLOAD_SCHEMAS doesn't always register collective schemas
    # (observed on some PyTorch versions where NCCL skips the typed-payload
    # path). Multi-device kernel activity is a robust signal for single-node
    # multi-rank training.
    if not distributed and "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        try:
            cur = adapter.execute(
                "SELECT COUNT(DISTINCT deviceId) FROM CUPTI_ACTIVITY_KIND_KERNEL"
            )
            row = cur.fetchone()
            if row and row[0] is not None and int(row[0]) > 1:
                distributed = True
        except DB_ERRORS:
            pass

    # Second fallback: NCCL collective kernels in the string pool. A multi-node
    # run is normally captured one report per rank, so the multi-device check
    # above sees a single device and would miss it — but a rank that runs NCCL
    # collectives is participating in a distributed job by definition.
    if not distributed and "StringIds" in tables:
        try:
            cur = adapter.execute("SELECT 1 FROM StringIds WHERE value LIKE '%nccl%' LIMIT 1")
            if cur.fetchone():
                distributed = True
        except DB_ERRORS:
            pass

    # RDMA-capable NIC hardware. Its mere presence does NOT mean the run spanned
    # nodes — most datacenter hosts ship one — so multi_node additionally
    # requires distributed evidence. Without that gate a single-GPU trace on a
    # Mellanox host reported the self-contradictory
    # "Distributed: no, Multi-node: yes". The hardware is still reported through
    # nic_summary regardless, so the observation is kept, only the claim is not.
    nic_present = False
    nic_summary = ""
    if "TARGET_INFO_NIC_INFO" in tables:
        try:
            vendor_keys = ",".join(map(str, KNOWN_NIC_VENDORS.keys()))
            cur = adapter.execute(
                f"SELECT vendorId, name FROM TARGET_INFO_NIC_INFO "
                f"WHERE CAST(vendorId AS INTEGER) IN ({vendor_keys}) "
                f"OR name LIKE 'mlx5_%' OR name LIKE 'cxi%' LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                nic_present = True
                v_id = -1
                try:
                    v_id = int(row[0])
                except (ValueError, TypeError):
                    pass
                vendor_name = KNOWN_NIC_VENDORS.get(v_id, "NIC")
                nic_summary = (
                    f"{vendor_name} hardware detected (vendorId: {row[0]}, name: {row[1]})"
                )
        except DB_ERRORS:
            pass
    multi_node = nic_present and distributed

    return ProfileFingerprint(
        framework=framework,
        distributed=distributed,
        multi_node=multi_node,
        nic_summary=nic_summary,
        precision_notes=[],
        framework_evidence=framework_evidence,
    )


# ---------------------------------------------------------------------------
# profile_id — stable content-derived identifier
# ---------------------------------------------------------------------------

PROFILE_ID_VERSION = "nsys1"
"""Algorithm tag for :func:`get_profile_id`. Bump (``nsys2``, …) if the
contributing columns or the canonical serialisation change."""


def get_profile_id(
    conn: typing.Any, *, fallback_path: str | os.PathLike[str] | None = None
) -> str:
    """Return a stable content-derived id for a Nsight Systems profile.

    The hash spans only fields stamped at *profile-capture* time, so it
    survives ``.nsys-rep`` → ``.sqlite`` re-export, ``VACUUM``,
    ``journal_mode`` changes, and filesystem moves:

      - ``TARGET_INFO_SESSION_START_TIME.utcEpochNs``
      - ``ANALYSIS_DETAILS`` rows (``duration / startTime / stopTime``)
      - ``TARGET_INFO_GPU`` rows (``id``, ``name``)
      - ``TARGET_INFO_GPU.uuid`` values when the column exists
        (newer Nsight); older exports keep ``uuid`` on
        ``TARGET_INFO_CUDA_DEVICE`` and that contribution degrades to
        empty rather than failing
      - distinct ``ANALYSIS_FILE.globalPid`` values
      - ``CUPTI_ACTIVITY_KIND_KERNEL`` row count

    Each contribution is JSON-encoded as a ``(label, value)`` pair before
    hashing, so values containing ``|`` / ``;`` / ``\\n`` can never collide
    with neighbouring values via separator ambiguity.

    Format::

        nsys1:sha256:<64-hex>   # preferred — content-derived
        nsys1:path:<64-hex>     # fallback — when no Nsight metadata is reachable

    The fallback fires for backends that expose only the parquet cache
    (``backend='parquetdir'``) where META_DATA / TARGET_INFO tables were
    never materialised. Callers should pass ``self.prof.conn`` (the
    SQLite source where available); the function never reads
    ``self.prof.db`` semantics — it just runs SQL.

    ``fallback_path`` is hashed verbatim — pass an absolute path if you
    want it to compare equal across working-directory changes.

    .. note::
       When *both* ``conn`` is None / empty and ``fallback_path`` is
       None, the function returns ``nsys1:sha256:<sha256 of empty>``.
       That is a recognisable null-id sentinel: every such caller
       collapses to the same value. Always pass ``fallback_path`` if
       cross-call distinguishability matters.

    .. note::
       ``NULLS LAST`` requires SQLite ≥ 3.30 (2019-10-04) and is native
       in DuckDB. Older SQLite raises ``OperationalError`` on the
       ``ORDER BY ... NULLS LAST`` clause; the offending contribution
       is caught and degraded to empty rather than crashing.
    """
    # Coerce ``fallback_path`` once: callers often pass ``pathlib.Path``
    # (e.g. via ``Profile(Path(...))`` → ``Profile.path``). Both fallback
    # branches below would otherwise crash with AttributeError on
    # ``.encode("utf-8")``.
    fallback_str = os.fspath(fallback_path) if fallback_path is not None else None

    def _path_id(p: str) -> str:
        digest = hashlib.sha256(p.encode("utf-8")).hexdigest()
        return f"{PROFILE_ID_VERSION}:path:{digest}"

    def _null_id() -> str:
        """The shared null-id sentinel — same value whether ``conn`` is
        None or merely empty, so consumers can detect "no usable
        identity" with a single equality check."""
        return f"{PROFILE_ID_VERSION}:sha256:{hashlib.sha256(b'').hexdigest()}"

    # None-safe shortcut: wrap_connection(None) returns a SQLiteAdapter
    # over None, whose .execute() raises AttributeError (not a DB error),
    # so the loop's try/except wouldn't catch it. Skip the queries.
    if conn is None:
        return _path_id(fallback_str) if fallback_str else _null_id()

    adapter = wrap_connection(conn)

    def _scalar(sql: str) -> typing.Any:
        """Return the single cell from a one-row/one-column query, or None."""
        try:
            row = adapter.execute(sql).fetchone()
            return row[0] if row and row[0] is not None else None
        except DB_ERRORS:
            return None

    def _rows(sql: str) -> list[list[typing.Any]]:
        """Return all rows as a list of lists (JSON-serialisable)."""
        try:
            return [list(r) for r in adapter.execute(sql).fetchall() if r]
        except DB_ERRORS:
            return []

    # NULLS LAST: SQLite defaults to NULLS FIRST, DuckDB to NULLS LAST —
    # pin it so two engine adapters on the same data hash identically.
    #
    # Determinism rule for ``_rows`` queries: ``ORDER BY`` must cover the
    # superset of every SELECTed column. Otherwise two rows that tie on
    # the partial ORDER BY can have *different* selected values, and the
    # relative order between them is engine-defined — VACUUM, schema
    # rebuild, or a different adapter can flip them and change the hash.
    parts: list[tuple[str, typing.Any]] = [
        ("session_start_utc_ns", _scalar("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME")),
        # ANALYSIS_DETAILS is conventionally single-row, but serialise all
        # rows in deterministic order so the contribution stays stable
        # if a future profile carries more than one row.
        (
            "analysis_details",
            _rows(
                "SELECT duration, startTime, stopTime FROM ANALYSIS_DETAILS "
                "ORDER BY startTime NULLS LAST, stopTime NULLS LAST, duration NULLS LAST"
            ),
        ),
        # GPU id + name works on every Nsight schema we've seen.
        (
            "gpus",
            _rows(
                "SELECT id, name FROM TARGET_INFO_GPU "
                "ORDER BY id NULLS LAST, name NULLS LAST"
            ),
        ),
        # uuid lives on TARGET_INFO_GPU in newer Nsight exports and on
        # TARGET_INFO_CUDA_DEVICE in older ones (and in our minimal test
        # fixture). Query each in its own contribution so a schema
        # mismatch degrades only that part, not the whole id.
        (
            "gpu_uuids",
            _rows(
                "SELECT uuid FROM TARGET_INFO_GPU "
                "ORDER BY id NULLS LAST, uuid NULLS LAST"
            ),
        ),
        # ``TARGET_INFO_CUDA_DEVICE`` commonly carries multiple rows per
        # (gpuId, cudaId) — one per process — so ``pid`` and ``uuid`` are
        # required tie-breakers to make the hash stable across engines.
        (
            "cuda_device_uuids",
            _rows(
                "SELECT uuid FROM TARGET_INFO_CUDA_DEVICE "
                "ORDER BY gpuId NULLS LAST, cudaId NULLS LAST, "
                "pid NULLS LAST, uuid NULLS LAST"
            ),
        ),
        (
            "pids",
            # DISTINCT guarantees no ties on the SELECTed column, so
            # ORDER BY globalPid alone is total.
            _rows(
                "SELECT DISTINCT globalPid FROM ANALYSIS_FILE "
                "ORDER BY globalPid NULLS LAST"
            ),
        ),
        ("kernel_count", _scalar("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL")),
    ]

    # If every contribution is missing/empty the conn carries no Nsight
    # metadata (e.g. parquetdir backend). Fall back to a clearly-labelled
    # path-derived id rather than collapse every such profile to the
    # same constant hash.
    def _empty(v: typing.Any) -> bool:
        return v is None or v == "" or v == [] or v == 0

    if all(_empty(v) for _, v in parts):
        # Same shape as the ``conn is None`` branch above: path-fallback
        # if available, else the null-id sentinel. This keeps the
        # "no usable identity" check a single equality.
        return _path_id(fallback_str) if fallback_str else _null_id()

    # JSON canonical: structured, unambiguous, no separator collisions.
    # ``sort_keys=False`` because ``parts`` is already ordered; flipping
    # the order would change the hash, which is intended (algorithm
    # change → bump PROFILE_ID_VERSION).
    canonical = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{PROFILE_ID_VERSION}:sha256:{digest}"
