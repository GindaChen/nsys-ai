"""Correlate CUTracer kernel names with nsys kernel names.

CUTracer extracts kernel names from SASS binary metadata, producing short
names like ``volta_h16gemm_32x32_nt_nn``.  nsys stores full C++ demangled
names including template parameters and argument lists, e.g.
``void volta_h16gemm_32x32_nt_nn<bool, int, 128>(cublas::...)``.

Strategy
--------
1. Strip leading ``void``, return types, and namespace prefixes.
2. Strip C++ template arguments ``<…>`` and function argument lists ``(…)``.
3. Collapse whitespace and lower-case.
4. Try exact match first; fall back to longest common substring (LCS).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_STRIP_RETURN_TYPE = re.compile(r"^(?:void|__global__|__device__|static)\s+")
_STRIP_TEMPLATES = re.compile(r"<[^<>]*>")   # non-nested; handles most cases
_STRIP_ARGS = re.compile(r"\([^()]*\)")       # non-nested
_STRIP_NAMESPACE = re.compile(r"\w+::")
_WHITESPACE = re.compile(r"\s+")

# Max iterations when stripping nested template / argument brackets.
# 4 levels handles >99% of real CUDA kernel names.
_MAX_STRIP_DEPTH: int = 4


def normalize_kernel_name(name: str) -> str:
    """Return a normalised, lowercased kernel name suitable for fuzzy matching.

    >>> normalize_kernel_name("void volta_h16gemm_32x32_nt_nn<bool, int>(cublas::...)")
    'volta_h16gemm_32x32_nt_nn'
    """
    s = name.strip()
    # Strip common return type / qualifier prefixes
    s = _STRIP_RETURN_TYPE.sub("", s)
    # Iteratively strip template args (handles single nesting level, which
    # covers >95% of CUDA kernel names in practice)
    for _ in range(_MAX_STRIP_DEPTH):
        s2 = _STRIP_TEMPLATES.sub("", s)
        if s2 == s:
            break
        s = s2
    # Strip argument lists
    for _ in range(_MAX_STRIP_DEPTH):
        s2 = _STRIP_ARGS.sub("", s)
        if s2 == s:
            break
        s = s2
    # Strip namespace qualifiers
    s = _STRIP_NAMESPACE.sub("", s)
    s = _WHITESPACE.sub("", s)
    return s.lower().rstrip("_")


def _lcs_ratio(a: str, b: str) -> float:
    """Return the longest-common-substring ratio between *a* and *b* (0–1)."""
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_kernels(
    nsys_kernels: list[str],
    cutracer_kernels: list[str],
    *,
    min_ratio: float = 0.75,
) -> dict[str, str | None]:
    """Match each nsys kernel name to the best-matching CUTracer kernel name.

    Parameters
    ----------
    nsys_kernels:
        Kernel names as stored in ``StringIds`` of the Nsight Systems profile.
    cutracer_kernels:
        Kernel names extracted from CUTracer histogram filenames.
    min_ratio:
        Minimum similarity ratio to accept a fuzzy (LCS) match (0–1).
        Only applies to step 3 — exact matches (step 1) and substring
        containment (step 2) are never gated by this threshold.
        Default 0.75: rejects spurious matches caused by shared affixes
        such as ``_fwd_``, ``_kernel``, ``_ring_ll`` across unrelated kernels.
        Pairs below this threshold are mapped to ``None``.

    Returns
    -------
    dict mapping each *nsys_kernel* → best-matching *cutracer_kernel* or ``None``.
    """
    norm_ct: dict[str, str] = {normalize_kernel_name(k): k for k in cutracer_kernels}

    result: dict[str, str | None] = {}
    for nsys_k in nsys_kernels:
        norm_nsys = normalize_kernel_name(nsys_k)

        # 1. Exact match after normalisation
        if norm_nsys in norm_ct:
            result[nsys_k] = norm_ct[norm_nsys]
            continue

        # 2. CUTracer name is a prefix/substring of the normalised nsys name
        #    (e.g. CUTracer: "volta_h16gemm", nsys: "volta_h16gemm_32x32_nt_nn")
        substring_match: str | None = None
        for norm_k, orig_k in norm_ct.items():
            if norm_k and norm_k in norm_nsys:
                if substring_match is None or len(norm_k) > len(
                    normalize_kernel_name(substring_match)
                ):
                    substring_match = orig_k
        if substring_match is not None:
            result[nsys_k] = substring_match
            continue

        # 3. Fuzzy LCS fallback
        best_orig: str | None = None
        best_ratio = 0.0
        for norm_k, orig_k in norm_ct.items():
            r = _lcs_ratio(norm_nsys, norm_k)
            if r > best_ratio:
                best_ratio = r
                best_orig = orig_k

        result[nsys_k] = best_orig if best_ratio >= min_ratio else None

    return result


def build_nsys_kernel_list(conn) -> list[str]:
    """Return all kernel names from the nsys profile connection.

    Works with both SQLite (raw) and DuckDB (parquet cache) connections.
    """
    from nsys_ai.connection import wrap_connection

    adapter = wrap_connection(conn)
    tables = adapter.get_table_names()

    # Parquet cache path — kernels table has a 'name' column
    if "kernels" in tables:
        try:
            rows = adapter.execute("SELECT DISTINCT name FROM kernels WHERE name IS NOT NULL").fetchall()
            return [r[0] for r in rows if r[0]]
        except Exception:
            pass

    # SQLite fallback — join shortName / demangledName with StringIds
    resolved = adapter.resolve_activity_tables()
    kernel_table = resolved.get("kernel", "CUPTI_ACTIVITY_KIND_KERNEL")
    try:
        rows = adapter.execute(
            f"""
            SELECT DISTINCT COALESCE(d.value, s.value) AS name
            FROM {kernel_table} k
            LEFT JOIN StringIds s ON k.shortName = s.id
            LEFT JOIN StringIds d ON k.demangledName = d.id
            WHERE COALESCE(d.value, s.value) IS NOT NULL
            """
        ).fetchall()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []
