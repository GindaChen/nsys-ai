"""NVTX layer-level auto-detection.

Scans kernel-to-NVTX attribution rows and finds the NVTX depth that
corresponds to "layer-level" annotations.  This enables the agent to
auto-discover ``layer_0, layer_1, …`` or ``TransformerBlock[0], …``
without the user specifying ``depth=2`` manually.

Detection heuristics (ordered by confidence):

1. **Numbered pattern**: names matching ``layer_0``, ``TransformerBlock[1]``,
   ``encoder.layers.2``, etc.  Regex captures trailing integer indices.

2. **Repeated siblings**: at some depth, identically-named nodes appear ≥ N
   times (e.g. PyTorch ``emit_nvtx`` producing ``aten::linear`` × 24).

3. **Fallback**: return ``layer_depth=None``.
"""

import re
from collections import Counter, defaultdict

# Matches trailing numeric index in various conventions:
#   layer_0, layer_1                 → _0, _1
#   TransformerBlock[0]              → [0]
#   encoder.layers.0                → .0
_NUMBERED_RE = re.compile(
    r"(?:"
    r"\[(\d+)\]"          # TransformerBlock[0]
    r"|\.(\d+)(?:\.|$)"   # encoder.layers.0
    r"|[_\-](\d+)(?:_|$)" # layer_0, layer_0_bwd, block-3
    r")"
)


def detect_layer_depth(attribution_rows: list[dict]) -> dict:
    """Scan attributed NVTX rows and find the layer-level depth.

    Args:
        attribution_rows: Output from ``attribute_kernels_to_nvtx()``,
            each row must have ``nvtx_path`` and ``nvtx_text`` keys.

    Returns:
        dict with keys:
          - ``layer_depth``: int or None (the detected depth)
          - ``layer_names``: list[str] of detected layer names
          - ``detection_method``: ``"numbered_pattern"`` | ``"repeated_siblings"``
                                  | ``"fallback_top_level"``
          - ``confidence``: float 0.0–1.0
    """
    if not attribution_rows:
        return _fallback()

    # Group NVTX texts by depth.
    # Kernels are attributed to the innermost NVTX, so nvtx_depth is always
    # the leaf level.  To discover layer names at intermediate depths (e.g.
    # layer_0 at depth 2), we walk the FULL path and register every ancestor.
    texts_by_depth: dict[int, list[str]] = defaultdict(list)
    for r in attribution_rows:
        text = r.get("nvtx_text", "")
        path = r.get("nvtx_path", "")
        if not text:
            continue
        parts = path.split(" > ") if path else [text]
        for i, part in enumerate(parts):
            texts_by_depth[i].append(part)

    if not texts_by_depth:
        return _fallback()

    # --- Heuristic 1: Numbered pattern ---
    best_numbered = _find_numbered_depth(texts_by_depth)
    if best_numbered is not None:
        found_depth, names = best_numbered
        return {
            "layer_depth": found_depth,
            "layer_names": sorted(set(names)),
            "detection_method": "numbered_pattern",
            "grouping_type": "layer",
            "confidence": min(1.0, len(set(names)) / 3),  # ≥3 layers → full confidence
        }

    # --- Heuristic 2: Repeated siblings ---
    best_repeated = _find_repeated_depth(texts_by_depth)
    if best_repeated is not None:
        found_depth, names = best_repeated
        return {
            "layer_depth": found_depth,
            "layer_names": sorted(set(names)),
            "detection_method": "repeated_siblings",
            "grouping_type": "operation",
            "confidence": 0.6,
        }

    # --- Heuristic 3: Fallback ---
    return _fallback()


def _find_numbered_depth(
    texts_by_depth: dict[int, list[str]],
) -> tuple[int, list[str]] | None:
    """Find depth with the most numbered-pattern names.

    Returns (depth, list_of_numbered_names) or None.
    """
    best_depth: int | None = None
    best_names: list[str] = []
    best_count = 0

    for depth in sorted(texts_by_depth):
        names = texts_by_depth[depth]
        numbered = [n for n in names if _NUMBERED_RE.search(n)]
        unique_numbered = set(numbered)
        if len(unique_numbered) >= 2 and len(unique_numbered) > best_count:
            best_depth = depth
            best_names = list(unique_numbered)
            best_count = len(unique_numbered)

    if best_depth is not None:
        return (best_depth, best_names)
    return None


def _find_repeated_depth(
    texts_by_depth: dict[int, list[str]],
) -> tuple[int, list[str]] | None:
    """Find depth with the largest cluster of identically-named nodes.

    For emit_nvtx style profiles where names like ``aten::linear`` repeat.
    Returns (depth, list_of_repeated_names) or None.
    """
    best_depth: int | None = None
    best_names: list[str] = []
    best_score = 0
    min_repeats = 2

    for depth in sorted(texts_by_depth):
        counts = Counter(texts_by_depth[depth])
        repeated = {name: cnt for name, cnt in counts.items() if cnt >= min_repeats}
        if not repeated:
            continue

        # Score = number of distinct repeated operations
        # This prevents a single repeated root scope at depth 0 from outscoring
        # a deeper hierarchical level that contains many distinct repeated operations.
        score = len(repeated)
        if score > best_score:
            best_depth = depth
            best_names = list(repeated.keys())
            best_score = score

    if best_depth is not None and best_score >= 2:
        return (best_depth, best_names)
    return None


def _fallback() -> dict:
    """Return fallback detection result (depth=None)."""
    return {
        "layer_depth": None,
        "layer_names": [],
        "detection_method": "fallback_top_level",
        "grouping_type": "flat",
        "confidence": 0.2,
    }


def is_outlier(value: float, all_values: list[float]) -> bool:
    """Detect outlier using IQR + minimum absolute threshold.

    Uses dual criteria (both must be true):
    1. Statistical: value > Q3 + 1.5 × IQR (or fence)
    2. Practical: value > median × 1.5

    For small sample sizes (< 4), falls back to value > median × 2.
    When IQR == 0 (uniform distribution), falls back to median × 2.
    """
    import statistics

    if len(all_values) < 2:
        return False

    med = statistics.median(all_values)

    if len(all_values) < 4:
        # Too few data points for IQR — use simple fallback
        return value > med * 2.0

    # quantiles with n=4 gives [Q1, Q2, Q3]
    q1, _, q3 = statistics.quantiles(all_values, n=4, method="inclusive")
    iqr = q3 - q1

    if iqr == 0:
        # All layers nearly identical — use percentage-based fallback
        fence = med * 2.0
    else:
        fence = q3 + 1.5 * iqr

    # Dual threshold: must exceed BOTH statistical AND practical
    return value > fence and value > med * 1.5
