"""SASS opcode → category mapping for Volta/Turing/Ampere/Hopper GPUs.

Categories:
  compute  — FP32/FP64/INT arithmetic (FFMA, IMAD, DADD, …)
  tensor   — Tensor Core matrix ops (HMMA, IMMA, BMMA, …)
  memory   — Global/shared/local memory (LDG, STG, LDS, STS, …)
  sync     — Barriers and memory fences (BAR, MEMBAR, …)
  control  — Flow control (BRA, RET, CALL, …)
  special  — Warp-level ops (SHFL, VOTE, …)
  other    — Anything unrecognised
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Category sets (prefix-matched — "FFMA" matches opcode "FFMA.FTZ" etc.)
# ---------------------------------------------------------------------------

_COMPUTE_PREFIXES: frozenset[str] = frozenset(
    {
        # FP32
        "FFMA", "FMUL", "FADD", "FCHK", "FSET", "FCMP", "FMNMX", "FSWZ",
        "FSEL", "FSETP", "MUFU",
        # FP16
        "HADD2", "HMUL2", "HFMA2", "HSET2", "HSETP2", "HMMA2",
        # FP64
        "DADD", "DFMA", "DMUL", "DSET", "DSETP", "DCMP",
        # Integer
        "IMAD", "IMUL", "IADD", "IADD3", "IMNMX", "ISET", "ISETP",
        "ICMP", "ISCADD", "XMAD",
        # Bitwise / shift
        "LOP", "LOP3", "SHF", "SHL", "SHR", "BFE", "BFI", "PRMT",
        "POPC", "FLO",
        # Select / move
        "SEL", "MOV", "CS2R", "S2R", "B2R", "R2B", "I2I", "I2F", "F2I", "F2F",
    }
)

_TENSOR_PREFIXES: frozenset[str] = frozenset(
    {
        "HMMA",   # FP16/BF16 Tensor Core (Volta+)
        "IMMA",   # INT8 Tensor Core (Turing+)
        "BMMA",   # Binary Tensor Core
        "DMMA",   # FP64 Tensor Core (Ampere A100)
    }
)

_MEMORY_PREFIXES: frozenset[str] = frozenset(
    {
        # Global memory
        "LDG", "STG", "LDGSTS",
        # Shared memory
        "LDS", "STS", "LDSM",
        # Local / generic
        "LD", "ST", "LDL", "STL", "LDC",
        # Atomic / reduction
        "ATOM", "ATOMS", "RED", "REDS",
        # Texture / surface
        "TEX", "TLD", "TLD4", "TMML", "TXQ",
        "SULD", "SUST", "SURED",
        # Cache control
        "CCTL", "CCTLL",
    }
)

_SYNC_PREFIXES: frozenset[str] = frozenset(
    {
        "BAR", "BSSY", "BSYNC",
        "SSY", "SYNC",
        "MEMBAR",
        "DEPBAR",
    }
)

_CONTROL_PREFIXES: frozenset[str] = frozenset(
    {
        "BRA", "BRAL", "BREAK", "BRX",
        "CAL", "PRET", "RET",
        "JMP", "JMXU",
        "EXIT", "BPT",
        "NOP",
        "RPCMOV", "ISBERD",
    }
)

_SPECIAL_PREFIXES: frozenset[str] = frozenset(
    {
        "SHFL", "SHFLI",
        "VOTE", "VOTEU",
        "MATCH",
        "REDUX",
        "LEPC", "GETCRSPTR", "GETLMEMBASE",
    }
)

# Ordered list for classify_opcode — tensor checked before compute because
# HMMA would otherwise match the HADD2-style compute check via prefix.
_ORDERED_CATEGORIES: list[tuple[str, frozenset[str]]] = [
    ("tensor", _TENSOR_PREFIXES),
    ("compute", _COMPUTE_PREFIXES),
    ("memory", _MEMORY_PREFIXES),
    ("sync", _SYNC_PREFIXES),
    ("control", _CONTROL_PREFIXES),
    ("special", _SPECIAL_PREFIXES),
]

# ---------------------------------------------------------------------------
# Ideal cycles-per-instruction for stall scoring
# (approximate; Volta SM throughput)
# ---------------------------------------------------------------------------
_IDEAL_CYCLES: dict[str, float] = {
    "FFMA": 4.0,
    "IMAD": 4.0,
    "HMMA": 8.0,
    "HADD2": 2.0,
    "HMUL2": 2.0,
    "HFMA2": 2.0,
    "LDS": 23.0,     # shared memory, ~L1 latency
    "STS": 20.0,
    "LDG": 30.0,     # global memory L2 hit; L1 miss → 100+; HBM → 500+
    "STG": 30.0,
    "BAR": 16.0,
    "SHFL": 8.0,
}
_DEFAULT_IDEAL_CYCLES = 8.0


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def classify_opcode(opcode: str) -> str:
    """Return the category string for a SASS opcode mnemonic.

    The input may include modifiers separated by dots (e.g. ``"FFMA.FTZ"``);
    only the first token is used for matching.

    Returns one of: ``"tensor"``, ``"compute"``, ``"memory"``, ``"sync"``,
    ``"control"``, ``"special"``, ``"other"``.
    """
    base = opcode.split(".")[0].upper()
    for category, prefixes in _ORDERED_CATEGORIES:
        if base in prefixes:
            return category
    return "other"


def ideal_cycles(opcode: str) -> float:
    """Return the approximate ideal cycles/instruction for *opcode*.

    Used as the denominator when computing stall scores.
    """
    base = opcode.split(".")[0].upper()
    return _IDEAL_CYCLES.get(base, _DEFAULT_IDEAL_CYCLES)


def stall_score(opcode: str, total_cycles: int, total_count: int) -> float:
    """Return a 0–1 stall score for an instruction bucket.

    A score of 0 means no stall (actual ≤ ideal).
    A score close to 1 means almost all cycles were stall cycles.

    Formula: ``max(0, (actual - ideal) / actual)``
    """
    if total_count == 0:
        return 0.0
    actual = total_cycles / total_count
    ideal = ideal_cycles(opcode)
    if actual <= ideal:
        return 0.0
    return (actual - ideal) / actual
