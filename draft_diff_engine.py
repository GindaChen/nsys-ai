"""
Draft implementation for the base Diff Engine focusing on Memory-Efficient SQL Aggregation
and Triton JIT Kernel Fuzzy Matching.
"""
import sqlite3
import re
from typing import Set, Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class KernelStats:
    name: str
    invocations: int
    total_ns: int
    avg_ns: float
    stddev_ns: float
    registers_per_thread: int
    dynamic_shared_mem: int
    avg_block_x: float
    avg_grid_x: float

def extract_aggregated_kernels(db_path: str, gpu_id: Optional[int] = None) -> List[KernelStats]:
    """
    Extremely memory-efficient SQL aggregation to prevent OOM on 10GB+ traces.
    Avoids loading millions of raw kernel rows into Python.
    Computes count, total, avg, stddev, and launch config parameters directly in SQLite.
    """
    conn = sqlite3.connect(db_path)
    
    # Base query for CUPTI_ACTIVITY_KIND_KERNEL via StringIds for demangled names
    # Notes: In actual nsys-ai schema, this requires joining CUPTI_ACTIVITY_KIND_KERNEL with StringIds
    # This is a conceptual draft representative of the required SQL GROUP BY structure
    
    gpu_filter = f"WHERE deviceId = {gpu_id}" if gpu_id is not None else ""
    
    query = f"""
    SELECT
        s.value AS name,
        COUNT(k.start) AS invocations,
        SUM(k.end - k.start) AS total_ns,
        AVG(k.end - k.start) AS avg_ns,
        -- SQLite doesn't have native STDEV; could use sum of squares trick or custom func
        -- For now, conceptual placeholder for stddev calculation
        0.0 AS stddev_ns, 
        MAX(k.registersPerThread) AS registers_per_thread,
        MAX(k.dynamicSharedMemory) AS dynamic_shared_mem,
        AVG(k.blockX) AS avg_block_x,
        AVG(k.gridX) AS avg_grid_x
    FROM
        CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN
        StringIds s ON k.demangledName = s.id
    {gpu_filter}
    GROUP BY
        s.value
    ORDER BY
        total_ns DESC
    """
    
    # In a real scenario, we'd handle the query carefully.
    # We yield/return the structured data classes
    kernels = []
    try:
        cursor = conn.cursor()
        for row in cursor.execute(query):
            kernels.append(KernelStats(
                name=row[0],
                invocations=row[1],
                total_ns=row[2],
                avg_ns=row[3],
                stddev_ns=row[4],
                registers_per_thread=row[5],
                dynamic_shared_mem=row[6],
                avg_block_x=row[7],
                avg_grid_x=row[8]
            ))
    except sqlite3.OperationalError as e:
        print(f"[ERROR] Failed to execute trace extraction (Schema Drift?): {e}")
    finally:
        conn.close()
        
    return kernels


def normalize_triton_kernel_name(raw_name: str) -> str:
    """
    Strips auto-generated sequence numbers and compilation hashes
    from standard PyTorch/Triton kernels to enable fuzzy matching.
    """
    # Remove trailing sequential numbers (e.g., triton_poi_fused_add_0 -> triton_poi_fused_add)
    name = re.sub(r'_\d+$', '', raw_name)
    
    # Remove obvious hexa-hashes (e.g., usually 6-8 chars long at the end)
    name = re.sub(r'_[a-fA-F0-9]{6,8}$', '', name)
    
    return name

def perform_fuzzy_kernel_match(before_kernels: List[KernelStats], after_kernels: List[KernelStats]) -> Dict[str, Optional[KernelStats]]:
    """
    Two-stage matching algorithm to pair up Base and Candidate kernels.
    Stage 1: Exact Name Match
    Stage 2: Fuzzy/Normalized Match for JIT kernels (Triton / torch.compile)
    
    Returns a mapping of { BeforeKernelName : AfterKernelStats (or None if removed) }
    """
    matched_pairs = {}
    
    after_dict = {k.name: k for k in after_kernels}
    unmatched_after = list(after_kernels)
    unmatched_before = []

    # --- Stage 1: Exact Match ---
    for b_kernel in before_kernels:
        if b_kernel.name in after_dict:
            matched_pairs[b_kernel.name] = after_dict[b_kernel.name]
            unmatched_after.remove(after_dict[b_kernel.name])
        else:
            unmatched_before.append(b_kernel)

    # Build normalized mapping for remaining After kernels
    # Format: { normalized_name: [KernelStats_1, KernelStats_2, ...] }
    normalized_after_map: Dict[str, List[KernelStats]] = {}
    for a_kernel in unmatched_after:
        norm_a = normalize_triton_kernel_name(a_kernel.name)
        normalized_after_map.setdefault(norm_a, []).append(a_kernel)

    # --- Stage 2: Fuzzy Normalized Match ---
    for b_kernel in unmatched_before:
        norm_b = normalize_triton_kernel_name(b_kernel.name)
        
        if norm_b in normalized_after_map and normalized_after_map[norm_b]:
            # Pop the first available match. (In a real scenario with multiple collisions,
            # we'd compare 'invocations' or Grid/Block configs to break the tie).
            matched_a = normalized_after_map[norm_b].pop(0)
            matched_pairs[b_kernel.name] = matched_a
        else:
            # Truly removed (or wrapper shift if NVTX depth changed, which requires NVTX path fallback)
            matched_pairs[b_kernel.name] = None 
            
    # Note: Any kernels left in `unmatched_after` at this point are classified as "New Kernels"
            
    return matched_pairs
