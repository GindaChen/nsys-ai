"""
Regression tests for nvtx_tree.build_nvtx_tree() against both NVTX schema variants.

- Schema A: NVTX_EVENTS has only inline text (no textId column).
- Schema B: NVTX_EVENTS has textId -> StringIds (COALESCE(text, s.value) for display).

Ensures cleanup does not hard-depend on textId so that both legal Nsight export
formats work and Tree/Timeline load without OperationalError.
"""
import sqlite3

import pytest

from nsys_ai.nvtx_tree import build_nvtx_tree
from nsys_ai import profile as _profile


def test_build_nvtx_tree_text_only_schema(minimal_nsys_db_path):
    """build_nvtx_tree works when NVTX_EVENTS has only 'text' column (no textId)."""
    with _profile.open(minimal_nsys_db_path) as prof:
        roots = build_nvtx_tree(prof, device=0, trim=(0, 5_000_000_000))
    assert isinstance(roots, list)
    # Fixture has NVTX 'train_step' / 'forward' and kernels; we may get 0 or more roots
    # depending on trim and thread mapping; main assertion is no OperationalError.


def test_build_nvtx_tree_text_id_schema(tmp_path):
    """build_nvtx_tree works when NVTX_EVENTS has textId -> StringIds (no inline text)."""
    db_path = tmp_path / "textid.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO StringIds VALUES (0, ''), (1, 'nvtx_span_a'), (2, 'kernel_A');
        CREATE TABLE TARGET_INFO_GPU (id INTEGER PRIMARY KEY, name TEXT, busLocation TEXT, totalMemory INTEGER, smCount INTEGER, chipName TEXT, memoryBandwidth INTEGER);
        INSERT INTO TARGET_INFO_GPU VALUES (0, 'Test GPU', '', 8589934592, 108, 'Test', 0);
        CREATE TABLE TARGET_INFO_CUDA_DEVICE (gpuId INT, cudaId INT, pid INT, uuid TEXT, numMultiprocessors INT);
        INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (0, 0, 100, '', 108);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (globalPid INT, deviceId INT, streamId INT, correlationId INT, start INT, end INT, shortName INT, demangledName INT, gridX INT, gridY INT, gridZ INT, blockX INT, blockY INT, blockZ INT);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 0, 7, 1, 1000000, 2000000, 2, 0, 1,1,1,1,1,1);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (globalTid INT, correlationId INT, start INT, end INT, nameId INT);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (100, 1, 500000, 1000000, 0);
        CREATE TABLE NVTX_EVENTS (globalTid INT, start INT NOT NULL, end INT NOT NULL, text TEXT, textId INTEGER, eventType INT DEFAULT 59, rangeId INT DEFAULT 0);
        INSERT INTO NVTX_EVENTS VALUES (100, 400000, 2500000, NULL, 1, 59, 0);
    """)
    conn.commit()
    conn.close()

    with _profile.open(str(db_path)) as prof:
        roots = build_nvtx_tree(prof, device=0, trim=(0, 5_000_000_000))
    assert isinstance(roots, list)
