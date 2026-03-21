"""
conftest.py — Shared pytest fixtures for nsys-ai tests.

All fixtures here are available to every test module without explicit imports.
"""

import sqlite3

import pytest

# ---------------------------------------------------------------------------
# Minimal in-memory SQLite database that mimics an Nsight Systems export.
# Only the tables and columns exercised by the test suite are created; this
# avoids the need for a real .sqlite profile during CI.
# ---------------------------------------------------------------------------

_NSYS_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS StringIds (
    id      INTEGER PRIMARY KEY,
    value   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS TARGET_INFO_GPU (
    id              INTEGER PRIMARY KEY,
    name            TEXT,
    busLocation     TEXT DEFAULT '',
    totalMemory     INTEGER DEFAULT 0,
    smCount         INTEGER DEFAULT 0,
    chipName        TEXT DEFAULT '',
    memoryBandwidth INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS TARGET_INFO_CUDA_DEVICE (
    gpuId    INTEGER,
    cudaId   INTEGER,
    pid      INTEGER DEFAULT 0,
    uuid     TEXT DEFAULT '',
    numMultiprocessors INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS CUPTI_ACTIVITY_KIND_KERNEL (
    globalPid       INTEGER DEFAULT 0,
    deviceId        INTEGER DEFAULT 0,
    streamId        INTEGER DEFAULT 0,
    correlationId   INTEGER DEFAULT 0,
    start           INTEGER NOT NULL,
    end             INTEGER NOT NULL,
    shortName       INTEGER NOT NULL,
    demangledName   INTEGER DEFAULT 0,
    gridX           INTEGER DEFAULT 1,
    gridY           INTEGER DEFAULT 1,
    gridZ           INTEGER DEFAULT 1,
    blockX          INTEGER DEFAULT 1,
    blockY          INTEGER DEFAULT 1,
    blockZ          INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS CUPTI_ACTIVITY_KIND_MEMCPY (
    globalPid       INTEGER DEFAULT 0,
    deviceId        INTEGER DEFAULT 0,
    streamId        INTEGER DEFAULT 0,
    correlationId   INTEGER DEFAULT 0,
    copyKind        INTEGER DEFAULT 0,
    bytes           INTEGER DEFAULT 0,
    srcKind         INTEGER DEFAULT 0,
    dstKind         INTEGER DEFAULT 0,
    start           INTEGER NOT NULL,
    end             INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS CUPTI_ACTIVITY_KIND_MEMSET (
    globalPid       INTEGER DEFAULT 0,
    deviceId        INTEGER DEFAULT 0,
    streamId        INTEGER DEFAULT 0,
    correlationId   INTEGER DEFAULT 0,
    bytes           INTEGER DEFAULT 0,
    value           INTEGER DEFAULT 0,
    start           INTEGER NOT NULL,
    end             INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS CUPTI_ACTIVITY_KIND_RUNTIME (
    globalTid       INTEGER DEFAULT 0,
    correlationId   INTEGER DEFAULT 0,
    start           INTEGER NOT NULL,
    end             INTEGER NOT NULL,
    nameId          INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS NVTX_EVENTS (
    globalTid       INTEGER DEFAULT 0,
    start           INTEGER NOT NULL,
    end             INTEGER DEFAULT -1,
    text            TEXT DEFAULT '',
    eventType       INTEGER DEFAULT 59,
    rangeId         INTEGER DEFAULT 0,
    textId          INTEGER DEFAULT NULL
);
"""

_NSYS_SEED_SQL = """\
INSERT INTO StringIds VALUES
    (1, 'kernel_A'), (2, 'kernel_B'), (10, 'nccl_AllReduce_kernel'),
    -- CUDA API names for anti-pattern detection tests
    (20, 'cudaDeviceSynchronize'),
    (21, 'cudaMemcpy'),
    (22, 'cudaMemcpyAsync'),
    (23, 'cudaMemset'),
    (24, 'cudaLaunchKernel'),
    -- Additional for V3 overlap diagnosis
    (25, 'cudaStreamSynchronize'),
    (11, 'nccl_ReduceScatter_kernel');

INSERT INTO TARGET_INFO_GPU VALUES
    (0, 'NVIDIA Test GPU', '0000:00:00.0', 8589934592, 108, 'TestChip', 0);

INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES
    (0, 0, 100, '', 108);

INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
    -- Stream 7: compute kernels
    (100, 0, 7, 1, 1000000, 2000000, 1, 1, 32, 1, 1, 256, 1, 1),
    (100, 0, 7, 2, 3000000, 4000000, 2, 2, 16, 1, 1, 128, 1, 1),
    -- Stream 8: NCCL on separate stream
    (100, 0, 8, 10, 2500000, 3500000, 10, 10, 1, 1, 1, 512, 1, 1),
    -- Stream 7: NCCL on SAME stream as compute (same-stream anti-pattern)
    (100, 0, 7, 12, 4500000, 5500000, 11, 11, 1, 1, 1, 256, 1, 1),
    -- Stream 7: another compute kernel after a large gap (for idle gap testing)
    (100, 0, 7, 13, 8000000, 9000000, 1, 1, 32, 1, 1, 256, 1, 1);

INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
    -- Kernel launches (existing)
    (100, 1,   900000,  1000000, 24),   -- cudaLaunchKernel
    (100, 2,  2900000,  3000000, 24),   -- cudaLaunchKernel
    (100, 10, 2400000,  2500000, 24),   -- cudaLaunchKernel
    (100, 12, 4400000,  4500000, 24),   -- cudaLaunchKernel (NCCL same-stream)
    (100, 13, 7900000,  8000000, 24),   -- cudaLaunchKernel (after gap)
    -- Sync API calls (anti-pattern: Excessive Synchronization)
    (100, 100, 5000000, 15000000, 20),  -- cudaDeviceSynchronize, 10ms
    (100, 101, 16000000, 22000000, 20), -- cudaDeviceSynchronize, 6ms
    -- cudaStreamSynchronize right after NCCL (sync-after-NCCL pattern)
    (100, 104, 5600000, 5800000, 25),   -- cudaStreamSynchronize, 0.2ms
    -- Sync memcpy (anti-pattern: Synchronous Memcpy)
    (100, 102, 23000000, 23500000, 21), -- cudaMemcpy (sync), 0.5ms
    -- cudaDeviceSynchronize during GPU idle gap (5.5ms..8ms)
    (100, 105, 5800000, 7800000, 20),   -- cudaDeviceSynchronize during gap, 2ms
    -- Sync memset (anti-pattern: Synchronous Memset)
    (100, 103, 24000000, 24200000, 23); -- cudaMemset (sync), 0.2ms

-- Sync memcpy correlated data (correlationId=102 matches cudaMemcpy runtime entry)
-- srcKind=1 = Pageable memory (see CUPTI schema docs)
-- Also add H2D memcpy at multiple time points for distribution testing
INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES
    (100, 0, 7, 102, 1, 1048576, 1, 2, 23100000, 23400000),
    -- H2D transfers at second 0 (init-heavy: 4MB)
    (100, 0, 7, 200, 1, 2097152, 7, 2, 100000, 200000),
    (100, 0, 7, 201, 1, 2097152, 7, 2, 300000, 400000),
    -- H2D transfer at second 2 (small: 0.1MB)
    (100, 0, 7, 202, 1, 104858, 7, 2, 2100000, 2200000),
    -- H2D transfer at second 5 (small: 0.1MB)
    (100, 0, 7, 203, 1, 104858, 7, 2, 5100000, 5200000);

-- Sync memset correlated data
INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES
    (100, 0, 7, 103, 4096, 0, 24050000, 24150000);

INSERT INTO NVTX_EVENTS (globalTid, start, end, text, eventType, rangeId) VALUES
    (100, 500000,  4500000, 'train_step', 59, 0),
    (100, 900000,  2100000, 'forward',    59, 1);
"""


@pytest.fixture
def minimal_nsys_conn():
    """Return a SQLite connection pre-populated with minimal Nsight tables.

    The connection uses :memory: and is closed automatically after the test.
    Use this fixture in unit tests that need a sqlite3.Connection object.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(_NSYS_SCHEMA_SQL)
    conn.executescript(_NSYS_SEED_SQL)
    yield conn
    conn.close()


@pytest.fixture
def minimal_nsys_db_path(tmp_path):
    """Write the minimal Nsight schema to a temporary .sqlite file and return its path.

    Use this fixture when a test requires an actual file path (e.g. for
    profile.open() or CLI subprocess calls).
    """
    db_path = tmp_path / "test_profile.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_NSYS_SCHEMA_SQL)
    conn.executescript(_NSYS_SEED_SQL)
    conn.close()
    return str(db_path)


# ---------------------------------------------------------------------------
# DuckDB in-memory fixture — exercises the DuckDB query code path.
# ---------------------------------------------------------------------------

# DuckDB uses "end" (double-quoted) instead of SQLite's `end` (unquoted).
# The schema below mirrors _NSYS_SCHEMA_SQL but with DuckDB syntax.
_DUCKDB_SCHEMA_SQL = """
CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value VARCHAR);
CREATE TABLE TARGET_INFO_GPU (
    id INTEGER PRIMARY KEY, name VARCHAR, busLocation VARCHAR DEFAULT '',
    totalMemory BIGINT DEFAULT 0, smCount INTEGER DEFAULT 0,
    chipName VARCHAR DEFAULT '', memoryBandwidth BIGINT DEFAULT 0
);
CREATE TABLE TARGET_INFO_CUDA_DEVICE (
    gpuId INTEGER, cudaId INTEGER, pid INTEGER DEFAULT 0,
    uuid VARCHAR DEFAULT '', numMultiprocessors INTEGER DEFAULT 0
);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
    globalPid INTEGER DEFAULT 0, deviceId INTEGER DEFAULT 0,
    streamId INTEGER DEFAULT 0, correlationId INTEGER DEFAULT 0,
    start BIGINT NOT NULL, "end" BIGINT NOT NULL,
    shortName INTEGER NOT NULL, demangledName INTEGER DEFAULT 0,
    gridX INTEGER DEFAULT 1, gridY INTEGER DEFAULT 1, gridZ INTEGER DEFAULT 1,
    blockX INTEGER DEFAULT 1, blockY INTEGER DEFAULT 1, blockZ INTEGER DEFAULT 1
);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
    globalPid INTEGER DEFAULT 0, deviceId INTEGER DEFAULT 0,
    streamId INTEGER DEFAULT 0, correlationId INTEGER DEFAULT 0,
    copyKind INTEGER DEFAULT 0, bytes BIGINT DEFAULT 0,
    srcKind INTEGER DEFAULT 0, dstKind INTEGER DEFAULT 0,
    start BIGINT NOT NULL, "end" BIGINT NOT NULL
);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (
    globalPid INTEGER DEFAULT 0, deviceId INTEGER DEFAULT 0,
    streamId INTEGER DEFAULT 0, correlationId INTEGER DEFAULT 0,
    bytes BIGINT DEFAULT 0, value INTEGER DEFAULT 0,
    start BIGINT NOT NULL, "end" BIGINT NOT NULL
);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
    globalTid INTEGER DEFAULT 0, correlationId INTEGER DEFAULT 0,
    start BIGINT NOT NULL, "end" BIGINT NOT NULL,
    nameId INTEGER DEFAULT 0
);
CREATE TABLE NVTX_EVENTS (
    globalTid INTEGER DEFAULT 0, start BIGINT NOT NULL, "end" BIGINT DEFAULT -1,
    text VARCHAR DEFAULT '', eventType INTEGER DEFAULT 59,
    rangeId INTEGER DEFAULT 0, textId INTEGER DEFAULT NULL
);
CREATE TABLE ThreadNames (
    globalTid INTEGER, nameId INTEGER, priority INTEGER DEFAULT 0
);
"""


@pytest.fixture
def duckdb_conn():
    """Return a DuckDB connection pre-populated with the minimal Nsight schema.

    Uses the same seed data as the SQLite fixtures. The DuckDB path is
    exercised by the ``_duckdb_query()`` method when ``Profile.db`` is set.
    """
    import duckdb  # noqa: F811

    db = duckdb.connect()
    # Create schema
    for stmt in _DUCKDB_SCHEMA_SQL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            db.execute(stmt)

    # Seed data — same values as _NSYS_SEED_SQL but without SQLite comments
    db.execute("""
        INSERT INTO StringIds VALUES
            (1, 'kernel_A'), (2, 'kernel_B'), (10, 'nccl_AllReduce_kernel'),
            (20, 'cudaDeviceSynchronize'), (21, 'cudaMemcpy'),
            (22, 'cudaMemcpyAsync'), (23, 'cudaMemset'),
            (24, 'cudaLaunchKernel'), (25, 'cudaStreamSynchronize'),
            (11, 'nccl_ReduceScatter_kernel')
    """)
    db.execute("""
        INSERT INTO TARGET_INFO_GPU VALUES
            (0, 'NVIDIA Test GPU', '0000:00:00.0', 8589934592, 108, 'TestChip', 0)
    """)
    db.execute("INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (0, 0, 100, '', 108)")
    db.execute("""
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
            (100, 0, 7, 1, 1000000, 2000000, 1, 1, 32, 1, 1, 256, 1, 1),
            (100, 0, 7, 2, 3000000, 4000000, 2, 2, 16, 1, 1, 128, 1, 1),
            (100, 0, 8, 10, 2500000, 3500000, 10, 10, 1, 1, 1, 512, 1, 1),
            (100, 0, 7, 12, 4500000, 5500000, 11, 11, 1, 1, 1, 256, 1, 1),
            (100, 0, 7, 13, 8000000, 9000000, 1, 1, 32, 1, 1, 256, 1, 1)
    """)
    db.execute("""
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
            (100, 1,   900000,  1000000, 24),
            (100, 2,  2900000,  3000000, 24),
            (100, 10, 2400000,  2500000, 24),
            (100, 12, 4400000,  4500000, 24),
            (100, 13, 7900000,  8000000, 24),
            (100, 100, 5000000, 15000000, 20),
            (100, 101, 16000000, 22000000, 20),
            (100, 104, 5600000, 5800000, 25),
            (100, 102, 23000000, 23500000, 21),
            (100, 105, 5800000, 7800000, 20),
            (100, 103, 24000000, 24200000, 23)
    """)
    db.execute("""
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES
            (100, 0, 7, 102, 1, 1048576, 1, 2, 23100000, 23400000),
            (100, 0, 7, 200, 1, 2097152, 7, 2, 100000, 200000),
            (100, 0, 7, 201, 1, 2097152, 7, 2, 300000, 400000),
            (100, 0, 7, 202, 1, 104858, 7, 2, 2100000, 2200000),
            (100, 0, 7, 203, 1, 104858, 7, 2, 5100000, 5200000)
    """)
    db.execute("""
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES
            (100, 0, 7, 103, 4096, 0, 24050000, 24150000)
    """)
    db.execute("""
        INSERT INTO NVTX_EVENTS (globalTid, start, "end", text, eventType, rangeId) VALUES
            (100, 500000,  4500000, 'train_step', 59, 0),
            (100, 900000,  2100000, 'forward',    59, 1)
    """)
    db.execute("""
        INSERT INTO ThreadNames VALUES (100, 24, 1)
    """)
    yield db
    db.close()



# ---------------------------------------------------------------------------
# Enhanced fixture with nested NVTX hierarchy (multi-layer)
# ---------------------------------------------------------------------------

_NESTED_NVTX_SEED_SQL = """\
INSERT INTO StringIds VALUES
    (1, 'sm80_xmma_gemm_f16f16'),
    (2, 'sm80_xmma_gemm_f32f32'),
    (10, 'nccl_AllReduce_kernel'),
    (24, 'cudaLaunchKernel');

INSERT INTO TARGET_INFO_GPU VALUES
    (0, 'NVIDIA Test GPU', '0000:00:00.0', 8589934592, 108, 'TestChip', 0);

INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES
    (0, 0, 100, '', 108);

-- NVTX Hierarchy (CPU thread 100, all ns):
-- train_step       [0           - 100_000_000]   depth 0
--   forward         [1_000_000  - 40_000_000]    depth 1
--     layer_0       [2_000_000  - 18_000_000]    depth 2
--       attention   [3_000_000  - 10_000_000]    depth 3
--       mlp         [11_000_000 - 17_000_000]    depth 3
--     layer_1       [19_000_000 - 38_000_000]    depth 2
--       attention   [20_000_000 - 28_000_000]    depth 3
--       mlp         [29_000_000 - 37_000_000]    depth 3
--   backward        [42_000_000 - 90_000_000]    depth 1
--     layer_1_bwd   [43_000_000 - 65_000_000]    depth 2
--       allreduce   [50_000_000 - 60_000_000]    depth 3
--     layer_0_bwd   [66_000_000 - 88_000_000]    depth 2
--       allreduce   [75_000_000 - 85_000_000]    depth 3
INSERT INTO NVTX_EVENTS (globalTid, start, end, text, eventType, rangeId) VALUES
    (100,  0,         100000000, 'train_step', 59, 0),
    (100,  1000000,    40000000, 'forward',    59, 1),
    (100,  2000000,    18000000, 'layer_0',    59, 2),
    (100,  3000000,    10000000, 'attention',  59, 3),
    (100, 11000000,    17000000, 'mlp',        59, 4),
    (100, 19000000,    38000000, 'layer_1',    59, 5),
    (100, 20000000,    28000000, 'attention',  59, 6),
    (100, 29000000,    37000000, 'mlp',        59, 7),
    (100, 42000000,    90000000, 'backward',   59, 8),
    (100, 43000000,    65000000, 'layer_1_bwd', 59, 9),
    (100, 50000000,    60000000, 'allreduce',  59, 10),
    (100, 66000000,    88000000, 'layer_0_bwd', 59, 11),
    (100, 75000000,    85000000, 'allreduce',  59, 12);

-- Kernels (GPU device 0):
-- Stream 7 (compute): GEMM kernels inside attention/mlp NVTX
INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
    -- layer_0 attention: 2 GEMM kernels
    (100, 0, 7, 1,  4000000,  6000000, 1, 1, 32,1,1, 256,1,1),
    (100, 0, 7, 2,  7000000,  9000000, 1, 1, 32,1,1, 256,1,1),
    -- layer_0 mlp: 2 GEMM kernels
    (100, 0, 7, 3, 12000000, 14000000, 2, 2, 16,1,1, 128,1,1),
    (100, 0, 7, 4, 15000000, 16500000, 2, 2, 16,1,1, 128,1,1),
    -- layer_1 attention: 2 GEMM kernels
    (100, 0, 7, 5, 21000000, 24000000, 1, 1, 32,1,1, 256,1,1),
    (100, 0, 7, 6, 25000000, 27000000, 1, 1, 32,1,1, 256,1,1),
    -- layer_1 mlp: 2 GEMM kernels
    (100, 0, 7, 7, 30000000, 33000000, 2, 2, 16,1,1, 128,1,1),
    (100, 0, 7, 8, 34000000, 36000000, 2, 2, 16,1,1, 128,1,1);

-- Stream 8 (NCCL): AllReduce kernels inside backward allreduce NVTX
INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
    -- layer_1_bwd allreduce
    (100, 0, 8, 11, 51000000, 58000000, 10, 10, 1,1,1, 512,1,1),
    -- layer_0_bwd allreduce
    (100, 0, 8, 12, 76000000, 84000000, 10, 10, 1,1,1, 512,1,1);

-- Runtime (CUDA API) — correlate kernels to CPU thread via correlationId
INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
    -- Compute kernel launches (on thread 100)
    (100, 1,  3500000,  4000000, 24),
    (100, 2,  6500000,  7000000, 24),
    (100, 3, 11500000, 12000000, 24),
    (100, 4, 14500000, 15000000, 24),
    (100, 5, 20500000, 21000000, 24),
    (100, 6, 24500000, 25000000, 24),
    (100, 7, 29500000, 30000000, 24),
    (100, 8, 33500000, 34000000, 24),
    -- NCCL kernel launches
    (100, 11, 50500000, 51000000, 24),
    (100, 12, 75500000, 76000000, 24);
"""


@pytest.fixture
def nested_nvtx_conn():
    """Return a SQLite connection with nested NVTX hierarchy for per-layer tests.

    Simulates a training step with:
      - 4-level nested NVTX (train_step > forward/backward > layer_0/layer_1 > attention/mlp)
      - 8 compute (GEMM) kernels on stream 7
      - 2 NCCL (AllReduce) kernels on stream 8
      - Proper temporal containment for attribution testing
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(_NSYS_SCHEMA_SQL)
    conn.executescript(_NESTED_NVTX_SEED_SQL)
    yield conn
    conn.close()

