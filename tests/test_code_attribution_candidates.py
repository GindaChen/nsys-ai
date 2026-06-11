import sqlite3


def test_code_attribution_candidates_ranks_containing_nvtx_path(nested_nvtx_conn):
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("code_attribution_candidates")
    rows = skill.execute(
        nested_nvtx_conn,
        start_ns=50_500_000,
        end_ns=58_500_000,
        device=0,
        limit=3,
    )

    assert rows
    top = rows[0]
    assert top["candidate"] == "train_step > backward > layer_1_bwd > allreduce"
    assert top["candidate_type"] == "nvtx_path"
    assert top["nvtx_text"] == "allreduce"
    assert top["confidence"] >= 0.9
    assert top["evidence"]["overlap_pct"] == 100.0
    assert top["evidence"]["contains_selection"] is True
    assert top["evidence"]["kernel_count"] == 1
    assert top["evidence"]["nccl_kernel_count"] == 1
    assert top["evidence"]["compute_kernel_count"] == 0
    assert top["evidence"]["top_kernels"][0]["kernel_name"] == "nccl_AllReduce_kernel"
    assert "temporal context" in top["limitations"][0]


def test_code_attribution_candidates_includes_parent_context(nested_nvtx_conn):
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("code_attribution_candidates")
    rows = skill.execute(
        nested_nvtx_conn,
        start_ns=21_000_000,
        end_ns=27_000_000,
        device=0,
        limit=5,
    )

    candidates = [r["candidate"] for r in rows]
    assert "train_step > forward > layer_1 > attention" in candidates
    assert "train_step > forward > layer_1" in candidates
    leaf = next(r for r in rows if r["candidate"].endswith("> attention"))
    assert leaf["evidence"]["compute_kernel_count"] == 2
    assert leaf["evidence"]["total_kernel_ms"] == 5.0


def test_code_attribution_candidates_format_mentions_limitations(nested_nvtx_conn):
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("code_attribution_candidates")
    rows = skill.execute(
        nested_nvtx_conn,
        start_ns=50_500_000,
        end_ns=58_500_000,
        device=0,
        limit=1,
    )

    text = skill.format_rows(rows)
    assert "Code Attribution Candidates" in text
    assert "train_step > backward > layer_1_bwd > allreduce" in text
    assert "Confidence:" in text
    assert "Limitations:" in text
    assert "not exact source-line attribution" in text


def test_code_attribution_candidates_reports_missing_nvtx(minimal_nsys_conn):
    from nsys_ai.skills.registry import get_skill

    minimal_nsys_conn.execute("DELETE FROM NVTX_EVENTS")
    skill = get_skill("code_attribution_candidates")

    rows = skill.execute(
        minimal_nsys_conn,
        start_ns=1_000_000,
        end_ns=2_000_000,
        device=0,
    )

    assert rows[0]["error"] == "No overlapping NVTX ranges found for the selected window"
    assert rows[0]["selection"]["start_ns"] == 1_000_000
    assert "limitations" in rows[0]


def test_code_attribution_candidates_validates_window(nested_nvtx_conn):
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("code_attribution_candidates")
    rows = skill.execute(nested_nvtx_conn, start_ns=10, end_ns=10)

    assert rows == [{"error": "end_ns must be greater than start_ns"}]


def test_code_attribution_candidates_validates_limit(nested_nvtx_conn):
    from nsys_ai.skills.registry import get_skill

    skill = get_skill("code_attribution_candidates")
    assert skill.execute(nested_nvtx_conn, start_ns=10, end_ns=20, limit=0) == [
        {"error": "limit must be >= 1"}
    ]
    assert skill.execute(nested_nvtx_conn, start_ns=10, end_ns=20, limit=-1) == [
        {"error": "limit must be >= 1"}
    ]


def test_code_attribution_candidates_does_not_cross_promote_duplicate_paths():
    from nsys_ai.skills.registry import get_skill

    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE TARGET_INFO_GPU (
                id INTEGER PRIMARY KEY,
                name TEXT,
                busLocation TEXT DEFAULT '',
                totalMemory INTEGER DEFAULT 0,
                smCount INTEGER DEFAULT 0,
                chipName TEXT DEFAULT '',
                memoryBandwidth INTEGER DEFAULT 0
            );
            CREATE TABLE TARGET_INFO_CUDA_DEVICE (
                gpuId INTEGER,
                cudaId INTEGER,
                pid INTEGER DEFAULT 0,
                uuid TEXT DEFAULT '',
                numMultiprocessors INTEGER DEFAULT 0
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                globalPid INTEGER DEFAULT 0,
                deviceId INTEGER DEFAULT 0,
                streamId INTEGER DEFAULT 0,
                correlationId INTEGER DEFAULT 0,
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                shortName INTEGER NOT NULL,
                demangledName INTEGER DEFAULT 0
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                globalTid INTEGER DEFAULT 0,
                correlationId INTEGER DEFAULT 0,
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                nameId INTEGER DEFAULT 0
            );
            CREATE TABLE NVTX_EVENTS (
                globalTid INTEGER DEFAULT 0,
                start INTEGER NOT NULL,
                end INTEGER DEFAULT -1,
                text TEXT DEFAULT NULL,
                eventType INTEGER DEFAULT 59,
                rangeId INTEGER DEFAULT 0,
                textId INTEGER DEFAULT NULL
            );
            INSERT INTO StringIds VALUES
                (1, 'kernel_x'),
                (2, 'cudaLaunchKernel');
            INSERT INTO TARGET_INFO_GPU VALUES
                (0, 'GPU', '', 0, 0, '', 0);
            INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES
                (0, 0, 1, '', 0);
            INSERT INTO NVTX_EVENTS VALUES
                (100, 0, 900, 'shared_region', 59, 0, NULL),
                (200, 0, 1000, 'shared_region', 59, 1, NULL);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                (1, 0, 1, 10, 300, 700, 1, 1);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
                (100, 10, 250, 300, 2);
            """
        )

        skill = get_skill("code_attribution_candidates")
        rows = skill.execute(conn, start_ns=0, end_ns=1000, device=0, limit=2)

        assert rows[0]["candidate"] == "shared_region"
        assert rows[0]["evidence"]["global_tid"] == 100
        assert rows[0]["evidence"]["kernel_count"] == 1
        assert rows[1]["candidate"] == "shared_region"
        assert rows[1]["evidence"]["global_tid"] == 200
        assert rows[1]["evidence"]["kernel_count"] == 0
    finally:
        conn.close()


def test_code_attribution_candidates_handles_textid_nvtx():
    from nsys_ai.skills.registry import get_skill

    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE TARGET_INFO_GPU (
                id INTEGER PRIMARY KEY,
                name TEXT,
                busLocation TEXT DEFAULT '',
                totalMemory INTEGER DEFAULT 0,
                smCount INTEGER DEFAULT 0,
                chipName TEXT DEFAULT '',
                memoryBandwidth INTEGER DEFAULT 0
            );
            CREATE TABLE TARGET_INFO_CUDA_DEVICE (
                gpuId INTEGER,
                cudaId INTEGER,
                pid INTEGER DEFAULT 0,
                uuid TEXT DEFAULT '',
                numMultiprocessors INTEGER DEFAULT 0
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                globalPid INTEGER DEFAULT 0,
                deviceId INTEGER DEFAULT 0,
                streamId INTEGER DEFAULT 0,
                correlationId INTEGER DEFAULT 0,
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                shortName INTEGER NOT NULL,
                demangledName INTEGER DEFAULT 0
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                globalTid INTEGER DEFAULT 0,
                correlationId INTEGER DEFAULT 0,
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                nameId INTEGER DEFAULT 0
            );
            CREATE TABLE NVTX_EVENTS (
                globalTid INTEGER DEFAULT 0,
                start INTEGER NOT NULL,
                end INTEGER DEFAULT -1,
                text TEXT DEFAULT NULL,
                eventType INTEGER DEFAULT 59,
                rangeId INTEGER DEFAULT 0,
                textId INTEGER DEFAULT NULL
            );
            INSERT INTO StringIds VALUES
                (1, 'source_region'),
                (2, 'kernel_x'),
                (3, 'cudaLaunchKernel');
            INSERT INTO TARGET_INFO_GPU VALUES
                (0, 'GPU', '', 0, 0, '', 0);
            INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES
                (0, 0, 1, '', 0);
            INSERT INTO NVTX_EVENTS VALUES
                (7, 100, 900, NULL, 59, 0, 1);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                (1, 0, 1, 10, 300, 700, 2, 2);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
                (7, 10, 250, 300, 3);
            """
        )

        skill = get_skill("code_attribution_candidates")
        rows = skill.execute(conn, start_ns=200, end_ns=800, device=0)

        assert rows[0]["candidate"] == "source_region"
        assert rows[0]["evidence"]["kernel_count"] == 1
    finally:
        conn.close()
