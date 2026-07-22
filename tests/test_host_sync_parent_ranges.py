"""Tests for the host_sync_parent_ranges builtin skill."""

import sqlite3

import pytest

from nsys_ai.skills.builtins import host_sync_parent_ranges as skill_mod


def _build_conn(rows: list[tuple], *, default_tid: int = 1) -> sqlite3.Connection:
    """Build a minimal sqlite3 connection with NVTX_EVENTS seeded.

    ``rows`` accepts either 3-tuples ``(label, start, end_ns)`` or 4-tuples
    ``(label, start, end_ns, globalTid)``. 3-tuples fall back to
    ``default_tid`` so parent/child ancestry joins cleanly.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (
            globalTid INTEGER DEFAULT 0,
            start     INTEGER NOT NULL,
            end       INTEGER NOT NULL,
            text      TEXT,
            textId    INTEGER,
            eventType INTEGER DEFAULT 59
        );
        """
    )
    for row in rows:
        if len(row) == 3:
            label, start, end_ns = row
            tid = default_tid
        else:
            label, start, end_ns, tid = row
        conn.execute(
            "INSERT INTO NVTX_EVENTS (globalTid, start, end, text) VALUES (?, ?, ?, ?)",
            (tid, start, end_ns, label),
        )
    conn.commit()
    return conn


class TestHostSyncParentRanges:
    def test_ranks_parents_by_sync_ns(self):
        # Two disjoint training phases; each contains exactly one sync child.
        # Durations chosen to round cleanly at 3 decimals in ms.
        conn = _build_conn(
            [
                # forward_pass contains aten::item — 300 µs = 0.3 ms
                ("forward_pass", 0, 10_000_000),
                ("aten::item", 1_000_000, 1_300_000),
                # optimizer contains aten::_local_scalar_dense — 500 µs = 0.5 ms
                ("optimizer", 20_000_000, 30_000_000),
                ("aten::_local_scalar_dense", 25_000_000, 25_500_000),
                # unrelated wallclock-only range — contains no sync events
                ("misc", 40_000_000, 50_000_000),
            ]
        )
        out = skill_mod._execute(conn)
        parents = {r["parent_range"] for r in out}
        assert {"forward_pass", "optimizer"} <= parents
        assert "misc" not in parents
        # Ordering: optimizer (500 µs) > forward_pass (300 µs)
        assert out[0]["parent_range"] == "optimizer"
        assert out[0]["sync_ns"] == 500_000
        assert out[0]["sync_ms"] == pytest.approx(0.5)
        assert out[1]["parent_range"] == "forward_pass"
        assert out[1]["sync_ns"] == 300_000
        assert out[1]["sync_ms"] == pytest.approx(0.3)

    def test_nested_parents_both_attributed(self):
        # When parents nest, each enclosing range gets independent credit.
        # Mirrors the real-world case where `aten::item` (outer pytorch wrapper)
        # contains `aten::_local_scalar_dense` (inner sync).
        conn = _build_conn(
            [
                ("train_step", 0, 10_000),  # outermost
                ("aten::item", 100, 600),   # parent-of-child AND is itself a child
                ("aten::_local_scalar_dense", 200, 500),  # inner child
            ]
        )
        out = skill_mod._execute(conn)
        parents = {r["parent_range"] for r in out}
        # Both train_step (contains item + _local_scalar_dense) AND aten::item
        # (contains _local_scalar_dense) should appear as parents.
        assert "train_step" in parents
        assert "aten::item" in parents

    def test_matches_via_textid_indirection(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS (
                globalTid INTEGER DEFAULT 0,
                start     INTEGER NOT NULL,
                end       INTEGER NOT NULL,
                text      TEXT,
                textId    INTEGER,
                eventType INTEGER DEFAULT 59
            );
            """
        )
        # Labels live in StringIds; NVTX_EVENTS.text is NULL — exercises the COALESCE path.
        conn.executescript(
            """
            INSERT INTO StringIds VALUES (1, 'train_step'), (2, 'aten::item');
            INSERT INTO NVTX_EVENTS (start, end, text, textId) VALUES
                (0, 10000, NULL, 1),
                (200, 600, NULL, 2);
            """
        )
        conn.commit()
        out = skill_mod._execute(conn)
        assert out and "error" not in out[0]
        assert out[0]["parent_range"] == "train_step"
        assert out[0]["n_syncs"] == 1
        assert out[0]["sync_ns"] == 400

    def test_empty_when_no_sync_events(self):
        conn = _build_conn(
            [
                ("train_step", 0, 10_000),
                ("forward_pass", 100, 500),
            ]
        )
        assert skill_mod._execute(conn) == []

    def test_custom_patterns_override_default(self):
        conn = _build_conn(
            [
                ("train_step", 0, 10_000),
                ("my_custom_sync_marker", 100, 600),
            ]
        )
        # Default patterns should not match this label.
        assert skill_mod._execute(conn) == []
        # But a user-supplied pattern should.
        out = skill_mod._execute(conn, patterns="custom_sync")
        assert len(out) == 1
        assert out[0]["parent_range"] == "train_step"
        assert out[0]["n_syncs"] == 1

    def test_cross_thread_parents_excluded(self):
        # DataLoader NVTX range (tid=99) wraps the same wall-clock window as
        # the main-thread sync, but is NOT on the same thread — must not be
        # attributed as a parent of the main-thread sync.
        conn = _build_conn(
            [
                ("dataloader_fetch",   0, 10_000_000,  99),
                ("train_step",         0, 10_000_000,   1),
                ("aten::item", 1_000_000,  1_500_000,   1),
            ]
        )
        out = skill_mod._execute(conn)
        parents = {r["parent_range"] for r in out}
        assert "train_step" in parents
        assert "dataloader_fetch" not in parents, (
            "cross-thread NVTX range must not be matched as parent"
        )

    def test_missing_nvtx_table_returns_error_row(self):
        conn = sqlite3.connect(":memory:")
        # Intentionally no NVTX_EVENTS table.
        out = skill_mod._execute(conn)
        assert len(out) == 1
        assert "error" in out[0]
        assert "NVTX" in out[0]["error"]

    def test_limit_caps_results(self):
        # 6 distinct parents, each with one matching child — ask for 2.
        rows = []
        for i in range(6):
            base = i * 1_000_000
            rows.append((f"train_step_{i}", base, base + 100_000))
            # child sync duration grows with i so ordering is deterministic
            rows.append(("aten::item", base + 100, base + 100 + (i + 1) * 100))
        conn = _build_conn(rows)
        out = skill_mod._execute(conn, limit=2)
        assert len(out) == 2
        # Largest sync_ns first → parent of i=5 (600ns), then i=4 (500ns)
        assert out[0]["parent_range"] == "train_step_5"
        assert out[1]["parent_range"] == "train_step_4"

    def test_format_renders_table(self):
        rows = [
            {
                "parent_range": "train_step",
                "n_syncs": 3,
                "sync_ns": 500_000,
                "top_child_label": "aten::item",
                "sync_ms": 0.5,
            }
        ]
        out = skill_mod._format(rows)
        assert "train_step" in out
        assert "aten::item" in out
        assert "0.500" in out

    def test_format_empty(self):
        assert "No host-sync events" in skill_mod._format([])

    def test_format_error(self):
        assert skill_mod._format([{"error": "boom"}]) == "Error: boom"

    def test_top_child_label_picks_max_sync_time_not_alphabetic(self):
        # Parent contains two sync children:
        #   aten::item                 — 100 µs single call
        #   cudaStreamSynchronize_foo  — 1 ms single call (10x larger)
        # Alphabetic MIN would pick `aten::item` (wrong); correct answer is
        # `cudaStreamSynchronize_foo` (larger aggregated sync_ns).
        conn = _build_conn(
            [
                ("train_step",               0,  10_000_000),
                ("aten::item",       1_000_000,   1_100_000),
                ("cudaStreamSynchronize_foo", 2_000_000, 3_000_000),
            ]
        )
        out = skill_mod._execute(conn)
        assert len(out) == 1
        assert out[0]["parent_range"] == "train_step"
        assert out[0]["top_child_label"] == "cudaStreamSynchronize_foo"

    def test_same_labeled_nested_ranges_still_attributed(self):
        # Nested same-label ranges — outer `train_step` contains inner
        # `train_step` (iteration inside iteration), and the inner range
        # contains the real sync. Outer should get credit; the previous
        # `parent.label != child.label` filter would have dropped this.
        conn = _build_conn(
            [
                ("train_step",     0,  10_000_000),  # outer
                ("train_step", 2_000_000,   8_000_000),  # inner — same label
                ("aten::item", 4_000_000,   4_500_000),
            ]
        )
        out = skill_mod._execute(conn)
        # Inner `train_step` and outer `train_step` share a parent_range in
        # the GROUP BY, so we get ONE row labeled `train_step` with 2 syncs
        # (outer contains aten::item AND outer contains inner train_step…
        # but inner train_step doesn't match the sync pattern, so it isn't
        # counted as a child — still 1 sync attributed but the outer range
        # is no longer filtered out).
        assert len(out) == 1
        assert out[0]["parent_range"] == "train_step"
        assert out[0]["top_child_label"] == "aten::item"

    def test_invalid_limit_returns_error(self):
        conn = _build_conn([("train_step", 0, 10_000_000)])
        # Zero
        out = skill_mod._execute(conn, limit=0)
        assert len(out) == 1 and "error" in out[0]
        # Negative
        out = skill_mod._execute(conn, limit=-5)
        assert len(out) == 1 and "error" in out[0]
        # Non-numeric
        out = skill_mod._execute(conn, limit="abc")
        assert len(out) == 1 and "error" in out[0]

    def test_limit_capped_at_max(self):
        # Above _MAX_LIMIT should silently clamp, not error.
        conn = _build_conn([("train_step", 0, 10_000_000)])
        out = skill_mod._execute(conn, limit=10_000)
        # No error row; just runs normally (empty since no syncs).
        assert out == []

    def test_case_insensitive_pattern_matching(self):
        # User supplies uppercase/mixed-case pattern; should still match
        # lowercase labels in the profile (and vice versa). DuckDB's LIKE is
        # case-sensitive by default, so this test guards the engine parity.
        conn = _build_conn(
            [
                ("train_step",   0, 10_000_000),
                ("aten::ITEM_UPPERCASE", 1_000_000, 1_500_000),
            ]
        )
        out = skill_mod._execute(conn, patterns="item")  # lowercase pattern
        assert len(out) == 1
        assert out[0]["parent_range"] == "train_step"
        # And vice versa — uppercase pattern matches lowercase label.
        conn2 = _build_conn(
            [
                ("train_step",  0, 10_000_000),
                ("aten::item",  1_000_000, 1_500_000),
            ]
        )
        out2 = skill_mod._execute(conn2, patterns="ITEM")
        assert len(out2) == 1
        assert out2[0]["parent_range"] == "train_step"

    def test_non_range_event_types_excluded(self):
        # Marker-type NVTX events (eventType not in (59, 60)) must not be
        # treated as ranges — they have zero duration semantics and would
        # pollute the ancestry join.
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS (
                globalTid INTEGER DEFAULT 0,
                start     INTEGER NOT NULL,
                end       INTEGER NOT NULL,
                text      TEXT,
                textId    INTEGER,
                eventType INTEGER DEFAULT 59
            );
            """
        )
        # A marker (eventType=75, some non-range code) containing an item
        # call — must be skipped, leaving `train_step` (eventType=59) as
        # the sole parent.
        conn.executescript(
            """
            INSERT INTO NVTX_EVENTS (globalTid, start, end, text, eventType) VALUES
                (1,        0, 10000000, 'train_step',         59),
                (1,        0, 10000000, 'fake_marker_range',  75),
                (1, 1000000,  1500000,  'aten::item',         59);
            """
        )
        conn.commit()
        out = skill_mod._execute(conn)
        parents = {r["parent_range"] for r in out}
        assert "train_step" in parents
        assert "fake_marker_range" not in parents

    def test_registered_in_builtin_registry(self):
        from nsys_ai.skills import registry

        s = registry.get_skill("host_sync_parent_ranges")
        assert s is not None
        assert s.name == "host_sync_parent_ranges"
        assert {p.name for p in s.params} == {"limit", "patterns"}


# Rows shaped (label, start, end, tid, eventType). One dataset exercising the
# cases the sweep must reproduce byte-for-byte against the SQL: multi-ancestor
# nesting, crossing StartEnd(60) ranges (where a stack-pop would diverge),
# an identical-coordinate twin (self-exclusion by coords), tid isolation, and
# a parent-order tie (the deterministic secondary key).
_PARITY_ROWS = [
    # tid 1: three-deep nesting — item credits train_step; _local_scalar_dense
    # credits train_step AND aten::item.
    ("train_step", 0, 10_000_000, 1, 59),
    ("aten::item", 1_000_000, 2_000_000, 1, 59),
    ("aten::_local_scalar_dense", 1_400_000, 1_600_000, 1, 59),
    # tid 2: two crossing StartEnd ranges. childA is inside both; childB starts
    # after rangeA has closed, so it credits rangeB only — the case a stack gets
    # wrong.
    ("rangeA", 0, 10_000_000, 2, 60),
    ("rangeB", 5_000_000, 15_000_000, 2, 60),
    ("cudaStreamSynchronize_a", 6_000_000, 7_000_000, 2, 60),
    ("cudaStreamSynchronize_b", 11_000_000, 12_000_000, 2, 60),
    # tid 3: identical-coordinate twin. Two distinct ranges share [0, 5_000_000];
    # a child at exactly [0, 5_000_000] is excluded from both (coord self-match),
    # while a strictly-inner child credits both twins.
    ("twin_one", 0, 5_000_000, 3, 59),
    ("twin_two", 0, 5_000_000, 3, 59),
    ("aten::item_exact", 0, 5_000_000, 3, 59),
    ("aten::item_inner", 1_000_000, 2_000_000, 3, 59),
    # tid 4: two parents with identical total sync_ns — pins the sync_ns-tie
    # ordering (deterministic parent_range ASC secondary).
    ("zeta_phase", 0, 10_000_000, 4, 59),
    ("aten::item_z", 1_000_000, 1_500_000, 4, 59),
    ("alpha_phase", 20_000_000, 30_000_000, 4, 59),
    ("aten::item_a", 21_000_000, 21_500_000, 4, 59),
]


def _build_duckdb(rows):
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute(
        'CREATE TABLE StringIds (id BIGINT, value VARCHAR); '
        'CREATE TABLE NVTX_EVENTS (globalTid BIGINT, start BIGINT, "end" BIGINT, '
        'text VARCHAR, "textId" BIGINT, "eventType" INTEGER);'
    )
    for label, start, end_ns, tid, etype in rows:
        con.execute(
            'INSERT INTO NVTX_EVENTS (globalTid, start, "end", text, "eventType") '
            "VALUES (?, ?, ?, ?, ?)",
            [tid, start, end_ns, label, etype],
        )
    return con


def _build_sqlite(rows):
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (
            globalTid INTEGER, start INTEGER, end INTEGER,
            text TEXT, textId INTEGER, eventType INTEGER
        );
        """
    )
    conn.executemany(
        "INSERT INTO NVTX_EVENTS (globalTid, start, end, text, eventType) "
        "VALUES (?, ?, ?, ?, ?)",
        [(tid, s, e, label, et) for (label, s, e, tid, et) in rows],
    )
    conn.commit()
    return conn


# The original containment CTE, kept here as an independent oracle. Production
# replaced it with the Python sweep (it hangs on SQLite and crashes DuckDB's
# sqlite_scanner on real profiles), but on a *native* DuckDB connection with
# small data it runs fine and defines the exact semantics the sweep must match.
_ORACLE_SQL = """
    WITH resolved AS (
        SELECT COALESCE(n.text, s.value) AS label, n.globalTid AS tid,
               n.start AS start, n.[end] AS end_ns
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
        WHERE n.[end] > n.start AND n.eventType IN (59, 60) {trim}
    ),
    sync_children AS (
        SELECT label, tid, start, end_ns FROM resolved
        WHERE ({like}) AND label IS NOT NULL
    ),
    matched AS (
        SELECT parent.label AS parent_range, child.label AS child_label,
               child.end_ns - child.start AS sync_ns
        FROM sync_children child JOIN resolved parent
          ON parent.tid = child.tid
         AND parent.start <= child.start AND parent.end_ns >= child.end_ns
         AND (parent.start != child.start OR parent.end_ns != child.end_ns)
        WHERE parent.label IS NOT NULL
    ),
    parent_totals AS (
        SELECT parent_range, COUNT(*) AS n_syncs, SUM(sync_ns) AS sync_ns
        FROM matched GROUP BY parent_range
    ),
    child_totals AS (
        SELECT parent_range, child_label, COUNT(*) AS child_n_syncs,
               SUM(sync_ns) AS child_sync_ns
        FROM matched GROUP BY parent_range, child_label
    ),
    ranked_children AS (
        SELECT parent_range, child_label, ROW_NUMBER() OVER (
            PARTITION BY parent_range
            ORDER BY child_sync_ns DESC, child_n_syncs DESC, child_label ASC
        ) AS rn FROM child_totals
    )
    SELECT pt.parent_range, pt.n_syncs, pt.sync_ns, rc.child_label AS top_child_label
    FROM parent_totals pt LEFT JOIN ranked_children rc
      ON rc.parent_range = pt.parent_range AND rc.rn = 1
    ORDER BY pt.sync_ns DESC, pt.parent_range ASC
    LIMIT {limit}
"""


def _oracle_rows(duckdb_conn, patterns, limit):
    from nsys_ai.connection import wrap_connection

    like_parts, like_params = [], []
    for p in patterns:
        like_parts.append("LOWER(label) LIKE ?")
        like_params.append(f"%{p.lower()}%")
    sql = _ORACLE_SQL.format(trim="", like=" OR ".join(like_parts), limit=int(limit))
    rows = wrap_connection(duckdb_conn).execute(sql, like_params).fetchall()
    out = []
    for parent_range, n_syncs, sync_ns, top_child_label in rows:
        sync_ns = int(sync_ns or 0)
        out.append(
            {
                "parent_range": parent_range,
                "n_syncs": n_syncs,
                "sync_ns": sync_ns,
                "sync_ms": round(sync_ns / 1e6, 3),
                "top_child_label": top_child_label,
            }
        )
    return out


def test_sweep_matches_the_sql_oracle_exactly():
    """The sweep must reproduce the original containment SQL byte-for-byte.

    Same synthetic data through the Python sweep and the reference CTE (run on
    a native DuckDB connection where it still works). If the sweep ever diverges
    on containment, crossing ranges, twin exclusion, tid isolation, top-child
    choice, or ordering, this fails.
    """
    patterns = ["item", "_local_scalar_dense", "cudastreamsynchronize"]
    sweep_out = skill_mod._execute(_build_sqlite(_PARITY_ROWS), limit=1000)
    oracle_out = _oracle_rows(_build_duckdb(_PARITY_ROWS), patterns, 1000)

    assert "error" not in (sweep_out[0] if sweep_out else {})
    assert sweep_out == oracle_out, "sweep diverged from the SQL oracle"
    # The crossing case specifically (a stack-pop would get this wrong): rangeA
    # must NOT be credited with the child that starts after it closed.
    a_rows = [r for r in sweep_out if r["parent_range"] == "rangeA"]
    assert a_rows and a_rows[0]["n_syncs"] == 1  # only cudaStreamSynchronize_a


def test_sweep_matches_across_engines():
    """The sweep is engine-agnostic: identical output on SQLite and DuckDB."""
    duck_out = skill_mod._execute(_build_duckdb(_PARITY_ROWS), limit=1000)
    lite_out = skill_mod._execute(_build_sqlite(_PARITY_ROWS), limit=1000)
    assert duck_out == lite_out


def test_sweep_matches_oracle_with_trim():
    """Parity with the oracle must also hold under a trim window."""
    patterns = ["item", "_local_scalar_dense", "cudastreamsynchronize"]
    trim = {"trim_start_ns": 0, "trim_end_ns": 12_000_000}
    sweep_out = skill_mod._execute(_build_sqlite(_PARITY_ROWS), limit=1000, **trim)

    from nsys_ai.connection import wrap_connection

    like_parts = ["LOWER(label) LIKE ?" for _ in patterns]
    like_params = [f"%{p.lower()}%" for p in patterns]
    sql = _ORACLE_SQL.format(
        trim="AND n.start >= 0 AND n.[end] <= 12000000",
        like=" OR ".join(like_parts),
        limit=1000,
    )
    rows = wrap_connection(_build_duckdb(_PARITY_ROWS)).execute(sql, like_params).fetchall()
    oracle_out = [
        {
            "parent_range": pr,
            "n_syncs": n,
            "sync_ns": int(s or 0),
            "sync_ms": round(int(s or 0) / 1e6, 3),
            "top_child_label": tc,
        }
        for pr, n, s, tc in rows
    ]
    assert sweep_out == oracle_out
