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

    def test_registered_in_builtin_registry(self):
        from nsys_ai.skills import registry

        s = registry.get_skill("host_sync_parent_ranges")
        assert s is not None
        assert s.name == "host_sync_parent_ranges"
        assert {p.name for p in s.params} == {"limit", "patterns"}
