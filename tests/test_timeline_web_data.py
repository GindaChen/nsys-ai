import json
import sqlite3
from pathlib import Path

from nsys_ai.profile import Profile
from nsys_ai.viewer import (
    build_timeline_gpu_data,
    build_timeline_gpu_data_lod,
    generate_timeline_data_json,
    generate_timeline_html,
    write_timeline_html,
)
from nsys_ai.web import (
    _aggregate_timeline_gpu_entry,
    _build_timeline_overview,
    _slice_nvtx_spans,
)


def test_timeline_web_lod_uses_interval_union_and_marks_aggregates():
    entry = {
        "id": 0,
        "kernels": [
            {"type": "kernel", "name": "long", "start_ns": 0, "end_ns": 60, "stream": 1},
            {"type": "kernel", "name": "overlap", "start_ns": 40, "end_ns": 100, "stream": 2},
            {"type": "kernel", "name": "short", "start_ns": 20, "end_ns": 30, "stream": 1},
        ],
        "nvtx_spans": [{"name": "phase", "start": 0, "end": 100}],
    }

    result = _aggregate_timeline_gpu_entry(entry, 0, 100, 2)

    assert result["lod"] == {
        "mode": "aggregate",
        "record_count": 3,
        "bucket_count": 2,
        "max_buckets": 2,
    }
    assert len(result["kernels"]) == 2
    assert all(row["aggregate"] is True for row in result["kernels"])
    assert [row["record_count"] for row in result["kernels"]] == [3, 2]
    # Both buckets are fully busy after merging overlapping stream intervals.
    assert [row["busy_ns"] for row in result["kernels"]] == [50, 50]
    assert all(row["occupancy"] == 1.0 for row in result["kernels"])
    assert result["nvtx_spans"] == entry["nvtx_spans"]


def test_timeline_web_lod_keeps_small_windows_exact():
    entry = {
        "id": 0,
        "kernels": [
            {"type": "kernel", "name": "one", "start_ns": 10, "end_ns": 20},
            {"type": "kernel", "name": "two", "start_ns": 30, "end_ns": 40},
        ],
        "nvtx_spans": [],
    }

    result = _aggregate_timeline_gpu_entry(entry, 0, 100, 2)

    assert result["lod"]["mode"] == "exact"
    assert result["kernels"] == entry["kernels"]


def test_timeline_web_duckdb_lod_returns_bounded_rows(minimal_nsys_db_path):
    with Profile(minimal_nsys_db_path) as prof:
        trim = (0, 5_000_000)
        result = build_timeline_gpu_data_lod(prof, [0], trim, max_buckets=2)

    assert result[0]["lod"]["mode"] == "aggregate"
    assert result[0]["lod"]["record_count"] > 2
    assert len(result[0]["kernels"]) <= 2
    assert all(row["aggregate"] is True for row in result[0]["kernels"])


def test_timeline_web_frontend_requests_max_buckets_and_marks_aggregates():
    javascript = Path("src/nsys_ai/templates/timeline.js").read_text(encoding="utf-8")

    assert "max_buckets=${maxBuckets}" in javascript
    assert "resolution=" not in javascript
    assert "k.aggregate === true" in javascript
    assert "Aggregate bucket" in javascript


def test_timeline_web_kernel_first_keeps_kernels_outside_nvtx(minimal_nsys_db_path):
    conn = sqlite3.connect(minimal_nsys_db_path)
    conn.execute("INSERT INTO StringIds(id, value) VALUES (?, ?)", (3, "kernel_C"))
    conn.execute(
        """
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
        (globalPid, deviceId, streamId, correlationId, start, end, shortName, demangledName, gridX, gridY, gridZ, blockX, blockY, blockZ)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (100, 0, 9, 3, 4_600_000, 4_800_000, 3, 3, 1, 1, 1, 1, 1, 1),
    )
    conn.commit()
    conn.close()

    with Profile(minimal_nsys_db_path) as prof:
        data = build_timeline_gpu_data(prof, 0, (0, 5_000_000))
        gpu0 = data[0]
        kernel_names = {k["name"] for k in gpu0["kernels"]}

        assert "kernel_C" in kernel_names
        assert len(gpu0["kernels"]) == 8

        k_c = next(k for k in gpu0["kernels"] if k["name"] == "kernel_C")
        assert k_c["path"] == "kernel_C"


def test_timeline_web_nvtx_slice_keeps_boundary_overlaps():
    spans = [
        {"name": "before", "start": 0, "end": 10},
        {"name": "inside", "start": 20, "end": 30},
        {"name": "after", "start": 40, "end": 50},
    ]

    assert [s["name"] for s in _slice_nvtx_spans(spans, 10, 40)] == [
        "before",
        "inside",
        "after",
    ]
    assert [s["name"] for s in _slice_nvtx_spans(spans, 11, 39)] == ["inside"]


def test_timeline_web_overview_covers_full_profile_without_loaded_tiles():
    bins, kernel_count = _build_timeline_overview(
        [
            {
                "id": 0,
                "kernels": [
                    {"start_ns": 100, "end_ns": 200, "duration_ms": 0.1},
                    {"start_ns": 2_100, "end_ns": 2_300, "duration_ms": 0.2},
                ],
            }
        ],
        (0, 4_000),
        bin_count=4,
    )

    assert kernel_count == 2
    assert bins[0] > 0
    assert bins[2] > 0
    assert bins[1] == 0
    assert bins[3] == 0


def test_timeline_web_trim_uses_overlap_not_containment(minimal_nsys_db_path):
    with Profile(minimal_nsys_db_path) as prof:
        gpu_data = build_timeline_gpu_data(prof, 0, (1_500_000, 1_600_000))
        kernels = gpu_data[0]["kernels"]
        names = [k["name"] for k in kernels]

        # kernel_A spans 1.0ms-2.0ms and must be included by overlap logic.
        assert names == ["kernel_A"]

        payload = json.loads(generate_timeline_data_json(prof, [0], (1_500_000, 1_600_000)))
        assert "gpus" in payload
        assert payload["gpus"][0]["kernels"][0]["name"] == "kernel_A"


def test_timeline_web_can_build_kernels_without_nvtx(minimal_nsys_db_path):
    with Profile(minimal_nsys_db_path) as prof:
        gpu_data = build_timeline_gpu_data(
            prof,
            0,
            (0, 5_000_000),
            include_kernels=True,
            include_nvtx=False,
        )
        entry = gpu_data[0]
        assert len(entry["kernels"]) == 7
        assert entry["nvtx_spans"] == []


def test_timeline_web_can_build_nvtx_without_kernels(minimal_nsys_db_path):
    with Profile(minimal_nsys_db_path) as prof:
        gpu_data = build_timeline_gpu_data(
            prof,
            0,
            (0, 5_000_000),
            include_kernels=False,
            include_nvtx=True,
        )
        entry = gpu_data[0]
        assert entry["kernels"] == []
        assert len(entry["nvtx_spans"]) >= 1
        assert "thread" in entry["nvtx_spans"][0]


def test_timeline_web_includes_memcpy_and_memset_events(minimal_nsys_db_path):
    conn = sqlite3.connect(minimal_nsys_db_path)
    conn.execute(
        """
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY
        (globalPid, deviceId, streamId, copyKind, bytes, srcKind, dstKind, start, end)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (100, 0, 11, 1, 4096, 1, 3, 1_200_000, 1_350_000),
    )
    conn.execute(
        """
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET
        (globalPid, deviceId, streamId, bytes, value, start, end)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (100, 0, 12, 8192, 0, 3_200_000, 3_450_000),
    )
    conn.commit()
    conn.close()

    with Profile(minimal_nsys_db_path) as prof:
        gpu_data = build_timeline_gpu_data(prof, 0, (1_000_000, 4_000_000))
        events = gpu_data[0]["kernels"]
        memcpy_events = [e for e in events if e["type"] == "memcpy"]
        memset_events = [e for e in events if e["type"] == "memset"]

    assert len(memcpy_events) == 2
    assert len(memset_events) == 1
    assert memcpy_events[0]["name"] == "[CUDA memcpy H2D]"
    assert memcpy_events[0]["path"] == "[CUDA memcpy H2D]"
    assert memset_events[0]["name"] == "[CUDA memset]"
    assert memset_events[0]["path"] == "[CUDA memset]"


def test_timeline_web_template_uses_external_assets(minimal_nsys_db_path):
    with Profile(minimal_nsys_db_path) as prof:
        html = generate_timeline_html(prof, [0], None)
    assert 'href="/assets/timeline.css"' in html
    assert 'src="/assets/timeline.js"' in html
    assert "window.__TIMELINE_BOOTSTRAP__" in html
    assert 'id="loopBtn"' in html
    assert 'id="loopSidebar"' in html
    assert 'id="inspectorRail"' in html
    assert 'id="inspectorTabs"' in html
    assert 'id="inspectorTabChat"' in html
    assert 'id="inspectorTabFindings"' in html
    assert 'id="inspectorTabLoop"' in html
    assert 'id="workflowEdgeTab"' in html
    assert 'id="workflowEdgeLabel"' in html
    assert 'id="workflowEdgeProgress"' in html
    assert 'aria-controls="inspectorRail"' in html
    assert 'onclick="toggleLoop()"' in html
    assert 'data-inspector-panel="chat"' in html
    assert 'data-inspector-panel="findings"' in html
    assert 'data-inspector-panel="loop"' in html
    assert "LOOP_TRIM_NS" in html
    assert "PROFILE_ID" in html


def test_timeline_web_template_has_nvtx_command_controls(minimal_nsys_db_path):
    with Profile(minimal_nsys_db_path) as prof:
        html = generate_timeline_html(prof, [0], None)
    assert 'id="searchInput"' in html
    assert "searchInput" in html
    assert "Kernels" in html and "NVTX" in html
    assert 'id="commandPalette"' in html
    assert 'id="commandInput"' in html
    assert 'id="settingsBtn"' in html
    assert 'id="settingsPanel"' in html
    assert 'id="settingsApplyLockBtn"' in html
    assert 'id="setRenderLockEnabled"' in html
    assert 'id="setRenderLockStart"' in html
    assert 'id="setRenderLockEnd"' in html
    assert 'id="setHierarchyLayout"' in html
    assert 'id="setRulerLabelMode"' in html
    assert 'id="detailResizeHandle"' in html
    assert 'id="overviewCanvas"' in html
    assert 'id="viewportReadout"' in html
    assert 'id="viewBackBtn"' in html
    assert 'id="gpuSel"' in html
    assert 'id="focusBtn"' in html
    assert 'id="chatCapabilities"' in html
    assert "fit_nvtx_range" in html
    assert "Go to NVTX" in html


def test_workflow_edge_tab_contract_is_wired():
    css = Path("src/nsys_ai/templates/timeline.css").read_text(encoding="utf-8")
    js = Path("src/nsys_ai/templates/timeline.js").read_text(encoding="utf-8")

    assert "#workflowEdgeTab[hidden]" in css
    assert "@media (max-width: 960px)" in css
    assert "function updateWorkflowEdgeTab()" in js
    assert "tab.hidden = railOpen" in js
    assert "loopRenderState()" in js


def test_timeline_template_declutters_inspector_and_annotations(minimal_nsys_db_path):
    with Profile(minimal_nsys_db_path) as prof:
        html = generate_timeline_html(
            prof,
            [0],
            None,
            findings_data=[
                {
                    "label": "Overlapping finding",
                    "severity": "warning",
                    "type": "region",
                    "start_ns": 1,
                    "end_ns": 2,
                }
            ],
        )
    assert 'onclick="closeInspector()"' in html
    assert 'data-inspector-panel="findings"' in html


def test_timeline_html_export_writes_sidecar_assets(minimal_nsys_db_path, tmp_path):
    out_html = tmp_path / "timeline.html"
    with Profile(minimal_nsys_db_path) as prof:
        write_timeline_html(prof, 0, (0, 5_000_000), str(out_html))

    out_css = tmp_path / "timeline.css"
    out_js = tmp_path / "timeline.js"
    out_tokens = tmp_path / "tokens.css"
    assert out_html.exists()
    assert out_css.exists()
    assert out_js.exists()
    assert out_tokens.exists()

    html_text = out_html.read_text(encoding="utf-8")
    assert 'href="timeline.css"' in html_text
    assert 'src="timeline.js"' in html_text
    assert '@import url("tokens.css")' in out_css.read_text(encoding="utf-8")
