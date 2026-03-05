import json
import sqlite3

from nsys_ai.profile import Profile
from nsys_ai.viewer import (
    build_timeline_gpu_data,
    generate_timeline_data_json,
    generate_timeline_html,
    write_timeline_html,
)


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
        assert len(gpu0["kernels"]) == 3

        k_c = next(k for k in gpu0["kernels"] if k["name"] == "kernel_C")
        assert k_c["path"] == "kernel_C"


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
        assert len(entry["kernels"]) == 2
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

    assert len(memcpy_events) == 1
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


def test_timeline_web_template_has_nvtx_command_controls(minimal_nsys_db_path):
    with Profile(minimal_nsys_db_path) as prof:
        html = generate_timeline_html(prof, [0], None)
    assert 'id="searchInput"' in html
    assert "search kernels + NVTX" in html
    assert 'id="commandPalette"' in html
    assert 'id="commandInput"' in html
    assert 'id="settingsBtn"' in html
    assert 'id="settingsPanel"' in html
    assert 'id="settingsApplyLockBtn"' in html
    assert 'id="setRenderLockEnabled"' in html
    assert 'id="setRenderLockStart"' in html
    assert 'id="setRenderLockEnd"' in html
    assert 'id="setHierarchyLayout"' in html
    assert 'id="detailResizeHandle"' in html
    assert 'id="chatCapabilities"' in html
    assert "fit_nvtx_range" in html
    assert "Go to NVTX" in html


def test_timeline_html_export_writes_sidecar_assets(minimal_nsys_db_path, tmp_path):
    out_html = tmp_path / "timeline.html"
    with Profile(minimal_nsys_db_path) as prof:
        write_timeline_html(prof, 0, (0, 5_000_000), str(out_html))

    out_css = tmp_path / "timeline.css"
    out_js = tmp_path / "timeline.js"
    assert out_html.exists()
    assert out_css.exists()
    assert out_js.exists()

    html_text = out_html.read_text(encoding="utf-8")
    assert 'href="timeline.css"' in html_text
    assert 'src="timeline.js"' in html_text
