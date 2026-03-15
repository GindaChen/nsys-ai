"""
viewer.py - Generate interactive HTML visualizations for Nsight profiles.

Uses string.Template with HTML template files for clean separation between
Python logic and HTML/CSS/JS presentation.
"""

import json
import os
from string import Template

from .tree import build_nvtx_tree, to_json

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
_CUDA_MEMCPY_KIND_LABELS = {
    1: "H2D",
    2: "D2H",
    8: "D2D",
    10: "P2P",
}


def _load_template(name: str) -> Template:
    """Load an HTML template from the templates directory."""
    path = os.path.join(_TEMPLATE_DIR, name)
    with open(path, encoding="utf-8") as f:
        return Template(f.read())


def _read_template_text(name: str) -> str:
    """Read a raw template/static file from templates directory."""
    path = os.path.join(_TEMPLATE_DIR, name)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _escape_json_for_html_script(json_str: str) -> str:
    """Escape '</' in JSON so it can be safely embedded in a <script> block."""
    return json_str.replace("</", "<\\/")


def generate_html(prof, device: int, trim: tuple[int, int]) -> str:
    """Generate a standalone HTML page showing the NVTX stack trace."""
    roots = build_nvtx_tree(prof, device, trim)
    tree_json = to_json(roots)

    gpu_info = prof.meta.gpu_info.get(device)
    gpu_label = f"GPU {device}"
    if gpu_info:
        gpu_label += (
            f" - {gpu_info.name} ({gpu_info.pci_bus}), "
            f"{gpu_info.sm_count} SMs, {gpu_info.memory_bytes / 1e9:.0f}GB"
        )

    # Stable id for this profile view (device + time window) for profile-bound chat history
    trim_sec = (trim[0] / 1e9, trim[1] / 1e9)
    profile_id = f"{device}_{trim_sec[0]:.1f}_{trim_sec[1]:.1f}"

    # Escape profile path for safe embedding in <script>
    safe_profile_path = _escape_json_for_html_script(json.dumps(prof.path))

    tmpl = _load_template("nvtx_tree.html")
    db_agent_flag = os.environ.get("NSYS_AI_DB_AGENT", "").strip().lower()
    db_agent_enabled = bool(db_agent_flag) and db_agent_flag not in ("0", "false", "no", "off")
    return tmpl.safe_substitute(
        DATA=json.dumps(tree_json),
        GPU_LABEL=gpu_label,
        TRIM_LABEL=f"{trim[0] / 1e9:.1f}s - {trim[1] / 1e9:.1f}s",
        PROFILE_ID=profile_id,
        PROFILE_PATH=safe_profile_path,
        DB_AGENT_ENABLED="1" if db_agent_enabled else "",
    )


def write_html(prof, device: int, trim: tuple[int, int], path: str):
    """Generate and write the HTML viewer to a file."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(generate_html(prof, device, trim))


def _collect_nvtx_annotations(
    nodes: list[dict],
    spans: list[dict],
    kernel_paths: dict[tuple, str],
    current_thread: str = "",
) -> None:
    """Collect flat NVTX spans and kernel-path annotations from a tree JSON."""
    for node in nodes:
        thread_name = node.get("thread_name") or current_thread
        ntype = node.get("type")
        if ntype == "nvtx":
            path = node.get("path", "")
            depth = max(len(path.split(" > ")) - 1, 0) if path else 0
            spans.append(
                {
                    "name": node.get("name", ""),
                    "start": node.get("start_ns", 0),
                    "end": node.get("end_ns", 0),
                    "depth": depth,
                    "path": path,
                    "dur": node.get("duration_ms", 0),
                    "thread": thread_name or "(unnamed)",
                }
            )
        elif ntype == "kernel":
            key = (
                node.get("start_ns"),
                node.get("end_ns"),
                node.get("stream"),
                node.get("name"),
            )
            kernel_paths[key] = node.get("path", "")

        children = node.get("children") or []
        if children:
            _collect_nvtx_annotations(children, spans, kernel_paths, thread_name)


def build_timeline_gpu_data(
    prof,
    device,
    trim: tuple[int, int],
    *,
    include_kernels: bool = True,
    include_nvtx: bool = True,
) -> list[dict]:
    """Build per-GPU timeline payload with kernel rows plus optional NVTX annotations."""
    from collections.abc import Sequence

    from .nvtx_tree import build_nvtx_tree as build_nvtx_tree_all_threads
    from .nvtx_tree import to_json as nvtx_to_json

    devices: list[int] = list(device) if isinstance(device, Sequence) else [device]
    gpu_entries: list[dict] = []

    for dev in devices:
        kernels = []
        if include_kernels:
            # 1) Kernel-first: authoritative source for timeline rows.
            kernel_sql = f"""
                SELECT k.start AS start_ns, k.[end] AS end_ns, k.streamId AS stream,
                       s.value AS name
                FROM {prof.schema.kernel_table} k
                JOIN StringIds s ON k.shortName = s.id
                WHERE k.deviceId = ? AND k.[end] >= ? AND k.start <= ?
                ORDER BY k.start
            """
            with prof._lock:
                rows = prof.conn.execute(kernel_sql, (dev, trim[0], trim[1])).fetchall()

            for r in rows:
                start_ns = int(r["start_ns"])
                end_ns = int(r["end_ns"])
                kernels.append(
                    {
                        "type": "kernel",
                        "name": r["name"],
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "duration_ms": round((end_ns - start_ns) / 1e6, 3),
                        "stream": r["stream"],
                        "path": "",
                    }
                )

            if "CUPTI_ACTIVITY_KIND_MEMCPY" in prof.schema.tables:
                memcpy_sql = """
                    SELECT m.start AS start_ns,
                           m.[end] AS end_ns,
                           m.streamId AS stream,
                           m.copyKind AS copy_kind
                    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
                    WHERE m.deviceId = ? AND m.[end] >= ? AND m.start <= ?
                    ORDER BY m.start
                """
                with prof._lock:
                    memcpy_rows = prof.conn.execute(memcpy_sql, (dev, trim[0], trim[1])).fetchall()

                for r in memcpy_rows:
                    start_ns = int(r["start_ns"])
                    end_ns = int(r["end_ns"])
                    copy_kind = int(r["copy_kind"])
                    copy_kind_label = _CUDA_MEMCPY_KIND_LABELS.get(copy_kind, f"kind={copy_kind}")
                    kernels.append(
                        {
                            "type": "memcpy",
                            "name": f"[CUDA memcpy {copy_kind_label}]",
                            "start_ns": start_ns,
                            "end_ns": end_ns,
                            "duration_ms": round((end_ns - start_ns) / 1e6, 3),
                            "stream": r["stream"],
                            "path": "",
                        }
                    )

            if "CUPTI_ACTIVITY_KIND_MEMSET" in prof.schema.tables:
                memset_sql = """
                    SELECT m.start AS start_ns,
                           m.[end] AS end_ns,
                           m.streamId AS stream
                    FROM CUPTI_ACTIVITY_KIND_MEMSET m
                    WHERE m.deviceId = ? AND m.[end] >= ? AND m.start <= ?
                    ORDER BY m.start
                """
                with prof._lock:
                    memset_rows = prof.conn.execute(memset_sql, (dev, trim[0], trim[1])).fetchall()

                for r in memset_rows:
                    start_ns = int(r["start_ns"])
                    end_ns = int(r["end_ns"])
                    kernels.append(
                        {
                            "type": "memset",
                            "name": "[CUDA memset]",
                            "start_ns": start_ns,
                            "end_ns": end_ns,
                            "duration_ms": round((end_ns - start_ns) / 1e6, 3),
                            "stream": r["stream"],
                            "path": "",
                        }
                    )

            kernels.sort(key=lambda k: (k["start_ns"], k["end_ns"]))

        # 2) NVTX-only annotations + kernel->path labels.
        #    NVTX is advisory metadata; missing mapping must not drop kernels.
        nvtx_spans: list[dict] = []
        kernel_paths: dict[tuple, str] = {}
        if include_nvtx:
            try:
                roots = build_nvtx_tree_all_threads(prof, dev, trim)
                tree_json = nvtx_to_json(roots)
            except Exception:
                tree_json = []
            _collect_nvtx_annotations(tree_json, nvtx_spans, kernel_paths)

        if include_kernels:
            for k in kernels:
                if k.get("type") != "kernel":
                    k["path"] = k["name"]
                    continue
                key = (k["start_ns"], k["end_ns"], k["stream"], k["name"])
                k["path"] = kernel_paths.get(key, k["name"])

        gpu_entries.append({"id": dev, "kernels": kernels, "nvtx_spans": nvtx_spans})

    return gpu_entries


def generate_timeline_data_json(prof, devices, trim: tuple[int, int]) -> str:
    """Return JSON string of per-GPU kernel/NVTX data for a time window.

    Called by the ``/api/data`` endpoint for on-demand tile loading.
    """
    gpu_entries = build_timeline_gpu_data(prof, devices, trim)
    return json.dumps({"gpus": gpu_entries})


def generate_timeline_html(
    prof,
    device,
    trim: tuple[int, int] | None = None,
    *,
    findings_data: list[dict] | None = None,
    timeline_css_href: str = "/assets/timeline.css",
    timeline_js_src: str = "/assets/timeline.js",
    api_prefix: str = "",
    profile_path: str = "",
) -> str:
    """Generate a standalone HTML page with the horizontal timeline viewer.

    *device* may be a single int or a list of ints.
    When *trim* is None, HTML is generated in progressive mode: ``$DATA``
    is ``null`` and the template fetches data via ``/api/data`` on demand.
    """
    from collections.abc import Sequence

    devices: list[int] = list(device) if isinstance(device, Sequence) else [device]

    # Build GPU info list (for dropdown) and compact label
    gpu_details = []
    gpu_type = "GPU"
    for dev in devices:
        gpu_info = prof.meta.gpu_info.get(dev)
        detail = {"id": dev, "name": "Unknown", "pci": "", "sms": 0, "mem_gb": 0}
        if gpu_info:
            detail["name"] = gpu_info.name
            detail["pci"] = gpu_info.pci_bus
            detail["sms"] = gpu_info.sm_count
            detail["mem_gb"] = round(gpu_info.memory_bytes / 1e9)
            gpu_type = gpu_info.name
        gpu_details.append(detail)
    gpu_info_json = json.dumps(gpu_details)
    gpu_label = f"{len(devices)}× {gpu_type}" if len(devices) > 1 else gpu_type
    gpu_label_json = json.dumps(gpu_label)

    if trim is not None:
        # Full data baked into HTML (kernel-first payload).
        gpu_entries = build_timeline_gpu_data(prof, devices, trim)
        data_json = json.dumps({"gpus": gpu_entries})
        trim_label = f"{trim[0] / 1e9:.1f}s - {trim[1] / 1e9:.1f}s"
        progressive = ""
    else:
        # Progressive mode: no data baked in
        data_json = "null"
        trim_label = "Progressive"
        progressive = "1"

    safe_data_json = _escape_json_for_html_script(data_json)
    safe_gpu_info_json = _escape_json_for_html_script(gpu_info_json)
    safe_gpu_label_json = _escape_json_for_html_script(gpu_label_json)
    safe_findings_json = _escape_json_for_html_script(json.dumps(findings_data or []))
    safe_profile_path_json = _escape_json_for_html_script(
        json.dumps(profile_path) if profile_path is not None else "null"
    )

    tmpl = _load_template("timeline.html")
    return tmpl.safe_substitute(
        DATA=safe_data_json,
        GPU_LABEL=gpu_label,
        GPU_LABEL_JSON=safe_gpu_label_json,
        GPU_INFO_JSON=safe_gpu_info_json,
        TRIM_LABEL=trim_label,
        PROGRESSIVE=progressive,
        TIMELINE_CSS_HREF=timeline_css_href,
        TIMELINE_JS_SRC=timeline_js_src,
        API_PREFIX=api_prefix,
        FINDINGS_JSON=safe_findings_json,
        PROFILE_PATH=safe_profile_path_json,
    )


def write_timeline_html(prof, device: int, trim: tuple[int, int], path: str):
    """Generate and write the timeline HTML viewer to a file."""
    out_dir = os.path.dirname(os.path.abspath(path))
    css_name = "timeline.css"
    js_name = "timeline.js"
    os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(
            generate_timeline_html(
                prof,
                device,
                trim,
                timeline_css_href=css_name,
                timeline_js_src=js_name,
            )
        )

    # Export sidecar static assets so generated HTML remains self-contained on disk.
    with open(os.path.join(out_dir, css_name), "w", encoding="utf-8", newline="\n") as f:
        f.write(_read_template_text("timeline.css"))
    with open(os.path.join(out_dir, js_name), "w", encoding="utf-8", newline="\n") as f:
        f.write(_read_template_text("timeline.js"))


def generate_evidence_html(
    prof,
    device,
    findings_data: list[dict],
    title: str = "Evidence View",
    *,
    trim: tuple[int, int] | None = None,
    evidence_css_href: str = "/assets/evidence.css",
    evidence_js_src: str = "/assets/evidence.js",
    api_prefix: str = "",
) -> str:
    """Generate the Evidence View HTML page.

    *findings_data* is a list of Finding dicts (from annotation.py).
    *device* may be a single int or list of ints.
    When *trim* is None, progressive mode is used (kernels fetched via API).
    """
    from collections.abc import Sequence

    devices: list[int] = list(device) if isinstance(device, Sequence) else [device]

    # GPU label
    gpu_type = "GPU"
    for dev in devices:
        gpu_info = prof.meta.gpu_info.get(dev)
        if gpu_info:
            gpu_type = gpu_info.name
    gpu_label = f"{len(devices)}× {gpu_type}" if len(devices) > 1 else gpu_type
    gpu_label_json = json.dumps(gpu_label)

    # Time range
    time_range = list(prof.meta.time_range)

    if trim is not None:
        # Bake kernel data into HTML
        gpu_entries = build_timeline_gpu_data(
            prof, devices, trim, include_kernels=True, include_nvtx=False
        )
        kernels = []
        streams_set = set()
        for entry in gpu_entries:
            for k in entry.get("kernels", []):
                kernels.append(k)
                streams_set.add(k.get("stream", 0))
        progressive = ""
    else:
        kernels = []
        streams_set = set()
        progressive = "1"

    streams = sorted(streams_set)

    # Load evidence template from disk when available; otherwise fall back to a
    # minimal inline template so this view does not fail when the asset is
    # missing from the templates directory.
    evidence_template_path = os.path.join(_TEMPLATE_DIR, "evidence.html")
    if os.path.exists(evidence_template_path):
        tmpl = _load_template("evidence.html")
    else:
        tmpl = Template(
            """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>${TITLE}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="${EVIDENCE_CSS_HREF}">
</head>
<body>
    <div id="app" data-api-prefix="${API_PREFIX}"
         data-gpu-label='${GPU_LABEL_JSON}'
         data-time-range='${TIME_RANGE_JSON}'
         data-findings='${FINDINGS_JSON}'
         data-kernels='${KERNELS_JSON}'
         data-streams='${STREAMS_JSON}'
         data-progressive="${PROGRESSIVE}">
    </div>
    <script src="${EVIDENCE_JS_SRC}"></script>
</body>
</html>
"""
        )
    return tmpl.safe_substitute(
        TITLE=title,
        FINDINGS_JSON=json.dumps(findings_data),
        KERNELS_JSON=json.dumps(kernels),
        STREAMS_JSON=json.dumps(streams),
        GPU_LABEL_JSON=gpu_label_json,
        TIME_RANGE_JSON=json.dumps(time_range),
        PROGRESSIVE=progressive,
        EVIDENCE_CSS_HREF=evidence_css_href,
        EVIDENCE_JS_SRC=evidence_js_src,
        API_PREFIX=api_prefix,
    )
