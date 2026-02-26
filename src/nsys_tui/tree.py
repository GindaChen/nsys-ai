"""
tree.py — Build hierarchical NVTX trees with nested GPU kernels.

Provides a "function stack trace" view: NVTX annotations form the tree
structure, GPU kernels are leaves attached to their innermost NVTX range.

Key design: builds the tree from the PRIMARY CPU thread for the target GPU
(the thread with the most correlated kernel launches on that device).
Mixing threads breaks containment-based nesting.
"""
from .projection import _compute_depth


def _find_primary_thread(profile, device: int) -> int:
    """Find the CPU thread with the most kernel launches on this device."""
    rows = profile.conn.execute("""
        SELECT r.globalTid, COUNT(*) as cnt
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON r.correlationId = k.correlationId
        WHERE k.deviceId = ?
        GROUP BY r.globalTid ORDER BY cnt DESC LIMIT 1
    """, (device,)).fetchone()
    return rows[0] if rows else 0


def build_nvtx_tree(profile, device: int, trim: tuple[int, int],
                    pad: int = int(5e9)) -> list[dict]:
    """
    Build a hierarchical tree of NVTX annotations with GPU kernels as leaves.

    Uses only the primary thread's NVTX events (the thread that launches the
    most kernels on the target device). This preserves the correct CPU call-stack
    nesting that gets destroyed when mixing events from multiple threads.

    Each node: {name, start, end, type: "nvtx"|"kernel", stream?, children: [...]}
    """
    kmap = profile.kernel_map(device)
    if not kmap:
        return []

    primary_tid = _find_primary_thread(profile, device)
    if not primary_tid:
        return []

    # Load runtime calls for primary thread
    rt_rows = profile.conn.execute("""
        SELECT start, [end], correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE globalTid = ? AND start >= ? AND [end] <= ?  ORDER BY start
    """, (primary_tid, trim[0] - pad, trim[1] + int(2e9))).fetchall()

    # Load NVTX for primary thread only
    nvtx_rows = profile.conn.execute("""
        SELECT text, start, [end] FROM NVTX_EVENTS
        WHERE text IS NOT NULL AND [end] > start
          AND globalTid = ?
          AND start >= ? AND start <= ?
        ORDER BY start
    """, (primary_tid, trim[0] - pad, trim[1])).fetchall()

    # Build projected entries: each NVTX span → projected GPU bounds + child kernels
    entries = []  # list of {name, gpu_start, gpu_end, cpu_start, cpu_end, kernels: [...]}

    for n in nvtx_rows:
        text, cpu_start, cpu_end = n["text"], n["start"], n["end"]
        if not text:
            continue

        # Find correlated kernels on this GPU
        child_kernels = []
        for rt in rt_rows:
            if rt["start"] > cpu_end:
                break
            if rt["start"] >= cpu_start and rt["end"] <= cpu_end:
                k = kmap.get(rt["correlationId"])
                if k:
                    child_kernels.append(dict(
                        name=k["name"], demangled=k.get("demangled",""),
                        start=k["start"], end=k["end"],
                        stream=k["stream"], type="kernel", children=[]))

        if not child_kernels:
            continue

        gpu_start = min(k["start"] for k in child_kernels)
        gpu_end = max(k["end"] for k in child_kernels)

        if gpu_end < trim[0] or gpu_start > trim[1]:
            continue

        entries.append(dict(
            name=text, start=gpu_start, end=gpu_end,
            cpu_start=cpu_start, cpu_end=cpu_end,
            type="nvtx", kernels=child_kernels, children=[]))

    # Nest by CPU time containment (parent = the NVTX whose CPU range contains ours)
    # Since entries are sorted by CPU start time (from the SQL ORDER BY), and NVTX
    # forms a proper stack (push/pop), we can use a simple stack.
    roots = []
    stack = []  # stack of entries (parents)

    for entry in entries:
        # Pop entries whose CPU range has ended before this one starts
        while stack and stack[-1]["cpu_end"] <= entry["cpu_start"]:
            stack.pop()

        node = dict(name=entry["name"], start=entry["start"], end=entry["end"],
                    type="nvtx", children=list(entry["kernels"]))

        if stack:
            stack[-1]["_node"]["children"].append(node)
        else:
            roots.append(node)

        entry["_node"] = node
        stack.append(entry)

    # Deduplicate: remove kernels from parents if a child NVTX claims them
    _deduplicate_kernels(roots)

    # Sort children: interleave by start time so NVTX nodes aren't buried under kernels
    _sort_children(roots)

    return roots


def _sort_children(nodes):
    """Recursively sort children by start time."""
    for node in nodes:
        if node.get("children"):
            node["children"].sort(key=lambda c: c["start"])
            _sort_children(node["children"])


def _deduplicate_kernels(nodes):
    """Remove kernel children that are also present in a deeper NVTX child."""
    for node in nodes:
        if node["type"] != "nvtx":
            continue
        # Collect all kernel start times claimed by NVTX children (recursively)
        child_kernel_starts = set()
        for child in node["children"]:
            if child["type"] == "nvtx":
                _collect_kernel_starts(child, child_kernel_starts)
        # Remove kernels that are claimed deeper
        node["children"] = [c for c in node["children"]
                           if c["type"] != "kernel" or c["start"] not in child_kernel_starts]
        _deduplicate_kernels(node["children"])


def _collect_kernel_starts(node, starts):
    for child in node.get("children", []):
        if child["type"] == "kernel":
            starts.add(child["start"])
        else:
            _collect_kernel_starts(child, starts)


def format_text(roots, indent=0) -> str:
    """Render tree as indented text (like a stack trace)."""
    lines = []
    for node in roots:
        dur_ms = (node["end"] - node["start"]) / 1e6
        prefix = "  " * indent
        icon = "📦" if node["type"] == "nvtx" else "⚡"
        extra = f" [stream {node['stream']}]" if "stream" in node else ""
        lines.append(f"{prefix}{icon} {node['name']}{extra}  ({dur_ms:.3f}ms)")
        if node.get("children"):
            lines.append(format_text(node["children"], indent + 1))
    return "\n".join(lines)


def to_json(roots, parent_duration_ms: float = 0, path: str = "") -> list[dict]:
    """
    Convert tree to JSON-serializable format with durations and heat metrics.

    Each node gets:
      - duration_ms: absolute duration
      - heat: 0.0-1.0 normalized to max sibling (for color encoding)
      - relative_pct: % of parent duration (for bar width)
      - start_ns, end_ns: timestamps for timeline views
      - path: NVTX path string (e.g. "sample_0 > Attention > flash_fwd_kernel")
    """
    out = []
    # Compute max sibling duration for heat normalization
    durations = [(node["end"] - node["start"]) / 1e6 for node in roots]
    max_dur = max(durations) if durations else 1

    for node, dur_ms in zip(roots, durations):
        dur_ms_r = round(dur_ms, 3)
        heat = round(dur_ms / max_dur, 3) if max_dur > 0 else 0
        rel_pct = round(100 * dur_ms / parent_duration_ms, 1) if parent_duration_ms > 0 else 100.0
        node_path = f"{path} > {node['name']}" if path else node["name"]

        d = dict(name=node["name"], type=node["type"],
                 duration_ms=dur_ms_r, heat=heat, relative_pct=rel_pct,
                 start_ns=node["start"], end_ns=node["end"],
                 path=node_path)
        if "stream" in node:
            d["stream"] = node["stream"]
        if node.get("demangled"):
            d["demangled"] = node["demangled"]
        if node.get("children"):
            d["children"] = to_json(node["children"], dur_ms, node_path)
        out.append(d)
    return out


def format_markdown(roots, depth=0) -> str:
    """
    Render tree as structured markdown for LLM agent consumption.

    NVTX nodes become headers, kernels become bullet items.
    Designed for easy parsing by language models analyzing profiles.
    """
    lines = []
    for node in roots:
        dur_ms = (node["end"] - node["start"]) / 1e6
        if node["type"] == "nvtx":
            # NVTX annotations become headers (h2-h6, capped at h6)
            level = min(depth + 2, 6)
            hdr = "#" * level
            children = node.get("children", [])
            nvtx_children = [c for c in children if c["type"] == "nvtx"]
            kern_children = [c for c in children if c["type"] == "kernel"]

            lines.append(f"{hdr} {node['name']} ({dur_ms:.1f}ms)")
            lines.append("")

            if kern_children:
                lines.append(f"| Kernel | Stream | Duration |")
                lines.append(f"|--------|--------|----------|")
                for k in kern_children:
                    k_dur = (k["end"] - k["start"]) / 1e6
                    lines.append(f"| {k['name']} | {k.get('stream', '?')} | {k_dur:.3f}ms |")
                lines.append("")

            for child in nvtx_children:
                lines.append(format_markdown([child], depth + 1))
        else:
            # Standalone kernel (shouldn't happen at root, but handle it)
            stream = node.get("stream", "?")
            lines.append(f"- ⚡ **{node['name']}** [stream {stream}] ({dur_ms:.3f}ms)")

    return "\n".join(lines)

