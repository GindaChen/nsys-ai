"""
nvtx_tree.py — Build hierarchical NVTX trees with nested GPU kernels.

Provides a "function stack trace" view: NVTX annotations form the tree
structure, GPU kernels are leaves attached to their innermost NVTX range.

Key design: builds separate per-thread trees, then merges them by GPU time.
Each thread's tree preserves correct CPU call-stack nesting. Multiple threads
are needed because PyTorch/autograd runs backward passes on separate
``pt_autograd_*`` threads that are invisible from the primary training thread.
"""


def _find_kernel_threads(profile, device: int,
                         min_pct: float = 0.5) -> list[int]:
    """Find CPU threads that are significant kernel launchers on this device.

    Only returns threads whose launch count is >= min_pct of the top thread's
    count.  This filters out cross-GPU NCCL threads that launch a few
    collectives on this device but bring unrelated NVTX context.
    """
    with profile._lock:
        rows = profile.conn.execute(f"""
            SELECT r.globalTid, COUNT(*) as cnt
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN {profile.schema.kernel_table} k ON r.correlationId = k.correlationId
            WHERE k.deviceId = ?
            GROUP BY r.globalTid ORDER BY cnt DESC
        """, (device,)).fetchall()
    if not rows:
        return []
    top_cnt = rows[0][1]
    threshold = top_cnt * min_pct
    return [r[0] for r in rows if r[1] >= threshold]


def _find_primary_thread(profile, device: int) -> int:
    """Find the CPU thread with the most kernel launches (backward-compat alias)."""
    tids = _find_kernel_threads(profile, device)
    return tids[0] if tids else 0



def _get_thread_name(profile, tid: int) -> str:
    """Look up thread name from ThreadNames+StringIds tables. Returns '' if unknown."""
    try:
        with profile._lock:
            row = profile.conn.execute("""
                SELECT s.value FROM ThreadNames t
                JOIN StringIds s ON t.nameId = s.id
                WHERE t.globalTid = ?
                ORDER BY t.priority DESC LIMIT 1
            """, (tid,)).fetchone()
        return row[0] if row else ""
    except Exception:
        return ""


def _build_single_thread_tree(profile, device: int, trim: tuple[int, int],
                              tid: int, kmap: dict,
                              pad: int = int(5e9)) -> list[dict]:
    """
    Build an NVTX tree for a single thread's events on the target GPU.

    Each node: {name, start, end, type: "nvtx"|"kernel", stream?, children: [...]}
    """
    # Load NVTX for this thread only.
    # Support both schemas: (1) only NVTX_EVENTS.text, (2) textId -> StringIds (COALESCE text, s.value).
    with profile._lock:
        if profile._nvtx_has_text_id:
            nvtx_rows = profile.conn.execute("""
                SELECT COALESCE(n.text, s.value) AS text, n.start, n.[end]
                FROM NVTX_EVENTS n
                LEFT JOIN StringIds s ON n.textId = s.id
                WHERE (n.text IS NOT NULL OR s.value IS NOT NULL) AND n.[end] > n.start
                  AND n.globalTid = ?
                  AND n.[end] >= ? AND n.start <= ?
                ORDER BY n.start
            """, (tid, trim[0] - pad, trim[1])).fetchall()
        else:
            nvtx_rows = profile.conn.execute("""
                SELECT text, start, [end]
                FROM NVTX_EVENTS
                WHERE text IS NOT NULL AND [end] > start
                  AND globalTid = ?
                  AND [end] >= ? AND start <= ?
                ORDER BY start
            """, (tid, trim[0] - pad, trim[1])).fetchall()
    if not nvtx_rows:
        return []

    # Load runtime calls for this thread covering the discovered NVTX span set.
    # Using NVTX-derived bounds keeps GPU projection (start/end/depth/path)
    # stable across adjacent timeline tiles near boundaries.
    rt_lo = min(int(n["start"]) for n in nvtx_rows)
    rt_hi = max(int(n["end"]) for n in nvtx_rows) + int(2e9)
    with profile._lock:
        rt_rows = profile.conn.execute("""
            SELECT start, [end], correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE globalTid = ? AND start >= ? AND [end] <= ?  ORDER BY start
        """, (tid, rt_lo, rt_hi)).fetchall()

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


def build_nvtx_tree(profile, device: int, trim: tuple[int, int],
                    pad: int = int(5e9)) -> list[dict]:
    """
    Build a hierarchical tree of NVTX annotations with GPU kernels as leaves.

    Discovers ALL CPU threads that launch kernels on the target device (e.g.
    the main training thread for forward pass, ``pt_autograd_*`` threads for
    backward pass), builds a per-thread NVTX tree for each, and merges the
    roots by GPU start time.

    Each root node is tagged with ``thread_name`` metadata.

    Each node: {name, start, end, type: "nvtx"|"kernel", stream?, children: [...]}
    """
    kmap = profile.kernel_map(device)
    if not kmap:
        return []

    kernel_tids = _find_kernel_threads(profile, device)
    if not kernel_tids:
        return []

    all_roots = []
    for tid in kernel_tids:
        thread_name = _get_thread_name(profile, tid)
        roots = _build_single_thread_tree(profile, device, trim, tid, kmap, pad)
        for r in roots:
            r["thread_name"] = thread_name
        all_roots.extend(roots)

    # Merge all threads' roots by GPU start time
    all_roots.sort(key=lambda n: n["start"])
    return all_roots


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
        if node.get("thread_name"):
            d["thread_name"] = node["thread_name"]
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
                lines.append("| Kernel | Stream | Duration |")
                lines.append("|--------|--------|----------|")
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
