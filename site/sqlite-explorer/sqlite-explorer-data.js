// sqlite-explorer-data.js — Data + logic for the Nsight SQLite Schema Explorer
// Separated from HTML to stay within generation limits.

const TC = {
    'CUPTI_ACTIVITY_KIND_KERNEL': 'var(--hl-kernel)',
    'CUPTI_ACTIVITY_KIND_RUNTIME': 'var(--hl-runtime)',
    'NVTX_EVENTS': 'var(--hl-nvtx)',
    'StringIds': 'var(--hl-string)',
    'TARGET_INFO_GPU': 'var(--hl-gpu)',
    'TARGET_INFO_CUDA_DEVICE': 'var(--hl-cuda)',
    'CUPTI_ACTIVITY_KIND_MEMCPY': 'var(--hl-memcpy)',
    'CUPTI_ACTIVITY_KIND_MEMSET': 'var(--hl-memset)',
    'TARGET_INFO_SESSION_START_TIME': 'var(--hl-session)',
    'OSRT_API': 'var(--hl-osrt)',
    'ThreadNames': 'var(--hl-thread)',
    'PROFILER_OVERHEAD': 'var(--hl-overhead)',
    'SCHED_EVENTS': 'var(--hl-sched)',
    'COMPOSITE_EVENTS': 'var(--hl-composite)',
    'SAMPLING_CALLCHAINS': 'var(--hl-sampling)',
    'ProcessStreams': 'var(--hl-process)',
    'CUDA_CALLCHAINS': 'var(--hl-cudacc)',
    'OSRT_CALLCHAINS': 'var(--hl-osrtcc)',
    'CUDA_GRAPH_EVENTS': 'var(--hl-cudagraph)',
    'FECS_EVENTS': 'var(--hl-fecs)',
};
function gc(n) { return TC[n] || 'var(--text-dim)'; }

const TABLES = [
    {
        name: 'CUPTI_ACTIVITY_KIND_KERNEL', cat: 'GPU Activity',
        desc: 'Every GPU kernel execution — the most important table. Tells you when each kernel ran, on which GPU/stream, and for how long.',
        cols: [['start', 'INT', 'Kernel start time (ns, GPU clock)'], ['end', 'INT', 'Kernel end time (ns)'], ['deviceId', 'INT', 'GPU index (0,1,2…)'], ['streamId', 'INT', 'CUDA stream ID'], ['shortName', 'INT', 'FK→StringIds — short name'], ['demangledName', 'INT', 'FK→StringIds — full C++ name'], ['correlationId', 'INT', 'Links to Runtime table'], ['gridX', 'INT', 'Grid blocks X'], ['gridY', 'INT', 'Grid blocks Y'], ['gridZ', 'INT', 'Grid blocks Z'], ['blockX', 'INT', 'Threads/block X'], ['blockY', 'INT', 'Threads/block Y'], ['blockZ', 'INT', 'Threads/block Z'], ['registersPerThread', 'INT', 'Registers per thread'], ['staticSharedMemory', 'INT', 'Static shared mem (bytes)'], ['dynamicSharedMemory', 'INT', 'Dynamic shared mem (bytes)']]
    },
    {
        name: 'CUPTI_ACTIVITY_KIND_RUNTIME', cat: 'CPU Activity',
        desc: 'CUDA Runtime API calls (cudaLaunchKernel, cudaMemcpy…). The correlationId links each CPU launch to its GPU kernel.',
        cols: [['start', 'INT', 'API call start (ns, CPU)'], ['end', 'INT', 'API call end (ns)'], ['globalTid', 'INT', 'CPU thread (PID×2²⁴+TID)'], ['correlationId', 'INT', 'Links to Kernel table'], ['nameId', 'INT', 'FK→StringIds — API name']]
    },
    {
        name: 'NVTX_EVENTS', cat: 'Annotations',
        desc: 'User-defined labels marking code regions ("forward_pass", "attention"). Run on CPU, projected to GPU via Runtime→Kernel correlation.',
        cols: [['start', 'INT', 'Range start (ns, CPU)'], ['end', 'INT', 'Range end (ns). NULL for marks'], ['text', 'TEXT', 'Annotation label'], ['globalTid', 'INT', 'CPU thread ID'], ['eventType', 'INT', '59=push/pop, 60=start/end, 34=mark, 33=category'], ['domainId', 'INT', 'NVTX domain'], ['category', 'INT', 'User category ID']]
    },
    {
        name: 'StringIds', cat: 'Lookup',
        desc: 'Global string table. Kernel names, API names stored as integer IDs elsewhere, resolved here.',
        cols: [['id', 'INT', 'Primary key'], ['value', 'TEXT', 'Actual string']]
    },
    {
        name: 'TARGET_INFO_GPU', cat: 'Hardware',
        desc: 'Physical GPU info — model name, SM count, memory, PCIe bus. Join with TARGET_INFO_CUDA_DEVICE to map deviceId.',
        cols: [['id', 'INT', 'Internal GPU ID'], ['name', 'TEXT', 'GPU model (e.g. "NVIDIA H200")'], ['busLocation', 'TEXT', 'PCIe bus'], ['smCount', 'INT', 'SM count'], ['totalMemory', 'INT', 'Memory (bytes)'], ['chipName', 'TEXT', 'Chip arch (e.g. "GH100")'], ['memoryBandwidth', 'INT', 'Bandwidth (bytes/s)']]
    },
    {
        name: 'TARGET_INFO_CUDA_DEVICE', cat: 'Hardware',
        desc: 'Maps CUDA deviceId (used in kernel events) to physical GPU ID in TARGET_INFO_GPU.',
        cols: [['cudaId', 'INT', 'CUDA device index = kernel.deviceId'], ['gpuId', 'INT', 'FK→TARGET_INFO_GPU.id']]
    },
    {
        name: 'CUPTI_ACTIVITY_KIND_MEMCPY', cat: 'GPU Activity',
        desc: 'Memory copies. copyKind: 1=H2D, 2=D2H, 8=D2D, 10=P2P.',
        cols: [['start', 'INT', 'Copy start (ns)'], ['end', 'INT', 'Copy end (ns)'], ['deviceId', 'INT', 'GPU'], ['streamId', 'INT', 'Stream'], ['copyKind', 'INT', '1=H2D,2=D2H,8=D2D,10=P2P'], ['bytes', 'INT', 'Bytes transferred'], ['srcKind', 'INT', 'Source memory kind'], ['dstKind', 'INT', 'Dest memory kind']]
    },
    {
        name: 'CUPTI_ACTIVITY_KIND_MEMSET', cat: 'GPU Activity',
        desc: 'GPU memory set operations (cudaMemset).',
        cols: [['start', 'INT', 'Start (ns)'], ['end', 'INT', 'End (ns)'], ['deviceId', 'INT', 'GPU'], ['streamId', 'INT', 'Stream'], ['bytes', 'INT', 'Bytes set'], ['value', 'INT', 'Fill value']]
    },
    {
        name: 'TARGET_INFO_SESSION_START_TIME', cat: 'Metadata',
        desc: 'Absolute wall-clock time when profiling started. Converts relative ns to real time.',
        cols: [['sessionStartTimeNs', 'INT', 'Epoch time (ns)']]
    },
    {
        name: 'OSRT_API', cat: 'CPU Activity',
        desc: 'OS runtime calls — pthread, mutex, file I/O. Shows CPU-side blocking.',
        cols: [['start', 'INT', 'Start (ns)'], ['end', 'INT', 'End (ns)'], ['globalTid', 'INT', 'Thread'], ['nameId', 'INT', 'FK→StringIds'], ['returnValue', 'INT', 'Return code']]
    },
    {
        name: 'ThreadNames', cat: 'Metadata',
        desc: 'Maps globalTid to human-readable thread names.',
        cols: [['globalTid', 'INT', 'Encoded thread ID'], ['nameId', 'INT', 'FK→StringIds — thread name']]
    },
    {
        name: 'SCHED_EVENTS', cat: 'CPU Activity',
        desc: 'CPU scheduler events — thread migrations, context switches.',
        cols: [['start', 'INT', 'Event time (ns)'], ['globalTid', 'INT', 'Thread'], ['cpuId', 'INT', 'CPU core'], ['reason', 'INT', 'Schedule reason']]
    },
    {
        name: 'COMPOSITE_EVENTS', cat: 'Sampling',
        desc: 'Aggregated CPU sampling events with cycle counts for utilization analysis.',
        cols: [['globalTid', 'INT', 'Thread'], ['cpuCycles', 'INT', 'CPU cycles sampled']]
    },
    {
        name: 'SAMPLING_CALLCHAINS', cat: 'Sampling',
        desc: 'CPU sampling call stacks — where the CPU was spending time.',
        cols: [['id', 'INT', 'Callchain ID'], ['stackDepth', 'INT', 'Frame depth'], ['symbol', 'INT', 'FK→StringIds — function name']]
    },
    {
        name: 'ProcessStreams', cat: 'Metadata',
        desc: 'Captured stdout/stderr from the profiled process.',
        cols: [['globalPid', 'INT', 'Process ID'], ['filenameId', 'INT', 'FK→StringIds'], ['contentId', 'INT', 'FK→StringIds']]
    },
    {
        name: 'PROFILER_OVERHEAD', cat: 'Metadata',
        desc: 'Time ranges where profiler instrumentation added significant overhead. Filter these out for accurate analysis.',
        cols: [['start', 'INT', 'Overhead start (ns)'], ['end', 'INT', 'Overhead end (ns)']]
    },
    {
        name: 'CUDA_CALLCHAINS', cat: 'Debug',
        desc: 'CUDA API call stacks (requires --cudabacktrace). Shows which source code launched each kernel.',
        cols: [['id', 'INT', 'Callchain ID'], ['stackDepth', 'INT', 'Frame depth'], ['symbol', 'INT', 'FK→StringIds — function'], ['callchainId', 'INT', 'Groups frames']]
    },
    {
        name: 'OSRT_CALLCHAINS', cat: 'Debug',
        desc: 'OS runtime call stacks — where OS calls (mutex, file) originated.',
        cols: [['id', 'INT', 'Callchain ID'], ['stackDepth', 'INT', 'Frame depth'], ['symbol', 'INT', 'FK→StringIds']]
    },
    {
        name: 'CUDA_GRAPH_EVENTS', cat: 'GPU Activity',
        desc: 'CUDA graph node creation and execution events.',
        cols: [['start', 'INT', 'Event time (ns)'], ['end', 'INT', 'End time'], ['graphId', 'INT', 'Graph ID'], ['nodeType', 'INT', 'Node type']]
    },
    {
        name: 'FECS_EVENTS', cat: 'GPU Activity',
        desc: 'GPU front-end context switch events — when GPU hardware switched between processes.',
        cols: [['start', 'INT', 'Event time (ns)'], ['globalTid', 'INT', 'Context'], ['deviceId', 'INT', 'GPU']]
    },
];

const CONCEPTS = [
    {
        title: '🔗 Q1: How do you associate NVTX with a kernel?',
        body: `<p>NVTX lives on <b>CPU</b>, kernels on <b>GPU</b>. They connect through <code>CUPTI_ACTIVITY_KIND_RUNTIME</code>:</p>
<div class="cdiag"><span class="hl" style="color:var(--hl-nvtx)">NVTX_EVENTS</span>           <span class="hl" style="color:var(--hl-runtime)">RUNTIME</span>                    <span class="hl" style="color:var(--hl-kernel)">KERNEL</span>
┌──────────────┐      ┌────────────────────┐      ┌──────────────────┐
│ text:"forward"│      │ globalTid (thread!) │      │ deviceId,streamId│
│ globalTid:1001│─────▶│ start, end (CPU)    │─────▶│ start, end (GPU) │
│ start,end CPU │ same │ <span class="hl" style="color:var(--accent)">correlationId</span>      │ same │ <span class="hl" style="color:var(--accent)">correlationId</span>    │
└──────────────┘ tid  └────────────────────┘  cid └──────────────────┘</div>
<ol style="margin:6px 0 0 18px;line-height:1.9;color:var(--text)">
<li>Find NVTX on <b>same thread</b> (<code>globalTid</code>) as Runtime calls</li>
<li>Find Runtime calls whose time <b>falls inside</b> the NVTX range</li>
<li>Use <code>correlationId</code> to look up the <b>GPU kernel</b></li></ol>`},
    {
        title: '🏔️ Q2: How do you find a kernel using the NVTX path?',
        body: `<p>NVTX annotations <b>nest like a call stack</b>. To traverse:</p>
<div class="cdiag"><span class="hl" style="color:var(--hl-nvtx)">sample_0(repeat=5)</span>                         ← outermost
  └── <span class="hl" style="color:var(--hl-nvtx)">TransformerLayer._forward_attention</span>    ← narrower time
        └── <span class="hl" style="color:var(--hl-nvtx)">TEDotProductAttention.forward</span>      ← even narrower
              └── <span class="hl" style="color:var(--hl-nvtx)">FlashAttention.run_attention</span>    ← innermost
                    └── <span class="hl" style="color:var(--hl-kernel)">⚡ flash_fwd_kernel</span> [stream 21]</div>
<p><b>Algorithm</b> (<code>tree.py</code>):</p>
<ol style="margin:6px 0 0 18px;line-height:1.9;color:var(--text)">
<li>Find <b>primary CPU thread</b> (most kernel launches on target GPU)</li>
<li>Load NVTX + Runtime calls for that thread, sorted by start</li>
<li>Build <code>correlationId → kernel</code> map</li>
<li>For each NVTX, find Runtime calls <b>inside</b> its CPU window</li>
<li><b>Stack-based nesting</b>: NVTX B starts before A ends → B is child of A</li>
<li>Kernels go to the <b>innermost</b> containing NVTX</li></ol>`},
    {
        title: '🔄 Q3: Can you derive a kernel\'s NVTX path?',
        body: `<p><b>Yes!</b> Reverse-trace:</p>
<ol style="margin:6px 0 0 18px;line-height:1.9;color:var(--text)">
<li>Kernel's <code>correlationId</code> → find Runtime call → get CPU thread + time</li>
<li>Find <b>all NVTX on that thread</b> containing the Runtime call time</li>
<li>Sort widest → narrowest → that's the NVTX path</li></ol>
<div class="cdiag"><span class="hl" style="color:var(--hl-kernel)">Kernel</span>: flash_fwd_kernel (correlationId=42)
  ↓
<span class="hl" style="color:var(--hl-runtime)">Runtime</span>: tid=1001, time=39.98s (correlationId=42)
  ↓ find containing NVTX on thread 1001
<span class="hl" style="color:var(--hl-nvtx)">Path</span>: sample_0 &gt; TransformerLayer &gt; FlashAttention</div>
<p>Names <em>can</em> duplicate (multiple iterations). Path uses unique timestamps internally.</p>`},
    {
        title: '⏱️ Q4: How do you get a kernel\'s time and stream?',
        body: `<p>Directly from <code>CUPTI_ACTIVITY_KIND_KERNEL</code>:</p>
<div class="cdiag">SELECT k.start, k.end,
       (k.end - k.start) / 1e6 AS duration_ms,
       k.deviceId, k.streamId,
       s.value AS kernel_name
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
WHERE k.deviceId = 4;</div>
<ul style="margin:6px 0 0 18px;line-height:1.9;color:var(--text)">
<li><code>start/end</code> — nanosecond GPU timestamps. Duration = end−start</li>
<li><code>deviceId</code> — GPU index. Join TARGET_INFO_CUDA_DEVICE→TARGET_INFO_GPU for model name</li>
<li><code>streamId</code> — CUDA stream (convention: 21=compute, 56=NCCL)</li>
<li>Name needs JOIN to <code>StringIds</code> via shortName or demangledName</li></ul>`},
];

const EXAMPLES = [
    {
        title: '🏷️ GPU Names → ID Mapping', intent: 'Maps CUDA device indices to GPU model names and specs.',
        tables: ['TARGET_INFO_GPU', 'TARGET_INFO_CUDA_DEVICE'],
        columns: ['cudaId', 'name', 'busLocation', 'smCount', 'totalMemory', 'gpuId', 'chipName'],
        sql: `SELECT c.cudaId AS device_id, g.name AS gpu_name,\n       g.busLocation AS pci_bus, g.smCount, g.totalMemory, g.chipName\nFROM TARGET_INFO_GPU g\nJOIN TARGET_INFO_CUDA_DEVICE c ON g.id = c.gpuId\nGROUP BY c.cudaId;`,
        output: [['device_id', 'INT', 'CUDA device index', '4'], ['gpu_name', 'TEXT', 'GPU model', 'NVIDIA H200 141GB HBM3'], ['pci_bus', 'TEXT', 'PCIe location', '00000000:1A:00.0'], ['smCount', 'INT', 'SM count', '132']],
        source: 'profile.py → _gpu_info()'
    },
    {
        title: '📊 Top Kernels by Duration', intent: 'Find most expensive kernels across all invocations.',
        tables: ['CUPTI_ACTIVITY_KIND_KERNEL', 'StringIds'],
        columns: ['demangledName', 'shortName', 'deviceId', 'start', 'end', 'id', 'value'],
        sql: `SELECT s.value AS kernel_name, COUNT(*) AS invocations,\n       SUM(k.end - k.start) AS total_ns\nFROM CUPTI_ACTIVITY_KIND_KERNEL k\nJOIN StringIds s ON k.demangledName = s.id\nGROUP BY k.demangledName ORDER BY total_ns DESC LIMIT 15;`,
        output: [['kernel_name', 'TEXT', 'Full name', 'void flash::flash_fwd_kernel<…>()'], ['invocations', 'INT', 'Call count', '128'], ['total_ns', 'INT', 'Total time (ns)', '3318000000']],
        source: 'analysis.js → getKernelSummary()'
    },
    {
        title: '🔗 NVTX → Kernel Mapping', intent: 'Map kernels to their innermost NVTX annotation via correlationId.',
        tables: ['NVTX_EVENTS', 'CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_KERNEL', 'StringIds'],
        columns: ['text', 'globalTid', 'start', 'end', 'correlationId', 'shortName', 'id', 'value', 'deviceId', 'streamId', 'eventType'],
        sql: `SELECT k.start, k.end, s.value AS kernel_name, n.text AS nvtx\nFROM CUPTI_ACTIVITY_KIND_KERNEL k\nJOIN StringIds s ON k.shortName = s.id\nJOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId = r.correlationId\nJOIN NVTX_EVENTS n ON n.globalTid = r.globalTid\n  AND n.start <= r.start AND n.end >= r.end AND n.eventType = 59\nWHERE k.deviceId = 4 ORDER BY n.start DESC LIMIT 10;`,
        output: [['kernel_name', 'TEXT', 'Kernel', 'flash_attn_fwd'], ['nvtx', 'TEXT', 'Innermost NVTX', 'FlashAttention.run_attention']],
        source: 'docs/02-sqlite-schema.md'
    },
    {
        title: '🧵 Find Primary Thread', intent: 'Which CPU thread launches the most kernels on a given GPU?',
        tables: ['CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_KERNEL'],
        columns: ['globalTid', 'correlationId', 'deviceId'],
        sql: `SELECT r.globalTid, COUNT(*) AS cnt\nFROM CUPTI_ACTIVITY_KIND_RUNTIME r\nJOIN CUPTI_ACTIVITY_KIND_KERNEL k ON r.correlationId = k.correlationId\nWHERE k.deviceId = 4\nGROUP BY r.globalTid ORDER BY cnt DESC LIMIT 1;`,
        output: [['globalTid', 'INT', 'CPU thread', '281474976841729'], ['cnt', 'INT', 'Launch count', '1847']],
        source: 'tree.py → _find_primary_thread()'
    },
    {
        title: '⏱️ Timeline Window', intent: 'Get kernels in a time range for rendering.',
        tables: ['CUPTI_ACTIVITY_KIND_KERNEL', 'StringIds'],
        columns: ['start', 'end', 'deviceId', 'streamId', 'shortName', 'demangledName', 'id', 'value', 'gridX', 'blockX'],
        sql: `SELECT k.start, k.end, k.deviceId, k.streamId,\n       sh.value AS short_name, s.value AS kernel_name\nFROM CUPTI_ACTIVITY_KIND_KERNEL k\nJOIN StringIds s ON k.demangledName = s.id\nJOIN StringIds sh ON k.shortName = sh.id\nWHERE k.start < ? AND k.end > ? AND k.deviceId = ?\nORDER BY k.start;`,
        output: [['short_name', 'TEXT', 'Short name', 'flash_attn_fwd'], ['kernel_name', 'TEXT', 'Full name', 'void flash::flash_fwd_kernel<…>()'], ['streamId', 'INT', 'Stream', '21']],
        source: 'analysis.js → getTimelineWindow()'
    },
    {
        title: '🕳️ Idle Gap Detection', intent: 'Find GPU idle periods (bubbles) between kernels.',
        tables: ['CUPTI_ACTIVITY_KIND_KERNEL'],
        columns: ['start', 'end', 'deviceId'],
        sql: `SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL\nWHERE deviceId = 4 ORDER BY start;\n-- In code: iterate, track max(end),\n-- if next.start > max_end = gap found`,
        output: [['gap_start', 'INT', 'Previous kernel end', '40006023456'], ['gap_end', 'INT', 'Next kernel start', '40006523456'], ['duration_us', 'TEXT', 'Gap size', '500.0']],
        source: 'analysis.js → detectIdleGaps()'
    },
    {
        title: '📡 NCCL Alignment', intent: 'Check if GPUs enter NCCL collectives at the same time.',
        tables: ['NVTX_EVENTS'],
        columns: ['start', 'end', 'text', 'globalTid'],
        sql: `SELECT start, end, text, globalTid\nFROM NVTX_EVENTS WHERE text LIKE 'nccl%' ORDER BY start;\n-- Group by op name + time proximity,\n-- compute skew = max(start) - min(start)`,
        output: [['op', 'TEXT', 'NCCL op', 'ncclAllReduce'], ['skewMs', 'TEXT', 'Time spread', '50.000'], ['latestGpu', 'INT', 'Straggler', '3']],
        source: 'analysis.js → analyzeNcclAlignment()'
    },
    {
        title: '🔎 Search Kernels', intent: 'Full-text search across kernel names using LIKE.',
        tables: ['CUPTI_ACTIVITY_KIND_KERNEL', 'StringIds'],
        columns: ['start', 'end', 'deviceId', 'streamId', 'demangledName', 'shortName', 'id', 'value'],
        sql: `SELECT k.start, k.end, k.deviceId, k.streamId, s.value AS name\nFROM CUPTI_ACTIVITY_KIND_KERNEL k\nJOIN StringIds s ON k.demangledName = s.id\nWHERE LOWER(s.value) LIKE '%flash%'\nORDER BY k.start LIMIT 100;`,
        output: [['name', 'TEXT', 'Kernel name', 'void flash::flash_fwd_kernel<…>()'], ['deviceId', 'INT', 'GPU', '4'], ['streamId', 'INT', 'Stream', '21']],
        source: 'search.py / server.js → /search'
    },
    {
        title: '🌿 Build NVTX Tree', intent: 'Build full NVTX call-stack tree for one GPU.',
        tables: ['CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_KERNEL', 'NVTX_EVENTS', 'StringIds'],
        columns: ['start', 'end', 'correlationId', 'globalTid', 'deviceId', 'streamId', 'text', 'shortName', 'id', 'value'],
        sql: `-- Step 1: primary thread\n-- Step 2: runtime calls for that thread\nSELECT start, end, correlationId\nFROM CUPTI_ACTIVITY_KIND_RUNTIME\nWHERE globalTid = ? AND start >= ? AND end <= ?\nORDER BY start;\n-- Step 3: NVTX for same thread\nSELECT text, start, end FROM NVTX_EVENTS\nWHERE globalTid = ? AND text IS NOT NULL\n  AND end > start ORDER BY start;`,
        output: [['name', 'TEXT', 'NVTX text or kernel name', 'TransformerLayer'], ['type', 'TEXT', 'nvtx or kernel', 'nvtx'], ['duration_ms', 'FLOAT', 'Duration', '43.5'], ['children', 'ARRAY', 'Sub-nodes', '[{…}]']],
        source: 'tree.py → build_nvtx_tree()'
    },
];

// ════════════════════════════════════════════════════
//  RENDERING
// ════════════════════════════════════════════════════
let selectedExample = null, hlMode = 0, activeColumn = null, activeColTable = null;
let colScope = 'global'; // 'global' or 'local'

function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

function colorizeSQL(sql) {
    // Extract alias-to-table map from raw SQL before HTML injection
    var aliasMap = {};
    for (var t of TABLES) {
        var re = new RegExp('(?:FROM|JOIN)\\s+' + t.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\s+(\\w{1,3})\\b', 'gi');
        var m; while ((m = re.exec(sql)) !== null) aliasMap[m[1]] = t.name;
    }
    var lines = sql.split('\n');
    return lines.map(function (line) {
        var cmtIdx = line.indexOf('--');
        var code = cmtIdx >= 0 ? line.slice(0, cmtIdx) : line;
        var comment = cmtIdx >= 0 ? line.slice(cmtIdx) : '';
        var out = colorizeCodePart(code, aliasMap);
        if (comment) out += '<span class="cmt">' + esc(comment) + '</span>';
        return out;
    }).join('\n');
}

function colorizeCodePart(s, aliasMap) {
    s = esc(s);
    // Table names
    for (var t of TABLES) {
        var re = new RegExp('\\b(' + t.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')\\b', 'g');
        s = s.replace(re, '<span class="tbl" data-table="' + t.name + '" style="color:' + gc(t.name) + ';background:' + gc(t.name) + '22" onmouseenter="hoverTable(\'' + t.name + '\')" onmouseleave="unhoverTable(\'' + t.name + '\')">' + t.name + '</span>');
    }
    // Column names
    var allCols = new Set();
    TABLES.forEach(function (t) { t.cols.forEach(function (c) { allCols.add(c[0]); }); });
    for (var col of allCols) {
        var re2 = new RegExp('(?<![\\w."])\\b(' + col + ')\\b(?![^<]*>)', 'g');
        s = s.replace(re2, '<span class="col" data-col="' + col + '" onclick="clickColumn(\'' + col + '\',event)" onmouseenter="hoverColumn(\'' + col + '\',event)" onmouseleave="unhoverColumn(\'' + col + '\',event)">' + '$1' + '</span>');
    }
    // Aliases - color matches their aliased table
    s = s.replace(/\b([a-z]{1,3})\.(?![^<]*>)/g, function (match, alias) {
        var tbl = aliasMap && aliasMap[alias];
        var color = tbl ? gc(tbl) : 'var(--text-muted)';
        return '<span style="color:' + color + '">' + alias + '.</span>';
    });
    // Keywords
    s = s.replace(/\b(SELECT|FROM|JOIN|LEFT|RIGHT|INNER|ON|WHERE|AND|OR|NOT|GROUP|BY|ORDER|ASC|DESC|LIMIT|AS|DISTINCT|COUNT|SUM|AVG|MIN|MAX|LIKE|IN|BETWEEN|IS|NULL|HAVING|WITH)\b(?![^<]*>)/gi, '<span class="kw">$1</span>');
    // Strings
    s = s.replace(/('(?:[^']|\\')*')(?![^<]*>)/g, '<span class="str">$1</span>');
    // Numbers
    s = s.replace(/\b(\d+(?:\.\d+)?(?:e\d+)?)\b(?![^<]*>)/g, '<span class="num">$1</span>');
    // Functions
    s = s.replace(/\b(ROUND|LOWER|UPPER|SUBSTR|LENGTH|COALESCE)\b(?![^<]*>)/gi, '<span class="fn">$1</span>');
    s = s.replace(/\?(?![^<]*>)/g, '<span class="num">?</span>');
    return s;
}

function renderTables() {
    document.getElementById('tableList').innerHTML = TABLES.map((t, i) => `
    <div class="tcard" data-table="${t.name}" id="tc-${i}">
      <span class="tname" style="color:${gc(t.name)};background:${gc(t.name)}22"
        onmouseenter="hoverTable('${t.name}')" onmouseleave="unhoverTable('${t.name}')">${t.name}</span>
      <span class="tbadge">${t.cat}</span>
      <div class="tdesc">${t.desc}</div>
      <table class="schema-tbl"><thead><tr><th>Column</th><th>Type</th><th>Description</th></tr></thead><tbody>
        ${t.cols.map(([n, ty, d]) => `<tr><td class="cn" data-col="${n}" onclick="clickColumn('${n}',event)" onmouseenter="hoverColumn('${n}',event)" onmouseleave="unhoverColumn('${n}',event)">${n}</td><td>${ty}</td><td>${d}</td></tr>`).join('')}
      </tbody></table>
    </div>`).join('');
    // Left sidebar index
    document.getElementById('leftIndex').innerHTML = TABLES.map((t, i) =>
        `<div class="sb-item" data-table="${t.name}" onclick="scrollToTable(${i})" onmouseenter="hoverTable('${t.name}')" onmouseleave="unhoverTable('${t.name}')"><span class="dot" style="background:${gc(t.name)}"></span>${t.name.replace('CUPTI_ACTIVITY_KIND_', '').replace('TARGET_INFO_', '')}</div>`
    ).join('');
}

function renderConcepts() {
    document.getElementById('tab-concepts').innerHTML = CONCEPTS.map(c =>
        `<div class="ccard"><div class="ctitle">${c.title}</div><div class="cbody">${c.body}</div></div>`).join('');
}

function renderExamples() {
    document.getElementById('tab-examples').innerHTML = EXAMPLES.map((ex, i) => `
    <div class="ecard" data-idx="${i}" id="ex-${i}" onclick="selectExample(${i})">
      <div class="etitle">${ex.title}</div>
      <div class="eintent">${ex.intent}</div>
      <div class="tags"><span style="color:var(--text-muted);font-size:10px;margin-right:3px">Tables:</span>
        ${ex.tables.map(t => `<span class="tag" style="color:${gc(t)};background:${gc(t)}22" onmouseenter="hoverTable('${t}')" onmouseleave="unhoverTable('${t}')">${t.replace('CUPTI_ACTIVITY_KIND_', '')}</span>`).join('')}
      </div>
      <div class="slabel">SQL Query</div>
      <div class="sql-block">${colorizeSQL(ex.sql)}</div>
      <div class="slabel">Output</div>
      <table class="otbl"><thead><tr><th>Column</th><th>Type</th><th>Meaning</th><th>Example</th></tr></thead><tbody>
        ${ex.output.map(([n, ty, d, ex2]) => `<tr><td>${n}</td><td>${ty}</td><td>${d}</td><td class="exr">${ex2}</td></tr>`).join('')}
      </tbody></table>
      <div style="font-size:10px;color:var(--text-muted);margin-top:4px">Source: <code style="background:var(--surface);padding:1px 3px;border-radius:3px;font-size:10px">${ex.source}</code></div>
    </div>`).join('');
    // Right sidebar index
    document.getElementById('rightIndex').innerHTML =
        '<h3 style="padding:10px 12px 4px;font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.5px">💡 Concepts</h3>' +
        CONCEPTS.map((c, i) => `<div class="sb-item" onclick="switchTab('concepts');document.querySelectorAll('.ccard')[${i}].scrollIntoView({behavior:'smooth'})">${c.title.slice(0, 30)}</div>`).join('') +
        '<h3 style="padding:10px 12px 4px;font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.5px">📝 Examples</h3>' +
        EXAMPLES.map((ex, i) => `<div class="sb-item" onclick="switchTab('examples');selectExample(${i});document.getElementById('ex-${i}').scrollIntoView({behavior:'smooth'})">${ex.title}</div>`).join('');
}

// ════════════════════════════════════════════════════
//  INTERACTIONS
// ════════════════════════════════════════════════════
function scrollToTable(i) {
    const el = document.getElementById('tc-' + i);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function selectExample(idx) {
    selectedExample = (selectedExample === idx) ? null : idx;
    document.querySelectorAll('.ecard').forEach((el, i) => el.classList.toggle('selected', i === selectedExample));
    if (hlMode) applyHighlight();
}

const HL_LABELS = ['🎨 Off', '🎨 Dim', '🎨 Focus'];
function toggleHighlight() {
    hlMode = (hlMode + 1) % 3;
    const btn = document.getElementById('hlBtn');
    btn.classList.toggle('active', hlMode > 0);
    btn.textContent = HL_LABELS[hlMode];
    applyHighlight();
}

function applyHighlight() {
    const cards = document.querySelectorAll('.tcard');
    if (!hlMode || selectedExample === null) {
        cards.forEach(c => { c.style.display = ''; c.classList.remove('dimmed', 'highlighted'); c.style.borderLeftColor = 'transparent'; c.querySelectorAll('.cn').forEach(td => td.classList.remove('active-col')); });
        return;
    }
    const used = new Set(EXAMPLES[selectedExample].tables);
    const usedC = new Set(EXAMPLES[selectedExample].columns || []);
    cards.forEach(card => {
        const n = card.dataset.table;
        if (used.has(n)) {
            card.style.display = ''; card.classList.remove('dimmed'); card.classList.add('highlighted'); card.style.borderLeftColor = gc(n);
            card.querySelectorAll('.cn').forEach(td => { td.classList.toggle('active-col', usedC.has(td.dataset.col)); });
        } else if (hlMode === 2) {
            card.style.display = 'none';
        } else {
            card.style.display = ''; card.classList.add('dimmed'); card.classList.remove('highlighted'); card.style.borderLeftColor = 'transparent'; card.querySelectorAll('.cn').forEach(td => td.classList.remove('active-col'));
        }
    });
}

function hoverTable(n) {
    document.querySelectorAll('[data-table="' + n + '"]').forEach(el => {
        if (el.classList.contains('tname') || el.classList.contains('tbl') || el.classList.contains('tag')) el.classList.add('active-hover');
        if (el.classList.contains('tcard')) el.classList.add('hovered-match');
        if (el.classList.contains('sb-item')) el.style.color = 'var(--accent)';
    });
}
function unhoverTable(n) {
    document.querySelectorAll('[data-table="' + n + '"]').forEach(el => { el.classList.remove('active-hover', 'hovered-match'); if (el.classList.contains('sb-item')) el.style.color = ''; });
}
function clickColumn(col, evt) {
    if (activeColumn === col) { activeColumn = null; activeColTable = null; document.querySelectorAll('.active-col').forEach(e => e.classList.remove('active-col')); return; }
    activeColumn = col; document.querySelectorAll('.active-col').forEach(e => e.classList.remove('active-col'));
    // Find parent table for local scope
    activeColTable = evt && evt.target ? evt.target.closest('.tcard, .ecard, .fk-card') : null;
    _highlightCol(col, activeColTable, true);
}
function hoverColumn(col, evt) {
    var parent = evt && evt.target ? evt.target.closest('.tcard, .ecard, .fk-card') : null;
    _highlightCol(col, parent, true);
}
function unhoverColumn(col, evt) {
    if (activeColumn !== col) {
        var parent = evt && evt.target ? evt.target.closest('.tcard, .ecard, .fk-card') : null;
        _highlightCol(col, parent, false);
    }
}
function _highlightCol(col, parentEl, on) {
    if (colScope === 'local' && parentEl) {
        // Only highlight within the same card
        parentEl.querySelectorAll('[data-col="' + col + '"]').forEach(e => e.classList.toggle('active-col', on));
    } else {
        // Global: highlight everywhere
        document.querySelectorAll('[data-col="' + col + '"]').forEach(e => e.classList.toggle('active-col', on));
    }
}
function toggleColScope() {
    colScope = colScope === 'global' ? 'local' : 'global';
    var btn = document.getElementById('colScopeBtn');
    btn.textContent = colScope === 'global' ? '🌐 Global' : '📌 Local';
    btn.classList.toggle('active', colScope === 'local');
    // Clear active column when switching scope
    activeColumn = null; activeColTable = null;
    document.querySelectorAll('.active-col').forEach(e => e.classList.remove('active-col'));
}

function filterTables() {
    const v = document.getElementById('tableFilter').value.trim();
    const cards = document.querySelectorAll('.tcard');
    if (!v) { cards.forEach(c => c.style.display = ''); return; }
    try { const re = new RegExp(v, 'i'); cards.forEach(c => { c.style.display = re.test(c.textContent) ? '' : 'none'; }); }
    catch (e) { const q = v.toLowerCase(); cards.forEach(c => { c.style.display = c.textContent.toLowerCase().includes(q) ? '' : 'none'; }); }
}

function switchTab(t) {
    document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x.dataset.tab === t));
    document.querySelectorAll('.tc').forEach(x => x.classList.toggle('active', x.id === 'tab-' + t));
}

// ════════════════════════════════════════════════════
//  FK RELATIONSHIPS
// ════════════════════════════════════════════════════
const FK_RELATIONSHIPS = [
    {
        group: '🔗 Kernel Launch Chain (the critical path)',
        rels: [
            { from: 'CUPTI_ACTIVITY_KIND_KERNEL', fromCol: 'correlationId', to: 'CUPTI_ACTIVITY_KIND_RUNTIME', toCol: 'correlationId', type: '1:1', desc: 'Links GPU kernel execution to the CPU-side launch call' },
            { from: 'CUPTI_ACTIVITY_KIND_RUNTIME', fromCol: 'globalTid', to: 'NVTX_EVENTS', toCol: 'globalTid', type: 'N:M', desc: 'Same CPU thread — NVTX ranges contain runtime calls by time overlap (not a strict FK, but the key join path)' },
            { from: 'CUPTI_ACTIVITY_KIND_KERNEL', fromCol: 'shortName', to: 'StringIds', toCol: 'id', type: 'N:1', desc: 'Resolves short kernel display name' },
            { from: 'CUPTI_ACTIVITY_KIND_KERNEL', fromCol: 'demangledName', to: 'StringIds', toCol: 'id', type: 'N:1', desc: 'Resolves full demangled C++ kernel name' },
            { from: 'CUPTI_ACTIVITY_KIND_RUNTIME', fromCol: 'nameId', to: 'StringIds', toCol: 'id', type: 'N:1', desc: 'Resolves CUDA API function name (e.g. "cudaLaunchKernel")' },
        ]
    },
    {
        group: '🖥️ Hardware Mapping',
        rels: [
            { from: 'CUPTI_ACTIVITY_KIND_KERNEL', fromCol: 'deviceId', to: 'TARGET_INFO_CUDA_DEVICE', toCol: 'cudaId', type: 'N:1', desc: 'Maps kernel\'s deviceId to CUDA device entry' },
            { from: 'TARGET_INFO_CUDA_DEVICE', fromCol: 'gpuId', to: 'TARGET_INFO_GPU', toCol: 'id', type: '1:1', desc: 'Maps CUDA device to physical GPU info (name, memory, SMs)' },
        ]
    },
    {
        group: '🧵 Thread & Process',
        rels: [
            { from: 'ThreadNames', fromCol: 'nameId', to: 'StringIds', toCol: 'id', type: 'N:1', desc: 'Resolves thread name' },
            { from: 'ProcessStreams', fromCol: 'filenameId', to: 'StringIds', toCol: 'id', type: 'N:1', desc: 'Resolves output filename' },
            { from: 'ProcessStreams', fromCol: 'contentId', to: 'StringIds', toCol: 'id', type: 'N:1', desc: 'Resolves captured output content' },
        ]
    },
    {
        group: '🔍 Debug / Backtraces',
        rels: [
            { from: 'CUDA_CALLCHAINS', fromCol: 'symbol', to: 'StringIds', toCol: 'id', type: 'N:1', desc: 'Resolves backtrace symbol name' },
            { from: 'OSRT_CALLCHAINS', fromCol: 'symbol', to: 'StringIds', toCol: 'id', type: 'N:1', desc: 'Resolves OS runtime backtrace symbol' },
            { from: 'OSRT_API', fromCol: 'nameId', to: 'StringIds', toCol: 'id', type: 'N:1', desc: 'Resolves OS API function name' },
        ]
    },
];

function renderRelationships() {
    const container = document.getElementById('tab-relationships');
    let html = '<div class="ccard"><div class="ctitle">Entity Relationships</div><div class="cbody"><p>Foreign key and join relationships between tables. Hover table names to highlight on the left panel. The <b>Kernel Launch Chain</b> is the most important — it connects GPU kernels back to CPU code through <code>correlationId</code> and <code>globalTid</code>.</p></div></div>';

    for (const group of FK_RELATIONSHIPS) {
        html += `<div class="fk-group-title">${group.group}</div>`;
        for (const r of group.rels) {
            html += `<div class="fk-card" onmouseenter="hoverTable('${r.from}');hoverTable('${r.to}')" onmouseleave="unhoverTable('${r.from}');unhoverTable('${r.to}')">
        <div class="fk-arrow">
          <span class="fk-tbl" data-table="${r.from}" style="color:${gc(r.from)};background:${gc(r.from)}22"
            onmouseenter="hoverTable('${r.from}')" onmouseleave="unhoverTable('${r.from}')">${r.from.replace('CUPTI_ACTIVITY_KIND_', '')}</span>
          <span class="fk-col" style="color:${gc(r.from)}">.${r.fromCol}</span>
          <span style="color:var(--text-muted)">→</span>
          <span class="fk-tbl" data-table="${r.to}" style="color:${gc(r.to)};background:${gc(r.to)}22"
            onmouseenter="hoverTable('${r.to}')" onmouseleave="unhoverTable('${r.to}')">${r.to}</span>
          <span class="fk-col" style="color:${gc(r.to)}">.${r.toCol}</span>
          <span class="fk-type">${r.type}</span>
        </div>
        <div class="fk-desc">${r.desc}</div>
      </div>`;
        }
    }
    container.innerHTML = html;

    // Also add relationships to right sidebar index
    const ri = document.getElementById('rightIndex');
    ri.innerHTML += '<h3 style="padding:10px 12px 4px;font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.5px">🔗 Relationships</h3>';
    ri.innerHTML += FK_RELATIONSHIPS.map((g, i) =>
        `<div class="sb-item" onclick="switchTab('relationships')">${g.group.slice(0, 28)}</div>`
    ).join('');
}

// Init
renderTables(); renderConcepts(); renderExamples(); renderRelationships();
