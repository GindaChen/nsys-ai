const BOOT = window.__TIMELINE_BOOTSTRAP__ || {};
const INITIAL_DATA = BOOT.INITIAL_DATA ?? null;
const PROGRESSIVE = BOOT.PROGRESSIVE === true;
const TILE_WINDOW_S = Number.isFinite(BOOT.TILE_WINDOW_S) ? BOOT.TILE_WINDOW_S : 5;  // seconds per tile
const GPU_INFO = Array.isArray(BOOT.GPU_INFO) ? BOOT.GPU_INFO : [];
const FINDINGS = Array.isArray(BOOT.FINDINGS) ? BOOT.FINDINGS : [];

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const wrap = document.getElementById('canvasWrap');

// ── Tile Cache ──
class TileCache {
    constructor(maxBytes = 100 * 1024 * 1024) {
        this.tiles = new Map();  // key "start_s-end_s" → {data, ts, sizeEst}
        this.maxBytes = maxBytes;
        this.currentBytes = 0;
        this.nvtxReady = new Set();
    }
    key(startS, endS) { return `${startS.toFixed(1)}-${endS.toFixed(1)}`; }
    has(startS, endS) { return this.tiles.has(this.key(startS, endS)); }
    nvtxKey(startS, endS, gpuId) { return `${this.key(startS, endS)}|gpu:${gpuId}`; }
    hasNvtx(startS, endS, gpuId) { return this.nvtxReady.has(this.nvtxKey(startS, endS, gpuId)); }
    markNvtx(startS, endS, gpuId) { this.nvtxReady.add(this.nvtxKey(startS, endS, gpuId)); }
    get(startS, endS) {
        const k = this.key(startS, endS);
        const entry = this.tiles.get(k);
        if (entry) entry.ts = Date.now();
        return entry ? entry.data : null;
    }
    put(startS, endS, data) {
        const k = this.key(startS, endS);
        const sizeEst = JSON.stringify(data).length * 2;  // rough byte estimate
        // No eviction — tiles are small with pre-built server cache
        this.tiles.set(k, { data, ts: Date.now(), sizeEst });
        this.currentBytes += sizeEst;
    }
    mergeNvtx(startS, endS, nvtxData, gpuId) {
        const k = this.key(startS, endS);
        const entry = this.tiles.get(k);
        if (!entry || !entry.data || !entry.data.gpus || !nvtxData || !nvtxData.gpus) return;
        let mergedTargetGpu = false;
        for (const ng of nvtxData.gpus) {
            if (ng.id !== gpuId) continue;
            const eg = entry.data.gpus.find(g => g.id === ng.id);
            if (!eg) continue;
            if (Array.isArray(ng.nvtx_spans)) eg.nvtx_spans = ng.nvtx_spans;
            mergedTargetGpu = true;
        }
        if (mergedTargetGpu) {
            this.markNvtx(startS, endS, gpuId);
        }
    }
    // Get all cached data merged
    allData() {
        const merged = { gpus: [] };
        const gpuMap = {};
        for (const [, entry] of this.tiles) {
            if (!entry.data || !entry.data.gpus) continue;
            for (const gpu of entry.data.gpus) {
                if (gpu.kernels) {
                    if (gpuMap[gpu.id] === undefined) {
                        gpuMap[gpu.id] = merged.gpus.length;
                        merged.gpus.push({ id: gpu.id, kernels: [], nvtx_spans: [] });
                    }
                    const idx = gpuMap[gpu.id];
                    if (gpu.kernels) {
                        for (let k = 0; k < gpu.kernels.length; k++) {
                            merged.gpus[idx].kernels.push(gpu.kernels[k]);
                        }
                    }
                    if (gpu.nvtx_spans) {
                        for (let s = 0; s < gpu.nvtx_spans.length; s++) {
                            merged.gpus[idx].nvtx_spans.push(gpu.nvtx_spans[s]);
                        }
                    }
                } else {
                    if (gpuMap[gpu.id] === undefined) {
                        gpuMap[gpu.id] = merged.gpus.length;
                        merged.gpus.push({ id: gpu.id, data: [] });
                    }
                    const idx = gpuMap[gpu.id];
                    if (gpu.data) {
                        for (let d = 0; d < gpu.data.length; d++) {
                            merged.gpus[idx].data.push(gpu.data[d]);
                        }
                    }
                }
            }
        }
        // De-duplicate entries that appear in adjacent tiles.
        for (const g of merged.gpus) {
            if (!g.kernels) continue;
            const kSeen = new Set();
            g.kernels = g.kernels.filter(k => {
                const key = `${k.start_ns}|${k.end_ns}|${k.stream}|${k.name}`;
                if (kSeen.has(key)) return false;
                kSeen.add(key);
                return true;
            });
            const nSeen = new Set();
            g.nvtx_spans = g.nvtx_spans.filter(s => {
                const key = `${s.start}|${s.end}|${s.depth}|${s.path}|${s.thread || ''}`;
                if (nSeen.has(key)) return false;
                nSeen.add(key);
                return true;
            });
        }
        return merged;
    }
}
const tileCache = new TileCache();
const nvtxInflight = new Set();
let profileMeta = null;  // {time_range_ns, gpus, device_ids}
let isLoading = false;

// ── Loading overlay ──
function showLoading(msg) {
    let overlay = document.getElementById('loadingOverlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'loadingOverlay';
        overlay.style.cssText = 'position:absolute;top:0;left:0;right:0;bottom:0;display:flex;align-items:center;justify-content:center;background:rgba(13,17,23,0.8);color:#79b8ff;font-size:14px;z-index:100;pointer-events:none;';
        wrap.style.position = 'relative';
        wrap.appendChild(overlay);
    }
    overlay.textContent = msg || 'Loading…';
    overlay.style.display = 'flex';
}
function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.style.display = 'none';
}

// ── Fetch data for a time window ──
async function fetchTile(startS, endS, { requestNvtx = false } = {}) {
    if (tileCache.has(startS, endS)) return tileCache.get(startS, endS);
    const pfx = BOOT.API_PREFIX || '';
    const resp = await fetch(`${pfx}/api/data?start_s=${startS}&end_s=${endS}&kernels=1&nvtx=0`);
    const data = await resp.json();
    if (data.error) { console.error('fetchTile error:', data.error); return null; }
    tileCache.put(startS, endS, data);
    if (requestNvtx && showNVTX) void fetchTileNvtx(startS, endS, currentActiveGpuId());
    return data;
}

function currentActiveGpuId() {
    if (isMultiGPU && streamIds[selectedStreamIdx]) {
        const parts = String(streamIds[selectedStreamIdx]).split(':');
        const maybeGpu = parseInt(parts[0], 10);
        if (!isNaN(maybeGpu)) return maybeGpu;
    }
    if (gpuIds.length > 0) return gpuIds[0];
    if (profileMeta && Array.isArray(profileMeta.device_ids) && profileMeta.device_ids.length > 0) {
        return profileMeta.device_ids[0];
    }
    return null;
}

async function fetchTileNvtx(startS, endS, gpuId) {
    const key = `${tileCache.key(startS, endS)}|gpu:${gpuId}`;
    if (tileCache.hasNvtx(startS, endS, gpuId) || nvtxInflight.has(key)) return;
    nvtxInflight.add(key);
    updateNvtxLoadingIndicator();
    try {
        const pfx = BOOT.API_PREFIX || '';
        const resp = await fetch(`${pfx}/api/data?start_s=${startS}&end_s=${endS}&kernels=0&nvtx=1&gpu=${gpuId}`);
        const data = await resp.json();
        if (data.error) { console.error('fetchTileNvtx error:', data.error); return; }
        tileCache.mergeNvtx(startS, endS, data, gpuId);
        rebuildDataFromCache();
        draw();
    } finally {
        nvtxInflight.delete(key);
        updateNvtxLoadingIndicator();
    }
}

function updateNvtxLoadingIndicator() {
    const el = document.getElementById('nvtxLoading');
    const text = document.getElementById('nvtxLoadingText');
    if (!el || !text) return;
    const active = showNVTX && nvtxInflight.size > 0;
    el.style.display = active ? 'inline-flex' : 'none';
    if (active) {
        text.textContent = `NVTX loading… (${nvtxInflight.size})`;
    }
}

// ── Quantize view to tile boundaries ──
function tileBounds(ns) {
    const s = ns / 1e9;
    const startS = Math.floor(s / TILE_WINDOW_S) * TILE_WINDOW_S;
    return [startS, startS + TILE_WINDOW_S];
}

// ── Load tiles covering the current viewport ──
async function ensureTilesForView(vStart, vEnd) {
    const startTile = tileBounds(vStart);
    const endTile = tileBounds(vEnd);
    const promises = [];
    for (let s = startTile[0]; s < endTile[1]; s += TILE_WINDOW_S) {
        if (!renderLockIntersects(s, s + TILE_WINDOW_S)) continue;
        if (!tileCache.has(s, s + TILE_WINDOW_S)) {
            promises.push(fetchTile(s, s + TILE_WINDOW_S));
        }
    }
    if (promises.length > 0) {
        isLoading = true;
        showLoading(`Loading ${promises.length} tile${promises.length > 1 ? 's' : ''}…`);
        await Promise.all(promises);
        isLoading = false;
        hideLoading();
        rebuildDataFromCache();
    }
    if (showNVTX) {
        maybeFetchNvtxForCurrentView();
    }
}

function maybeFetchNvtxForCurrentView() {
    if (!PROGRESSIVE || !showNVTX) return;
    const startTile = tileBounds(viewStart);
    const endTile = tileBounds(viewEnd);
    const activeGpu = currentActiveGpuId();
    if (activeGpu === null || activeGpu === undefined) return;
    for (let s = startTile[0]; s < endTile[1]; s += TILE_WINDOW_S) {
        if (!renderLockIntersects(s, s + TILE_WINDOW_S)) continue;
        if (tileCache.has(s, s + TILE_WINDOW_S)) {
            void fetchTileNvtx(s, s + TILE_WINDOW_S, activeGpu);
        }
    }
}

// ── Background prefetch ±1 tile on idle ──
function prefetchAdjacent() {
    if (!PROGRESSIVE || typeof requestIdleCallback === 'undefined') return;
    requestIdleCallback(() => {
        const startTile = tileBounds(viewStart);
        const endTile = tileBounds(viewEnd);
        // Prefetch one tile before and after
        const before = [startTile[0] - TILE_WINDOW_S, startTile[0]];
        const after = [endTile[1], endTile[1] + TILE_WINDOW_S];
        if (profileMeta) {
            const pStartS = profileMeta.time_range_ns[0] / 1e9;
            const pEndS = profileMeta.time_range_ns[1] / 1e9;
            if (before[0] >= pStartS && renderLockIntersects(before[0], before[1]) && !tileCache.has(before[0], before[1])) {
                fetchTile(before[0], before[1], { requestNvtx: false }).then(() => { rebuildDataFromCache(); draw(); });
            }
            if (after[1] <= pEndS && renderLockIntersects(after[0], after[1]) && !tileCache.has(after[0], after[1])) {
                fetchTile(after[0], after[1], { requestNvtx: false }).then(() => { rebuildDataFromCache(); draw(); });
            }
        }
    });
}

// ── Data extraction (multi-GPU aware) ──
let allNodes = []; let kernels = []; let nvtxSpans = [];
let gpuIds = [];
let isMultiGPU = false;

function extractData(nodes, path, depth, gpuId) {
    for (const n of nodes) {
        const np = path ? path + ' > ' + n.name : n.name;
        n._path = np; n._depth = depth; n._gpu = gpuId;
        allNodes.push(n);
        if (n.type === 'kernel') { n._path = np; kernels.push(n); }
        else if (n.type === 'nvtx' && n.start_ns && n.end_ns) {
            nvtxSpans.push({ name: n.name, start: n.start_ns, end: n.end_ns, depth, path: np, dur: n.duration_ms, gpu: gpuId });
        }
        if (n.children) extractData(n.children, np, depth + 1, gpuId);
    }
}

// Handle kernel-first format: {gpus: [{id, kernels, nvtx_spans}, ...]}
// and legacy tree format: {gpus: [{id, data}, ...]} / [node, ...]
function loadFromData(DATA) {
    if (!DATA) return;
    if (DATA.gpus) {
        for (const gpuEntry of DATA.gpus) {
            if (!gpuIds.includes(gpuEntry.id)) gpuIds.push(gpuEntry.id);
            if (gpuEntry.kernels) {
                for (const k of gpuEntry.kernels) {
                    const kk = { ...k, _gpu: gpuEntry.id, _depth: 0, _path: k.path || k.name || '' };
                    allNodes.push(kk);
                    kernels.push(kk);
                }
                for (const s of (gpuEntry.nvtx_spans || [])) {
                    nvtxSpans.push({
                        name: s.name,
                        start: s.start,
                        end: s.end,
                        depth: s.depth || 0,
                        path: s.path || '',
                        dur: s.dur || 0,
                        thread: s.thread || '(unnamed)',
                        gpu: gpuEntry.id
                    });
                }
            } else {
                extractData(gpuEntry.data, '', 0, gpuEntry.id);
            }
        }
    } else if (Array.isArray(DATA)) {
        if (!gpuIds.includes(0)) gpuIds.push(0);
        extractData(DATA, '', 0, 0);
    }
    isMultiGPU = gpuIds.length > 1;
}

// Streams — rebuilt with each data load
let streamMap = {};
let streamIds = [];
let gpuBands = [];
let timeStart = 0, timeEnd = 1, timeSpan = 1;
let nvtxMaxDepth = 0;
let nvtxDepthByGpu = new Map();

function rebuildDataFromCache() {
    // Clear and rebuild from all cached tiles (progressive) or from baked data
    allNodes = []; kernels = []; nvtxSpans = [];
    gpuIds = []; isMultiGPU = false;

    if (PROGRESSIVE) {
        loadFromData(tileCache.allData());
    } else {
        loadFromData(INITIAL_DATA);
    }
    kernels.sort((a, b) => a.start_ns - b.start_ns);

    // Rebuild stream map
    streamMap = {};
    kernels.forEach(k => {
        const rawStream = k.stream !== undefined ? String(k.stream) : '?';
        const key = isMultiGPU ? `${k._gpu}:${rawStream}` : rawStream;
        k._streamKey = key;
        if (!streamMap[key]) streamMap[key] = [];
        streamMap[key].push(k);
    });

    // Build ordered stream list: group by GPU
    streamIds = [];
    gpuBands = [];
    for (const gpuId of gpuIds) {
        const startIdx = streamIds.length;
        const gpuStreams = Object.keys(streamMap)
            .filter(k => isMultiGPU ? k.startsWith(gpuId + ':') : true)
            .sort((a, b) => {
                const aNum = parseInt(a.split(':').pop());
                const bNum = parseInt(b.split(':').pop());
                if (!isNaN(aNum) && !isNaN(bNum)) return aNum - bNum;
                return a.localeCompare(b);
            });
        for (const s of gpuStreams) {
            if (!streamIds.includes(s)) streamIds.push(s);
        }
        gpuBands.push({ gpuId, startIdx, endIdx: streamIds.length });
    }
    Object.values(streamMap).forEach(ks => ks.sort((a, b) => a.start_ns - b.start_ns));

    // Time bounds
    timeStart = kernels.length ? Math.min(...kernels.map(k => k.start_ns), ...nvtxSpans.map(s => s.start)) : 0;
    timeEnd = kernels.length ? Math.max(...kernels.map(k => k.end_ns), ...nvtxSpans.map(s => s.end)) : 1;
    timeSpan = Math.max(timeEnd - timeStart, 1);

    // NVTX depth
    nvtxMaxDepth = nvtxSpans.length ? Math.min(6, Math.max(...nvtxSpans.map(s => s.depth)) + 1) : 0;
    nvtxDepthByGpu = new Map();
    for (const s of nvtxSpans) {
        const gpu = s.gpu;
        const d = Math.min(6, (s.depth || 0) + 1);
        const prev = nvtxDepthByGpu.get(gpu) || 0;
        if (d > prev) nvtxDepthByGpu.set(gpu, d);
    }

    // Stats
    const totalKernels = kernels.length;
    const totalNVTX = nvtxSpans.length;
    const totalMs = kernels.reduce((a, k) => a + k.duration_ms, 0);
    const gpuCountLabel = gpuIds.length > 1 ? `GPUs: <strong>${gpuIds.length}</strong> &nbsp; ` : '';
    document.getElementById('stats').innerHTML =
        gpuCountLabel +
        (PROGRESSIVE ? `<span style="color:#238636">● Progressive</span> &nbsp; ` : '') +
        `<span>Kernels: <strong>${totalKernels}</strong></span> &nbsp; ` +
        `<span>NVTX: <strong>${totalNVTX}</strong></span> &nbsp; ` +
        `<span>GPU time: <strong>${totalMs.toFixed(1)}ms</strong></span>`;
    ensureRenderLockDefaults();
    updateNvtxThreadOptions();
}

// ── Initialization ──
async function initData() {
    if (INITIAL_DATA !== null) {
        // Baked-in data mode (--trim was specified)
        rebuildDataFromCache();
        viewStart = timeStart - timeSpan * 0.02;
        viewEnd = timeEnd + timeSpan * 0.02;
        resize();
        return;
    }
    // Progressive mode: fetch metadata, then first window
    showLoading('Loading profile metadata…');
    try {
        const pfx = BOOT.API_PREFIX || '';
        const metaResp = await fetch(`${pfx}/api/meta`);
        profileMeta = await metaResp.json();
        // Check for saved viewport
        let restored = false;
        try {
            const saved = JSON.parse(localStorage.getItem(_viewKey));
            if (saved && saved.s !== undefined && saved.e !== undefined) {
                // Set view first so NVTX fetch targets the correct viewport/GPU.
                viewStart = saved.s;
                viewEnd = saved.e;
                await ensureTilesForView(viewStart, viewEnd);
                rebuildDataFromCache();
                restored = true;
            }
        } catch (e) { }
        if (!restored) {
            // Default view: first 5 seconds
            const firstEnd = TILE_WINDOW_S;
            viewStart = 0;
            viewEnd = firstEnd * 1e9;
            await ensureTilesForView(viewStart, viewEnd);
            rebuildDataFromCache();
        }
        hideLoading();
        resize();
        if (showNVTX) maybeFetchNvtxForCurrentView();
        prefetchAdjacent();
    } catch (e) {
        hideLoading();
        console.error('initData failed:', e);
        document.getElementById('detail').textContent = 'Failed to load profile: ' + e.message;
    }
}

// ── State ──
let viewStart = 0;
let viewEnd = 1;
let selectedKernel = null;
let selectedNvtx = null;
let selectedStreamIdx = 0;
let showNVTX = true;
let selectedNvtxThread = 'auto';
let searchQuery = '';
let searchKernelMatches = new Set();
let searchNvtxMatches = new Set();
let isDragging = false, dragStartX = 0, dragViewStart = 0, dragViewEnd = 0;
let isSelecting = false, selectStartX = 0, selectStartNs = 0, selectEndNs = 0;
let isResizingDetail = false, detailResizeStartY = 0, detailResizeStartH = 0;
let bookmarks = [];
let hiddenStreams = new Set();  // stream keys to hide
const RENDER_SETTINGS_KEY = 'timeline-render-settings-v1';
const RENDER_LOCK_KEY = 'timeline-render-lock-v1';
const DEFAULT_RENDER_SETTINGS = Object.freeze({
    kernelSearchNonMatchAlpha: 0.16,
    kernelWhenNvtxSelectedAlpha: 0.08,
    nvtxWhenKernelSelectedAlpha: 0.08,
    nvtxNonSelectedAlpha: 0.08,
    nvtxSelectedAlpha: 0.95,
    hierarchyLayout: 'horizontal',
    rulerLabelMode: 'absolute',
});
const DEFAULT_RENDER_LOCK = Object.freeze({
    enabled: false,
    startNs: 0,
    endNs: 0,
});
let renderSettings = loadRenderSettings();
let renderLock = loadRenderLock();
let fitZoomState = null;

// ── Layout constants ──
const LABEL_W = isMultiGPU ? 110 : 90;
const RULER_H = 24;
const FINDINGS_LANE_H = 28;  // dedicated annotation lane for AI findings
const NVTX_ROW_H = 20;
const NVTX_PIN_ROWS = 5;
const STREAM_H = 32;
const STREAM_GAP = 2;
const GPU_SEP_H = isMultiGPU ? 22 : 0;
const MIN_BLOCK_W = 2;
const DPR = window.devicePixelRatio || 1;

// Height of the findings annotation lane (0 when no findings loaded)
function findingsLaneH() { return FINDINGS.length > 0 ? FINDINGS_LANE_H : 0; }

function clampNum(v, minV, maxV, fallback) {
    const n = Number(v);
    if (!Number.isFinite(n)) return fallback;
    return Math.min(maxV, Math.max(minV, n));
}

function loadRenderSettings() {
    try {
        const raw = localStorage.getItem(RENDER_SETTINGS_KEY);
        if (!raw) return { ...DEFAULT_RENDER_SETTINGS };
        const parsed = JSON.parse(raw);
        return {
            kernelSearchNonMatchAlpha: clampNum(
                parsed.kernelSearchNonMatchAlpha, 0.03, 0.9, DEFAULT_RENDER_SETTINGS.kernelSearchNonMatchAlpha
            ),
            kernelWhenNvtxSelectedAlpha: clampNum(
                parsed.kernelWhenNvtxSelectedAlpha, 0.03, 0.9, DEFAULT_RENDER_SETTINGS.kernelWhenNvtxSelectedAlpha
            ),
            nvtxWhenKernelSelectedAlpha: clampNum(
                parsed.nvtxWhenKernelSelectedAlpha, 0.03, 0.9, DEFAULT_RENDER_SETTINGS.nvtxWhenKernelSelectedAlpha
            ),
            nvtxNonSelectedAlpha: clampNum(
                parsed.nvtxNonSelectedAlpha, 0.03, 0.95, DEFAULT_RENDER_SETTINGS.nvtxNonSelectedAlpha
            ),
            nvtxSelectedAlpha: clampNum(
                parsed.nvtxSelectedAlpha, 0.1, 1.0, DEFAULT_RENDER_SETTINGS.nvtxSelectedAlpha
            ),
            hierarchyLayout: parsed.hierarchyLayout === 'vertical' ? 'vertical' : 'horizontal',
            rulerLabelMode: parsed.rulerLabelMode === 'anchored' ? 'anchored' : 'absolute',
        };
    } catch (e) {
        return { ...DEFAULT_RENDER_SETTINGS };
    }
}

function saveRenderSettings() {
    try {
        localStorage.setItem(RENDER_SETTINGS_KEY, JSON.stringify(renderSettings));
    } catch (e) { }
}

function loadRenderLock() {
    try {
        const raw = localStorage.getItem(RENDER_LOCK_KEY);
        if (!raw) return { ...DEFAULT_RENDER_LOCK };
        const parsed = JSON.parse(raw);
        const startNs = clampNum(parsed.startNs, 0, Number.MAX_SAFE_INTEGER, 0);
        const endNs = clampNum(parsed.endNs, 0, Number.MAX_SAFE_INTEGER, 0);
        return {
            enabled: parsed.enabled === true && endNs > startNs,
            startNs,
            endNs,
        };
    } catch (e) {
        return { ...DEFAULT_RENDER_LOCK };
    }
}

function saveRenderLock() {
    try {
        localStorage.setItem(RENDER_LOCK_KEY, JSON.stringify(renderLock));
    } catch (e) { }
}

function renderLockIntersects(startS, endS) {
    if (!renderLock.enabled) return true;
    const lockStartS = renderLock.startNs / 1e9;
    const lockEndS = renderLock.endNs / 1e9;
    return endS > lockStartS && startS < lockEndS;
}

function profileMaxNs() {
    if (profileMeta && Array.isArray(profileMeta.time_range_ns) && Number.isFinite(profileMeta.time_range_ns[1])) {
        return profileMeta.time_range_ns[1];
    }
    return Number.isFinite(timeEnd) ? timeEnd : 0;
}

function ensureRenderLockDefaults() {
    const maxNs = profileMaxNs();
    if (!Number.isFinite(maxNs) || maxNs <= 0) return;
    let changed = false;
    if (renderLock.startNs < 0) {
        renderLock.startNs = 0;
        changed = true;
    }
    if (renderLock.endNs <= 0) {
        renderLock.endNs = maxNs;
        changed = true;
    }
    if (renderLock.endNs > maxNs) {
        renderLock.endNs = maxNs;
        changed = true;
    }
    if (renderLock.enabled && renderLock.endNs <= renderLock.startNs) {
        renderLock.enabled = false;
        changed = true;
    }
    if (changed) saveRenderLock();
}

function syncSettingsPanel() {
    const bind = (inputId, valueId, key) => {
        const inp = document.getElementById(inputId);
        const val = document.getElementById(valueId);
        if (!inp || !val) return;
        inp.value = String(renderSettings[key]);
        val.textContent = Number(renderSettings[key]).toFixed(2);
    };
    bind('setKernelSearchDim', 'setKernelSearchDimVal', 'kernelSearchNonMatchAlpha');
    bind('setKernelWhenNvtxDim', 'setKernelWhenNvtxDimVal', 'kernelWhenNvtxSelectedAlpha');
    bind('setNvtxWhenKernelDim', 'setNvtxWhenKernelDimVal', 'nvtxWhenKernelSelectedAlpha');
    bind('setNvtxNonSelected', 'setNvtxNonSelectedVal', 'nvtxNonSelectedAlpha');
    bind('setNvtxSelected', 'setNvtxSelectedVal', 'nvtxSelectedAlpha');
    const hierarchySel = document.getElementById('setHierarchyLayout');
    if (hierarchySel) hierarchySel.value = renderSettings.hierarchyLayout || 'horizontal';
    const rulerSel = document.getElementById('setRulerLabelMode');
    if (rulerSel) rulerSel.value = renderSettings.rulerLabelMode || 'absolute';
    ensureRenderLockDefaults();
    const lockEnabled = document.getElementById('setRenderLockEnabled');
    const lockStart = document.getElementById('setRenderLockStart');
    const lockEnd = document.getElementById('setRenderLockEnd');
    const lockState = document.getElementById('setRenderLockState');
    if (lockEnabled) lockEnabled.checked = renderLock.enabled;
    if (lockStart) lockStart.value = (renderLock.startNs / 1e9).toFixed(3);
    if (lockEnd) lockEnd.value = (renderLock.endNs / 1e9).toFixed(3);
    if (lockState) lockState.textContent = renderLock.enabled ? 'on' : 'off';
}

function wireSettingsPanel() {
    const bind = (inputId, valueId, key, minV, maxV) => {
        const inp = document.getElementById(inputId);
        const val = document.getElementById(valueId);
        if (!inp || !val) return;
        inp.addEventListener('input', () => {
            renderSettings[key] = clampNum(inp.value, minV, maxV, renderSettings[key]);
            val.textContent = Number(renderSettings[key]).toFixed(2);
            saveRenderSettings();
            draw();
        });
    };
    bind('setKernelSearchDim', 'setKernelSearchDimVal', 'kernelSearchNonMatchAlpha', 0.03, 0.9);
    bind('setKernelWhenNvtxDim', 'setKernelWhenNvtxDimVal', 'kernelWhenNvtxSelectedAlpha', 0.03, 0.9);
    bind('setNvtxWhenKernelDim', 'setNvtxWhenKernelDimVal', 'nvtxWhenKernelSelectedAlpha', 0.03, 0.9);
    bind('setNvtxNonSelected', 'setNvtxNonSelectedVal', 'nvtxNonSelectedAlpha', 0.03, 0.95);
    bind('setNvtxSelected', 'setNvtxSelectedVal', 'nvtxSelectedAlpha', 0.1, 1.0);
    const hierarchySel = document.getElementById('setHierarchyLayout');
    if (hierarchySel) {
        hierarchySel.addEventListener('change', () => {
            renderSettings.hierarchyLayout = hierarchySel.value === 'vertical' ? 'vertical' : 'horizontal';
            saveRenderSettings();
            if (selectedNvtx) showDetail(selectedNvtx);
            else if (selectedKernel) showDetail(selectedKernel);
        });
    }
    const rulerSel = document.getElementById('setRulerLabelMode');
    if (rulerSel) {
        rulerSel.addEventListener('change', () => {
            renderSettings.rulerLabelMode = rulerSel.value === 'anchored' ? 'anchored' : 'absolute';
            saveRenderSettings();
            draw();
        });
    }
    syncSettingsPanel();
}

function toggleSettingsMenu() {
    const p = document.getElementById('settingsPanel');
    if (!p) return;
    const opening = p.style.display === 'none';
    p.style.display = opening ? 'block' : 'none';
    if (opening) syncSettingsPanel();
}

function resetRenderSettings() {
    renderSettings = { ...DEFAULT_RENDER_SETTINGS };
    saveRenderSettings();
    syncSettingsPanel();
    if (selectedNvtx) showDetail(selectedNvtx);
    else if (selectedKernel) showDetail(selectedKernel);
    draw();
}

function applyRenderLockSettings() {
    const lockEnabled = document.getElementById('setRenderLockEnabled');
    const lockStart = document.getElementById('setRenderLockStart');
    const lockEnd = document.getElementById('setRenderLockEnd');
    const enabled = !!(lockEnabled && lockEnabled.checked);
    const startS = Number(lockStart ? lockStart.value : NaN);
    const endS = Number(lockEnd ? lockEnd.value : NaN);
    if (enabled) {
        if (!Number.isFinite(startS) || !Number.isFinite(endS) || endS <= startS) {
            showToast('Invalid lock range: end must be > start');
            return;
        }
        renderLock = {
            enabled: true,
            startNs: Math.max(0, Math.round(startS * 1e9)),
            endNs: Math.max(0, Math.round(endS * 1e9)),
        };
    } else {
        renderLock = {
            enabled: false,
            startNs: Math.max(0, Number.isFinite(startS) ? Math.round(startS * 1e9) : 0),
            endNs: Math.max(0, Number.isFinite(endS) ? Math.round(endS * 1e9) : 0),
        };
    }
    saveRenderLock();
    syncSettingsPanel();
    showToast(renderLock.enabled
        ? `Render lock applied: ${(renderLock.startNs / 1e9).toFixed(3)}s–${(renderLock.endNs / 1e9).toFixed(3)}s`
        : 'Render lock disabled');
    if (PROGRESSIVE) {
        void ensureTilesForView(viewStart, viewEnd);
    }
    draw();
}

// ── Colors (vivid so bars stand out on dark background; background unchanged) ──
const STREAM_COLORS = ['#3fb950', '#58a6ff', '#bc8cff', '#ff7b7b', '#d4a72c', '#79b8ff', '#56d364'];
const NVTX_COLORS = ['#5b8dc9', '#8b7ab8', '#3d9b6e', '#c76b7a', '#b89b4d', '#6b9bc4'];
const GPU_SEP_COLORS = ['#58a6ff', '#bc8cff', '#3fb950', '#ff7b7b'];

function streamColor(idx) { return STREAM_COLORS[idx % STREAM_COLORS.length]; }
function nvtxColor(depth) { return NVTX_COLORS[depth % NVTX_COLORS.length]; }

// ── Formatting ──
function fmtDur(ms) { return ms >= 1 ? ms.toFixed(2) + 'ms' : (ms * 1000).toFixed(0) + 'μs'; }
function fmtNs(ns) { return (ns / 1e6).toFixed(3) + 'ms'; }
function fmtTimeS(ns) { return (ns / 1e9).toFixed(3) + 's'; }
function nvtxKey(span) { return `${span.gpu}|${span.start}|${span.end}|${span.name}|${span.thread || ''}`; }

function activeGpuNvtxSpans() {
    const activeGpu = currentActiveGpuId();
    const spans = isMultiGPU ? nvtxSpans.filter(s => s.gpu === activeGpu) : nvtxSpans.slice();
    return { activeGpu, spans };
}

function dominantNvtxThread(spans) {
    const counts = new Map();
    for (const s of spans) {
        const t = s.thread || '(unnamed)';
        counts.set(t, (counts.get(t) || 0) + 1);
    }
    let best = null;
    let bestCount = -1;
    for (const [t, c] of counts.entries()) {
        if (c > bestCount) {
            bestCount = c;
            best = t;
        }
    }
    return best;
}

function activeNvtxSpans() {
    const { activeGpu, spans } = activeGpuNvtxSpans();
    if (!spans.length) return { activeGpu, thread: null, spans: [] };
    const thread = selectedNvtxThread === 'auto' ? dominantNvtxThread(spans) : selectedNvtxThread;
    const filtered = thread ? spans.filter(s => (s.thread || '(unnamed)') === thread) : spans;
    return { activeGpu, thread, spans: filtered };
}

function layoutNvtxSpans(spans, maxRows = NVTX_PIN_ROWS, forceKey = null) {
    const ordered = [...spans].sort((a, b) => (a.start - b.start) || (a.depth - b.depth) || (a.end - b.end));
    const laidOut = [];
    let clipped = 0;
    for (const span of ordered) {
        const depth = Number.isFinite(span.depth) ? Math.max(0, Math.floor(span.depth)) : 0;
        const key = nvtxKey(span);
        const isForced = forceKey && forceKey === key;
        if (depth >= maxRows && !isForced) {
            clipped += 1;
            continue;
        }
        const lane = depth >= maxRows ? (maxRows - 1) : depth;
        laidOut.push({ ...span, _lane: lane, _key: key, _clipped: depth >= maxRows });
    }
    return { spans: laidOut, clipped };
}

function activeNvtxLayout() {
    const { activeGpu, thread, spans } = activeNvtxSpans();
    const forceKey = selectedNvtx ? selectedNvtx.key : null;
    const laid = layoutNvtxSpans(spans, NVTX_PIN_ROWS, forceKey);
    return { activeGpu, thread, spans: laid.spans, clipped: laid.clipped };
}

function nvtxLoadingForGpu(gpuId) {
    if (gpuId === null || gpuId === undefined) return false;
    const suffix = `|gpu:${gpuId}`;
    for (const key of nvtxInflight) {
        if (key.endsWith(suffix)) return true;
    }
    return false;
}

function activeNvtxMaxDepth() {
    const { activeGpu } = activeGpuNvtxSpans();
    if (!showNVTX) return 0;
    if (nvtxLoadingForGpu(activeGpu)) return NVTX_PIN_ROWS;
    return NVTX_PIN_ROWS;
}

function updateNvtxThreadOptions() {
    const sel = document.getElementById('nvtxThreadSel');
    if (!sel) return;
    const { spans } = activeGpuNvtxSpans();
    const counts = new Map();
    for (const s of spans) {
        const t = s.thread || '(unnamed)';
        counts.set(t, (counts.get(t) || 0) + 1);
    }
    const options = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    const current = selectedNvtxThread;
    sel.innerHTML = '';
    const autoOpt = document.createElement('option');
    autoOpt.value = 'auto';
    autoOpt.textContent = 'NVTX: Auto thread';
    sel.appendChild(autoOpt);
    for (const [t, c] of options) {
        const opt = document.createElement('option');
        opt.value = t;
        opt.textContent = `${t} (${c})`;
        sel.appendChild(opt);
    }
    const hasCurrent = current !== 'auto' && options.some(([t]) => t === current);
    selectedNvtxThread = hasCurrent ? current : 'auto';
    sel.value = selectedNvtxThread;
}

function onNvtxThreadChange() {
    const sel = document.getElementById('nvtxThreadSel');
    selectedNvtxThread = sel ? sel.value : 'auto';
    draw();
}

// ── Resize ──
function contentHeight() {
    // Total height needed: ruler + findings lane + NVTX + GPU separators + all stream rows
    const gpuSeps = isMultiGPU ? gpuBands.length * GPU_SEP_H : 0;
    return RULER_H + findingsLaneH() + nvtxAreaH() + gpuSeps + streamIds.length * (STREAM_H + STREAM_GAP) + 20;
}
function resize() {
    const r = wrap.getBoundingClientRect();
    const needH = Math.max(r.height, contentHeight());
    canvas.width = r.width * DPR;
    canvas.height = needH * DPR;
    canvas.style.width = r.width + 'px';
    canvas.style.height = needH + 'px';
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    draw();
}
window.addEventListener('resize', resize);

// ── Coordinate helpers ──
function nsToX(ns) {
    const w = canvas.width / DPR - LABEL_W;
    return LABEL_W + (ns - viewStart) / (viewEnd - viewStart) * w;
}
function xToNs(x) {
    const w = canvas.width / DPR - LABEL_W;
    return viewStart + (x - LABEL_W) / w * (viewEnd - viewStart);
}

function nvtxAreaH() {
    const d = activeNvtxMaxDepth();
    return showNVTX && d > 0 ? d * NVTX_ROW_H + 4 : 0;
}

// Compute Y position for a stream index, accounting for GPU separator rows
function streamY(idx) {
    let y = RULER_H + findingsLaneH() + nvtxAreaH();
    let sepsAbove = 0;
    if (isMultiGPU) {
        for (const band of gpuBands) {
            if (idx >= band.startIdx && band.startIdx > 0) sepsAbove++;
        }
    }
    return y + idx * (STREAM_H + STREAM_GAP) + sepsAbove * GPU_SEP_H;
}

// ── Draw ──
function draw() {
    const W = canvas.width / DPR;
    const H = canvas.height / DPR;
    ctx.clearRect(0, 0, W, H);
    ctx.font = '11px SF Mono, Cascadia Code, Fira Code, monospace';

    drawRuler(W);
    if (showNVTX) drawNVTX(W);
    drawStreams(W, H);
    if (FINDINGS.length) drawFindings(W, H);
}

function drawRuler(W) {
    const tw = W - LABEL_W;
    const viewSpan = viewEnd - viewStart;
    // Nice interval
    const rawInterval = viewSpan / (tw / 80);
    const mag = Math.pow(10, Math.floor(Math.log10(rawInterval)));
    const nice = [1, 2, 5, 10].find(n => n * mag >= rawInterval) * mag;
    const mode = renderSettings.rulerLabelMode === 'anchored' ? 'anchored' : 'absolute';
    const anchorNs = Math.floor(viewStart / 1e9) * 1e9;
    const rulerUnit = chooseRulerUnit(nice);

    ctx.fillStyle = '#161b22';
    ctx.fillRect(0, 0, W, RULER_H);
    ctx.strokeStyle = '#30363d';
    ctx.beginPath(); ctx.moveTo(0, RULER_H - 0.5); ctx.lineTo(W, RULER_H - 0.5); ctx.stroke();

    const start = Math.ceil(viewStart / nice) * nice;
    ctx.fillStyle = '#8b949e';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    let prevLabel = null;
    for (let t = start; t <= viewEnd; t += nice) {
        const x = nsToX(t);
        if (x < LABEL_W || x > W) continue;
        ctx.strokeStyle = '#30363d';
        ctx.beginPath(); ctx.moveTo(x, RULER_H - 6); ctx.lineTo(x, RULER_H); ctx.stroke();
        const valueNs = mode === 'anchored' ? (t - anchorNs) : t;
        let decimals = rulerUnit.decimals;
        let label = formatTickValue(valueNs, rulerUnit.div, decimals);
        while (prevLabel !== null && label === prevLabel && decimals < 7) {
            decimals += 1;
            label = formatTickValue(valueNs, rulerUnit.div, decimals);
        }
        prevLabel = label;
        ctx.fillText(label, x, RULER_H - 7);
    }

    // Label
    ctx.textAlign = 'right';
    ctx.fillStyle = '#8b949e';
    if (mode === 'anchored') {
        const anchorLabel = formatTickValue(anchorNs, 1e9, 0);
        ctx.fillText(`Time ${anchorLabel}s + (${rulerUnit.unit})`, LABEL_W - 6, RULER_H - 7);
    } else {
        ctx.fillText(`Time (${rulerUnit.unit})`, LABEL_W - 6, RULER_H - 7);
    }
}

function chooseRulerUnit(stepNs) {
    const secStep = stepNs / 1e9;
    const secDecimals = Math.min(6, Math.max(0, Math.ceil(-Math.log10(Math.max(secStep, 1e-18)))));
    if (secDecimals <= 3) return { unit: 's', div: 1e9, decimals: secDecimals };

    const msStep = stepNs / 1e6;
    const msDecimals = Math.min(4, Math.max(0, Math.ceil(-Math.log10(Math.max(msStep, 1e-18)))));
    if (msDecimals <= 3) return { unit: 'ms', div: 1e6, decimals: msDecimals };

    const usStep = stepNs / 1e3;
    const usDecimals = Math.min(3, Math.max(0, Math.ceil(-Math.log10(Math.max(usStep, 1e-18)))));
    if (usDecimals <= 2) return { unit: 'μs', div: 1e3, decimals: usDecimals };

    return { unit: 'ns', div: 1, decimals: 0 };
}

function formatTickValue(ns, div, decimals) {
    const v = ns / div;
    const fmt = new Intl.NumberFormat(undefined, {
        useGrouping: true,
        minimumFractionDigits: 0,
        maximumFractionDigits: Math.max(0, decimals),
    });
    return fmt.format(v);
}

function drawNVTX(W) {
    const baseY = RULER_H + findingsLaneH();
    ctx.fillStyle = 'rgba(22, 27, 34, 0.85)';
    ctx.fillRect(0, baseY, W, nvtxAreaH());

    const { activeGpu, thread, spans: visibleSpans, clipped } = activeNvtxLayout();
    const depthMax = activeNvtxMaxDepth();

    for (const span of visibleSpans) {
        const x1 = Math.max(LABEL_W, nsToX(span.start));
        const x2 = Math.min(W, nsToX(span.end));
        if (x2 < LABEL_W || x1 > W) continue;
        const w = Math.max(MIN_BLOCK_W, x2 - x1);
        const y = baseY + span._lane * NVTX_ROW_H + 1;
        const h = NVTX_ROW_H - 2;
        const isSel = selectedNvtx && selectedNvtx.key === span._key;
        const isSearchMatch = !searchQuery || searchNvtxMatches.has(span._key);

        ctx.fillStyle = nvtxColor(span.depth);
        let alpha = 0.82;
        if (selectedKernel) alpha = renderSettings.nvtxWhenKernelSelectedAlpha;
        if (selectedNvtx) alpha = isSel ? renderSettings.nvtxSelectedAlpha : renderSettings.nvtxNonSelectedAlpha;
        if (searchQuery && !isSearchMatch) alpha *= 0.3;
        ctx.globalAlpha = alpha;
        ctx.fillRect(x1, y, w, h);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = nvtxColor(span.depth);
        ctx.globalAlpha = isSel ? 1 : 0.75;
        ctx.strokeRect(x1 + 0.5, y + 0.5, w - 1, h - 1);
        ctx.globalAlpha = 1;

        // Label if wide enough
        if (w > 40) {
            ctx.fillStyle = isSel ? '#ffffff' : (searchQuery && !isSearchMatch ? '#6b7280' : '#e6edf3');
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.save();
            ctx.beginPath(); ctx.rect(x1 + 2, y, w - 4, h); ctx.clip();
            ctx.font = '10px SF Mono, Cascadia Code, monospace';
            ctx.fillText(span.name, x1 + 4, y + h / 2);
            ctx.restore();
            ctx.font = '11px SF Mono, Cascadia Code, Fira Code, monospace';
        }
    }

    // NVTX depth labels
    for (let d = 0; d < depthMax; d++) {
        ctx.fillStyle = '#8b949e';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.font = '9px SF Mono, monospace';
        ctx.fillText('L' + d, LABEL_W - 6, baseY + d * NVTX_ROW_H + NVTX_ROW_H / 2);
    }

    if (!visibleSpans.length && nvtxLoadingForGpu(activeGpu) && depthMax > 0) {
        ctx.fillStyle = '#d4a72c';
        ctx.font = '10px SF Mono, monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText('Loading NVTX…', LABEL_W + 8, baseY + NVTX_ROW_H / 2);
    } else if (!visibleSpans.length && depthMax > 0) {
        ctx.fillStyle = '#8b949e';
        ctx.font = '10px SF Mono, monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText('No NVTX in current filter', LABEL_W + 8, baseY + NVTX_ROW_H / 2);
    }

    // Show which GPU/thread NVTX is displayed
    if (depthMax > 0) {
        ctx.fillStyle = GPU_SEP_COLORS[gpuIds.indexOf(activeGpu) % GPU_SEP_COLORS.length];
        ctx.font = '9px SF Mono, monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        const threadLabel = thread ? ` • ${thread}` : '';
        ctx.fillText(`NVTX: GPU ${activeGpu}${threadLabel}`, LABEL_W + 4, baseY + 2);
        if (clipped > 0) {
            ctx.fillStyle = '#8b949e';
            ctx.textAlign = 'right';
            ctx.fillText(`+${clipped} clipped`, W - 8, baseY + 2);
        }
    }
    ctx.font = '11px SF Mono, Cascadia Code, Fira Code, monospace';

    // Separator
    const sepY = baseY + nvtxAreaH() - 1;
    ctx.strokeStyle = '#30363d';
    ctx.beginPath(); ctx.moveTo(0, sepY); ctx.lineTo(W, sepY); ctx.stroke();
}

function drawStreams(W, H) {
    // Draw GPU separator rows first (if multi-GPU)
    if (isMultiGPU) {
        for (let gi = 0; gi < gpuBands.length; gi++) {
            const band = gpuBands[gi];
            if (band.startIdx === 0) continue;  // No separator before first GPU
            const sepY = streamY(band.startIdx) - GPU_SEP_H;
            const sepColor = GPU_SEP_COLORS[gi % GPU_SEP_COLORS.length];

            // Background bar
            ctx.fillStyle = '#0d1117';
            ctx.fillRect(0, sepY, W, GPU_SEP_H);

            // Colored accent line
            ctx.fillStyle = sepColor;
            ctx.globalAlpha = 0.6;
            ctx.fillRect(0, sepY, W, 2);
            ctx.globalAlpha = 1;

            // GPU label
            ctx.fillStyle = sepColor;
            ctx.font = '10px SF Mono, Cascadia Code, Fira Code, monospace';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText(`── GPU ${band.gpuId} ──`, 8, sepY + GPU_SEP_H / 2 + 1);
            ctx.font = '11px SF Mono, Cascadia Code, Fira Code, monospace';
        }
        // First GPU header at top of streams area
        if (gpuBands.length > 0) {
            const band = gpuBands[0];
            const topY = streamY(0) - GPU_SEP_H;
            if (topY >= RULER_H + findingsLaneH() + nvtxAreaH() - GPU_SEP_H) {
                const sepColor = GPU_SEP_COLORS[0];
                ctx.fillStyle = '#0d1117';
                ctx.fillRect(0, RULER_H + findingsLaneH() + nvtxAreaH(), W, GPU_SEP_H);
                ctx.fillStyle = sepColor;
                ctx.globalAlpha = 0.6;
                ctx.fillRect(0, RULER_H + findingsLaneH() + nvtxAreaH(), W, 2);
                ctx.globalAlpha = 1;
                ctx.fillStyle = sepColor;
                ctx.font = '10px SF Mono, Cascadia Code, Fira Code, monospace';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.fillText(`── GPU ${band.gpuId} ──`, 8, RULER_H + findingsLaneH() + nvtxAreaH() + GPU_SEP_H / 2 + 1);
                ctx.font = '11px SF Mono, Cascadia Code, Fira Code, monospace';
            }
        }
    }

    for (let si = 0; si < streamIds.length; si++) {
        const sid = streamIds[si];
        if (hiddenStreams.has(sid)) continue;  // skip hidden streams
        const ks = streamMap[sid];
        const y = streamY(si) + (isMultiGPU && gpuBands[0].startIdx === 0 ? GPU_SEP_H : 0);
        const isSelected = si === selectedStreamIdx;
        const color = streamColor(si);

        // Background
        if (isSelected) {
            ctx.fillStyle = 'rgba(55, 65, 85, 0.5)';
            ctx.fillRect(0, y, W, STREAM_H);
        }

        // Stream label
        ctx.fillStyle = isSelected ? '#79b8ff' : '#8b949e';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        const isNccl = ks.some(k => k.name.toLowerCase().includes('nccl'));
        let streamLabel;
        if (isMultiGPU) {
            // "G0:S21" format
            const parts = sid.split(':');
            streamLabel = (isNccl ? '⚡ ' : '') + 'G' + parts[0] + ':S' + parts[1];
        } else {
            streamLabel = (isNccl ? '⚡ ' : '') + 'S' + sid;
        }
        ctx.fillText(streamLabel, LABEL_W - 6, y + STREAM_H / 2);

        // Kernel blocks
        for (const k of ks) {
            let x1 = nsToX(k.start_ns);
            let x2 = nsToX(k.end_ns);
            if (x2 < LABEL_W || x1 > W) continue;
            x1 = Math.max(LABEL_W, x1);
            x2 = Math.min(W, x2);
            const w = Math.max(MIN_BLOCK_W, x2 - x1);
            const bY = y + 2;
            const bH = STREAM_H - 4;

            const isMatch = !searchQuery || searchKernelMatches.has(k);
            const isSel = k === selectedKernel;
            const isNcclK = k.name.toLowerCase().includes('nccl');

            // Color (higher base alpha so blocks are clearly visible on dark bg)
            let fillColor = isNcclK ? '#d2a8ff' : color;
            let alpha = 0.78 + 0.22 * (k.heat || 0);
            if (searchQuery && !isMatch) alpha = renderSettings.kernelSearchNonMatchAlpha;
            if (selectedNvtx && !isSel) alpha = renderSettings.kernelWhenNvtxSelectedAlpha;
            if (isSel) { fillColor = '#79b8ff'; alpha = 1; }

            ctx.globalAlpha = alpha;
            ctx.fillStyle = fillColor;
            ctx.fillRect(x1, bY, w, bH);
            ctx.globalAlpha = 1;
            // Subtle border so blocks don't blend into background
            if (!isSel) {
                ctx.strokeStyle = fillColor;
                ctx.globalAlpha = 0.5;
                ctx.strokeRect(x1 + 0.5, bY + 0.5, w - 1, bH - 1);
                ctx.globalAlpha = 1;
            }

            if (isSel) {
                ctx.strokeStyle = '#79b8ff';
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, bY, w, bH);
                ctx.lineWidth = 1;
            }

            // Kernel name label
            if (w > 20) {
                ctx.fillStyle = isSel
                    ? '#fff'
                    : (searchQuery && !isMatch ? '#555' : '#1c1c1c');
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.save();
                ctx.beginPath(); ctx.rect(x1 + 1, bY, w - 2, bH); ctx.clip();
                const short = k.name.length > 40 ? k.name.slice(0, 38) + '…' : k.name;
                // Name first, duration only if space
                const label = w > 100 ? short + ' ' + fmtDur(k.duration_ms) : short;
                if (label) ctx.fillText(label, x1 + 3, bY + bH / 2);
                ctx.restore();
            }
        }

        // Gridline
        if (si < streamIds.length - 1) {
            ctx.strokeStyle = '#1c2230';
            ctx.beginPath(); ctx.moveTo(LABEL_W, y + STREAM_H + 1); ctx.lineTo(W, y + STREAM_H + 1); ctx.stroke();
        }
    }
}

// ── Hit testing ──
function hitTest(mx, my) {
    if (showNVTX) {
        const baseY = RULER_H + findingsLaneH();
        const endY = baseY + nvtxAreaH();
        if (my >= baseY && my <= endY) {
            const { spans } = activeNvtxLayout();
            for (let i = spans.length - 1; i >= 0; i--) {
                const span = spans[i];
                const x1 = Math.max(LABEL_W, nsToX(span.start));
                const x2 = Math.min(canvas.width / DPR, nsToX(span.end));
                const w = Math.max(MIN_BLOCK_W, x2 - x1);
                const y = baseY + span._lane * NVTX_ROW_H + 1;
                const h = NVTX_ROW_H - 2;
                if (mx >= x1 && mx <= x1 + w && my >= y && my <= y + h) {
                    return { nvtx: span, streamIdx: selectedStreamIdx };
                }
            }
            return { nvtx: null, streamIdx: selectedStreamIdx };
        }
    }

    for (let si = 0; si < streamIds.length; si++) {
        const y = streamY(si) + (isMultiGPU && gpuBands[0].startIdx === 0 ? GPU_SEP_H : 0);
        if (my < y || my > y + STREAM_H) continue;
        const ks = streamMap[streamIds[si]];
        for (const k of ks) {
            const x1 = Math.max(LABEL_W, nsToX(k.start_ns));
            const x2 = Math.min(canvas.width / DPR, nsToX(k.end_ns));
            if (mx >= x1 && mx <= Math.max(x1 + MIN_BLOCK_W, x2)) {
                return { kernel: k, streamIdx: si };
            }
        }
        return { kernel: null, streamIdx: si };
    }
    return null;
}

// ── Detail panel ──
function inferKernelPathFromNvtx(k) {
    const gpu = k._gpu;
    let candidates = nvtxSpans.filter(s =>
        s.gpu === gpu && s.start <= k.start_ns && s.end >= k.end_ns
    );
    if (!candidates.length) return '';
    if (selectedNvtxThread && selectedNvtxThread !== 'auto') {
        const byThread = candidates.filter(
            s => (s.thread || '(unnamed)') === selectedNvtxThread
        );
        if (byThread.length) candidates = byThread;
    }
    candidates.sort((a, b) =>
        ((a.end - a.start) - (b.end - b.start)) || ((b.depth || 0) - (a.depth || 0))
    );
    const best = candidates[0];
    if (!best) return '';
    return best.path ? `${best.path} > ${k.name}` : k.name;
}

function kernelPathForDisplay(k) {
    const p = String(k._path || '').trim();
    if (p && p !== k.name) return p;
    return inferKernelPathFromNvtx(k) || p || k.name;
}

function showDetail(item) {
    if (!item) {
        document.getElementById('detail').className = 'empty';
        document.getElementById('detail').innerHTML = 'Click a kernel or NVTX to see details. Press A for AI chat. Keyboard: ←/→ pan, +/- zoom, ↑/↓ stream, / search, ? help';
        return;
    }
    if (item._kind === 'nvtx') {
        const pathInfo = renderPathHtml(item.path || '');
        const durMs = (item.end - item.start) / 1e6;
        const gpuLabel = isMultiGPU ? ` &nbsp;|&nbsp; GPU ${item.gpu}` : '';
        const threadLabel = item.thread ? ` &nbsp;|&nbsp; ${escH(item.thread)}` : '';
        document.getElementById('detail').className = '';
        document.getElementById('detail').innerHTML =
            `<div class="detail-name">📦 ${escH(item.name)}</div>` +
            `<div class="detail-dur">${fmtDur(durMs)}${gpuLabel}${threadLabel} &nbsp;|&nbsp; ${fmtNs(item.start)} → ${fmtNs(item.end)}</div>` +
            (pathInfo.html ? `<div class="detail-path ${pathInfo.layout}">🧭 ${pathInfo.html}</div>` : '');
        return;
    }

    const k = item;
    const displayPath = kernelPathForDisplay(k);
    const pathInfo = renderPathHtml(displayPath);
    const gpuLabel = isMultiGPU ? ` &nbsp;|&nbsp; GPU ${k._gpu}` : '';

    document.getElementById('detail').className = '';
    document.getElementById('detail').innerHTML =
        `<div class="detail-name">⚡ ${escH(k.name)}</div>` +
        `<div class="detail-dur">${fmtDur(k.duration_ms)} &nbsp;|&nbsp; Stream ${k.stream ?? '?'}${gpuLabel} &nbsp;|&nbsp; ${fmtNs(k.start_ns)} → ${fmtNs(k.end_ns)}</div>` +
        (pathInfo.html ? `<div class="detail-path ${pathInfo.layout}">📦 ${pathInfo.html}</div>` : '');
}

function escH(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

function renderPathHtml(path) {
    const parts = String(path || '').split(' > ').filter(Boolean);
    if (!parts.length) return { html: '', layout: 'horizontal' };
    const layout = renderSettings.hierarchyLayout === 'vertical' ? 'vertical' : 'horizontal';
    if (layout === 'vertical') {
        const html = parts
            .map((p, idx) => {
                const cls = idx === parts.length - 1 ? 'node current' : 'node';
                const indent = idx * 14;
                return `<span class="vline" style="padding-left:${indent}px"><span class="${cls}">${escH(p)}</span></span>`;
            })
            .join('');
        return { html, layout };
    }
    const html = parts
        .map((p, idx) => {
            const cls = idx === parts.length - 1 ? 'node current' : 'node';
            return `<span class="${cls}">${escH(p)}</span>`;
        })
        .join('<span class="sep">›</span>');
    return { html, layout };
}

// ── Navigation ──
// ── Progressive tile loading on viewport change ──
let _viewChangeTimer = null;
const _viewLabel = (typeof BOOT.GPU_LABEL === 'string' && BOOT.GPU_LABEL) ? BOOT.GPU_LABEL : 'timeline';
const _viewKey = 'timeline-view-' + _viewLabel.replace(/[^a-zA-Z0-9]/g, '_');
function afterViewChange() {
    // Persist viewport to localStorage
    try { localStorage.setItem(_viewKey, JSON.stringify({ s: viewStart, e: viewEnd, t: Date.now() })); } catch (e) { }
    if (PROGRESSIVE) {
        ensureTilesForView(viewStart, viewEnd).catch(console.error);
        prefetchAdjacent();
    }
    if (window.parent && window.parent !== window) {
        window.parent.postMessage({ type: 'T_VIEW_CHANGE', startNs: viewStart, endNs: viewEnd }, '*');
    }
    clearTimeout(_viewChangeTimer);
    _viewChangeTimer = setTimeout(async () => {
        resize();
    }, 150);
}

function clampView() {
    const minNs = 0;  // time is never negative
    const maxNs = profileMeta ? profileMeta.time_range_ns[1] : timeEnd;
    const span = viewEnd - viewStart;
    if (viewStart < minNs) { viewStart = minNs; viewEnd = minNs + span; }
    if (viewEnd > maxNs) { viewEnd = maxNs; viewStart = maxNs - span; }
    // Final clamp in case span > total range
    if (viewStart < minNs) viewStart = minNs;
}

function panBy(fraction) {
    const span = viewEnd - viewStart;
    const delta = span * fraction;
    viewStart += delta; viewEnd += delta;
    clampView();
    draw();
    afterViewChange();
}
function zoomAt(factor, centerNs) {
    if (centerNs === undefined) centerNs = (viewStart + viewEnd) / 2;
    const span = viewEnd - viewStart;
    const newSpan = span * factor;
    const minSpan = 100; // 100 ns minimum
    if (newSpan < minSpan) return;
    const ratio = (centerNs - viewStart) / span;
    viewStart = centerNs - newSpan * ratio;
    viewEnd = centerNs + newSpan * (1 - ratio);
    clampView();
    draw();
    afterViewChange();
}
function zoomIn() { zoomAt(0.5); }
function zoomOut() { zoomAt(2); }
function fitAll() {
    const maxNs = profileMeta ? profileMeta.time_range_ns[1] : timeEnd;
    viewStart = 0;
    viewEnd = maxNs;
    clampView();
    draw();
    afterViewChange();
}

function selectKernel(k) {
    selectedKernel = k;
    selectedNvtx = null;
    showDetail(k);
    draw();
}

function selectNvtx(span) {
    selectedKernel = null;
    selectedNvtx = { ...span, key: span._key, _kind: 'nvtx' };
    showDetail(selectedNvtx);
    draw();
}

function computeFitViewRange(startNs, endNs) {
    const minSpan = 1000;
    const span = Math.max(minSpan, endNs - startNs);
    const pad = Math.max(span * 0.03, 500);
    return { start: startNs - pad, end: endNs + pad };
}

function isViewCloseTo(startNs, endNs) {
    const span = Math.max(endNs - startNs, 1);
    const tol = Math.max(span * 0.02, 2000);
    return Math.abs(viewStart - startNs) <= tol && Math.abs(viewEnd - endNs) <= tol;
}

function toggleFitToElement(kind, key, startNs, endNs) {
    const fit = computeFitViewRange(startNs, endNs);
    const canRestore = fitZoomState
        && fitZoomState.kind === kind
        && fitZoomState.key === key
        && isViewCloseTo(fitZoomState.fitStart, fitZoomState.fitEnd);
    if (canRestore) {
        viewStart = fitZoomState.prevStart;
        viewEnd = fitZoomState.prevEnd;
        fitZoomState = null;
        clampView();
        draw();
        afterViewChange();
        return;
    }
    fitZoomState = {
        kind,
        key,
        prevStart: viewStart,
        prevEnd: viewEnd,
        fitStart: fit.start,
        fitEnd: fit.end,
    };
    viewStart = fit.start;
    viewEnd = fit.end;
    clampView();
    draw();
    afterViewChange();
}

function zoomToNsRange(startNs, endNs) {
    const minSpan = 1000;
    const span = Math.max(minSpan, endNs - startNs);
    const pad = Math.max(span * 0.25, minSpan);
    viewStart = startNs - pad;
    viewEnd = endNs + pad;
    clampView();
    draw();
    afterViewChange();
}

function parseTimeInputToNs(raw) {
    if (!raw) return null;
    const s = String(raw).trim().toLowerCase();
    if (!s) return null;
    if (s.endsWith('ms')) {
        const v = parseFloat(s.slice(0, -2).trim());
        return Number.isFinite(v) ? v * 1e6 : null;
    }
    if (s.endsWith('us')) {
        const v = parseFloat(s.slice(0, -2).trim());
        return Number.isFinite(v) ? v * 1e3 : null;
    }
    if (s.endsWith('s')) {
        const v = parseFloat(s.slice(0, -1).trim());
        return Number.isFinite(v) ? v * 1e9 : null;
    }
    const v = parseFloat(s);
    return Number.isFinite(v) ? v * 1e9 : null; // default seconds
}

function gotoTimePrompt() {
    const current = fmtTimeS((viewStart + viewEnd) / 2);
    const raw = prompt(`Go to time (supports s/ms/us). Current center: ${current}`, '');
    if (raw === null) return;
    const ns = parseTimeInputToNs(raw);
    if (!Number.isFinite(ns)) {
        showToast('Invalid time format');
        return;
    }
    const span = viewEnd - viewStart;
    viewStart = ns - span / 2;
    viewEnd = ns + span / 2;
    clampView();
    draw();
    afterViewChange();
}

function gotoNvtxByQuery(query) {
    const q = String(query || '').trim().toLowerCase();
    if (!q) {
        showToast('NVTX query is empty');
        return;
    }
    const { spans } = activeGpuNvtxSpans();
    const matches = spans.filter(s => (s.name || '').toLowerCase().includes(q));
    if (!matches.length) {
        showToast(`No NVTX match: ${query}`);
        return;
    }
    const target = matches[0];
    showNVTX = true;
    document.getElementById('nvtxBtn').classList.add('active');
    if (target.thread) {
        selectedNvtxThread = target.thread;
        updateNvtxThreadOptions();
    }
    const laid = layoutNvtxSpans([target], NVTX_PIN_ROWS).spans[0]
        || { ...target, _lane: NVTX_PIN_ROWS - 1, _key: nvtxKey(target), _clipped: true };
    selectNvtx(laid);
    if ((target.depth || 0) >= NVTX_PIN_ROWS) {
        showToast(`NVTX depth ${target.depth} clipped to row ${NVTX_PIN_ROWS - 1}`);
    }
    zoomToNsRange(target.start, target.end);
}

function gotoNvtxPrompt() {
    const raw = prompt('Go to NVTX (substring match on active GPU)', '');
    if (raw === null) return;
    gotoNvtxByQuery(raw);
}

function gotoKernelByQuery(query) {
    const q = String(query || '').trim().toLowerCase();
    if (!q) {
        showToast('Kernel query is empty');
        return;
    }
    const target = kernels.find(k => (k.name || '').toLowerCase().includes(q));
    if (!target) {
        showToast(`No kernel match: ${query}`);
        return;
    }
    const si = streamIds.indexOf(target._streamKey);
    if (si >= 0) selectedStreamIdx = si;
    selectKernel(target);
    ensureVisible(target);
}

function runCommandText(raw) {
    const cmd = String(raw || '').trim();
    if (!cmd) return;
    const sp = cmd.split(/\s+/, 2);
    const op = sp[0].toLowerCase();
    const arg = cmd.slice(op.length).trim();
    if (op === 'g' || op === 't') {
        const ns = parseTimeInputToNs(arg);
        if (!Number.isFinite(ns)) {
            showToast('Invalid time format');
            return;
        }
        const span = viewEnd - viewStart;
        viewStart = ns - span / 2;
        viewEnd = ns + span / 2;
        clampView();
        draw();
        afterViewChange();
        return;
    }
    if (op === 'n') {
        gotoNvtxByQuery(arg);
        return;
    }
    if (op === 'k') {
        gotoKernelByQuery(arg);
        return;
    }
    if (op === '/' || op === 's') {
        document.getElementById('searchInput').value = arg;
        onSearch();
        return;
    }
    // Treat plain text as search
    document.getElementById('searchInput').value = cmd;
    onSearch();
}

function setCommandPalette(open, seed = '') {
    const palette = document.getElementById('commandPalette');
    const input = document.getElementById('commandInput');
    if (!palette || !input) return;
    palette.style.display = open ? 'block' : 'none';
    if (open) {
        input.value = seed || '';
        setTimeout(() => {
            input.focus();
            input.select();
        }, 0);
    } else if (document.activeElement === input) {
        input.blur();
    }
}

function openCommandPalette(seed = '') {
    setCommandPalette(true, seed);
}

function nextKernel(dir) {
    const sid = streamIds[selectedStreamIdx];
    const ks = streamMap[sid];
    if (!ks || !ks.length) return;
    // Find current kernel in this stream using _streamKey
    if (!selectedKernel || selectedKernel._streamKey !== sid) {
        selectKernel(ks[0]); ensureVisible(ks[0]); return;
    }
    const idx = ks.indexOf(selectedKernel);
    if (idx === -1) { selectKernel(ks[0]); ensureVisible(ks[0]); return; }
    const next = idx + dir;
    if (next >= 0 && next < ks.length) {
        selectKernel(ks[next]); ensureVisible(ks[next]);
    }
}

function ensureNvtxVisible(span) {
    const spanNs = viewEnd - viewStart;
    if (span.start < viewStart || span.end > viewEnd) {
        const center = (span.start + span.end) / 2;
        viewStart = center - spanNs / 2;
        viewEnd = center + spanNs / 2;
        clampView();
        draw();
        afterViewChange();
    }
}

function nextNvtx(dir) {
    const { spans } = activeNvtxLayout();
    if (!spans.length) return false;
    const ordered = spans
        .slice()
        .sort((a, b) => (a.start - b.start) || (a.end - b.end) || a.name.localeCompare(b.name));

    if (!selectedNvtx || !selectedNvtx.key) {
        const first = dir >= 0 ? ordered[0] : ordered[ordered.length - 1];
        selectNvtx(first);
        ensureNvtxVisible(first);
        return true;
    }

    const currentIdx = ordered.findIndex(s => s._key === selectedNvtx.key);
    if (currentIdx < 0) {
        const first = dir >= 0 ? ordered[0] : ordered[ordered.length - 1];
        selectNvtx(first);
        ensureNvtxVisible(first);
        return true;
    }

    const current = ordered[currentIdx];
    const sameLevel = ordered.filter(s => s._lane === current._lane);
    if (!sameLevel.length) return false;
    const laneIdx = sameLevel.findIndex(s => s._key === current._key);
    const nextIdx = laneIdx + dir;
    if (nextIdx < 0 || nextIdx >= sameLevel.length) return false;
    const target = sameLevel[nextIdx];
    selectNvtx(target);
    ensureNvtxVisible(target);
    return true;
}

function ensureVisible(k) {
    const span = viewEnd - viewStart;
    if (k.start_ns < viewStart || k.end_ns > viewEnd) {
        const center = (k.start_ns + k.end_ns) / 2;
        viewStart = center - span / 2;
        viewEnd = center + span / 2;
        clampView();
        draw();
        afterViewChange();
    }
}

// ── GPU Info dropdown ──
function toggleGpuInfo() {
    const panel = document.getElementById('gpuInfoPanel');
    if (panel.style.display === 'none') {
        let html = '<table style="border-collapse:collapse;width:100%">';
        html += '<tr style="color:#58a6ff"><th style="text-align:left;padding:2px 6px">GPU</th><th style="text-align:left;padding:2px 6px">Name</th><th style="text-align:right;padding:2px 6px">SMs</th><th style="text-align:right;padding:2px 6px">Mem</th><th style="text-align:left;padding:2px 6px">PCI</th></tr>';
        GPU_INFO.forEach(g => {
            html += `<tr><td style="padding:2px 6px;color:#7ee787">${g.id}</td><td style="padding:2px 6px">${g.name}</td><td style="padding:2px 6px;text-align:right">${g.sms}</td><td style="padding:2px 6px;text-align:right">${g.mem_gb}GB</td><td style="padding:2px 6px;color:#8b949e">${g.pci}</td></tr>`;
        });
        html += '</table>';
        document.getElementById('gpuInfoContent').innerHTML = html;
        panel.style.display = 'block';
    } else {
        panel.style.display = 'none';
    }
}

// ── Bookmarks ──
function saveBookmark() {
    const k = selectedKernel;
    const time_ns = k ? (k.start_ns + k.end_ns) / 2 : (viewStart + viewEnd) / 2;
    const bm = {
        label: k ? k.name.slice(0, 30) : `@${(time_ns / 1e9).toFixed(2)}s`,
        time_ns,
        nvtx_path: k ? kernelPathForDisplay(k) : '',
        nvtx_index: k ? nvtxSpans.findIndex(s => time_ns >= s.start && time_ns <= s.end) : -1,
        kernel_index: k ? kernels.indexOf(k) : -1,
        kernel_name: k ? k.name : '',
        stream: k ? (k._streamKey || '') : '',
        gpu: k ? (k._gpu ?? '') : '',
    };
    bookmarks.push(bm);
    showToast(`Bookmark #${bookmarks.length}: ${bm.label}`);
    draw();
}

function jumpToBookmark(idx) {
    if (idx < 0 || idx >= bookmarks.length) return;
    const bm = bookmarks[idx];
    const span = viewEnd - viewStart;
    viewStart = bm.time_ns - span / 2;
    viewEnd = bm.time_ns + span / 2;
    clampView();
    draw();
    afterViewChange();
    // Try to select the kernel if we have an index
    if (bm.kernel_index >= 0 && bm.kernel_index < kernels.length) {
        selectKernel(kernels[bm.kernel_index]);
    }
    showToast(`→ Bookmark #${idx + 1}: ${bm.label}`);
}

function toggleBookmarkList() {
    const panel = document.getElementById('gpuInfoPanel');
    if (bookmarks.length === 0) { showToast('No bookmarks. Press B to save one.'); return; }
    let html = '<div style="color:#58a6ff;margin-bottom:4px;font-weight:600">📌 Bookmarks</div>';
    html += '<table style="border-collapse:collapse;width:100%">';
    bookmarks.forEach((bm, i) => {
        const ts = (bm.time_ns / 1e9).toFixed(3) + 's';
        html += `<tr style="cursor:pointer" onclick="jumpToBookmark(${i});document.getElementById('gpuInfoPanel').style.display='none'"><td style="padding:2px 4px;color:#7ee787">${i + 1}</td><td style="padding:2px 4px">${bm.label}</td><td style="padding:2px 4px;color:#8b949e">${ts}</td><td style="padding:2px 4px;color:#8b949e">${bm.nvtx_path ? bm.nvtx_path.split(' > ').pop() : ''}</td></tr>`;
    });
    html += '</table>';
    document.getElementById('gpuInfoContent').innerHTML = html;
    panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
}

function showToast(msg) {
    let toast = document.getElementById('toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'toast';
        toast.style.cssText = 'position:fixed;bottom:60px;left:50%;transform:translateX(-50%);background:#21262d;color:#e6edf3;padding:6px 16px;border-radius:6px;border:1px solid #30363d;font-size:12px;z-index:200;opacity:0;transition:opacity 0.3s';
        document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.style.opacity = '1';
    setTimeout(() => { toast.style.opacity = '0'; }, 2000);
}

function toggleNVTX() {
    showNVTX = !showNVTX;
    document.getElementById('nvtxBtn').classList.toggle('active', showNVTX);
    updateNvtxThreadOptions();
    updateNvtxLoadingIndicator();
    draw();
    if (showNVTX) {
        void ensureTilesForView(viewStart, viewEnd);
    }
}

function isGpuInfoPanelOpen() {
    const panel = document.getElementById('gpuInfoPanel');
    return !!panel && panel.style.display !== 'none';
}

function streamNumberKey(sid) {
    if (!isMultiGPU) return String(sid);
    const text = String(sid);
    const idx = text.indexOf(':');
    return idx >= 0 ? text.slice(idx + 1) : text;
}

function streamNumberGroups() {
    const groups = new Map();
    for (const sid of streamIds) {
        const key = streamNumberKey(sid);
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(sid);
    }
    return [...groups.entries()].sort((a, b) => {
        const an = Number(a[0]);
        const bn = Number(b[0]);
        if (Number.isFinite(an) && Number.isFinite(bn)) return an - bn;
        return String(a[0]).localeCompare(String(b[0]));
    });
}

function renderStreamFilterPanel() {
    const panel = document.getElementById('gpuInfoPanel');
    let html = '<div style="color:#58a6ff;margin-bottom:6px;font-weight:600">📺 Stream Number Visibility</div>';
    const visibleCount = streamIds.length - hiddenStreams.size;
    html += `<div style="margin-bottom:6px;font-size:11px;color:#8b949e">Visible ${visibleCount}/${streamIds.length} · ` +
        `<a href="#" onclick="setAllStreams(true);return false" style="color:#58a6ff;text-decoration:none">All</a> ` +
        `<a href="#" onclick="setAllStreams(false);return false" style="color:#58a6ff;text-decoration:none">None</a> ` +
        `<a href="#" onclick="toggleSelectedStreamNumber();return false" style="color:#58a6ff;text-decoration:none">Toggle selected number</a> ` +
        `<a href="#" onclick="invertStreams();return false" style="color:#58a6ff;text-decoration:none">Invert</a></div>`;

    const groups = streamNumberGroups();
    for (const [streamNum, sids] of groups) {
        const visible = sids.filter(sid => !hiddenStreams.has(sid)).length;
        const total = sids.length;
        const checked = visible === total ? 'checked' : '';
        const totalKernels = sids.reduce((acc, sid) => acc + ((streamMap[sid] || []).length), 0);
        html += `<label style="display:block;padding:1px 0;cursor:pointer"><input type="checkbox" ${checked} onchange="toggleStreamNumber('${streamNum}')" style="margin-right:4px">S${streamNum} <span style="color:#484f58">(${visible}/${total} GPUs, ${totalKernels} kernels)</span></label>`;
    }
    document.getElementById('gpuInfoContent').innerHTML = html;
    return panel;
}

function toggleStreamFilter() {
    const panel = document.getElementById('gpuInfoPanel');
    renderStreamFilterPanel();
    panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
}

function toggleStream(sid) {
    if (hiddenStreams.has(sid)) hiddenStreams.delete(sid);
    else hiddenStreams.add(sid);
    resize();
    if (isGpuInfoPanelOpen()) renderStreamFilterPanel();
}

function toggleStreamNumber(streamNum) {
    const members = streamIds.filter(sid => streamNumberKey(sid) === String(streamNum));
    if (members.length === 0) return;
    const allVisible = members.every(sid => !hiddenStreams.has(sid));
    for (const sid of members) {
        if (allVisible) hiddenStreams.add(sid);
        else hiddenStreams.delete(sid);
    }
    resize();
    if (isGpuInfoPanelOpen()) renderStreamFilterPanel();
}

function setAllStreams(visible) {
    if (visible) hiddenStreams.clear();
    else streamIds.forEach(sid => hiddenStreams.add(sid));
    resize();
    if (isGpuInfoPanelOpen()) renderStreamFilterPanel();
}

function invertStreams() {
    const next = new Set();
    streamIds.forEach(sid => {
        if (!hiddenStreams.has(sid)) next.add(sid);
    });
    hiddenStreams = next;
    resize();
    if (isGpuInfoPanelOpen()) renderStreamFilterPanel();
}

function toggleSelectedStreamNumber() {
    const sid = streamIds[selectedStreamIdx];
    if (!sid) return;
    toggleStreamNumber(streamNumberKey(sid));
}

function toggleHelp() {
    const h = document.getElementById('helpOverlay');
    h.style.display = h.style.display === 'none' ? 'block' : 'none';
}

// ── Search ──
function onSearch() {
    searchQuery = document.getElementById('searchInput').value.trim().toLowerCase();
    searchKernelMatches = new Set();
    searchNvtxMatches = new Set();
    if (searchQuery) {
        kernels.forEach(k => { if (k.name.toLowerCase().includes(searchQuery)) searchKernelMatches.add(k); });
        nvtxSpans.forEach(s => {
            if ((s.name || '').toLowerCase().includes(searchQuery)) searchNvtxMatches.add(nvtxKey(s));
        });
        document.getElementById('searchHint').textContent = `${searchKernelMatches.size} kernels, ${searchNvtxMatches.size} NVTX`;
    } else {
        document.getElementById('searchHint').textContent = '';
    }
    draw();
}

function gotoFirstSearchMatch() {
    if (!searchQuery) return;
    const kernel = kernels.find(k => searchKernelMatches.has(k));
    if (kernel) {
        const si = streamIds.indexOf(kernel._streamKey);
        if (si >= 0) selectedStreamIdx = si;
        selectKernel(kernel);
        ensureVisible(kernel);
        return;
    }
    const { spans } = activeGpuNvtxSpans();
    const nvtx = spans.find(s => searchNvtxMatches.has(nvtxKey(s)));
    if (nvtx) gotoNvtxByQuery(nvtx.name);
}

// ── Mouse events ──
canvas.addEventListener('mousedown', e => {
    if (e.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    if (mx < LABEL_W) return;

    // Check findings annotation lane + overlays first
    if (FINDINGS.length) {
        const fi = _findingsHitTest(mx, my);
        if (fi >= 0) {
            selectFinding(fi);
            // Ensure sidebar is open
            const sidebar = document.getElementById('findingsSidebar');
            if (sidebar && !sidebar.classList.contains('open')) toggleFindings();
            return;
        }
    }

    const hit = hitTest(mx, my);
    if (hit) {
        if (hit.streamIdx !== null && hit.streamIdx !== undefined) {
            selectedStreamIdx = hit.streamIdx;
            updateNvtxThreadOptions();
            maybeFetchNvtxForCurrentView();
        }
        if (hit.nvtx) { selectNvtx(hit.nvtx); return; }
        if (hit.kernel) { selectKernel(hit.kernel); return; }
    }

    // Shift+drag = pan (old behavior), plain drag = select time range
    if (e.shiftKey) {
        isDragging = true;
        dragStartX = e.clientX;
        dragViewStart = viewStart;
        dragViewEnd = viewEnd;
        canvas.style.cursor = 'grabbing';
    } else {
        isSelecting = true;
        selectStartX = mx;
        selectStartNs = xToNs(mx);
        selectEndNs = selectStartNs;
        canvas.style.cursor = 'col-resize';
    }
});

canvas.addEventListener('dblclick', e => {
    if (e.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    if (mx < LABEL_W) return;
    const hit = hitTest(mx, my);
    if (!hit) return;
    if (hit.nvtx) {
        const n = hit.nvtx;
        selectNvtx(n);
        toggleFitToElement('nvtx', n._key, n.start, n.end);
        return;
    }
    if (hit.kernel) {
        const k = hit.kernel;
        selectedStreamIdx = hit.streamIdx ?? selectedStreamIdx;
        selectKernel(k);
        const kKey = `${k.start_ns}|${k.end_ns}|${k.stream}|${k.name}|${k._gpu ?? ''}`;
        toggleFitToElement('kernel', kKey, k.start_ns, k.end_ns);
    }
});

canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;

    if (isDragging) {
        const dx = e.clientX - dragStartX;
        const w = canvas.width / DPR - LABEL_W;
        const nsPerPx = (dragViewEnd - dragViewStart) / w;
        viewStart = dragViewStart - dx * nsPerPx;
        viewEnd = dragViewEnd - dx * nsPerPx;
        clampView();
        afterViewChange();
        draw();
        return;
    }

    if (isSelecting) {
        selectEndNs = xToNs(Math.max(LABEL_W, Math.min(mx, canvas.width / DPR)));
        draw();
        // Draw selection overlay
        const x1 = Math.max(LABEL_W, nsToX(Math.min(selectStartNs, selectEndNs)));
        const x2 = Math.min(canvas.width / DPR, nsToX(Math.max(selectStartNs, selectEndNs)));
        ctx.fillStyle = 'rgba(88, 166, 255, 0.15)';
        ctx.fillRect(x1, 0, x2 - x1, canvas.height / DPR);
        ctx.strokeStyle = '#58a6ff';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(x1, 0); ctx.lineTo(x1, canvas.height / DPR); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x2, 0); ctx.lineTo(x2, canvas.height / DPR); ctx.stroke();
        ctx.setLineDash([]);
        return;
    }

    // Tooltip
    const hit = hitTest(mx, my);
    const tt = document.getElementById('tooltip');
    if (hit && hit.kernel) {
        const k = hit.kernel;
        const isNccl = k.name.toLowerCase().includes('nccl');
        tt.innerHTML = `<div style="color:${isNccl ? '#d2a8ff' : '#7ee787'};font-weight:600">${escH(k.name)}</div>` +
            `<div style="color:#8b949e">${fmtDur(k.duration_ms)} &nbsp;|&nbsp; Stream ${k.stream ?? '?'}</div>`;
        tt.style.display = 'block';
        tt.style.left = (e.clientX + 12) + 'px';
        tt.style.top = (e.clientY - 10) + 'px';
        // Keep on screen
        const tr = tt.getBoundingClientRect();
        if (tr.right > window.innerWidth) tt.style.left = (e.clientX - tr.width - 8) + 'px';
        if (tr.bottom > window.innerHeight) tt.style.top = (e.clientY - tr.height - 8) + 'px';
    } else if (hit && hit.nvtx) {
        const n = hit.nvtx;
        tt.innerHTML = `<div style="color:#79b8ff;font-weight:600">${escH(n.name)}</div>` +
            `<div style="color:#8b949e">${fmtDur((n.end - n.start) / 1e6)} &nbsp;|&nbsp; ${(n.thread || '(unnamed)')}</div>`;
        tt.style.display = 'block';
        tt.style.left = (e.clientX + 12) + 'px';
        tt.style.top = (e.clientY - 10) + 'px';
    } else {
        // Check findings hover
        if (FINDINGS.length) {
            const fi = _findingsHitTest(mx, my);
            if (fi >= 0) {
                const f = FINDINGS[fi];
                const sevCol = _findingSevColor(f.severity);
                const rangeStr = f.end_ns
                    ? _fmtFindingNs(f.start_ns) + ' → ' + _fmtFindingNs(f.end_ns)
                    : '@ ' + _fmtFindingNs(f.start_ns);
                tt.innerHTML =
                    `<div style="color:${sevCol};font-weight:600">📋 #${fi + 1} ${escH(f.label)}</div>` +
                    `<div style="color:#8b949e">${(f.severity || 'info').toUpperCase()} · ${f.type}` +
                    (f.gpu_id != null ? ` · GPU ${f.gpu_id}` : '') +
                    ` · ${rangeStr}</div>` +
                    (f.note ? `<div style="color:#c9d1d9;margin-top:3px;max-width:280px;white-space:normal;line-height:1.3">${escH(f.note.slice(0, 120))}</div>` : '');
                tt.style.display = 'block';
                tt.style.left = (e.clientX + 12) + 'px';
                tt.style.top = (e.clientY - 10) + 'px';
                const tr = tt.getBoundingClientRect();
                if (tr.right > window.innerWidth) tt.style.left = (e.clientX - tr.width - 8) + 'px';
                if (tr.bottom > window.innerHeight) tt.style.top = (e.clientY - tr.height - 8) + 'px';
                canvas.style.cursor = 'pointer';
            } else {
                tt.style.display = 'none';
                if (!isDragging && !isSelecting) canvas.style.cursor = 'crosshair';
            }
        } else {
            tt.style.display = 'none';
        }
    }
});

canvas.addEventListener('mouseup', () => {
    if (isSelecting) {
        isSelecting = false;
        canvas.style.cursor = 'crosshair';
        // Zoom to selection if it's wide enough
        const sNs = Math.min(selectStartNs, selectEndNs);
        const eNs = Math.max(selectStartNs, selectEndNs);
        if (eNs - sNs > 1000) {
            viewStart = sNs; viewEnd = eNs;
            clampView();
            draw();
            afterViewChange();
        } else {
            draw();  // redraw to clear selection overlay
        }
        return;
    }
    isDragging = false; canvas.style.cursor = 'crosshair';
});
canvas.addEventListener('mouseleave', () => {
    isDragging = false; isSelecting = false; canvas.style.cursor = 'crosshair';
    document.getElementById('tooltip').style.display = 'none';
});

canvas.addEventListener('wheel', e => {
    e.preventDefault();
    if (e.shiftKey || e.ctrlKey) {
        // Shift+scroll or pinch (ctrlKey on Mac) = zoom
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const ns = xToNs(mx);
        const factor = e.deltaY > 0 ? 1.15 : 1 / 1.15;
        zoomAt(factor, ns);
    } else {
        // Plain scroll = pan (trackpad swipe)
        const span = viewEnd - viewStart;
        const dx = e.deltaX * span * 0.001;
        viewStart += dx; viewEnd += dx;
        // Vertical scroll moves canvas scroll position
        wrap.scrollTop += e.deltaY;
        clampView();
        draw();
        afterViewChange();
    }
}, { passive: false });

const detailEl = document.getElementById('detail');
const detailResizeHandle = document.getElementById('detailResizeHandle');
if (detailEl && detailResizeHandle) {
    detailResizeHandle.addEventListener('mousedown', e => {
        if (e.button !== 0) return;
        e.preventDefault();
        isResizingDetail = true;
        detailResizeStartY = e.clientY;
        detailResizeStartH = detailEl.getBoundingClientRect().height;
        document.body.style.cursor = 'ns-resize';
        document.body.style.userSelect = 'none';
    });
    document.addEventListener('mousemove', e => {
        if (!isResizingDetail) return;
        const dy = detailResizeStartY - e.clientY;
        const minH = 48;
        const maxH = Math.max(120, Math.floor(window.innerHeight * 0.7));
        const h = Math.min(maxH, Math.max(minH, detailResizeStartH + dy));
        detailEl.style.height = `${h}px`;
        resize();
    });
    document.addEventListener('mouseup', () => {
        if (!isResizingDetail) return;
        isResizingDetail = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
    });
}

const searchInputEl = document.getElementById('searchInput');
if (searchInputEl) {
    searchInputEl.addEventListener('keydown', e => {
        if (e.key === 'Enter') {
            e.preventDefault();
            gotoFirstSearchMatch();
        }
    });
}

const commandPaletteBackdrop = document.getElementById('commandPaletteBackdrop');
if (commandPaletteBackdrop) {
    commandPaletteBackdrop.addEventListener('click', () => setCommandPalette(false));
}

const commandInputEl = document.getElementById('commandInput');
if (commandInputEl) {
    commandInputEl.addEventListener('keydown', e => {
        if (e.key === 'Enter') {
            e.preventDefault();
            const cmd = commandInputEl.value;
            setCommandPalette(false);
            runCommandText(cmd);
            return;
        }
        if (e.key === 'Escape') {
            e.preventDefault();
            setCommandPalette(false);
        }
        e.stopPropagation();
    });
}

document.addEventListener('click', e => {
    const panel = document.getElementById('settingsPanel');
    const btn = document.getElementById('settingsBtn');
    if (!panel || !btn || panel.style.display === 'none') return;
    const t = e.target;
    if (panel.contains(t) || btn.contains(t)) return;
    panel.style.display = 'none';
});

// ── Keyboard ──
document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        openCommandPalette();
        return;
    }

    if (e.key === 'Escape') {
        const palette = document.getElementById('commandPalette');
        if (palette && palette.style.display !== 'none') {
            e.preventDefault();
            setCommandPalette(false);
            return;
        }
        const settingsPanel = document.getElementById('settingsPanel');
        if (settingsPanel && settingsPanel.style.display !== 'none') {
            e.preventDefault();
            settingsPanel.style.display = 'none';
            return;
        }
    }

    const tag = e.target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') {
        if (e.key === 'Escape') {
            e.target.blur(); searchQuery = ''; searchKernelMatches.clear(); searchNvtxMatches.clear();
            document.getElementById('searchInput').value = ''; document.getElementById('searchHint').textContent = ''; draw();
        }
        return;
    }

    const helpEl = document.getElementById('helpOverlay');
    if (helpEl.style.display === 'block' && e.key !== '?') { helpEl.style.display = 'none'; return; }

    switch (e.key) {
        case 'ArrowLeft': case 'h': e.preventDefault(); panBy(-0.15); break;
        case 'ArrowRight': case 'l': e.preventDefault(); panBy(0.15); break;
        case 'ArrowUp': case 'k':
            e.preventDefault();
            selectedStreamIdx = Math.max(0, selectedStreamIdx - 1);
            updateNvtxThreadOptions();
            maybeFetchNvtxForCurrentView();
            draw();
            break;
        case 'ArrowDown': case 'j':
            e.preventDefault();
            selectedStreamIdx = Math.min(streamIds.length - 1, selectedStreamIdx + 1);
            updateNvtxThreadOptions();
            maybeFetchNvtxForCurrentView();
            draw();
            break;
        case '+': case '=': zoomIn(); break;
        case '-': case '_': zoomOut(); break;
        case 'Home': case '0': fitAll(); break;
        case 'Tab':
            e.preventDefault();
            if (selectedNvtx) {
                if (!nextNvtx(e.shiftKey ? -1 : 1)) {
                    showToast('No more NVTX at this level');
                }
            } else {
                nextKernel(e.shiftKey ? -1 : 1);
            }
            break;
        case 'g': case 'G': e.preventDefault(); gotoTimePrompt(); break;
        case 'v': case 'V': e.preventDefault(); gotoNvtxPrompt(); break;
        case '/': e.preventDefault(); document.getElementById('searchInput').focus(); break;
        case 'Escape':
            selectedKernel = null; selectedNvtx = null; selectedFindingIdx = -1;
            document.querySelectorAll('.finding-card').forEach(el => el.classList.remove('active'));
            showDetail(null);
            searchQuery = ''; searchKernelMatches.clear(); searchNvtxMatches.clear();
            document.getElementById('searchInput').value = '';
            document.getElementById('searchHint').textContent = '';
            document.getElementById('gpuInfoPanel').style.display = 'none';
            draw(); break;
        case 'n': case 'N': toggleNVTX(); break;
        case 'a': case 'A': toggleChat(); break;
        case 'e': case 'E': if (FINDINGS.length) toggleFindings(); break;
        case 'f': case 'F': fitAll(); break;
        case 'b': saveBookmark(); break;
        case 'B': toggleBookmarkList(); break;
        case 's': case 'S': toggleStreamFilter(); break;
        case '[':
            if (FINDINGS.length) {
                e.preventDefault();
                const prevIdx = selectedFindingIdx <= 0 ? FINDINGS.length - 1 : selectedFindingIdx - 1;
                selectFinding(prevIdx);
                if (!document.getElementById('findingsSidebar').classList.contains('open')) toggleFindings();
            }
            break;
        case ']':
            if (FINDINGS.length) {
                e.preventDefault();
                const nextIdx = selectedFindingIdx >= FINDINGS.length - 1 ? 0 : selectedFindingIdx + 1;
                selectFinding(nextIdx);
                if (!document.getElementById('findingsSidebar').classList.contains('open')) toggleFindings();
            }
            break;
        case '?': toggleHelp(); break;
        default:
            // 1-9 jump to bookmark
            if (e.key >= '1' && e.key <= '9') {
                const bi = parseInt(e.key) - 1;
                if (bi < bookmarks.length) { jumpToBookmark(bi); }
            }
            break;
    }
});

// ── AI Chat ──
let chatHistory = [];  // [{role, content}]
let chatStreaming = false;

function toggleChat() {
    const sidebar = document.getElementById('chatSidebar');
    const btn = document.getElementById('chatBtn');
    sidebar.classList.toggle('open');
    btn.classList.toggle('active', sidebar.classList.contains('open'));
    if (sidebar.classList.contains('open')) {
        document.getElementById('chatInput').focus();
    }
    setTimeout(resize, 0);
}

function buildUIContext() {
    return {
        selected_kernel: selectedKernel ? {
            name: selectedKernel.name,
            duration_ms: selectedKernel.duration_ms,
            stream: selectedKernel.stream,
        } : null,
        view_state: {
            time_range_ns: [viewStart, viewEnd],
            visible_span_ms: (viewEnd - viewStart) / 1e6,
            mode: 'WEB',
        },
        stats: { kernel_count: kernels.length, stream_count: streamIds.length },
    };
}

function appendChatMsg(role, content, cls) {
    const msgs = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'chat-msg ' + (cls || role);
    div.innerHTML = formatChatContent(content);
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    return div;
}

function formatChatContent(text) {
    // Simple markdown-ish rendering
    let html = text
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/```([\s\S]*?)```/g, '<pre>$1</pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
    // Linkify [Finding N] references → click to zoom to evidence
    html = html.replace(/\[Finding\s+(\d+)\]/gi, (match, n) => {
        const idx = parseInt(n) - 1;
        return '<a href="#" class="finding-link" onclick="selectFinding(' + idx + ');return false">' + match + '</a>';
    });
    return html;
}

async function sendChat() {
    const input = document.getElementById('chatInput');
    const msg = input.value.trim();
    if (!msg || chatStreaming) return;
    input.value = '';

    appendChatMsg('user', msg);
    chatHistory.push({ role: 'user', content: msg });

    chatStreaming = true;
    const sendBtn = document.getElementById('chatSendBtn');
    sendBtn.disabled = true;
    sendBtn.textContent = '...';

    const aiDiv = appendChatMsg('ai', '');
    aiDiv.classList.add('chat-streaming');
    let fullContent = '';
    const ctrl = new AbortController();
    const maxMs = 180000;
    const abortTimer = setTimeout(() => {
        try { ctrl.abort(); } catch (e) { }
    }, maxMs);
    let streamDone = false;

    try {
        const modelSel = document.getElementById('chatModel');
        const model = modelSel.value || undefined;
        const body = {
            messages: chatHistory,
            model: model,
            stream: true,
            ui_context: buildUIContext(),
            profile_path: BOOT.PROFILE_PATH || undefined,
        };
        lastChatTs = Date.now();

        const pfx = BOOT.API_PREFIX || '';
        const resp = await fetch(`${pfx}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: ctrl.signal,
        });

        if (!resp.ok) {
            const err = await resp.text();
            aiDiv.innerHTML = formatChatContent('Error: ' + err);
            aiDiv.classList.remove('chat-streaming');
            chatStreaming = false;
            sendBtn.disabled = false;
            sendBtn.textContent = 'Send';
            return;
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            if (streamDone) {
                try { await reader.cancel(); } catch (e) { }
                break;
            }
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            // Parse SSE blocks separated by blank line.
            while (true) {
                const sep = buffer.indexOf('\n\n');
                if (sep < 0) break;
                const rawEvent = buffer.slice(0, sep);
                buffer = buffer.slice(sep + 2);
                const eventLines = rawEvent.split('\n');
                let currentEvent = 'message';
                let data = '';
                for (const rawLine of eventLines) {
                    const line = rawLine.replace(/\r$/, '');
                    if (line.startsWith('event:')) {
                        currentEvent = line.slice(6).trim();
                        continue;
                    }
                    if (line.startsWith('data:')) {
                        data += line.slice(5).trimStart();
                    }
                }
                if (!data || data === '[DONE]') continue;
                try {
                    const payload = JSON.parse(data);
                    if (currentEvent === 'text') {
                        const chunk = payload.chunk || payload.content || '';
                        fullContent += chunk;
                        aiDiv.innerHTML = formatChatContent(fullContent);
                        document.getElementById('chatMessages').scrollTop =
                            document.getElementById('chatMessages').scrollHeight;
                    } else if (currentEvent === 'system') {
                        appendChatMsg('system', payload.content || '', 'system');
                    } else if (currentEvent === 'done') {
                        // done event — may contain usage info
                        if (payload.content) {
                            fullContent = payload.content;
                            aiDiv.innerHTML = formatChatContent(fullContent);
                        }
                        streamDone = true;
                    } else if (currentEvent === 'action') {
                        executeAIAction(payload);
                    } else if (currentEvent === 'finding') {
                        const idx = Number.isFinite(payload.index)
                            ? payload.index
                            : addFinding(payload);
                        appendChatMsg('system',
                            `📋 Finding ${idx}: ${payload.label || ''}`, 'system');
                    }
                } catch (e) { /* ignore parse errors in SSE */ }
                if (streamDone) break;
            }
        }

        chatHistory.push({ role: 'assistant', content: fullContent });

        // Check for inline actions in the response
        parseAndExecuteActions(fullContent);

    } catch (err) {
        aiDiv.innerHTML = formatChatContent('Connection error: ' + err.message);
    } finally {
        clearTimeout(abortTimer);
        aiDiv.classList.remove('chat-streaming');
        chatStreaming = false;
        sendBtn.disabled = false;
        sendBtn.textContent = 'Send';
        input.focus();
    }
}

function executeAIAction(action) {
    if (action.type === 'navigate_to_kernel' || (action.action_type === 'navigate_to_kernel')) {
        const target = action.target_name || action.target;
        if (target) {
            const k = kernels.find(k => k.name.includes(target));
            if (k) {
                const streamKey = k._streamKey != null ? k._streamKey : String(k.stream);
                const si = streamIds.indexOf(streamKey);
                if (si >= 0) selectedStreamIdx = si;
                updateNvtxThreadOptions();
                maybeFetchNvtxForCurrentView();
                selectKernel(k);
                ensureVisible(k);
                appendChatMsg('system', '→ Navigated to: ' + k.name.slice(0, 60), 'system');
            }
        }
    } else if (action.type === 'zoom_to_time_range' || (action.action_type === 'zoom_to_time_range')) {
        const startS = action.start_s;
        const endS = action.end_s;
        if (startS !== undefined && endS !== undefined) {
            viewStart = startS * 1e9;
            viewEnd = endS * 1e9;
            draw();
            appendChatMsg('system', '→ Zoomed to ' + startS.toFixed(3) + 's – ' + endS.toFixed(3) + 's', 'system');
        }
    } else if (action.type === 'fit_nvtx_range' || (action.action_type === 'fit_nvtx_range')) {
        const targetName = action.nvtx_name || action.target_name || action.target;
        if (targetName) {
            const q = String(targetName).toLowerCase();
            const all = activeGpuNvtxSpans().spans;
            const occ = Math.max(1, parseInt(action.occurrence_index || 1, 10) || 1);
            const matches = all.filter(s => (s.name || '').toLowerCase().includes(q));
            const target = matches[Math.min(occ - 1, matches.length - 1)];
            if (target) {
                showNVTX = true;
                document.getElementById('nvtxBtn').classList.add('active');
                if (target.thread) {
                    selectedNvtxThread = target.thread;
                    updateNvtxThreadOptions();
                }
                const laid = layoutNvtxSpans([target], NVTX_PIN_ROWS).spans[0]
                    || { ...target, _lane: NVTX_PIN_ROWS - 1, _key: nvtxKey(target), _clipped: true };
                selectNvtx(laid);
                const fit = computeFitViewRange(target.start, target.end);
                viewStart = fit.start;
                viewEnd = fit.end;
                clampView();
                draw();
                afterViewChange();
                appendChatMsg('system', '→ Fitted NVTX: ' + target.name, 'system');
                return;
            }
        }
        const startS = action.start_s;
        const endS = action.end_s;
        if (startS !== undefined && endS !== undefined) {
            const fit = computeFitViewRange(startS * 1e9, endS * 1e9);
            viewStart = fit.start;
            viewEnd = fit.end;
            clampView();
            draw();
            afterViewChange();
            appendChatMsg('system', '→ Fitted NVTX range ' + startS.toFixed(3) + 's – ' + endS.toFixed(3) + 's', 'system');
        }
    }
}

function parseAndExecuteActions(content) {
    // Look for JSON action blocks in the response
    const actionRe = /```json\s*({[^}]*"type"\s*:\s*"(navigate_to_kernel|zoom_to_time_range|fit_nvtx_range)"[^}]*})\s*```/g;
    let match;
    while ((match = actionRe.exec(content)) !== null) {
        try {
            executeAIAction(JSON.parse(match[1]));
        } catch (e) { /* ignore */ }
    }
}

// Chat input: Enter to send
document.getElementById('chatInput').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChat();
    }
    if (e.key === 'Escape') {
        e.target.blur();
    }
    e.stopPropagation();  // Don't trigger timeline shortcuts
});

// Load available models
async function loadModels() {
    try {
        const pfx = BOOT.API_PREFIX || '';
        const resp = await fetch(`${pfx}/api/models`);
        const data = await resp.json();
        const sel = document.getElementById('chatModel');
        if (data.options && data.options.length) {
            data.options.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.id;
                opt.textContent = m.label;
                if (m.id === data.default) opt.selected = true;
                sel.appendChild(opt);
            });
        } else {
            const opt = document.createElement('option');
            opt.textContent = 'No models configured';
            opt.disabled = true;
            sel.appendChild(opt);
        }
    } catch (e) {
        console.warn('Failed to load models:', e);
    }
}
loadModels();
wireSettingsPanel();

window.addEventListener('message', e => {
    if (e.data && e.data.type === 'T_SET_VIEW') {
        viewStart = e.data.startNs;
        viewEnd = e.data.endNs;
        clampView();
        draw();
        if (PROGRESSIVE) ensureTilesForView(viewStart, viewEnd);
    }
});

// ── Init ──
setTimeout(initData, 0);

// ── Findings Evidence Layer ──
let selectedFindingIdx = -1;

function _findingSevColor(sev) {
    if (sev === 'critical') return '#ff4444';
    if (sev === 'warning') return '#d4a72c';
    return '#58a6ff';
}

function _findingStreamRange(f) {
    // Returns { y, h } in canvas coordinates for this finding's target area
    const topY = RULER_H + findingsLaneH() + nvtxAreaH();
    const totalH = streamIds.length * (STREAM_H + STREAM_GAP) +
        (isMultiGPU ? gpuBands.length * GPU_SEP_H : 0);

    if (f.gpu_id != null && isMultiGPU) {
        const band = gpuBands.find(b => b.gpuId === f.gpu_id);
        if (band) {
            if (f.stream != null) {
                const streamKey = band.gpuId + ':' + f.stream;
                const si = streamIds.indexOf(streamKey);
                if (si >= 0) {
                    const y = streamY(si) + GPU_SEP_H;
                    return { y, h: STREAM_H };
                }
            }
            const y0 = streamY(band.startIdx) + GPU_SEP_H;
            const yEnd = band.endIdx < streamIds.length
                ? streamY(band.endIdx)
                : topY + totalH;
            return { y: y0, h: yEnd - y0 };
        }
    }

    if (f.stream != null && !isMultiGPU) {
        const si = streamIds.indexOf(String(f.stream));
        if (si >= 0) {
            return { y: streamY(si), h: STREAM_H };
        }
    }

    return { y: topY, h: totalH };
}

function drawFindings(W, H) {
    const font = ctx.font;
    ctx.save();

    const hasActive = selectedFindingIdx >= 0 && selectedFindingIdx < FINDINGS.length;
    const laneY = RULER_H;  // annotation lane starts right below the ruler
    const laneH = findingsLaneH();

    // ── Pass 0: Draw annotation lane background ──
    if (laneH > 0) {
        ctx.fillStyle = '#0f141d';
        ctx.fillRect(0, laneY, W, laneH);
        ctx.strokeStyle = '#21262d';
        ctx.beginPath();
        ctx.moveTo(0, laneY + laneH - 0.5);
        ctx.lineTo(W, laneY + laneH - 0.5);
        ctx.stroke();

        // Label on the left
        ctx.fillStyle = '#6e7681';
        ctx.font = '9px SF Mono, monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText('Findings', LABEL_W - 6, laneY + laneH / 2);
    }

    // ── Pass 1: Focus dim — darken everything OUTSIDE the active finding ──
    if (hasActive) {
        const af = FINDINGS[selectedFindingIdx];
        const aStart = af.start_ns;
        const aEnd = af.end_ns ?? aStart;
        const ax1 = nsToX(aStart);
        const ax2 = nsToX(aEnd);
        const sevCol = _findingSevColor(af.severity);

        // Heavy dim
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        if (ax1 > LABEL_W) ctx.fillRect(LABEL_W, laneY + laneH, ax1 - LABEL_W, H - laneY - laneH);
        if (ax2 < W) ctx.fillRect(ax2, laneY + laneH, W - ax2, H - laneY - laneH);

        // Bright accent border around the un-dimmed window
        ctx.strokeStyle = sevCol;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        ctx.moveTo(ax1, laneY + laneH); ctx.lineTo(ax1, H);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(ax2, laneY + laneH); ctx.lineTo(ax2, H);
        ctx.stroke();
        ctx.lineWidth = 1;
        ctx.globalAlpha = 1;
    }

    // ── Pass 2: Draw region/highlight/marker overlays on streams ──
    for (let i = 0; i < FINDINGS.length; i++) {
        const f = FINDINGS[i];
        const startNs = f.start_ns;
        const endNs = f.end_ns ?? startNs;
        if (endNs < viewStart || startNs > viewEnd) continue;

        const x1 = Math.max(LABEL_W, nsToX(startNs));
        const x2 = Math.min(W, nsToX(endNs));
        const pixW = Math.max(x2 - x1, 2);
        const isActive = i === selectedFindingIdx;
        const sevCol = _findingSevColor(f.severity);
        const { y: targetY, h: targetH } = _findingStreamRange(f);
        const baseAlpha = hasActive && !isActive ? 0.1 : 1;

        if (f.type === 'region' || f.type === 'highlight') {
            ctx.globalAlpha = baseAlpha;

            // Fill
            const fillAlpha = isActive ? 0.25 : 0.08;
            ctx.fillStyle = sevCol;
            ctx.globalAlpha = baseAlpha * fillAlpha;
            ctx.fillRect(x1, targetY, pixW, targetH);

            // Glow for active
            if (isActive) {
                ctx.save();
                ctx.shadowColor = sevCol;
                ctx.shadowBlur = 12;
                ctx.strokeStyle = sevCol;
                ctx.lineWidth = 2.5;
                ctx.globalAlpha = 0.8;
                ctx.strokeRect(x1, targetY, pixW, targetH);
                ctx.restore();
            } else {
                // Subtle border for inactive
                ctx.strokeStyle = sevCol;
                ctx.lineWidth = 1;
                ctx.globalAlpha = baseAlpha * 0.4;
                ctx.setLineDash([4, 4]);
                ctx.strokeRect(x1, targetY, pixW, targetH);
                ctx.setLineDash([]);
            }
            ctx.lineWidth = 1;

        } else if (f.type === 'marker') {
            const x = nsToX(startNs);
            if (x < LABEL_W || x > W) continue;

            ctx.globalAlpha = baseAlpha;
            ctx.strokeStyle = sevCol;
            ctx.lineWidth = isActive ? 3 : 2;
            ctx.globalAlpha = baseAlpha * (isActive ? 1 : 0.6);
            ctx.setLineDash([6, 4]);
            ctx.beginPath();
            ctx.moveTo(x, laneY + laneH);
            ctx.lineTo(x, targetY + targetH);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.lineWidth = 1;
        }

        ctx.globalAlpha = 1;
    }

    // ── Pass 3: Draw annotation lane pills + connecting lines ──
    if (laneH > 0) {
        for (let i = 0; i < FINDINGS.length; i++) {
            const f = FINDINGS[i];
            const startNs = f.start_ns;
            const endNs = f.end_ns ?? startNs;
            if (endNs < viewStart || startNs > viewEnd) continue;

            const x1 = Math.max(LABEL_W, nsToX(startNs));
            const x2 = Math.min(W, nsToX(endNs));
            const midX = (x1 + x2) / 2;
            const isActive = i === selectedFindingIdx;
            const sevCol = _findingSevColor(f.severity);
            const baseAlpha = hasActive && !isActive ? 0.2 : 1;
            const { y: targetY, h: targetH } = _findingStreamRange(f);

            // ── Connecting dotted line from pill to target region ──
            ctx.globalAlpha = baseAlpha * (isActive ? 0.6 : 0.25);
            ctx.strokeStyle = sevCol;
            ctx.lineWidth = isActive ? 1.5 : 1;
            ctx.setLineDash(isActive ? [6, 3] : [3, 5]);
            ctx.beginPath();
            ctx.moveTo(midX, laneY + laneH);
            ctx.lineTo(midX, targetY);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.lineWidth = 1;

            // ── Pill bar in annotation lane ──
            const pillH = laneH - 6;
            const pillY = laneY + 3;
            const pillX1 = Math.max(LABEL_W + 2, x1);
            const pillX2 = Math.min(W - 2, x2);
            let pillW = Math.max(pillX2 - pillX1, 24);  // min 24px
            // Prevent pills from going off screen
            const clampedX = Math.max(LABEL_W + 2, Math.min(pillX1, W - pillW - 2));

            // Pill background
            ctx.globalAlpha = baseAlpha * (isActive ? 0.9 : 0.5);
            ctx.fillStyle = sevCol;
            ctx.beginPath();
            ctx.roundRect(clampedX, pillY, pillW, pillH, 4);
            ctx.fill();

            // Active glow
            if (isActive) {
                ctx.save();
                ctx.shadowColor = sevCol;
                ctx.shadowBlur = 8;
                ctx.strokeStyle = sevCol;
                ctx.lineWidth = 1.5;
                ctx.globalAlpha = 0.6;
                ctx.beginPath();
                ctx.roundRect(clampedX, pillY, pillW, pillH, 4);
                ctx.stroke();
                ctx.restore();
            }

            // Number circle inside pill
            const circR = pillH / 2 - 1;
            const circX = clampedX + circR + 3;
            const circCY = pillY + pillH / 2;
            ctx.globalAlpha = baseAlpha;
            ctx.fillStyle = '#0d1117';
            ctx.beginPath();
            ctx.arc(circX, circCY, circR, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = sevCol;
            ctx.font = 'bold 9px SF Mono, monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(i + 1), circX, circCY);

            // Label text (truncated to fit)
            if (pillW > 40) {
                ctx.fillStyle = '#0d1117';
                ctx.font = (isActive ? 'bold ' : '') + '9px SF Mono, monospace';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                const textStart = circX + circR + 4;
                const maxTextW = pillW - (textStart - clampedX) - 4;
                if (maxTextW > 10) {
                    const maxChars = Math.floor(maxTextW / 5.5);
                    const truncLabel = f.label.length > maxChars
                        ? f.label.slice(0, maxChars - 1) + '…' : f.label;
                    ctx.fillText(truncLabel, textStart, circCY);
                }
            }

            ctx.globalAlpha = 1;
        }
    }

    // ── Pass 4: Active finding stream label flash ──
    if (hasActive) {
        const af = FINDINGS[selectedFindingIdx];
        const sevCol = _findingSevColor(af.severity);
        // Flash the stream label background
        if (af.gpu_id != null && isMultiGPU) {
            const band = gpuBands.find(b => b.gpuId === af.gpu_id);
            if (band) {
                for (let si = band.startIdx; si < band.endIdx; si++) {
                    const sy = streamY(si) + (si === band.startIdx ? GPU_SEP_H : 0);
                    ctx.fillStyle = sevCol;
                    ctx.globalAlpha = 0.12;
                    ctx.fillRect(0, sy, LABEL_W, STREAM_H);
                    ctx.globalAlpha = 1;
                }
            }
        } else if (af.stream != null && !isMultiGPU) {
            const si = streamIds.indexOf(String(af.stream));
            if (si >= 0) {
                const sy = streamY(si);
                ctx.fillStyle = sevCol;
                ctx.globalAlpha = 0.12;
                ctx.fillRect(0, sy, LABEL_W, STREAM_H);
                ctx.globalAlpha = 1;
            }
        }
    }

    ctx.restore();
    ctx.font = font;
}

// ── Findings hit-test (for annotation lane pills and region overlays) ──
function _findingsHitTest(mx, my) {
    const laneY = RULER_H;
    const laneH = findingsLaneH();

    // Check annotation lane pills
    if (laneH > 0 && my >= laneY && my <= laneY + laneH) {
        for (let i = FINDINGS.length - 1; i >= 0; i--) {
            const f = FINDINGS[i];
            const startNs = f.start_ns;
            const endNs = f.end_ns ?? startNs;
            if (endNs < viewStart || startNs > viewEnd) continue;
            const x1 = Math.max(LABEL_W, nsToX(startNs));
            const x2 = Math.min(canvas.width / DPR, nsToX(endNs));
            const pillW = Math.max(x2 - x1, 24);
            const clampedX = Math.max(LABEL_W + 2, Math.min(x1, canvas.width / DPR - pillW - 2));
            if (mx >= clampedX && mx <= clampedX + pillW) {
                return i;
            }
        }
    }

    // Check stream-area region/highlight overlays
    for (let i = FINDINGS.length - 1; i >= 0; i--) {
        const f = FINDINGS[i];
        const startNs = f.start_ns;
        const endNs = f.end_ns ?? startNs;
        if (endNs < viewStart || startNs > viewEnd) continue;
        const x1 = Math.max(LABEL_W, nsToX(startNs));
        const x2 = Math.min(canvas.width / DPR, nsToX(endNs));
        const { y: targetY, h: targetH } = _findingStreamRange(f);
        if (mx >= x1 && mx <= x2 && my >= targetY && my <= targetY + targetH) {
            return i;
        }
    }
    return -1;
}


// ── Findings sidebar ──
function _fmtFindingNs(ns) {
    const s = ns / 1e9;
    if (s >= 1) return s.toFixed(3) + 's';
    const ms = ns / 1e6;
    if (ms >= 1) return ms.toFixed(2) + 'ms';
    return (ns / 1e3).toFixed(1) + 'μs';
}

function _initFindingsSidebar() {
    if (!FINDINGS.length) return;

    const btn = document.getElementById('findingsBtn');
    if (btn) btn.style.display = '';

    const badge = document.getElementById('findingsCount');
    if (badge) {
        const crit = FINDINGS.filter(f => f.severity === 'critical').length;
        badge.textContent = FINDINGS.length + ' finding' + (FINDINGS.length !== 1 ? 's' : '');
        if (crit) badge.style.background = 'rgba(255,68,68,0.2)';
    }

    const list = document.getElementById('findingsList');
    if (!list) return;
    list.innerHTML = '';  // Clear existing cards for re-initialization

    FINDINGS.forEach((f, i) => {
        const card = document.createElement('div');
        card.className = 'finding-card';
        card.dataset.idx = i;

        const sevCol = _findingSevColor(f.severity);
        const rangeText = f.end_ns
            ? _fmtFindingNs(f.start_ns) + ' – ' + _fmtFindingNs(f.end_ns)
            : '@ ' + _fmtFindingNs(f.start_ns);

        const noteSnippet = (f.note || '').replace(/\n/g, ' ').slice(0, 80);

        card.innerHTML =
            '<div class="fc-header">' +
                '<span class="fc-badge" style="background:' + sevCol + '">' + (i + 1) + '</span>' +
                '<span class="fc-label">' + _esc(f.label) + '</span>' +
            '</div>' +
            '<div class="fc-severity ' + (f.severity || 'info') + '">' +
                (f.severity || 'info').toUpperCase() + ' · ' + f.type +
                (f.gpu_id != null ? ' · GPU ' + f.gpu_id : '') +
            '</div>' +
            '<div class="fc-range">' + rangeText + '</div>' +
            (noteSnippet ? '<div class="fc-note">' + _esc(noteSnippet) + '</div>' : '');

        card.style.borderLeftColor = sevCol;

        card.addEventListener('click', () => selectFinding(i));
        list.appendChild(card);
    });
}

function selectFinding(idx) {
    selectedFindingIdx = idx;
    const f = FINDINGS[idx];
    if (!f) return;

    // Update sidebar active state
    document.querySelectorAll('.finding-card').forEach((el, i) => {
        el.classList.toggle('active', i === idx);
    });
    const activeCard = document.querySelector('.finding-card.active');
    if (activeCard) activeCard.scrollIntoView({ block: 'nearest' });

    // Update detail panel
    const detail = document.getElementById('detail');
    if (detail) {
        detail.classList.remove('empty', 'collapsed');
        const sev = f.severity || 'info';
        const rangeText = f.end_ns
            ? _fmtFindingNs(f.start_ns) + ' → ' + _fmtFindingNs(f.end_ns) +
              ' (' + _fmtFindingNs(f.end_ns - f.start_ns) + ')'
            : '@ ' + _fmtFindingNs(f.start_ns);
        detail.innerHTML =
            '<div style="color:' + _findingSevColor(sev) + ';font-weight:600;font-size:13px">' +
                '📋 ' + _esc(f.label) + '</div>' +
            '<div style="font-size:11px;color:#8b949e;margin-top:2px">' +
                sev.toUpperCase() + ' · ' + f.type + ' · ' + rangeText +
                (f.gpu_id != null ? ' · GPU ' + f.gpu_id : '') +
                (f.stream != null ? ' · Stream ' + f.stream : '') +
            '</div>' +
            (f.note ? '<div style="font-size:11px;color:#e6edf3;margin-top:6px;white-space:pre-wrap;line-height:1.5">' +
                _esc(f.note) + '</div>' : '');
    }

    // ── Smart zoom: center the finding, range = 50% of viewport ──
    const startNs = f.start_ns;
    const endNs = f.end_ns ?? startNs;
    const span = endNs - startNs || 1e6;  // at least 1ms for markers
    // Finding should occupy half the viewport → viewport = span * 2
    // Center it → pad each side by span * 0.5
    const pad = span * 0.5;
    viewStart = startNs - pad;
    viewEnd = endNs + pad;
    clampView();
    draw();
    if (PROGRESSIVE) ensureTilesForView(viewStart, viewEnd);
}

function toggleFindings() {
    const sidebar = document.getElementById('findingsSidebar');
    const btn = document.getElementById('findingsBtn');
    if (!sidebar) return;
    sidebar.classList.toggle('open');
    if (btn) btn.classList.toggle('active', sidebar.classList.contains('open'));
    setTimeout(resize, 0);

    if (sidebar.classList.contains('open') && selectedFindingIdx < 0 && FINDINGS.length) {
        selectFinding(0);
    }
}

function _esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Analyze button + live finding injection ──

async function runAnalyze() {
    const btn = document.getElementById('analyzeBtn');
    if (!btn) return;
    btn.disabled = true;
    btn.textContent = '⏳ Analyzing...';
    try {
        const pfx = BOOT.API_PREFIX || '';
        const resp = await fetch(`${pfx}/api/analyze`, { method: 'POST' });
        if (!resp.ok) throw new Error(await resp.text());
        const newFindings = await resp.json();
        FINDINGS.length = 0;
        FINDINGS.push(...newFindings);
        selectedFindingIdx = -1;
        _initFindingsSidebar();
        // Auto-open sidebar + select first finding
        const sidebar = document.getElementById('findingsSidebar');
        if (sidebar && FINDINGS.length > 0) {
            sidebar.classList.add('open');
            const fbtn = document.getElementById('findingsBtn');
            if (fbtn) { fbtn.classList.add('active'); fbtn.style.display = ''; }
            setTimeout(resize, 0);
            selectFinding(0);
        }
        btn.textContent = `✓ ${FINDINGS.length} findings`;
    } catch (err) {
        btn.textContent = '❌ Error';
        console.error('Analyze failed:', err);
    }
    setTimeout(() => {
        btn.disabled = false;
        btn.textContent = '🔍 Analyze';
    }, 3000);
}

function addFinding(findingDict) {
    FINDINGS.push(findingDict);
    selectedFindingIdx = -1;
    _initFindingsSidebar();
    // Show sidebar if hidden
    const sidebar = document.getElementById('findingsSidebar');
    const fbtn = document.getElementById('findingsBtn');
    if (sidebar && !sidebar.classList.contains('open')) {
        sidebar.classList.add('open');
        if (fbtn) { fbtn.classList.add('active'); fbtn.style.display = ''; }
        setTimeout(resize, 0);
    }
    draw();
    return FINDINGS.length;  // 1-based index
}

// Initialize findings sidebar
_initFindingsSidebar();

// Auto-open findings sidebar if findings exist
if (FINDINGS.length > 0) {
    setTimeout(() => toggleFindings(), 300);
}
