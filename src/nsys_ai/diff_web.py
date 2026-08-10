"""
diff_web.py — Web compare server for before/after profiles.

This is a lightweight Phase B implementation that reuses the shared diff
engine (`diff.ProfileDiffSummary`) and exposes a small JSON API plus a
simple HTML shell.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from .diff import ProfileDiffSummary, diff_profiles
from .diff_render import to_diff_json
from .profile import Profile
from .viewer import build_timeline_gpu_data, generate_timeline_html
from .web import _TEMPLATE_DIR, _bind_local_server, _run_server


class _DiffHandler(BaseHTTPRequestHandler):
    """Serve a minimal web diff viewer and JSON APIs."""

    before: Profile | None = None
    after: Profile | None = None
    gpu: int | None = None
    trim: tuple[int, int] | None = None
    summary: ProfileDiffSummary | None = None

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/diff/meta":
            self._handle_meta()
            return
        if path == "/api/diff/summary":
            self._handle_summary()
            return

        # Iframe timelines
        if path.startswith("/timeline"):
            qs = parse_qs(urlparse(self.path).query)
            side = str(qs.get("side", ["before"])[0]).lower()
            self._serve_timeline_iframe(side)
            return

        if path.startswith("/api/timeline/"):
            parts = path.split("/")
            if len(parts) >= 6 and parts[4] == "api":
                side = parts[3]
                endpoint = parts[5]
                if endpoint == "meta":
                    self._handle_timeline_meta(side)
                    return
                elif endpoint == "data":
                    self._handle_timeline_data(side)
                    return
                elif endpoint == "models":
                    self._json_response({"options": [], "default": None})
                    return
                elif endpoint == "chat":
                    self._json_response(
                        {"error": "Chat is not available in diff-web iframe"},
                        status=501,
                    )
                    return
                else:
                    self._json_response(
                        {"error": "Unknown timeline endpoint", "endpoint": endpoint},
                        status=404,
                    )
                    return

        # Serve timeline assets if requested
        if path == "/assets/timeline.css":
            self._serve_asset("timeline.css", "text/css; charset=utf-8")
            return

        if path == "/assets/timeline.js":
            self._serve_asset("timeline.js", "application/javascript; charset=utf-8")
            return

        # Fallback: serve HTML shell.
        self._serve_html()

    def _serve_asset(self, filename: str, content_type: str):
        # Reuse timeline assets from the single-profile viewer
        try:
            path = os.path.join(_TEMPLATE_DIR, filename)
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ── JSON helpers ──────────────────────────────────────────────

    def _json_response(self, obj, status: int = 200):
        body = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_meta(self):
        if self.__class__.before is None or self.__class__.after is None:
            self._json_response({"error": "profiles not loaded"}, 500)
            return
        before = self.__class__.before
        after = self.__class__.after
        gpu = self.__class__.gpu
        trim = self.__class__.trim
        self._json_response(
            {
                "before_path": getattr(before, "path", ""),
                "after_path": getattr(after, "path", ""),
                "gpu": gpu,
                "trim_ns": list(trim) if trim else None,
                "before_devices": getattr(before.meta, "devices", []),
                "after_devices": getattr(after.meta, "devices", []),
            }
        )

    def _handle_summary(self):
        summary = self.__class__.summary
        if summary is None:
            self._json_response({"error": "summary not ready"}, 500)
            return

        # reuse the CLI diff JSON renderer
        body = to_diff_json(summary).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_timeline_meta(self, side: str):
        prof = self.__class__.after if side == "after" else self.__class__.before
        if not prof:
            self._json_response({"error": "profile not loaded"}, 500)
            return

        gpu_filter = self.__class__.gpu
        devices = [gpu_filter] if gpu_filter is not None else getattr(prof.meta, "devices", [])

        gpu_infos = []
        for dev in devices:
            info = prof.meta.gpu_info.get(dev)
            label = f"GPU {dev}"
            if info:
                label += f" - {info.name} ({info.pci_bus}), {info.sm_count} SMs, {info.memory_bytes / 1e9:.0f}GB"
            gpu_infos.append({"id": dev, "label": label})

        trim = self.__class__.trim
        t_start, t_end = prof.meta.time_range
        if trim:
            t_start, t_end = trim

        self._json_response(
            {
                "time_range_ns": [t_start, t_end],
                "gpus": gpu_infos,
                "device_ids": devices,
            }
        )

    def _handle_timeline_data(self, side: str):
        qs = parse_qs(urlparse(self.path).query)
        prof = self.__class__.after if side == "after" else self.__class__.before

        if not prof:
            self._json_response({"error": "profile not loaded"}, 500)
            return

        try:
            start_s = float(qs.get("start_s", [0])[0])
            end_s = float(qs.get("end_s", [5])[0])
        except (ValueError, IndexError):
            start_s, end_s = 0, 5

        gpu_filter = self.__class__.gpu
        start_ns = int(start_s * 1e9)
        end_ns = int(end_s * 1e9)
        devices = [gpu_filter] if gpu_filter is not None else getattr(prof.meta, "devices", [])

        data = build_timeline_gpu_data(
            prof,
            devices,
            (start_ns, end_ns),
            include_kernels=True,
            include_nvtx=True,
        )

        body = json.dumps({"gpus": data}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_timeline_iframe(self, side: str):
        prof = self.__class__.after if side == "after" else self.__class__.before
        if not prof:
            self.send_error(404)
            return

        gpu_filter = self.__class__.gpu
        devices = [gpu_filter] if gpu_filter is not None else getattr(prof.meta, "devices", [])

        html = generate_timeline_html(
            prof,
            devices,
            None,  # Progressive
            api_prefix=f"/api/timeline/{side}",
        )
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ── HTML shell ────────────────────────────────────────────────

    def _serve_html(self):
        html = _DIFF_HTML
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Quieter server logs.
        pass


_DIFF_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>nsys-ai diff</title>
    <link rel="stylesheet" href="/assets/timeline.css" />
    <style>
      body { font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
             margin: 0; padding: 1.5rem; background: #050816; color: #f5f5f5;
             min-height: 100vh; overflow-y: auto; overflow-x: hidden; }
      h1 { margin-top: 0; font-size: 1.4rem; }
      .paths { font-size: 0.9rem; margin-bottom: 1rem; }
      .paths code { word-break: break-all; }
      .layout { display: grid; grid-template-columns: 1.4fr 1fr; gap: 1.5rem; align-items: flex-start; }
      .layout-timeline { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1.5rem; }
      .timeline-container { height: 400px; position: relative; border-radius: 0.5rem; overflow: hidden; border: 1px solid rgba(148,163,184,0.25); background: #000; }
      .card { background: rgba(15,23,42,0.9); border-radius: 0.75rem; padding: 1rem 1.25rem; box-shadow: 0 18px 45px rgba(15,23,42,0.9); border: 1px solid rgba(148,163,184,0.25);}
      .card h2 { margin: 0 0 0.5rem 0; font-size: 1.0rem; }

      .pill { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px; background: rgba(148,163,184,0.25); font-size: 0.75rem; margin-right: 0.25rem; }
      table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
      th, td { padding: 0.35rem 0.4rem; text-align: right; border-bottom: 1px solid rgba(30,41,59,0.9); }
      th:nth-child(2), td:nth-child(2) { text-align: left; }
      th { font-weight: 600; color: #e5e7eb; background: rgba(15,23,42,0.9); position: sticky; top: 0; }
      tbody tr:nth-child(even) { background: rgba(15,23,42,0.6); }
      .delta-bad { color: #f87171; }
      .delta-good { color: #4ade80; }
      .warnings { color: #facc15; font-size: 0.85rem; }
      .warnings li { margin-bottom: 0.15rem; }
      .section-title { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: #9ca3af; margin-bottom: 0.35rem; }
      .overlap-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.4rem 0.75rem; font-size: 0.8rem; }
      .overlap-grid div span { display: block; }
      .overlap-label { color: #9ca3af; }
      .overlap-value { font-weight: 600; }
      .badge { font-size: 0.75rem; padding: 0.05rem 0.4rem; border-radius: 999px; border: 1px solid rgba(148,163,184,0.4); }
    </style>
  </head>
  <body>
    <h1>nsys-ai &mdash; Profile Diff (Web)</h1>
    <div class="paths" id="paths"></div>
    <div class="layout-timeline">
      <div style="display: flex; flex-direction: column;">
        <div style="margin-bottom: 0.5rem; font-weight: 600;">Before</div>
        <div class="timeline-container">
          <iframe id="frame-before" src="/timeline?side=before" style="width:100%; height:100%; border:none;"></iframe>
        </div>
      </div>
      <div style="display: flex; flex-direction: column;">
        <div style="margin-bottom: 0.5rem; font-weight: 600;">After</div>
        <div class="timeline-container">
          <iframe id="frame-after" src="/timeline?side=after" style="width:100%; height:100%; border:none;"></iframe>
        </div>
      </div>
    </div>
    <div class="layout" style="margin-top: 1.5rem;">
      <section class="card">
        <div class="section-title">Summary</div>
        <h2>Overall &amp; Overlap</h2>
        <div id="summary-overall"></div>
        <div style="margin-top:0.75rem;">
          <div class="section-title">Warnings</div>
          <ul class="warnings" id="warnings"></ul>
        </div>
      </section>
      <section class="card">
        <div class="section-title">NVTX</div>
        <h2>NVTX Changes</h2>
        <div id="nvtx"></div>
      </section>
    </div>
    <section class="card" style="margin-top:1.5rem;">
      <div class="section-title">Kernels</div>
      <h2>Top regressions &amp; improvements</h2>
      <div id="kernels"></div>
    </section>
    <script>
      let syncIsActive = false;

      async function loadDiff() {
        const [metaRes, sumRes] = await Promise.all([
          fetch('/api/diff/meta'),
          fetch('/api/diff/summary'),
        ]);
        const meta = await metaRes.json();
        const data = await sumRes.json();
        renderPaths(meta, data);
        renderSummary(data);
        renderKernels(data);
        renderNvtx(data);

        // Setup Sync Logic for the Iframes
        window.addEventListener('message', e => {
            if (e.data && e.data.type === 'T_VIEW_CHANGE') {
                if (syncIsActive) return;
                syncIsActive = true;

                const msg = { type: 'T_SET_VIEW', startNs: e.data.startNs, endNs: e.data.endNs };
                const fb = document.getElementById('frame-before');
                const fa = document.getElementById('frame-after');

                if (e.source === fb.contentWindow) {
                    fa.contentWindow.postMessage(msg, '*');
                } else if (e.source === fa.contentWindow) {
                    fb.contentWindow.postMessage(msg, '*');
                }

                setTimeout(() => { syncIsActive = false; }, 50);
            }
        });
      }

      function fmtNs(ns) {
        const ms = ns / 1e6;
        if (Math.abs(ms) >= 1000) return (ms/1000).toFixed(2) + 's';
        if (Math.abs(ms) >= 1) return ms.toFixed(2) + 'ms';
        const us = ns / 1e3;
        if (Math.abs(us) >= 1) return us.toFixed(2) + 'µs';
        return ns + 'ns';
      }

      function renderPaths(meta, data) {
        const el = document.getElementById('paths');
        const gpu = data.before.gpu === null ? 'all GPUs' : 'GPU ' + data.before.gpu;
        el.innerHTML = '<div><span class=\"badge\">Before</span> <code>' +
          meta.before_path +
          '</code></div><div><span class=\"badge\">After</span> <code>' +
          meta.after_path +
          '</code></div><div style=\"margin-top:0.25rem;color:#9ca3af;font-size:0.8rem;\">View: ' +
          gpu + '</div>';
      }

      function renderSummary(d) {
        const el = document.getElementById('summary-overall');
        const totalDelta = d.after.total_gpu_ns - d.before.total_gpu_ns;
        const cls = totalDelta > 0 ? 'delta-bad' : (totalDelta < 0 ? 'delta-good' : '');
        let html = '<div><span class=\"overlap-label\">Total GPU:</span> ' +
          '<span class=\"overlap-value\">' + fmtNs(d.before.total_gpu_ns) + ' → ' +
          fmtNs(d.after.total_gpu_ns) + '</span> ' +
          '<span class=\"' + cls + '\">(' + (totalDelta >= 0 ? '+' : '') + fmtNs(totalDelta) + ')</span></div>';

        const keys = ['compute_only_ms','nccl_only_ms','overlap_ms','idle_ms','total_ms','overlap_pct'];
        const labels = {
          compute_only_ms: 'Compute only',
          nccl_only_ms: 'NCCL only',
          overlap_ms: 'Overlap',
          idle_ms: 'Idle',
          total_ms: 'Wall-clock span',
          overlap_pct: 'Overlap %',
        };
        html += '<div class=\"overlap-grid\" style=\"margin-top:0.75rem;\">';
        for (const k of keys) {
          if (!d.overlap || !(k in d.overlap.before) || !(k in d.overlap.after)) continue;
          const b = d.overlap.before[k];
          const a = d.overlap.after[k];
          const delta = d.overlap.delta[k] ?? (a - b);
          const kindClass = delta > 0 ? (k === 'idle_ms' ? 'delta-bad' : 'delta-bad') :
                            delta < 0 ? (k === 'idle_ms' ? 'delta-good' : 'delta-good') : '';
          html += '<div><span class=\"overlap-label\">' + labels[k] +
                  '</span><span class=\"overlap-value\">' + b + ' → ' + a +
                  '</span><span class=\"' + kindClass + '\">(' + (delta >= 0 ? '+' : '') +
                  parseFloat(delta).toFixed(1) + (k === 'overlap_pct' ? '%':'') + ')</span></div>';
        }

        html += '</div>';
        el.innerHTML = html;

        const w = document.getElementById('warnings');
        w.innerHTML = '';
        if (d.warnings && d.warnings.length) {
          d.warnings.forEach(msg => {
            const li = document.createElement('li');
            li.textContent = msg;
            w.appendChild(li);
          });
        } else {
          const li = document.createElement('li');
          li.textContent = 'No major sanity warnings.';
          w.appendChild(li);
        }
      }

      function renderKernels(d) {
        const el = document.getElementById('kernels');
        function table(rows, title) {
          if (!rows || !rows.length) {
            return '<div><strong>' + title + ':</strong> none</div>';
          }
          let html = '<div style=\"margin-top:0.5rem;\"><div class=\"section-title\">' +
                     title + '</div><table><thead><tr>' +
                     '<th>Δ time</th><th>kernel</th><th>before</th><th>after</th><th>count</th></tr></thead><tbody>';
          for (const k of rows) {
            const cls = k.delta_ns > 0 ? 'delta-bad' : (k.delta_ns < 0 ? 'delta-good' : '');
            const deltaStr = fmtNs(k.delta_ns);
            const cnt = k.before_count + ' → ' + k.after_count;
            html += '<tr><td class=\"' + cls + '\">' + (k.delta_ns >= 0 ? '+' : '') +
                    deltaStr + '</td><td>' + k.name + '</td>' +
                    '<td>' + fmtNs(k.before_total_ns) + '</td>' +
                    '<td>' + fmtNs(k.after_total_ns) + '</td>' +
                    '<td>' + cnt + '</td></tr>';
          }
          html += '</tbody></table></div>';
          return html;
        }
        el.innerHTML = table(d.top_regressions, 'Top regressions') +
                       table(d.top_improvements, 'Top improvements');
      }

      function renderNvtx(d) {
        const el = document.getElementById('nvtx');
        const reg = d.nvtx_regressions || [];
        const imp = d.nvtx_improvements || [];
        if (!reg.length && !imp.length) {
          el.innerHTML = '<div>no significant NVTX changes.</div>';
          return;
        }
        function list(items, title, cls) {
          if (!items.length) return '';
          let html = '<div style=\"margin-top:0.5rem;\"><div class=\"section-title\">' +
                     title + '</div><ul style=\"list-style:none;padding-left:0;font-size:0.8rem;\">';
          for (const n of items) {
            const deltaStr = fmtNs(n.delta_ns);
            html += '<li><span class=\"' + cls + '\">(' + (n.delta_ns >= 0 ? '+' : '') +
                    deltaStr + ')</span> ' + n.text +
                    ' <span style=\"color:#9ca3af;\">[' + n.before_total_ns + ' → ' +
                    n.after_total_ns + ' ns]</span></li>';
          }
          html += '</ul></div>';
          return html;
        }
        el.innerHTML = list(reg, 'NVTX regressions', 'delta-bad') +
                       list(imp, 'NVTX improvements', 'delta-good');
      }

      loadDiff().catch(err => {
        document.body.innerHTML = '<pre style="color:#f97373;">Failed to load diff: ' +
          String(err) + '</pre>';
      });
    </script>
  </body>
</html>
"""


def serve_diff_web(
    before: Profile,
    after: Profile,
    *,
    gpu: int | None,
    trim: tuple[int, int] | None,
    port: int = 8145,
    open_browser: bool = True,
) -> None:
    """Start a local HTTP server serving a minimal diff viewer."""
    summary = diff_profiles(before, after, gpu=gpu, trim=trim, limit=15)

    _DiffHandler.before = before
    _DiffHandler.after = after
    _DiffHandler.gpu = gpu
    _DiffHandler.trim = trim
    _DiffHandler.summary = summary

    server = _bind_local_server(port, _DiffHandler)

    open_url = f"http://127.0.0.1:{server.server_address[1]}" if open_browser else None
    _run_server(server, open_url, before)
