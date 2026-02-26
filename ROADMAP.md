# 🗺️ nsys-ai Roadmap

Two pillars: **UI** (making profiles effortless to view) and **AI** (making profiles effortless to understand).

---

## 🖥️ Pillar 1 — UI

> Goal: Zero-friction viewing of Nsight profiles across every surface — terminal, browser, VS Code.

### One-Click Perfetto (Server → Local)
- [ ] VSCode transport: remote SSH profile → local Perfetto UI in one click
- [ ] Auto-detect `.sqlite` / `.nsys-rep` on remote host, convert + stream to local
- [ ] `nsys-ai open profile.sqlite` — single command that picks the best viewer

### TUI
- [ ] Timeline TUI polish (bookmarks, annotation overlay, multi-GPU stacked view)
- [ ] Tree TUI improvements (inline flame-graph sparklines, diff mode for two profiles)
- [ ] Unified TUI launcher — auto-select timeline vs tree based on profile shape

### Web UI (custom)
- [ ] Self-hosted web viewer (`nsys-ai web`) — richer than Perfetto, profile-aware
- [ ] NVTX-aware flame chart with collapsible hierarchy
- [ ] Side-by-side comparison mode (two profiles / two iterations)
- [ ] Shareable links (serve from remote, view locally)

### Packaging & DX
- [ ] `pip install nsys-ai` → everything works, zero config
- [ ] VS Code extension stub (open `.sqlite` → launch viewer)
- [ ] Jupyter widget for inline profile viewing in notebooks

---

## 🤖 Pillar 2 — AI

> Goal: AI that understands GPU profiles as a first-class concept — integrated everywhere, not bolted on.

### AI in every interface
- [ ] TUI: inline AI commentary panel (press `?` on any kernel → explain)
- [ ] Web UI: chat widget — ask questions about the profile in natural language
- [ ] CLI: `nsys-ai ask "why is iteration 142 slow?"` → answer from profile data

### AI CLI (nsys as subcomponent)
- [ ] `nsys-ai analyze profile.sqlite` — full auto-report (bottlenecks, recommendations)
- [ ] `nsys-ai diff a.sqlite b.sqlite` — AI-narrated performance comparison
- [ ] `nsys-ai suggest` — NVTX annotation suggestions for unannotated regions
- [ ] `nsys-ai explain <kernel>` — deep-dive on a specific kernel's behavior

### AI backend
- [ ] Profile-aware RAG — embed kernel/NVTX data for context-rich answers
- [ ] Multi-model support (Claude, GPT, local models via Ollama)
- [ ] Cost-gated: only call LLM when user explicitly requests (no surprise API bills)
- [ ] Caching layer — don't re-analyze the same profile region twice

### AI-powered automation
- [ ] Auto-detect training iterations + flag regressions across iterations
- [ ] Anomaly detection — highlight kernels that deviate from the norm
- [ ] CI integration — `nsys-ai check profile.sqlite --baseline baseline.sqlite` → pass/fail

---

## 🏗️ What's shipped

- [x] Timeline TUI (v0.1.0)
- [x] Tree TUI (v0.1.0)
- [x] HTML viewer export (v0.1.0)
- [x] Perfetto JSON export + `perfetto` command (v0.1.5)
- [x] Web UI server (`nsys-ai web`) (v0.2.0)
- [x] AI module — auto-commentary, NVTX suggestions, bottleneck detection (v0.1.0)
- [x] PyPI package as `nsys-ai` (v0.2.1)
