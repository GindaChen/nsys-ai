# 🗺️ nsys-ai Roadmap

Three pillars: **UI** (effortless viewing), **AI** (effortless understanding), and **Agent & Skills** (first-principles diagnosis).

> Informed by competitive analysis of [NAV](https://github.com/eshama1/NSYS-Analyzer-and-Visualizer), [nsys_recipes](https://github.com/hyxcl/nsys_recipes), [nsys_easy](https://github.com/harrism/nsys_easy), and [Profiling-AI-Software-Bootcamp](https://github.com/openhackathons-org/Profiling-AI-Software-Bootcamp).

---

## Priority Order

### 🔴 P0 — Critical (next sprint)

| # | Item | Pillar | Status |
|---|------|--------|--------|
| [#1](../../issues/1) | **`nsys-ai analyze`** — full auto-report from a profile | AI | 🟡 In progress |
| [#2](../../issues/2) | **One-click Perfetto** — server → local transport, zero friction | UI | 🟡 In progress |

### 🟠 P1 — High (near term)

| # | Item | Pillar | Inspiration |
|---|------|--------|-------------|
| [#4](../../issues/4) | **`nsys-ai diff`** — multi-trace comparison with AI narration | AI | 🆕 NAV's comparative analysis |
| [#3](../../issues/3) | `nsys-ai ask` — natural language queries on profiles | AI | |
| [#15](../../issues/15) | Agent skill expansion — more built-in analysis skills | Agent | |
| [#5](../../issues/5) | TUI inline AI — press `?` to explain any kernel | AI+UI | |
| [#6](../../issues/6) | Web UI chat widget — ask questions in the browser | AI+UI | |
| [#23](../../issues/23) | **Kernel overlap matrix** — comm×comm, comm×compute, compute×compute | Agent | 🆕 nsys_recipes |
| [#24](../../issues/24) | **Per-stream NCCL breakdown** — parallelism-aware comm stats | Agent | 🆕 nsys_recipes |

### 🟡 P2 — Medium

| # | Item | Pillar | Inspiration |
|---|------|--------|-------------|
| [#7](../../issues/7) | Custom web flame chart with NVTX-aware hierarchy | UI | |
| [#8](../../issues/8) | Multi-model AI backend + caching layer | AI | |
| [#9](../../issues/9) | TUI polish — multi-GPU stacked view, diff mode | UI | |
| [#10](../../issues/10) | `nsys-ai suggest` — NVTX annotation suggestions | AI | |
| [#25](../../issues/25) | **`nsys-ai profile`** — wrapper for easy profiling | UI | 🆕 nsys_easy |
| [#26](../../issues/26) | **Statistical visualizations** — histograms, violin plots | UI | 🆕 NAV |
| [#27](../../issues/27) | **Batch processing** — parallel multi-profile extraction | Agent | 🆕 NAV |

### 🟣 P3 — Nice to have (longer term)

| # | Item | Pillar | Inspiration |
|---|------|--------|-------------|
| [#13](../../issues/13) | CI integration — `nsys-ai check` for perf regression gating | AI | 🆕 NAV regression testing |
| [#14](../../issues/14) | Anomaly detection across training iterations | AI | |
| [#11](../../issues/11) | VS Code extension — open `.sqlite` → launch viewer | UI | |
| [#12](../../issues/12) | Jupyter widget for inline profile viewing | UI | |
| [#28](../../issues/28) | **Guided optimization mode** — interactive profile→fix→reprofile loop | AI+UI | 🆕 Bootcamp |
| [#29](../../issues/29) | **Tensor Core utilization skill** — FP16/BF16 TC usage analysis | Agent | 🆕 Bootcamp |

---

## 🖥️ Pillar 1 — UI

> Zero-friction viewing of Nsight profiles across every surface — terminal, browser, VS Code.

**One-Click Perfetto (Server → Local)** — VSCode transport: remote SSH profile → local Perfetto in one click. Auto-detect `.sqlite` / `.nsys-rep`, convert + stream. Single command: `nsys-ai open profile.sqlite`.

**TUI** — Timeline polish (bookmarks, annotation overlay, multi-GPU stacked view). Tree improvements (sparklines, diff mode). Unified launcher that auto-selects timeline vs tree.

**Web UI** — Self-hosted viewer richer than Perfetto. NVTX-aware flame chart, side-by-side comparison, shareable links. **Statistical visualizations** (histograms, violin plots for kernel duration distributions — inspired by NAV).

**Profiling Convenience** — `nsys-ai profile ./train.py` wrapper with ML-optimized defaults (CUDA+NVTX+NCCL traces, inspired by nsys_easy). Completes the collect → analyze → report cycle.

**Packaging** — VS Code extension stub, Jupyter widget, zero-config pip install.

---

## 🤖 Pillar 2 — AI

> AI that understands GPU profiles as a first-class concept — integrated everywhere, not bolted on.

**AI in every interface** — TUI: inline commentary panel. Web: chat widget. CLI: `nsys-ai ask "why is iteration 142 slow?"`.

**AI CLI** — `analyze` (auto-report), `diff` (narrated multi-trace comparison), `suggest` (NVTX annotations), `explain` (kernel deep-dive).

**Multi-Trace Diff** — Load 2+ profiles, label them, compare kernel distributions. Output regressions/improvements with statistical significance. (Inspired by NAV's comparative analysis and CI regression workflows.)

**Backend** — Profile-aware RAG, multi-model support (Claude/GPT/Ollama), cost-gated, caching.

**Automation** — Iteration regression detection, anomaly flagging, CI pass/fail gating. **Guided optimization mode**: interactive walkthrough that chains profile → diagnose → suggest fix → re-profile (inspired by OpenHackathons bootcamp pedagogy).

---

## 🧠 Pillar 3 — Agent & Skills

> An intelligent agent that uses standardized SQL skills to diagnose GPU performance problems from first principles.

**Skills Foundation** — 8 built-in SQL skills: `top_kernels`, `memory_transfers`, `nvtx_kernel_map`, `gpu_idle_gaps`, `nccl_breakdown`, `kernel_launch_overhead`, `thread_utilization`, `schema_inspect`. User-extensible skill registry.

**New Skills (from competitive research):**

| Skill | Source | What it adds |
|-------|--------|-------------|
| `kernel_overlap_matrix` | nsys_recipes | comm×comm, comm×compute, compute×compute overlap matrices for multi-parallelism debugging |
| `per_stream_nccl` | nsys_recipes | Per-stream comm breakdown to distinguish TP vs PP vs DP communication |
| `tensor_core_usage` | Bootcamp | Check if kernels leverage FP16/BF16 Tensor Cores |

**Agent Persona** — CUDA ML systems expert with deep knowledge of nsys, Megatron, SGLang, vLLM. Follows evidence-based analysis: orient → identify → hypothesize → investigate → diagnose → recommend → verify.

**Book of Root Causes** — Living document of GPU performance problems. Quick-reference table, 10 detailed root cause writeups, and 38 veteran diagnostic questions.

**Batch Processing** — Thread-pool for processing multiple profiles simultaneously. Critical for CI regression testing workflows. (Inspired by NAV's parallel extraction achieving 23× speedup on large traces.)

**Benchmarking Problems** (planned):
1. Identifying pipeline "bubbles" and stalls
2. Calculating Model Flops Utilization (MFU)
3. Determining if kernels achieve ideal arithmetic intensity for a given GPU
4. Analyzing network overlap and bandwidth vs. compute-communication balance
5. Investigating module loading or kernel compilation elongating forward/backward passes
6. Assessing how memory/GC affects performance

---

## 🏆 Competitive Positioning

```
┌─────────────────────────────────────────────────────┐
│  Our unique strengths (no other tool has these):    │
│  ✅ AI/LLM-powered analysis & natural language      │
│  ✅ Interactive terminal TUI (tree + timeline)       │
│  ✅ NVTX hierarchy navigation & tree browser        │
│  ✅ Minimal-dependency core (rich + textual only)    │
│  ✅ Extensible skill system + Book of Root Causes    │
│  ✅ Agentic NVTX annotation (kernel→source mapping)  │
├─────────────────────────────────────────────────────┤
│  Gaps we're closing (learned from competitors):     │
│  🔧 Multi-trace diff & regression testing  ← NAV    │
│  🔧 Kernel overlap matrices               ← recipes │
│  🔧 Statistical visualizations            ← NAV    │
│  🔧 Profile collection wrapper            ← easy   │
│  🔧 Per-stream comm breakdown             ← recipes │
│  🔧 Guided optimization workflows         ← bootcamp│
└─────────────────────────────────────────────────────┘
```

---

## ✅ Shipped

- [x] Timeline TUI (v0.1.0)
- [x] Tree TUI (v0.1.0)
- [x] HTML viewer export (v0.1.0)
- [x] Perfetto JSON export + `perfetto` command (v0.1.5)
- [x] Web UI server — `nsys-ai web` (v0.2.0)
- [x] AI module — auto-commentary, NVTX suggestions, bottleneck detection (v0.1.0)
- [x] PyPI package as `nsys-ai` (v0.2.1)
- [x] Agent skill system — 8 built-in SQL skills + registry + CLI (v0.3.0)
- [x] Agent persona + analysis loop — `nsys-ai agent analyze|ask` (v0.3.0)
- [x] Book of Root Causes — quick-ref, long-form, veteran questions (v0.3.0)
- [x] Modular packaging — `[agent]`, `[all]` extras (v0.3.0)
