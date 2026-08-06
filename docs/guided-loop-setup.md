# Guided loop setup (Direction 2)

Step-by-step setup for the **diagnose → propose → re-profile → diff → accept** workflow in the timeline web UI. This path is **not AI-based** — diagnose uses local SQL/skills, diff uses deterministic before/after comparison, and you record accept/reject yourself.

---

## What you need

| Requirement | Notes |
|-------------|--------|
| **Python 3.10+** | Matches `pyproject.toml` / CI |
| **Git** | basic cli commands |
| **Browser** | For `timeline-web` (default) |
| **Optional: `hf` CLI** | Only to download the H100 example dataset |
| **Optional: NVIDIA Nsight Systems** | Only if you capture your own profiles; not needed for the H100 replay |

You do **not** need `ANTHROPIC_API_KEY` or `pip install -e '.[agent]'` for the guided loop.

---

## 1. Get example profiles (H100 FA2 vs FA3)

The canonical replay scenario uses the Hugging Face dataset [rich7421/fastvideo-wan-h100-sp1-nsys](https://huggingface.co/datasets/rich7421/fastvideo-wan-h100-sp1-nsys):

| File | Role |
|------|------|
| `profiles/perf_h100_sp1.sqlite` | Baseline (FlashAttention-2) |
| `profiles/perf_h100_sp1_fa3.sqlite` | Candidate (FlashAttention-3) |

Install the Hugging Face Hub CLI and download only the two profile files:

```bash
pip install -U huggingface_hub
hf download rich7421/fastvideo-wan-h100-sp1-nsys --repo-type dataset \
  profiles/perf_h100_sp1.sqlite profiles/perf_h100_sp1_fa3.sqlite
```

Expected cache layout (revision hash may differ):

```text
~/.cache/huggingface/hub/datasets--rich7421--fastvideo-wan-h100-sp1-nsys/snapshots/<rev>/profiles/
  perf_h100_sp1.sqlite
  perf_h100_sp1_fa3.sqlite
```

Check that nsys-ai can see them:

```bash
python -c "from nsys_ai.loop_state import detect_h100_replay_preset; print(detect_h100_replay_preset())"
```

You should see `before_path` and `after_path` pointing at the two files.

---

## 2. Start the guided loop (web UI)

Recommended command (H100 preset + a timeline trim window for faster loading):

```bash
nsys-ai loop --h100-preset --trim 49 90 --port 8144
```

| Flag | Purpose |
|------|---------|
| `--h100-preset` | Auto-fill baseline + candidate paths from the HF cache |
| `--trim START END` | **Viewer + diagnose** time window in **seconds** (optional but speeds up large traces) |
| `--port` | HTTP port (default `8144`) |
| `--no-browser` | Print URL only; do not open a tab |
| `--gpu N` | GPU device id (default: first GPU in the profile) |

Open the UI manually if needed: `http://127.0.0.1:8144/`

Click **Loop** in the toolbar (or press **W**) to open the guided sidebar.

### Trim vs diff (important)

- **`--trim`** narrows what you **see** in the timeline and what **diagnose** analyzes.
- **Diff always compares the full profiles** (not the trim window). A narrow trim can make the timeline look fine while diff still reports the true end-to-end step change (~31% improvement for FA2→FA3 on the H100 pair).

---

## 3. Walk through the five steps

Work top-to-bottom in the loop sidebar. The primary button advances the suggested step.

| Step | Action | What happens |
|------|--------|----------------|
| **1. Diagnose** | Run diagnose on baseline | Local evidence builder runs; findings appear in the findings sidebar |
| **2. Propose** | Enter change + expected impact, save | Text is stored in loop state (no code is modified) |
| **3. Re-profile** | Set candidate `.sqlite` path | Registers the after profile (preset fills `perf_h100_sp1_fa3.sqlite` when using `--h100-preset`) |
| **4. Diff** | Run diff | Compares baseline vs candidate; **All steps & status** panel appears with verdict and stats |
| **5. Decide** | Accept or reject | Records your decision in session state only |

Example proposal (H100 replay):

- **Change:** Switch attention backend from FlashAttention-2 to FlashAttention-3 on H100.
- **Expected impact:** Lower end-to-end step time by ~30–40%, mainly from faster attention kernels.

After a successful diff on the H100 pair you should see roughly:

- **Verdict:** Likely faster
- **Step Δ:** about −3.3 min (−31%)
- **Comparability:** ~99% (High)
