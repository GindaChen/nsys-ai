# Skill audits

Reproducible audits of nsys-ai built-in skills against public Nsight profiles.

## L40S FastVideo audit (Direction 1)

**Profile:** `perf_compile.sqlite` from [rich7421/fastvideo-wan-l40s-nsys](https://huggingface.co/datasets/rich7421/fastvideo-wan-l40s-nsys)  
**Workload:** FastVideo `Wan2.1-T2V-1.3B`, 4× L40S (PCIe Gen4, no NVLink) — system-level, comm-bound, same-stream NCCL/compute  
**Reports:** [l40s-fastvideo-skill-audit.md](l40s-fastvideo-skill-audit.md), [l40s-fastvideo-gaps.md](l40s-fastvideo-gaps.md)

### Dataset files (not in git)

| File | Description |
|------|-------------|
| `profiles/perf_compile.sqlite` | **Primary audit profile** (`torch.compile` on) |
| `profiles/perf.sqlite` | Non-compile baseline |
| `analysis_measurements.md` | Maintainer reference measurements |
| `plots/` | Static plots (stream 7, NCCL bimodal, etc.) |

### Download profile

```bash
pip install huggingface_hub
hf download rich7421/fastvideo-wan-l40s-nsys --repo-type dataset \
  --local-dir ~/.cache/nsys-ai-datasets/fastvideo-wan-l40s-nsys

export L40S_PROFILE=~/.cache/nsys-ai-datasets/fastvideo-wan-l40s-nsys/profiles/perf_compile.sqlite
```

### Run audit batch

```bash
cd /path/to/nsys-ai
pip install -e '.[dev]'

python scripts/batch_audit_skills.py "$L40S_PROFILE" audit/l40s-perf_compile
```

The audit table has two layers: **Batch** (mechanical run from `batch_audit_skills.py`) and **Grade** (human score vs ground truth in `analysis_measurements.md`). Batch JSON under `audit/` is gitignored; re-run the script to refresh `_batch_summary.json`.

Outputs go under `audit/l40s-perf_compile/` (gitignored: `json/`, `logs/`, `text/`). Only `docs/audits/` and `scripts/` belong in the PR.

**Do not commit:** `nsys-ai profiling.pdf`, profile `.sqlite`, or `audit/**` output files.

### Headline numbers (`perf_compile.sqlite`, device 0)

| Metric | Value |
|--------|-------|
| Overlap | ~0% |
| Same stream | 100% compute + NCCL on **stream 7** |
| NCCL payload | ~8,936 GiB |
| NCCL msg p50 / p99 | 13.18 / 39.55 MiB |
| Idle | ~13.9% |

### Orientation

```bash
nsys-ai info "$L40S_PROFILE"
nsys-ai skill run overlap_breakdown "$L40S_PROFILE"
nsys-ai skill run nccl_payload_breakdown "$L40S_PROFILE" --format json
nsys-ai skill run profile_health_manifest "$L40S_PROFILE" --format json
nsys-ai timeline-web "$L40S_PROFILE"
nsys-ai evidence build "$L40S_PROFILE" --format json -o /tmp/findings.json
```

Ground truth detail: dataset `analysis_measurements.md` and mentoring deck `nsys-ai profiling.pdf` (do not commit the PDF).