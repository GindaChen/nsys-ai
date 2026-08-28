# 🔬 nsys-ai Examples

Hands-on examples for getting started with nsys-ai. Each example includes a download script and a step-by-step quick start.

> **Low numbers = simpler.** Numbers 01–09 are reserved for introductory examples.

## Examples

| # | Example | Profile | GPUs | Difficulty |
|---|---------|---------|------|------------|
| 10 | [FastVideo Inference](example-10-fastvideo-inference/) | Inference (video generation) | 4× H100 | ⭐⭐ Intermediate |
| 20 | [Megatron-LM DistCA](example-20-megatron-distca/) | Training (Transformer Engine) | 8× H200 | ⭐⭐⭐ Advanced |

## Quick Start (Example 20)

Example 20 is the one whose capture is published, so it runs end to end today.
It downloads a `.nsys-rep`, so Nsight Systems must be on your `PATH` — nsys-ai
converts the capture by shelling out to `nsys export`:

```bash
pip install nsys-ai
cd example-20-megatron-distca
python download_data.py
nsys-ai open output/megatron_distca.nsys-rep
```

Example 10 has no profile on HuggingFace yet; its download step fails until one
is uploaded, or until you capture your own with the Modal script it documents.

## Adding New Examples

Follow the convention:

```
example-NN-short-description/
├── .gitignore         # Ignores output/ directory
├── README.md          # Step-by-step guide
├── download_data.py   # Data download script
└── output/            # Downloaded data (gitignored)
```

Reserve number ranges:
- `01–09` — Introductory / synthetic profiles
- `10–19` — Single-model inference profiling
- `20–29` — Distributed training profiling
