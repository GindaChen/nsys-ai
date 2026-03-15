# Evidence Schema — Finding JSON Reference

**Purpose**: When an external AI agent reaches a conclusion about a profile, it should produce a `findings.json` file that highlights the supporting timeline ranges for human verification.

---

## Quick Example

```bash
# Agent writes findings based on its analysis
cat > /tmp/findings.json << 'EOF'
{
  "title": "Pipeline Parallelism Bubble Analysis",
  "profile_path": "fastvideo.sqlite",
  "findings": [
    {
      "type": "region",
      "label": "PP Bubble — 21s GPU idle",
      "start_ns": 89000000000,
      "end_ns": 110000000000,
      "gpu_id": 0,
      "severity": "critical",
      "note": "Largest idle gap between micro-batches"
    },
    {
      "type": "highlight",
      "label": "Dominant NCCL: SendRecv (98%)",
      "start_ns": 117142000000,
      "end_ns": 117154000000,
      "severity": "warning",
      "note": "SendRecv=98% confirms Pipeline Parallelism"
    }
  ]
}
EOF

# Open timeline with findings overlay
nsys-ai timeline-web fastvideo.sqlite --findings /tmp/findings.json
```

---

## Finding Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | ✅ | `"region"` (shaded area), `"highlight"` (specific kernel), or `"marker"` (point in time) |
| `label` | string | ✅ | Short label shown in the evidence sidebar (keep ≤ 40 chars) |
| `start_ns` | integer | ✅ | Start timestamp in **nanoseconds** (absolute, from profile epoch) |
| `end_ns` | integer | for region/highlight | End timestamp in nanoseconds. Omit for `"marker"` type |
| `gpu_id` | integer | optional | GPU device ID (0-indexed). Omit if finding spans all GPUs |
| `stream` | string | optional | CUDA stream ID for highlighting a specific stream |
| `severity` | string | optional | `"critical"` (red), `"warning"` (orange), `"info"` (blue). Default: `"info"` |
| `note` | string | optional | Longer explanation shown on hover / in sidebar detail |
| `color` | string | optional | Override color, e.g. `"rgba(255,68,68,0.3)"`. Default by severity |

## EvidenceReport Wrapper

```json
{
  "title": "Human-readable title for the report",
  "profile_path": "path/to/profile.sqlite",
  "findings": [ ... ]
}
```

- `title`: Displayed at the top of the evidence sidebar
- `profile_path`: Optional, for reference only
- `findings`: Array of Finding objects (see above)

---

## How to Get Nanosecond Timestamps

Run skills with `--format json` to get timing data with nanosecond precision:

```bash
# GPU idle gaps → use gap_start / gap_end as start_ns / end_ns
nsys-ai skill run gpu_idle_gaps profile.sqlite --format json
# Returns: [{"gap_start": 89000000000, "gap_end": 110000000000, "gap_ns": 21065000000, ...}]

# Top kernels → use start / end
nsys-ai skill run top_kernels profile.sqlite --format json
# Returns: [{"name": "...", "start": 117142000000, "end": 117154000000, "dur_ns": 12390000, ...}]

# NCCL breakdown → use start / end for individual instances
nsys-ai skill run nccl_breakdown profile.sqlite --format json
# Returns: [{"name": "ncclDevKernel_SendRecv", "total_ns": 6496000000, ...}]
```

> **Tip**: If a skill returns aggregated data (e.g. `nccl_breakdown` returns totals per collective type), use `query_profile_db` or `skill run top_kernels` to find specific kernel instances with exact timestamps.

---

## When to Use Each Finding Type

| Type | Use for | Example |
|------|---------|---------|
| `region` | A time range where something happened | "GPU idle for 21s" |
| `highlight` | A specific kernel instance | "ncclDevKernel_SendRecv took 12ms" |
| `marker` | A point in time (no duration) | "Training step boundary" |

---

## Severity Guidelines

| Severity | When to use |
|----------|-------------|
| `critical` | Findings that are definite bottlenecks (>10% of profile time) |
| `warning` | Noteworthy issues that may not be the primary bottleneck |
| `info` | Context or reference points (e.g. hotspot kernels for comparison) |

---

## Viewing Findings

```bash
# From a pre-existing findings.json
nsys-ai timeline-web profile.sqlite --findings /tmp/findings.json

# Auto-generate findings (uses built-in heuristics, not agent conclusions)
nsys-ai timeline-web profile.sqlite --auto-analyze
```

In the viewer:
- **Evidence sidebar** (right panel) lists all findings as numbered cards
- **Click a finding** → timeline zooms to that time range
- **Colored overlays** appear on the timeline at each finding's location
