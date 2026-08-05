# Command: `nsys-ai evidence build`

Build evidence findings from heuristic analyzers for timeline overlay visualization.

> **When to use**: After analysis is complete, generate `findings.json` with exact
> nanosecond timestamps for visual verification on the timeline viewer.
> This is distinct from `skill run`, which returns aggregate analysis data.

---

## Usage

```bash
nsys-ai evidence build <profile.sqlite> [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--format {text,json}` | Output format (default: `json`) |
| `--analyzers a,b,c` | Comma-separated list of analyzers to run (default: all) |
| `--trim START_S END_S` | Restrict analysis to a time window (seconds) |
| `--gpu N` | GPU device ID (default: 0) |
| `-o, --output FILE` | Write findings JSON to file (also prints to stdout) |

---

## Available Analyzers

| Analyzer Name | What It Detects |
|---------------|----------------|
| `slow_iterations` | Training iterations with duration >1.5× median |
| `idle_gaps` | Top 5 GPU idle gaps with CPU attribution |
| `nccl_stalls` | Top 3 longest NCCL kernel instances |
| `kernel_hotspots` | Top 3 longest compute kernels |
| `overlap_ratio` | Low compute/NCCL overlap + same-stream diagnosis |
| `memory_anomalies` | Large memory transfers (>10MB) |
| `h2d_spikes` | H2D burst windows (>3× median) |

---

## Examples

```bash
# Run all 7 analyzers → JSON findings
nsys-ai evidence build profile.sqlite --format json

# Run only idle gaps and NCCL analyzers
nsys-ai evidence build profile.sqlite --analyzers idle_gaps,nccl_stalls --format json

# Save to file for timeline overlay
nsys-ai evidence build profile.sqlite -o /tmp/findings.json
nsys-ai timeline-web profile.sqlite --findings /tmp/findings.json

# With time window
nsys-ai evidence build profile.sqlite --trim 5.0 15.0 --format json
```

---

## Output Format

The JSON output is an EvidenceReport object following the [Evidence Schema](evidence_schema.md):

```json
{
  "title": "Idle gaps and NCCL stalls",
  "profile_path": "profile.sqlite",
  "findings": [
    {
      "type": "region",
      "label": "GPU Idle Gap (12.34ms)",
      "start_ns": 89000000000,
      "end_ns": 89012340000,
      "gpu_id": 0,
      "stream": "7",
      "severity": "warning",
      "note": "Stream 7: 12.34ms idle — CPU: cudaLaunchKernel (11.2ms)"
    }
  ],
  "skipped": []
}
```

`skipped` lists the analyzers that could not run on this profile — a missing
memcpy table, a profile with no NVTX annotation — with the reason each one
gave. It is emitted even when empty, so `[]` means "everything ran" rather
than "this build does not report it". `--format text` prints the same list as
a `Skipped` section after the findings; `--format json` prints it to stderr so
stdout stays a single JSON document. See the
[Evidence Schema](evidence_schema.md#evidencereport-wrapper) for the field.

### Finding Types

| Type | Visual | Use Case |
|------|--------|----------|
| `region` | Shaded time range | Slow iterations, idle gaps, overlap issues |
| `highlight` | Emphasized kernel bar | Hotspot kernels, long NCCL calls |
| `marker` | Point marker | Events, boundaries |

### Severity Levels

| Severity | Color | When |
|----------|-------|------|
| `critical` | 🔴 Red | Communication dominated, NCCL >5ms |
| `warning` | 🟡 Yellow | Slow iterations, idle gaps, low overlap |
| `info` | 🔵 Blue | Hotspots, memory transfers, summaries |

---

## Difference from `skill run`

| | `skill run` | `evidence build` |
|---|-------------|------------------|
| **Output** | Aggregate statistics | Instance-level ns timestamps |
| **Purpose** | Data analysis | Timeline visualization |
| **Use case** | Agent triage & drill-down | Human verification |
| **Example field** | `total_ms: 57639` | `start_ns: 89000000000` |
