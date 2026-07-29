# Evidence Schema — Finding JSON Reference

**Purpose**: When an AI agent reaches a conclusion about a profile, it should produce a `findings.json` file that highlights the supporting timeline ranges for human verification.

**Schema version**: `0.1`. Older payloads (pre-`schema_version`, only the legacy display fields on `Finding`) still load unchanged — every new field is optional.

---

## Quick Example (minimal — legacy fields only)

This minimal form is what existed before v0.1 and still loads identically today — no envelope, no v0.1 optional fields.

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

> The v0.1 envelope keys (`schema_version` / `producer` / `producer_version`) and Finding optional fields (`category`, `confidence`, `evidence`, `selection`, etc.) are documented in the sections below. Use them when you want category / confidence, structured evidence, or selection-based navigation — see [Enriched (v0.1) Example](#enriched-v01-example).

---

## Finding Fields

Required fields and pre-v0.1 display fields (rendered by every consumer):

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
| `color` | string | optional | Reserved for a future per-finding color override. The timeline-web viewer today derives overlay color exclusively from `severity` and ignores this field; setting it has no visible effect yet. |

### v0.1 optional fields

All new fields default to `null` (or are omitted entirely) and are dropped from `to_dict()` output when unset, so legacy producers keep emitting the compact pre-v0.1 shape:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Stable identifier for this finding (e.g. `"idle_gap_gpu0_stream7_500"`). Used by tooling for cross-references and provenance. |
| `category` | string | Step-time bucket: `"compute"`, `"communication"`, `"launch_overhead"`, `"idle"`. Orthogonal tags also accepted: `"memory"`, `"sync"`, `"nvtx"`, `"profile_quality"`, `"kernel_internal"`, `"framework"`. |
| `confidence` | float | Producer's confidence in [0.0, 1.0]. Low values flag noisy / borderline findings; high values flag clear bottlenecks. |
| `evidence` | list[EvidenceRow] | Structured rows backing the finding — metric values, units, provenance. See [EvidenceRow](#evidencerow). |
| `selection` | TraceSelection | The region of the profile the finding refers to (time + GPU + stream + NVTX path). See [TraceSelection](#traceselection). |
| `explanation` | string | Longer-form prose for a human reader. Free of profile-specific numbers; the `evidence` rows carry the numbers. |
| `suggested_actions` | list[string] | Concrete next steps the user could take (e.g. `"Check for explicit cudaDeviceSynchronize calls"`). |
| `false_positive_notes` | list[string] | Known caveats / when this finding may not be a real bottleneck. |
| `provenance` | dict | Producer metadata (skill name, row kind, etc.). Free-form. |
| `diff_lineage` | DiffLineage | When a finding came from a diff comparison, identifies which diff and the finding's role. See [DiffLineage](#difflineage). |

## Good vs Bad Findings

Findings should encode **conclusions**, not raw data. The agent must reason about the data before writing findings.

```diff
- BAD: label is just raw data, no reasoning
  {
    "label": "GPU Idle Gap (21065ms)",
    "note": "Stream 7: 21065ms idle"
  }

+ GOOD: label states the conclusion, note explains why
  {
    "label": "PP Bubble: 21s idle caused by serialized NCCL",
    "note": "GPU idle for 21s after AllReduce — NCCL not overlapping with compute. This is the largest pipeline bubble."
  }
```

> **Rule**: If your `label` could be generated by a SQL query alone, it's not a conclusion.
> A conclusion requires the agent to reason about *why* something happened.

## EvidenceReport Wrapper

```json
{
  "schema_version": "0.1",
  "producer": "nsys-ai",
  "producer_version": "0.2.2",
  "title": "Human-readable title for the report",
  "profile_id": "nsys1:sha256:7d2f6013fb98855bda6a5448a91af563553cc2c013f42ca6d9cdbb3b737ac738",
  "profile_path": "path/to/profile.sqlite",
  "findings": [ ... ]
}
```

- `schema_version`: Current evidence-artifact schema version (`"0.1"`). Bumped on breaking changes; additive new optional fields do not bump it.
- `producer`: Always `"nsys-ai"` for reports emitted by this tool.
- `producer_version`: Version of the `nsys-ai` package that wrote the report.
- `title`: Displayed at the top of the evidence sidebar.
- `profile_id`: Content-derived stable identifier produced by `fingerprint.get_profile_id`. Format: `nsys1:sha256:<64-hex>` (or `nsys1:path:<64-hex>` fallback when the source carries no Nsight metadata, e.g. a parquetdir-only backend). Two surfaces looking at the same profile agree on this value without depending on the filesystem path. **Every `TraceSelection.profile_id` produced during the same build carries the same value** — readers can join across surfaces by equality.
- `profile_path`: Filesystem path the producer opened. Reference only — not an identifier.
- `findings`: Array of Finding objects (see above).

Legacy payloads without the envelope keys (`schema_version` / `producer` / `producer_version` / `profile_id`) load identically — readers ignore unknown / missing envelope fields, and `profile_id` defaults to `""` for pre-`profile_id` payloads.

---

## Supporting Types

### EvidenceRow

One row of measured evidence backing a Finding. A skill emits zero or more rows; an evidence-citing Finding embeds them directly under `Finding.evidence`. Each row may additionally carry a `selection_id` that cross-references a `TraceSelection` (see below). There is no Finding→EvidenceRow id reference field — Findings always embed their evidence inline.

```json
{
  "id": "ev_idle_gap_gpu0_stream7_500",
  "source_skill": "gpu_idle_gaps",
  "values": {
    "gap_ms": 12.5,
    "gap_ns": 12500000,
    "top_cpu_api": "cudaLaunchKernel",
    "top_cpu_api_ms": 11.2
  },
  "units": {
    "gap_ms": "ms",
    "gap_ns": "ns",
    "top_cpu_api_ms": "ms"
  },
  "selection_id": "sel_idle_gap_gpu0_stream7_500",
  "provenance": {
    "row_kind": "per_gap",
    "device": 0,
    "stream": 7
  }
}
```

- `id`: Stable id for this row.
- `source_skill`: Name of the skill that produced this row.
- `values`: Raw metric values (free-form dict). Numeric values should be machine-readable (no formatting suffixes).
- `units`: Maps each value key to its unit string (e.g. `"ms"`, `"ns"`, `"count"`, `"percent"`).
- `selection_id`: Optional cross-reference to the `TraceSelection` this row applies to.
- `provenance`: Free-form producer metadata.

### TraceSelection

A region of a profile. All location fields are optional — a selection may be time-only, GPU-only, NVTX-only, or any combination.

```json
{
  "id": "sel_idle_gap_gpu0_stream7_500",
  "profile_id": "nsys1:sha256:7d2f6013fb98855bda6a5448a91af563553cc2c013f42ca6d9cdbb3b737ac738",
  "source": "skill:gpu_idle_gaps",
  "start_ns": 500,
  "end_ns": 12500500,
  "gpu_ids": [0],
  "stream_ids": [7],
  "nvtx_path": ["iteration_142", "backward"],
  "label": "12.50ms idle gap"
}
```

- `id`: Stable id for this selection.
- `profile_id`: Content-derived stable identifier — matches the enclosing `EvidenceReport.profile_id` and is sourced from `fingerprint.get_profile_id`. Format: `nsys1:sha256:<64-hex>` (or `nsys1:path:<64-hex>` fallback when no Nsight metadata is reachable). Treated by readers as an equality key — two surfaces looking at the same profile agree on the value without depending on the filesystem path.
- `source`: Who produced the selection — `"skill:<name>"`, `"gui"`, `"user"`, or `"diff"`.
- `start_ns` / `end_ns`: Time window (nanoseconds, absolute from profile epoch).
- `gpu_ids` / `rank_ids` / `stream_ids`: Location lists. Empty list (`[]`) means "no GPUs/ranks/streams" — distinct from omission, which means "unspecified".
- `nvtx_path`: NVTX hierarchy from root to leaf, e.g. `["iteration_142", "backward"]`.
- `event_ids`: Specific event ids when known.
- `label`: Short human-readable summary.

### DiffLineage

Set on findings that originated from a before/after diff comparison.

```json
{
  "diff_id": "diff_20260512",
  "role": "regression",
  "rank": 1,
  "baseline_profile_id": "nsys1:sha256:6b1f9a0e0b3e2c7a8b8d5f0e1b2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d"
}
```

- `diff_id`: Id of the diff run this finding came from.
- `role`: `"regression"`, `"improvement"`, or `"stable"`.
- `rank`: 0-indexed position in the diff's top-regressions / top-improvements list.
- `baseline_profile_id`: Identifier of the baseline side of the diff.

### Diagnostic

A higher-level summary an agent emits after analyzing a profile or a diff. Pairs a runnable verification command with the findings that justify the diagnosis.

```json
{
  "id": "diag_20260512_idle",
  "summary": "Backward pass is exposed-NCCL bound on rank 5.",
  "recommendation": "Inspect DDP bucket timing and overlap settings.",
  "verification_command": "nsys-ai diff before.sqlite after.sqlite --axis communication --axis step",
  "confidence": 0.86,
  "primary_findings": [ ... Finding objects ... ],
  "root_cause_hypotheses": [
    "DDP bucket boundary serializes the AllReduce after the last bucket's backward"
  ]
}
```

- `verification_command` is a **runnable** `nsys-ai` command, not a description. If no runnable command can be constructed, an agent should say so explicitly rather than narrate one here.

> **Transport note**: at v0.1, `Diagnostic` is a standalone schema type. It is **not** carried inside `EvidenceReport` / `findings.json` — that wrapper only knows about `findings`. A producer emitting a diagnosis should serialize it to its own JSON document (`diagnostics.json`). Top-level `diagnostic` / `diagnostics` keys added to a `findings.json` payload will be silently dropped by `EvidenceReport.from_dict()`.

#### Producing `diagnostics.json` from the CLI

`ask`, `agent ask`, and `agent analyze` can each emit their diagnosis as a structured `diagnostics.json` alongside the human-readable answer:

```bash
nsys-ai ask profile.sqlite "why is this slow?" --diagnostics            # writes ./diagnostics.json
nsys-ai agent ask profile.sqlite "why is this slow?" --diagnostics out/diag.json
nsys-ai agent analyze profile.sqlite --diagnostics
```

- `--diagnostics` without a value writes `diagnostics.json` in the current directory; omitting the flag preserves the previous behavior. It is independent of `agent analyze --evidence -o findings.json`.
- `id` is deterministic — derived from the profile id, the mode (`ask` / `analyze`), the question, and the full-profile or trim scope — so re-running the same command reproduces the same artifact id without colliding with a different analysis window.
- `primary_findings` are converted from the skill rows the run already executed (via each skill's v0.1 `to_findings_fn` converter); no extra profile queries are run to build the JSON.
- Load it back with `nsys_ai.annotation.load_diagnostic(path)` (writer: `save_diagnostic(diagnostic, path)`).

The printed evidence-first answer and the JSON are rendered from the same structured `Diagnostic`, so the sections always match:

| Human section | JSON field |
|---------------|------------|
| Summary | `summary` |
| Primary Diagnosis | `root_cause_hypotheses[0]` |
| Evidence | `primary_findings[*].evidence` |
| Confidence | `confidence` |
| Recommended Action | `recommendation` |
| Verify | `verification_command` |

---

## Enriched (v0.1) Example

Same finding as the [Quick Example](#quick-example-minimal--legacy-fields-only) above, written with the v0.1 fields populated:

```json
{
  "schema_version": "0.1",
  "producer": "nsys-ai",
  "producer_version": "0.2.2",
  "title": "Pipeline Parallelism Bubble Analysis",
  "profile_id": "nsys1:sha256:7d2f6013fb98855bda6a5448a91af563553cc2c013f42ca6d9cdbb3b737ac738",
  "profile_path": "fastvideo.sqlite",
  "findings": [
    {
      "id": "pp_bubble_iter142",
      "type": "region",
      "label": "PP Bubble — 21s GPU idle",
      "start_ns": 89000000000,
      "end_ns": 110000000000,
      "gpu_id": 0,
      "severity": "critical",
      "note": "Largest idle gap between micro-batches",
      "category": "idle",
      "confidence": 0.92,
      "evidence": [
        {
          "id": "ev_pp_bubble_iter142",
          "source_skill": "gpu_idle_gaps",
          "values": {"gap_ms": 21000.0, "pct_of_profile": 24.5},
          "units": {"gap_ms": "ms", "pct_of_profile": "percent"}
        }
      ],
      "selection": {
        "id": "sel_pp_bubble_iter142",
        "profile_id": "nsys1:sha256:7d2f6013fb98855bda6a5448a91af563553cc2c013f42ca6d9cdbb3b737ac738",
        "source": "skill:gpu_idle_gaps",
        "start_ns": 89000000000,
        "end_ns": 110000000000,
        "gpu_ids": [0]
      },
      "explanation": "GPU 0 is idle for 21s between micro-batches. This is the largest pipeline bubble in the trace; the dominant cause is a serialized NCCL collective immediately preceding the idle window.",
      "suggested_actions": [
        "Inspect the pipeline-parallel scheduler for stage-boundary syncs",
        "Verify the collective uses async/overlapping launch settings"
      ],
      "false_positive_notes": [
        "Brief idle windows (<1ms) are kernel-launch overhead, not real stalls"
      ],
      "provenance": {"skill": "gpu_idle_gaps", "row_kind": "per_gap"}
    }
  ]
}
```

---

## How to Get Nanosecond Timestamps

### Option A: `kernel_instances` skill (recommended for targeted findings)

```bash
# Get specific kernel instances with exact ns timestamps
nsys-ai skill run kernel_instances profile.sqlite --format json -p name=flash_bwd -p limit=3
# Returns: [{"kernel_name": "...", "start_ns": 89886440111, "end_ns": 89982440111, "duration_ms": 96.0, ...}]

# All top instances (no name filter)
nsys-ai skill run kernel_instances profile.sqlite --format json -p limit=5
```

### Option B: `evidence build` (automatic evidence generation)

```bash
# Run all 7 heuristic analyzers → findings JSON with ns timestamps
nsys-ai evidence build profile.sqlite --format json -o /tmp/findings.json

# Run specific analyzers only
nsys-ai evidence build profile.sqlite --analyzers idle_gaps,nccl_stalls --format json
```

See [evidence-build.md](evidence-build.md) for full analyzer list and options.

### Option C: Other skills with ns data

```bash
# GPU idle gaps → use start_ns / end_ns directly for finding start/end
nsys-ai skill run gpu_idle_gaps profile.sqlite --format json
# Returns: [{"start_ns": 89000000000, "end_ns": 110000000000, "gap_ns": 21065000000, ...}]

# NCCL breakdown → aggregate totals per collective type (no instance-level ns)
nsys-ai skill run nccl_breakdown profile.sqlite --format json
# Returns: [{"name": "ncclDevKernel_SendRecv", "total_ns": 6496000000, ...}]
# → Use kernel_instances -p name=nccl for instance-level timestamps
```

> **Tip**: `kernel_instances` is the primary way to get instance-level ns timestamps for any kernel type. Use it to bridge the gap between aggregate skill output and findings.json evidence.

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

---

## End-to-End Workflow

After analysis, an AI agent should produce findings and open the web viewer.
Three approaches, from most control to least:

### Option A: Agent-Driven (recommended)

The agent runs skills, reasons about results, writes conclusions, then opens viewer.

```bash
# 1. COLLECT — run skills to gather data
nsys-ai skill run gpu_idle_gaps profile.sqlite --format json > /tmp/gaps.json
nsys-ai skill run nccl_breakdown profile.sqlite --format json > /tmp/nccl.json
nsys-ai skill run top_kernels profile.sqlite --format json > /tmp/kernels.json

# 2. REASON — agent analyzes the collected data (LLM reasoning, not a CLI command)
#    Cross-reference gaps, NCCL, and kernel data to identify root causes.

# 3. WRITE — agent writes findings.json with conclusions + nanosecond time ranges
cat > /tmp/findings.json << 'EOF'
{
  "title": "Pipeline Parallelism Bubble Analysis",
  "findings": [
    {
      "type": "region",
      "label": "PP Bubble: 21s idle caused by serialized NCCL",
      "start_ns": 89886440111,
      "end_ns": 110951683466,
      "severity": "critical",
      "note": "GPU idle for 21s after AllReduce — NCCL not overlapping with compute"
    }
  ]
}
EOF

# 4. VIEW — open timeline with evidence overlay for human verification
nsys-ai timeline-web profile.sqlite --findings /tmp/findings.json
```

### Option B: Auto-Evidence via `agent analyze`

Built-in `EvidenceBuilder` runs 7 heuristic analyzers (slow iterations, GPU idle gaps,
NCCL stalls, kernel hotspots, overlap ratio, memory anomalies, H2D spikes) and produces
findings automatically.

```bash
# Analyze + generate evidence in one step
nsys-ai agent analyze profile.sqlite --evidence -o /tmp/findings.json

# Then open the viewer
nsys-ai timeline-web profile.sqlite --findings /tmp/findings.json
```

### Option C: One-Step Auto-Analyze

Combines analysis + viewer launch. No `findings.json` file is written to disk.

```bash
nsys-ai timeline-web profile.sqlite --auto-analyze
```

> **When to use which**: Option A produces the highest-quality findings because the
> agent reasons about root causes. Option B/C use heuristics only (no LLM reasoning).
> Option A is strongly recommended for most AI agents, falling back to Option B or C when speed matters more than custom reasoning.
