#!/usr/bin/env bash
# Smoke test for nsys-ai plugin — validates that every CLI command referenced
# in skills/analyze/SKILL.md + M*.md still works with the installed nsys-ai version.
#
# Usage:  ./scripts/smoke_test.sh <profile.sqlite> [nvtx-profile.sqlite] [before.sqlite] [after.sqlite]
#   profile.sqlite      — any GPU profile (NCCL preferred for Mode 2 coverage)
#   nvtx-profile.sqlite — profile with NVTX_EVENTS; enables Mode 5/9 skill checks
#                         (optional; NVTX skills SKIP when omitted)
#   before.sqlite       — baseline profile for Mode 8 Diff (optional; falls back to profile × profile)
#   after.sqlite        — candidate profile for Mode 8 Diff (optional; same fallback)
# Exit 0 if all checks succeed, non-zero otherwise.
#
# Stage A:  Mode 1 end-to-end (manifest → evidence → timeline surface check)
# Stage B1: Mode 2 + Mode 6 drill-down skills + field-shape validation
# Stage B2: Mode 3, Mode 4, Mode 5 drill-down skills + field-shape validation
# Stage C1: Mode 7 CUTracer plan + script generation
# Stage C2: Mode 8 Diff + Mode 9 Variance/iteration skills

set -euo pipefail

PROFILE="${1:-}"
if [[ -z "$PROFILE" || ! -f "$PROFILE" ]]; then
  echo "usage: $0 <profile.sqlite> [nvtx-profile.sqlite] [before.sqlite] [after.sqlite]" >&2
  exit 2
fi
if [[ -n "${2:-}" ]]; then
  if [[ ! -f "$2" ]]; then
    echo "error: nvtx-profile.sqlite not found: $2" >&2
    echo "usage: $0 <profile.sqlite> [nvtx-profile.sqlite] [before.sqlite] [after.sqlite]" >&2
    exit 2
  fi
  NVTX_PROFILE="$2"
else
  NVTX_PROFILE="$PROFILE"
fi

# Mode 8 Diff: use dedicated before/after pair when supplied; otherwise same-file fallback.
DIFF_BEFORE="${3:-$PROFILE}"
DIFF_AFTER="${4:-$PROFILE}"
if [[ -n "${3:-}" && ! -f "$3" ]]; then
  echo "error: before.sqlite not found: $3" >&2; exit 2
fi
if [[ -n "${4:-}" && ! -f "$4" ]]; then
  echo "error: after.sqlite not found: $4" >&2; exit 2
fi

FAIL=0

run() {
  local label="$1"; shift
  printf "  %-55s " "$label"
  if "$@" >/dev/null 2>&1; then
    echo "OK"
  else
    echo "FAIL"
    FAIL=$((FAIL+1))
  fi
}

run_capture() {
  local label="$1"; shift
  local out="$1"; shift
  printf "  %-55s " "$label"
  if "$@" >"$out" 2>/dev/null; then
    echo "OK"
  else
    echo "FAIL"
    FAIL=$((FAIL+1))
  fi
}

# Tolerant run: treats non-zero exit as OK if stdout is valid JSON (list or
# dict). Used for skills that return error-JSON on minimal fixtures (e.g.
# tests/fixtures/mock.sqlite); the follow-up check_field_conditional SKIPs gracefully.
run_capture_json() {
  local label="$1"; shift
  local out="$1"; shift
  printf "  %-55s " "$label"
  "$@" >"$out" 2>/dev/null || true
  # Pass path as argv so paths containing quotes/backslashes don't break parsing.
  if python3 -c 'import json, sys; json.load(open(sys.argv[1]))' "$out" 2>/dev/null; then
    echo "OK"
  else
    echo "FAIL (invalid JSON output)"
    FAIL=$((FAIL+1))
  fi
}

check_regex() {
  local label="$1"; local file="$2"; local pattern="$3"
  printf "  %-55s " "$label"
  if grep -qE "$pattern" "$file" 2>/dev/null; then
    echo "OK"
  else
    echo "FAIL (pattern /$pattern/ not found)"
    FAIL=$((FAIL+1))
  fi
}

check_json_key() {
  # Verify a key path exists and is non-null in a JSON file.
  # key_path uses dot notation: e.g. "nccl.collectives" or "idle.idle_pct"
  local label="$1"; local file="$2"; local key_path="$3"
  printf "  %-55s " "$label"
  local py_expr
  py_expr="import json,sys; d=json.load(open('$file')); d=d[0] if isinstance(d,list) else d"
  for part in ${key_path//./ }; do
    py_expr+="; d=d['$part']"
  done
  py_expr+="; sys.exit(0 if d is not None else 1)"
  if python3 -c "$py_expr" 2>/dev/null; then
    echo "OK"
  else
    echo "FAIL (key $key_path missing or null)"
    FAIL=$((FAIL+1))
  fi
}

check_field_conditional() {
  # Run check_regex but SKIP when JSON has no real data rows.
  # Skips rows with: empty list, _metadata, _summary, _diagnostic, or top-level 'error' key.
  local label="$1"; local file="$2"; local pattern="$3"; local skip_msg="$4"
  printf "  %-55s " "$label"
  EMPTY=$(python3 -c "
import json, sys
try:
    d = json.load(open('$file'))
    rows = d if isinstance(d, list) else [d]
    data = [r for r in rows if not r.get('_metadata') and not r.get('_summary')
            and not r.get('_diagnostic') and 'error' not in r]
    print('yes' if not data else 'no')
except Exception:
    print('yes')
" 2>/dev/null || echo "yes")
  if [[ "$EMPTY" == "yes" ]]; then
    echo "SKIP ($skip_msg)"
  elif grep -qE "$pattern" "$file" 2>/dev/null; then
    echo "OK"
  else
    echo "FAIL (pattern /$pattern/ not found)"
    FAIL=$((FAIL+1))
  fi
}

# ── Profile capability detection ──────────────────────────────────────────────
# Use the skill itself to probe NVTX — sqlite3 can't read DuckDB parquet caches.
NVTX_PROBE_OUT="$(mktemp)"
nsys-ai skill run nvtx_layer_breakdown "$NVTX_PROFILE" --format json --max-rows 2 \
  >"$NVTX_PROBE_OUT" 2>/dev/null || true
# Detection succeeds when either real region rows or a _detection_meta row exists.
# Probe two rows so a prepended metadata row does not hide the first real NVTX region row.
HAS_NVTX=$(python3 -c "
import json
d = json.load(open('$NVTX_PROBE_OUT'))
has_regions = any('nvtx_region' in r for r in d)
has_meta = any('_detection_meta' in r for r in d)
print('1' if (has_regions or has_meta) else '0')
" 2>/dev/null || echo 0)
NVTX_ROWS="$HAS_NVTX"

# Temp files
MANIFEST_OUT="$(mktemp)"
OVL_OUT="$(mktemp)"
NCCL_BD_OUT="$(mktemp)"
NCCL_AN_OUT="$(mktemp)"
NCCL_COMM_OUT="$(mktemp)"
TC_OUT="$(mktemp)"
TK_OUT="$(mktemp)"
KLP_OUT="$(mktemp)"
AI_OUT="$(mktemp)"
MEM_BW_OUT="$(mktemp)"
MEM_XFER_OUT="$(mktemp)"
H2D_OUT="$(mktemp)"
GAPS_OUT="$(mktemp)"
SYNC_OUT="$(mktemp)"
CPU_GPU_OUT="$(mktemp)"
ITER_OUT="$(mktemp)"
NLB_OUT="$(mktemp)"
NKM_OUT="$(mktemp)"
SE_OUT="$(mktemp)"
CUPLAN_OUT="$(mktemp)"
CUPLAN_SCRIPT_OUT="$(mktemp)"
_cleanup() {
  rm -f "$MANIFEST_OUT" "$OVL_OUT" "$NCCL_BD_OUT" "$NCCL_AN_OUT" "$NCCL_COMM_OUT" \
        "$TC_OUT" "$TK_OUT" "$KLP_OUT" "$AI_OUT" \
        "$MEM_BW_OUT" "$MEM_XFER_OUT" "$H2D_OUT" \
        "$GAPS_OUT" "$SYNC_OUT" "$CPU_GPU_OUT" \
        "$ITER_OUT" "$NLB_OUT" "$NKM_OUT" "$SE_OUT" "$NVTX_PROBE_OUT" \
        "$CUPLAN_OUT" "$CUPLAN_SCRIPT_OUT" \
        /tmp/findings_smoke.json
}
trap _cleanup EXIT

# ── Top-level ─────────────────────────────────────────────────────────────────
echo "== Top-level commands =="
run "nsys-ai --help"             nsys-ai --help
run "nsys-ai skill list"         nsys-ai skill list
run "schema_inspect"             nsys-ai skill run schema_inspect "$PROFILE" --format json

# ── Mode 1: manifest + field validation ───────────────────────────────────────
echo "== Mode 1 — profile_health_manifest + field validation =="
run_capture "profile_health_manifest" "$MANIFEST_OUT" \
  nsys-ai skill run profile_health_manifest "$PROFILE" --format json
check_regex    "  manifest: gpu field"              "$MANIFEST_OUT" '"gpu"'
check_regex    "  manifest: profile_span_ms"        "$MANIFEST_OUT" '"profile_span_ms"'
check_regex    "  manifest: suspected_bottleneck"   "$MANIFEST_OUT" '"suspected_bottleneck"'
check_json_key "  manifest: nccl.collectives"       "$MANIFEST_OUT" "nccl.collectives"
check_json_key "  manifest: idle.idle_pct"          "$MANIFEST_OUT" "idle.idle_pct"
# overlap.overlap_pct absent when device 0 has no kernels (overlap.error present instead)
printf "  %-55s " "  manifest: overlap.overlap_pct"
OVERLAP_ERR=$(python3 -c "
import json; d=json.load(open('$MANIFEST_OUT'))
d=d[0] if isinstance(d,list) else d
print('yes' if 'error' in d.get('overlap',{}) else 'no')
" 2>/dev/null || echo "no")
if [[ "$OVERLAP_ERR" == "yes" ]]; then
  echo "SKIP (device 0 empty — overlap.error present; auto-retry needed)"
else
  if python3 -c "
import json,sys; d=json.load(open('$MANIFEST_OUT'))
d=d[0] if isinstance(d,list) else d
assert d['overlap']['overlap_pct'] is not None
" 2>/dev/null; then
    echo "OK"
  else
    echo "FAIL (key overlap.overlap_pct missing or null)"
    FAIL=$((FAIL+1))
  fi
fi
run "root_cause_matcher"         nsys-ai skill run root_cause_matcher "$PROFILE" --format json
if [[ "$HAS_NVTX" -gt 0 && "$NVTX_ROWS" -gt 0 ]]; then
  run "nvtx_layer_breakdown (probe)" nsys-ai skill run nvtx_layer_breakdown "$NVTX_PROFILE" --format json --max-rows 1
else
  printf "  %-55s " "nvtx_layer_breakdown (probe)"; echo "SKIP (no NVTX_EVENTS)"
fi

# ── Mode 2: comms ─────────────────────────────────────────────────────────────
echo "== Mode 2 — comms (NCCL / overlap) =="
run_capture "overlap_breakdown" "$OVL_OUT" \
  nsys-ai skill run overlap_breakdown "$PROFILE" --format json
# overlap_pct absent when device 0 has no kernels (error key present instead)
OVL_ERR=$(python3 -c "
import json; d=json.load(open('$OVL_OUT'))
d=d[0] if isinstance(d,list) else d
print('yes' if 'error' in d else 'no')
" 2>/dev/null || echo "no")
printf "  %-55s " "  overlap_breakdown: overlap_pct field"
if [[ "$OVL_ERR" == "yes" ]]; then
  echo "SKIP (device error — no overlap data on device 0)"
else
  if grep -qE '"overlap_pct"' "$OVL_OUT" 2>/dev/null; then echo "OK"
  else echo "FAIL (overlap_pct missing)"; FAIL=$((FAIL+1)); fi
fi

run_capture "nccl_breakdown" "$NCCL_BD_OUT" \
  nsys-ai skill run nccl_breakdown "$PROFILE" --format json
check_field_conditional \
  "  nccl_breakdown: total_ms field" "$NCCL_BD_OUT" '"total_ms"' "no NCCL data on this profile"

run "kernel_overlap_matrix"      nsys-ai skill run kernel_overlap_matrix "$PROFILE" --format json

run_capture "nccl_anomaly" "$NCCL_AN_OUT" \
  nsys-ai skill run nccl_anomaly "$PROFILE" --format json -p threshold=3.0
check_field_conditional \
  "  nccl_anomaly: ratio_to_avg field" "$NCCL_AN_OUT" '"ratio_to_avg"' "no anomalies detected"

run_capture "nccl_communicator_analysis" "$NCCL_COMM_OUT" \
  nsys-ai skill run nccl_communicator_analysis "$PROFILE" --format json
check_field_conditional \
  "  nccl_communicator_analysis: communicator_id" "$NCCL_COMM_OUT" '"communicator_id"' \
  "no communicator blobs (re-export with --include-blobs=true)"

# ── Mode 3: compute ───────────────────────────────────────────────────────────
echo "== Mode 3 — compute (kernels / tensor core / MFU) =="
run_capture_json "tensor_core_usage" "$TC_OUT" \
  nsys-ai skill run tensor_core_usage "$PROFILE" --format json
check_field_conditional \
  "  tensor_core_usage: tc_achieved_pct" "$TC_OUT" '"tc_achieved_pct"' "no TC-capable kernels"

run_capture "top_kernels" "$TK_OUT" \
  nsys-ai skill run top_kernels "$PROFILE" --format json --max-rows 5
check_regex "  top_kernels: total_ms field"        "$TK_OUT" '"total_ms"'

# kernel_instances: default (longest) + specific-kernel sub-focus
run "kernel_instances (longest)"  nsys-ai skill run kernel_instances "$PROFILE" --format json --max-rows 3
KERNEL_NAME=$(python3 -c "
import json
d = json.load(open('$TK_OUT'))
if d: print(d[0]['kernel_name'])
" 2>/dev/null || echo "")
printf "  %-55s " "kernel_instances (by name)"
if [[ -n "$KERNEL_NAME" ]]; then
  if nsys-ai skill run kernel_instances "$PROFILE" --format json -p name="$KERNEL_NAME" --max-rows 3 >/dev/null 2>&1; then
    echo "OK"
  else
    echo "FAIL"
    FAIL=$((FAIL+1))
  fi
else
  echo "SKIP (no kernel name from top_kernels)"
fi

run_capture "kernel_launch_pattern" "$KLP_OUT" \
  nsys-ai skill run kernel_launch_pattern "$PROFILE" --format json
check_regex "  kernel_launch_pattern: sync_stalls"  "$KLP_OUT" '"sync_stalls"'

# arithmetic_intensity requires theoretical_flops — use a sentinel value (1e12 FLOPs)
run_capture "arithmetic_intensity" "$AI_OUT" \
  nsys-ai skill run arithmetic_intensity "$PROFILE" --format json \
  -p theoretical_flops=1000000000000
check_field_conditional \
  "  arithmetic_intensity: classification" "$AI_OUT" '"classification"' \
  "GPU not in hardware specs — roofline unavailable"

# ── Mode 4: memory ────────────────────────────────────────────────────────────
echo "== Mode 4 — memory (H2D / D2H / bandwidth) =="
run_capture "memory_bandwidth" "$MEM_BW_OUT" \
  nsys-ai skill run memory_bandwidth "$PROFILE" --format json
check_field_conditional \
  "  memory_bandwidth: avg_bandwidth_gbps" "$MEM_BW_OUT" '"avg_bandwidth_gbps"' \
  "no CUDA memory copies in this profile"

run_capture "memory_transfers" "$MEM_XFER_OUT" \
  nsys-ai skill run memory_transfers "$PROFILE" --format json
check_field_conditional \
  "  memory_transfers: total_ms field" "$MEM_XFER_OUT" '"total_ms"' \
  "No memory transfers found"

run_capture "h2d_distribution" "$H2D_OUT" \
  nsys-ai skill run h2d_distribution "$PROFILE" --format json
# pattern metadata row only appended when H2D data rows exist
printf "  %-55s " "  h2d_distribution: pattern type"
H2D_NONEMPTY=$(python3 -c "
import json; d=json.load(open('$H2D_OUT'))
print('yes' if d else 'no')
" 2>/dev/null || echo "no")
if [[ "$H2D_NONEMPTY" == "no" ]]; then
  echo "SKIP (no H2D transfers in this profile)"
else
  if grep -qE '"type"' "$H2D_OUT" 2>/dev/null; then
    echo "OK"
  else
    echo "FAIL (pattern /\"type\"/ not found)"
    FAIL=$((FAIL+1))
  fi
fi

# ── Mode 5: NVTX / code mapping ──────────────────────────────────────────────
echo "== Mode 5 — NVTX / code mapping =="
# Tolerant run: iteration_timing errors with "no NVTX_EVENTS table" on minimal
# fixtures; the field_conditional check SKIPs on error-JSON.
run_capture_json "iteration_timing" "$ITER_OUT" \
  nsys-ai skill run iteration_timing "$NVTX_PROFILE" --format json
check_field_conditional \
  "  iteration_timing: duration_ms field" "$ITER_OUT" '"duration_ms"' \
  "no iterations detected"

if [[ "$HAS_NVTX" -gt 0 && "$NVTX_ROWS" -gt 0 ]]; then
  run_capture "nvtx_layer_breakdown" "$NLB_OUT" \
    nsys-ai skill run nvtx_layer_breakdown "$NVTX_PROFILE" --format json --max-rows 20
  check_field_conditional \
    "  nvtx_layer_breakdown: total_gpu_ms"  "$NLB_OUT" '"total_gpu_ms"' "no NVTX regions found"
  check_field_conditional \
    "  nvtx_layer_breakdown: tc_achieved_pct" "$NLB_OUT" '"tc_achieved_pct"' "no NVTX regions found"

  run_capture "nvtx_kernel_map" "$NKM_OUT" \
    nsys-ai skill run nvtx_kernel_map "$NVTX_PROFILE" --format json --max-rows 5
  check_field_conditional \
    "  nvtx_kernel_map: kernel_name field"  "$NKM_OUT" '"kernel_name"' "no NVTX→kernel mappings"

  # iteration_detail: use iteration=1 (safe default)
  run "iteration_detail iter=1" \
    nsys-ai skill run iteration_detail "$NVTX_PROFILE" --format json -p iteration=1

  # speedup_estimator: extract median_ms from iteration_timing
  ITER_MS=$(python3 -c "
import json
rows=[r for r in json.load(open('$ITER_OUT')) if 'duration_ms' in r]
if rows:
    durs=[r['duration_ms'] for r in rows]
    print(sorted(durs)[len(durs)//2])
else:
    print(100)
" 2>/dev/null || echo 100)
  run_capture "speedup_estimator" "$SE_OUT" \
    nsys-ai skill run speedup_estimator "$NVTX_PROFILE" --format json \
    -p iteration_ms="$ITER_MS"
  check_field_conditional \
    "  speedup_estimator: speedup field"    "$SE_OUT" '"speedup"' "no speedup opportunities found"
else
  printf "  %-55s " "nvtx_layer_breakdown";   echo "SKIP (no NVTX_EVENTS — pass nvtx-profile as \$2)"
  printf "  %-55s " "nvtx_kernel_map";        echo "SKIP (no NVTX_EVENTS)"
  printf "  %-55s " "iteration_detail iter=1"; echo "SKIP (no NVTX_EVENTS)"
  printf "  %-55s " "speedup_estimator";       echo "SKIP (no NVTX_EVENTS)"
fi

# ── Mode 6: idle / sync ───────────────────────────────────────────────────────
echo "== Mode 6 — idle / sync =="
run_capture "gpu_idle_gaps" "$GAPS_OUT" \
  nsys-ai skill run gpu_idle_gaps "$PROFILE" --format json -p min_gap_ns=1000000
# attribution only present on top 5 gaps; check conditionally
printf "  %-55s " "  gpu_idle_gaps: attribution field"
ATTR_ROWS=$(python3 -c "
import json
d=json.load(open('$GAPS_OUT'))
rows=[r for r in d if not r.get('_summary') and r.get('attribution')]
print('yes' if rows else 'no')
" 2>/dev/null || echo "no")
if [[ "$ATTR_ROWS" == "yes" ]]; then
  if grep -qE '"description"' "$GAPS_OUT" 2>/dev/null; then
    echo "OK"
  else
    echo "FAIL (attribution.description missing)"
    FAIL=$((FAIL+1))
  fi
else
  echo "SKIP (no gaps large enough for CPU attribution)"
fi

run "stream_concurrency"   nsys-ai skill run stream_concurrency "$PROFILE" --format json

run_capture "sync_cost_analysis" "$SYNC_OUT" \
  nsys-ai skill run sync_cost_analysis "$PROFILE" --format json
check_regex "  sync_cost_analysis: sync_density_pct" "$SYNC_OUT" '"sync_density_pct"'

HOST_SYNC_OUT="$(mktemp)"
run_capture_json "host_sync_parent_ranges" "$HOST_SYNC_OUT" \
  nsys-ai skill run host_sync_parent_ranges "$NVTX_PROFILE" --format json
check_field_conditional \
  "  host_sync_parent_ranges: parent_range" "$HOST_SYNC_OUT" '"parent_range"' \
  "no NVTX or no matching sync events"

run "kernel_launch_overhead" nsys-ai skill run kernel_launch_overhead "$PROFILE" --format json

run_capture "cpu_gpu_pipeline" "$CPU_GPU_OUT" \
  nsys-ai skill run cpu_gpu_pipeline "$PROFILE" --format json
check_field_conditional \
  "  cpu_gpu_pipeline: starvation_events"  "$CPU_GPU_OUT" '"starvation_events"' \
  "no dispatch data"

run "thread_utilization"   nsys-ai skill run thread_utilization "$PROFILE" --format json
run "module_loading"       nsys-ai skill run module_loading "$PROFILE" --format json
run "gc_impact"            nsys-ai skill run gc_impact "$PROFILE" --format json
run "pipeline_bubble_metrics" nsys-ai skill run pipeline_bubble_metrics "$PROFILE" --format json

# ── Stage A evidence / timeline surface ───────────────────────────────────────
echo "== Stage A — evidence build + timeline-web surface =="
run "evidence build"          nsys-ai evidence build "$PROFILE" --format json -o /tmp/findings_smoke.json
run "findings JSON valid"     bash -c "python3 -c 'import json; json.load(open(\"/tmp/findings_smoke.json\"))'"
run "findings has findings key" bash -c "python3 -c 'import json; assert \"findings\" in json.load(open(\"/tmp/findings_smoke.json\"))'"
run "timeline-web --help"     nsys-ai timeline-web --help

# ── Mode 7: CUTracer ─────────────────────────────────────────────────────────
echo "== Mode 7 — CUTracer (SASS) =="
# cutracer check: exit 0 = full SASS mode; exit 1 = .so missing (kernel-launch-logger fallback)
# or Python package missing (hard block). Both cases are valid smoke outcomes.
printf "  %-55s " "cutracer check"
CUTRACER_CHECK_OUT="$(mktemp)"
if nsys-ai cutracer check >"$CUTRACER_CHECK_OUT" 2>&1; then
  echo "OK (full SASS mode)"
elif grep -q "Python package.*OK" "$CUTRACER_CHECK_OUT" 2>/dev/null; then
  echo "SKIP (.so not built — kernel-launch-logger fallback available)"
else
  echo "SKIP (cutracer Python package not installed)"
fi
rm -f "$CUTRACER_CHECK_OUT"
# cutracer plan --top-n 3: default output is a human-readable table/summary.
# Either the "Kernel" selection header OR the "Skipped" fallback is accepted —
# minimal fixtures have only trivial kernels and hit the Skipped path.
run_capture "cutracer plan --top-n 3" "$CUPLAN_OUT" \
  nsys-ai cutracer plan "$PROFILE" --top-n 3
check_regex "  cutracer plan: Kernel column or Skipped note" "$CUPLAN_OUT" 'Kernel|Skipped'
# cutracer plan --script: should emit a bash script containing cutracer reference
run_capture "cutracer plan --script" "$CUPLAN_SCRIPT_OUT" \
  nsys-ai cutracer plan "$PROFILE" --script --launch-cmd 'echo test'
check_regex "  cutracer plan script: cutracer ref" "$CUPLAN_SCRIPT_OUT" 'cutracer'

# ── Mode 8: Diff ──────────────────────────────────────────────────────────────
echo "== Mode 8 — Diff =="
DIFF_OUT="$(mktemp)"

# 8a. Global diff — check all M8_DIFF.md top-level JSON keys
run_capture "diff global --format json" "$DIFF_OUT" \
  nsys-ai diff "$DIFF_BEFORE" "$DIFF_AFTER" --format json
check_json_key "  diff: top_regressions key"   "$DIFF_OUT" 'top_regressions'
check_json_key "  diff: top_improvements key"  "$DIFF_OUT" 'top_improvements'
check_json_key "  diff: nvtx_regressions key"  "$DIFF_OUT" 'nvtx_regressions'
check_json_key "  diff: nvtx_improvements key" "$DIFF_OUT" 'nvtx_improvements'
check_json_key "  diff: overlap key"           "$DIFF_OUT" 'overlap'
check_json_key "  diff: warnings key"          "$DIFF_OUT" 'warnings'

# 8b. Same-file path: regressions and improvements must both be empty (comparing a
#     profile to itself should yield zero kernel changes — §C2 acceptance test 2 CLI side).
#     The same-inode warning is surfaced by the AI skill layer, not the CLI.
DIFF_SAME_OUT="$(mktemp)"
run_capture "diff same-file --format json" "$DIFF_SAME_OUT" \
  nsys-ai diff "$PROFILE" "$PROFILE" --format json
printf "  %-55s " "diff same-file: zero regressions + improvements"
SAME_CHANGES=$(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(len(d.get("top_regressions", [])) + len(d.get("top_improvements", [])))
' "$DIFF_SAME_OUT" 2>/dev/null || echo 1)
if [[ "$SAME_CHANGES" -eq 0 ]]; then echo "OK"; else echo "FAIL (expected 0 changes, got $SAME_CHANGES)"; FAIL=$((FAIL+1)); fi
rm -f "$DIFF_SAME_OUT"

# 8c. diff --gpu 0 (GPU-scoped diff — M8_DIFF.md §3 command 2c)
DIFF_GPU_OUT="$(mktemp)"
run_capture "diff --gpu 0 --format json" "$DIFF_GPU_OUT" \
  nsys-ai diff "$DIFF_BEFORE" "$DIFF_AFTER" --format json --gpu 0
check_json_key "  diff --gpu 0: top_regressions key" "$DIFF_GPU_OUT" 'top_regressions'
rm -f "$DIFF_GPU_OUT"

# 8d. diff --iteration 1 (iteration-scoped — M8_DIFF.md §3 command 2b)
#     Only run when both profiles have ≥ 2 iterations; otherwise the command
#     exits without JSON and check_json_key would false-fail.
DIFF_ITER_OUT="$(mktemp)"
# Count iterations robustly: handles error dicts, empty lists, and pipefail.
_count_iters() {
  local prof="$1"
  local tmp
  tmp=$(mktemp)
  # Route JSON through a tempfile instead of argv — argv has an OS-imposed
  # ARG_MAX (~2 MB on Linux) and large profiles can produce bigger payloads.
  nsys-ai skill run iteration_timing "$prof" --format json >"$tmp" 2>/dev/null || true
  python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print(0); sys.exit(0)
print(len(d) if isinstance(d, list) else 0)
' "$tmp" 2>/dev/null || echo 0
  rm -f "$tmp"
}
BEFORE_ITERS=$(_count_iters "$DIFF_BEFORE")
AFTER_ITERS=$(_count_iters "$DIFF_AFTER")
printf "  %-55s " "diff --iteration 1 --format json"
if [[ "$BEFORE_ITERS" -ge 2 && "$AFTER_ITERS" -ge 2 ]]; then
  run_capture "diff --iteration 1 --format json" "$DIFF_ITER_OUT" \
    nsys-ai diff "$DIFF_BEFORE" "$DIFF_AFTER" --format json --iteration 1
  check_json_key "  diff --iteration 1: top_regressions key" "$DIFF_ITER_OUT" 'top_regressions'
else
  echo "SKIP (iterations unavailable: before=$BEFORE_ITERS after=$AFTER_ITERS)"
fi
rm -f "$DIFF_ITER_OUT"

# 8e. search --type nvtx (M8_DIFF.md §3 command 3 — discover NVTX names before drilling)
#     Gate on HAS_NVTX: "No results found." is non-empty so -s alone is not sufficient.
SEARCH_OUT="$(mktemp)"
printf "  %-55s " "search --type nvtx --query '' --limit 5"
if [[ "${HAS_NVTX:-0}" == "1" ]]; then
  if nsys-ai search "$NVTX_PROFILE" --query '' --type nvtx --limit 5 >"$SEARCH_OUT" 2>/dev/null; then
    echo "OK"
  else
    echo "FAIL"; FAIL=$((FAIL+1))
  fi
else
  echo "SKIP (no NVTX events in profile)"
fi
rm -f "$SEARCH_OUT"
rm -f "$DIFF_OUT"

# ── Mode 9: Variance ──────────────────────────────────────────────────────────
echo "== Mode 9 — Variance =="

# 9a. iteration_timing (precondition gate — M9_VARIANCE.md §1).
#     On minimal fixtures without NVTX (e.g. tests/fixtures/mock.sqlite) the skill
#     errors with "no NVTX_EVENTS table"; M9_VARIANCE §1 mandates the mode treat
#     this as a precondition-fail (route to Mode 5 Path B). In smoke context,
#     accept list OR NVTX-missing error as correct behavior; FAIL only on
#     unexpected shapes.
ITER_TIMING_OUT="$(mktemp)"
# Discard stderr so nsys-ai's cache-build banner lines don't corrupt the JSON
# parser (they are printed to stderr but a merged `2>&1` would break json.load).
nsys-ai skill run iteration_timing "$NVTX_PROFILE" --format json >"$ITER_TIMING_OUT" 2>/dev/null || true
printf "  %-55s " "skill iteration_timing --format json"
ITER_SHAPE=$(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("invalid"); sys.exit(0)
if isinstance(d, list):
    print("list")
elif isinstance(d, dict) and "NVTX" in d.get("error", {}).get("message", ""):
    # Only SKIP on the specific "no NVTX events" error — any other error
    # (SQL/schema/etc.) falls through to `other` and FAILs.
    print("nvtx_missing")
else:
    print("other")
' "$ITER_TIMING_OUT" 2>/dev/null || echo invalid)
case "$ITER_SHAPE" in
  list) echo "OK" ;;
  nvtx_missing) echo "SKIP (no NVTX events — precondition blocks per M9_VARIANCE.md §1)" ;;
  *) echo "FAIL (unexpected shape: $ITER_SHAPE)"; FAIL=$((FAIL+1)) ;;
esac
printf "  %-55s " "iteration_timing: returns list"
if [[ "$ITER_SHAPE" == "list" ]]; then
  echo "OK"
elif [[ "$ITER_SHAPE" == "nvtx_missing" ]]; then
  echo "SKIP (no NVTX)"
else
  echo "FAIL (expected list or nvtx_missing)"; FAIL=$((FAIL+1))
fi

# 9b. iteration_detail on iteration 1 (skip index 0 per M9_VARIANCE.md §4)
ITER_DETAIL_OUT="$(mktemp)"
ITER_COUNT=$(python3 -c "
import json
try:
    d = json.load(open('$ITER_TIMING_OUT'))
except Exception:
    print(0); exit()
print(len(d) if isinstance(d, list) else 0)
" 2>/dev/null || echo 0)
if [[ "$ITER_COUNT" -ge 2 ]]; then
  run_capture "skill iteration_detail -p iteration=1 --format json" "$ITER_DETAIL_OUT" \
    nsys-ai skill run iteration_detail "$NVTX_PROFILE" --format json -p iteration=1
  printf "  %-55s " "iteration_detail: has duration_ms field"
  DET_FIELDS=$(python3 -c "
import json; d=json.load(open('$ITER_DETAIL_OUT'))
rows = d if isinstance(d,list) else [d]
print('ok' if rows and 'duration_ms' in rows[0] else 'missing')
" 2>/dev/null || echo missing)
  if [[ "$DET_FIELDS" == "ok" ]]; then echo "OK"; else echo "FAIL"; FAIL=$((FAIL+1)); fi
else
  printf "  %-55s SKIP (< 2 iterations detected)\n" "skill iteration_detail"
fi
rm -f "$ITER_DETAIL_OUT"

# 9c. nccl_anomaly (Mode 9 NCCL straggler check — M9_VARIANCE.md §3 command 6).
#     Accepts empty list (no NCCL) as correct behavior on minimal profiles.
NCCL_ANOM_OUT="$(mktemp)"
# Discard stderr so cache-build banners don't corrupt the JSON parser.
# `nccl_anomaly` returns `[]` on profiles with no NCCL (no error dict needed),
# so any error-shaped payload here indicates a real bug (SQL/schema/skill) and
# must FAIL — we deliberately do NOT treat `{"error": ...}` as a soft SKIP.
nsys-ai skill run nccl_anomaly "$NVTX_PROFILE" --format json >"$NCCL_ANOM_OUT" 2>/dev/null || true
printf "  %-55s " "skill nccl_anomaly --format json"
NCCL_SHAPE=$(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("invalid"); sys.exit(0)
if isinstance(d, list):
    print("list")
else:
    print("unexpected")
' "$NCCL_ANOM_OUT" 2>/dev/null || echo invalid)
case "$NCCL_SHAPE" in
  list) echo "OK" ;;
  *) echo "FAIL (unexpected shape: $NCCL_SHAPE)"; FAIL=$((FAIL+1)) ;;
esac
printf "  %-55s " "nccl_anomaly: returns list"
if [[ "$NCCL_SHAPE" == "list" ]]; then
  echo "OK"
else
  echo "FAIL (expected list; got $NCCL_SHAPE)"; FAIL=$((FAIL+1))
fi
rm -f "$NCCL_ANOM_OUT" "$ITER_TIMING_OUT"

# ── Plugin skill name check ───────────────────────────────────────────────────
echo "== Plugin skill name (/nsys-ai) =="
SKILL_NAME=$(python3 -c "
import re, pathlib
content = pathlib.Path('skills/analyze/SKILL.md').read_text()
m = re.search(r'^name:\s*(\S+)', content, re.MULTILINE)
print(m.group(1) if m else 'NOT_FOUND')
" 2>/dev/null || echo NOT_FOUND)
printf "  %-55s " "SKILL.md name == nsys-ai"
if [[ "$SKILL_NAME" == "nsys-ai" ]]; then
  echo "OK"
else
  echo "FAIL (got: $SKILL_NAME)"
  FAIL=$((FAIL+1))
fi

echo ""
if [[ $FAIL -eq 0 ]]; then
  echo "All checks passed."
  exit 0
else
  echo "$FAIL check(s) FAILED."
  exit 1
fi
