#!/usr/bin/env bash
# Smoke test for nsys-ai plugin — validates that every CLI command referenced
# in skills/analyze/SKILL.md + M*.md still works with the installed nsys-ai version.
#
# Usage:  ./scripts/smoke_test.sh <profile.sqlite>
# Exit 0 if all commands succeed, non-zero otherwise.
#
# Stage A extension: adds Mode 1 end-to-end dry-run covering the three CLI boundaries
# that the plugin actually orchestrates (manifest → evidence → timeline). See
# docs/claude-skill-plan.md §11 Stage A acceptance tests.

set -euo pipefail

PROFILE="${1:-}"
if [[ -z "$PROFILE" || ! -f "$PROFILE" ]]; then
  echo "usage: $0 <profile.sqlite>" >&2
  exit 2
fi

FAIL=0
run() {
  local label="$1"; shift
  printf "  %-45s " "$label"
  if "$@" >/dev/null 2>&1; then
    echo "OK"
  else
    echo "FAIL"
    FAIL=$((FAIL+1))
  fi
}

run_capture() {
  # Like run, but captures stdout to a temp file for subsequent regex checks.
  local label="$1"; shift
  local out="$1"; shift
  printf "  %-45s " "$label"
  if "$@" >"$out" 2>/dev/null; then
    echo "OK"
  else
    echo "FAIL"
    FAIL=$((FAIL+1))
  fi
}

check_regex() {
  local label="$1"; local file="$2"; local pattern="$3"
  printf "  %-45s " "$label"
  if grep -qE "$pattern" "$file" 2>/dev/null; then
    echo "OK"
  else
    echo "FAIL (pattern /$pattern/ not found)"
    FAIL=$((FAIL+1))
  fi
}

echo "== Top-level commands =="
run "nsys-ai help"           nsys-ai --help
run "nsys-ai skill list"     nsys-ai skill list
# cutracer check exits non-zero if .so is not built — informational only
printf "  %-45s " "nsys-ai cutracer check (info)"
if nsys-ai cutracer check >/dev/null 2>&1; then
  echo "OK (.so installed)"
else
  echo "SKIP (.so not built — expected for analysis-only installs)"
fi

echo "== Mode 1 routing skills (profile_health_manifest + drill-down targets) =="
MANIFEST_OUT="$(mktemp)"
trap 'rm -f "$MANIFEST_OUT" /tmp/findings_smoke.json' EXIT
run_capture "profile_health_manifest" "$MANIFEST_OUT" \
  nsys-ai skill run profile_health_manifest "$PROFILE" --format json
check_regex "manifest has gpu field"         "$MANIFEST_OUT" '"gpu"'
check_regex "manifest has profile_span_ms"   "$MANIFEST_OUT" '"profile_span_ms"'
check_regex "manifest has suspected_bottleneck" "$MANIFEST_OUT" '"suspected_bottleneck"'
# NVTX-dependent skills fail on profiles without an NVTX_EVENTS table.
# Missing NVTX is a non-blocking scenario for the plugin (Mode 5 falls back to Path B),
# so we skip rather than fail here.
HAS_NVTX=$(sqlite3 "$PROFILE" "SELECT COUNT(*) FROM sqlite_master WHERE name='NVTX_EVENTS';" 2>/dev/null || echo 0)
if [[ "$HAS_NVTX" -gt 0 ]]; then
  run "nvtx_layer_breakdown"   nsys-ai skill run nvtx_layer_breakdown "$PROFILE" --format json --max-rows 1
else
  printf "  %-45s " "nvtx_layer_breakdown"
  echo "SKIP (no NVTX_EVENTS table)"
fi
run "root_cause_matcher"       nsys-ai skill run root_cause_matcher "$PROFILE" --format json

echo "== Mode 2 drill-down skills (comms) =="
run "overlap_breakdown"        nsys-ai skill run overlap_breakdown "$PROFILE" --format json
run "nccl_breakdown"           nsys-ai skill run nccl_breakdown "$PROFILE" --format json
run "kernel_overlap_matrix"    nsys-ai skill run kernel_overlap_matrix "$PROFILE" --format json

echo "== Mode 6 drill-down skills (idle / sync) =="
run "gpu_idle_gaps"            nsys-ai skill run gpu_idle_gaps "$PROFILE" --format json -p min_gap_ns=1000000
run "sync_cost_analysis"       nsys-ai skill run sync_cost_analysis "$PROFILE" --format json
run "kernel_launch_overhead"   nsys-ai skill run kernel_launch_overhead "$PROFILE" --format json
run "cpu_gpu_pipeline"         nsys-ai skill run cpu_gpu_pipeline "$PROFILE" --format json
run "stream_concurrency"       nsys-ai skill run stream_concurrency "$PROFILE" --format json
run "pipeline_bubble_metrics"  nsys-ai skill run pipeline_bubble_metrics "$PROFILE" --format json

echo "== Mode 3 drill-down skills (compute hotspot) =="
run "top_kernels"              nsys-ai skill run top_kernels "$PROFILE" --format json --max-rows 5
run "tensor_core_usage"        nsys-ai skill run tensor_core_usage "$PROFILE" --format json
run "kernel_launch_pattern"    nsys-ai skill run kernel_launch_pattern "$PROFILE" --format json

echo "== Mode 4 drill-down skills (memory) =="
run "memory_bandwidth"         nsys-ai skill run memory_bandwidth "$PROFILE" --format json
run "memory_transfers"         nsys-ai skill run memory_transfers "$PROFILE" --format json
run "h2d_distribution"         nsys-ai skill run h2d_distribution "$PROFILE" --format json

echo "== Mode 5 drill-down skills (code mapping) =="
run "iteration_timing"         nsys-ai skill run iteration_timing "$PROFILE" --format json

echo "== Stage A Mode 1 smoke — evidence/timeline CLI surface =="
# The plugin ends every mode with evidence build + timeline-web. In CI we only smoke
# the CLI boundary here: build findings JSON, validate its shape, and verify the
# timeline-web command is available. This does not start the HTTP server.
run "evidence build"           nsys-ai evidence build "$PROFILE" --format json -o /tmp/findings_smoke.json
run "findings JSON is valid"   bash -c "python3 -c 'import json,sys; json.load(open(\"/tmp/findings_smoke.json\"))'"
run "findings has findings key" bash -c "python3 -c 'import json; assert \"findings\" in json.load(open(\"/tmp/findings_smoke.json\"))'"
run "timeline-web --help"      nsys-ai timeline-web --help

echo ""
if [[ $FAIL -eq 0 ]]; then
  echo "All checks passed."
  exit 0
else
  echo "$FAIL check(s) FAILED."
  exit 1
fi
