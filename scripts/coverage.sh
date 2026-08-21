#!/usr/bin/env bash
# Run pytest with coverage for both the pytest process and CLI subprocesses.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_cmd="${PYTHON:-python}"
if ! command -v "$python_cmd" >/dev/null 2>&1; then
    python_cmd=python3
fi

coverage_file="${COVERAGE_FILE:-$repo_root/.coverage}"
export COVERAGE_FILE="$coverage_file"

"$python_cmd" -m coverage erase

if (($# == 0)); then
    set -- tests/
fi

# pytest-cov reports the parent process and tests/conftest.py passes the
# startup hook to every CLI child. Suppress its inline report so the explicit
# combine/report below is the single final report.
"$python_cmd" -m pytest "$@" --cov=src/nsys_ai --cov-report=

# pytest-cov normally combines these itself. Keep the explicit step for
# sidecars left by a child that exits after the plugin's reporting phase, and
# make the script useful with older pytest-cov versions too.
shopt -s nullglob
parts=("${coverage_file}".*)
if ((${#parts[@]})); then
    "$python_cmd" -m coverage combine --data-file "$coverage_file" "$repo_root"
fi

"$python_cmd" -m coverage report --data-file "$coverage_file" --show-missing
