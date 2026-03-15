#!/usr/bin/env bash
set -euo pipefail
# nsys_profile_wrapper.sh — Profile + export to .sqlite in one step.
#
# Usage:
#   ./nsys_profile_wrapper.sh <output_name> <command...>
#
# Example:
#   ./nsys_profile_wrapper.sh exp_12 uv run train.py
#
# Produces: <output_name>.nsys-rep and <output_name>.sqlite
# The .sqlite path is printed as the last line of stdout.

if [ $# -lt 2 ]; then
    echo "Usage: $0 <output_name> <command...>"
    echo ""
    echo "Examples:"
    echo "  $0 run1 python train.py"
    echo "  $0 exp_12 uv run train.py --batch-size 32"
    echo ""
    echo "This script:"
    echo "  1. Profiles the command with nsys"
    echo "  2. Exports the .nsys-rep to .sqlite"
    echo "  3. Prints the .sqlite path on the last line"
    exit 1
fi

NAME="${1}"
shift

echo ">>> nsys profile -o ${NAME} $*" >&2
nsys profile --stats=true -o "${NAME}" "$@"

REP="${NAME}.nsys-rep"
SQLITE="${NAME}.sqlite"

echo ">>> nsys export --type sqlite ${REP}" >&2
nsys export --type sqlite --output "${SQLITE}" "${REP}"

echo ">>> Profile ready: ${SQLITE}" >&2
echo "${SQLITE}"
