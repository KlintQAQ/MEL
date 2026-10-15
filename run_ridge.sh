#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [ "$#" -eq 0 ]; then
    printf 'Usage: bash run_ridge.sh sub01-scan01 [more scan identifiers]\n' >&2
    exit 2
fi
"${PYTHON:-python}" model/run.py --scans "$@" --readouts ridge --output-dir "${OUTPUT_DIR:-results/main_ridge}" --check-reference
