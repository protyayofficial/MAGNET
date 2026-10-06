#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DIFFEO_METRICS_DIR="${DIFFEO_METRICS_DIR:-${ROOT_DIR}/external/DiffeoCFM}"
export DIFFEO_METRICS_DIR
if [[ ! -f "${DIFFEO_METRICS_DIR}/deterministic_distribution_metrics.py" ]]; then
  echo "Clone https://github.com/antoinecollas/DiffeoCFM into ${DIFFEO_METRICS_DIR} for its metric definitions." >&2
  exit 1
fi
"${PYTHON_BIN}" -u "${ROOT_DIR}/runners/zero_shot_hierarchical_eval/evaluate_zero_shot_hierarchical.py" \
  --gdt-results-dir "${ROOT_DIR}/results/table1/msdl/gdt" \
  --output-root "${ROOT_DIR}/results/zero_shot_magnet" \
  --datasets adni,oasis3 --quality-metrics full "$@"
