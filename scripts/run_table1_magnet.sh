#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

for atlas in msdl aal116 cc200; do
  datasets=abide
  if [[ "${atlas}" == msdl ]]; then
    datasets=abide,adni,oasis3
  fi
  "${PYTHON_BIN}" -u "${ROOT_DIR}/src/train_gdt.py" \
    --datasets "${datasets}" --atlas "${atlas}" \
    --results-dir "${ROOT_DIR}/results/table1/${atlas}/gdt" \
    --n-splits 10 --test-size 0.1 --seed 42 --n-jobs 1 \
    --gdt-epochs 200 --gdt-batch-size 64 --gdt-ddim-steps 6 \
    --gdt-ddim-eta 0.05 --gdt-cfg-scale 1.5 \
    --gdt-spectral-feature-mode amortized_bridge_v1 \
    --gdt-subtype-conditioning-mode hard --gdt-min-samples-subtype 80 \
    --gdt-use-prior-slow-noising 1 --gdt-corruption-law adaptive \
    --gdt-prior-kappa 0.5 --gdt-prior-power 3 --gdt-prior-temp 0.5 \
    --skip-existing
done
