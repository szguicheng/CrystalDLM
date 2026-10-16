#!/usr/bin/env bash
set -euo pipefail
stage_root=$(pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$stage_root/.venv/bin/python" -m pytest "$stage_root/research/crystal_sharing_20261012/test_sharing.py" -q
"$stage_root/.venv/bin/python" "$stage_root/research/crystal_sharing_20261012/run_sharing.py"
