#!/usr/bin/env bash
# Execute only inside a Slurm CPU allocation; no LLM credentials are copied.
set -euo pipefail
stage_root=$(pwd)
stage_dir="$stage_root/research/crystal_preannotation_20261012"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HTTP_PROXY=http://proxy.cityu.edu.hk:8080 HTTPS_PROXY=http://proxy.cityu.edu.hk:8080
/scratch/chenggui2/anaconda3/bin/python -m venv "$stage_root/.venv"
"$stage_root/.venv/bin/pip" install -r "$stage_dir/requirements-cpu.txt" > "$stage_dir/install.log" 2>&1
"$stage_root/.venv/bin/python" -m pytest "$stage_dir/test_preprocess.py" -q > "$stage_dir/tests.log" 2>&1
"$stage_root/.venv/bin/python" "$stage_dir/run_full.py"
