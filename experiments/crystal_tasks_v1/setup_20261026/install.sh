#!/usr/bin/env bash
set -euo pipefail
cd /home/liugang/src/crystalDLM
source research/liugang_setup_20261015/runtime.sh
export CONDA_REGISTER_ENVS=false CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
OUT="$CRYSTAL_PROJECT/experiments/crystal_tasks_v1/setup_20261026"
trap 'code=$?; echo "$code" > "$OUT/install.exit"' EXIT
MAMBA=/home/liugang/software/miniforge3/condabin/mamba
PREFIX="$CRYSTAL_PROJECT/envs/crystaldlm"
bash "$CRYSTAL_PROJECT/experiments/crystal_tasks_v1/python.sh" -m pip install --report "$OUT/install-report.json" -c "$OUT/constraints.txt" -r "$OUT/resolved-requirements.txt" -e ./work/LLaMA-Factory-0.9.5
bash "$CRYSTAL_PROJECT/experiments/crystal_tasks_v1/python.sh" -m pip check
bash "$CRYSTAL_PROJECT/experiments/crystal_tasks_v1/python.sh" -m pip freeze > "$OUT/pip-after.txt"
bash "$CRYSTAL_PROJECT/experiments/crystal_tasks_v1/python.sh" -c 'import llamafactory, accelerate, torch; print("INSTALLED",llamafactory.__version__,accelerate.__version__,torch.__version__)'
