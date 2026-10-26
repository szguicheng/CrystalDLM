#!/usr/bin/env bash
set -euo pipefail
cd /home/liugang/src/crystalDLM
source research/liugang_setup_20261015/runtime.sh
EXP=experiments/structure_conditioned_hybrid8b_20261018/validation_001
trap 'code=$?; echo "$code" > "$EXP/source-exit.txt"' EXIT
test -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
CUDA_VISIBLE_DEVICES=0 /home/liugang/software/miniforge3/condabin/mamba run --prefix envs/crystaldlm python -u "$EXP/evaluate.py" --arm source --samplers original confidence remask > "$EXP/source-evaluation.log" 2>&1
