#!/usr/bin/env bash
set -euo pipefail
cd /home/liugang/src/crystalDLM
source research/liugang_setup_20261015/runtime.sh
EXP=experiments/flare_crystal_qwen3_8b_20261019
ARM="$1"
GPU="$2"
[[ "$ARM" == ar || "$ARM" == flare || "$ARM" == source ]]
[[ "$GPU" == 0 || "$GPU" == 1 || "$GPU" == 2 ]]
trap 'code=$?; echo "$code" > "$EXP/$ARM-exit.txt"' EXIT
test -z "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES="$GPU"
MAMBA=/home/liugang/software/miniforge3/condabin/mamba
if [[ "$ARM" != source ]]; then
  "$MAMBA" run --prefix envs/crystaldlm python -u "$EXP/run.py" --arm "$ARM" --out "$EXP/$ARM" --stop-after-update 2 > "$EXP/$ARM-smoke.log" 2>&1
  "$MAMBA" run --prefix envs/crystaldlm python -u "$EXP/run.py" --arm "$ARM" --out "$EXP/$ARM" --resume > "$EXP/$ARM-training.log" 2>&1
fi
"$MAMBA" run --prefix envs/crystaldlm python -u "$EXP/evaluate.py" --arm "$ARM" > "$EXP/$ARM-evaluation.log" 2>&1
