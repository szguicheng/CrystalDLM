#!/usr/bin/env bash
set -euo pipefail
cd /home/liugang/src/crystalDLM
source research/liugang_setup_20261015/runtime.sh
OUT=experiments/structure_request_eval_20261017
trap 'code=$?; echo "$code" > "$OUT/calibration-exit.txt"' EXIT
active=$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader)
test -z "$active"
CUDA_VISIBLE_DEVICES=2 TF_CPP_MIN_LOG_LEVEL=2 /home/liugang/software/miniforge3/condabin/mamba run --prefix "$CRYSTAL_PROJECT/envs/grace" python "$OUT/grace_score.py" --mode calibrate
