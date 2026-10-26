#!/usr/bin/env bash
set -euo pipefail
cd /home/liugang/src/crystalDLM
source research/liugang_setup_20261015/runtime.sh
OUT=experiments/structure_request_eval_20261017
trap 'code=$?; echo "$code" > "$OUT/preflight-002-exit.txt"' EXIT
MAMBA=/home/liugang/software/miniforge3/condabin/mamba
"$MAMBA" run --prefix "$CRYSTAL_PROJECT/envs/crystaldlm" python -m unittest discover -s "$OUT" -p test_run.py -v
active=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)
test -z "$active"
CUDA_VISIBLE_DEVICES=0 "$MAMBA" run --prefix "$CRYSTAL_PROJECT/envs/crystaldlm" python "$OUT/run.py" --out "$OUT/preflight_002" --limit 1
