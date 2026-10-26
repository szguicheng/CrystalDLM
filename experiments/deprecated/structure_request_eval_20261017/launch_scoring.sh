#!/usr/bin/env bash
set -euo pipefail
cd /home/liugang/src/crystalDLM
source research/liugang_setup_20261015/runtime.sh
OUT=experiments/structure_request_eval_20261017
MAMBA=/home/liugang/software/miniforge3/condabin/mamba
PREFIX="$CRYSTAL_PROJECT/envs/grace"
trap 'code=$?; echo "$code" > "$OUT/scoring-pipeline-exit.txt"' EXIT
active=$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader)
test -z "$active"
sha256sum "$OUT/grace_score.py" "$OUT/test_grace_score.py" "$OUT/launch_scoring.sh" > "$OUT/scoring-source-sha256.txt"
CUDA_VISIBLE_DEVICES=2 TF_CPP_MIN_LOG_LEVEL=2 "$MAMBA" run --prefix "$PREFIX" python "$OUT/grace_score.py" --mode score --run-dir "$OUT/run_001" > "$OUT/score-preview.log" 2>&1
printf 'waiting_for_generation\n' > "$OUT/scoring-stage.txt"
while test ! -f "$OUT/full-exit.json"; do sleep 30; done
"$MAMBA" run --prefix "$PREFIX" python -c 'import json; from pathlib import Path; d=json.loads(Path("experiments/structure_request_eval_20261017/full-exit.json").read_text()); assert d=={"shard_0":0,"shard_1":0},d'
active=$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader)
test -z "$active"
printf 'final_scoring\n' > "$OUT/scoring-stage.txt"
CUDA_VISIBLE_DEVICES=2 TF_CPP_MIN_LOG_LEVEL=2 "$MAMBA" run --prefix "$PREFIX" python "$OUT/grace_score.py" --mode score --run-dir "$OUT/run_001" --resume > "$OUT/score-final.log" 2>&1
printf 'complete\n' > "$OUT/scoring-stage.txt"
