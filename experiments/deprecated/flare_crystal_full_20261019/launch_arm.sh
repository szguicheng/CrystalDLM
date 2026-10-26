#!/usr/bin/env bash
set -euo pipefail
cd /home/liugang/src/crystalDLM
source research/liugang_setup_20261015/runtime.sh
EXP=experiments/flare_crystal_full_20261019
ARM="$1"
[[ "$ARM" == ar || "$ARM" == flare ]]
MAMBA=/home/liugang/software/miniforge3/condabin/mamba
if [[ "$ARM" == flare ]]; then GPUS=0,1; EVAL_GPU=0; WORLD=2; else GPUS=2; EVAL_GPU=2; WORLD=1; fi
trap 'code=$?; echo "$code" > "$EXP/$ARM-exit.txt"' EXIT
check_gpus() { test -z "$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader)"; }
check_gpus "$GPUS"
printf 'training_smoke\n' > "$EXP/$ARM-stage.txt"
CUDA_VISIBLE_DEVICES="$GPUS" "$MAMBA" run --prefix envs/crystaldlm torchrun --standalone --nproc_per_node="$WORLD" "$EXP/run.py" --mode train --arm "$ARM" --out "$EXP/$ARM" --stop-after-update 2 > "$EXP/$ARM-smoke.log" 2>&1
printf 'training\n' > "$EXP/$ARM-stage.txt"
check_gpus "$GPUS"
CUDA_VISIBLE_DEVICES="$GPUS" "$MAMBA" run --prefix envs/crystaldlm torchrun --standalone --nproc_per_node="$WORLD" "$EXP/run.py" --mode train --arm "$ARM" --out "$EXP/$ARM" --resume > "$EXP/$ARM-training.log" 2>&1
printf 'generation_preflight\n' > "$EXP/$ARM-stage.txt"
check_gpus "$EVAL_GPU"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$MAMBA" run --prefix envs/crystaldlm python -u "$EXP/evaluate.py" --arm "$ARM" --out "$EXP/${ARM}_preflight" --limit 4 > "$EXP/$ARM-preflight.log" 2>&1
"$MAMBA" run --prefix envs/crystaldlm python "$EXP/evaluate.py" --mode summarize --arm "$ARM" --out "$EXP/${ARM}_preflight" > "$EXP/$ARM-preflight-summary.log"
"$MAMBA" run --prefix envs/crystaldlm python - "$EXP/${ARM}_preflight/summary.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['completed_attempts']==28
assert r['strict_valid_structures']>0, 'quality gate: all 28 end-to-end attempts invalid; stop for diagnosis'
PY
printf 'validation_900\n' > "$EXP/$ARM-stage.txt"
check_gpus "$EVAL_GPU"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$MAMBA" run --prefix envs/crystaldlm python -u "$EXP/run.py" --mode validate --arm "$ARM" --out "$EXP/$ARM" > "$EXP/$ARM-validation.log" 2>&1
printf 'generation_700\n' > "$EXP/$ARM-stage.txt"
check_gpus "$GPUS"
if [[ "$ARM" == flare ]]; then
  CUDA_VISIBLE_DEVICES=0 "$MAMBA" run --prefix envs/crystaldlm python -u "$EXP/evaluate.py" --arm "$ARM" --out "$EXP/${ARM}_evaluation" --num-shards 2 --shard-index 0 > "$EXP/$ARM-generation-0.log" 2>&1 &
  pid0=$!
  CUDA_VISIBLE_DEVICES=1 "$MAMBA" run --prefix envs/crystaldlm python -u "$EXP/evaluate.py" --arm "$ARM" --out "$EXP/${ARM}_evaluation" --num-shards 2 --shard-index 1 > "$EXP/$ARM-generation-1.log" 2>&1 &
  pid1=$!
  set +e
  wait "$pid0"; code0=$?
  wait "$pid1"; code1=$?
  set -e
  printf '{"shard_0":%s,"shard_1":%s}\n' "$code0" "$code1" > "$EXP/flare-generation-exit.json"
  test "$code0" -eq 0 && test "$code1" -eq 0
else
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$MAMBA" run --prefix envs/crystaldlm python -u "$EXP/evaluate.py" --arm "$ARM" --out "$EXP/${ARM}_evaluation" > "$EXP/$ARM-generation.log" 2>&1
fi
"$MAMBA" run --prefix envs/crystaldlm python "$EXP/evaluate.py" --mode summarize --arm "$ARM" --out "$EXP/${ARM}_evaluation" > "$EXP/$ARM-generation-summary.log"
printf 'grace_scoring\n' > "$EXP/$ARM-stage.txt"
check_gpus "$EVAL_GPU"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" TF_CPP_MIN_LOG_LEVEL=2 "$MAMBA" run --prefix envs/grace python experiments/structure_request_eval_20261017/grace_score.py --mode score --run-dir "$EXP/${ARM}_evaluation" > "$EXP/$ARM-grace.log" 2>&1
printf 'complete\n' > "$EXP/$ARM-stage.txt"
