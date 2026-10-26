#!/usr/bin/env bash
set -euo pipefail
cd /home/liugang/src/crystalDLM
source research/liugang_setup_20261015/runtime.sh
EXP=experiments/structure_conditioned_hybrid8b_20261018
MAMBA=/home/liugang/software/miniforge3/condabin/mamba
PREFIX="$CRYSTAL_PROJECT/envs/crystaldlm"
BASE="$CRYSTAL_PROJECT/models/Qwen3-8B-b968826d9c46"
trap 'code=$?; echo "$code" > "$EXP/pipeline-exit.txt"' EXIT
check_gpus() {
  active=$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader)
  test -z "$active"
}
printf 'cpu_checks\n' > "$EXP/stage.txt"
"$MAMBA" run --prefix "$PREFIX" python -m unittest discover -s "$EXP" -p 'test_*.py' -v
printf 'training_preflight\n' > "$EXP/stage.txt"
check_gpus 0,1
CUDA_VISIBLE_DEVICES=0,1 "$MAMBA" run --prefix "$PREFIX" torchrun --standalone --nproc_per_node=2 "$EXP/train.py" --mode preflight --model-path "$BASE" --out "$EXP/preflight_001" > "$EXP/preflight.log" 2>&1
printf 'training\n' > "$EXP/stage.txt"
check_gpus 0,1
CUDA_VISIBLE_DEVICES=0,1 "$MAMBA" run --prefix "$PREFIX" torchrun --standalone --nproc_per_node=2 "$EXP/train.py" --mode train --model-path "$BASE" --out "$EXP/run_001" --preflight-receipt "$EXP/preflight_001/summary.json" > "$EXP/train.log" 2>&1
printf 'generation_preflight\n' > "$EXP/stage.txt"
check_gpus 0
CUDA_VISIBLE_DEVICES=0 "$MAMBA" run --prefix "$PREFIX" python "$EXP/evaluate.py" --model-path "$BASE" --checkpoint "$EXP/run_001/checkpoint.pt" --out "$EXP/evaluation_preflight_001" --limit 1 > "$EXP/generation-preflight.log" 2>&1
printf 'generation_700\n' > "$EXP/stage.txt"
check_gpus 0,1
CUDA_VISIBLE_DEVICES=0 "$MAMBA" run --prefix "$PREFIX" python "$EXP/evaluate.py" --model-path "$BASE" --checkpoint "$EXP/run_001/checkpoint.pt" --out "$EXP/evaluation_001" --num-shards 2 --shard-index 0 > "$EXP/generation-shard-0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$MAMBA" run --prefix "$PREFIX" python "$EXP/evaluate.py" --model-path "$BASE" --checkpoint "$EXP/run_001/checkpoint.pt" --out "$EXP/evaluation_001" --num-shards 2 --shard-index 1 > "$EXP/generation-shard-1.log" 2>&1 &
pid1=$!
set +e
wait "$pid0"; code0=$?
wait "$pid1"; code1=$?
printf '{"shard_0":%s,"shard_1":%s}\n' "$code0" "$code1" > "$EXP/generation-exit.json"
set -e
"$MAMBA" run --prefix "$PREFIX" python "$EXP/evaluate.py" --mode summarize --out "$EXP/evaluation_001" > "$EXP/generation-summary.log"
test "$code0" -eq 0 && test "$code1" -eq 0
printf 'grace_scoring\n' > "$EXP/stage.txt"
check_gpus 0
CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=2 "$MAMBA" run --prefix "$CRYSTAL_PROJECT/envs/grace" python experiments/structure_request_eval_20261017/grace_score.py --mode score --run-dir "$EXP/evaluation_001" > "$EXP/grace-scoring.log" 2>&1
printf 'complete\n' > "$EXP/stage.txt"
