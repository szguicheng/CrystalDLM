#!/usr/bin/env bash
set -euo pipefail
cd /home/liugang/src/crystalDLM
OUT="$PWD/experiments/crystal_tasks_v1/phase3_understanding_20261026"
PY="$PWD/experiments/crystal_tasks_v1/python.sh"
mkdir -p "$OUT"
trap 'code=$?; printf "%s\n" "$code" > "$OUT/pipeline.exit"' EXIT

check_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits > "$OUT/gpu-before-$1.csv"
  bash "$PY" -c 'import sys; rows=[line.split(",") for line in open(sys.argv[1])]; assert all(int(used)<512 for index,used in rows if int(index) in (0,1,2)), "Selected GPU is occupied"' "$OUT/gpu-before-$1.csv"
}

printf 'overfit_training\n' > "$OUT/stage.txt"
check_gpus overfit
CUDA_VISIBLE_DEVICES=0,1,2 bash "$PY" -m torch.distributed.run --standalone --nproc_per_node=3 --module experiments.crystal_tasks_v1.train_ar --mode overfit --run-dir "$OUT/overfit" > "$OUT/overfit.log" 2>&1

printf 'overfit_greedy_and_nll\n' > "$OUT/stage.txt"
CUDA_VISIBLE_DEVICES=0 bash "$PY" -m experiments.crystal_tasks_v1.evaluate_ar --checkpoint "$OUT/overfit/final" --split training --overfit32 --output "$OUT/overfit_check" > "$OUT/overfit_check.log" 2>&1

printf 'full_training\n' > "$OUT/stage.txt"
check_gpus full
CUDA_VISIBLE_DEVICES=0,1,2 bash "$PY" -m torch.distributed.run --standalone --nproc_per_node=3 --module experiments.crystal_tasks_v1.train_ar --mode full --run-dir "$OUT/full" > "$OUT/full.log" 2>&1

printf 'full_validation_and_test\n' > "$OUT/stage.txt"
CUDA_VISIBLE_DEVICES=0 bash "$PY" -m experiments.crystal_tasks_v1.evaluate_ar --checkpoint "$OUT/full/final" --split validation --output "$OUT/final_validation" > "$OUT/final_validation.log" 2>&1 &
val_pid=$!
CUDA_VISIBLE_DEVICES=1 bash "$PY" -m experiments.crystal_tasks_v1.evaluate_ar --checkpoint "$OUT/full/final" --split test --output "$OUT/final_test" > "$OUT/final_test.log" 2>&1 &
test_pid=$!
wait "$val_pid"
wait "$test_pid"

bash "$PY" - "$OUT" <<'PY'
import json,hashlib,sys
from pathlib import Path
p=Path(sys.argv[1]);m=json.loads((p/'full/run_meta.json').read_text());best=Path(m['best_checkpoint'])
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
same=sha(best/'adapter_model.safetensors')==sha(p/'full/final/adapter_model.safetensors')
(p/'checkpoint-selection.json').write_text(json.dumps({'best_checkpoint':str(best),'final_checkpoint':str(p/'full/final'),'same_adapter_weights':same},indent=2)+'\n')
if not same:(p/'best-to-evaluate.txt').write_text(str(best)+'\n')
PY
if [ -f "$OUT/best-to-evaluate.txt" ]; then
  best_checkpoint=$(cat "$OUT/best-to-evaluate.txt")
  printf 'best_validation_and_test\n' > "$OUT/stage.txt"
  CUDA_VISIBLE_DEVICES=0 bash "$PY" -m experiments.crystal_tasks_v1.evaluate_ar --checkpoint "$best_checkpoint" --split validation --output "$OUT/best_validation" > "$OUT/best_validation.log" 2>&1 &
  val_pid=$!
  CUDA_VISIBLE_DEVICES=1 bash "$PY" -m experiments.crystal_tasks_v1.evaluate_ar --checkpoint "$best_checkpoint" --split test --output "$OUT/best_test" > "$OUT/best_test.log" 2>&1 &
  test_pid=$!
  wait "$val_pid"
  wait "$test_pid"
fi
printf 'complete\n' > "$OUT/stage.txt"
