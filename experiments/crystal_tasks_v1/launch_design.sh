#!/usr/bin/env bash
set -euo pipefail
cd /home/liugang/src/crystalDLM
OUT="${1:-$PWD/experiments/crystal_tasks_v1/phase3_design_20261026}"
overfit_epochs="${2:-60}"
PY="$PWD/experiments/crystal_tasks_v1/python.sh"
devices="${CUDA_VISIBLE_DEVICES:?select three free GPU indices before launch}"
first_gpu="${devices%%,*}"
mkdir "$OUT"
trap 'code=$?; printf "%s\n" "$code" > "$OUT/pipeline.exit"' EXIT
DATA=$(bash "$PY" -c 'from experiments.crystal_tasks_v1.configuration import task_data;print(task_data("ar_design")["dataset_dir"])')

check_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits > "$OUT/gpu-before-$1.csv"
  bash "$PY" - "$OUT/gpu-before-$1.csv" "$devices" <<'PY'
import sys
selected=sys.argv[2].split(',');assert len(selected)==len(set(selected))==3
memory={str(int(line.split(',')[0])):int(line.split(',')[1]) for line in open(sys.argv[1])}
assert all(memory[gpu]<512 for gpu in selected),'Selected GPU is occupied'
PY
}

printf 'overfit_training\n' > "$OUT/stage.txt"
check_gpus overfit
CUDA_VISIBLE_DEVICES="$devices" bash "$PY" -m torch.distributed.run --standalone --nproc_per_node=3 --module experiments.crystal_tasks_v1.train_ar --task ar_design --mode overfit --overfit-epochs "$overfit_epochs" --run-dir "$OUT/overfit" > "$OUT/overfit.log" 2>&1

printf 'overfit_greedy_and_nll\n' > "$OUT/stage.txt"
CUDA_VISIBLE_DEVICES="$first_gpu" bash "$PY" -m experiments.crystal_tasks_v1.generate_design --mode overfit --training-run-dir "$OUT/overfit" --checkpoint "$OUT/overfit/final" --output "$OUT/overfit_check" > "$OUT/overfit_check.log" 2>&1

printf 'full_training\n' > "$OUT/stage.txt"
check_gpus full
CUDA_VISIBLE_DEVICES="$devices" bash "$PY" -m torch.distributed.run --standalone --nproc_per_node=3 --module experiments.crystal_tasks_v1.train_ar --task ar_design --mode full --run-dir "$OUT/full" > "$OUT/full.log" 2>&1

CUDA_VISIBLE_DEVICES='' bash "$PY" -m experiments.crystal_tasks_v1.evaluation.plan --dataset-dir "$DATA" --output "$OUT/attempt-plan.jsonl" > "$OUT/plan.log" 2>&1
printf 'final_evaluation\n' > "$OUT/stage.txt"
CUDA_VISIBLE_DEVICES='' bash "$PY" -m experiments.crystal_tasks_v1.run_design_evaluation --checkpoint "$OUT/full/final" --attempt-plan "$OUT/attempt-plan.jsonl" --gpus "$devices" --output "$OUT/final_evaluation" > "$OUT/final_evaluation.log" 2>&1

bash "$PY" - "$OUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);meta=json.loads((p/'full/run_meta.json').read_text());best=Path(meta['best_checkpoint'])
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
same=sha(best/'adapter_model.safetensors')==sha(p/'full/final/adapter_model.safetensors')
(p/'checkpoint-selection.json').write_text(json.dumps({'best_checkpoint':str(best),'final_checkpoint':str(p/'full/final'),'same_adapter_weights':same},indent=2)+'\n')
if not same:(p/'best-to-evaluate.txt').write_text(str(best)+'\n')
PY
if [ -f "$OUT/best-to-evaluate.txt" ]; then
  best_checkpoint=$(cat "$OUT/best-to-evaluate.txt")
  printf 'best_evaluation\n' > "$OUT/stage.txt"
  CUDA_VISIBLE_DEVICES='' bash "$PY" -m experiments.crystal_tasks_v1.run_design_evaluation --checkpoint "$best_checkpoint" --attempt-plan "$OUT/attempt-plan.jsonl" --gpus "$devices" --output "$OUT/best_evaluation" > "$OUT/best_evaluation.log" 2>&1
fi
printf 'complete\n' > "$OUT/stage.txt"
