#!/usr/bin/env bash
set -euo pipefail
source /home/liugang/src/crystalDLM/research/liugang_setup_20261015/runtime.sh
cd "$CRYSTAL_PROJECT"
MAMBA=/home/liugang/software/miniforge3/condabin/mamba
PREFIX="$CRYSTAL_PROJECT/envs/crystaldlm"
EXPERIMENT=experiments/ehull_pure_ar_qwen3_8b_20261015
MODEL_PATH="$CRYSTAL_PROJECT/models/Qwen3-8B-b968826d9c46"
trap 'code=$?; printf "%s\n" "$code" > "$CRYSTAL_PROJECT/logs/experiment-exit-code.txt"' EXIT

check_selected_gpus() {
  python3 - "$1" <<'PY'
import subprocess,sys
ids=sys.argv[1]
active=subprocess.check_output(['nvidia-smi','-i',ids,'--query-compute-apps=pid',
                                '--format=csv,noheader'],text=True).strip()
assert not active, f'selected GPUs {ids} already have compute processes: {active}'
print('SELECTED_FREE_GPUS',ids,flush=True)
PY
}

printf 'STAGE cpu_tests\n'
CUDA_VISIBLE_DEVICES='' "$MAMBA" run --prefix "$PREFIX" python -m unittest discover \
  -s "$EXPERIMENT" -p test_train.py -v
printf 'STAGE scorer_regression\n'
CUDA_VISIBLE_DEVICES='' "$MAMBA" run --prefix "$PREFIX" python \
  research/liugang_setup_20261015/check_scorer.py

printf 'STAGE preflight\n'
check_selected_gpus 0,1
CUDA_VISIBLE_DEVICES=0,1 "$MAMBA" run --prefix "$PREFIX" torchrun \
  --standalone --nproc_per_node=2 "$EXPERIMENT/train.py" \
  --mode preflight --model-path "$MODEL_PATH" \
  --out "$EXPERIMENT/preflight_001"

printf 'STAGE formal_train\n'
check_selected_gpus 0,1
CUDA_VISIBLE_DEVICES=0,1 "$MAMBA" run --prefix "$PREFIX" torchrun \
  --standalone --nproc_per_node=2 "$EXPERIMENT/train.py" \
  --mode train --model-path "$MODEL_PATH" --out "$EXPERIMENT/run_001" \
  --preflight-receipt "$EXPERIMENT/preflight_001/summary.json"

printf 'STAGE generated_structure_evaluation\n'
check_selected_gpus 0
CUDA_VISIBLE_DEVICES=0 "$MAMBA" run --prefix "$PREFIX" torchrun \
  --standalone --nproc_per_node=1 "$EXPERIMENT/train.py" \
  --mode evaluate --model-path "$MODEL_PATH" \
  --checkpoint "$EXPERIMENT/run_001/checkpoint.pt" \
  --out "$EXPERIMENT/evaluation_001"
printf 'STAGE complete\n'
