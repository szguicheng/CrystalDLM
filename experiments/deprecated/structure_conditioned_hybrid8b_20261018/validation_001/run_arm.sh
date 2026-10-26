#!/usr/bin/env bash
set -euo pipefail
cd /home/liugang/src/crystalDLM
source research/liugang_setup_20261015/runtime.sh
EXP=experiments/structure_conditioned_hybrid8b_20261018/validation_001
ARM="$1"
GPU="$2"
[[ "$ARM" == control || "$ARM" == highmask ]]
[[ "$GPU" == 1 || "$GPU" == 2 ]]
MAMBA=/home/liugang/software/miniforge3/condabin/mamba
trap 'code=$?; echo "$code" > "$EXP/$ARM-exit.txt"' EXIT
test -z "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES="$GPU"
"$MAMBA" run --prefix envs/crystaldlm python -u "$EXP/continuation.py" --arm "$ARM" --out "$EXP/$ARM" --stop-after-update 2 > "$EXP/$ARM-smoke.log" 2>&1
"$MAMBA" run --prefix envs/crystaldlm python - "$EXP/$ARM" <<'PY'
import json
import sys
from pathlib import Path
import torch
p = Path(sys.argv[1])
c = torch.load(p / 'checkpoints/step_002.pt', map_location='cpu', weights_only=False)
rows = [json.loads(s) for s in (p / 'updates.jsonl').read_text().splitlines()]
assert c['completed_updates'] == 2 and len(rows) == 2
assert c['trainable_state_sha256'] != c['source_trainable_sha256']
assert all(r['finite_gradients'] and r['grad_norm'] > 0 for r in rows)
print('two-update smoke and changed-weight check passed', p)
PY
"$MAMBA" run --prefix envs/crystaldlm python -u "$EXP/continuation.py" --arm "$ARM" --out "$EXP/$ARM" --resume "$EXP/$ARM/checkpoints/step_002.pt" > "$EXP/$ARM-training.log" 2>&1
test -z "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)"
"$MAMBA" run --prefix envs/crystaldlm python -u "$EXP/evaluate.py" --arm "$ARM" --samplers original > "$EXP/$ARM-evaluation.log" 2>&1
