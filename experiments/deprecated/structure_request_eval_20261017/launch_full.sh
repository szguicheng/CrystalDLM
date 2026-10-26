#!/usr/bin/env bash
set -euo pipefail
cd /home/liugang/src/crystalDLM
source research/liugang_setup_20261015/runtime.sh
OUT=experiments/structure_request_eval_20261017
MAMBA=/home/liugang/software/miniforge3/condabin/mamba
PREFIX="$CRYSTAL_PROJECT/envs/crystaldlm"
"$MAMBA" run --prefix "$PREFIX" python -m unittest discover -s "$OUT" -p test_run.py -q
active=$(nvidia-smi -i 0,1 --query-compute-apps=pid --format=csv,noheader)
test -z "$active"
sha256sum "$OUT/run.py" "$OUT/test_run.py" "$OUT/launch_full.sh" > "$OUT/full-source-sha256.txt"
CUDA_VISIBLE_DEVICES=0 "$MAMBA" run --prefix "$PREFIX" python "$OUT/run.py" --out "$OUT/run_001" --num-shards 2 --shard-index 0 > "$OUT/full-shard-0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$MAMBA" run --prefix "$PREFIX" python "$OUT/run.py" --out "$OUT/run_001" --num-shards 2 --shard-index 1 > "$OUT/full-shard-1.log" 2>&1 &
pid1=$!
set +e
wait "$pid0"; code0=$?
wait "$pid1"; code1=$?
printf '{"shard_0":%s,"shard_1":%s}\n' "$code0" "$code1" > "$OUT/full-exit.json"
set -e
"$MAMBA" run --prefix "$PREFIX" python "$OUT/run.py" --mode summarize --out "$OUT/run_001" > "$OUT/full-summary.log"
test "$code0" -eq 0 && test "$code1" -eq 0
