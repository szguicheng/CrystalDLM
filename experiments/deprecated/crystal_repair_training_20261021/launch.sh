#!/usr/bin/env bash
set -euo pipefail
cd /home/liugang/src/crystalDLM
source research/liugang_setup_20261015/runtime.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
experiment=experiments/crystal_repair_training_20261021
mkdir -p "$experiment/logs"
trap 'code=$?; printf "%s\n" "$code" > "$experiment/exit.txt"; if ((code!=0)); then printf "%s\n" failed > "$experiment/stage.txt"; fi' EXIT
python3 - <<'PY'
import shutil, subprocess
read=lambda fields,kind:subprocess.check_output(['nvidia-smi',f'--query-{kind}={fields}','--format=csv,noheader,nounits'],text=True).splitlines()
chosen={row.split(',')[1].strip() for row in read('index,uuid','gpu') if int(row.split(',')[0]) in (1,2,3)}
assert len(chosen)==3
assert not [row for row in read('gpu_uuid,pid','compute-apps') if row.split(',')[0].strip() in chosen], 'Selected GPUs are occupied'
assert shutil.disk_usage('.').free > 6*1024**3, 'Need 6 GiB free for checkpoints and outputs'
PY
mamba_bin=/home/liugang/software/miniforge3/condabin/mamba
resume=()
if [[ -f "$experiment/run_001/latest.pt" ]]; then resume=(--resume); fi
printf '%s\n' training > "$experiment/stage.txt"
CUDA_VISIBLE_DEVICES=1,2,3 "$mamba_bin" run --prefix "$CRYSTAL_PROJECT/envs/crystaldlm" \
  python -m torch.distributed.run --standalone --nproc-per-node=3 "$experiment/run.py" \
  --mode train --data "$experiment/data_002" --out "$experiment/run_001" "${resume[@]}" \
  >> "$experiment/logs/train.log" 2>&1
printf '%s\n' validation > "$experiment/stage.txt"
CUDA_VISIBLE_DEVICES=1,2,3 "$mamba_bin" run --prefix "$CRYSTAL_PROJECT/envs/crystaldlm" \
  python -m torch.distributed.run --standalone --nproc-per-node=3 "$experiment/run.py" \
  --mode validate --data "$experiment/data_002" --out "$experiment/run_001" \
  >> "$experiment/logs/validation.log" 2>&1
printf '%s\n' paired-generation > "$experiment/stage.txt"
pids=()
for shard in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((shard+1)) "$mamba_bin" run --prefix "$CRYSTAL_PROJECT/envs/crystaldlm" \
    python "$experiment/evaluate.py" --data "$experiment/data_002" --run-dir "$experiment/run_001" \
    --out "$experiment/evaluation_001" --shard-index "$shard" --num-shards 3 --gpu "$((shard+1))" --resume \
    >> "$experiment/logs/generation-$shard.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if ((failed)); then exit 1; fi
CUDA_VISIBLE_DEVICES= "$mamba_bin" run --prefix "$CRYSTAL_PROJECT/envs/crystaldlm" \
  python "$experiment/evaluate.py" --mode summarize --out "$experiment/evaluation_001" \
  >> "$experiment/logs/summary.log" 2>&1
printf '%s\n' complete > "$experiment/stage.txt"
