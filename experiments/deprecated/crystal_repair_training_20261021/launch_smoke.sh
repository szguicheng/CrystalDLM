#!/usr/bin/env bash
set -euo pipefail
cd /home/liugang/src/crystalDLM
source research/liugang_setup_20261015/runtime.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
experiment=experiments/crystal_repair_training_20261021
mkdir -p "$experiment/logs"
python3 - <<'PY'
import subprocess
read=lambda f,k:subprocess.check_output(['nvidia-smi',f'--query-{k}={f}','--format=csv,noheader,nounits'],text=True).splitlines()
chosen={s.split(',')[1].strip() for s in read('index,uuid','gpu') if int(s.split(',')[0]) in (1,2,3)}
assert len(chosen)==3
assert not [s for s in read('gpu_uuid,pid','compute-apps') if s.split(',')[0].strip() in chosen], 'Selected GPUs are occupied'
PY
trap 'code=$?; printf "%s\n" "$code" > "$experiment/smoke-exit.txt"; if ((code!=0)); then printf "%s\n" failed > "$experiment/smoke-stage.txt"; fi' EXIT
mamba_bin=/home/liugang/software/miniforge3/condabin/mamba
printf '%s\n' two-updates > "$experiment/smoke-stage.txt"
CUDA_VISIBLE_DEVICES=1,2,3 "$mamba_bin" run --prefix "$CRYSTAL_PROJECT/envs/crystaldlm" \
  python -m torch.distributed.run --standalone --nproc-per-node=3 "$experiment/run.py" \
  --mode train --data "$experiment/smoke_data_002" --out "$experiment/smoke_run_002" \
  --max-updates 2 > "$experiment/logs/smoke-v2-two.log" 2>&1
printf '%s\n' resume-third-update > "$experiment/smoke-stage.txt"
CUDA_VISIBLE_DEVICES=1,2,3 "$mamba_bin" run --prefix "$CRYSTAL_PROJECT/envs/crystaldlm" \
  python -m torch.distributed.run --standalone --nproc-per-node=3 "$experiment/run.py" \
  --mode train --data "$experiment/smoke_data_002" --out "$experiment/smoke_run_002" \
  --max-updates 3 --resume > "$experiment/logs/smoke-v2-resume.log" 2>&1
printf '%s\n' finalize-completed-checkpoint > "$experiment/smoke-stage.txt"
CUDA_VISIBLE_DEVICES=1,2,3 "$mamba_bin" run --prefix "$CRYSTAL_PROJECT/envs/crystaldlm" \
  python -m torch.distributed.run --standalone --nproc-per-node=3 "$experiment/run.py" \
  --mode train --data "$experiment/smoke_data_002" --out "$experiment/smoke_run_002" \
  --max-updates 3 --resume > "$experiment/logs/smoke-v2-finalize.log" 2>&1
printf '%s\n' validation > "$experiment/smoke-stage.txt"
CUDA_VISIBLE_DEVICES=1,2,3 "$mamba_bin" run --prefix "$CRYSTAL_PROJECT/envs/crystaldlm" \
  python -m torch.distributed.run --standalone --nproc-per-node=3 "$experiment/run.py" \
  --mode validate --data "$experiment/smoke_data_002" --out "$experiment/smoke_run_002" \
  > "$experiment/logs/smoke-v2-validation.log" 2>&1
printf '%s\n' complete > "$experiment/smoke-stage.txt"
