#!/usr/bin/env bash
set -euo pipefail
cd /home/liugang/src/crystalDLM
source research/liugang_setup_20261015/runtime.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
experiment=experiments/structure_feedback_loop_20261020
case "${1:?usage: launch.sh smoke|pilot}" in
  smoke) groups=1; output="$experiment/smoke_001" ;;
  pilot) groups=10; output="$experiment/run_001" ;;
  *) exit 2 ;;
esac
python3 - <<'PY'
import subprocess
from pathlib import Path
read = lambda fields, kind: subprocess.check_output(
    ['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'], text=True).splitlines()
selected = {line.split(',')[1].strip() for line in read('index,uuid', 'gpu')
            if int(line.split(',')[0]) in (0, 1)}
assert len(selected) == 2
processes = read('gpu_uuid,pid', 'compute-apps')
busy = [line for line in processes if line.split(',')[0].strip() in selected]
assert not busy, f'Selected GPUs already have processes: {busy}'
project_gpus = set()
for line in processes:
    uuid, pid = map(str.strip, line.split(','))
    try:
        cwd = Path(f'/proc/{pid}/cwd').resolve(strict=True)
    except (PermissionError, FileNotFoundError):
        continue
    if cwd.is_relative_to(Path.cwd()):
        project_gpus.add(uuid)
assert len(project_gpus | selected) <= 3, 'Existing project tasks would exceed the three-GPU limit'
PY
mkdir -p "$output"
trap 'code=$?; printf "%s\n" "$code" > "$output/exit.txt"; if (( code != 0 )); then printf "%s\n" failed > "$output/stage.txt"; fi' EXIT
if [[ ! -f "$output/started.txt" ]]; then date -Iseconds > "$output/started.txt"; fi
printf '%s\n' sampling > "$output/stage.txt"
mamba_bin=/home/liugang/software/miniforge3/condabin/mamba
resume0=(); resume1=()
if [[ -f "$output/shard-00-of-02/manifest.json" ]]; then resume0=(--resume); fi
if [[ -f "$output/shard-01-of-02/manifest.json" ]]; then resume1=(--resume); fi
CUDA_VISIBLE_DEVICES=0 "$mamba_bin" run --prefix "$CRYSTAL_PROJECT/envs/crystaldlm" \
  python "$experiment/run.py" --out "$output" --groups-per-sg "$groups" \
  --num-shards 2 --shard-index 0 --gpu 0 "${resume0[@]}" >> "$output/shard-0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$mamba_bin" run --prefix "$CRYSTAL_PROJECT/envs/crystaldlm" \
  python "$experiment/run.py" --out "$output" --groups-per-sg "$groups" \
  --num-shards 2 --shard-index 1 --gpu 1 "${resume1[@]}" >> "$output/shard-1.log" 2>&1 &
pid1=$!
set +e
wait "$pid0"; code0=$?
wait "$pid1"; code1=$?
set -e
printf '%s\n' "$code0" > "$output/shard-0-exit.txt"
printf '%s\n' "$code1" > "$output/shard-1-exit.txt"
if (( code0 != 0 || code1 != 0 )); then
  printf '%s\n' failed > "$output/stage.txt"
  exit 1
fi
printf '%s\n' summarizing > "$output/stage.txt"
"$mamba_bin" run --prefix "$CRYSTAL_PROJECT/envs/crystaldlm" \
  python "$experiment/summarize.py" --out "$output" > "$output/summary.log" 2>&1
date -Iseconds > "$output/finished.txt"
printf '%s\n' complete > "$output/stage.txt"
