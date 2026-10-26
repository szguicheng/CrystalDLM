#!/usr/bin/env bash
set -euo pipefail
cd /home/liugang/src/crystalDLM
source research/liugang_setup_20261015/runtime.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
experiment=experiments/repair_feedback_full_20261022
mode=${1:?Use smoke or formal}
case "$mode" in
  smoke) out="$experiment/smoke_001"; groups=1 ;;
  formal) out="$experiment/run_001"; groups=100 ;;
  *) exit 2 ;;
esac
mkdir -p "$experiment/logs"
trap 'code=$?; printf "%s\n" "$code" > "$experiment/$mode-exit.txt"; if ((code!=0)); then printf "%s\n" failed > "$experiment/$mode-stage.txt"; fi' EXIT
check_gpus() {
  python3 - "$@" <<'PY'
import subprocess,sys
selected=set(map(int,sys.argv[1:]))
read=lambda fields,kind:subprocess.check_output(['nvidia-smi',f'--query-{kind}={fields}','--format=csv,noheader,nounits'],text=True).splitlines()
uuids={row.split(',')[1].strip() for row in read('index,uuid','gpu') if int(row.split(',')[0]) in selected}
assert len(uuids)==len(selected)
assert not [row for row in read('gpu_uuid,pid','compute-apps') if row.split(',')[0].strip() in uuids], 'Selected GPU is occupied'
PY
}
test "$(findmnt -n -o TARGET -T "$CRYSTAL_PROJECT")" = /data1
mamba_bin=/home/liugang/software/miniforge3/condabin/mamba
check_gpus 1 2 3
resume=()
if [[ -f "$out/run_manifest.json" ]]; then resume=(--resume); fi
printf '%s\n' generation > "$experiment/$mode-stage.txt"
pids=()
for shard in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((shard+1)) "$mamba_bin" run --prefix "$CRYSTAL_PROJECT/envs/crystaldlm" \
    python -u "$experiment/run.py" --out "$out" --groups-per-sg "$groups" \
    --num-shards 3 --shard-index "$shard" --gpu "$((shard+1))" "${resume[@]}" \
    >> "$experiment/logs/$mode-generation-$shard.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if ((failed)); then exit 1; fi
printf '%s\n' scoring-export > "$experiment/$mode-stage.txt"
CUDA_VISIBLE_DEVICES= "$mamba_bin" run --prefix "$CRYSTAL_PROJECT/envs/crystaldlm" \
  python "$experiment/score.py" --mode export --run-dir "$out" \
  >> "$experiment/logs/$mode-export.log" 2>&1
printf '%s\n' grace-scoring > "$experiment/$mode-stage.txt"
check_gpus 1 2
pids=()
gpu=1
for arm in independent feedback; do
  score_resume=()
  if [[ -f "$out/selected/$arm/grace_score/scoring_manifest.json" ]]; then score_resume=(--resume); fi
  CUDA_VISIBLE_DEVICES="$gpu" TF_CPP_MIN_LOG_LEVEL=2 "$mamba_bin" run --prefix "$CRYSTAL_PROJECT/envs/grace" \
    python -u experiments/structure_request_eval_20261017/grace_score.py --mode score \
    --run-dir "$out/selected/$arm" "${score_resume[@]}" \
    >> "$experiment/logs/$mode-grace-$arm.log" 2>&1 &
  pids+=("$!")
  gpu=$((gpu+1))
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if ((failed)); then exit 1; fi
printf '%s\n' summary > "$experiment/$mode-stage.txt"
CUDA_VISIBLE_DEVICES= "$mamba_bin" run --prefix "$CRYSTAL_PROJECT/envs/crystaldlm" \
  python "$experiment/score.py" --mode summarize --run-dir "$out" \
  >> "$experiment/logs/$mode-summary.log" 2>&1
printf '%s\n' complete > "$experiment/$mode-stage.txt"
