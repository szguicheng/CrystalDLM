#!/bin/bash
#SBATCH --job-name=crystaldlm-preprocess
#SBATCH --partition=tiny
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
PYTHON=/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python
if [[ "${SMOKE:-0}" == 1 ]]; then
  "$PYTHON" preprocess.py --split train --limit 1000 --workers 4 --output processed_smoke
else
  splits=(train val test)
  "$PYTHON" preprocess.py --split "${splits[$SLURM_ARRAY_TASK_ID]}" --workers 4 --output processed
fi
