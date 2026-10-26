#!/bin/bash
#SBATCH --job-name=crystaldlm-token-pilot
#SBATCH --partition=stingy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled
cd "$SLURM_SUBMIT_DIR"
PYTHON=/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python
variants=(factorized compositional)
if [[ "${SMOKE:-0}" == 1 ]]; then
  for variant in "${variants[@]}"; do
    "$PYTHON" train_profile_model.py --variant "$variant" --smoke --steps 5 --eval-every 5 --val-limit 32
  done
else
  for variant in "${variants[@]}"; do
    "$PYTHON" train_profile_model.py --variant "$variant" --steps 2000 --eval-every 500 --val-limit 2048
  done
fi
