#!/bin/bash
#SBATCH --job-name=crystaldlm-multiplicity-model
#SBATCH --partition=stingy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
PYTHON=/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python
for variant in flat degeneracy hierarchical; do
  if [[ "${SMOKE:-0}" == 1 ]]; then
    "$PYTHON" train_multiplicity_model.py --variant "$variant" --smoke --steps 5 --eval-every 5 --val-limit 32
  else
    "$PYTHON" train_multiplicity_model.py --variant "$variant" --seed "${SEED:-17}"
  fi
done
if [[ "${SMOKE:-0}" != 1 ]]; then
  "$PYTHON" compare_multiplicity_models.py --seed "${SEED:-17}"
fi
