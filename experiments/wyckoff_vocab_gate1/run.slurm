#!/bin/bash
#SBATCH --job-name=crystaldlm-gate1
#SBATCH --partition=tiny
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.mplconfig"
cd "$SLURM_SUBMIT_DIR"
/home/chenggui2/scratch/anaconda3/bin/python3 count_vocab.py --input input --output results --max-atoms 40
