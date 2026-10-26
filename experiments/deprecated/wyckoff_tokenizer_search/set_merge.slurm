#!/bin/bash
#SBATCH --job-name=crystaldlm-set-merges
#SBATCH --partition=tiny
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python learn_set_merges.py --merges 256 --min-support 20 --max-orbits 4
