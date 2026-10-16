#!/bin/bash
#SBATCH --job-name=crystal-cohort-probe
#SBATCH --partition=stingy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:40:00
#SBATCH --output=logs/cohort-probe-%j.out
#SBATCH --error=logs/cohort-probe-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME="$PWD/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
env/bin/python -u diagnostics/cohort_probe.py --data data_002 --cohort cohort_001 --out cohort_probe_001
