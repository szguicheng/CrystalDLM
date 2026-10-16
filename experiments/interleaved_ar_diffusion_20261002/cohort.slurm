#!/bin/bash
#SBATCH --job-name=crystal-unseen-cohort
#SBATCH --partition=stingy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/cohort-%j.out
#SBATCH --error=logs/cohort-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME="$PWD/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export HTTP_PROXY=http://proxy.cityu.edu.hk:8080 HTTPS_PROXY=http://proxy.cityu.edu.hk:8080
env/bin/python -u cohort.py --data data_002 --initial alignment_002 --out cohort_001
