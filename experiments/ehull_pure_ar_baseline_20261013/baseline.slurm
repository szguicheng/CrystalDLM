#!/bin/bash
#SBATCH --job-name=ehull-pure-ar
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/baseline-%j.out
#SBATCH --error=logs/baseline-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/scratch/chenggui2/CrystalDLM/experiments/interleaved_ar_diffusion_20261002/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export HTTP_PROXY=http://proxy.cityu.edu.hk:8080 HTTPS_PROXY=http://proxy.cityu.edu.hk:8080
PY=/scratch/chenggui2/CrystalDLM/experiments/interleaved_ar_diffusion_20261002/env/bin/python
"$PY" -m unittest discover -s . -p test_baseline.py -v
"$PY" -u baseline.py --data data_001 --out run_001
