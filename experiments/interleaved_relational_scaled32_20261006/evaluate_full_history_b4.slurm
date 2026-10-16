#!/bin/bash
#SBATCH --job-name=rel-history-b4-eval
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=experiments/interleaved_relational_scaled32_20261006/evaluate-full-history-b4-%j.out
#SBATCH --error=experiments/interleaved_relational_scaled32_20261006/evaluate-full-history-b4-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME="$PWD/experiments/interleaved_ar_diffusion_20261002/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
"$PWD/experiments/interleaved_ar_diffusion_20261002/env/bin/python" -u \
  experiments/interleaved_relational_scaled32_20261006/evaluate_full_history_b4.py \
  --pin experiments/interleaved_relational_scaled32_20261006/eval_full_history_b4_input_pin.json \
  --out experiments/interleaved_relational_scaled32_20261006/eval_full_history_b4_001 \
  --prepare-pin
