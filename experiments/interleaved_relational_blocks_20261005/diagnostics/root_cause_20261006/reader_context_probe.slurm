#!/bin/bash
#SBATCH --job-name=rel-reader-context
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --output=experiments/interleaved_relational_blocks_20261005/diagnostics/root_cause_20261006/reader-context-%j.out
#SBATCH --error=experiments/interleaved_relational_blocks_20261005/diagnostics/root_cause_20261006/reader-context-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME="$PWD/experiments/interleaved_ar_diffusion_20261002/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
"$PWD/experiments/interleaved_ar_diffusion_20261002/env/bin/python" -u \
  experiments/interleaved_relational_blocks_20261005/diagnostics/root_cause_20261006/reader_context_probe.py
