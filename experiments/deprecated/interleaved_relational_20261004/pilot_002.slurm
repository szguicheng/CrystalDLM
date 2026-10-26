#!/bin/bash
# Fixed continuation of pilot_001: do not retrain AR and do not overwrite pilot_001.
# Submit this draft from the project root only after independent review.
#SBATCH --job-name=rel-cnl-pilot-002
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=experiments/interleaved_relational_20261004/pilot_002-%j.out
#SBATCH --error=experiments/interleaved_relational_20261004/pilot_002-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME="$PWD/experiments/interleaved_ar_diffusion_20261002/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1

PYTHON_BIN="$PWD/experiments/interleaved_ar_diffusion_20261002/env/bin/python"
test -x "$PYTHON_BIN"
"$PYTHON_BIN" -u experiments/interleaved_relational_20261004/test_checks.py \
  > "experiments/interleaved_relational_20261004/pilot_002-checks-$SLURM_JOB_ID.txt"
"$PYTHON_BIN" -u experiments/interleaved_relational_20261004/run.py --check-pack
"$PYTHON_BIN" -u experiments/interleaved_relational_20261004/resume_pilot_001.py --offline-check
"$PYTHON_BIN" -u experiments/interleaved_relational_20261004/resume_pilot_001.py
