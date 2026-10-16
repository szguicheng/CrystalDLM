#!/bin/bash
# Draft only: do not submit until the offline adapter checks and independent review pass.
# The runner probes both arms first, discards probe models/optimizers/RNG state,
# then creates fresh same-seed models for the frozen 512-update pilot.
#SBATCH --job-name=rel-cnl-pilot-001
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=experiments/interleaved_relational_20261004/pilot_001-%j.out
#SBATCH --error=experiments/interleaved_relational_20261004/pilot_001-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME="$PWD/experiments/interleaved_ar_diffusion_20261002/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1

PYTHON_BIN="$PWD/experiments/interleaved_ar_diffusion_20261002/env/bin/python"
test -x "$PYTHON_BIN"
"$PYTHON_BIN" -u experiments/interleaved_relational_20261004/test_checks.py \
  > "experiments/interleaved_relational_20261004/pilot_001-checks-$SLURM_JOB_ID.txt"
"$PYTHON_BIN" -u experiments/interleaved_relational_20261004/run.py \
  --data experiments/interleaved_relational_20261004/data_pilot_001 \
  --check-pack
"$PYTHON_BIN" -u experiments/interleaved_relational_20261004/run.py \
  --data experiments/interleaved_relational_20261004/data_pilot_001 \
  --out experiments/interleaved_relational_20261004/pilot_001
