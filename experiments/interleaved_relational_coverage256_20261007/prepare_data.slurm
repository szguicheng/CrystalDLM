#!/bin/bash
#SBATCH --job-name=rel-cnl-coverage256-prep
#SBATCH --partition=stingy
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/interleaved_relational_coverage256_20261007/prepare-%j.out
#SBATCH --error=experiments/interleaved_relational_coverage256_20261007/prepare-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="$PWD/experiments/interleaved_ar_diffusion_20261002/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
PYTHON_BIN="$PWD/experiments/interleaved_ar_diffusion_20261002/env/bin/python"
test -x "$PYTHON_BIN"
"$PYTHON_BIN" -u experiments/interleaved_relational_coverage256_20261007/prepare_data.py --prepare
