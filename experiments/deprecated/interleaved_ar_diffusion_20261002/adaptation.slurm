#!/bin/bash
#SBATCH --job-name=crystal-adapt-v2
#SBATCH --partition=stingy
#SBATCH --nodelist=gpu-a100-05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/adapt-%j.out
#SBATCH --error=logs/adapt-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME="$PWD/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export HTTP_PROXY=http://proxy.cityu.edu.hk:8080 HTTPS_PROXY=http://proxy.cityu.edu.hk:8080
PY="$PWD/env/bin/python"
mkdir -p "allocations/$SLURM_JOB_ID"
"$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name()); x=torch.randn(32,32,device="cuda"); assert torch.isfinite(x@x).all()' > "allocations/$SLURM_JOB_ID/cuda_probe.txt"
"$PY" -u test_checks.py > "allocations/$SLURM_JOB_ID/checks.txt"
"$PY" -u adapt.py --data smoke_data_002 --out adaptation_001
