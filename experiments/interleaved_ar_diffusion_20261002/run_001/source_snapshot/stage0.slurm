#!/bin/bash
#SBATCH --job-name=crystal-ar-diff-stage0
#SBATCH --partition=stingy
#SBATCH --nodelist=gpu-a100-05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=gpu-a100-08,gpu-a100-11
#SBATCH --time=02:00:00
#SBATCH --output=logs/stage0-%j.out
#SBATCH --error=logs/stage0-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME="$PWD/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export HTTP_PROXY=http://proxy.cityu.edu.hk:8080 HTTPS_PROXY=http://proxy.cityu.edu.hk:8080
PY="$PWD/env/bin/python"
mkdir -p "allocations/$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv > "allocations/$SLURM_JOB_ID/gpu.txt"
"$PY" -c 'import torch; assert torch.cuda.is_available(); x=torch.randn(64,64,device="cuda"); print(torch.__version__,torch.cuda.get_device_name(),float((x@x).mean()))' > "allocations/$SLURM_JOB_ID/cuda_probe.txt"
"$PY" -u test_checks.py > "allocations/$SLURM_JOB_ID/checks.txt"
"$PY" -u run.py --data smoke_data_002 --out run_001 --updates 200
